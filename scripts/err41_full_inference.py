#!/usr/bin/env python3
"""Run ERR41 end-to-end 4D trajectory inference for a case directory.

This script executes a full ERR41 run with:
- graph scenario preparation (edge filtering by avoided sectors),
- automatic checkpoint resolution,
- TResPASS transition recomputation,
- backward SVI/sampling via ``compute_4d_path_for_dataset``,
- and run-level reporting/debug artifact export.

It creates a run folder under ``<case_dir>/err41/<run_name>`` and writes:
- ``run_summary.json`` on success,
- ``run_error.json`` on failure,
- ``inference_outputs/`` and ``inference_cache/``,
- optional ``trespass_debug/`` files when failure debugging is enabled.

Usage examples
--------------
Default run (auto-select latest checkpoint in case defaults):
    python scripts/err41_full_inference.py --case-dir "/mnt/d/project-equinox/data/cases/LGAV_LFPG" --run-name "run_20260226_auto_duration" --flight-id "392AECAFR98CQ" --n-samples 100 --policy sample --initial-k-policy uniform --no-use-cache

Run with explicit checkpoint and custom run name:
    python scripts/err41_full_inference.py \\
      --case-dir data/cases/LGAV_LFPG \\
      --checkpoint-path data/cases/LGAV_LFPG/results_full/checkpoint_iter_100.pt \\
      --run-name run_20260226_manual

Run for a specific flight and disable metadata-based duration:
    python scripts/err41_full_inference.py \\
      --case-dir data/cases/LGAV_LFPG \\
      --flight-id AEE123 \\
      --takeoff-timestamp 1700000000 \\
      --no-use-flight-metadata-duration \\
      --max-flight-duration-hours 3.5

Primary inputs
--------------
- ``--case-dir``: case root containing graph/data/model artifacts.
- Checkpoint source:
  - ``--checkpoint-path`` (explicit file), or
  - ``--checkpoint-dir`` (directory with ``checkpoint_iter_*.pt``), or
  - auto-discovery from known case subdirectories.
- Flight selector (optional): ``--flight-id`` and/or ``--takeoff-timestamp``.
- Sampling controls: ``--n-samples``, ``--policy``, ``--initial-k-policy``.
- Graph scenario controls: ``--sectors-to-avoid``, ``--sectors-geojson-path``,
  ``--connectivity-repair-iterations``.

Output examples
---------------
Success:
    {
      "status": "ok",
      "run_dir": "data/cases/LGAV_LFPG/err41/run_20260226_123456",
      "output_dir": ".../inference_outputs",
      "n_samples": 100,
      "first_route": ["LGAV", "...", "LFPG"]
    }

Failure:
    {
      "status": "error",
      "error": "ValueError('No valid transitions ...')",
      "run_dir": "data/cases/LGAV_LFPG/err41/run_20260226_123456",
      "trespass_debug": {
        "status": "ok",
        "files": {
          "forward_transitions": ".../FW_<flight>_<takeoff>.pkl",
          "summary_json": ".../debug_summary_<flight>_<takeoff>.json"
        }
      }
    }
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import networkx as nx
import torch

from equinox.dp.trespass.thinning import thin_closures
from equinox.dp.trespass.transition_utils import parse_transition
from equinox.dp.trespass.tres_backward import tres_backward
from equinox.dp.trespass.tres_forward import tres_forward
from equinox.helpers.datetimeh import datestr_to_seconds_since_midnight
from equinox.sampling.pipeline import (
    GraphScenario,
    _ensure_required_flight_config_fields,
    _infer_origin_goal,
    _prepare_graph_for_scenario,
    _resolve_auto_cruise_altitude,
    _resolve_takeoff_fields,
    _resize_cost_model,
    compute_4d_path_for_dataset,
)
from equinox.sampling.trespass.inference import (
    _resolve_wind_model,
    load_case,
    load_cost_model,
    load_flight_artifacts,
)


def _get_latest_checkpoint(results_dir: Path) -> Optional[Path]:
    pattern = re.compile(r"checkpoint_iter_(\d+)\.pt$")
    candidates: list[tuple[int, Path]] = []
    for file_path in results_dir.glob("checkpoint_iter_*.pt"):
        match = pattern.search(file_path.name)
        if not match:
            continue
        candidates.append((int(match.group(1)), file_path))
    if not candidates:
        final_results = results_dir / "final_results.pt"
        if final_results.exists():
            return final_results
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def _resolve_checkpoint(case_dir: Path, checkpoint_path: Optional[str], checkpoint_dir: Optional[str]) -> Path:
    if checkpoint_path:
        resolved = Path(checkpoint_path)
        if not resolved.exists():
            raise FileNotFoundError(f"checkpoint_path does not exist: {resolved}")
        return resolved

    candidate_dirs: list[Path] = []
    if checkpoint_dir:
        candidate_dirs.append(Path(checkpoint_dir))
    else:
        candidate_dirs.extend(
            [
                case_dir / "batch_sgd_results",
                case_dir / "results_nogauge",
                case_dir / "results_full",
                case_dir / "results_nogaugezeroridge",
            ]
        )

    available: list[Path] = []
    for directory in candidate_dirs:
        if directory.exists() and directory.is_dir():
            available.append(directory)

    if not available:
        raise FileNotFoundError(
            "No checkpoint directory found. Provide --checkpoint-path or --checkpoint-dir."
        )

    # Prefer the latest iteration among all available candidate dirs.
    best: Optional[tuple[int, Path]] = None
    fallback: Optional[Path] = None
    iter_pattern = re.compile(r"checkpoint_iter_(\d+)\.pt$")
    for directory in available:
        ckpt = _get_latest_checkpoint(directory)
        if ckpt is None:
            continue
        m = iter_pattern.search(ckpt.name)
        if m:
            item = (int(m.group(1)), ckpt)
            if best is None or item[0] > best[0]:
                best = item
        else:
            # final_results.pt fallback if no iter checkpoints found anywhere
            fallback = ckpt
    if best is not None:
        return best[1]
    if fallback is not None:
        return fallback
    raise FileNotFoundError("No checkpoint_iter_*.pt or final_results.pt found in candidate dirs.")


def _state_from_base(base: tuple, *, is_src: bool) -> tuple[int, int, int, float, int]:
    if is_src:
        return (int(base[0]), int(base[1]), int(base[2]), float(base[3]), int(base[4]))
    return (int(base[5]), int(base[6]), int(base[7]), float(base[8]), int(base[9]))


def _state_preview(states: set[tuple[int, int, int, float, int]], limit: int = 8) -> list[dict[str, Any]]:
    ordered = sorted(states, key=lambda s: (s[1], s[2], s[3], s[4], s[0]))
    return [
        {
            "node_idx": int(s[0]),
            "k_idx": int(s[1]),
            "rho_idx": int(s[2]),
            "alt_ft": float(s[3]),
            "phase_idx": int(s[4]),
        }
        for s in ordered[:limit]
    ]


def _diagnose_thinning(
    closures: list[tuple],
    *,
    origin_node_idx: int,
    goal_node_idx: int,
    max_rho: Optional[int],
) -> dict[str, Any]:
    if not closures:
        return {
            "thinning_outcome": "empty",
            "reason": "backward_produced_no_closures",
            "closure_count": 0,
        }

    graph = nx.DiGraph()
    all_states: set[tuple[int, int, int, float, int]] = set()
    for transition in closures:
        base, _, _ = parse_transition(transition)
        u_state = _state_from_base(base, is_src=True)
        v_state = _state_from_base(base, is_src=False)
        graph.add_edge(u_state, v_state)
        all_states.add(u_state)
        all_states.add(v_state)

    inferred_max_rho = max(
        max(int(c[2]) for c in closures),
        max(int(c[7]) for c in closures),
    )
    max_rho_effective = inferred_max_rho if max_rho is None else int(max_rho)

    origin_states = {
        state for state in all_states if state[0] == int(origin_node_idx) and state[2] == max_rho_effective
    }
    goal_states = {state for state in all_states if state[0] == int(goal_node_idx)}

    diagnostics: dict[str, Any] = {
        "thinning_outcome": "unknown",
        "closure_count": len(closures),
        "closure_state_count": len(all_states),
        "closure_edge_count": int(graph.number_of_edges()),
        "origin_node_idx": int(origin_node_idx),
        "goal_node_idx": int(goal_node_idx),
        "max_rho_input": None if max_rho is None else int(max_rho),
        "max_rho_inferred": int(inferred_max_rho),
        "max_rho_effective": int(max_rho_effective),
        "origin_state_count": len(origin_states),
        "goal_state_count": len(goal_states),
        "origin_states_preview": _state_preview(origin_states),
        "goal_states_preview": _state_preview(goal_states),
    }

    if not origin_states:
        diagnostics["thinning_outcome"] = "empty"
        diagnostics["reason"] = "no_origin_state_with_max_rho"
        return diagnostics
    if not goal_states:
        diagnostics["thinning_outcome"] = "empty"
        diagnostics["reason"] = "no_goal_state_in_closures"
        return diagnostics

    reachable_from_origins: set[tuple[int, int, int, float, int]] = set()
    for start_state in origin_states:
        reachable_from_origins.add(start_state)
        reachable_from_origins.update(nx.descendants(graph, start_state))

    can_reach_goals: set[tuple[int, int, int, float, int]] = set()
    reversed_graph = nx.reverse_view(graph)
    for end_state in goal_states:
        can_reach_goals.add(end_state)
        can_reach_goals.update(nx.descendants(reversed_graph, end_state))

    valid_states = reachable_from_origins.intersection(can_reach_goals)
    diagnostics.update(
        {
            "reachable_from_origin_count": len(reachable_from_origins),
            "can_reach_goal_count": len(can_reach_goals),
            "valid_state_count": len(valid_states),
        }
    )

    if not reachable_from_origins:
        diagnostics["thinning_outcome"] = "empty"
        diagnostics["reason"] = "no_states_reachable_from_origin"
        return diagnostics
    if not can_reach_goals:
        diagnostics["thinning_outcome"] = "empty"
        diagnostics["reason"] = "no_states_can_reach_goal"
        return diagnostics
    if not valid_states:
        diagnostics["thinning_outcome"] = "empty"
        diagnostics["reason"] = "origin_reachable_and_goal_reachable_state_sets_do_not_intersect"
        return diagnostics

    diagnostics["thinning_outcome"] = "non_empty_possible"
    diagnostics["reason"] = "valid_state_intersection_exists"
    return diagnostics


def _dump_trespass_debug(
    *,
    case_dir: Path,
    resolved_checkpoint: Path,
    graph_scenario: GraphScenario,
    run_dir: Path,
    flight_id: Optional[str],
    takeoff_timestamp: Optional[int],
    estimated_landing_time_str: Optional[str] = None,
) -> dict[str, Any]:
    config, source_components = load_case(str(case_dir))
    (
        selected_flight_id,
        selected_takeoff_ts,
        _dataset_transitions,
        _dataset_tailwind,
        flight_meta,
    ) = load_flight_artifacts(
        str(case_dir),
        flight_id=flight_id,
        takeoff_timestamp=takeoff_timestamp,
    )

    _infer_origin_goal(source_components, flight_meta.get("flight_metadata", {}))
    _resolve_auto_cruise_altitude(
        config,
        source_components,
        flight_meta.get("flight_metadata", {}),
        flight_id=selected_flight_id,
        takeoff_timestamp=selected_takeoff_ts,
    )
    if source_components.get("origin_node") is None or source_components.get("goal_node") is None:
        raise ValueError("Origin/goal nodes are missing after metadata resolution.")

    components, graph_report = _prepare_graph_for_scenario(
        config=config,
        components=source_components,
        graph_scenario=graph_scenario,
    )
    _ensure_required_flight_config_fields(config)

    source_cost_model, _checkpoint_metadata = load_cost_model(
        resolved_checkpoint,
        config=config,
        num_waypoints=source_components["num_nodes"],
        device=components["device"],
    )
    if components["num_nodes"] != source_components["num_nodes"]:
        cost_model = _resize_cost_model(
            source_cost_model,
            num_waypoints=components["num_nodes"],
            device=components["device"],
        )
    else:
        cost_model = source_cost_model

    selected_takeoff_ts, takeoff_time_str = _resolve_takeoff_fields(takeoff_timestamp=selected_takeoff_ts)
    if estimated_landing_time_str is None:
        estimated_landing_time_str = (
            datetime.strptime(takeoff_time_str, "%Y-%m-%d %H:%M:%S")
            + timedelta(hours=float(config.max_flight_duration_hours))
        ).strftime("%Y-%m-%d %H:%M:%S")

    wind_model = _resolve_wind_model(config, selected_takeoff_ts)
    performance_model = components.get("performance_model") or config.initialize_performance_model()
    forward_transitions = tres_forward(
        graph=components["graph"],
        source_node_id=components["origin_node"],
        takeoff_time_str=takeoff_time_str,
        source_elevation_ft=float(config.source_elevation_ft or 0.0),
        goal_elevation_ft=float(config.goal_elevation_ft or 0.0),
        cost_model=cost_model,
        wind_model=wind_model,
        performance_model=performance_model,
        dist_matrix_np=components["dist_matrix"],
        ac_matrix_np=components["ac_matrix"],
        initial_alt_ft=float(config.initial_alt_ft or 0.0),
        delta_t_seconds=config.delta_t_seconds,
        max_flight_duration_hours=config.max_flight_duration_hours,
        etto_delta_t_seconds=config.etto_delta_t_seconds,
        max_elapsed_time_since_takeoff_hours=config.max_elapsed_time_since_takeoff_hours,
        device=components["device"],
    )[3]

    backward_closures: list[tuple] = []
    thinned_transitions: list[tuple] = []
    avg_tailwind_knots: Optional[torch.Tensor] = None
    max_rho_val: Optional[int] = None
    min_wall_clock_time_sec: Optional[float] = None

    if forward_transitions:
        backward_closures = tres_backward(
            graph=components["graph"],
            goal_node_id=components["goal_node"],
            estimated_landing_time_str=estimated_landing_time_str,
            origin_elevation_ft=float(config.source_elevation_ft or 0.0),
            destination_elevation_ft=float(config.goal_elevation_ft or 0.0),
            wind_model=wind_model,
            performance_model=performance_model,
            transitions_list=forward_transitions,
            eta_takeoff_str=takeoff_time_str,
            final_alt_ft=0.0,
            delta_t_seconds_wall_clock=config.delta_t_seconds,
            delta_t_seconds_climb=config.etto_delta_t_seconds,
            max_flight_duration_hours=config.max_flight_duration_hours,
            climb_phase_switch_allowance_climb_time_bins=config.climb_phase_switch_allowance_climb_time_bins,
            device=components["device"],
        )

    if backward_closures:
        max_rho_val = max(
            max(int(t[2]) for t in backward_closures),
            max(int(t[7]) for t in backward_closures),
        )
        thinned_transitions = thin_closures(
            components["origin_node_idx"],
            components["goal_node_idx"],
            max_rho_val,
            components["graph"],
            backward_closures,
            wallclock_time_bin_k_tolerance_s=config.delta_t_seconds,
            delta_t_seconds_wall_clock=config.delta_t_seconds,
            include_wait_edges_in_output=False,
        )
        if thinned_transitions:
            estimated_landing_ssm = datestr_to_seconds_since_midnight(estimated_landing_time_str)
            min_wall_clock_time_sec = float(estimated_landing_ssm - config.max_flight_duration_hours * 3600)
            avg_tailwind_knots = wind_model.get_average_tailwind_on_edges_knots(
                transitions=thinned_transitions,
                node_coords_deg=components["node_coords_deg"],
                min_wall_clock_time_sec=min_wall_clock_time_sec,
                delta_t_wall_clock_sec=config.delta_t_seconds,
                num_integration_steps=3,
            )

    diagnostics = _diagnose_thinning(
        backward_closures,
        origin_node_idx=int(components["origin_node_idx"]),
        goal_node_idx=int(components["goal_node_idx"]),
        max_rho=max_rho_val,
    )
    diagnostics["forward_transition_count"] = len(forward_transitions)
    diagnostics["backward_closure_count"] = len(backward_closures)
    diagnostics["thinned_transition_count"] = len(thinned_transitions)
    diagnostics["min_wall_clock_time_sec"] = min_wall_clock_time_sec

    debug_dir = run_dir / "trespass_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    safe_flight_id = selected_flight_id or "UNKNOWN_FLIGHT"
    safe_takeoff_ts = int(selected_takeoff_ts)
    fw_path = debug_dir / f"FW_{safe_flight_id}_{safe_takeoff_ts}.pkl"
    bw_path = debug_dir / f"BW_{safe_flight_id}_{safe_takeoff_ts}.pkl"
    clsr_path = debug_dir / f"CLSR_{safe_flight_id}_{safe_takeoff_ts}.pkl"
    wind_path = debug_dir / f"WIND_{safe_flight_id}_{safe_takeoff_ts}.pt"
    debug_summary_path = debug_dir / f"debug_summary_{safe_flight_id}_{safe_takeoff_ts}.json"

    # Debug outputs are intentionally write-only and never auto-loaded by this script.
    with open(fw_path, "wb") as f_fw:
        pickle.dump(forward_transitions, f_fw)
    with open(bw_path, "wb") as f_bw:
        pickle.dump(backward_closures, f_bw)
    with open(clsr_path, "wb") as f_clsr:
        pickle.dump(thinned_transitions, f_clsr)
    if avg_tailwind_knots is not None:
        torch.save(avg_tailwind_knots.detach().cpu(), wind_path)

    payload: dict[str, Any] = {
        "status": "ok",
        "debug_dir": str(debug_dir),
        "flight_id": selected_flight_id,
        "takeoff_timestamp": safe_takeoff_ts,
        "files": {
            "forward_transitions": str(fw_path),
            "backward_closures": str(bw_path),
            "thinned_transitions": str(clsr_path),
            "avg_tailwind_knots": str(wind_path) if avg_tailwind_knots is not None else None,
            "summary_json": str(debug_summary_path),
        },
        "counts": {
            "forward_transitions": len(forward_transitions),
            "backward_closures": len(backward_closures),
            "thinned_transitions": len(thinned_transitions),
            "tailwind_size": int(avg_tailwind_knots.shape[0]) if avg_tailwind_knots is not None else 0,
        },
        "graph_report": graph_report,
        "diagnostics": diagnostics,
    }

    with open(debug_summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload


def _resolve_estimated_landing_time_str(
    *,
    case_dir: Path,
    flight_id: Optional[str],
    takeoff_timestamp: Optional[int],
    max_flight_duration_hours: float,
    use_flight_metadata_duration: bool,
) -> tuple[str, float, Optional[float], str]:
    (
        _selected_flight_id,
        selected_takeoff_ts,
        _transitions,
        _tailwind,
        metadata,
    ) = load_flight_artifacts(
        str(case_dir),
        flight_id=flight_id,
        takeoff_timestamp=takeoff_timestamp,
    )
    flight_meta = metadata.get("flight_metadata", {}) if isinstance(metadata, dict) else {}

    inferred_duration_hours: Optional[float] = None
    try:
        flight_time_s = flight_meta.get("flight_time_s")
        if flight_time_s is not None:
            inferred_duration_hours = float(flight_time_s) / 3600.0
    except (TypeError, ValueError):
        inferred_duration_hours = None

    if inferred_duration_hours is None:
        try:
            takeoff_s = flight_meta.get("takeoff_time")
            landing_s = flight_meta.get("landing_time")
            if takeoff_s is not None and landing_s is not None:
                inferred_duration_hours = max(0.0, (float(landing_s) - float(takeoff_s)) / 3600.0)
        except (TypeError, ValueError):
            inferred_duration_hours = None

    effective_duration_hours = float(max_flight_duration_hours)
    duration_source = "fixed_max_flight_duration_hours"
    if use_flight_metadata_duration and inferred_duration_hours is not None and inferred_duration_hours > 0:
        effective_duration_hours = float(inferred_duration_hours)
        duration_source = "flight_metadata"

    takeoff_dt = datetime.fromtimestamp(int(selected_takeoff_ts))
    landing_dt = takeoff_dt + timedelta(hours=float(effective_duration_hours))
    return (
        landing_dt.strftime("%Y-%m-%d %H:%M:%S"),
        float(effective_duration_hours),
        None if inferred_duration_hours is None else float(inferred_duration_hours),
        duration_source,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "ERR41 full run: graph scenario + TResPASS recompute + backward SVI + sampling."
        )
    )
    parser.add_argument("--case-dir", default="data/cases/LGAV_LFPG", help="Case directory.")
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="Explicit checkpoint .pt path. If omitted, latest is auto-selected.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Directory containing checkpoint_iter_*.pt (used when checkpoint-path is omitted).",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Run folder name under <case_dir>/err41/. Defaults to timestamp.",
    )
    parser.add_argument(
        "--sectors-to-avoid",
        nargs="+",
        default=["LFEEE"],
        help="Sector IDs to avoid.",
    )
    parser.add_argument(
        "--sectors-geojson-path",
        default="data/airspace/sectors.geojson",
        help="Path to sectors GeoJSON.",
    )
    parser.add_argument(
        "--connectivity-repair-iterations",
        type=int,
        default=20,
        help="Max iterations for post-sector connectivity repair.",
    )
    parser.add_argument("--flight-id", default=None, help="Optional flight ID for dataset-backed selection.")
    parser.add_argument(
        "--takeoff-timestamp",
        type=int,
        default=None,
        help="Optional takeoff timestamp for dataset-backed selection.",
    )
    parser.add_argument("--n-samples", type=int, default=100, help="Number of sampled trajectories.")
    parser.add_argument("--policy", default="sample", choices=["sample", "greedy"], help="Sampling policy.")
    parser.add_argument(
        "--initial-k-policy",
        default="uniform",
        choices=["uniform", "latest"],
        help="Initial wall-clock bin sampling policy.",
    )
    parser.add_argument("--gamma", type=float, default=None, help="Override gamma; default reads from checkpoint/config.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed.")
    parser.add_argument(
        "--return-4d",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to reconstruct 4D trajectories.",
    )
    parser.add_argument(
        "--write-4d-csv",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to export 4D CSV outputs.",
    )
    parser.add_argument(
        "--use-cache",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use SVI cache. Default false for clean from-scratch runs.",
    )
    parser.add_argument("--output-dir", default=None, help="Override inference output dir.")
    parser.add_argument("--cache-dir", default=None, help="Override inference cache dir.")
    parser.add_argument(
        "--save-trespass-debug",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "On failure, re-run TResPASS once and save FW/BW/CLSR/WIND artifacts under "
            "<run_dir>/trespass_debug for debugging. These files are never auto-loaded."
        ),
    )
    parser.add_argument(
        "--max-flight-duration-hours",
        type=float,
        default=(3.0 + 20.0 / 60.0),
        help=(
            "Maximum flight duration used to derive estimated landing time for backward "
            "TResPASS. Defaults to 3h20m. Overriden by metadata if --use-flight-metadata-duration is True."
        ),
    )
    parser.add_argument(
        "--use-flight-metadata-duration",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use per-flight duration from flight metadata (flight_time_s or landing-takeoff) "
            "to derive estimated landing time. Falls back to --max-flight-duration-hours."
        ),
    )
    args = parser.parse_args()

    case_dir = Path(args.case_dir)
    if not case_dir.exists():
        raise FileNotFoundError(f"case_dir does not exist: {case_dir}")

    run_name = args.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = case_dir / "err41" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    resolved_checkpoint = _resolve_checkpoint(
        case_dir=case_dir,
        checkpoint_path=args.checkpoint_path,
        checkpoint_dir=args.checkpoint_dir,
    )

    output_dir = Path(args.output_dir) if args.output_dir else (run_dir / "inference_outputs")
    cache_dir = Path(args.cache_dir) if args.cache_dir else (run_dir / "inference_cache")
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    graph_scenario = GraphScenario(
        mode="edge_filter",
        sectors_to_avoid=args.sectors_to_avoid,
        sectors_geojson_path=args.sectors_geojson_path,
        post_sector_connectivity_repair=True,
        connectivity_repair_iterations=args.connectivity_repair_iterations,
    )

    summary_path = run_dir / "run_summary.json"
    error_path = run_dir / "run_error.json"
    resolved_flight_id = args.flight_id
    resolved_takeoff_timestamp = args.takeoff_timestamp
    if args.flight_id is not None and args.takeoff_timestamp is None:
        selected_flight_id, selected_takeoff_ts, _, _, _ = load_flight_artifacts(
            str(case_dir),
            flight_id=args.flight_id,
            takeoff_timestamp=None,
        )
        resolved_flight_id = selected_flight_id
        resolved_takeoff_timestamp = int(selected_takeoff_ts)

    (
        estimated_landing_time_str,
        effective_duration_hours,
        inferred_duration_hours,
        duration_source,
    ) = _resolve_estimated_landing_time_str(
        case_dir=case_dir,
        flight_id=resolved_flight_id,
        takeoff_timestamp=resolved_takeoff_timestamp,
        max_flight_duration_hours=args.max_flight_duration_hours,
        use_flight_metadata_duration=args.use_flight_metadata_duration,
    )

    try:
        result = compute_4d_path_for_dataset(
            case_dir=str(case_dir),
            checkpoint_path=str(resolved_checkpoint),
            flight_id=resolved_flight_id,
            takeoff_timestamp=resolved_takeoff_timestamp,
            gamma=args.gamma,
            n_samples=args.n_samples,
            policy=args.policy,
            initial_k_policy=args.initial_k_policy,
            return_4d=args.return_4d,
            seed=args.seed,
            output_dir=str(output_dir),
            cache_dir=str(cache_dir),
            use_cache=args.use_cache,
            write_4d_csv=args.write_4d_csv,
            graph_scenario=graph_scenario,
            transition_mode="recompute",
            estimated_landing_time_str=estimated_landing_time_str,
        )
    except Exception as exc:
        debug_payload: Optional[dict[str, Any]] = None
        if args.save_trespass_debug:
            try:
                debug_payload = _dump_trespass_debug(
                    case_dir=case_dir,
                    resolved_checkpoint=resolved_checkpoint,
                    graph_scenario=graph_scenario,
                    run_dir=run_dir,
                    flight_id=resolved_flight_id,
                    takeoff_timestamp=resolved_takeoff_timestamp,
                    estimated_landing_time_str=estimated_landing_time_str,
                )
            except Exception as debug_exc:
                debug_payload = {
                    "status": "error",
                    "error": repr(debug_exc),
                    "traceback": traceback.format_exc(),
                }

        payload = {
            "status": "error",
            "error": repr(exc),
            "case_dir": str(case_dir),
            "run_dir": str(run_dir),
            "checkpoint_path": str(resolved_checkpoint),
            "sectors_to_avoid": args.sectors_to_avoid,
            "output_dir": str(output_dir),
            "cache_dir": str(cache_dir),
            "flight_id": resolved_flight_id,
            "takeoff_timestamp": resolved_takeoff_timestamp,
            "max_flight_duration_hours": float(args.max_flight_duration_hours),
            "effective_flight_duration_hours": float(effective_duration_hours),
            "metadata_flight_duration_hours": inferred_duration_hours,
            "duration_source": duration_source,
            "estimated_landing_time_str": estimated_landing_time_str,
            "trespass_debug": debug_payload,
        }
        with open(error_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        raise

    payload = {
        "status": "ok",
        "case_dir": str(case_dir),
        "run_dir": str(run_dir),
        "checkpoint_path": str(resolved_checkpoint),
        "sectors_to_avoid": args.sectors_to_avoid,
        "output_dir": str(output_dir),
        "cache_dir": str(cache_dir),
        "flight_id": resolved_flight_id,
        "takeoff_timestamp": resolved_takeoff_timestamp,
        "max_flight_duration_hours": float(args.max_flight_duration_hours),
        "effective_flight_duration_hours": float(effective_duration_hours),
        "metadata_flight_duration_hours": inferred_duration_hours,
        "duration_source": duration_source,
        "estimated_landing_time_str": estimated_landing_time_str,
        "n_samples": len(result.samples),
        "metadata": result.metadata,
        "first_route": (result.samples[0].route if result.samples else []),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

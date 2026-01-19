from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from equinox.dp.trespass.thinning import thin_closures
from equinox.dp.trespass.tres_backward import tres_backward
from equinox.dp.trespass.tres_forward import tres_forward
from equinox.helpers.datetimeh import datestr_to_seconds_since_midnight, seconds_to_hhmmss
from equinox.sampling.trespass.inference import (
    compute_checkpoint_hash,
    graph_fingerprint,
    load_case,
    load_cached_svi,
    load_cost_model,
    load_flight_artifacts,
    _resolve_wind_model,
    resolve_checkpoint,
    route_to_4d,
    run_backward_svi,
    sample_paths,
    save_cached_svi,
    states_to_route,
)


@dataclass
class SampledPath:
    states: List[tuple]
    route: List[str]
    transition_costs: List[float]
    total_cost: float
    trajectory_4d: Optional[Dict[str, Any]]


@dataclass
class Compute4DPathResult:
    case_dir: str
    checkpoint_path: str
    flight_id: Optional[str]
    takeoff_timestamp: Optional[int]
    gamma: float
    samples: List[SampledPath]
    metadata: Dict[str, Any]


def _infer_origin_goal(components: Dict[str, Any], flight_metadata: Dict[str, Any]) -> None:
    origin = flight_metadata.get("origin") or flight_metadata.get("origin_node")
    destination = flight_metadata.get("destination") or flight_metadata.get("goal_node")
    if origin:
        components["origin_node"] = origin
    if destination:
        components["goal_node"] = destination

    if components.get("origin_node") in components["node_to_idx"]:
        components["origin_node_idx"] = components["node_to_idx"][components["origin_node"]]
    if components.get("goal_node") in components["node_to_idx"]:
        components["goal_node_idx"] = components["node_to_idx"][components["goal_node"]]


def _set_origin_goal(components: Dict[str, Any], origin_node: Optional[str], goal_node: Optional[str]) -> None:
    if origin_node:
        components["origin_node"] = origin_node
    if goal_node:
        components["goal_node"] = goal_node

    if components.get("origin_node") in components["node_to_idx"]:
        components["origin_node_idx"] = components["node_to_idx"][components["origin_node"]]
    if components.get("goal_node") in components["node_to_idx"]:
        components["goal_node_idx"] = components["node_to_idx"][components["goal_node"]]


def _seconds_to_hhmmss_int(seconds: float) -> int:
    total = int(round(seconds))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return int(f"{hours:02d}{minutes:02d}{secs:02d}")


def _write_outputs(
    output_dir: Path,
    *,
    result: Compute4DPathResult,
    write_4d_csv: bool,
    tranche_altitudes_ft: List[float],
    components: Dict[str, Any],
    config: Any,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    routes_path = output_dir / "routes.txt"
    with open(routes_path, "w", encoding="utf-8") as f:
        for sample in result.samples:
            route_line = " ".join(sample.route)
            f.write(f"{sample.total_cost},{route_line}\n")

    metadata_path = output_dir / "metadata.pt"
    torch.save(result.metadata, metadata_path)

    for idx, sample in enumerate(result.samples):
        if sample.trajectory_4d is None:
            continue
        traj_path = output_dir / f"trajectory_{idx}.pt"
        torch.save(sample.trajectory_4d, traj_path)

    if write_4d_csv:
        if not result.samples or any(sample.trajectory_4d is None for sample in result.samples):
            raise ValueError("4D CSV requested but 4D trajectories are missing.")

        from equinox.posttrain.vertical_tranchification import VerticalTranchifier

        waypoints_csv = output_dir / "shortest_path_4d_waypoints.csv"
        _write_4d_waypoints_csv(waypoints_csv, result)

        base_csv = output_dir / "shortest_path_4d_trajectories.csv"
        _write_4d_segments_csv(base_csv, result, components)

        tranchifier = VerticalTranchifier(
            performance=config.initialize_performance_model(),
            tranche_altitudes=tranche_altitudes_ft,
        )
        tranched_csv = output_dir / "shortest_path_4d_trajectories_tranched.csv"
        tranchifier.process_trajectory(str(base_csv), str(tranched_csv))


def _write_4d_waypoints_csv(output_path: Path, result: Compute4DPathResult) -> None:
    phase_map = {0: "CLIMB", 1: "CRUISE", 2: "DESCENT"}
    header = [
        "sample_id",
        "waypoint_index",
        "waypoint",
        "phase",
        "phase_name",
        "eta_seconds",
        "eta_hhmmss",
        "distance_nm",
        "altitude_ft",
        "takeoff_time_str",
        "absolute_timestamp",
        "absolute_time_str",
        "route",
    ]

    rows = []
    for sample_idx, sample in enumerate(result.samples):
        traj = sample.trajectory_4d
        if traj is None:
            continue
        waypoints = list(traj.get("waypoints", []))
        phases = traj.get("phase")
        etas = traj.get("eta")
        distances = traj.get("distance_nm")
        altitudes = traj.get("altitude_ft")
        takeoff_time_str = traj.get("takeoff_time_str")

        if hasattr(phases, "tolist"):
            phases = phases.tolist()
        if hasattr(etas, "tolist"):
            etas = etas.tolist()
        if hasattr(distances, "tolist"):
            distances = distances.tolist()
        if hasattr(altitudes, "tolist"):
            altitudes = altitudes.tolist()

        if not (len(waypoints) == len(phases) == len(etas) == len(distances) == len(altitudes)):
            raise ValueError("Trajectory arrays have inconsistent lengths.")

        route_str = " ".join(waypoints)
        for idx, waypoint in enumerate(waypoints):
            eta_seconds = float(etas[idx])
            phase_val = int(phases[idx])
            abs_ts = None
            abs_time_str = None
            if result.takeoff_timestamp is not None:
                abs_ts = int(result.takeoff_timestamp + round(eta_seconds))
                abs_time_str = datetime.fromtimestamp(abs_ts).strftime("%Y-%m-%d %H:%M:%S")

            rows.append(
                {
                    "sample_id": sample_idx,
                    "waypoint_index": idx,
                    "waypoint": waypoint,
                    "phase": phase_val,
                    "phase_name": phase_map.get(phase_val, str(phase_val)),
                    "eta_seconds": eta_seconds,
                    "eta_hhmmss": seconds_to_hhmmss(eta_seconds),
                    "distance_nm": float(distances[idx]),
                    "altitude_ft": float(altitudes[idx]),
                    "takeoff_time_str": takeoff_time_str,
                    "absolute_timestamp": abs_ts,
                    "absolute_time_str": abs_time_str,
                    "route": route_str,
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def _write_4d_segments_csv(
    output_path: Path,
    result: Compute4DPathResult,
    components: Dict[str, Any],
) -> None:
    header = [
        "segment_identifier",
        "origin_aerodrome",
        "destination_aerodrome",
        "time_begin_segment",
        "time_end_segment",
        "flight_level_begin",
        "flight_level_end",
        "latitude_begin",
        "longitude_begin",
        "latitude_end",
        "longitude_end",
        "flight_identifier",
        "route",
    ]

    graph = components["graph"]
    rows = []
    for sample_idx, sample in enumerate(result.samples):
        traj = sample.trajectory_4d
        if traj is None:
            continue
        waypoints = list(traj.get("waypoints", []))
        etas = traj.get("eta")
        altitudes = traj.get("altitude_ft")

        if hasattr(etas, "tolist"):
            etas = etas.tolist()
        if hasattr(altitudes, "tolist"):
            altitudes = altitudes.tolist()

        if len(waypoints) < 2:
            continue

        route_str = " ".join(waypoints)
        origin = waypoints[0]
        destination = waypoints[-1]
        flight_identifier = f"sample_{sample_idx}"

        for idx in range(len(waypoints) - 1):
            wp_begin = waypoints[idx]
            wp_end = waypoints[idx + 1]

            lat_begin = graph.nodes[wp_begin]["lat"] if wp_begin in graph.nodes else -1
            lon_begin = graph.nodes[wp_begin]["lon"] if wp_begin in graph.nodes else -1
            lat_end = graph.nodes[wp_end]["lat"] if wp_end in graph.nodes else -1
            lon_end = graph.nodes[wp_end]["lon"] if wp_end in graph.nodes else -1

            if hasattr(lat_begin, "item"):
                lat_begin = lat_begin.item()
            if hasattr(lon_begin, "item"):
                lon_begin = lon_begin.item()
            if hasattr(lat_end, "item"):
                lat_end = lat_end.item()
            if hasattr(lon_end, "item"):
                lon_end = lon_end.item()

            time_begin = _seconds_to_hhmmss_int(float(etas[idx]))
            time_end = _seconds_to_hhmmss_int(float(etas[idx + 1]))

            rows.append(
                {
                    "segment_identifier": f"{wp_begin}_{wp_end}",
                    "origin_aerodrome": origin,
                    "destination_aerodrome": destination,
                    "time_begin_segment": time_begin,
                    "time_end_segment": time_end,
                    "flight_level_begin": int(float(altitudes[idx]) / 100),
                    "flight_level_end": int(float(altitudes[idx + 1]) / 100),
                    "latitude_begin": lat_begin,
                    "longitude_begin": lon_begin,
                    "latitude_end": lat_end,
                    "longitude_end": lon_end,
                    "flight_identifier": flight_identifier,
                    "route": route_str,
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def compute_4d_path_for_dataset(
    *,
    case_dir: str,
    checkpoint_path: Optional[str] = None,
    flight_id: Optional[str] = None,
    takeoff_timestamp: Optional[int] = None,
    gamma: Optional[float] = None,
    n_samples: int = 1,
    policy: str = "sample",
    return_4d: bool = True,
    seed: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    output_dir: Optional[str] = None,
    cache_dir: Optional[str] = None,
    use_cache: bool = True,
    write_4d_csv: bool = False,
    tranche_altitudes_ft: Optional[List[float]] = None,
) -> Compute4DPathResult:
    if write_4d_csv and output_dir is None:
        raise ValueError("write_4d_csv requires output_dir to be set.")
    if write_4d_csv and not return_4d:
        raise ValueError("write_4d_csv requires return_4d=True.")
    if tranche_altitudes_ft is None:
        tranche_altitudes_ft = [10000, 15000, 20000, 24000, 28000, 32000]

    config, components = load_case(case_dir, device=device)

    resolved_checkpoint = resolve_checkpoint(case_dir, checkpoint_path)
    cost_model, checkpoint_metadata = load_cost_model(
        resolved_checkpoint,
        config=config,
        num_waypoints=components["num_nodes"],
        device=components["device"],
    )
    components["cost_model"] = cost_model

    flight_id, takeoff_timestamp, transitions, tailwind, flight_meta = load_flight_artifacts(
        case_dir,
        flight_id=flight_id,
        takeoff_timestamp=takeoff_timestamp,
    )
    _infer_origin_goal(components, flight_meta.get("flight_metadata", {}))

    if components.get("origin_node") is None or components.get("goal_node") is None:
        raise ValueError("Origin/goal nodes are missing; check case config or flight metadata.")

    effective_gamma = gamma
    if effective_gamma is None:
        batch_config = checkpoint_metadata.get("batch_config", {})
        effective_gamma = batch_config.get("gamma") or config.gamma or 0.1

    checkpoint_hash = compute_checkpoint_hash(resolved_checkpoint)
    svi_cache_hit = False
    svi_metadata = {}

    if use_cache:
        if cache_dir is None:
            cache_dir = os.path.join(case_dir, "inference_cache")
        cached = load_cached_svi(
            Path(cache_dir),
            checkpoint_hash=checkpoint_hash,
            flight_id=flight_id,
            takeoff_timestamp=takeoff_timestamp,
            gamma=effective_gamma,
            device=components["device"],
        )
        if cached is not None:
            V_bwd, edge_costs = cached
            svi_cache_hit = True
        else:
            V_bwd, edge_costs, svi_metadata = run_backward_svi(
                transitions,
                tailwind,
                components=components,
                cost_model=cost_model,
                gamma=effective_gamma,
            )
            save_cached_svi(
                Path(cache_dir),
                checkpoint_hash=checkpoint_hash,
                flight_id=flight_id,
                takeoff_timestamp=takeoff_timestamp,
                gamma=effective_gamma,
                V_bwd=V_bwd,
                edge_costs=edge_costs,
            )
    else:
        V_bwd, edge_costs, svi_metadata = run_backward_svi(
            transitions,
            tailwind,
            components=components,
            cost_model=cost_model,
            gamma=effective_gamma,
        )

    trajectories, cost_lists = sample_paths(
        V_bwd,
        edge_costs,
        components=components,
        n_samples=n_samples,
        gamma=effective_gamma,
        policy=policy,
        seed=seed,
    )

    samples: List[SampledPath] = []
    for trajectory, costs in zip(trajectories, cost_lists):
        route = states_to_route(trajectory)
        total_cost = float(np.sum(costs)) if costs else 0.0
        trajectory_4d = None
        if return_4d:
            trajectory_4d = route_to_4d(
                route,
                components=components,
                config=config,
                takeoff_timestamp=takeoff_timestamp,
            )
        samples.append(
            SampledPath(
                states=trajectory,
                route=route,
                transition_costs=list(costs),
                total_cost=total_cost,
                trajectory_4d=trajectory_4d,
            )
        )

    graph_sig = graph_fingerprint(components["graph"], components["node_to_idx"])
    checkpoint_sig = (
        checkpoint_metadata.get("edge_list_fingerprint")
        or checkpoint_metadata.get("graph_fingerprint")
    )
    graph_match = graph_sig == checkpoint_sig if checkpoint_sig else None

    metadata = {
        "checkpoint_hash": checkpoint_hash,
        "checkpoint_metadata_keys": sorted(checkpoint_metadata.keys()),
        "flight_metadata": flight_meta,
        "graph_fingerprint": graph_sig,
        "checkpoint_graph_fingerprint": checkpoint_sig,
        "graph_fingerprint_match": graph_match,
        "svi_cache_hit": svi_cache_hit,
        "svi_metadata": svi_metadata,
        "V_bwd_shape": tuple(V_bwd.shape),
    }

    result = Compute4DPathResult(
        case_dir=case_dir,
        checkpoint_path=str(resolved_checkpoint),
        flight_id=flight_id,
        takeoff_timestamp=takeoff_timestamp,
        gamma=effective_gamma,
        samples=samples,
        metadata=metadata,
    )

    if output_dir is not None:
        _write_outputs(
            Path(output_dir),
            result=result,
            write_4d_csv=write_4d_csv,
            tranche_altitudes_ft=tranche_altitudes_ft,
            components=components,
            config=config,
        )

    return result


def compute_4d_path_for_flight(
    *,
    case_dir: str,
    checkpoint_path: Optional[str] = None,
    origin_node: Optional[str] = None,
    goal_node: Optional[str] = None,
    flight_id: Optional[str] = None,
    takeoff_timestamp: Optional[int] = None,
    takeoff_time_str: Optional[str] = None,
    estimated_landing_time_str: Optional[str] = None,
    gamma: Optional[float] = None,
    n_samples: int = 1,
    policy: str = "sample",
    return_4d: bool = True,
    seed: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    output_dir: Optional[str] = None,
    cache_dir: Optional[str] = None,
    use_cache: bool = True,
    write_4d_csv: bool = False,
    tranche_altitudes_ft: Optional[List[float]] = None,
) -> Compute4DPathResult:
    if takeoff_timestamp is None and takeoff_time_str is None:
        raise ValueError("takeoff_timestamp or takeoff_time_str is required for flight inference.")
    if write_4d_csv and output_dir is None:
        raise ValueError("write_4d_csv requires output_dir to be set.")
    if write_4d_csv and not return_4d:
        raise ValueError("write_4d_csv requires return_4d=True.")
    if tranche_altitudes_ft is None:
        tranche_altitudes_ft = [10000, 15000, 20000, 24000, 28000, 32000]

    if takeoff_time_str is None and takeoff_timestamp is not None:
        takeoff_time_str = datetime.fromtimestamp(int(takeoff_timestamp)).strftime("%Y-%m-%d %H:%M:%S")
    if takeoff_timestamp is None and takeoff_time_str is not None:
        takeoff_dt = datetime.strptime(takeoff_time_str, "%Y-%m-%d %H:%M:%S")
        takeoff_timestamp = int(takeoff_dt.timestamp())
    if takeoff_time_str is None or takeoff_timestamp is None:
        raise ValueError("Failed to resolve takeoff_time_str and takeoff_timestamp.")

    config, components = load_case(case_dir, device=device)
    required_fields = {
        "delta_t_seconds": config.delta_t_seconds,
        "etto_delta_t_seconds": config.etto_delta_t_seconds,
        "max_elapsed_time_since_takeoff_hours": config.max_elapsed_time_since_takeoff_hours,
        "max_flight_duration_hours": config.max_flight_duration_hours,
        "climb_phase_switch_allowance_climb_time_bins": config.climb_phase_switch_allowance_climb_time_bins,
    }
    missing = [name for name, value in required_fields.items() if value is None]
    if missing:
        raise ValueError(f"Missing required config fields for flight inference: {', '.join(missing)}")
    _set_origin_goal(components, origin_node or config.origin_node, goal_node or config.goal_node)

    if components.get("origin_node") is None or components.get("goal_node") is None:
        raise ValueError("origin_node and goal_node must be provided or present in the case config.")

    resolved_checkpoint = resolve_checkpoint(case_dir, checkpoint_path)
    cost_model, checkpoint_metadata = load_cost_model(
        resolved_checkpoint,
        config=config,
        num_waypoints=components["num_nodes"],
        device=components["device"],
    )
    components["cost_model"] = cost_model

    wind_model = _resolve_wind_model(config, takeoff_timestamp)
    components["wind_model"] = wind_model
    performance_model = components.get("performance_model") or config.initialize_performance_model()

    if estimated_landing_time_str is None:
        landing_dt = datetime.strptime(takeoff_time_str, "%Y-%m-%d %H:%M:%S") + timedelta(
            hours=float(config.max_flight_duration_hours)
        )
        estimated_landing_time_str = landing_dt.strftime("%Y-%m-%d %H:%M:%S")

    _, _, _, forward_transitions = tres_forward(
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
    )

    if not forward_transitions:
        raise RuntimeError("Forward TResPASS produced no transitions.")

    closure_list = tres_backward(
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

    if not closure_list:
        raise RuntimeError("Backward TResPASS produced no closures.")

    max_rho_val = max(t[2] for t in closure_list)
    thinned_transitions = thin_closures(
        components["origin_node_idx"],
        components["goal_node_idx"],
        max_rho_val,
        components["graph"],
        closure_list,
    )
    if not thinned_transitions:
        raise RuntimeError("Thinning removed all transitions.")

    estimated_landing_ssm = datestr_to_seconds_since_midnight(estimated_landing_time_str)
    min_wall_clock_time_sec = float(estimated_landing_ssm - config.max_flight_duration_hours * 3600)
    avg_tailwind_knots = wind_model.get_average_tailwind_on_edges_knots(
        transitions=thinned_transitions,
        node_coords_deg=components["node_coords_deg"],
        min_wall_clock_time_sec=min_wall_clock_time_sec,
        delta_t_wall_clock_sec=config.delta_t_seconds,
        num_integration_steps=3,
    )

    effective_gamma = gamma
    if effective_gamma is None:
        batch_config = checkpoint_metadata.get("batch_config", {})
        effective_gamma = batch_config.get("gamma") or config.gamma or 0.1

    checkpoint_hash = compute_checkpoint_hash(resolved_checkpoint)
    svi_cache_hit = False
    svi_metadata = {}
    cache_key_id = flight_id or f"{components['origin_node']}_{components['goal_node']}"

    if use_cache:
        if cache_dir is None:
            cache_dir = os.path.join(case_dir, "inference_cache")
        cached = load_cached_svi(
            Path(cache_dir),
            checkpoint_hash=checkpoint_hash,
            flight_id=cache_key_id,
            takeoff_timestamp=takeoff_timestamp,
            gamma=effective_gamma,
            device=components["device"],
        )
        if cached is not None:
            V_bwd, edge_costs = cached
            svi_cache_hit = True
        else:
            V_bwd, edge_costs, svi_metadata = run_backward_svi(
                thinned_transitions,
                avg_tailwind_knots,
                components=components,
                cost_model=cost_model,
                gamma=effective_gamma,
            )
            save_cached_svi(
                Path(cache_dir),
                checkpoint_hash=checkpoint_hash,
                flight_id=cache_key_id,
                takeoff_timestamp=takeoff_timestamp,
                gamma=effective_gamma,
                V_bwd=V_bwd,
                edge_costs=edge_costs,
            )
    else:
        V_bwd, edge_costs, svi_metadata = run_backward_svi(
            thinned_transitions,
            avg_tailwind_knots,
            components=components,
            cost_model=cost_model,
            gamma=effective_gamma,
        )

    trajectories, cost_lists = sample_paths(
        V_bwd,
        edge_costs,
        components=components,
        n_samples=n_samples,
        gamma=effective_gamma,
        policy=policy,
        seed=seed,
    )

    samples: List[SampledPath] = []
    for trajectory, costs in zip(trajectories, cost_lists):
        route = states_to_route(trajectory)
        total_cost = float(np.sum(costs)) if costs else 0.0
        trajectory_4d = None
        if return_4d:
            trajectory_4d = route_to_4d(
                route,
                components=components,
                config=config,
                takeoff_timestamp=takeoff_timestamp,
            )
        samples.append(
            SampledPath(
                states=trajectory,
                route=route,
                transition_costs=list(costs),
                total_cost=total_cost,
                trajectory_4d=trajectory_4d,
            )
        )

    graph_sig = graph_fingerprint(components["graph"], components["node_to_idx"])
    checkpoint_sig = (
        checkpoint_metadata.get("edge_list_fingerprint")
        or checkpoint_metadata.get("graph_fingerprint")
    )
    graph_match = graph_sig == checkpoint_sig if checkpoint_sig else None

    metadata = {
        "checkpoint_hash": checkpoint_hash,
        "checkpoint_metadata_keys": sorted(checkpoint_metadata.keys()),
        "graph_fingerprint": graph_sig,
        "checkpoint_graph_fingerprint": checkpoint_sig,
        "graph_fingerprint_match": graph_match,
        "svi_cache_hit": svi_cache_hit,
        "svi_metadata": svi_metadata,
        "tres_forward_transitions": len(forward_transitions),
        "tres_backward_closures": len(closure_list),
        "thinned_transitions": len(thinned_transitions),
        "takeoff_time_str": takeoff_time_str,
        "estimated_landing_time_str": estimated_landing_time_str,
        "min_wall_clock_time_sec": min_wall_clock_time_sec,
        "V_bwd_shape": tuple(V_bwd.shape),
    }

    result = Compute4DPathResult(
        case_dir=case_dir,
        checkpoint_path=str(resolved_checkpoint),
        flight_id=flight_id,
        takeoff_timestamp=takeoff_timestamp,
        gamma=effective_gamma,
        samples=samples,
        metadata=metadata,
    )

    if output_dir is not None:
        _write_outputs(
            Path(output_dir),
            result=result,
            write_4d_csv=write_4d_csv,
            tranche_altitudes_ft=tranche_altitudes_ft,
            components=components,
            config=config,
        )

    return result


def compute_4d_path(*args, **kwargs) -> Compute4DPathResult:
    print(f"compute_4d_path implies compute_4d_path_for_dataset and will be deprecated in the future.")
    return compute_4d_path_for_dataset(*args, **kwargs)

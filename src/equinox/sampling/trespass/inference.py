from __future__ import annotations

import csv
import hashlib
import os
import pickle
import re
from dataclasses import fields, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import yaml

from equinox.config import RunConfiguration
from equinox.dp.trespass.amorwin.backward_svi_log_cost_temp import backward_soft_value_iteration
from equinox.helpers.datetimeh import datestr_to_seconds_since_midnight
from equinox.preferences.disentanglement import build_edge_list
from equinox.route.fms import get_4d_trajectory
from equinox.sampling.trespass.sampler_log import sample_tres_trajectory
from equinox.vnav.vnav_performance import get_eta_and_distance_climb, get_eta_and_distance_descent
from equinox.wind.wind_date import WindDate
from equinox.wind.wind_free import WindFree


def _filter_config_dict(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    allowed = {field.name for field in fields(RunConfiguration)}
    return {key: value for key, value in config_dict.items() if key in allowed}


def load_run_configuration(config_path: str, *, allow_legacy: bool = False) -> RunConfiguration:
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    if not isinstance(config_dict, dict):
        raise ValueError(f"Invalid configuration in {config_path}; expected a mapping.")

    schema_version = config_dict.get("schema_version")
    if schema_version is None:
        if "cost_model_version" in config_dict:
            schema_version = 2
        elif any(key.startswith("cost_model_beta") for key in config_dict):
            schema_version = 1

    if schema_version == 1 and not allow_legacy:
        raise ValueError(
            "Legacy config detected (cost_model_beta*). "
            "Use a case config with cost_model_version: lin_disent."
        )

    filtered = _filter_config_dict(config_dict)
    if filtered.get("cost_model_version") != "lin_disent":
        raise ValueError(
            f"Unsupported cost_model_version '{filtered.get('cost_model_version')}'. "
            "Only lin_disent is supported for this inference pipeline."
        )
    return RunConfiguration.from_dict(filtered)


def load_case(
    case_dir: str,
    *,
    config_path: Optional[str] = None,
    device: Optional[Union[torch.device, str]] = None,
) -> Tuple[RunConfiguration, Dict[str, Any]]:
    if config_path is None:
        config_path = os.path.join(case_dir, "default.yaml")

    config = load_run_configuration(config_path)
    components = config.initialize_all_components(manual_cost_model_init=True)

    if device is not None:
        components["device"] = torch.device(device)

    components["origin_node"] = config.origin_node
    components["goal_node"] = config.goal_node

    return config, components


def _edge_list_fingerprint(edge_u: torch.Tensor, edge_v: torch.Tensor) -> str:
    if edge_u.numel() != edge_v.numel():
        raise ValueError("edge_u and edge_v must have the same number of elements.")
    edge_u_cpu = edge_u.detach().to(device="cpu", dtype=torch.int64).contiguous()
    edge_v_cpu = edge_v.detach().to(device="cpu", dtype=torch.int64).contiguous()
    hasher = hashlib.blake2b(digest_size=16)
    hasher.update(edge_u_cpu.numpy().tobytes())
    hasher.update(b"|")
    hasher.update(edge_v_cpu.numpy().tobytes())
    return f"{edge_u_cpu.numel()}:{hasher.hexdigest()}"


def graph_fingerprint(graph: Any, node_to_idx: Dict[str, int]) -> str:
    edge_u, edge_v = build_edge_list(graph, node_to_idx)
    return _edge_list_fingerprint(edge_u, edge_v)


def resolve_checkpoint(case_dir: str, checkpoint_path: Optional[str] = None) -> Path:
    if checkpoint_path:
        resolved = Path(checkpoint_path)
        if resolved.is_dir():
            checkpoint_path = None
        else:
            if not resolved.exists():
                raise FileNotFoundError(f"Checkpoint not found: {resolved}")
            return resolved

    results_dir = Path(case_dir) / "batch_sgd_results"
    if not results_dir.exists():
        raise FileNotFoundError(f"batch_sgd_results directory not found under {case_dir}")

    checkpoint_pattern = re.compile(r"^checkpoint_iter_(\d+)\.pt$")
    best_iter = None
    best_path = None
    for path in results_dir.iterdir():
        match = checkpoint_pattern.match(path.name)
        if not match:
            continue
        iteration = int(match.group(1))
        if best_iter is None or iteration > best_iter:
            best_iter = iteration
            best_path = path

    if best_path is not None:
        return best_path

    final_results = results_dir / "final_results.pt"
    if final_results.exists():
        return final_results

    raise FileNotFoundError(f"No checkpoint_iter_*.pt or final_results.pt found in {results_dir}")


def compute_checkpoint_hash(checkpoint_path: Path) -> str:
    hasher = hashlib.blake2b(digest_size=16)
    with open(checkpoint_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _extract_state_dict(checkpoint_obj: Any) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    if isinstance(checkpoint_obj, dict):
        for key in ("model_state_dict", "final_model_state", "state_dict"):
            state_dict = checkpoint_obj.get(key)
            if isinstance(state_dict, dict):
                return state_dict, checkpoint_obj
        if all(isinstance(val, torch.Tensor) for val in checkpoint_obj.values()):
            return checkpoint_obj, {}

    raise ValueError("Checkpoint does not contain a recognizable state dict.")


def load_cost_model(
    checkpoint_path: Path,
    *,
    config: RunConfiguration,
    num_waypoints: int,
    device: torch.device,
    cost_model_version: Optional[str] = None,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    checkpoint_obj = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict, metadata = _extract_state_dict(checkpoint_obj)

    effective_version = cost_model_version or metadata.get("cost_model_version") or config.cost_model_version
    if effective_version != "lin_disent":
        raise ValueError(f"Unsupported cost_model_version '{effective_version}'.")

    cruise_speed_kts = config.cruise_speed_kts
    config_snapshot = (
        metadata.get("run_config")
        or metadata.get("config_snapshot")
        or metadata.get("config")
        or {}
    )
    if isinstance(config_snapshot, dict):
        cruise_speed_kts = config_snapshot.get("cruise_speed_kts", cruise_speed_kts)

    config_for_model = replace(config)
    config_for_model.cruise_speed_kts = cruise_speed_kts

    model = config_for_model.build_cost_model_from_state_dict(
        cost_model_state=state_dict,
        num_waypoints=num_waypoints,
        device=device,
        cost_model_version=effective_version,
    )
    return model, metadata


def _index_tres_batch_files(
    tres_dir: Path,
) -> Dict[Tuple[str, int], Tuple[Optional[Path], Optional[Path], Optional[Path]]]:
    pattern = re.compile(r"^(?P<prefix>CLSR|WIND)_(?P<flight_id>.+)_(?P<ts>\d+)\.(?P<ext>pkl|pt)$")
    index: Dict[Tuple[str, int], Tuple[Optional[Path], Optional[Path], Optional[Path]]] = {}
    for batch_dir in tres_dir.glob("batch*"):
        if not batch_dir.is_dir():
            continue
        for path in batch_dir.iterdir():
            match = pattern.match(path.name)
            if match is None:
                continue
            flight_id = match.group("flight_id")
            ts = int(match.group("ts"))
            key = (flight_id, ts)
            clsr_path, wind_path, _ = index.get(key, (None, None, None))
            if match.group("prefix") == "CLSR":
                clsr_path = path
            else:
                wind_path = path
            index[key] = (clsr_path, wind_path, batch_dir)
    return index


def _select_flight_key(
    index: Dict[Tuple[str, int], Tuple[Optional[Path], Optional[Path], Optional[Path]]],
    flight_id: Optional[str],
    takeoff_timestamp: Optional[int],
) -> Tuple[str, int]:
    keys = sorted(index.keys())
    if not keys:
        raise FileNotFoundError("No CLSR/WIND files found in tres_runs.")

    if flight_id is None and takeoff_timestamp is None:
        return keys[0]

    if flight_id is not None and takeoff_timestamp is None:
        candidates = sorted([key for key in keys if key[0] == flight_id])
        if not candidates:
            raise FileNotFoundError(f"No flights found for flight_id={flight_id}")
        return candidates[0]

    if flight_id is None and takeoff_timestamp is not None:
        candidates = sorted([key for key in keys if key[1] == takeoff_timestamp])
        if not candidates:
            raise FileNotFoundError(f"No flights found for takeoff_timestamp={takeoff_timestamp}")
        return candidates[0]

    key = (flight_id, int(takeoff_timestamp))
    if key not in index:
        raise FileNotFoundError(f"Flight artifacts not found for {flight_id}_{takeoff_timestamp}")
    return key


def _load_flight_metadata(case_dir: str, flight_id: str, takeoff_timestamp: int) -> Dict[str, Any]:
    tres_dir = Path(case_dir) / "tres_runs"
    csv_paths = list(tres_dir.glob("batch*/flights.csv"))
    csv_paths.append(tres_dir / "all_routes_feasibly_snapped.csv")

    ts_keys = ("takeoff_time", "takeoff_timestamp", "takeoff")
    for csv_path in csv_paths:
        if not csv_path.exists():
            continue
        try:
            with open(csv_path, newline="", encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    if row.get("flight_id") != flight_id:
                        continue
                    for key in ts_keys:
                        if key in row:
                            try:
                                row_ts = int(float(row[key]))
                            except (TypeError, ValueError):
                                row_ts = None
                            if row_ts == takeoff_timestamp:
                                row["source_csv"] = str(csv_path)
                                return row
        except Exception:
            continue

    return {}


def load_flight_artifacts(
    case_dir: str,
    *,
    flight_id: Optional[str] = None,
    takeoff_timestamp: Optional[int] = None,
) -> Tuple[str, int, list, torch.Tensor, Dict[str, Any]]:
    tres_dir = Path(case_dir) / "tres_runs"
    if not tres_dir.exists():
        raise FileNotFoundError(f"tres_runs directory not found under {case_dir}")

    index = _index_tres_batch_files(tres_dir)
    selected_flight_id, selected_ts = _select_flight_key(index, flight_id, takeoff_timestamp)
    clsr_path, wind_path, batch_dir = index[(selected_flight_id, selected_ts)]
    if clsr_path is None or wind_path is None:
        raise FileNotFoundError(
            f"Missing CLSR/WIND for {selected_flight_id}_{selected_ts} in {batch_dir}"
        )

    with open(clsr_path, "rb") as f:
        thinned_transitions = pickle.load(f)
    avg_tailwind_knots = torch.load(wind_path, weights_only=False)

    if len(thinned_transitions) != int(avg_tailwind_knots.shape[0]):
        raise ValueError(
            f"Tailwind length mismatch for {selected_flight_id}_{selected_ts}: "
            f"{len(thinned_transitions)} transitions vs {avg_tailwind_knots.shape[0]} tailwind values."
        )

    metadata = {
        "clsr_path": str(clsr_path),
        "wind_path": str(wind_path),
        "batch_dir": str(batch_dir) if batch_dir is not None else None,
        "flight_metadata": _load_flight_metadata(case_dir, selected_flight_id, selected_ts),
    }
    return selected_flight_id, selected_ts, thinned_transitions, avg_tailwind_knots, metadata


def run_backward_svi(
    thinned_transitions: list,
    avg_tailwind_knots: torch.Tensor,
    *,
    components: Dict[str, Any],
    cost_model: torch.nn.Module,
    gamma: float,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    if not thinned_transitions:
        raise ValueError("No thinned transitions provided for backward SVI.")

    device = components["device"]
    dist_matrix = torch.tensor(components["dist_matrix"], dtype=torch.float32, device=device)
    ac_matrix = torch.tensor(components["ac_matrix"], dtype=torch.float32, device=device)

    max_k_val = max(max(t[1] for t in thinned_transitions), max(t[6] for t in thinned_transitions))
    max_rho_val = max(max(t[2] for t in thinned_transitions), max(t[7] for t in thinned_transitions))
    max_phase_val = max(max(t[4] for t in thinned_transitions), max(t[9] for t in thinned_transitions))

    num_time_bins_wall_clock = max_k_val + 1
    num_rho_bins = max_rho_val + 1
    num_phases = max_phase_val + 1

    V_bwd, edge_costs = backward_soft_value_iteration(
        state_transitions=thinned_transitions,
        avg_tailwind_knots_per_transition=avg_tailwind_knots.to(device),
        G=components["graph"],
        idx_to_node=components["idx_to_node"],
        goal_node_idx=components["goal_node_idx"],
        cost_model=cost_model,
        num_nodes=components["num_nodes"],
        num_time_bins_wall_clock=num_time_bins_wall_clock,
        num_rho_bins=num_rho_bins,
        num_phases=num_phases,
        distance_matrix_d=dist_matrix,
        airspace_charge_matrix_ac=ac_matrix,
        device=device,
        gamma=gamma,
        verbose=False,
    )

    metadata = {
        "num_time_bins_wall_clock": num_time_bins_wall_clock,
        "num_rho_bins": num_rho_bins,
        "num_phases": num_phases,
    }
    return V_bwd, edge_costs, metadata


def _cache_key(checkpoint_hash: str, flight_id: str, takeoff_timestamp: int, gamma: float) -> str:
    gamma_str = str(gamma).replace(".", "p")
    return f"{checkpoint_hash}_{flight_id}_{takeoff_timestamp}_g{gamma_str}"


def load_cached_svi(
    cache_dir: Path,
    *,
    checkpoint_hash: str,
    flight_id: str,
    takeoff_timestamp: int,
    gamma: float,
    device: torch.device,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"svi_{_cache_key(checkpoint_hash, flight_id, takeoff_timestamp, gamma)}.pt"
    if not cache_path.exists():
        return None

    cache_obj = torch.load(cache_path, map_location="cpu", weights_only=False)
    if not isinstance(cache_obj, dict):
        return None

    V_bwd = cache_obj.get("V_bwd")
    edge_costs = cache_obj.get("edge_costs")
    if V_bwd is None or edge_costs is None:
        return None

    return V_bwd.to(device), edge_costs.to(device)


def save_cached_svi(
    cache_dir: Path,
    *,
    checkpoint_hash: str,
    flight_id: str,
    takeoff_timestamp: int,
    gamma: float,
    V_bwd: torch.Tensor,
    edge_costs: torch.Tensor,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"svi_{_cache_key(checkpoint_hash, flight_id, takeoff_timestamp, gamma)}.pt"
    torch.save(
        {
            "V_bwd": V_bwd.detach().cpu(),
            "edge_costs": edge_costs.detach().cpu(),
        },
        cache_path,
    )
    return cache_path


def _resolve_wind_model(config: RunConfiguration, takeoff_timestamp: Optional[int]) -> Any:
    date_str = config.wind_date
    if date_str is None and takeoff_timestamp is not None:
        date_str = datetime.fromtimestamp(int(takeoff_timestamp)).strftime("%Y-%m-%d")
    if date_str:
        try:
            return WindDate(date_str=date_str, data_dir=config.wind_data_dir)
        except Exception:
            return WindFree()
    return WindFree()


def route_to_4d(
    route: Sequence[str],
    *,
    components: Dict[str, Any],
    config: RunConfiguration,
    takeoff_timestamp: Optional[int],
) -> Dict[str, Any]:
    if takeoff_timestamp is None:
        raise ValueError("takeoff_timestamp is required to build a 4D trajectory.")

    takeoff_dt = datetime.fromtimestamp(int(takeoff_timestamp))
    takeoff_time_str = takeoff_dt.strftime("%Y-%m-%d %H:%M:%S")
    takeoff_ssm = datestr_to_seconds_since_midnight(takeoff_time_str)

    performance_model = components.get("performance_model") or config.initialize_performance_model()
    origin_elev = float(config.source_elevation_ft or 0.0)
    dest_elev = float(config.goal_elevation_ft or 0.0)

    climb_perf = get_eta_and_distance_climb(performance_model, origin_elev)
    descent_perf = get_eta_and_distance_descent(performance_model, dest_elev)
    wind_model = _resolve_wind_model(config, takeoff_timestamp)

    waypoint_string = " ".join(route)
    phase, eta, distance, altitude = get_4d_trajectory(
        waypoint_string,
        components["graph"],
        origin_elev,
        dest_elev,
        takeoff_ssm,
        climb_perf,
        descent_perf,
        wind_model,
    )

    return {
        "waypoints": list(route),
        "takeoff_time_str": takeoff_time_str,
        "phase": phase,
        "eta": eta,
        "distance_nm": distance,
        "altitude_ft": altitude,
    }


def sample_paths(
    V_bwd: torch.Tensor,
    edge_costs_uv: torch.Tensor,
    *,
    components: Dict[str, Any],
    n_samples: int,
    gamma: float,
    policy: str,
    initial_k_policy: str = "uniform",
    seed: Optional[int] = None,
    max_steps: int = 200,
    initial_rho: Optional[int] = None,
    initial_phase: int = 0,
) -> Tuple[List[list], List[list]]:
    rng = np.random.default_rng(seed) if seed is not None else None

    if initial_rho is None:
        initial_rho = V_bwd.shape[2] - 1 if V_bwd.shape[2] > 0 else 0

    edge_costs_uv = edge_costs_uv.coalesce()

    trajectories = []
    trajectory_costs = []
    for _ in range(n_samples):
        trajectory, costs = sample_tres_trajectory(
            G=components["graph"],
            node_to_idx=components["node_to_idx"],
            idx_to_node=components["idx_to_node"],
            origin_node_id=components["origin_node"],
            goal_node_id=components["goal_node"],
            initial_rho=initial_rho,
            initial_phase=initial_phase,
            soft_cost_to_go=V_bwd,
            edge_costs_uv=edge_costs_uv,
            gamma=gamma,
            max_steps=max_steps,
            policy=policy,
            initial_k_policy=initial_k_policy,
            rng=rng,
        )
        trajectories.append(trajectory)
        trajectory_costs.append(costs)

    return trajectories, trajectory_costs


def states_to_route(states: Sequence[Tuple[str, int, int, int]]) -> List[str]:
    route = []
    for node_id, _, _, _ in states:
        if not route or route[-1] != node_id:
            route.append(node_id)
    return route

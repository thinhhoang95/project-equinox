from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
import csv
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import networkx as nx
import numpy as np
import torch

from equinox.posttrain.preference_remap import remap_checkpoint_preferences_to_graph
from equinox.dp.trespass.thinning import thin_closures
from equinox.dp.trespass.tres_backward import tres_backward
from equinox.dp.trespass.tres_forward import tres_forward
from equinox.helpers.datetimeh import datestr_to_seconds_since_midnight, seconds_to_hhmmss
from equinox.training.prep.remove_edges_for_sectors import remove_edges_through_sectors
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


@dataclass
class GraphRegenerateArgs:
    nodes_only_graph_path: str
    routes_dir: str
    source_id: Optional[str] = None
    destination_id: Optional[str] = None
    minimum_detour_allowed: float = 0.025
    n_iter: int = 10
    max_allowed_deviation_angle: float = 90.0
    delete_isolated_nodes: bool = False
    remove_collinear_edges_option: bool = True
    remove_backtracking_edges_option: bool = True
    remove_unreachable_nodes_option: bool = True
    make_acyclic_option: bool = True
    improve_connectivity_option: bool = True
    charges_csv_path: str = "data/ufir/fir_charges.csv"


@dataclass
class GraphScenario:
    sectors_to_avoid: Optional[List[str]] = None
    sectors_geojson_path: str = "data/airspace/sectors.geojson"
    mode: Literal["edge_filter", "regenerate"] = "edge_filter"
    regenerate_args: Optional[GraphRegenerateArgs] = None
    post_sector_connectivity_repair: bool = False
    connectivity_repair_iterations: int = 20


@dataclass
class PreferenceScenario:
    source: Literal["checkpoint"] = "checkpoint"
    remap_method: Literal["node_nn"] = "node_nn"
    max_nn_distance_nm: Optional[float] = None
    unmatched_value: float = 0.0
    zero_all: bool = False
    zero_edges: Optional[List[Tuple[str, str]]] = None


def _stable_hash(payload: Dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.blake2b(encoded, digest_size=10).hexdigest()


def _with_scenario_cache_key(base_key: str, scenario_hash: Optional[str]) -> str:
    if not scenario_hash:
        return base_key
    return f"{base_key}__scn_{scenario_hash}"


def _ensure_required_flight_config_fields(config: Any) -> None:
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


def _resolve_takeoff_fields(
    *,
    takeoff_timestamp: Optional[int] = None,
    takeoff_time_str: Optional[str] = None,
) -> Tuple[int, str]:
    if takeoff_time_str is None and takeoff_timestamp is None:
        raise ValueError("takeoff_timestamp or takeoff_time_str is required.")
    if takeoff_time_str is None and takeoff_timestamp is not None:
        takeoff_time_str = datetime.fromtimestamp(int(takeoff_timestamp)).strftime("%Y-%m-%d %H:%M:%S")
    if takeoff_timestamp is None and takeoff_time_str is not None:
        takeoff_dt = datetime.strptime(takeoff_time_str, "%Y-%m-%d %H:%M:%S")
        takeoff_timestamp = int(takeoff_dt.timestamp())
    if takeoff_time_str is None or takeoff_timestamp is None:
        raise ValueError("Failed to resolve takeoff_time_str and takeoff_timestamp.")
    return int(takeoff_timestamp), takeoff_time_str


def _rebuild_graph_components(components: Dict[str, Any], graph: nx.DiGraph) -> Dict[str, Any]:
    out = dict(components)
    node_to_idx = {node: i for i, node in enumerate(graph.nodes())}
    idx_to_node = {i: node for node, i in node_to_idx.items()}
    node_coords_deg = torch.zeros((len(node_to_idx), 2), dtype=torch.float32)
    for node, idx in node_to_idx.items():
        node_coords_deg[idx, 0] = float(graph.nodes[node]["lat"])
        node_coords_deg[idx, 1] = float(graph.nodes[node]["lon"])
    out["graph"] = graph
    out["node_to_idx"] = node_to_idx
    out["idx_to_node"] = idx_to_node
    out["node_coords_deg"] = node_coords_deg
    out["num_nodes"] = len(node_to_idx)
    if out.get("origin_node") in node_to_idx:
        out["origin_node_idx"] = node_to_idx[out["origin_node"]]
    if out.get("goal_node") in node_to_idx:
        out["goal_node_idx"] = node_to_idx[out["goal_node"]]
    return out


def _build_charge_matrix_for_graph(graph: nx.DiGraph, charges_csv_path: str) -> np.ndarray:
    import pandas as pd

    from equinox.feateng.airspace_charges import compute_charges_for_graph

    charges_df = pd.read_csv(charges_csv_path)
    charge_graph = compute_charges_for_graph(graph, charges_df)
    node_to_idx = {node: i for i, node in enumerate(graph.nodes())}
    matrix = np.zeros((len(node_to_idx), len(node_to_idx)), dtype=np.float32)
    for u, v, data in charge_graph.edges(data=True):
        matrix[node_to_idx[u], node_to_idx[v]] = float(data.get("airspace_charge", 0.0))
    return matrix


def _validate_graph_for_inference(graph: nx.DiGraph, origin_node: str, goal_node: str) -> None:
    if origin_node not in graph.nodes or goal_node not in graph.nodes:
        raise ValueError("origin/goal node is missing from the scenario graph.")
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("Scenario graph must remain acyclic for backward SVI.")
    if not nx.has_path(graph, origin_node, goal_node):
        raise ValueError("No path from origin to goal after applying graph scenario.")


def _compute_initial_bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
    lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)
    dlon = lon2_rad - lon1_rad
    x = math.sin(dlon) * math.cos(lat2_rad)
    y = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def _nodes_that_cannot_reach_goal(graph: nx.DiGraph, goal_node: str) -> List[str]:
    reachable = set(nx.ancestors(graph, goal_node))
    reachable.add(goal_node)
    return list(set(graph.nodes()) - reachable)


def _nodes_source_cannot_reach(graph: nx.DiGraph, source_node: str) -> List[str]:
    reachable = set(nx.descendants(graph, source_node))
    reachable.add(source_node)
    return list(set(graph.nodes()) - reachable)


def _repair_connectivity_after_sector_removal(
    graph: nx.DiGraph,
    *,
    source_id: str,
    destination_id: str,
    n_iter: int,
) -> Dict[str, Any]:
    from equinox.training.prep.graph_scripts.improve_connectivity import improve_graph_connectivity

    if source_id not in graph.nodes or destination_id not in graph.nodes:
        return {
            "enabled": True,
            "applied": False,
            "reason": "origin_or_goal_missing_in_graph",
            "iterations": int(n_iter),
            "edges_added_total": 0,
        }

    main_bearing = _compute_initial_bearing_deg(
        float(graph.nodes[source_id]["lat"]),
        float(graph.nodes[source_id]["lon"]),
        float(graph.nodes[destination_id]["lat"]),
        float(graph.nodes[destination_id]["lon"]),
    )
    edges_added_total = 0
    iterations_executed = 0
    for _ in range(max(0, int(n_iter))):
        orphan_goal_nodes = _nodes_that_cannot_reach_goal(graph, destination_id)
        orphan_source_nodes = _nodes_source_cannot_reach(graph, source_id)
        edge_before = graph.number_of_edges()
        graph = improve_graph_connectivity(
            graph,
            orphan_goal_nodes,
            orphan_source_nodes,
            main_bearing=main_bearing,
            radius_nm=400,
            bearing_tolerance_deg=85,
            n_degree_connections=4,
            n_nearest_connections=4,
        )
        edge_after = graph.number_of_edges()
        added = max(0, edge_after - edge_before)
        edges_added_total += added
        iterations_executed += 1
        if added == 0:
            break

    return {
        "enabled": True,
        "applied": True,
        "iterations_requested": int(n_iter),
        "iterations_executed": int(iterations_executed),
        "edges_added_total": int(edges_added_total),
        "remaining_goal_orphans": len(_nodes_that_cannot_reach_goal(graph, destination_id)),
        "remaining_source_orphans": len(_nodes_source_cannot_reach(graph, source_id)),
    }


def _prepare_graph_for_scenario(
    *,
    config: Any,
    components: Dict[str, Any],
    graph_scenario: Optional[GraphScenario],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if graph_scenario is None:
        return components, {"applied": False}

    mode = graph_scenario.mode
    if mode not in {"edge_filter", "regenerate"}:
        raise ValueError(f"Unsupported graph scenario mode '{mode}'.")

    if mode == "edge_filter":
        graph = components["graph"].copy()
        removed_edges = 0
        connectivity_repair_report: Dict[str, Any] = {"enabled": False}
        if graph_scenario.sectors_to_avoid:
            before = graph.number_of_edges()
            graph, _ = remove_edges_through_sectors(
                graph,
                graph_scenario.sectors_to_avoid,
                graph_scenario.sectors_geojson_path,
                output_dir=None,
            )
            removed_edges = before - graph.number_of_edges()
            if graph_scenario.post_sector_connectivity_repair:
                connectivity_repair_report = _repair_connectivity_after_sector_removal(
                    graph,
                    source_id=components["origin_node"],
                    destination_id=components["goal_node"],
                    n_iter=graph_scenario.connectivity_repair_iterations,
                )
        updated = dict(components)
        updated["graph"] = graph
        _validate_graph_for_inference(graph, updated["origin_node"], updated["goal_node"])
        return updated, {
            "applied": True,
            "mode": mode,
            "sectors_to_avoid": graph_scenario.sectors_to_avoid or [],
            "sectors_geojson_path": graph_scenario.sectors_geojson_path,
            "edges_removed": removed_edges,
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "connectivity_repair": connectivity_repair_report,
        }

    # mode == regenerate
    regen_args = graph_scenario.regenerate_args
    if regen_args is None:
        raise ValueError("graph_scenario.mode='regenerate' requires regenerate_args.")

    from equinox.feateng.distance import haversine_distance_matrix
    from equinox.training.prep.prep_graph import process_and_save_graph

    source_id = regen_args.source_id or components.get("origin_node") or config.origin_node
    destination_id = regen_args.destination_id or components.get("goal_node") or config.goal_node
    if source_id is None or destination_id is None:
        raise ValueError("source_id/destination_id are required for regenerate graph mode.")

    graph = process_and_save_graph(
        nodes_only_graph_path=regen_args.nodes_only_graph_path,
        source_id=source_id,
        destination_id=destination_id,
        routes_dir=regen_args.routes_dir,
        delete_isolated_nodes=regen_args.delete_isolated_nodes,
        minimum_detour_allowed=regen_args.minimum_detour_allowed,
        n_iter=regen_args.n_iter,
        max_allowed_deviation_angle=regen_args.max_allowed_deviation_angle,
        output_path=None,
        remove_collinear_edges_option=regen_args.remove_collinear_edges_option,
        remove_backtracking_edges_option=regen_args.remove_backtracking_edges_option,
        remove_unreachable_nodes_option=regen_args.remove_unreachable_nodes_option,
        make_acyclic_option=regen_args.make_acyclic_option,
        sectors_to_avoid=graph_scenario.sectors_to_avoid or [],
        improve_connectivity_option=regen_args.improve_connectivity_option,
    )
    updated = _rebuild_graph_components(components, graph)
    updated["dist_matrix"] = haversine_distance_matrix(graph)
    updated["ac_matrix"] = _build_charge_matrix_for_graph(graph, regen_args.charges_csv_path)
    _validate_graph_for_inference(graph, updated["origin_node"], updated["goal_node"])
    return updated, {
        "applied": True,
        "mode": mode,
        "sectors_to_avoid": graph_scenario.sectors_to_avoid or [],
        "sectors_geojson_path": graph_scenario.sectors_geojson_path,
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "charges_csv_path": regen_args.charges_csv_path,
    }


def _resize_cost_model(
    source_cost_model: torch.nn.Module,
    *,
    num_waypoints: int,
    device: torch.device,
) -> torch.nn.Module:
    from equinox.cost.cost_linear_disentangled import CostLinearDisentangled

    if not isinstance(source_cost_model, CostLinearDisentangled):
        raise TypeError("Only CostLinearDisentangled is supported for resizing.")
    target = CostLinearDisentangled(
        common_weights=tuple(float(v) for v in source_cost_model.common_weights.detach().cpu().tolist()),
        preference_weights=float(source_cost_model.preference_weights),
        alpha_pref_reg=float(source_cost_model.alpha_pref_reg.detach().cpu().item()),
        num_waypoints=num_waypoints,
        device=device,
        cruise_speed_kts=float(source_cost_model.cruise_speed_kts),
    )
    with torch.no_grad():
        target.common_weights.copy_(
            source_cost_model.common_weights.detach().to(device=target.common_weights.device)
        )
        target.preference_matrix_p.zero_()
    return target


def _apply_preference_scenario(
    *,
    cost_model: torch.nn.Module,
    source_preference_matrix: Optional[torch.Tensor],
    source_graph: nx.DiGraph,
    source_node_to_idx: Dict[str, int],
    target_graph: nx.DiGraph,
    target_node_to_idx: Dict[str, int],
    preference_scenario: PreferenceScenario,
) -> Dict[str, Any]:
    if preference_scenario.source != "checkpoint":
        raise ValueError(f"Unsupported preference source '{preference_scenario.source}'.")
    if preference_scenario.remap_method != "node_nn":
        raise ValueError(f"Unsupported preference remap_method '{preference_scenario.remap_method}'.")
    preference_matrix = source_preference_matrix
    if preference_matrix is None:
        preference_matrix = cost_model.preference_matrix_p.detach()

    pref_matrix, report = remap_checkpoint_preferences_to_graph(
        preference_matrix=preference_matrix,
        source_graph=source_graph,
        source_node_to_idx=source_node_to_idx,
        target_graph=target_graph,
        target_node_to_idx=target_node_to_idx,
        max_nn_distance_nm=preference_scenario.max_nn_distance_nm,
        unmatched_value=preference_scenario.unmatched_value,
        zero_all=preference_scenario.zero_all,
        zero_edges=preference_scenario.zero_edges,
    )
    with torch.no_grad():
        cost_model.preference_matrix_p.copy_(
            pref_matrix.to(
                device=cost_model.preference_matrix_p.device,
                dtype=cost_model.preference_matrix_p.dtype,
            )
        )
    report["scenario"] = {
        "source": preference_scenario.source,
        "remap_method": preference_scenario.remap_method,
        "max_nn_distance_nm": preference_scenario.max_nn_distance_nm,
        "unmatched_value": preference_scenario.unmatched_value,
        "zero_all": preference_scenario.zero_all,
        "zero_edges_count": len(preference_scenario.zero_edges or []),
    }
    return report


def _compute_transitions_for_intent(
    *,
    config: Any,
    components: Dict[str, Any],
    cost_model: torch.nn.Module,
    takeoff_time_str: str,
    takeoff_timestamp: int,
    estimated_landing_time_str: Optional[str] = None,
) -> Dict[str, Any]:
    _ensure_required_flight_config_fields(config)
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

    max_rho_val = max(
        max(t[2] for t in closure_list),
        max(t[7] for t in closure_list),
    )
    thinned_transitions = thin_closures(
        components["origin_node_idx"],
        components["goal_node_idx"],
        max_rho_val,
        components["graph"],
        closure_list,
        wallclock_time_bin_k_tolerance_s=config.delta_t_seconds,
        delta_t_seconds_wall_clock=config.delta_t_seconds,
        include_wait_edges_in_output=False,
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
    return {
        "thinned_transitions": thinned_transitions,
        "avg_tailwind_knots": avg_tailwind_knots,
        "forward_transitions": len(forward_transitions),
        "backward_closures": len(closure_list),
        "thinned_count": len(thinned_transitions),
        "takeoff_time_str": takeoff_time_str,
        "estimated_landing_time_str": estimated_landing_time_str,
        "min_wall_clock_time_sec": min_wall_clock_time_sec,
    }


def _run_inference_from_transitions(
    *,
    case_dir: str,
    resolved_checkpoint: Path,
    config: Any,
    components: Dict[str, Any],
    cost_model: torch.nn.Module,
    checkpoint_metadata: Dict[str, Any],
    transitions: List[tuple],
    tailwind: torch.Tensor,
    flight_id: Optional[str],
    takeoff_timestamp: Optional[int],
    gamma: Optional[float],
    n_samples: int,
    policy: str,
    initial_k_policy: str,
    return_4d: bool,
    seed: Optional[int],
    cache_dir: Optional[str],
    use_cache: bool,
    output_dir: Optional[str],
    write_4d_csv: bool,
    tranche_altitudes_ft: List[float],
    cache_key_id: Optional[str] = None,
    use_adaptive_initial_state: bool = False,
    metadata_extra: Optional[Dict[str, Any]] = None,
) -> Compute4DPathResult:
    effective_gamma = gamma
    if effective_gamma is None:
        batch_config = checkpoint_metadata.get("batch_config", {})
        effective_gamma = batch_config.get("gamma") or config.gamma or 0.1

    checkpoint_hash = compute_checkpoint_hash(resolved_checkpoint)
    svi_cache_hit = False
    svi_metadata: Dict[str, Any] = {}
    effective_cache_key = cache_key_id or flight_id or f"{components['origin_node']}_{components['goal_node']}"

    if use_cache:
        if cache_dir is None:
            cache_dir = os.path.join(case_dir, "inference_cache")
        cached = load_cached_svi(
            Path(cache_dir),
            checkpoint_hash=checkpoint_hash,
            flight_id=effective_cache_key,
            takeoff_timestamp=int(takeoff_timestamp) if takeoff_timestamp is not None else 0,
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
                flight_id=effective_cache_key,
                takeoff_timestamp=int(takeoff_timestamp) if takeoff_timestamp is not None else 0,
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

    sample_kwargs: Dict[str, Any] = {}
    if use_adaptive_initial_state:
        initial_rho, initial_phase = _select_initial_state(
            V_bwd,
            origin_node_idx=components["origin_node_idx"],
            phase_order=[0, 1, 2],
        )
        sample_kwargs["initial_rho"] = initial_rho
        sample_kwargs["initial_phase"] = initial_phase

    trajectories, cost_lists = sample_paths(
        V_bwd,
        edge_costs,
        components=components,
        n_samples=n_samples,
        gamma=effective_gamma,
        policy=policy,
        initial_k_policy=initial_k_policy,
        seed=seed,
        **sample_kwargs,
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
        "V_bwd_shape": tuple(V_bwd.shape),
    }
    if metadata_extra:
        metadata.update(metadata_extra)

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
            write_pt_files=False,
        )
    return result


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


def _resolve_auto_cruise_altitude(
    config: Any,
    components: Dict[str, Any],
    flight_metadata: Dict[str, Any],
    *,
    flight_id: Optional[str],
    takeoff_timestamp: Optional[int],
) -> None:
    cruise_altitude_setting = config.cruise_altitude_ft
    if not (isinstance(cruise_altitude_setting, str) and cruise_altitude_setting.strip().lower() == "auto"):
        return

    if not flight_id or takeoff_timestamp is None:
        raise ValueError("cruise_altitude_ft=auto requires flight_id and takeoff_timestamp.")

    cruise_altitude_m = flight_metadata.get("cruise_altitude") if flight_metadata else None
    if cruise_altitude_m is None or str(cruise_altitude_m).strip() == "":
        raise ValueError(
            f"cruise_altitude_ft=auto but cruise_altitude missing for {flight_id}_{takeoff_timestamp}."
        )
    try:
        cruise_altitude_m = float(cruise_altitude_m)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid cruise_altitude '{cruise_altitude_m}' for {flight_id}_{takeoff_timestamp}."
        ) from exc

    config.cruise_altitude_ft = cruise_altitude_m * 3.280839895
    components["performance_model"] = config.initialize_performance_model()


def _seconds_to_hhmmss_int(seconds: float) -> int:
    total = int(round(seconds))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return int(f"{hours:02d}{minutes:02d}{secs:02d}")


def _select_initial_state(
    V_bwd: torch.Tensor,
    *,
    origin_node_idx: int,
    phase_order: Optional[List[int]] = None,
) -> tuple[int, int]:
    if V_bwd.ndim < 4:
        raise ValueError("V_bwd must be 4D (node, k, rho, phase).")
    num_rho = V_bwd.shape[2]
    num_phase = V_bwd.shape[3]
    if num_rho == 0 or num_phase == 0:
        return 0, 0
    if phase_order is None:
        phase_order = list(range(num_phase))
    for phase in phase_order:
        for rho in range(num_rho - 1, -1, -1):
            cost_slice = V_bwd[origin_node_idx, :, rho, phase]
            if torch.isfinite(cost_slice).any():
                return rho, phase
    return 0, 0


def _write_outputs(
    output_dir: Path,
    *,
    result: Compute4DPathResult,
    write_4d_csv: bool,
    tranche_altitudes_ft: List[float],
    components: Dict[str, Any],
    config: Any,
    write_pt_files: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    routes_path = output_dir / "routes.txt"
    with open(routes_path, "w", encoding="utf-8") as f:
        for sample in result.samples:
            route_line = " ".join(sample.route)
            f.write(f"{sample.total_cost},{route_line}\n")

    if write_pt_files:
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
    initial_k_policy: str = "uniform",
    return_4d: bool = True,
    seed: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    output_dir: Optional[str] = None,
    cache_dir: Optional[str] = None,
    use_cache: bool = True,
    write_4d_csv: bool = False,
    tranche_altitudes_ft: Optional[List[float]] = None,
    graph_scenario: Optional[GraphScenario] = None,
    preference_scenario: Optional[PreferenceScenario] = None,
    transition_mode: Literal["dataset", "recompute"] = "dataset",
    estimated_landing_time_str: Optional[str] = None,
) -> Compute4DPathResult:
    if write_4d_csv and output_dir is None:
        raise ValueError("write_4d_csv requires output_dir to be set.")
    if write_4d_csv and not return_4d:
        raise ValueError("write_4d_csv requires return_4d=True.")
    if tranche_altitudes_ft is None:
        tranche_altitudes_ft = [10000, 15000, 20000, 24000, 28000, 32000]

    if transition_mode not in {"dataset", "recompute"}:
        raise ValueError("transition_mode must be one of {'dataset', 'recompute'}.")

    config, source_components = load_case(case_dir, device=device)
    flight_id, takeoff_timestamp, dataset_transitions, dataset_tailwind, flight_meta = load_flight_artifacts(
        case_dir,
        flight_id=flight_id,
        takeoff_timestamp=takeoff_timestamp,
    )
    _infer_origin_goal(source_components, flight_meta.get("flight_metadata", {}))
    _resolve_auto_cruise_altitude(
        config,
        source_components,
        flight_meta.get("flight_metadata", {}),
        flight_id=flight_id,
        takeoff_timestamp=takeoff_timestamp,
    )
    if source_components.get("origin_node") is None or source_components.get("goal_node") is None:
        raise ValueError("Origin/goal nodes are missing; check case config or flight metadata.")
    components, graph_report = _prepare_graph_for_scenario(
        config=config,
        components=source_components,
        graph_scenario=graph_scenario,
    )

    resolved_checkpoint = resolve_checkpoint(case_dir, checkpoint_path)
    source_cost_model, checkpoint_metadata = load_cost_model(
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
    components["cost_model"] = cost_model

    preference_report: Dict[str, Any] = {"applied": False}
    if preference_scenario is not None:
        preference_report = _apply_preference_scenario(
            cost_model=cost_model,
            source_preference_matrix=source_cost_model.preference_matrix_p.detach(),
            source_graph=source_components["graph"],
            source_node_to_idx=source_components["node_to_idx"],
            target_graph=components["graph"],
            target_node_to_idx=components["node_to_idx"],
            preference_scenario=preference_scenario,
        )
        preference_report["applied"] = True

    auto_switched_to_recompute = False
    effective_transition_mode = transition_mode
    if (
        effective_transition_mode == "dataset"
        and (graph_scenario is not None or preference_scenario is not None)
    ):
        effective_transition_mode = "recompute"
        auto_switched_to_recompute = True

    transition_metadata: Dict[str, Any] = {"transition_mode": effective_transition_mode}
    if effective_transition_mode == "recompute":
        takeoff_timestamp, takeoff_time_str = _resolve_takeoff_fields(
            takeoff_timestamp=takeoff_timestamp,
        )
        transition_payload = _compute_transitions_for_intent(
            config=config,
            components=components,
            cost_model=cost_model,
            takeoff_time_str=takeoff_time_str,
            takeoff_timestamp=takeoff_timestamp,
            estimated_landing_time_str=estimated_landing_time_str,
        )
        transitions = transition_payload["thinned_transitions"]
        tailwind = transition_payload["avg_tailwind_knots"]
        transition_metadata.update(
            {
                "tres_forward_transitions": transition_payload["forward_transitions"],
                "tres_backward_closures": transition_payload["backward_closures"],
                "thinned_transitions": transition_payload["thinned_count"],
                "takeoff_time_str": transition_payload["takeoff_time_str"],
                "estimated_landing_time_str": transition_payload["estimated_landing_time_str"],
                "min_wall_clock_time_sec": transition_payload["min_wall_clock_time_sec"],
            }
        )
    else:
        transitions = dataset_transitions
        tailwind = dataset_tailwind

    scenario_active = (
        graph_scenario is not None
        or preference_scenario is not None
        or effective_transition_mode == "recompute"
    )
    scenario_hash = None
    if scenario_active:
        scenario_hash = _stable_hash(
            {
                "graph": graph_report,
                "graph_fingerprint": graph_fingerprint(components["graph"], components["node_to_idx"]),
                "preference": preference_report,
                "transition_mode": effective_transition_mode,
            }
        )
    base_cache_key = flight_id or f"{components['origin_node']}_{components['goal_node']}"
    cache_key_id = _with_scenario_cache_key(base_cache_key, scenario_hash)

    return _run_inference_from_transitions(
        case_dir=case_dir,
        resolved_checkpoint=resolved_checkpoint,
        config=config,
        components=components,
        cost_model=cost_model,
        checkpoint_metadata=checkpoint_metadata,
        transitions=transitions,
        tailwind=tailwind,
        flight_id=flight_id,
        takeoff_timestamp=takeoff_timestamp,
        gamma=gamma,
        n_samples=n_samples,
        policy=policy,
        initial_k_policy=initial_k_policy,
        return_4d=return_4d,
        seed=seed,
        cache_dir=cache_dir,
        use_cache=use_cache,
        output_dir=output_dir,
        write_4d_csv=write_4d_csv,
        tranche_altitudes_ft=tranche_altitudes_ft,
        cache_key_id=cache_key_id,
        metadata_extra={
            "flight_metadata": flight_meta,
            "graph_scenario": graph_report,
            "preference_scenario": preference_report,
            "transition_metadata": transition_metadata,
            "scenario_hash": scenario_hash,
            "auto_switched_to_recompute": auto_switched_to_recompute,
        },
    )


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
    initial_k_policy: str = "uniform",
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
    takeoff_timestamp, takeoff_time_str = _resolve_takeoff_fields(
        takeoff_timestamp=takeoff_timestamp,
        takeoff_time_str=takeoff_time_str,
    )
    config, components = load_case(case_dir, device=device)
    _set_origin_goal(components, origin_node or config.origin_node, goal_node or config.goal_node)
    if components.get("origin_node") is None or components.get("goal_node") is None:
        raise ValueError("origin_node and goal_node must be provided or present in the case config.")
    _ensure_required_flight_config_fields(config)

    resolved_checkpoint = resolve_checkpoint(case_dir, checkpoint_path)
    cost_model, checkpoint_metadata = load_cost_model(
        resolved_checkpoint,
        config=config,
        num_waypoints=components["num_nodes"],
        device=components["device"],
    )
    components["cost_model"] = cost_model

    transition_payload = _compute_transitions_for_intent(
        config=config,
        components=components,
        cost_model=cost_model,
        takeoff_time_str=takeoff_time_str,
        takeoff_timestamp=takeoff_timestamp,
        estimated_landing_time_str=estimated_landing_time_str,
    )
    cache_key_id = flight_id or f"{components['origin_node']}_{components['goal_node']}"
    return _run_inference_from_transitions(
        case_dir=case_dir,
        resolved_checkpoint=resolved_checkpoint,
        config=config,
        components=components,
        cost_model=cost_model,
        checkpoint_metadata=checkpoint_metadata,
        transitions=transition_payload["thinned_transitions"],
        tailwind=transition_payload["avg_tailwind_knots"],
        flight_id=flight_id,
        takeoff_timestamp=takeoff_timestamp,
        gamma=gamma,
        n_samples=n_samples,
        policy=policy,
        initial_k_policy=initial_k_policy,
        return_4d=return_4d,
        seed=seed,
        cache_dir=cache_dir,
        use_cache=use_cache,
        output_dir=output_dir,
        write_4d_csv=write_4d_csv,
        tranche_altitudes_ft=tranche_altitudes_ft,
        cache_key_id=cache_key_id,
        use_adaptive_initial_state=True,
        metadata_extra={
            "transition_metadata": {
                "transition_mode": "recompute",
                "tres_forward_transitions": transition_payload["forward_transitions"],
                "tres_backward_closures": transition_payload["backward_closures"],
                "thinned_transitions": transition_payload["thinned_count"],
                "takeoff_time_str": transition_payload["takeoff_time_str"],
                "estimated_landing_time_str": transition_payload["estimated_landing_time_str"],
                "min_wall_clock_time_sec": transition_payload["min_wall_clock_time_sec"],
            }
        },
    )


def compute_4d_path(*args, **kwargs) -> Compute4DPathResult:
    print(f"compute_4d_path implies compute_4d_path_for_dataset and will be deprecated in the future.")
    return compute_4d_path_for_dataset(*args, **kwargs)

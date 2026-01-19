from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from equinox.sampling.trespass.inference import (
    compute_checkpoint_hash,
    graph_fingerprint,
    load_case,
    load_cached_svi,
    load_cost_model,
    load_flight_artifacts,
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
    flight_id: str
    takeoff_timestamp: int
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


def _write_outputs(
    output_dir: Path,
    *,
    result: Compute4DPathResult,
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


def compute_4d_path(
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
) -> Compute4DPathResult:
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
        _write_outputs(Path(output_dir), result=result)

    return result

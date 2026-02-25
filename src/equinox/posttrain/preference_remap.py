from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import networkx as nx
import numpy as np
import torch
from sklearn.neighbors import KDTree


def haversine_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r_nm = 3440.065
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2.0) ** 2
    )
    return 2.0 * r_nm * math.asin(math.sqrt(a))


def _coords_by_index(graph: nx.Graph, node_to_idx: Mapping[str, int]) -> Tuple[np.ndarray, Dict[int, str]]:
    idx_to_node = {idx: node for node, idx in node_to_idx.items()}
    coords = np.zeros((len(idx_to_node), 2), dtype=np.float64)
    for idx, node in idx_to_node.items():
        coords[idx, 0] = float(graph.nodes[node]["lat"])
        coords[idx, 1] = float(graph.nodes[node]["lon"])
    return coords, idx_to_node


def remap_checkpoint_preferences_to_graph(
    *,
    preference_matrix: torch.Tensor,
    source_graph: nx.Graph,
    source_node_to_idx: Mapping[str, int],
    target_graph: nx.Graph,
    target_node_to_idx: Mapping[str, int],
    max_nn_distance_nm: Optional[float] = None,
    unmatched_value: float = 0.0,
    zero_all: bool = False,
    zero_edges: Optional[Iterable[Tuple[str, str]]] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    if preference_matrix.ndim != 2:
        raise ValueError("preference_matrix must be 2-dimensional.")

    source_coords, source_idx_to_node = _coords_by_index(source_graph, source_node_to_idx)
    target_coords, target_idx_to_node = _coords_by_index(target_graph, target_node_to_idx)

    pref_cpu = preference_matrix.detach().cpu()
    out = torch.full(
        (len(target_idx_to_node), len(target_idx_to_node)),
        fill_value=float(unmatched_value),
        dtype=pref_cpu.dtype,
    )
    if zero_all:
        out.zero_()

    tree = KDTree(source_coords, metric="euclidean")
    nearest_dist_deg, nearest_idx = tree.query(target_coords, k=1)
    nearest_dist_deg = nearest_dist_deg.reshape(-1)
    nearest_idx = nearest_idx.reshape(-1)

    target_to_source: Dict[int, Optional[int]] = {}
    distances_nm: Dict[int, float] = {}
    for tgt_idx in range(len(target_idx_to_node)):
        src_idx = int(nearest_idx[tgt_idx])
        tgt_node = target_idx_to_node[tgt_idx]
        src_node = source_idx_to_node[src_idx]
        d_nm = haversine_nm(
            float(target_graph.nodes[tgt_node]["lat"]),
            float(target_graph.nodes[tgt_node]["lon"]),
            float(source_graph.nodes[src_node]["lat"]),
            float(source_graph.nodes[src_node]["lon"]),
        )
        if max_nn_distance_nm is not None and d_nm > float(max_nn_distance_nm):
            target_to_source[tgt_idx] = None
        else:
            target_to_source[tgt_idx] = src_idx
            distances_nm[tgt_idx] = d_nm

    mapped_edges = 0
    unmatched_edges = 0
    if not zero_all:
        for u, v in target_graph.edges():
            if u not in target_node_to_idx or v not in target_node_to_idx:
                continue
            u_t = target_node_to_idx[u]
            v_t = target_node_to_idx[v]
            u_s = target_to_source.get(u_t)
            v_s = target_to_source.get(v_t)
            if u_s is None or v_s is None:
                unmatched_edges += 1
                continue
            out[u_t, v_t] = pref_cpu[u_s, v_s]
            mapped_edges += 1

    zeroed_edges = 0
    if zero_edges:
        for u, v in zero_edges:
            if u in target_node_to_idx and v in target_node_to_idx:
                out[target_node_to_idx[u], target_node_to_idx[v]] = 0.0
                zeroed_edges += 1

    report: Dict[str, Any] = {
        "target_nodes": len(target_idx_to_node),
        "source_nodes": len(source_idx_to_node),
        "matched_nodes": sum(1 for v in target_to_source.values() if v is not None),
        "unmatched_nodes": sum(1 for v in target_to_source.values() if v is None),
        "mapped_edges": int(mapped_edges),
        "unmatched_edges": int(unmatched_edges),
        "used_zero_all": bool(zero_all),
        "zeroed_edges": int(zeroed_edges),
        "max_nn_distance_nm": max(distances_nm.values()) if distances_nm else None,
        "mean_nn_distance_nm": (float(np.mean(list(distances_nm.values()))) if distances_nm else None),
        "max_nn_distance_limit_nm": max_nn_distance_nm,
        "nearest_distance_deg_mean": float(np.mean(nearest_dist_deg)) if len(nearest_dist_deg) else None,
    }
    return out, report

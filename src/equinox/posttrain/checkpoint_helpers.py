"""
See docs at POSTTRAIN.md
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import networkx as nx
import torch

from equinox.cost.cost_linear_disentangled import DEFAULT_FEATURE_NAMES


def _load_checkpoint_payload(
    checkpoint_path: str,
    *,
    map_location: Optional[torch.device] = None,
) -> Dict[str, Any]:
    if map_location is None:
        map_location = torch.device("cpu")
    payload = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint payload must be a dict.")
    return payload


def _extract_cost_model_state(payload: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
    if "model_state_dict" in payload:
        return payload["model_state_dict"]
    if "final_model_state" in payload:
        return payload["final_model_state"]
    if "common_weights" in payload and "preference_matrix_p" in payload:
        return payload  # already a state_dict-like payload
    raise KeyError(
        "Could not locate a cost model state dict in the checkpoint payload. "
        "Expected 'model_state_dict', 'final_model_state', or a state dict with "
        "'common_weights' and 'preference_matrix_p'."
    )


def load_cost_model_parameters(
    checkpoint_path: str,
    *,
    map_location: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Load common feature weights and the edge preference matrix from a checkpoint.

    Returns:
        dict with keys:
          - common: mapping from feature name -> weight
          - preference_matrix: torch.Tensor [num_nodes, num_nodes]
          - feature_names: tuple of feature names
    """
    payload = _load_checkpoint_payload(checkpoint_path, map_location=map_location)
    state_dict = _extract_cost_model_state(payload)

    common_weights = state_dict.get("common_weights")
    if common_weights is None:
        raise KeyError("Checkpoint state dict missing 'common_weights'.")

    preference_matrix = state_dict.get("preference_matrix_p")
    if preference_matrix is None:
        raise KeyError("Checkpoint state dict missing 'preference_matrix_p'.")

    common_values = common_weights.detach().cpu().tolist()
    common = {name: float(value) for name, value in zip(DEFAULT_FEATURE_NAMES, common_values)}

    return {
        "common": common,
        "preference_matrix": preference_matrix.detach().cpu(),
        "feature_names": DEFAULT_FEATURE_NAMES,
    }


def load_graph_from_gml(
    gml_path: str,
) -> Tuple[nx.Graph, Dict[str, int], Dict[int, str]]:
    graph = nx.read_gml(gml_path)
    node_to_idx = {node: i for i, node in enumerate(graph.nodes())}
    idx_to_node = {i: node for i, node in enumerate(graph.nodes())}
    return graph, node_to_idx, idx_to_node


def map_preferences_to_graph_edges(
    preference_matrix: torch.Tensor,
    graph: nx.Graph,
    node_to_idx: Mapping[str, int],
) -> Dict[Tuple[str, str], float]:
    if preference_matrix.ndim != 2:
        raise ValueError("preference_matrix must be 2-dimensional.")
    num_nodes = len(node_to_idx)
    if preference_matrix.shape[0] < num_nodes or preference_matrix.shape[1] < num_nodes:
        raise ValueError(
            "preference_matrix shape does not match number of nodes in the graph."
        )

    edge_preferences: Dict[Tuple[str, str], float] = {}
    pref_cpu = preference_matrix.detach().cpu()
    for u, v in graph.edges():
        if u not in node_to_idx or v not in node_to_idx:
            continue
        u_idx = node_to_idx[u]
        v_idx = node_to_idx[v]
        edge_preferences[(u, v)] = float(pref_cpu[u_idx, v_idx].item())
    return edge_preferences


def load_checkpoint_edge_preferences(
    checkpoint_path: str,
    gml_path: str,
    *,
    map_location: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Convenience wrapper: load checkpoint params and map preferences to graph edges.

    Returns:
        dict with keys:
          - common: mapping from feature name -> weight
          - preference_matrix: torch.Tensor [num_nodes, num_nodes]
          - edge_preferences: mapping (u, v) -> preference value
          - graph: networkx graph from GML
          - node_to_idx: node name -> index
    """
    params = load_cost_model_parameters(checkpoint_path, map_location=map_location)
    graph, node_to_idx, _ = load_graph_from_gml(gml_path)
    edge_preferences = map_preferences_to_graph_edges(
        params["preference_matrix"], graph, node_to_idx
    )
    params.update(
        {
            "edge_preferences": edge_preferences,
            "graph": graph,
            "node_to_idx": node_to_idx,
        }
    )
    return params


def attach_preferences_to_graph(
    graph: nx.Graph,
    node_to_idx: Mapping[str, int],
    preference_matrix: torch.Tensor,
    *,
    attr_name: str = "preference",
    copy_graph: bool = True,
) -> nx.Graph:
    if copy_graph:
        graph = graph.copy()

    pref_cpu = preference_matrix.detach().cpu()
    for u, v in graph.edges():
        if u not in node_to_idx or v not in node_to_idx:
            continue
        u_idx = node_to_idx[u]
        v_idx = node_to_idx[v]
        graph.edges[u, v][attr_name] = float(pref_cpu[u_idx, v_idx].item())
    return graph


def percentile_filter(
    values: np.ndarray,
    percentile: float,
    *,
    mode: str = "both",
) -> np.ndarray:
    """
    Return a boolean mask for values within a percentile slice of the distribution.

    Args:
        values: Array of numeric values to filter.
        percentile: Fraction of the distribution to keep (0 < percentile <= 1).
        mode: "upper", "lower", "both", or "contrast".
            - "upper": keep the lowest `percentile` fraction.
            - "lower": keep the highest `percentile` fraction.
            - "both": keep the central `percentile` fraction.
            - "contrast": keep the lowest and highest `percentile / 2` fractions.
    """
    values_np = np.asarray(values, dtype=np.float64)
    if percentile <= 0 or percentile > 1:
        raise ValueError("percentile must be in (0, 1].")
    if values_np.size == 0:
        return np.zeros(0, dtype=bool)

    mode_norm = mode.lower()
    if percentile == 1:
        return np.ones_like(values_np, dtype=bool)
    if mode_norm == "both":
        tail = (1.0 - percentile) / 2.0
        lower_q = np.quantile(values_np, tail)
        upper_q = np.quantile(values_np, 1.0 - tail)
        return (values_np >= lower_q) & (values_np <= upper_q)
    if mode_norm == "contrast":
        tail = percentile / 2.0
        lower_q = np.quantile(values_np, tail)
        upper_q = np.quantile(values_np, 1.0 - tail)
        return (values_np <= lower_q) | (values_np >= upper_q)
    if mode_norm == "upper":
        upper_q = np.quantile(values_np, percentile)
        return values_np <= upper_q
    if mode_norm == "lower":
        lower_q = np.quantile(values_np, 1.0 - percentile)
        return values_np >= lower_q
    raise ValueError("mode must be 'upper', 'lower', 'both', or 'contrast'.")


def plot_edge_preferences_cartopy(
    graph: nx.Graph,
    *,
    edge_preferences: Optional[Mapping[Tuple[str, str], float]] = None,
    preference_attr: str = "preference cost",
    ax=None,
    cmap: str = "coolwarm",
    linewidth: float = 1.5,
    alpha: float = 0.9,
    preference_color_range: Optional[Tuple[float, float]] = None,
    percentile: Optional[float] = None,
    percentile_mode: str = "both",
    show_colorbar: bool = True,
    show_waypoints: bool = False,
    show: bool = True,
):
    """
    Plot edge preferences on a Cartopy map using colored links.

    If edge_preferences is provided, it will be used; otherwise, this function
    expects the graph edges to have `preference_attr`.
    """
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize

    created_ax = ax is None
    if created_ax:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.add_feature(cfeature.LAND)
    # ax.add_feature(cfeature.OCEAN)
    # ax.add_feature(cfeature.LAKES)
    # ax.add_feature(cfeature.RIVERS)

    segments = []
    values = []
    segment_lons = []
    segment_lats = []
    segment_nodes = []

    for u, v in graph.edges():
        if edge_preferences is not None:
            pref = edge_preferences.get((u, v))
        else:
            pref = graph.edges[u, v].get(preference_attr)
        if pref is None:
            continue
        u_data = graph.nodes[u]
        v_data = graph.nodes[v]
        lon1, lat1 = u_data["lon"], u_data["lat"]
        lon2, lat2 = v_data["lon"], v_data["lat"]
        segments.append([(lon1, lat1), (lon2, lat2)])
        values.append(float(pref))
        segment_lons.append((lon1, lon2))
        segment_lats.append((lat1, lat2))
        segment_nodes.append((u, v))

    if not segments:
        raise ValueError("No edge preferences found to plot.")

    values_np = np.asarray(values, dtype=np.float64)
    if percentile is not None:
        mask = percentile_filter(values_np, percentile, mode=percentile_mode)
        if not mask.any():
            raise ValueError("Percentile filter removed all edge preferences.")
        segments = [seg for seg, keep in zip(segments, mask) if keep]
        values_np = values_np[mask]
        segment_lons = [lons for lons, keep in zip(segment_lons, mask) if keep]
        segment_lats = [lats for lats, keep in zip(segment_lats, mask) if keep]
        segment_nodes = [nodes for nodes, keep in zip(segment_nodes, mask) if keep]

    used_lons = [lon for pair in segment_lons for lon in pair]
    used_lats = [lat for pair in segment_lats for lat in pair]
    if show_waypoints:
        used_nodes = {node for pair in segment_nodes for node in pair}
        for node in sorted(used_nodes):
            data = graph.nodes[node]
            lon, lat = data["lon"], data["lat"]
            ax.plot(lon, lat, "o", color="blue", markersize=3, transform=ccrs.Geodetic())
            ax.text(lon + 0.01, lat + 0.01, str(node), fontsize=6, transform=ccrs.Geodetic())
    if preference_color_range is not None:
        vmin, vmax = preference_color_range
        if vmin >= vmax:
            raise ValueError("preference_color_range must be (min, max) with min < max.")
    else:
        vmin = float(values_np.min())
        vmax = float(values_np.max())
    norm = Normalize(vmin=vmin, vmax=vmax)
    line_collection = LineCollection(
        segments,
        cmap=cmap,
        norm=norm,
        linewidths=linewidth,
        alpha=alpha,
        transform=ccrs.Geodetic(),
    )
    line_collection.set_array(values_np)
    ax.add_collection(line_collection)

    if used_lons and used_lats:
        buffer = 1.0
        min_lon, max_lon = min(used_lons) - buffer, max(used_lons) + buffer
        min_lat, max_lat = min(used_lats) - buffer, max(used_lats) + buffer
        ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())

    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

    if show_colorbar:
        plt.colorbar(line_collection, ax=ax, orientation="vertical", label=preference_attr)

    if created_ax and show:
        plt.show()

    return ax

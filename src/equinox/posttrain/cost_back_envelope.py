"""
Utilities for back-of-the-envelope cost breakdowns on concrete routes.

This module loads the cost model parameters from a checkpoint, pulls the
route inputs (distance, charges, wind), and computes the per-edge
component contributions for a chosen flight.
"""

from __future__ import annotations

import argparse
import ast
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

from equinox.config import RunConfiguration
from equinox.posttrain.checkpoint_helpers import load_cost_model_parameters


@dataclass(frozen=True)
class RouteSelection:
    flight_id: str
    takeoff_time: int
    route_nodes: List[str]


def _load_run_config(case_dir: str) -> RunConfiguration:
    config_path = Path(case_dir) / "default.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)
    return RunConfiguration(**config_data)


def _parse_route(route_value: Any) -> List[str]:
    if isinstance(route_value, (list, tuple)):
        return [str(node) for node in route_value]
    if not isinstance(route_value, str):
        raise TypeError(f"Unsupported route type: {type(route_value)}")

    route_str = route_value.strip()
    if not route_str:
        return []

    if route_str.startswith("[") or route_str.startswith("("):
        try:
            parsed = ast.literal_eval(route_str)
            if isinstance(parsed, (list, tuple)):
                return [str(node) for node in parsed]
        except (SyntaxError, ValueError):
            pass

    if "->" in route_str:
        return [part.strip() for part in route_str.split("->") if part.strip()]

    return [node for node in route_str.split() if node]


def _parse_routes_txt(routes_path: Path) -> List[List[str]]:
    routes: List[List[str]] = []
    for line in routes_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if "," in line:
            _, route_part = line.split(",", 1)
        else:
            route_part = line
        nodes = route_part.strip().split()
        if nodes:
            routes.append(nodes)
    return routes


def _select_route_from_csv(
    routes_csv_path: str,
    *,
    flight_id: Optional[str] = None,
    takeoff_time: Optional[int] = None,
    row_idx: int = 0,
) -> RouteSelection:
    df = pd.read_csv(routes_csv_path)
    if df.empty:
        raise ValueError(f"No routes found in {routes_csv_path}")

    if flight_id is not None:
        match = df.loc[df["flight_id"] == flight_id]
        if takeoff_time is not None:
            match = match.loc[match["takeoff_time"] == int(takeoff_time)]
        if match.empty:
            if takeoff_time is None:
                raise ValueError(f"Flight ID {flight_id} not found in {routes_csv_path}")
            raise ValueError(
                f"Flight ID {flight_id} with takeoff_time {takeoff_time} not found in {routes_csv_path}"
            )
        if takeoff_time is None and len(match) > 1:
            raise ValueError(
                f"Flight ID {flight_id} is not unique in {routes_csv_path}; provide takeoff_time"
            )
        row = match.iloc[0]
    else:
        row = df.iloc[row_idx]

    route_nodes = _parse_route(row["route"])
    if len(route_nodes) < 2:
        raise ValueError("Route must contain at least two nodes.")

    return RouteSelection(
        flight_id=str(row["flight_id"]),
        takeoff_time=int(row["takeoff_time"]),
        route_nodes=route_nodes,
    )


def _find_case_graph_gml(case_dir: Path) -> Path:
    preferred = case_dir / "graphs" / "routes.gml"
    if preferred.exists():
        return preferred

    gml_files = sorted(case_dir.rglob("*.gml"))
    if not gml_files:
        raise FileNotFoundError(f"No .gml files found under {case_dir}")

    routes_named = [path for path in gml_files if path.name == "routes.gml"]
    if len(routes_named) == 1:
        return routes_named[0]

    if len(gml_files) == 1:
        return gml_files[0]

    graphs_dir = [path for path in gml_files if "graphs" in path.parts]
    if len(graphs_dir) == 1:
        return graphs_dir[0]

    candidates = "\n".join(str(path) for path in gml_files)
    raise ValueError(
        "Multiple .gml files found; provide graph_path explicitly:\n"
        f"{candidates}"
    )


def _find_tres_wind_files(
    case_dir: str,
    flight_id: str,
    takeoff_time: int,
) -> Optional[Tuple[Path, Path]]:
    tres_dir = Path(case_dir) / "tres_runs"
    for batch_dir in tres_dir.glob("batch*"):
        wind_file = batch_dir / f"WIND_{flight_id}_{takeoff_time}.pt"
        clsr_file = batch_dir / f"CLSR_{flight_id}_{takeoff_time}.pkl"
        if wind_file.exists() and clsr_file.exists():
            return wind_file, clsr_file
    return None


def _load_tailwind_by_edge(
    wind_file: Path,
    clsr_file: Path,
) -> Dict[Tuple[int, int], float]:
    with open(clsr_file, "rb") as f:
        transitions = pickle.load(f)
    tailwind = torch.load(wind_file).detach().cpu().tolist()
    if len(transitions) != len(tailwind):
        raise ValueError("Transition list and wind vector lengths do not match.")

    sums: Dict[Tuple[int, int], float] = {}
    counts: Dict[Tuple[int, int], int] = {}
    for transition, wind_value in zip(transitions, tailwind):
        if not np.isfinite(wind_value):
            continue
        u_idx = int(transition[0])
        v_idx = int(transition[5])
        key = (u_idx, v_idx)
        sums[key] = sums.get(key, 0.0) + float(wind_value)
        counts[key] = counts.get(key, 0) + 1

    return {key: sums[key] / counts[key] for key in sums}


def _compute_edge_breakdown(
    *,
    route_nodes: Sequence[str],
    node_to_idx: Dict[str, int],
    dist_matrix: np.ndarray,
    charges_matrix: np.ndarray,
    preference_matrix: torch.Tensor,
    common_weights: Dict[str, float],
    cruise_speed_kts: float,
    tailwind_by_edge: Optional[Dict[Tuple[int, int], float]] = None,
    tailwind_fallback_kts: float = 0.0,
) -> List[Dict[str, Any]]:
    weights = {
        "bias": float(common_weights["bias"]),
        "ac_dist": float(common_weights["ac_dist"]),
        "time": float(common_weights["time"]),
    }
    pref_cpu = preference_matrix.detach().cpu().numpy()

    breakdown: List[Dict[str, Any]] = []
    for u_node, v_node in zip(route_nodes[:-1], route_nodes[1:]):
        if u_node not in node_to_idx or v_node not in node_to_idx:
            raise ValueError(f"Route edge {u_node} -> {v_node} not found in graph.")
        u_idx = node_to_idx[u_node]
        v_idx = node_to_idx[v_node]

        dist = float(dist_matrix[u_idx, v_idx])
        charge = float(charges_matrix[u_idx, v_idx])
        if tailwind_by_edge is None:
            tailwind_kts = float(tailwind_fallback_kts)
            tailwind_is_fallback = True
        else:
            value = tailwind_by_edge.get((u_idx, v_idx))
            if value is None:
                tailwind_kts = float(tailwind_fallback_kts)
                tailwind_is_fallback = True
            else:
                tailwind_kts = float(value)
                tailwind_is_fallback = False

        ac_dist = charge * dist / 100.0
        time_feature = 60.0 * dist / (cruise_speed_kts + tailwind_kts)

        bias_cost = weights["bias"]
        ac_cost = weights["ac_dist"] * ac_dist
        time_cost = weights["time"] * time_feature
        pref_cost = float(pref_cpu[u_idx, v_idx])

        breakdown.append(
            {
                "u_node": u_node,
                "v_node": v_node,
                "u_idx": u_idx,
                "v_idx": v_idx,
                "distance": dist,
                "airspace_charge": charge,
                "tailwind_kts": tailwind_kts,
                "tailwind_is_fallback": tailwind_is_fallback,
                "feature_bias": 1.0,
                "feature_ac_dist": ac_dist,
                "feature_time": time_feature,
                "weight_bias": weights["bias"],
                "weight_ac_dist": weights["ac_dist"],
                "weight_time": weights["time"],
                "cost_bias": bias_cost,
                "cost_ac_dist": ac_cost,
                "cost_time": time_cost,
                "cost_preference": pref_cost,
                "cost_common": bias_cost + ac_cost + time_cost,
                "cost_total": bias_cost + ac_cost + time_cost + pref_cost,
            }
        )

    return breakdown


def compute_route_cost_breakdown(
    *,
    checkpoint_path: str,
    case_dir: str,
    routes_csv_path: Optional[str] = None,
    flight_id: Optional[str] = None,
    takeoff_time: Optional[int] = None,
    row_idx: int = 0,
    use_tres_wind: bool = True,
    tailwind_fallback_kts: float = 0.0,
) -> Dict[str, Any]:
    """
    Compute a per-edge cost component breakdown for a selected flight route.

    Args:
        checkpoint_path: Path to the cost model checkpoint.
        case_dir: Case directory containing default.yaml and case assets.
        routes_csv_path: Optional path to a routes CSV. Defaults to all_routes_sculpted.csv.
        flight_id: Optional flight identifier to select the route. If omitted, row_idx is used.
        takeoff_time: Optional takeoff timestamp to disambiguate flight_id.
        row_idx: Row index to use when flight_id is not provided.
        use_tres_wind: Use precomputed WIND/CLSR files to estimate tailwind per edge.
        tailwind_fallback_kts: Tailwind fallback if wind data is unavailable.

    Returns:
        Dict with selection metadata, per-edge breakdown list, and totals.
    """
    config = _load_run_config(case_dir)
    routes_csv = (
        routes_csv_path
        if routes_csv_path is not None
        else str(Path(case_dir) / "all_routes_sculpted.csv")
    )
    selection = _select_route_from_csv(
        routes_csv,
        flight_id=flight_id,
        takeoff_time=takeoff_time,
        row_idx=row_idx,
    )

    components = config.initialize_all_components(
        cost_model_version="lin_disent",
        manual_cost_model_init=True,
    )
    node_to_idx = components["node_to_idx"]
    dist_matrix = components["dist_matrix"]
    charges_matrix = components["ac_matrix"]

    cost_params = load_cost_model_parameters(checkpoint_path)
    tailwind_by_edge: Optional[Dict[Tuple[int, int], float]] = None
    if use_tres_wind:
        found = _find_tres_wind_files(
            case_dir, selection.flight_id, selection.takeoff_time
        )
        if found is not None:
            wind_file, clsr_file = found
            tailwind_by_edge = _load_tailwind_by_edge(wind_file, clsr_file)

    breakdown = _compute_edge_breakdown(
        route_nodes=selection.route_nodes,
        node_to_idx=node_to_idx,
        dist_matrix=dist_matrix,
        charges_matrix=charges_matrix,
        preference_matrix=cost_params["preference_matrix"],
        common_weights=cost_params["common"],
        cruise_speed_kts=config.cruise_speed_kts,
        tailwind_by_edge=tailwind_by_edge,
        tailwind_fallback_kts=tailwind_fallback_kts,
    )

    totals = {
        "cost_bias": sum(edge["cost_bias"] for edge in breakdown),
        "cost_ac_dist": sum(edge["cost_ac_dist"] for edge in breakdown),
        "cost_time": sum(edge["cost_time"] for edge in breakdown),
        "cost_preference": sum(edge["cost_preference"] for edge in breakdown),
        "cost_common": sum(edge["cost_common"] for edge in breakdown),
        "cost_total": sum(edge["cost_total"] for edge in breakdown),
    }

    return {
        "flight_id": selection.flight_id,
        "takeoff_time": selection.takeoff_time,
        "route_nodes": selection.route_nodes,
        "breakdown": breakdown,
        "totals": totals,
        "weights": cost_params["common"],
        "use_tres_wind": use_tres_wind,
        "tailwind_fallback_kts": tailwind_fallback_kts,
    }


def compute_routes_cost_breakdowns(
    *,
    checkpoint_path: str,
    case_dir: str,
    routes_path: str,
    graph_path: Optional[str] = None,
    flight_id: Optional[str] = None,
    takeoff_time: Optional[int] = None,
    use_tres_wind: bool = True,
    tailwind_fallback_kts: float = 0.0,
    plot_route: bool = True,
    show_waypoints: bool = False,
    route_alpha: float = 1.0,
) -> List[Dict[str, Any]]:
    """
    Compute and print per-edge cost breakdowns for each unique route in routes_path.

    When plot_route is True, a mini cartopy plot is rendered for each route.
    """
    if not 0.0 <= route_alpha <= 1.0:
        raise ValueError("route_alpha must be between 0.0 and 1.0")

    routes = _parse_routes_txt(Path(routes_path))
    if not routes:
        raise ValueError(f"No routes found in {routes_path}")

    unique_routes: List[List[str]] = []
    seen_routes: set[Tuple[str, ...]] = set()
    for route in routes:
        key = tuple(route)
        if key in seen_routes:
            continue
        seen_routes.add(key)
        unique_routes.append(route)

    config = _load_run_config(case_dir)
    components = config.initialize_all_components(
        cost_model_version="lin_disent",
        manual_cost_model_init=True,
    )
    node_to_idx = components["node_to_idx"]
    dist_matrix = components["dist_matrix"]
    charges_matrix = components["ac_matrix"]

    cost_params = load_cost_model_parameters(checkpoint_path)
    tailwind_by_edge: Optional[Dict[Tuple[int, int], float]] = None
    if use_tres_wind and flight_id is not None and takeoff_time is not None:
        found = _find_tres_wind_files(case_dir, flight_id, takeoff_time)
        if found is not None:
            wind_file, clsr_file = found
            tailwind_by_edge = _load_tailwind_by_edge(wind_file, clsr_file)

    graph = None
    if plot_route:
        import cartopy.crs as ccrs
        import matplotlib.pyplot as plt
        import networkx as nx

        from equinox.helpers.plotters import plot_routes_on_map

        graph_path_value = (
            Path(graph_path) if graph_path is not None else _find_case_graph_gml(Path(case_dir))
        )
        graph = nx.read_gml(graph_path_value)

    results: List[Dict[str, Any]] = []
    for route_idx, route_nodes in enumerate(unique_routes, start=1):
        breakdown = _compute_edge_breakdown(
            route_nodes=route_nodes,
            node_to_idx=node_to_idx,
            dist_matrix=dist_matrix,
            charges_matrix=charges_matrix,
            preference_matrix=cost_params["preference_matrix"],
            common_weights=cost_params["common"],
            cruise_speed_kts=config.cruise_speed_kts,
            tailwind_by_edge=tailwind_by_edge,
            tailwind_fallback_kts=tailwind_fallback_kts,
        )
        totals = {
            "cost_bias": sum(edge["cost_bias"] for edge in breakdown),
            "cost_ac_dist": sum(edge["cost_ac_dist"] for edge in breakdown),
            "cost_time": sum(edge["cost_time"] for edge in breakdown),
            "cost_preference": sum(edge["cost_preference"] for edge in breakdown),
            "cost_common": sum(edge["cost_common"] for edge in breakdown),
            "cost_total": sum(edge["cost_total"] for edge in breakdown),
        }

        display_frame = breakdown_to_frame(breakdown)
        columns = [
            "u_node",
            "v_node",
            "distance",
            "airspace_charge",
            "tailwind_kts",
            "tailwind_is_fallback",
            "cost_bias",
            "cost_ac_dist",
            "cost_time",
            "cost_preference",
            "cost_total",
        ]
        if all(column in display_frame.columns for column in columns):
            display_frame = display_frame[columns]
        float_cols = [
            column
            for column in display_frame.columns
            if display_frame[column].dtype.kind in "fc"
        ]
        display_frame = display_frame.copy()
        display_frame[float_cols] = display_frame[float_cols].round(3)

        print(
            f"Route {route_idx}/{len(unique_routes)}: "
            f"{' -> '.join(route_nodes)}"
        )
        _display_frame(display_frame)
        print(
            "Totals: "
            f"bias={totals['cost_bias']:.3f}, "
            f"ac_dist={totals['cost_ac_dist']:.3f}, "
            f"time={totals['cost_time']:.3f}, "
            f"preference={totals['cost_preference']:.3f}, "
            f"total={totals['cost_total']:.3f}"
        )

        fig = None
        ax = None
        if plot_route:
            fig = plt.figure(figsize=(7, 4))
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
            plot_routes_on_map(
                graph,
                [route_nodes],
                ax=ax,
                show_waypoints=show_waypoints,
                route_alpha=route_alpha,
            )
            ax.set_title(f"Route {route_idx}")
            plt.show()

        results.append(
            {
                "route_nodes": route_nodes,
                "breakdown": breakdown,
                "totals": totals,
                "weights": cost_params["common"],
                "use_tres_wind": use_tres_wind,
                "tailwind_fallback_kts": tailwind_fallback_kts,
                "figure": fig,
                "axes": ax,
            }
        )

    return results


def breakdown_to_frame(breakdown: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    """Convert breakdown list to a pandas DataFrame for inspection/export."""
    return pd.DataFrame(list(breakdown))


def _display_frame(frame: pd.DataFrame) -> None:
    try:
        from IPython.display import display  # type: ignore
    except ImportError:
        with pd.option_context("display.max_columns", None, "display.width", 200):
            print(frame.to_string(index=False))
    else:
        display(frame)


def _parse_routes_breakdown_args(
    argv: Iterable[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print per-route cost breakdown tables for a routes file."
    )
    parser.add_argument(
        "--checkpoint-path",
        required=True,
        type=str,
        help="Path to the cost model checkpoint.",
    )
    parser.add_argument(
        "--case-dir",
        required=True,
        type=str,
        help="Case directory containing default.yaml and case assets.",
    )
    parser.add_argument(
        "--routes-path",
        required=True,
        type=str,
        help="Path to routes.txt (lines are 'cost,WAYPOINT ...').",
    )
    parser.add_argument(
        "--graph-path",
        type=str,
        default=None,
        help="Optional graph .gml path (overrides auto-detect).",
    )
    parser.add_argument(
        "--flight-id",
        type=str,
        default=None,
        help="Optional flight_id for loading TRES wind files.",
    )
    parser.add_argument(
        "--takeoff-time",
        type=int,
        default=None,
        help="Optional takeoff_time for loading TRES wind files.",
    )
    parser.add_argument(
        "--no-tres-wind",
        action="store_true",
        help="Disable loading TRES wind data.",
    )
    parser.add_argument(
        "--tailwind-fallback-kts",
        type=float,
        default=0.0,
        help="Tailwind fallback (kts) if wind data is unavailable.",
    )
    parser.add_argument(
        "--plot-route",
        dest="plot_route",
        action="store_true",
        help="Plot each route on a cartopy map (default).",
    )
    parser.add_argument(
        "--no-plot-route",
        dest="plot_route",
        action="store_false",
        help="Disable route plotting.",
    )
    parser.set_defaults(plot_route=True)
    parser.add_argument(
        "--show-waypoints",
        action="store_true",
        help="Show waypoint markers and labels on the plots.",
    )
    parser.add_argument(
        "--route-alpha",
        type=float,
        default=1.0,
        help="Opacity for route lines (0.0 to 1.0).",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_routes_breakdown_args(argv)
    compute_routes_cost_breakdowns(
        checkpoint_path=args.checkpoint_path,
        case_dir=args.case_dir,
        routes_path=args.routes_path,
        graph_path=args.graph_path,
        flight_id=args.flight_id,
        takeoff_time=args.takeoff_time,
        use_tres_wind=not args.no_tres_wind,
        tailwind_fallback_kts=args.tailwind_fallback_kts,
        plot_route=args.plot_route,
        show_waypoints=args.show_waypoints,
        route_alpha=args.route_alpha,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

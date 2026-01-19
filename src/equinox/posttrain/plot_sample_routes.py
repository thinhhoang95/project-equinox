from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import networkx as nx
import cartopy.crs as ccrs

from equinox.helpers.plotters import plot_routes_on_map


def find_case_graph_gml(case_dir: Path) -> Path:
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
        "Multiple .gml files found; provide --graph-path explicitly:\n"
        f"{candidates}"
    )


def parse_routes_txt(routes_path: Path) -> list[list[str]]:
    routes: list[list[str]] = []
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


def plot_sample_routes(
    case_dir: Path,
    routes_path: Path,
    graph_path: Path | None = None,
    output_path: Path | None = None,
    show: bool = True,
    show_waypoints: bool = True,
    route_alpha: float = 1.0,
) -> tuple[plt.Figure, plt.Axes]:
    if graph_path is None:
        graph_path = find_case_graph_gml(case_dir)

    graph = nx.read_gml(graph_path)
    routes = parse_routes_txt(routes_path)
    if not 0.0 <= route_alpha <= 1.0:
        raise ValueError("route_alpha must be between 0.0 and 1.0")

    fig = plt.figure(figsize=(18, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    plot_routes_on_map(
        graph,
        routes,
        ax=ax,
        show_waypoints=show_waypoints,
        route_alpha=route_alpha,
    )
    ax.set_title(f"Sample routes ({routes_path.name})")

    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight")
    if show:
        plt.show()

    return fig, ax


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot sampled routes on a cartopy map."
    )
    parser.add_argument(
        "--case-dir",
        required=True,
        type=Path,
        help="Case directory containing the graph and inference outputs.",
    )
    parser.add_argument(
        "--routes-path",
        required=True,
        type=Path,
        help="Path to routes.txt (lines are 'cost,WAYPOINT ...').",
    )
    parser.add_argument(
        "--graph-path",
        type=Path,
        default=None,
        help="Optional explicit graph .gml path (overrides auto-detect).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional output image path to save the plot.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display the plot window.",
    )
    parser.add_argument(
        "--hide-waypoints",
        action="store_true",
        help="Do not plot waypoint markers or labels.",
    )
    parser.add_argument(
        "--route-alpha",
        type=float,
        default=1.0,
        help="Opacity for route lines (0.0 to 1.0).",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    plot_sample_routes(
        case_dir=args.case_dir,
        routes_path=args.routes_path,
        graph_path=args.graph_path,
        output_path=args.output_path,
        show=not args.no_show,
        show_waypoints=not args.hide_waypoints,
        route_alpha=args.route_alpha,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

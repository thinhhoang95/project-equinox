#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import networkx as nx

from equinox.sampling.pipeline import GraphScenario, _prepare_graph_for_scenario
from equinox.sampling.trespass.inference import load_case
from equinox.training.prep.prep_graph import plot_route_graph_with_sectors_pdf


def run_graph_export(
    *,
    case_dir: str,
    sectors_geojson_path: str,
    sectors_to_avoid: list[str],
    connectivity_repair_iterations: int,
    run_name: str | None = None,
) -> dict:
    case_path = Path(case_dir)
    if run_name is None:
        run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    out_dir = case_path / "err41" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    config, components = load_case(case_dir)
    origin = config.origin_node or components.get("origin_node")
    goal = config.goal_node or components.get("goal_node")
    if origin is None or goal is None:
        raise ValueError("Could not resolve origin/goal from case configuration.")

    scenario = GraphScenario(
        mode="edge_filter",
        sectors_to_avoid=sectors_to_avoid,
        sectors_geojson_path=sectors_geojson_path,
        post_sector_connectivity_repair=True,
        connectivity_repair_iterations=connectivity_repair_iterations,
    )
    updated_components, graph_report = _prepare_graph_for_scenario(
        config=config,
        components=components,
        graph_scenario=scenario,
    )

    base_gml = out_dir / "graph_base.gml"
    scenario_gml = out_dir / "graph_after_reroute.gml"
    nx.write_gml(components["graph"], base_gml)
    nx.write_gml(updated_components["graph"], scenario_gml)

    all_sectors = gpd.read_file(sectors_geojson_path)
    excluded = all_sectors[all_sectors["sector_id"].isin(sectors_to_avoid)]

    base_pdf = out_dir / "graph_base.pdf"
    scenario_pdf = out_dir / "graph_after_reroute.pdf"
    plot_route_graph_with_sectors_pdf(
        components["graph"],
        show_label=False,
        output_path=str(base_pdf),
        sectors_gdf=excluded,
        node_origin=origin,
        node_destination=goal,
    )
    plot_route_graph_with_sectors_pdf(
        updated_components["graph"],
        show_label=False,
        output_path=str(scenario_pdf),
        sectors_gdf=excluded,
        node_origin=origin,
        node_destination=goal,
    )

    report_path = out_dir / "graph_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(graph_report, f, indent=2)

    result = {
        "out_dir": str(out_dir),
        "base_pdf": str(base_pdf),
        "scenario_pdf": str(scenario_pdf),
        "base_gml": str(base_gml),
        "scenario_gml": str(scenario_gml),
        "graph_report_json": str(report_path),
        "graph_report": graph_report,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "ERR41 graph-only reroute export: load case graph, remove sector edges, "
            "repair connectivity, and export base/scenario GML+PDF."
        )
    )
    parser.add_argument(
        "--case-dir",
        default="data/cases/LGAV_LFPG",
        help="Case directory path.",
    )
    parser.add_argument(
        "--sectors-geojson-path",
        default="data/airspace/sectors.geojson",
        help="Path to sectors GeoJSON.",
    )
    parser.add_argument(
        "--sectors-to-avoid",
        nargs="+",
        default=["LFEEE"],
        help="Sector IDs to remove from the graph.",
    )
    parser.add_argument(
        "--connectivity-repair-iterations",
        type=int,
        default=20,
        help="Max iterations for post-removal connectivity repair.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run name under <case_dir>/err41/. Default is timestamped.",
    )
    args = parser.parse_args()

    result = run_graph_export(
        case_dir=args.case_dir,
        sectors_geojson_path=args.sectors_geojson_path,
        sectors_to_avoid=args.sectors_to_avoid,
        connectivity_repair_iterations=args.connectivity_repair_iterations,
        run_name=args.run_name,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

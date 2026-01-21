from __future__ import annotations

import ast
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import networkx as nx

from equinox.helpers.haversine import bearing, destination_point, haversine

Point = Tuple[float, float]


@dataclass(frozen=True)
class FlightReference:
    flight_id: str
    takeoff_time: int
    route_nodes: List[str]


def find_case_yaml(case_dir: Path) -> Path:
    case_path = Path(case_dir)
    if not case_path.exists():
        raise FileNotFoundError(f"Case directory not found: {case_dir}")
    default_path = case_path / "default.yaml"
    if default_path.exists():
        return default_path
    yaml_paths = sorted(case_path.glob("*.yaml"))
    if not yaml_paths:
        raise FileNotFoundError(f"No YAML configuration file found in {case_dir}")
    return yaml_paths[0]


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
        "Multiple .gml files found; provide case_dir that narrows it down:\n"
        f"{candidates}"
    )


def find_snapped_routes_csv(case_dir: Path) -> Path:
    preferred = case_dir / "tres_runs" / "all_routes_feasibly_snapped.csv"
    if preferred.exists():
        return preferred

    csv_candidates = sorted(case_dir.rglob("all_routes_feasibly_snapped.csv"))
    if not csv_candidates:
        raise FileNotFoundError(
            "No all_routes_feasibly_snapped.csv found under "
            f"{case_dir}. Provide reference inputs for a case that includes "
            "tres_runs outputs."
        )
    if len(csv_candidates) == 1:
        return csv_candidates[0]

    candidates = "\n".join(str(path) for path in csv_candidates)
    raise ValueError(
        "Multiple all_routes_feasibly_snapped.csv files found; narrow the case "
        f"directory:\n{candidates}"
    )


def parse_route_value(route_value: object) -> List[str]:
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


def load_snapped_routes(csv_path: Path) -> List[FlightReference]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Reference CSV {csv_path} has no headers.")
        if "flight_id" not in reader.fieldnames:
            raise ValueError(f"Reference CSV {csv_path} missing flight_id column.")

        timestamp_key = None
        for key in ("takeoff_time", "takeoff_ts", "takeoff_timestamp", "takeoff"):
            if key in reader.fieldnames:
                timestamp_key = key
                break
        if timestamp_key is None:
            raise ValueError(
                f"Reference CSV {csv_path} missing takeoff_time column."
            )

        flights: List[FlightReference] = []
        for row in reader:
            flight_id = (row.get("flight_id") or "").strip()
            if not flight_id:
                continue
            raw_ts = row.get(timestamp_key)
            if raw_ts is None:
                continue
            try:
                takeoff_time = int(float(raw_ts))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid takeoff_time {raw_ts!r} for flight {flight_id} in {csv_path}"
                ) from exc

            route_nodes = parse_route_value(row.get("route", ""))
            flights.append(
                FlightReference(
                    flight_id=flight_id,
                    takeoff_time=takeoff_time,
                    route_nodes=route_nodes,
                )
            )
    return flights


def collapse_consecutive(route_nodes: Sequence[str]) -> List[str]:
    collapsed: List[str] = []
    for node in route_nodes:
        if not collapsed or collapsed[-1] != node:
            collapsed.append(node)
    return collapsed


def build_node_coords(graph: nx.Graph) -> Dict[str, Point]:
    coords: Dict[str, Point] = {}
    for node, attrs in graph.nodes(data=True):
        if "lat" not in attrs or "lon" not in attrs:
            raise ValueError(f"Node {node!r} missing lat/lon attributes.")
        coords[str(node)] = (float(attrs["lat"]), float(attrs["lon"]))
    return coords


def normalize_route_nodes(
    route_nodes: Sequence[str],
    node_coords: Dict[str, Point],
    *,
    missing_policy: str = "strict",
) -> Tuple[List[str], List[str]]:
    if missing_policy not in {"strict", "drop"}:
        raise ValueError("missing_policy must be 'strict' or 'drop'.")

    cleaned: List[str] = []
    missing: List[str] = []
    for node in route_nodes:
        node_str = str(node)
        if node_str not in node_coords:
            missing.append(node_str)
            if missing_policy == "strict":
                raise KeyError(f"Waypoint {node_str!r} not found in graph.")
            continue
        if not cleaned or cleaned[-1] != node_str:
            cleaned.append(node_str)
    return cleaned, missing


def route_nodes_to_coords(
    route_nodes: Sequence[str],
    node_coords: Dict[str, Point],
) -> List[Point]:
    coords: List[Point] = []
    for node in route_nodes:
        if node not in node_coords:
            raise KeyError(f"Waypoint {node!r} not found in graph.")
        coords.append(node_coords[node])
    return coords


def resample_polyline(points: Sequence[Point], spacing_nm: float) -> List[Point]:
    if spacing_nm <= 0 or len(points) < 2:
        return list(points)

    distances = [0.0]
    for idx in range(1, len(points)):
        prev = points[idx - 1]
        curr = points[idx]
        seg_len = haversine(prev[0], prev[1], curr[0], curr[1])
        distances.append(distances[-1] + float(seg_len))

    total = distances[-1]
    if total == 0.0:
        return [points[0]]

    targets: List[float] = []
    current = 0.0
    while current < total:
        targets.append(current)
        current += spacing_nm
    targets.append(total)

    resampled: List[Point] = []
    seg_idx = 0
    for target in targets:
        while seg_idx < len(distances) - 1 and distances[seg_idx + 1] < target:
            seg_idx += 1

        if seg_idx >= len(points) - 1:
            point = points[-1]
        else:
            start = points[seg_idx]
            end = points[seg_idx + 1]
            seg_start = distances[seg_idx]
            seg_len = distances[seg_idx + 1] - seg_start
            offset = target - seg_start
            if seg_len <= 0 or offset <= 0:
                point = start
            elif offset >= seg_len:
                point = end
            else:
                seg_bearing = bearing(start, end)
                point = destination_point(start, seg_bearing, offset)

        if not resampled or point != resampled[-1]:
            resampled.append(point)

    return resampled

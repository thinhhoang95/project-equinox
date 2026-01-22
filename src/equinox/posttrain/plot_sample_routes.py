from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import cartopy.crs as ccrs
import xarray as xr
import yaml

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


def load_reference_route(
    case_dir: Path,
    flight_id: str,
    takeoff_timestamp: str,
) -> list[str]:
    csv_path = find_snapped_routes_csv(case_dir)
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Reference CSV {csv_path} has no headers.")
        if "flight_id" not in reader.fieldnames:
            raise ValueError(f"Reference CSV {csv_path} missing flight_id column.")

        timestamp_key = None
        if "takeoff_time" in reader.fieldnames:
            timestamp_key = "takeoff_time"
        elif "takeoff_ts" in reader.fieldnames:
            timestamp_key = "takeoff_ts"
        else:
            raise ValueError(
                f"Reference CSV {csv_path} missing takeoff_time column."
            )

        takeoff_timestamp = str(takeoff_timestamp).strip()
        matches = [
            row
            for row in reader
            if row.get("flight_id") == flight_id
            and row.get(timestamp_key) == takeoff_timestamp
        ]

    if not matches:
        raise ValueError(
            "No reference route found for flight_id "
            f"{flight_id!r} and takeoff timestamp {takeoff_timestamp!r} "
            f"in {csv_path}."
        )
    if len(matches) > 1:
        raise ValueError(
            "Multiple reference routes found for flight_id "
            f"{flight_id!r} and takeoff timestamp {takeoff_timestamp!r} "
            f"in {csv_path}."
        )

    route_text = matches[0].get("route", "").strip()
    if not route_text:
        raise ValueError(
            f"Reference route entry in {csv_path} has an empty route field."
        )
    return route_text.split()


def load_snapped_routes(case_dir: Path) -> list[list[str]]:
    csv_path = find_snapped_routes_csv(case_dir)
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Reference CSV {csv_path} has no headers.")
        if "route" not in reader.fieldnames:
            raise ValueError(f"Reference CSV {csv_path} missing route column.")

        routes: list[list[str]] = []
        for row in reader:
            route_text = (row.get("route") or "").strip()
            if not route_text:
                continue
            nodes = route_text.split()
            if nodes:
                routes.append(nodes)
    return routes


def plot_reference_route(
    graph: nx.Graph,
    case_dir: Path,
    flight_id: str,
    takeoff_timestamp: str,
    ax: plt.Axes,
) -> None:
    route_nodes = load_reference_route(case_dir, flight_id, takeoff_timestamp)
    route_lons: list[float] = []
    route_lats: list[float] = []
    for waypoint_name in route_nodes:
        if waypoint_name in graph.nodes:
            node_data = graph.nodes[waypoint_name]
            route_lons.append(node_data["lon"])
            route_lats.append(node_data["lat"])
        else:
            print(
                "Warning: Waypoint "
                f"{waypoint_name!r} in reference route not found in graph."
            )

    if route_lons and route_lats:
        ax.plot(
            route_lons,
            route_lats,
            linestyle=":",
            linewidth=2.5,
            color="black",
            transform=ccrs.Geodetic(),
            label="Reference route",
        )


def plot_sample_routes(
    case_dir: Path,
    routes_path: Path,
    graph_path: Path | None = None,
    output_path: Path | None = None,
    show: bool = True,
    show_waypoints: bool = True,
    route_alpha: float = 1.0,
    reference_flight_id: str | None = None,
    reference_takeoff_timestamp: str | None = None,
    plot_wind: bool = False,
    wind_date: str | None = None,
    wind_data_dir: Path | None = None,
    wind_time_idx: int | None = None,
    wind_altitude_ft: float | None = None,
    show_quiver: bool = True,
    thickness: float = 1,
    plot_title: str = None,
) -> tuple[plt.Figure, plt.Axes]:
    if graph_path is None:
        graph_path = find_case_graph_gml(case_dir)

    graph = nx.read_gml(graph_path)
    routes = parse_routes_txt(routes_path)
    if not 0.0 <= route_alpha <= 1.0:
        raise ValueError("route_alpha must be between 0.0 and 1.0")

    fig = plt.figure(figsize=(18, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_prop_cycle(color=["white"])
    if plot_wind:
        extent = _get_routes_extent(graph, routes, show_waypoints=show_waypoints)
        _plot_wind_map(
            case_dir=case_dir,
            ax=ax,
            extent=extent,
            wind_date=wind_date,
            wind_data_dir=wind_data_dir,
            wind_time_idx=wind_time_idx,
            wind_altitude_ft=wind_altitude_ft,
            reference_takeoff_timestamp=reference_takeoff_timestamp,
            show_quiver=show_quiver,
        )
    plot_routes_on_map(
        graph,
        routes,
        ax=ax,
        show_waypoints=show_waypoints,
        route_alpha=route_alpha,
        thickness=thickness
    )
    if reference_flight_id or reference_takeoff_timestamp:
        if not reference_flight_id or not reference_takeoff_timestamp:
            raise ValueError(
                "Both reference_flight_id and reference_takeoff_timestamp "
                "must be provided to plot a reference route."
            )
        plot_reference_route(
            graph,
            case_dir,
            reference_flight_id,
            reference_takeoff_timestamp,
            ax=ax,
        )
    if plot_title is None:
        ax.set_title(f"Sample routes ({routes_path.name})")
    else:
        ax.set_title(plot_title)

    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight")
    if show:
        plt.show()

    return fig, ax


def plot_snapped_routes(
    case_dir: Path,
    graph_path: Path | None = None,
    output_path: Path | None = None,
    show: bool = True,
    show_waypoints: bool = True,
    route_alpha: float = 1.0,
    thickness: float = 1,
    plot_title: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    if graph_path is None:
        graph_path = find_case_graph_gml(case_dir)

    graph = nx.read_gml(graph_path)
    routes = load_snapped_routes(case_dir)
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
        thickness=thickness,
    )

    if plot_title is None:
        ax.set_title("All snapped routes")
    else:
        ax.set_title(plot_title)

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
    parser.add_argument(
        "--reference-flight-id",
        type=str,
        default=None,
        help="Optional flight_id to overlay as a dotted reference route.",
    )
    parser.add_argument(
        "--reference-takeoff-timestamp",
        type=str,
        default=None,
        help="Takeoff timestamp matching the reference flight_id.",
    )
    parser.add_argument(
        "--plot-wind",
        action="store_true",
        help="Overlay ERA5 wind speed + direction on the map.",
    )
    parser.add_argument(
        "--wind-date",
        type=str,
        default=None,
        help="ERA5 date (YYYY-MM-DD). Defaults to case config or takeoff timestamp.",
    )
    parser.add_argument(
        "--wind-data-dir",
        type=Path,
        default=None,
        help="Directory containing ERA5 NetCDF files (default: data/era5).",
    )
    parser.add_argument(
        "--wind-time-idx",
        type=int,
        default=None,
        help="ERA5 time index to plot (default: nearest takeoff time or 0).",
    )
    parser.add_argument(
        "--wind-altitude-ft",
        type=float,
        default=None,
        help="Altitude (ft) used to select the closest pressure level.",
    )
    parser.add_argument(
        "--no-wind-quiver",
        action="store_true",
        help="Disable wind-direction quivers on the wind overlay.",
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
        reference_flight_id=args.reference_flight_id,
        reference_takeoff_timestamp=args.reference_takeoff_timestamp,
        plot_wind=args.plot_wind,
        wind_date=args.wind_date,
        wind_data_dir=args.wind_data_dir,
        wind_time_idx=args.wind_time_idx,
        wind_altitude_ft=args.wind_altitude_ft,
        show_quiver=not args.no_wind_quiver,
    )
    return 0


def _load_case_config(case_dir: Path) -> dict:
    config_path = case_dir / "default.yaml"
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _parse_takeoff_timestamp(value: str | None) -> datetime | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    if raw.isdigit():
        return datetime.utcfromtimestamp(int(raw))
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def _pressure_to_altitude_m(p_hpa: np.ndarray) -> np.ndarray:
    p0 = 1013.25
    t0 = 288.15
    lapse = 0.0065
    r_specific = 287.058
    g = 9.80665
    exponent = (r_specific * lapse) / g
    p_ratio = np.maximum(p_hpa, 1e-3) / p0
    return (t0 / lapse) * (1 - p_ratio**exponent)


def _get_routes_extent(
    graph: nx.Graph,
    routes: list[list[str]],
    show_waypoints: bool,
    buffer_deg: float = 1.0,
) -> list[float] | None:
    lons: list[float] = []
    lats: list[float] = []

    if show_waypoints:
        for _, data in graph.nodes(data=True):
            lons.append(data["lon"])
            lats.append(data["lat"])

    for route in routes:
        for waypoint_name in route:
            if waypoint_name in graph.nodes:
                node_data = graph.nodes[waypoint_name]
                lons.append(node_data["lon"])
                lats.append(node_data["lat"])

    if not lons or not lats:
        return None

    min_lon, max_lon = min(lons) - buffer_deg, max(lons) + buffer_deg
    min_lat, max_lat = min(lats) - buffer_deg, max(lats) + buffer_deg
    return [min_lon, max_lon, min_lat, max_lat]


def _plot_wind_map(
    case_dir: Path,
    ax: plt.Axes,
    extent: list[float] | None,
    wind_date: str | None,
    wind_data_dir: Path | None,
    wind_time_idx: int | None,
    wind_altitude_ft: float | None,
    reference_takeoff_timestamp: str | None,
    show_quiver: bool,
) -> None:
    case_config = _load_case_config(case_dir)
    if wind_data_dir is None:
        wind_data_dir_value = case_config.get("wind_data_dir")
        wind_data_dir = Path(wind_data_dir_value) if wind_data_dir_value else Path("data/era5")

    takeoff_dt = _parse_takeoff_timestamp(reference_takeoff_timestamp)
    if wind_date is None and takeoff_dt is not None:
        wind_date = takeoff_dt.date().isoformat()

    if wind_date is None:
        raise ValueError(
            "plot_wind=True requires a takeoff timestamp to infer wind_date "
            "or an explicit --wind-date."
        )

    era_path = wind_data_dir / f"{wind_date}.nc"
    if not era_path.exists():
        raise FileNotFoundError(f"ERA5 file not found: {era_path}")

    ds = xr.open_dataset(era_path)
    time_dim = "valid_time" if "valid_time" in ds.dims else "time"

    if wind_time_idx is None and takeoff_dt is not None:
        times = np.asarray(ds[time_dim].values).astype("datetime64[ns]")
        target = np.datetime64(takeoff_dt)
        wind_time_idx = int(np.argmin(np.abs(times - target)))
    if wind_time_idx is None:
        wind_time_idx = 0

    pressure_levels = None
    level_idx = None
    u_var = "u10"
    v_var = "v10"

    if "pressure_level" in ds.dims:
        pressure_levels = np.asarray(ds["pressure_level"].values, dtype=float)
        altitudes_m = _pressure_to_altitude_m(pressure_levels)
        altitudes_ft = altitudes_m * 3.28084
        if wind_altitude_ft is not None:
            level_idx = int(np.argmin(np.abs(altitudes_ft - wind_altitude_ft)))
        else:
            level_idx = int(np.argmax(altitudes_ft))
        u_var = "u"
        v_var = "v"

    if level_idx is not None:
        u = ds[u_var].isel({time_dim: wind_time_idx, "pressure_level": level_idx})
        v = ds[v_var].isel({time_dim: wind_time_idx, "pressure_level": level_idx})
        level_label = f"{pressure_levels[level_idx]:.0f} hPa"
    else:
        u = ds[u_var].isel({time_dim: wind_time_idx})
        v = ds[v_var].isel({time_dim: wind_time_idx})
        level_label = "10 m"

    speed = np.sqrt(u**2 + v**2)
    levels = np.linspace(0, float(np.nanmax(speed.values)), 31)

    if extent:
        ax.set_extent(extent, crs=ccrs.PlateCarree())

    contour = ax.contourf(
        u.longitude,
        u.latitude,
        speed,
        levels=levels,
        cmap="viridis",
        alpha=1.0,
        transform=ccrs.PlateCarree(),
        zorder=1,
    )
    ax.figure.colorbar(contour, ax=ax, orientation="vertical", label="Wind speed (m/s)")

    if show_quiver:
        quiver_step = 5
        lon_2d, lat_2d = np.meshgrid(u.longitude.values, u.latitude.values)
        ax.quiver(
            lon_2d[::quiver_step, ::quiver_step],
            lat_2d[::quiver_step, ::quiver_step],
            u.values[::quiver_step, ::quiver_step],
            v.values[::quiver_step, ::quiver_step],
            scale=700,
            width=0.003,
            headwidth=3,
            headlength=4,
            color="white",
            alpha=0.3,
            transform=ccrs.PlateCarree(),
            zorder=2,
        )

    time_value = ds[time_dim].values[wind_time_idx]
    ax.text(
        0.01,
        0.01,
        f"Wind {level_label} @ {time_value}",
        transform=ax.transAxes,
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.7, boxstyle="round,pad=0.2"),
        zorder=3,
    )


if __name__ == "__main__":
    raise SystemExit(main())

"""Evaluate sampled routes with discrete Frechet distance.

Usage example:
    /Users/thinhhoang/miniforge3/envs/equinox/bin/python -m equinox.evals.eval_sampling_frechet \
        --case-dir data/cases/LGAV_LFPG \
        --results-dir results_nogaugezeroridge \
        --n-samples 100 \
        --policy sample \
        --resample-spacing-nm 25 \
        --device cpu \
        --top-k 3

Metric notes (all distances are in nautical miles; smaller is better):
    min_frechet_nm_all: Best (closest) sampled route to the reference; lower means
        the model can generate a very close route.
    mean_frechet_nm_all: Average distance across all samples (duplicates weighted);
        lower means more probability mass near the reference.
    median_frechet_nm_all: Median distance across all samples; lower means typical
        samples are closer to the reference.
    max_frechet_nm_all: Worst sampled distance; larger indicates outlier routes.
    var_frechet_nm2_all: Variance of distances; larger means more spread.

    *_unique metrics are computed after de-duplicating sampled routes (support view).
    coverage_all_tau_*: Fraction of samples with distance <= tau; higher is better.
    hit_tau_*: Whether any sample is within tau; 1.0 means at least one close route.
    min_frechet_nm_top_k_cost: For each flight, select the lowest Frechet distance
        among the top-k lowest-cost samples; aggregated in summary.json.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import torch
import yaml

from equinox.evals.frechet import discrete_frechet
from equinox.evals.routes import (
    build_node_coords,
    find_case_graph_gml,
    find_case_yaml,
    find_snapped_routes_csv,
    load_snapped_routes,
    normalize_route_nodes,
    resample_polyline,
    route_nodes_to_coords,
)
from equinox.evals.wind_optimal_baseline import choose_min_time_sample
from equinox.helpers.haversine import haversine
from equinox.sampling.pipeline import compute_4d_path_for_dataset
from equinox.sampling.trespass.inference import compute_checkpoint_hash, resolve_checkpoint

Point = Tuple[float, float]

_WORKER_CONTEXT: Dict[str, object] = {}


@dataclass(frozen=True)
class FlightSpec:
    flight_id: str
    takeoff_time: int
    reference_route: List[str]


def _stable_seed(text: str) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def _sanitize_flight_key(flight_id: str, takeoff_time: int) -> str:
    safe_flight = re.sub(r"[^A-Za-z0-9_.-]+", "_", flight_id.strip())
    return f"{safe_flight}_{takeoff_time}"


def _point_distance_nm(left: Point, right: Point) -> float:
    return float(haversine(left[0], left[1], right[0], right[1]))


def _prepare_route(
    route_nodes: Sequence[str],
    node_coords: Dict[str, Point],
    *,
    missing_policy: str,
    resample_spacing_nm: Optional[float],
) -> Tuple[List[str], List[Point], List[str]]:
    cleaned_nodes, missing = normalize_route_nodes(
        route_nodes,
        node_coords,
        missing_policy=missing_policy,
    )
    if len(cleaned_nodes) < 2:
        raise ValueError("Route has fewer than 2 valid waypoints after cleanup.")
    coords = route_nodes_to_coords(cleaned_nodes, node_coords)
    if resample_spacing_nm:
        coords = resample_polyline(coords, resample_spacing_nm)
    if len(coords) < 2:
        raise ValueError("Route has fewer than 2 points after resampling.")
    return cleaned_nodes, coords, missing


def _compute_metrics(
    *,
    sample_routes: Sequence[Sequence[str]],
    sample_costs: Optional[Sequence[float]],
    reference_route: Sequence[str],
    node_coords: Dict[str, Point],
    missing_policy: str,
    resample_spacing_nm: Optional[float],
    coverage_thresholds: Sequence[float],
    top_k: int,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    if sample_costs is not None and len(sample_costs) != len(sample_routes):
        raise ValueError("sample_costs must have the same length as sample_routes.")

    reference_nodes, reference_coords, missing_ref = _prepare_route(
        reference_route,
        node_coords,
        missing_policy=missing_policy,
        resample_spacing_nm=resample_spacing_nm,
    )

    route_counts: Dict[Tuple[str, ...], int] = {}
    route_coords: Dict[Tuple[str, ...], List[Point]] = {}
    cost_samples: List[Tuple[Tuple[str, ...], float]] = []
    dropped_samples = 0
    missing_samples = 0

    for idx, route in enumerate(sample_routes):
        try:
            cleaned_nodes, coords, missing = _prepare_route(
                route,
                node_coords,
                missing_policy=missing_policy,
                resample_spacing_nm=resample_spacing_nm,
            )
        except KeyError:
            if missing_policy == "strict":
                raise
            dropped_samples += 1
            continue
        except Exception:
            dropped_samples += 1
            continue
        missing_samples += len(missing)
        key = tuple(cleaned_nodes)
        route_counts[key] = route_counts.get(key, 0) + 1
        if key not in route_coords:
            route_coords[key] = coords
        if sample_costs is not None:
            try:
                cost_value = float(sample_costs[idx])
            except (TypeError, ValueError):
                cost_value = None
            if cost_value is not None and math.isfinite(cost_value):
                cost_samples.append((key, cost_value))

    if not route_counts:
        raise ValueError("No valid sampled routes to evaluate.")

    distance_by_route: Dict[Tuple[str, ...], float] = {}
    for route_key, coords in route_coords.items():
        distance_by_route[route_key] = discrete_frechet(
            coords,
            reference_coords,
            point_dist_fn=_point_distance_nm,
        )

    distances_unique = [distance_by_route[key] for key in route_counts]
    distances_all: List[float] = []
    counts = []
    for route_key, count in route_counts.items():
        dist = distance_by_route[route_key]
        distances_all.extend([dist] * count)
        counts.append(count)

    distances_all_arr = np.array(distances_all, dtype=float)
    distances_unique_arr = np.array(distances_unique, dtype=float)

    n_samples_valid = int(distances_all_arr.size)
    n_unique = int(distances_unique_arr.size)
    duplicate_rate = float(n_unique / n_samples_valid) if n_samples_valid else 0.0

    probs = np.array(counts, dtype=float) / n_samples_valid
    entropy_nats = float(-(probs * np.log(probs)).sum()) if n_samples_valid else 0.0
    entropy_bits = float(entropy_nats / math.log(2.0)) if n_samples_valid else 0.0

    metrics: Dict[str, object] = {
        "n_samples_valid": n_samples_valid,
        "n_unique_routes": n_unique,
        "duplicate_rate": duplicate_rate,
        "entropy_bits": entropy_bits,
        "min_frechet_nm_all": float(np.min(distances_all_arr)),
        "mean_frechet_nm_all": float(np.mean(distances_all_arr)),
        "median_frechet_nm_all": float(np.median(distances_all_arr)),
        "max_frechet_nm_all": float(np.max(distances_all_arr)),
        "var_frechet_nm2_all": float(np.var(distances_all_arr)),
        "min_frechet_nm_unique": float(np.min(distances_unique_arr)),
        "mean_frechet_nm_unique": float(np.mean(distances_unique_arr)),
        "median_frechet_nm_unique": float(np.median(distances_unique_arr)),
        "max_frechet_nm_unique": float(np.max(distances_unique_arr)),
        "var_frechet_nm2_unique": float(np.var(distances_unique_arr)),
        "dropped_waypoints_reference": len(missing_ref),
        "dropped_waypoints_samples": int(missing_samples),
        "invalid_sample_routes": int(dropped_samples),
    }

    for tau in coverage_thresholds:
        label = _format_tau_label(tau)
        metrics[f"coverage_all_tau_{label}"] = float(
            np.mean(distances_all_arr <= tau)
        )
        metrics[f"hit_tau_{label}"] = float(np.min(distances_all_arr) <= tau)

    if top_k > 0 and cost_samples:
        sorted_by_cost = sorted(cost_samples, key=lambda item: item[1])
        best_by_cost = sorted_by_cost[: min(top_k, len(sorted_by_cost))]
        best_by_cost_distances = [
            distance_by_route[route_key] for route_key, _ in best_by_cost
        ]
        if best_by_cost_distances:
            metrics["min_frechet_nm_top_k_cost"] = float(
                min(best_by_cost_distances)
            )

    closest_routes: List[Dict[str, object]] = []
    if top_k > 0:
        sorted_routes = sorted(
            route_counts.items(),
            key=lambda item: distance_by_route[item[0]],
        )[:top_k]
        for route_key, count in sorted_routes:
            closest_routes.append(
                {
                    "route": list(route_key),
                    "distance_nm": float(distance_by_route[route_key]),
                    "count": int(count),
                }
            )

    return metrics, closest_routes


def _format_tau_label(tau: float) -> str:
    label = f"{tau:g}"
    return label.replace(".", "p")


def _init_worker(context: Dict[str, object], torch_threads: int) -> None:
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = context
    if torch_threads > 0:
        torch.set_num_threads(torch_threads)
        try:
            torch.set_num_interop_threads(torch_threads)
        except RuntimeError:
            pass


def _evaluate_flight(spec: FlightSpec) -> Dict[str, object]:
    context = _WORKER_CONTEXT
    case_dir = context["case_dir"]
    checkpoint_path = context["checkpoint_path"]
    n_samples = context["n_samples"]
    policy = context["policy"]
    initial_k_policy = context["initial_k_policy"]
    gamma = context["gamma"]
    device = context["device"]
    cache_dir = context["cache_dir"]
    use_cache = context["use_cache"]
    output_root = context["output_root"]
    resample_spacing_nm = context["resample_spacing_nm"]
    missing_policy = context["missing_policy"]
    coverage_thresholds = context["coverage_thresholds"]
    top_k = context["top_k"]
    base_seed = context["seed"]
    node_coords = context["node_coords"]
    delta_t_seconds_wall_clock = context["delta_t_seconds_wall_clock"]

    flight_key = _sanitize_flight_key(spec.flight_id, spec.takeoff_time)
    output_dir = Path(output_root) / "inference" / flight_key
    routes_txt_path = output_dir / "routes.txt"

    if base_seed is None:
        seed = None
    else:
        seed_text = f"{spec.flight_id}:{spec.takeoff_time}:{base_seed}"
        seed = _stable_seed(seed_text)

    try:
        result = compute_4d_path_for_dataset(
            case_dir=str(case_dir),
            checkpoint_path=str(checkpoint_path),
            flight_id=spec.flight_id,
            takeoff_timestamp=spec.takeoff_time,
            gamma=gamma,
            n_samples=n_samples,
            policy=policy,
            initial_k_policy=initial_k_policy,
            return_4d=False,
            seed=seed,
            device=device,
            output_dir=str(output_dir),
            cache_dir=str(cache_dir) if cache_dir else None,
            use_cache=use_cache,
        )

        sample_routes = [sample.route for sample in result.samples]
        metrics, closest_routes = _compute_metrics(
            sample_routes=sample_routes,
            sample_costs=[sample.total_cost for sample in result.samples],
            reference_route=spec.reference_route,
            node_coords=node_coords,
            missing_policy=missing_policy,
            resample_spacing_nm=resample_spacing_nm,
            coverage_thresholds=coverage_thresholds,
            top_k=top_k,
        )

        wind_optimal_elapsed_time_s = None
        wind_optimal_frechet_nm = None
        try:
            states_list = [sample.states for sample in result.samples]
            choice = choose_min_time_sample(states_list, delta_t_seconds_wall_clock)
            wind_optimal_elapsed_time_s = float(choice.elapsed_time_seconds)
            baseline_route = result.samples[choice.sample_index].route
            _, baseline_coords, _ = _prepare_route(
                baseline_route,
                node_coords,
                missing_policy=missing_policy,
                resample_spacing_nm=resample_spacing_nm,
            )
            _, reference_coords, _ = _prepare_route(
                spec.reference_route,
                node_coords,
                missing_policy=missing_policy,
                resample_spacing_nm=resample_spacing_nm,
            )
            wind_optimal_frechet_nm = float(
                discrete_frechet(
                    baseline_coords,
                    reference_coords,
                    point_dist_fn=_point_distance_nm,
                )
            )
        except Exception:
            wind_optimal_elapsed_time_s = None
            wind_optimal_frechet_nm = None

        if closest_routes:
            closest_path = Path(output_root) / "closest_routes" / f"{flight_key}.json"
            closest_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "flight_id": spec.flight_id,
                "takeoff_time": spec.takeoff_time,
                "routes": closest_routes,
            }
            closest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        return {
            "flight_id": spec.flight_id,
            "takeoff_time": spec.takeoff_time,
            "status": "ok",
            "error_message": "",
            "n_samples_requested": int(n_samples),
            "routes_txt": str(routes_txt_path),
            "svi_cache_hit": bool(result.metadata.get("svi_cache_hit")),
            "wind_optimal_from_samples_elapsed_time_s": wind_optimal_elapsed_time_s,
            "wind_optimal_from_samples_frechet_nm": wind_optimal_frechet_nm,
            **metrics,
        }
    except Exception as exc:
        return {
            "flight_id": spec.flight_id,
            "takeoff_time": spec.takeoff_time,
            "status": "error",
            "error_message": str(exc),
            "n_samples_requested": int(n_samples),
            "routes_txt": str(routes_txt_path),
        }


def _aggregate_metrics(
    results: Sequence[Dict[str, object]],
    *,
    coverage_thresholds: Sequence[float],
) -> Dict[str, object]:
    ok_results = [row for row in results if row.get("status") == "ok"]
    metrics = {}

    def _summarize(values: List[float]) -> Optional[Dict[str, float]]:
        if not values:
            return None
        arr = np.array(values, dtype=float)
        return {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
        }

    metric_keys = [
        "min_frechet_nm_all",
        "mean_frechet_nm_all",
        "median_frechet_nm_all",
        "max_frechet_nm_all",
        "var_frechet_nm2_all",
        "min_frechet_nm_top_k_cost",
        "min_frechet_nm_unique",
        "mean_frechet_nm_unique",
        "median_frechet_nm_unique",
        "max_frechet_nm_unique",
        "var_frechet_nm2_unique",
        "duplicate_rate",
        "entropy_bits",
        "wind_optimal_from_samples_elapsed_time_s",
        "wind_optimal_from_samples_frechet_nm",
    ]

    for key in metric_keys:
        values = [float(row[key]) for row in ok_results if row.get(key) is not None]
        summary = _summarize(values)
        if summary:
            metrics[key] = summary

    coverage_summary = {}
    for tau in coverage_thresholds:
        label = _format_tau_label(tau)
        coverage_values = [
            float(row[f"coverage_all_tau_{label}"])
            for row in ok_results
            if row.get(f"coverage_all_tau_{label}") is not None
        ]
        hit_values = [
            float(row[f"hit_tau_{label}"])
            for row in ok_results
            if row.get(f"hit_tau_{label}") is not None
        ]
        coverage_stats = _summarize(coverage_values)
        hit_rate = float(np.mean(hit_values)) if hit_values else None
        if coverage_stats or hit_rate is not None:
            coverage_summary[str(tau)] = {
                "coverage": coverage_stats,
                "hit_rate": hit_rate,
            }

    status_counts = {}
    for row in results:
        status_counts[row.get("status", "unknown")] = (
            status_counts.get(row.get("status", "unknown"), 0) + 1
        )

    return {
        "metrics": metrics,
        "coverage": coverage_summary,
        "status_counts": status_counts,
        "num_flights_ok": len(ok_results),
        "num_flights_total": len(results),
    }


def _write_metrics_csv(
    output_path: Path,
    results: Sequence[Dict[str, object]],
    coverage_thresholds: Sequence[float],
) -> None:
    base_fields = [
        "flight_id",
        "takeoff_time",
        "status",
        "error_message",
        "n_samples_requested",
        "n_samples_valid",
        "n_unique_routes",
        "duplicate_rate",
        "entropy_bits",
        "min_frechet_nm_all",
        "mean_frechet_nm_all",
        "median_frechet_nm_all",
        "max_frechet_nm_all",
        "var_frechet_nm2_all",
        "min_frechet_nm_top_k_cost",
        "min_frechet_nm_unique",
        "mean_frechet_nm_unique",
        "median_frechet_nm_unique",
        "max_frechet_nm_unique",
        "var_frechet_nm2_unique",
        "dropped_waypoints_reference",
        "dropped_waypoints_samples",
        "invalid_sample_routes",
        "svi_cache_hit",
        "wind_optimal_from_samples_elapsed_time_s",
        "wind_optimal_from_samples_frechet_nm",
        "routes_txt",
    ]

    coverage_fields = []
    for tau in coverage_thresholds:
        label = _format_tau_label(tau)
        coverage_fields.append(f"coverage_all_tau_{label}")
        coverage_fields.append(f"hit_tau_{label}")

    fieldnames = base_fields + coverage_fields
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _load_graph(case_dir: Path, config_path: Optional[Path]) -> nx.Graph:
    graph_path = None
    if config_path and config_path.exists():
        try:
            from equinox.config import RunConfiguration

            config = RunConfiguration.load_from_yaml(str(config_path))
            if config.graph_file_path:
                graph_path = Path(config.graph_file_path)
                if not graph_path.is_absolute():
                    graph_path = case_dir / graph_path
        except Exception:
            graph_path = None

    if graph_path is None or not graph_path.exists():
        graph_path = find_case_graph_gml(case_dir)

    return nx.read_gml(graph_path)


def _load_delta_t_seconds(config_path: Path) -> float:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config_dict = yaml.safe_load(handle)
    if not isinstance(config_dict, dict):
        raise ValueError(f"Invalid configuration in {config_path}; expected a mapping.")
    delta_t_seconds = config_dict.get("delta_t_seconds")
    if delta_t_seconds is None:
        raise ValueError(f"delta_t_seconds is missing in the case config: {config_path}")
    try:
        delta_t_seconds = float(delta_t_seconds)
    except (TypeError, ValueError) as exc:
        raise ValueError("delta_t_seconds must be a numeric value.") from exc
    if delta_t_seconds <= 0 or not math.isfinite(delta_t_seconds):
        raise ValueError("delta_t_seconds must be a positive finite value.")
    return delta_t_seconds


def _build_run_id(
    checkpoint_hash: str,
    n_samples: int,
    gamma: Optional[float],
    policy: str,
    resample_spacing_nm: Optional[float],
) -> str:
    gamma_part = f"{gamma:g}" if gamma is not None else "auto"
    gamma_part = gamma_part.replace(".", "p")
    resample_part = (
        f"rs{resample_spacing_nm:g}".replace(".", "p") if resample_spacing_nm else "rs0"
    )
    return f"{checkpoint_hash[:8]}_n{n_samples}_g{gamma_part}_p{policy}_{resample_part}"


def _parse_thresholds(raw: str) -> List[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def _resolve_checkpoint_with_results_dir(
    case_dir: Path,
    checkpoint_path: Optional[str],
    results_dir: Optional[str],
) -> Path:
    if checkpoint_path:
        return resolve_checkpoint(str(case_dir), checkpoint_path)

    if results_dir is None:
        return resolve_checkpoint(str(case_dir), None)

    results_path = Path(results_dir)
    if not results_path.is_absolute():
        results_path = case_dir / results_path

    if not results_path.exists():
        raise FileNotFoundError(f"Results directory not found: {results_path}")

    checkpoint_pattern = re.compile(r"^checkpoint_iter_(\d+)\.pt$")
    best_iter = None
    best_path = None
    for path in results_path.iterdir():
        match = checkpoint_pattern.match(path.name)
        if not match:
            continue
        iteration = int(match.group(1))
        if best_iter is None or iteration > best_iter:
            best_iter = iteration
            best_path = path

    if best_path is not None:
        return best_path

    final_results = results_path / "final_results.pt"
    if final_results.exists():
        return final_results

    raise FileNotFoundError(
        f"No checkpoint_iter_*.pt or final_results.pt found in {results_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate sampled routes with discrete Frechet distance."
    )
    parser.add_argument("--case-dir", required=True, help="Case directory path.")
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="Checkpoint file (default: newest in batch_sgd_results).",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Results directory containing checkpoints (default: batch_sgd_results).",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Case configuration YAML (default: case-dir/default.yaml).",
    )
    parser.add_argument("--n-samples", type=int, default=200, help="Samples per flight.")
    parser.add_argument(
        "--policy",
        default="sample",
        choices=["sample", "greedy"],
        help="Sampling policy.",
    )
    parser.add_argument(
        "--initial-k-policy",
        default="uniform",
        choices=["uniform", "zb"],
        help="Sampling policy for the initial wall-clock time bin k.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="Gamma override (default: checkpoint/config).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for inference (cpu, cuda, cuda:0).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Max parallel workers (default: cpu count or 1 for cuda).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Global seed for per-flight deterministic sampling.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Output root directory (default: case-dir/eval_temp/<run_id>).",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Cache directory for SVI (default: case-dir/inference_cache).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable SVI cache.",
    )
    parser.add_argument(
        "--missing-policy",
        default="strict",
        choices=["strict", "drop"],
        help="How to handle missing waypoints in graph.",
    )
    parser.add_argument(
        "--resample-spacing-nm",
        type=float,
        default=None,
        help="Resample route points to uniform spacing in NM.",
    )
    parser.add_argument(
        "--coverage-thresholds",
        default="5,10,20,50,100",
        help="Comma-separated Frechet thresholds in NM.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Store top-k closest unique routes per flight and compute top-k cost stats.",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help="Torch intra/inter-op threads per worker.",
    )
    parser.add_argument(
        "--flight-id",
        default=None,
        help="Optional single flight_id filter.",
    )
    parser.add_argument(
        "--takeoff-time",
        type=int,
        default=None,
        help="Optional takeoff time filter (requires flight-id for uniqueness).",
    )

    args = parser.parse_args()

    case_dir = Path(args.case_dir)
    config_path = Path(args.config) if args.config else find_case_yaml(case_dir)
    snapped_csv = find_snapped_routes_csv(case_dir)
    delta_t_seconds_wall_clock = _load_delta_t_seconds(config_path)

    checkpoint_path = _resolve_checkpoint_with_results_dir(
        case_dir,
        args.checkpoint_path,
        args.results_dir,
    )
    checkpoint_hash = compute_checkpoint_hash(checkpoint_path)

    coverage_thresholds = _parse_thresholds(args.coverage_thresholds)
    run_id = _build_run_id(
        checkpoint_hash=checkpoint_hash,
        n_samples=args.n_samples,
        gamma=args.gamma,
        policy=args.policy,
        resample_spacing_nm=args.resample_spacing_nm,
    )
    output_root = Path(args.output_root) if args.output_root else (
        case_dir / "eval_temp" / run_id
    )
    output_root.mkdir(parents=True, exist_ok=True)

    graph = _load_graph(case_dir, config_path)
    node_coords = build_node_coords(graph)

    flights = load_snapped_routes(snapped_csv)
    if args.flight_id:
        flights = [
            flight
            for flight in flights
            if flight.flight_id == args.flight_id
            and (args.takeoff_time is None or flight.takeoff_time == args.takeoff_time)
        ]
        if not flights:
            raise ValueError("No flights match the requested filters.")

    specs = [
        FlightSpec(
            flight_id=flight.flight_id,
            takeoff_time=flight.takeoff_time,
            reference_route=flight.route_nodes,
        )
        for flight in flights
    ]

    device = args.device
    if args.max_workers is None:
        if str(device).startswith("cuda"):
            max_workers = 1
        else:
            cpu_count = os.cpu_count() or 1
            max_workers = max(1, cpu_count - 1)
    else:
        max_workers = max(1, args.max_workers)

    cache_dir = Path(args.cache_dir) if args.cache_dir else case_dir / "inference_cache"

    context = {
        "case_dir": str(case_dir),
        "checkpoint_path": str(checkpoint_path),
        "n_samples": args.n_samples,
        "policy": args.policy,
        "initial_k_policy": args.initial_k_policy,
        "gamma": args.gamma,
        "device": device,
        "cache_dir": str(cache_dir),
        "use_cache": not args.no_cache,
        "output_root": str(output_root),
        "resample_spacing_nm": args.resample_spacing_nm,
        "missing_policy": args.missing_policy,
        "coverage_thresholds": coverage_thresholds,
        "top_k": args.top_k,
        "seed": args.seed,
        "node_coords": node_coords,
        "delta_t_seconds_wall_clock": delta_t_seconds_wall_clock,
    }

    from concurrent.futures import ProcessPoolExecutor, as_completed

    start_time = time.time()
    results: List[Dict[str, object]] = []
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker,
        initargs=(context, args.torch_threads),
    ) as executor:
        futures = {executor.submit(_evaluate_flight, spec): spec for spec in specs}
        total = len(futures)
        completed = 0
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1
            if completed == total or completed % 10 == 0:
                elapsed = time.time() - start_time
                print(f"Completed {completed}/{total} flights in {elapsed:.1f}s")

    metrics_csv = output_root / "metrics_per_flight.csv"
    _write_metrics_csv(metrics_csv, results, coverage_thresholds)

    summary = _aggregate_metrics(results, coverage_thresholds=coverage_thresholds)
    summary.update(
        {
            "case_dir": str(case_dir),
            "config_path": str(config_path),
            "snapped_routes_csv": str(snapped_csv),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_hash": checkpoint_hash,
            "results_dir": args.results_dir,
            "run_id": run_id,
            "n_samples": args.n_samples,
            "policy": args.policy,
            "gamma": args.gamma,
            "device": device,
            "max_workers": max_workers,
            "seed": args.seed,
            "missing_policy": args.missing_policy,
            "resample_spacing_nm": args.resample_spacing_nm,
            "coverage_thresholds": coverage_thresholds,
            "top_k": args.top_k,
            "cache_dir": str(cache_dir),
            "use_cache": not args.no_cache,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )

    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote metrics to {metrics_csv}")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()

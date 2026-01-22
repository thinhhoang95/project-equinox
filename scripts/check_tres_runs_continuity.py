#!/usr/bin/env python3
import argparse
import math
import os
import pickle
import re
from collections import Counter, defaultdict

import pandas as pd

from equinox.config import RunConfiguration
from equinox.dp.trespass.continuity import match_next_states


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check snapped route continuity against CLSR transitions."
    )
    parser.add_argument(
        "--case-dir",
        default="data/cases/LGAV_LFPG",
        help="Case directory containing default.yaml and runs folder.",
    )
    parser.add_argument(
        "--runs-dir",
        default=None,
        help="Runs directory containing batch* outputs (default: <case-dir>/tres_runs).",
    )
    parser.add_argument(
        "--routes-csv",
        default=None,
        help="Snapped routes CSV (default: <runs-dir>/all_routes_feasibly_snapped.csv).",
    )
    parser.add_argument(
        "--k-tolerance-bins",
        type=int,
        default=0,
        help="Allowed k-bin tolerance for continuity matching (default: 0).",
    )
    parser.add_argument(
        "--k-tolerance-seconds",
        type=float,
        default=None,
        help="Optional tolerance in seconds (converted using config delta_t_seconds).",
    )
    parser.add_argument(
        "--log-backward-snap-segments",
        action="store_true",
        help="Log segments that only match via backward-k snap.",
    )
    return parser.parse_args()


def load_route_map(path: str):
    df = pd.read_csv(path)
    if "takeoff_time" not in df.columns:
        raise ValueError(f"Expected takeoff_time column in {path}")
    route_map = {}
    for _, row in df.iterrows():
        key = (str(row["flight_id"]), int(row["takeoff_time"]))
        route_map[key] = row["route"].split()
    return route_map


def parse_clsr_filename(filename: str):
    match = re.match(r"^CLSR_(.+)_(\d+)\.pkl$", filename)
    if not match:
        return None
    return match.group(1), int(match.group(2))


def build_adjacency(transitions):
    edge_adj = defaultdict(list)
    for t in transitions:
        if len(t) < 10:
            continue
        u_idx, k_u, rho_u, alt_u, ph_u, v_idx, k_v, rho_v, alt_v, ph_v = t[:10]
        u_state = (u_idx, k_u, rho_u, ph_u)
        v_state = (v_idx, k_v, rho_v, ph_v)
        edge_adj[(u_idx, v_idx)].append((u_state, v_state))
    return edge_adj


def has_continuous_path(
    route_nodes,
    node_to_idx,
    edge_adj,
    k_tolerance_bins,
    backward_snap_counts=None,
):
    if len(route_nodes) < 2:
        return False, "route_too_short"
    for node in route_nodes:
        if node not in node_to_idx:
            return False, f"unknown_node:{node}"

    u0 = node_to_idx[route_nodes[0]]
    v0 = node_to_idx[route_nodes[1]]
    first_opts = edge_adj.get((u0, v0), [])
    if not first_opts:
        return False, f"missing_edge:{route_nodes[0]}->{route_nodes[1]}"

    current_states = {u_state for (u_state, _) in first_opts}

    for u_name, v_name in zip(route_nodes[:-1], route_nodes[1:]):
        u_idx = node_to_idx[u_name]
        v_idx = node_to_idx[v_name]
        opts = edge_adj.get((u_idx, v_idx), [])
        if not opts:
            return False, f"missing_edge:{u_name}->{v_name}"
        next_states, match_kind = match_next_states(
            opts, current_states, k_tolerance_bins
        )
        if not next_states:
            return False, f"no_chain:{u_name}->{v_name}"
        if match_kind == "backward" and backward_snap_counts is not None:
            backward_snap_counts[(u_name, v_name)] += 1
        current_states = next_states

    return True, ""


def main():
    args = parse_args()
    runs_dir = args.runs_dir or os.path.join(args.case_dir, "tres_runs")
    routes_csv = args.routes_csv or os.path.join(
        runs_dir, "all_routes_feasibly_snapped.csv"
    )
    config_path = os.path.join(args.case_dir, "default.yaml")

    if not os.path.exists(runs_dir):
        raise FileNotFoundError(f"Runs dir not found: {runs_dir}")
    if not os.path.exists(routes_csv):
        raise FileNotFoundError(f"Snapped routes CSV not found: {routes_csv}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = RunConfiguration.load_from_yaml(config_path)
    components = config.initialize_all_components()
    node_to_idx = components["node_to_idx"]
    k_tolerance_bins = args.k_tolerance_bins
    if k_tolerance_bins is None:
        k_tolerance_bins = 0
    if args.k_tolerance_seconds is not None:
        if config.delta_t_seconds is None or config.delta_t_seconds <= 0:
            raise ValueError(
                "delta_t_seconds must be set in config when using --k-tolerance-seconds."
            )
        k_tolerance_bins = int(
            math.ceil(args.k_tolerance_seconds / config.delta_t_seconds)
        )
    if k_tolerance_bins < 0:
        raise ValueError("k-tolerance-bins must be non-negative.")

    route_map = load_route_map(routes_csv)

    failures = []
    total = 0
    backward_snap_counts = Counter()

    for root, _, files in os.walk(runs_dir):
        for fname in files:
            parsed = parse_clsr_filename(fname)
            if not parsed:
                continue
            flight_id, takeoff_ts = parsed
            total += 1
            route = route_map.get((flight_id, takeoff_ts))
            if not route:
                failures.append((flight_id, takeoff_ts, "missing_route"))
                continue

            clsr_path = os.path.join(root, fname)
            try:
                with open(clsr_path, "rb") as f:
                    transitions = pickle.load(f)
            except Exception as exc:
                failures.append((flight_id, takeoff_ts, f"load_error:{exc}"))
                continue

            edge_adj = build_adjacency(transitions)
            ok, reason = has_continuous_path(
                route,
                node_to_idx,
                edge_adj,
                k_tolerance_bins,
                backward_snap_counts,
            )
            if not ok:
                failures.append((flight_id, takeoff_ts, reason))

    print(f"Checked {total} CLSR files in {runs_dir}.")
    total_backward_snaps = sum(backward_snap_counts.values())
    print(f"Backward snap segments: {total_backward_snaps}")
    if args.log_backward_snap_segments and total_backward_snaps:
        for (u_name, v_name), count in backward_snap_counts.most_common(50):
            print(f"  {u_name}->{v_name}: {count}")
    if failures:
        print(f"Failures: {len(failures)}")
        for flight_id, takeoff_ts, reason in failures[:50]:
            print(f"  {flight_id}_{takeoff_ts}: {reason}")
        raise SystemExit(1)

    print("All routes have a continuous state path.")


if __name__ == "__main__":
    main()

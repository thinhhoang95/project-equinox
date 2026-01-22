#!/usr/bin/env python3
import argparse
import pickle
from collections import Counter
from pathlib import Path
from typing import Iterable, Tuple

import networkx as nx
import pandas as pd
import torch

from equinox.config import RunConfiguration
from equinox.posttrain import load_cost_model_parameters
from equinox.preferences.disentanglement import (
    build_edge_list,
    compute_empirical_counts_from_routes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose 'Switzerland Paradox' edges: show empirical counts, learned preferences, "
            "and whether an edge ever appears in CLSR transitions."
        )
    )
    parser.add_argument(
        "--case-dir",
        default="data/cases/LGAV_LFPG",
        help="Case directory containing default.yaml, graphs/, and tres_runs/.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Checkpoint path (default: <case-dir>/results_full/final_results.pt if present, "
            "else the latest checkpoint_iter_*.pt)."
        ),
    )
    parser.add_argument(
        "--routes-csv",
        default=None,
        help="Routes CSV (default: <case-dir>/tres_runs/all_routes_feasibly_snapped.csv).",
    )
    parser.add_argument(
        "--edge",
        action="append",
        default=[],
        help="Edge to inspect as 'U->V'. Can be provided multiple times.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="How many extreme edges to print for missing-in-CLSR edges (default: 10).",
    )
    return parser.parse_args()


def _parse_edge_specs(edge_specs: Iterable[str]) -> list[Tuple[str, str]]:
    edges: list[Tuple[str, str]] = []
    for spec in edge_specs:
        if not isinstance(spec, str) or not spec.strip():
            continue
        if "->" not in spec:
            raise ValueError(f"Invalid --edge value {spec!r}; expected format 'U->V'.")
        left, right = spec.split("->", 1)
        u = left.strip()
        v = right.strip()
        if not u or not v:
            raise ValueError(f"Invalid --edge value {spec!r}; expected format 'U->V'.")
        edges.append((u, v))
    return edges


def _resolve_checkpoint(case_dir: Path, checkpoint_arg: str | None) -> Path:
    if checkpoint_arg:
        return Path(checkpoint_arg)

    final = case_dir / "results_full" / "final_results.pt"
    if final.exists():
        return final

    candidates = sorted((case_dir / "results_full").glob("checkpoint_iter_*.pt"))
    if not candidates:
        raise FileNotFoundError(
            "Could not locate a checkpoint. Expected final_results.pt or checkpoint_iter_*.pt "
            f"under {case_dir}/results_full."
        )
    return candidates[-1]


def _iter_clsr_files(case_dir: Path) -> Iterable[Path]:
    runs_dir = case_dir / "tres_runs"
    for path in runs_dir.glob("batch*/CLSR_*.pkl"):
        if path.name.startswith("._"):
            continue
        yield path


def _collect_clsr_edge_support(case_dir: Path) -> Counter:
    counts: Counter = Counter()
    for clsr_path in _iter_clsr_files(case_dir):
        with open(clsr_path, "rb") as f:
            transitions = pickle.load(f)
        for t in transitions:
            if not isinstance(t, tuple) or len(t) < 6:
                continue
            counts[(int(t[0]), int(t[5]))] += 1
    return counts


def main() -> None:
    args = parse_args()
    case_dir = Path(args.case_dir)
    if not case_dir.exists():
        raise FileNotFoundError(f"Case dir not found: {case_dir}")

    config_path = case_dir / "default.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = RunConfiguration.load_from_yaml(str(config_path))
    graph_path = Path(config.graph_file_path)
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph file not found: {graph_path}")

    routes_csv = args.routes_csv or str(case_dir / "tres_runs" / "all_routes_feasibly_snapped.csv")
    if not Path(routes_csv).exists():
        raise FileNotFoundError(f"Routes CSV not found: {routes_csv}")

    checkpoint_path = _resolve_checkpoint(case_dir, args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    inspected_edges = _parse_edge_specs(args.edge)
    if not inspected_edges:
        inspected_edges = [("LGAV", "OKIPA"), ("LGAV", "URELO"), ("LGAV", "OKIRA")]

    graph = nx.read_gml(str(graph_path))
    node_to_idx = {node: i for i, node in enumerate(graph.nodes())}
    idx_to_node = {i: node for node, i in node_to_idx.items()}

    routes_df = pd.read_csv(routes_csv)
    empirical = compute_empirical_counts_from_routes(
        routes_df["route"], node_to_idx, len(graph.nodes())
    )

    params = load_cost_model_parameters(str(checkpoint_path))
    pref = params["preference_matrix"].to(dtype=torch.float64)

    clsr_edge_counts = _collect_clsr_edge_support(case_dir)

    print(f"Case dir: {case_dir}")
    print(f"Graph: {graph_path} | nodes={graph.number_of_nodes()} edges={graph.number_of_edges()}")
    print(f"Routes: {routes_csv} | rows={len(routes_df)}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"CLSR files: {sum(1 for _ in _iter_clsr_files(case_dir))}")
    print(f"Unique (u_idx,v_idx) in CLSR: {len(clsr_edge_counts)}")

    print("\n**Inspected Edges**")
    for u, v in inspected_edges:
        in_graph = graph.has_node(u) and graph.has_node(v) and graph.has_edge(u, v)
        u_idx = node_to_idx.get(u)
        v_idx = node_to_idx.get(v)
        emp = float(empirical[u_idx, v_idx].item()) if u_idx is not None and v_idx is not None else None
        p_val = float(pref[u_idx, v_idx].item()) if u_idx is not None and v_idx is not None else None
        clsr_support = clsr_edge_counts.get((int(u_idx), int(v_idx)), 0) if u_idx is not None and v_idx is not None else 0
        print(
            f"- {u}->{v}: in_graph={in_graph} emp_count={emp} pref_p={p_val} "
            f"clsr_transitions={clsr_support}"
        )

    edge_u, edge_v = build_edge_list(graph, node_to_idx)
    graph_edges = {(int(u), int(v)) for u, v in zip(edge_u.tolist(), edge_v.tolist())}
    missing = sorted(graph_edges - set(clsr_edge_counts.keys()))
    if missing:
        print(f"\nEdges in routes.gml but NEVER appear in any CLSR transitions: {len(missing)}")
        top_k = max(int(args.top_k), 0)
        if top_k:
            missing_pref = []
            for u_idx, v_idx in missing:
                missing_pref.append((float(pref[u_idx, v_idx].item()), u_idx, v_idx))
            missing_pref.sort(key=lambda x: x[0])
            print(f"\nLowest preference p among missing edges (top {top_k}):")
            for p_val, u_idx, v_idx in missing_pref[:top_k]:
                print(
                    f"  {idx_to_node[u_idx]}->{idx_to_node[v_idx]} p={p_val:.6f} "
                    f"emp={float(empirical[u_idx, v_idx].item()):.0f}"
                )
            print(f"\nHighest preference p among missing edges (top {top_k}):")
            for p_val, u_idx, v_idx in reversed(missing_pref[-top_k:]):
                print(
                    f"  {idx_to_node[u_idx]}->{idx_to_node[v_idx]} p={p_val:.6f} "
                    f"emp={float(empirical[u_idx, v_idx].item()):.0f}"
                )
    else:
        print("\nAll graph edges appear at least once in CLSR transitions.")


if __name__ == "__main__":
    main()


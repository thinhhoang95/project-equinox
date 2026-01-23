import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable, Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import torch

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.equinox.config import RunConfiguration
from src.equinox.dp.trespass.amorwin.forward_svi_log_temp import forward_soft_value_iteration
from src.equinox.dp.trespass.amorwin.backward_svi_log_cost_temp import backward_soft_value_iteration
from src.equinox.dp.trespass.amorwin.backward_gradient import backward_gradient_pass
from src.equinox.training.batch_sgd_pipeline_parallel_tsb_truellh import (
    load_flight_tres_results,
    compute_empirical_counts_for_flight,
)


logger = logging.getLogger(__name__)


class ShapedCostModel(torch.nn.Module):
    def __init__(self, base_model: torch.nn.Module, phi: torch.Tensor) -> None:
        super().__init__()
        self.base_model = base_model
        self.register_buffer("phi", phi)

    def forward(
        self,
        edge_indices: Tuple[torch.Tensor, torch.Tensor],
        distance_matrix_d: torch.Tensor,
        airspace_charge_matrix_ac: torch.Tensor,
        tailwind_values_w: torch.Tensor,
    ) -> torch.Tensor:
        base_cost = self.base_model(
            edge_indices,
            distance_matrix_d,
            airspace_charge_matrix_ac,
            tailwind_values_w,
        )
        u_idx, v_idx = edge_indices
        shape_term = self.phi[v_idx] - self.phi[u_idx]
        if shape_term.dtype != base_cost.dtype:
            shape_term = shape_term.to(dtype=base_cost.dtype)
        return base_cost + shape_term


def _route_links(
    route: str,
    node_to_idx: Dict[str, int],
) -> List[Tuple[int, int]]:
    waypoints = route.split()
    links: List[Tuple[int, int]] = []
    for i in range(len(waypoints) - 1):
        u = waypoints[i]
        v = waypoints[i + 1]
        if u not in node_to_idx or v not in node_to_idx:
            return []
        links.append((node_to_idx[u], node_to_idx[v]))
    return links


def _state_space_dims(transitions: Iterable[Tuple[Any, ...]]) -> Tuple[int, int, int]:
    max_k_val = max(max(t[1] for t in transitions), max(t[6] for t in transitions))
    max_rho_val = max(max(t[2] for t in transitions), max(t[7] for t in transitions))
    max_phase_val = max(max(t[4] for t in transitions), max(t[9] for t in transitions))
    return max_k_val + 1, max_rho_val + 1, max_phase_val + 1


def _select_flight_with_full_links(
    routes_df: pd.DataFrame,
    node_to_idx: Dict[str, int],
    case_dir: str,
    match_index: int,
) -> Tuple[Dict[str, Any], List[Tuple], torch.Tensor]:
    if match_index < 0:
        raise ValueError("match_index must be non-negative.")
    match_count = 0
    for row in routes_df.itertuples(index=False):
        flight_id = getattr(row, "flight_id")
        takeoff_timestamp = int(getattr(row, "takeoff_time"))
        try:
            _, _, thinned_transitions, avg_tailwind_knots = load_flight_tres_results(
                case_dir, flight_id, takeoff_timestamp
            )
        except Exception as exc:
            logger.debug("Skipping %s_%s: %s", flight_id, takeoff_timestamp, exc)
            continue

        links = _route_links(getattr(row, "route"), node_to_idx)
        if not links:
            logger.debug("Skipping %s_%s: missing waypoints in graph", flight_id, takeoff_timestamp)
            continue

        transitions_by_link = {(t[0], t[5]) for t in thinned_transitions}
        if all(link in transitions_by_link for link in links):
            if match_count == match_index:
                flight_data = row._asdict()
                flight_data["takeoff_timestamp"] = takeoff_timestamp
                return flight_data, thinned_transitions, avg_tailwind_knots
            match_count += 1

    raise RuntimeError(
        "No flight found with full empirical links present in thinned transitions "
        f"at match index {match_index}."
    )


def _load_flight_by_id(
    routes_df: pd.DataFrame,
    node_to_idx: Dict[str, int],
    case_dir: str,
    flight_id: str,
    takeoff_timestamp: int,
) -> Tuple[Dict[str, Any], List[Tuple], torch.Tensor]:
    match = routes_df[(routes_df["flight_id"] == flight_id) & (routes_df["takeoff_time"] == takeoff_timestamp)]
    if match.empty:
        raise RuntimeError(f"Flight {flight_id}_{takeoff_timestamp} not found in routes CSV.")
    if len(match) > 1:
        raise RuntimeError(f"Flight {flight_id}_{takeoff_timestamp} appears multiple times in routes CSV.")

    row = match.iloc[0]
    _, _, thinned_transitions, avg_tailwind_knots = load_flight_tres_results(
        case_dir, flight_id, int(row["takeoff_time"])
    )

    links = _route_links(str(row["route"]), node_to_idx)
    if not links:
        raise RuntimeError(f"Flight {flight_id}_{takeoff_timestamp}: route contains missing waypoints in graph.")

    transitions_by_link = {(t[0], t[5]) for t in thinned_transitions}
    missing_links = [link for link in links if link not in transitions_by_link]
    if missing_links:
        raise RuntimeError(
            f"Flight {flight_id}_{takeoff_timestamp}: {len(missing_links)} route links missing in thinned transitions."
        )

    flight_data = row.to_dict()
    flight_data["takeoff_timestamp"] = int(row["takeoff_time"])
    return flight_data, thinned_transitions, avg_tailwind_knots


def _dump_route_link_breakdown(
    *,
    flight_data: Dict[str, Any],
    thinned_transitions: List[Tuple],
    avg_tailwind_knots: torch.Tensor,
    components: Dict[str, Any],
    cost_model_base: torch.nn.Module,
    phi: torch.Tensor,
) -> None:
    node_to_idx = components["node_to_idx"]
    idx_to_node = components["idx_to_node"]
    route = str(flight_data["route"])
    waypoints = route.split()
    if not waypoints:
        print("Route link breakdown: empty route string")
        return

    origin_wp = str(flight_data["origin"])
    dest_wp = str(flight_data["destination"])
    route_start = waypoints[0]
    route_end = waypoints[-1]

    print("Route consistency")
    print(f"  origin_col: {origin_wp}")
    print(f"  destination_col: {dest_wp}")
    print(f"  route_start: {route_start}")
    print(f"  route_end: {route_end}")
    print(f"  route_len_waypoints: {len(waypoints)}")
    print(f"  route_len_links: {max(0, len(waypoints) - 1)}")

    transitions_by_link: Dict[Tuple[int, int], List[int]] = {}
    for i, t in enumerate(thinned_transitions):
        transitions_by_link.setdefault((t[0], t[5]), []).append(i)

    sum_phi = 0.0
    sum_base = 0.0

    print("Per-link breakdown")
    print("  #  u -> v | base_cost | shape(phi[v]-phi[u]) | base+shape | n_trans | avg_tailwind_kts")
    with torch.no_grad():
        for i in range(len(waypoints) - 1):
            u_wp = waypoints[i]
            v_wp = waypoints[i + 1]
            u_idx = node_to_idx[u_wp]
            v_idx = node_to_idx[v_wp]

            transition_indices = transitions_by_link.get((u_idx, v_idx))
            if not transition_indices:
                raise RuntimeError(f"Missing link ({u_wp}, {v_wp}) in thinned transitions (unexpected).")
            avg_tailwind_for_link = avg_tailwind_knots[transition_indices].mean().to(components["device"])

            edge_u_indices = torch.tensor([u_idx], device=components["device"], dtype=torch.long)
            edge_v_indices = torch.tensor([v_idx], device=components["device"], dtype=torch.long)
            base_cost = cost_model_base(
                (edge_u_indices, edge_v_indices),
                components["dist_matrix"],
                components["ac_matrix"],
                avg_tailwind_for_link.unsqueeze(0),
            ).item()

            shape_term = (phi[v_idx] - phi[u_idx]).item()
            sum_phi += shape_term
            sum_base += base_cost

            u_name = idx_to_node[u_idx]
            v_name = idx_to_node[v_idx]
            print(
                f"  {i+1:2d}  {u_name} -> {v_name} | "
                f"{base_cost: .6f} | {shape_term: .6f} | {(base_cost + shape_term): .6f} | "
                f"{len(transition_indices):3d} | {avg_tailwind_for_link.item(): .6f}"
            )

    print("Route totals")
    print(f"  sum_base_cost: {sum_base:.6f}")
    print(f"  sum_shape_terms: {sum_phi:.6f}")
    if origin_wp in node_to_idx and dest_wp in node_to_idx:
        origin_idx = node_to_idx[origin_wp]
        dest_idx = node_to_idx[dest_wp]
        start_idx = node_to_idx[route_start]
        end_idx = node_to_idx[route_end]
        print(f"  phi(origin): {phi[origin_idx].item():.6f}")
        print(f"  phi(route_start): {phi[start_idx].item():.6f}")
        print(f"  phi(route_end): {phi[end_idx].item():.6f}")
        print(f"  phi(dest): {phi[dest_idx].item():.6f}")
        print(f"  telescoping(phi(route_end)-phi(route_start)): {(phi[end_idx]-phi[start_idx]).item():.6f}")
        print(f"  expected_DP_shift(phi(dest)-phi(origin)): {(phi[dest_idx]-phi[origin_idx]).item():.6f}")


def _compute_c_xi(
    *,
    cost_model: torch.nn.Module,
    route_links: List[Tuple[int, int]],
    thinned_transitions: List[Tuple],
    avg_tailwind_knots: torch.Tensor,
    dist_matrix: torch.Tensor,
    ac_matrix: torch.Tensor,
    device: torch.device,
) -> float:
    transitions_by_link: Dict[Tuple[int, int], List[int]] = {}
    for i, t in enumerate(thinned_transitions):
        transitions_by_link.setdefault((t[0], t[5]), []).append(i)

    c_xi = 0.0
    with torch.no_grad():
        for u_idx, v_idx in route_links:
            transition_indices = transitions_by_link.get((u_idx, v_idx))
            if not transition_indices:
                raise RuntimeError(f"Missing link ({u_idx}, {v_idx}) in thinned transitions.")

            avg_tailwind_for_link = avg_tailwind_knots[transition_indices].mean()
            edge_u_indices = torch.tensor([u_idx], device=device, dtype=torch.long)
            edge_v_indices = torch.tensor([v_idx], device=device, dtype=torch.long)
            link_cost = cost_model(
                (edge_u_indices, edge_v_indices),
                dist_matrix,
                ac_matrix,
                avg_tailwind_for_link.to(device).unsqueeze(0),
            ).item()
            c_xi += link_cost

    return c_xi


def _run_log_likelihood(
    *,
    cost_model: torch.nn.Module,
    flight_data: Dict[str, Any],
    thinned_transitions: List[Tuple],
    avg_tailwind_knots: torch.Tensor,
    components: Dict[str, Any],
    gamma: float,
) -> Tuple[float, float, float]:
    num_nodes = components["num_nodes"]
    num_time_bins_wall_clock, num_rho_bins, num_phases = _state_space_dims(thinned_transitions)

    if isinstance(avg_tailwind_knots, np.ndarray):
        avg_tailwind_knots = torch.from_numpy(avg_tailwind_knots)
    avg_tailwind_knots = avg_tailwind_knots.to(components["device"])

    v_f = forward_soft_value_iteration(
        state_transitions=thinned_transitions,
        avg_tailwind_knots_per_transition=avg_tailwind_knots,
        G=components["graph"],
        idx_to_node=components["idx_to_node"],
        origin_node_idx=components["node_to_idx"][flight_data["origin"]],
        cost_model=cost_model,
        num_nodes=num_nodes,
        num_time_bins_wall_clock=num_time_bins_wall_clock,
        num_rho_bins=num_rho_bins,
        num_phases=num_phases,
        distance_matrix_d=components["dist_matrix"],
        airspace_charge_matrix_ac=components["ac_matrix"],
        device=components["device"],
        gamma=gamma,
        verbose=False,
    )

    v_b, _ = backward_soft_value_iteration(
        state_transitions=thinned_transitions,
        avg_tailwind_knots_per_transition=avg_tailwind_knots,
        G=components["graph"],
        idx_to_node=components["idx_to_node"],
        goal_node_idx=components["node_to_idx"][flight_data["destination"]],
        cost_model=cost_model,
        num_nodes=num_nodes,
        num_time_bins_wall_clock=num_time_bins_wall_clock,
        num_rho_bins=num_rho_bins,
        num_phases=num_phases,
        distance_matrix_d=components["dist_matrix"],
        airspace_charge_matrix_ac=components["ac_matrix"],
        device=components["device"],
        gamma=gamma,
        verbose=False,
    )

    empirical_counts = compute_empirical_counts_for_flight(
        flight_data, components["node_to_idx"], num_nodes
    ).to(components["device"])

    _, _, log_partition_z = backward_gradient_pass(
        state_transitions=thinned_transitions,
        avg_tailwind_knots_per_transition=avg_tailwind_knots,
        V_f=v_f,
        V_b=v_b,
        cost_model=cost_model,
        empirical_counts=empirical_counts,
        origin_node_idx=components["node_to_idx"][flight_data["origin"]],
        num_nodes=num_nodes,
        distance_matrix_d=components["dist_matrix"],
        airspace_charge_matrix_ac=components["ac_matrix"],
        device=components["device"],
        gamma=gamma,
        verbose=False,
    )

    route_links = _route_links(flight_data["route"], components["node_to_idx"])
    c_xi = _compute_c_xi(
        cost_model=cost_model,
        route_links=route_links,
        thinned_transitions=thinned_transitions,
        avg_tailwind_knots=avg_tailwind_knots,
        dist_matrix=components["dist_matrix"],
        ac_matrix=components["ac_matrix"],
        device=components["device"],
    )

    if torch.isinf(torch.tensor(c_xi)) or torch.isinf(log_partition_z):
        log_likelihood = -float("inf")
    else:
        log_likelihood = (-c_xi + log_partition_z.item()) / gamma

    return log_likelihood, c_xi, log_partition_z.item()


def main() -> None:
    parser = argparse.ArgumentParser(description="Potential-shaping invariance litmus test.")
    parser.add_argument(
        "--case-dir",
        type=str,
        default="data/cases/LGAV_LFPG",
        help="Case directory with default.yaml and tres_runs.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config yaml (defaults to case_dir/default.yaml).",
    )
    parser.add_argument(
        "--phi-scale",
        type=float,
        default=1.0,
        help="Stddev scale for random phi.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for phi.",
    )
    parser.add_argument(
        "--max-flights",
        type=int,
        default=None,
        help="Limit number of flights scanned when searching for full-link coverage.",
    )
    parser.add_argument(
        "--match-index",
        type=int,
        default=0,
        help="Select the Nth flight that passes full-link coverage (0-based).",
    )
    parser.add_argument(
        "--flight-id",
        type=str,
        default=None,
        help="Select a specific flight_id from the routes CSV (requires --takeoff-timestamp).",
    )
    parser.add_argument(
        "--takeoff-timestamp",
        type=int,
        default=None,
        help="Takeoff timestamp (unix seconds) for --flight-id selection.",
    )
    parser.add_argument(
        "--dump-links",
        action="store_true",
        help="Print per-link breakdown for the selected flight.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    case_dir = args.case_dir
    config_path = args.config or str(Path(case_dir) / "default.yaml")
    config = RunConfiguration.load_from_yaml(config_path)

    components = config.initialize_all_components()
    components["device"] = torch.device("cpu")
    components["dist_matrix"] = torch.as_tensor(
        components["dist_matrix"], dtype=torch.float64, device=components["device"]
    )
    components["ac_matrix"] = torch.as_tensor(
        components["ac_matrix"], dtype=torch.float64, device=components["device"]
    )
    components["cost_model"] = components["cost_model"].to(components["device"])

    routes_path = Path(case_dir) / "tres_runs" / "all_routes_feasibly_snapped.csv"
    routes_df = pd.read_csv(routes_path)
    if args.max_flights is not None:
        routes_df = routes_df.head(args.max_flights)

    if args.flight_id is not None or args.takeoff_timestamp is not None:
        if args.flight_id is None or args.takeoff_timestamp is None:
            raise SystemExit("--flight-id and --takeoff-timestamp must be provided together.")
        flight_data, thinned_transitions, avg_tailwind_knots = _load_flight_by_id(
            routes_df,
            components["node_to_idx"],
            case_dir,
            args.flight_id,
            int(args.takeoff_timestamp),
        )
    else:
        flight_data, thinned_transitions, avg_tailwind_knots = _select_flight_with_full_links(
            routes_df, components["node_to_idx"], case_dir, args.match_index
        )
    logger.info(
        "Selected flight %s_%s (%s -> %s)",
        flight_data["flight_id"],
        flight_data["takeoff_timestamp"],
        flight_data["origin"],
        flight_data["destination"],
    )

    torch.manual_seed(args.seed)
    phi = torch.randn(
        components["num_nodes"],
        dtype=torch.float64,
        device=components["device"],
    ) * float(args.phi_scale)
    goal_idx = components["node_to_idx"][flight_data["destination"]]
    phi = phi - phi[goal_idx]

    base_ll, base_c, base_v = _run_log_likelihood(
        cost_model=components["cost_model"],
        flight_data=flight_data,
        thinned_transitions=thinned_transitions,
        avg_tailwind_knots=avg_tailwind_knots,
        components=components,
        gamma=float(config.gamma),
    )

    shaped_model = ShapedCostModel(components["cost_model"], phi)
    shaped_ll, shaped_c, shaped_v = _run_log_likelihood(
        cost_model=shaped_model,
        flight_data=flight_data,
        thinned_transitions=thinned_transitions,
        avg_tailwind_knots=avg_tailwind_knots,
        components=components,
        gamma=float(config.gamma),
    )

    delta = shaped_ll - base_ll

    if args.dump_links:
        _dump_route_link_breakdown(
            flight_data=flight_data,
            thinned_transitions=thinned_transitions,
            avg_tailwind_knots=torch.as_tensor(avg_tailwind_knots, device=components["device"]),
            components=components,
            cost_model_base=components["cost_model"],
            phi=phi,
        )

    print("Shaping invariance litmus")
    print(f"  flight_id: {flight_data['flight_id']}")
    print(f"  takeoff_timestamp: {flight_data['takeoff_timestamp']}")
    print(f"  phi_scale: {args.phi_scale}")
    print(f"  base_ll: {base_ll:.6f}")
    print(f"  shaped_ll: {shaped_ll:.6f}")
    print(f"  delta: {delta:.6f}")
    print(f"  base_c_xi: {base_c:.6f}")
    print(f"  shaped_c_xi: {shaped_c:.6f}")
    print(f"  base_V: {base_v:.6f}")
    print(f"  shaped_V: {shaped_v:.6f}")


if __name__ == "__main__":
    main()

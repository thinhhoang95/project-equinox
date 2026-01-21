import math
import networkx as nx
from collections import defaultdict
from typing import Optional

from equinox.dp.trespass.transition_utils import BASE_TRANSITION_LEN, parse_transition

# closures: [(3, 56, 0, 32653, 2, 54, 60, 0, 0, 2)...]
# (u_idx, k_u, rho_u, alt_u, phase_u, v_idx, k_v, rho_v, alt_v, phase_v)

def infer_max_rho_from_closures(
    closures: list[tuple],
) -> int:
    """
    Infer the maximum rho index present in a list of closure/transition tuples.

    The project convention is:
    - rho is the "remaining climb time bin index"
    - closures are 10-tuples:
      (u_idx, k_u, rho_u, alt_u, phase_u, v_idx, k_v, rho_v, alt_v, phase_v)
      Optional fields can append (eta_u_abs_s, eta_v_abs_s).

    This helper implements "Option A": derive max_rho from the closures themselves,
    using both rho_u (index 2) and rho_v (index 7).
    """
    if not closures:
        raise ValueError("Cannot infer max_rho from an empty closures list.")

    max_rho = -1
    for c in closures:
        # Be defensive: we only need indices 2 and 7.
        if len(c) <= 7:
            raise ValueError(
                "Closure tuple is too short to contain rho_u/rho_v at indices 2 and 7. "
                f"Expected 10-tuple, got length={len(c)} value={c!r}"
            )
        try:
            rho_u = int(c[2])
            rho_v = int(c[7])
        except Exception as exc:
            raise ValueError(f"Failed to parse rho indices from closure tuple: {c!r}") from exc
        max_rho = max(max_rho, rho_u, rho_v)

    if max_rho < 0:
        raise ValueError("Inferred max_rho < 0; closures list appears invalid.")
    return max_rho


def thin_closures(
    source_node_idx: int,
    goal_node_idx: int,
    max_rho: Optional[int],
    G: nx.DiGraph,
    closures: list[tuple],
    wallclock_time_bin_k_tolerance_s: Optional[float] = None,
    delta_t_seconds_wall_clock: Optional[float] = None,
    include_wait_edges_in_output: bool = False,
):
    """
    Prunes the closures list to keep only transitions that are part of a valid path
    from a source configuration to a goal configuration.

    A state is defined as (waypoint_idx, k_std_idx, rho_std_idx, alt_std, phase_std).
    closures are tuples: (u_idx, k_u, rho_u, alt_u, phase_u, v_idx, k_v, rho_v, alt_v, phase_v)
    Optionally, closures may append (eta_u_abs_s, eta_v_abs_s) as fields 11 and 12.

    wallclock_time_bin_k_tolerance_s enables virtual wait edges between same-waypoint states
    with small absolute-time mismatches; delta_t_seconds_wall_clock is used to approximate
    eta when closures do not include absolute times. Set include_wait_edges_in_output to
    True to emit those wait edges into the returned closures list.
    """
    if not closures:
        return []

    if max_rho is None:
        max_rho = infer_max_rho_from_closures(closures)

    if wallclock_time_bin_k_tolerance_s is None:
        if delta_t_seconds_wall_clock is not None:
            wallclock_time_bin_k_tolerance_s = float(delta_t_seconds_wall_clock)
        else:
            wallclock_time_bin_k_tolerance_s = 0.0

    if delta_t_seconds_wall_clock is None and wallclock_time_bin_k_tolerance_s is not None:
        delta_t_seconds_wall_clock = float(wallclock_time_bin_k_tolerance_s)

    use_wait_edges = (
        wallclock_time_bin_k_tolerance_s is not None
        and wallclock_time_bin_k_tolerance_s > 0.0
    )

    # 1. Build graph from closures
    # Nodes are states: (waypoint_idx, k_idx, rho_idx, altitude, phase_idx)
    # Edges represent transitions in closures.
    graph = nx.DiGraph()
    all_states_in_closures = set()
    state_eta = {}
    wait_edges = []

    def _sanitize_eta(eta_val: Optional[float]) -> Optional[float]:
        if eta_val is None:
            return None
        try:
            eta_float = float(eta_val)
        except (TypeError, ValueError):
            return None
        if math.isnan(eta_float) or math.isinf(eta_float):
            return None
        return eta_float

    def _resolve_eta(eta_val: Optional[float], k_idx: int) -> Optional[float]:
        eta_clean = _sanitize_eta(eta_val)
        if eta_clean is not None:
            return eta_clean
        if delta_t_seconds_wall_clock is None:
            return None
        return float(k_idx) * float(delta_t_seconds_wall_clock)

    # Assuming closure tuple structure from markdown:
    # c[0]=u_idx, c[1]=k_u, c[2]=rho_u, c[3]=alt_u, c[4]=phase_u
    # c[5]=v_idx, c[6]=k_v, c[7]=rho_v, c[8]=alt_v, c[9]=phase_v
    for c in closures:
        base, eta_u_abs_s, eta_v_abs_s = parse_transition(c)
        u_state = (base[0], base[1], base[2], float(base[3]), base[4])
        v_state = (base[5], base[6], base[7], float(base[8]), base[9])

        graph.add_edge(u_state, v_state)
        all_states_in_closures.add(u_state)
        all_states_in_closures.add(v_state)

        if use_wait_edges:
            eta_u_resolved = _resolve_eta(eta_u_abs_s, base[1])
            if eta_u_resolved is not None:
                prev_eta = state_eta.get(u_state)
                if prev_eta is None or eta_u_resolved < prev_eta:
                    state_eta[u_state] = eta_u_resolved

            eta_v_resolved = _resolve_eta(eta_v_abs_s, base[6])
            if eta_v_resolved is not None:
                prev_eta = state_eta.get(v_state)
                if prev_eta is None or eta_v_resolved < prev_eta:
                    state_eta[v_state] = eta_v_resolved

    # Add virtual "wait" edges for small wall-clock bin mismatches.
    if use_wait_edges and state_eta:
        tolerance_s = float(wallclock_time_bin_k_tolerance_s)
        grouped_states = defaultdict(list)
        for state, eta_val in state_eta.items():
            group_key = (state[0], state[2], state[3], state[4])
            grouped_states[group_key].append((eta_val, state))

        for group_states in grouped_states.values():
            if len(group_states) < 2:
                continue
            group_states.sort(key=lambda x: x[0])
            for i, (eta_i, state_i) in enumerate(group_states):
                for j in range(i + 1, len(group_states)):
                    eta_j, state_j = group_states[j]
                    if eta_j - eta_i > tolerance_s:
                        break
                    graph.add_edge(state_i, state_j)
                    wait_edges.append((state_i, state_j))

    # 2. Identify valid origin states and potential goal states
    # Valid origin state: at source_node_idx, rho = max_rho
    # Potential goal state: at goal_node_idx (any k, rho, alt, phase)
    
    origin_nodes = set()
    goal_nodes = set()

    for state in all_states_in_closures:
        # state: (idx, k, rho, alt, phase)
        waypoint_idx, _, rho_idx, _, _ = state
        if waypoint_idx == source_node_idx and rho_idx == max_rho:
            if graph.has_node(state): # Ensure it's part of the graph built from edges
                 origin_nodes.add(state)
        
        if waypoint_idx == goal_node_idx:
            if graph.has_node(state): # Ensure it's part of the graph built from edges
                 goal_nodes.add(state)

    if not origin_nodes or not goal_nodes:
        return [] # No possible path if no origins or no goals

    # 3. Find all states reachable from any valid origin state
    reachable_from_origins = set()
    for start_node in origin_nodes:
        reachable_from_origins.add(start_node) # Add the origin node itself
        # nx.descendants returns nodes reachable FROM start_node, EXCLUDING start_node
        reachable_from_origins.update(nx.descendants(graph, start_node))

    if not reachable_from_origins:
        return []

    # 4. Find all states that can reach any potential goal state (backward reachability)
    # Build a reversed view of the graph for this
    reversed_graph = nx.reverse_view(graph)
    can_reach_goals = set()
    for end_node in goal_nodes:
        can_reach_goals.add(end_node) # Add the goal node itself
        # Descendants in reversed_graph are predecessors in the original graph
        can_reach_goals.update(nx.descendants(reversed_graph, end_node))
    
    if not can_reach_goals:
        return []

    # 5. Valid states are the intersection
    valid_states = reachable_from_origins.intersection(can_reach_goals)

    if not valid_states:
        return []

    # 6. Filter original closures
    thinned_closures = []
    for c in closures:
        base, _, _ = parse_transition(c)
        u_state = (base[0], base[1], base[2], float(base[3]), base[4])
        v_state = (base[5], base[6], base[7], float(base[8]), base[9])
        if u_state in valid_states and v_state in valid_states:
            thinned_closures.append(c)

    if include_wait_edges_in_output and use_wait_edges and wait_edges:
        output_len = max(len(c) for c in closures) if closures else BASE_TRANSITION_LEN
        base_closure_set = {parse_transition(c)[0] for c in thinned_closures}
        for u_state, v_state in wait_edges:
            if u_state not in valid_states or v_state not in valid_states:
                continue
            base = (
                u_state[0], u_state[1], u_state[2], float(u_state[3]), u_state[4],
                v_state[0], v_state[1], v_state[2], float(v_state[3]), v_state[4],
            )
            if base in base_closure_set:
                continue
            if output_len >= BASE_TRANSITION_LEN + 2:
                eta_u = state_eta.get(u_state)
                eta_v = state_eta.get(v_state)
                thinned_closures.append(base + (eta_u, eta_v))
            else:
                thinned_closures.append(base)
            base_closure_set.add(base)
            
    return thinned_closures

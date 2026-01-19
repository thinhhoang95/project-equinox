import networkx as nx
from typing import Optional

# closures: [(3, 56, 0, 32653, 2, 54, 60, 0, 0, 2)...]
# (u_idx, k_u, rho_u, alt_u, phase_u, v_idx, k_v, rho_v, alt_v, phase_v)

def infer_max_rho_from_closures(
    closures: list[tuple[int, int, int, float, int, int, int, int, float, int]],
) -> int:
    """
    Infer the maximum rho index present in a list of closure/transition tuples.

    The project convention is:
    - rho is the "remaining climb time bin index"
    - closures are 10-tuples:
      (u_idx, k_u, rho_u, alt_u, phase_u, v_idx, k_v, rho_v, alt_v, phase_v)

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
    closures: list[tuple[int, int, int, float, int, int, int, int, float, int]],
):
    """
    Prunes the closures list to keep only transitions that are part of a valid path
    from a source configuration to a goal configuration.

    A state is defined as (waypoint_idx, k_std_idx, rho_std_idx, alt_std, phase_std).
    closures are tuples: (u_idx, k_u, rho_u, alt_u, phase_u, v_idx, k_v, rho_v, alt_v, phase_v)
    """
    if not closures:
        return []

    if max_rho is None:
        max_rho = infer_max_rho_from_closures(closures)

    # 1. Build graph from closures
    # Nodes are states: (waypoint_idx, k_idx, rho_idx, altitude, phase_idx)
    # Edges represent transitions in closures.
    graph = nx.DiGraph()
    all_states_in_closures = set()

    # Assuming closure tuple structure from markdown:
    # c[0]=u_idx, c[1]=k_u, c[2]=rho_u, c[3]=alt_u, c[4]=phase_u
    # c[5]=v_idx, c[6]=k_v, c[7]=rho_v, c[8]=alt_v, c[9]=phase_v
    for c in closures:
        # Ensure closure has 10 elements before unpacking, if there's uncertainty.
        # For now, assuming all closures adhere to the 10-tuple structure.
        u_state = (c[0], c[1], c[2], float(c[3]), c[4])
        v_state = (c[5], c[6], c[7], float(c[8]), c[9])
        
        graph.add_edge(u_state, v_state)
        all_states_in_closures.add(u_state)
        all_states_in_closures.add(v_state)

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
        u_state = (c[0], c[1], c[2], float(c[3]), c[4])
        v_state = (c[5], c[6], c[7], float(c[8]), c[9])
        if u_state in valid_states and v_state in valid_states:
            thinned_closures.append(c)
            
    return thinned_closures
"""
Thinning module for pruning closure/transition tuples to valid paths.

This module implements state space thinning for dynamic programming transitions in the
TRESPASS (Trajectory REasoning with State Space Pruning and Sampling) system. It prunes
transitions (closures) to keep only those that are part of valid paths from a source
configuration to a goal configuration, optionally merging adjacent time states within
a tolerance window.

Key Concepts:
-------------
- **State**: A 5-tuple (waypoint_idx, k_idx, rho_idx, altitude, phase_idx) representing
  a flight configuration at a specific waypoint, time bin, remaining climb time bin,
  altitude, and flight phase.
  
- **Closure/Transition**: A tuple representing a state transition. The base format is a
  10-tuple: (u_idx, k_u, rho_u, alt_u, phase_u, v_idx, k_v, rho_v, alt_v, phase_v),
  where u and v are the source and destination states. Optionally, closures may include
  absolute ETA times as fields 11 and 12: (..., eta_u_abs_s, eta_v_abs_s).

- **Thinning**: The process of filtering closures to keep only transitions that are part
  of valid paths from origin states (at source_node_idx with rho=max_rho) to goal states
  (at goal_node_idx).

- **Morphing**: The process of merging adjacent k (time bin) states at the same waypoint,
  rho, altitude, and phase when their ETA ranges are within a tolerance window. This
  reduces state space size by canonicalizing time bins.

Main Function:
-------------
`thin_closures()`: Prunes closures to valid paths, optionally with k-state morphing.

Usage Examples:
--------------
Basic usage without morphing:

    >>> import networkx as nx
    >>> from equinox.dp.trespass.thinning import thin_closures
    >>> 
    >>> G = nx.DiGraph()  # Graph structure (used for validation)
    >>> closures = [
    ...     (0, 0, 2, 0, 0, 1, 1, 1, 100, 1),  # Transition from waypoint 0 to 1
    ...     (1, 1, 1, 100, 1, 2, 2, 0, 200, 1),  # Transition from waypoint 1 to 2
    ...     (0, 0, 2, 0, 0, 3, 1, 1, 100, 1),  # Dead-end transition
    ... ]
    >>> 
    >>> thinned = thin_closures(
    ...     source_node_idx=0,
    ...     goal_node_idx=2,
    ...     max_rho=2,
    ...     G=G,
    ...     closures=closures,
    ...     wallclock_time_bin_k_tolerance_s=0.0,  # No morphing
    ... )
    >>> # Returns only closures on valid paths: [(0,0,2,0,0,1,1,1,100,1), (1,1,1,100,1,2,2,0,200,1)]

With morphing to merge adjacent k states:

    >>> closures_with_eta = [
    ...     (0, 0, 1, 0, 0, 1, 1, 0, 100, 1, 0.0, 600.0),
    ...     (0, 0, 1, 0, 0, 1, 2, 0, 100, 1, 0.0, 610.0),  # k=2 within tolerance of k=1
    ...     (1, 2, 0, 100, 1, 2, 3, 0, 100, 1, 610.0, 1200.0),
    ... ]
    >>> 
    >>> thinned = thin_closures(
    ...     source_node_idx=0,
    ...     goal_node_idx=2,
    ...     max_rho=1,
    ...     G=G,
    ...     closures=closures_with_eta,
    ...     wallclock_time_bin_k_tolerance_s=20.0,  # 20 second tolerance
    ...     delta_t_seconds_wall_clock=600.0,
    ... )
    >>> # k=2 states are morphed to k=1, reducing state space

Input/Output Format:
-------------------
Input:
    - closures: List of transition tuples, each with at least 10 fields:
      (u_idx, k_u, rho_u, alt_u, phase_u, v_idx, k_v, rho_v, alt_v, phase_v)
      Optional fields 11-12: (eta_u_abs_s, eta_v_abs_s)
    
    - source_node_idx: Waypoint index where valid paths start (must have rho=max_rho)
    - goal_node_idx: Waypoint index where valid paths end
    - max_rho: Maximum remaining climb time bin index (None to infer from closures)
    - G: NetworkX DiGraph (used for validation, typically the route graph)
    - wallclock_time_bin_k_tolerance_s: Tolerance for merging adjacent k states (seconds)
    - delta_t_seconds_wall_clock: Time bin duration for ETA approximation

Output:
    - List of closure tuples (same format as input) that are part of valid paths.
      If morphing is enabled, k indices may be canonicalized to earlier values.

Example Closure Tuple Structure:
--------------------------------
    # Base 10-tuple:
    closure = (3, 56, 0, 32653, 2, 54, 60, 0, 0, 2)
    #      u: (3, 56, 0, 32653, 2) -> v: (54, 60, 0, 0, 2)
    #      waypoint 3, k=56, rho=0, alt=32653, phase=2
    #      -> waypoint 54, k=60, rho=0, alt=0, phase=2
    
    # With ETA times (12-tuple):
    closure_with_eta = (3, 56, 0, 32653, 2, 54, 60, 0, 0, 2, 33600.0, 36000.0)
    #      eta_u_abs_s=33600.0, eta_v_abs_s=36000.0

Helper Functions:
---------------
- `infer_max_rho_from_closures()`: Extracts maximum rho index from closure list
- Internal functions handle ETA sanitization, state grouping, and graph reachability
"""

import math
import networkx as nx
from collections import defaultdict
from typing import Optional

from equinox.dp.trespass.transition_utils import BASE_TRANSITION_LEN, parse_transition

# closures: [(3, 56, 0, 32653, 2, 54, 60, 0, 0, 2)...]
# (u_idx, k_u, rho_u, alt_u, phase_u, v_idx, k_v, rho_v, alt_v, phase_v)


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


def _resolve_eta(
    eta_val: Optional[float],
    k_idx: int,
    delta_t_seconds_wall_clock: Optional[float],
) -> Optional[float]:
    eta_clean = _sanitize_eta(eta_val)
    if eta_clean is not None:
        return eta_clean
    if delta_t_seconds_wall_clock is None:
        return None
    return float(k_idx) * float(delta_t_seconds_wall_clock)


def _update_eta_bounds(
    state_eta_bounds: dict[tuple, tuple[float, float]],
    state: tuple,
    eta_val: float,
) -> None:
    prev = state_eta_bounds.get(state)
    if prev is None:
        state_eta_bounds[state] = (eta_val, eta_val)
        return
    eta_min, eta_max = prev
    state_eta_bounds[state] = (min(eta_min, eta_val), max(eta_max, eta_val))


def _compute_state_eta_bounds(
    closures: list[tuple],
    delta_t_seconds_wall_clock: Optional[float],
) -> dict[tuple, tuple[float, float]]:
    state_eta_bounds: dict[tuple, tuple[float, float]] = {}
    for c in closures:
        base, eta_u_abs_s, eta_v_abs_s = parse_transition(c)
        u_state = (base[0], base[1], base[2], float(base[3]), base[4])
        v_state = (base[5], base[6], base[7], float(base[8]), base[9])

        eta_u_resolved = _resolve_eta(eta_u_abs_s, base[1], delta_t_seconds_wall_clock)
        if eta_u_resolved is not None:
            _update_eta_bounds(state_eta_bounds, u_state, eta_u_resolved)

        eta_v_resolved = _resolve_eta(eta_v_abs_s, base[6], delta_t_seconds_wall_clock)
        if eta_v_resolved is not None:
            _update_eta_bounds(state_eta_bounds, v_state, eta_v_resolved)
    return state_eta_bounds


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[tuple, tuple] = {}
        self.rank: dict[tuple, int] = {}

    def add(self, item: tuple) -> None:
        if item in self.parent:
            return
        self.parent[item] = item
        self.rank[item] = 0

    def find(self, item: tuple) -> tuple:
        parent = self.parent.get(item)
        if parent is None:
            self.add(item)
            return item
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, a: tuple, b: tuple) -> None:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return
        rank_a = self.rank[root_a]
        rank_b = self.rank[root_b]
        if rank_a < rank_b:
            self.parent[root_a] = root_b
        elif rank_b < rank_a:
            self.parent[root_b] = root_a
        else:
            self.parent[root_b] = root_a
            self.rank[root_a] += 1


def _gap_between_eta_bounds(a: tuple[float, float], b: tuple[float, float]) -> float:
    eta_min_a, eta_max_a = a
    eta_min_b, eta_max_b = b
    return max(0.0, eta_min_b - eta_max_a, eta_min_a - eta_max_b)


def _build_canonical_k_map(
    state_eta_bounds: dict[tuple, tuple[float, float]],
    tolerance_s: float,
) -> dict[tuple, int]:
    if tolerance_s <= 0.0 or not state_eta_bounds:
        return {}

    grouped_states: dict[tuple, dict[int, tuple]] = defaultdict(dict)
    for state in state_eta_bounds:
        group_key = (state[0], state[2], state[3], state[4])
        grouped_states[group_key][state[1]] = state

    uf = _UnionFind()
    for group_states in grouped_states.values():
        if len(group_states) < 2:
            continue
        for state in group_states.values():
            uf.add(state)
        for k_idx in sorted(group_states):
            state = group_states[k_idx]
            state_next = group_states.get(k_idx + 1)
            if state_next is None:
                continue
            gap = _gap_between_eta_bounds(
                state_eta_bounds[state],
                state_eta_bounds[state_next],
            )
            if gap <= tolerance_s:
                uf.union(state, state_next)

    if not uf.parent:
        return {}

    min_k_by_root: dict[tuple, int] = {}
    for state in uf.parent:
        root = uf.find(state)
        min_k_by_root[root] = min(min_k_by_root.get(root, state[1]), state[1])

    state_to_canon_k: dict[tuple, int] = {}
    for state in uf.parent:
        state_to_canon_k[state] = min_k_by_root[uf.find(state)]
    return state_to_canon_k


def _rewrite_transition_base(transition: tuple, new_base: tuple) -> tuple:
    if len(transition) <= BASE_TRANSITION_LEN:
        return tuple(new_base)
    return tuple(new_base) + tuple(transition[BASE_TRANSITION_LEN:])


def _morph_closures_by_k(
    closures: list[tuple],
    state_to_canon_k: dict[tuple, int],
) -> list[tuple]:
    if not state_to_canon_k:
        return list(closures)

    seen_bases = set()
    morphed = []
    for c in closures:
        base, _, _ = parse_transition(c)
        u_state = (base[0], base[1], base[2], float(base[3]), base[4])
        v_state = (base[5], base[6], base[7], float(base[8]), base[9])
        k_u_canon = state_to_canon_k.get(u_state, base[1])
        k_v_canon = state_to_canon_k.get(v_state, base[6])

        if k_u_canon != base[1] or k_v_canon != base[6]:
            new_base = (
                base[0],
                k_u_canon,
                base[2],
                base[3],
                base[4],
                base[5],
                k_v_canon,
                base[7],
                base[8],
                base[9],
            )
            new_transition = _rewrite_transition_base(tuple(c), new_base)
        else:
            new_transition = tuple(c)

        base_key = tuple(new_transition[:BASE_TRANSITION_LEN])
        if base_key in seen_bases:
            continue
        seen_bases.add(base_key)
        morphed.append(new_transition)

    return morphed

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

    wallclock_time_bin_k_tolerance_s controls morphing: adjacent k states at the same
    waypoint/rho/alt/phase are merged onto the earlier k when their ETA ranges are
    within tolerance. delta_t_seconds_wall_clock is used to approximate ETA when
    closures do not include absolute times. include_wait_edges_in_output is ignored.
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

    morph_tolerance_s = float(wallclock_time_bin_k_tolerance_s or 0.0)
    if morph_tolerance_s > 0.0:
        state_eta_bounds = _compute_state_eta_bounds(
            closures,
            delta_t_seconds_wall_clock,
        )
        state_to_canon_k = _build_canonical_k_map(state_eta_bounds, morph_tolerance_s)
        closures = _morph_closures_by_k(closures, state_to_canon_k)

    # 1. Build graph from morphed closures
    # Nodes are states: (waypoint_idx, k_idx, rho_idx, altitude, phase_idx)
    # Edges represent transitions in closures.
    graph = nx.DiGraph()
    all_states_in_closures = set()

    # Assuming closure tuple structure from markdown:
    # c[0]=u_idx, c[1]=k_u, c[2]=rho_u, c[3]=alt_u, c[4]=phase_u
    # c[5]=v_idx, c[6]=k_v, c[7]=rho_v, c[8]=alt_v, c[9]=phase_v
    for c in closures:
        base, _, _ = parse_transition(c)
        u_state = (base[0], base[1], base[2], float(base[3]), base[4])
        v_state = (base[5], base[6], base[7], float(base[8]), base[9])

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

    # 6. Filter morphed closures
    thinned_closures = []
    for c in closures:
        base, _, _ = parse_transition(c)
        u_state = (base[0], base[1], base[2], float(base[3]), base[4])
        v_state = (base[5], base[6], base[7], float(base[8]), base[9])
        if u_state in valid_states and v_state in valid_states:
            thinned_closures.append(c)
            
    return thinned_closures

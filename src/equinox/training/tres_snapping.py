"""
Route Snapping to Feasible Graph

This module provides utilities for snapping flight routes to a feasible graph after
thinning operations in the TRES (Trespass) batch processing pipeline. The main purpose
is to map original route strings to valid graph edges that have been identified as
feasible through dynamic programming passes and thinning operations.

The workflow typically involves:
1. Extracting feasible waypoint transitions from thinned transitions
2. Creating a subgraph containing only these feasible edges
3. Snapping the original route string to this feasible graph using Viterbi matching

Usage Example:
-------------
```python
from equinox.training.tres_snapping import (
    extract_feasible_waypoint_transitions,
    create_feasible_graph,
    snap_route_to_feasible_graph
)
import networkx as nx

# Assume you have thinned_transitions from a TRES pass
# thinned_transitions is a list of tuples: (u_idx, ..., v_idx, ...)
thinned_transitions = [
    (0, 1.5, 100, 200, 300, 5, ...),  # u_idx=0, v_idx=5
    (5, 2.0, 150, 250, 350, 10, ...), # u_idx=5, v_idx=10
    (10, 1.8, 200, 300, 400, 15, ...) # u_idx=10, v_idx=15
]

# Step 1: Extract feasible waypoint transitions
waypoint_transitions = extract_feasible_waypoint_transitions(thinned_transitions)
# Output: {(0, 5), (5, 10), (10, 15)}

# Step 2: Create feasible graph from original graph
original_graph = nx.DiGraph()  # Your full airspace graph
# ... populate original_graph with nodes and edges ...
idx_to_node = {0: "WAYPOINT_A", 5: "WAYPOINT_B", 10: "WAYPOINT_C", 15: "WAYPOINT_D"}

feasible_graph = create_feasible_graph(original_graph, waypoint_transitions, idx_to_node)
# Output: NetworkX DiGraph containing only edges (WAYPOINT_A->WAYPOINT_B, etc.)

# Step 3: Snap original route to feasible graph
original_route_str = "WAYPOINT_A WAYPOINT_X WAYPOINT_B WAYPOINT_Y WAYPOINT_C"
snapped_route_str = snap_route_to_feasible_graph(
    original_route_str, 
    feasible_graph, 
    original_graph
)
# Output: "WAYPOINT_A WAYPOINT_B WAYPOINT_C" (snapped to feasible path)
```

Input/Output Examples:
---------------------

extract_feasible_waypoint_transitions:
    Input:
        thinned_transitions: List of transition tuples where:
            - transition[0] is the source waypoint index (u_idx)
            - transition[5] is the destination waypoint index (v_idx)
        Example: [(0, 1.5, 100, 200, 300, 5, ...), (5, 2.0, 150, 250, 350, 10, ...)]
    
    Output:
        Set of (u_idx, v_idx) tuples representing unique waypoint transitions
        Example: {(0, 5), (5, 10)}

create_feasible_graph:
    Input:
        original_graph: NetworkX DiGraph with full airspace structure
            - Nodes have attributes: "lat", "lon"
            - Edges may have attributes: "length_nm", etc.
        waypoint_transitions: Set of (u_idx, v_idx) tuples from step 1
        idx_to_node: Dictionary mapping waypoint indices to node names
            Example: {0: "KJFK", 5: "KORD", 10: "KLAX"}
    
    Output:
        NetworkX DiGraph containing only the feasible edges
            - Same node structure as original_graph
            - Only edges corresponding to waypoint_transitions are included

snap_route_to_feasible_graph:
    Input:
        original_route_str: Space-separated string of waypoint names
            Example: "KJFK KORD KLAX"
        feasible_graph: NetworkX DiGraph from create_feasible_graph
        original_graph: Full NetworkX DiGraph (for node coordinate lookup)
    
    Output:
        Space-separated string of waypoint names snapped to feasible path
            Example: "KJFK KORD KLAX" (if all waypoints are feasible)
            Raises if:
                - original_route_str is empty or invalid
                - Not enough waypoints found (< 2)
                - Feasible graph has no edges
                - Viterbi matching fails
"""

import logging
from bisect import bisect_right
from collections import defaultdict, deque

import networkx as nx

from equinox.training.prep.resculpt_viterbi import viterbi_match, haversine_nm


def extract_feasible_waypoint_transitions(thinned_transitions):
    """
    Extract unique waypoint transitions (u_idx, v_idx) from thinned transitions.
    """
    waypoint_transitions = set()
    for transition in thinned_transitions:
        u_idx = transition[0]
        v_idx = transition[5]
        waypoint_transitions.add((u_idx, v_idx))
    return waypoint_transitions


def create_feasible_graph(original_graph, waypoint_transitions, idx_to_node):
    """
    Create a subgraph containing only feasible edges based on waypoint transitions.
    """
    feasible_graph = nx.DiGraph()

    for node, data in original_graph.nodes(data=True):
        feasible_graph.add_node(node, **data)

    for u_idx, v_idx in waypoint_transitions:
        u_node = idx_to_node.get(u_idx)
        v_node = idx_to_node.get(v_idx)

        if u_node is not None and v_node is not None and original_graph.has_edge(u_node, v_node):
            edge_data = original_graph.edges[u_node, v_node]
            feasible_graph.add_edge(u_node, v_node, **edge_data)

    return feasible_graph


def build_state_adjacency(thinned_transitions):
    """
    Build adjacency structures for state-level continuity checks and repairs.
    """
    edge_adj = defaultdict(list)
    state_adj = defaultdict(list)
    states_by_waypoint = defaultdict(set)

    for transition in thinned_transitions:
        if len(transition) < 10:
            continue
        (
            u_idx,
            k_u,
            rho_u,
            _alt_u,
            ph_u,
            v_idx,
            k_v,
            rho_v,
            _alt_v,
            ph_v,
        ) = transition[:10]
        u_state = (u_idx, k_u, rho_u, ph_u)
        v_state = (v_idx, k_v, rho_v, ph_v)
        edge_adj[(u_idx, v_idx)].append((u_state, v_state))
        state_adj[u_state].append(v_state)
        states_by_waypoint[u_idx].add(u_state)
        states_by_waypoint[v_idx].add(v_state)

    return edge_adj, state_adj, states_by_waypoint


def _index_current_states(current_states):
    index = defaultdict(list)
    for u_idx, k_idx, rho, phase in current_states:
        index[(u_idx, rho, phase)].append(k_idx)
    for k_list in index.values():
        k_list.sort()
    return index


def _pick_k_forward(k_list, k_req, k_tolerance_bins):
    low = k_req - k_tolerance_bins
    idx = bisect_right(k_list, k_req)
    if idx == 0:
        return None
    candidate = k_list[idx - 1]
    if candidate < low:
        return None
    return candidate


def _pick_k_backward(k_list, k_req, k_tolerance_bins):
    high = k_req + k_tolerance_bins
    idx = bisect_right(k_list, k_req)
    if idx >= len(k_list):
        return None
    candidate = k_list[idx]
    if candidate > high:
        return None
    return candidate


def _match_next_states_with_backpointer(opts, current_states, k_tolerance_bins):
    if not current_states:
        return {}, "none"

    exact = {}
    for u_state, v_state in opts:
        if u_state in current_states and v_state not in exact:
            exact[v_state] = u_state
    if exact:
        return exact, "exact"
    if k_tolerance_bins <= 0:
        return {}, "none"

    index = _index_current_states(current_states)
    forward = {}
    backward = {}
    for u_state, v_state in opts:
        u_idx, k_req, rho, phase = u_state
        k_list = index.get((u_idx, rho, phase))
        if not k_list:
            continue
        k_match = _pick_k_forward(k_list, k_req, k_tolerance_bins)
        if k_match is not None:
            if v_state not in forward:
                forward[v_state] = (u_idx, k_match, rho, phase)
            continue
        k_match = _pick_k_backward(k_list, k_req, k_tolerance_bins)
        if k_match is not None:
            if v_state not in backward:
                backward[v_state] = (u_idx, k_match, rho, phase)

    if forward:
        return forward, "forward"
    if backward:
        return backward, "backward"
    return {}, "none"


def realize_state_chain_for_route(
    route_nodes,
    node_to_idx,
    edge_adj,
    *,
    k_tolerance_bins=0,
    states_by_waypoint=None,
):
    """
    Attempt to realize a waypoint route as a continuous state chain.
    """
    if len(route_nodes) < 2:
        return None, [], {
            "reason": "route_too_short",
            "index": None,
            "current_states": set(),
        }

    for node in route_nodes:
        if node not in node_to_idx:
            return None, [], {
                "reason": f"unknown_node:{node}",
                "index": None,
                "current_states": set(),
            }

    backpointers = []
    match_kinds = []

    u_name = route_nodes[0]
    v_name = route_nodes[1]
    u_idx = node_to_idx[u_name]
    v_idx = node_to_idx[v_name]
    opts = edge_adj.get((u_idx, v_idx), [])
    if not opts:
        current_states = set()
        if states_by_waypoint is not None:
            current_states = states_by_waypoint.get(u_idx, set())
        return None, match_kinds, {
            "reason": f"missing_edge:{u_name}->{v_name}",
            "index": 0,
            "current_states": current_states,
        }

    step_mapping = {}
    for u_state, v_state in opts:
        if v_state not in step_mapping:
            step_mapping[v_state] = u_state
    backpointers.append(step_mapping)
    match_kinds.append("init")
    current_states = set(step_mapping.keys())

    for i in range(1, len(route_nodes) - 1):
        u_name = route_nodes[i]
        v_name = route_nodes[i + 1]
        u_idx = node_to_idx[u_name]
        v_idx = node_to_idx[v_name]
        opts = edge_adj.get((u_idx, v_idx), [])
        if not opts:
            return None, match_kinds, {
                "reason": f"missing_edge:{u_name}->{v_name}",
                "index": i,
                "current_states": current_states,
            }
        step_mapping, match_kind = _match_next_states_with_backpointer(
            opts, current_states, k_tolerance_bins
        )
        if not step_mapping:
            return None, match_kinds, {
                "reason": f"no_chain:{u_name}->{v_name}",
                "index": i,
                "current_states": current_states,
            }
        backpointers.append(step_mapping)
        match_kinds.append(match_kind)
        current_states = set(step_mapping.keys())

    if not current_states:
        return None, match_kinds, {
            "reason": "no_terminal_state",
            "index": len(route_nodes) - 2,
            "current_states": current_states,
        }

    end_state = next(iter(current_states))
    state_chain = [end_state]
    for step_idx in reversed(range(len(backpointers))):
        prev_state = backpointers[step_idx][state_chain[-1]]
        state_chain.append(prev_state)
    state_chain.reverse()
    return state_chain, match_kinds, None


def _reconstruct_state_path(predecessor_map, end_state):
    path = [end_state]
    while predecessor_map[path[-1]] is not None:
        path.append(predecessor_map[path[-1]])
    path.reverse()
    return path


def find_state_path_to_waypoint(
    state_adj,
    start_states,
    target_waypoint_idx,
    *,
    max_hops=None,
    max_nodes=None,
):
    if not start_states:
        return None

    queue = deque(start_states)
    predecessor_map = {state: None for state in start_states}
    depth = {state: 0 for state in start_states}

    if any(state[0] == target_waypoint_idx for state in start_states):
        for state in start_states:
            if state[0] == target_waypoint_idx:
                return [state]

    while queue:
        current = queue.popleft()
        current_depth = depth[current]
        if max_hops is not None and current_depth >= max_hops:
            continue
        for nxt in state_adj.get(current, []):
            if nxt in predecessor_map:
                continue
            predecessor_map[nxt] = current
            depth[nxt] = current_depth + 1
            if max_nodes is not None and len(predecessor_map) >= max_nodes:
                return None
            if nxt[0] == target_waypoint_idx:
                return _reconstruct_state_path(predecessor_map, nxt)
            queue.append(nxt)
    return None


def _state_path_to_waypoints(state_path, idx_to_node):
    if not state_path:
        return None
    waypoint_indices = [state[0] for state in state_path]
    condensed = [waypoint_indices[0]]
    for idx in waypoint_indices[1:]:
        if idx != condensed[-1]:
            condensed.append(idx)
    try:
        return [idx_to_node[idx] for idx in condensed]
    except KeyError:
        return None


def _realize_route_with_state_repair(
    route_nodes,
    node_to_idx,
    edge_adj,
    state_adj,
    states_by_waypoint,
    idx_to_node,
    *,
    k_tolerance_bins=0,
    max_state_repair_attempts=3,
    max_state_repair_hops=12,
    max_state_repair_nodes=50000,
):
    repairs = 0
    route_nodes = list(route_nodes)

    while True:
        state_chain, match_kinds, failure = realize_state_chain_for_route(
            route_nodes,
            node_to_idx,
            edge_adj,
            k_tolerance_bins=k_tolerance_bins,
            states_by_waypoint=states_by_waypoint,
        )
        if state_chain:
            return route_nodes, state_chain, match_kinds, repairs

        if not failure or repairs >= max_state_repair_attempts:
            return None

        reason = failure.get("reason", "")
        if reason.startswith("unknown_node") or reason == "route_too_short":
            return None

        fail_idx = failure.get("index")
        if fail_idx is None or fail_idx >= len(route_nodes) - 1:
            return None

        u_name = route_nodes[fail_idx]
        v_name = route_nodes[fail_idx + 1]
        u_idx = node_to_idx.get(u_name)
        v_idx = node_to_idx.get(v_name)
        if u_idx is None or v_idx is None:
            return None

        current_states = failure.get("current_states") or states_by_waypoint.get(
            u_idx, set()
        )
        if not current_states:
            return None

        path_states = find_state_path_to_waypoint(
            state_adj,
            current_states,
            v_idx,
            max_hops=max_state_repair_hops,
            max_nodes=max_state_repair_nodes,
        )
        if not path_states:
            return None

        path_nodes = _state_path_to_waypoints(path_states, idx_to_node)
        if not path_nodes or path_nodes[0] != u_name or path_nodes[-1] != v_name:
            return None
        if len(path_nodes) <= 2:
            return None

        logging.info(
            "Repairing snapped route segment %s->%s with %s hops",
            u_name,
            v_name,
            len(path_nodes) - 1,
        )
        route_nodes = (
            route_nodes[: fail_idx + 1]
            + path_nodes[1:]
            + route_nodes[fail_idx + 2 :]
        )
        repairs += 1


def snap_route_to_feasible_graph(
    original_route_str,
    feasible_graph,
    original_graph,
    *,
    origin=None,
    destination=None,
    thinned_transitions=None,
    node_to_idx=None,
    idx_to_node=None,
    k_tolerance_bins=0,
    allow_state_repair=True,
    max_state_repair_hops=12,
    max_state_repair_nodes=50000,
    max_state_repair_attempts=3,
    return_details=False,
):
    """
    Snap the original route to the feasible graph using viterbi matching.

    When thinned_transitions and node_to_idx are provided, the snapped route is
    post-validated against state transitions and optionally repaired to enforce
    continuity (raises on matching or state-chain failures).
    """
    if not original_route_str or not isinstance(original_route_str, str):
        raise ValueError("original_route_str is empty or invalid.")

    waypoint_names = original_route_str.split()

    obs_pts = []
    for waypoint_name in waypoint_names:
        if waypoint_name in original_graph.nodes:
            node_data = original_graph.nodes[waypoint_name]
            obs_pts.append((node_data["lat"], node_data["lon"]))
        else:
            logging.warning("Waypoint '%s' not found in original graph", waypoint_name)

    if len(obs_pts) < 2:
        raise RuntimeError(f"Not enough waypoints found in graph ({len(obs_pts)}).")

    if feasible_graph.number_of_edges() == 0:
        raise RuntimeError("Feasible graph has no edges.")

    for u, v in feasible_graph.edges():
        if "length_nm" not in feasible_graph.edges[u, v]:
            lat1, lon1 = feasible_graph.nodes[u]["lat"], feasible_graph.nodes[u]["lon"]
            lat2, lon2 = feasible_graph.nodes[v]["lat"], feasible_graph.nodes[v]["lon"]
            feasible_graph.edges[u, v]["length_nm"] = haversine_nm(lat1, lon1, lat2, lon2)

    start_node = origin if origin else None
    end_node = destination if destination else None
    _, best_edges = viterbi_match(
        feasible_graph,
        obs_pts,
        k=50,
        beta=0.5,
        start_node=start_node,
        end_node=end_node,
    )
    if not best_edges:
        raise RuntimeError("No edges found in snapped route.")

    full_nodes = [best_edges[0][0]] + [edge[1] for edge in best_edges]
    snapped_nodes = full_nodes
    state_chain = None
    match_kinds = None
    repairs = 0

    if thinned_transitions is not None and node_to_idx is not None:
        edge_adj, state_adj, states_by_waypoint = build_state_adjacency(
            thinned_transitions
        )
        if idx_to_node is None:
            idx_to_node = {v: k for k, v in node_to_idx.items()}

        if allow_state_repair:
            result = _realize_route_with_state_repair(
                snapped_nodes,
                node_to_idx,
                edge_adj,
                state_adj,
                states_by_waypoint,
                idx_to_node,
                k_tolerance_bins=k_tolerance_bins,
                max_state_repair_attempts=max_state_repair_attempts,
                max_state_repair_hops=max_state_repair_hops,
                max_state_repair_nodes=max_state_repair_nodes,
            )
            if not result:
                raise RuntimeError("State repair failed for snapped route.")
            snapped_nodes, state_chain, match_kinds, repairs = result
        else:
            state_chain, match_kinds, failure = realize_state_chain_for_route(
                snapped_nodes,
                node_to_idx,
                edge_adj,
                k_tolerance_bins=k_tolerance_bins,
                states_by_waypoint=states_by_waypoint,
            )
            if not state_chain:
                reason = failure["reason"] if failure else "unknown"
                raise RuntimeError(
                    f"State realization failed for snapped route: {reason}"
                )
    elif thinned_transitions is not None or node_to_idx is not None:
        logging.warning(
            "State realization skipped (thinned_transitions=%s, node_to_idx=%s)",
            thinned_transitions is not None,
            node_to_idx is not None,
        )

    route_str = " ".join(snapped_nodes)
    if return_details:
        rho_sequence = (
            [state[2] for state in state_chain] if state_chain else None
        )
        return {
            "route": route_str,
            "state_chain": state_chain,
            "rho_sequence": rho_sequence,
            "match_kinds": match_kinds,
            "repairs": repairs,
        }
    return route_str

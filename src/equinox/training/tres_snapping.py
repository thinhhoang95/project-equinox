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
            Returns None if:
                - original_route_str is empty or invalid
                - Not enough waypoints found (< 2)
                - Feasible graph has no edges
                - Viterbi matching fails
"""

import logging

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


def snap_route_to_feasible_graph(original_route_str, feasible_graph, original_graph):
    """
    Snap the original route to the feasible graph using viterbi matching.
    """
    if not original_route_str or not isinstance(original_route_str, str):
        return None

    waypoint_names = original_route_str.split()

    obs_pts = []
    for waypoint_name in waypoint_names:
        if waypoint_name in original_graph.nodes:
            node_data = original_graph.nodes[waypoint_name]
            obs_pts.append((node_data["lat"], node_data["lon"]))
        else:
            logging.warning("Waypoint '%s' not found in original graph", waypoint_name)

    if len(obs_pts) < 2:
        logging.warning("Not enough waypoints found in graph (%s)", len(obs_pts))
        return None

    if feasible_graph.number_of_edges() == 0:
        logging.warning("Feasible graph has no edges")
        return None

    for u, v in feasible_graph.edges():
        if "length_nm" not in feasible_graph.edges[u, v]:
            lat1, lon1 = feasible_graph.nodes[u]["lat"], feasible_graph.nodes[u]["lon"]
            lat2, lon2 = feasible_graph.nodes[v]["lat"], feasible_graph.nodes[v]["lon"]
            feasible_graph.edges[u, v]["length_nm"] = haversine_nm(lat1, lon1, lat2, lon2)

    try:
        _, best_edges = viterbi_match(feasible_graph, obs_pts, k=50, beta=0.5)
        if not best_edges:
            logging.warning("No edges found in snapped route")
            return None

        full_nodes = [best_edges[0][0]] + [edge[1] for edge in best_edges]
        return " ".join(full_nodes)

    except Exception as e:
        logging.error("Viterbi matching failed: %s", e)
        return None

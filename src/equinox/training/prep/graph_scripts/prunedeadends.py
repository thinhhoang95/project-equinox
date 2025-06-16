import networkx as nx
import math

def get_turn_angle(pos1, pos2, pos3):
    """
    Calculate the interior angle (in degrees) formed by the segments
    (pos1 -> pos2) and (pos2 -> pos3) at pos2. Assumes planar geometry.
    Returns an angle between 0 (reverse) and 180 (forward).

    Parameters
    ----------
    pos1, pos2, pos3 : tuple
        Coordinate pairs (e.g., (longitude, latitude) or (x, y)).
    """
    # Vector p1 -> p2
    v1x = pos2[0] - pos1[0]
    v1y = pos2[1] - pos1[1]
    # Vector p2 -> p3
    v2x = pos3[0] - pos2[0]
    v2y = pos3[1] - pos2[1]

    dot = v1x * v2x + v1y * v2y
    norm1 = math.sqrt(v1x**2 + v1y**2)
    norm2 = math.sqrt(v2x**2 + v2y**2)

    if norm1 == 0 or norm2 == 0: return 180.0 # Treat zero-length segments as straight

    cos_theta = max(-1.0, min(1.0, dot / (norm1 * norm2))) # Clamp for precision
    angle_rad = math.acos(cos_theta)
    return math.degrees(angle_rad)

def prune_dead_ends_and_sharp_turns(Gm, source, target, min_turn_angle=30):
    """
    Remove from Gm all nodes (and their edges) that are not on any path
    between `source` and `target` that meets the minimum turn angle requirement.

    Uses a fast iterative approach for DiGraphs when angle checking is enabled.
    Falls back to all_simple_paths for undirected graphs if angle checking is needed.
    Modifies Gm in-place. Assumes nodes have 'lat' and 'lon' attributes.

    Parameters
    ----------
    Gm : networkx.Graph or networkx.DiGraph
        Your route graph. Nodes must have 'lat' and 'lon' attributes.
    source : node-ID
        The origin waypoint.
    target : node-ID
        The destination waypoint.
    min_turn_angle : float, optional
        Minimum interior angle (degrees) for turns. Uses planar approximation
        for lat/lon. Defaults to 30. Value <= 0 disables angle checks.
    """
    if not Gm.has_node(source) or not Gm.has_node(target):
        raise KeyError(f"Either source ({source}) or target ({target}) not in graph")

    initial_node_count = Gm.number_of_nodes() # For checking if anything changed

    if min_turn_angle <= 0:
        # --- No Angle Checks: Use original fast logic ---
        if Gm.is_directed():
            reachable_from_src = set(nx.descendants(Gm, source)) | {source}
            can_reach_tgt    = set(nx.ancestors(Gm, target))    | {target}
            valid_nodes = reachable_from_src & can_reach_tgt
        else: # Undirected Graph
            valid_nodes = set()
            try:
                for path in nx.all_simple_paths(Gm, source, target):
                    valid_nodes.update(path)
            except nx.NetworkXNoPath:
                pass # No paths found
            # Ensure source and target are included if they exist
            if source in Gm: valid_nodes.add(source)
            if target in Gm: valid_nodes.add(target)

    elif Gm.is_directed():
        # --- Angle Checks & DiGraph: Use Fast Iterative Method ---

        # 1. Initial reachability filter
        reachable_from_src = set(nx.descendants(Gm, source)) | {source}
        can_reach_tgt = set(nx.ancestors(Gm, target)) | {target}
        potentially_valid_nodes = reachable_from_src & can_reach_tgt

        # Handle cases where no path exists initially
        if not (source in potentially_valid_nodes and target in potentially_valid_nodes):
             # Keep only source/target if they were in the graph, otherwise empty
             nodes_to_keep = set()
             if source in Gm: nodes_to_keep.add(source)
             if target in Gm: nodes_to_keep.add(target)
             Gm.remove_nodes_from([n for n in Gm.nodes() if n not in nodes_to_keep])
             return # Nothing more to do

        # Pre-cache coordinates for potentially valid nodes
        coords = {}
        try:
            for node in potentially_valid_nodes:
                # Use get() for slightly safer access, though attribute should exist
                lon = Gm.nodes[node].get('lon')
                lat = Gm.nodes[node].get('lat')
                if lon is None or lat is None:
                    raise ValueError(f"Node {node} is missing required 'lat' or 'lon' attribute.")
                coords[node] = (lon, lat)
        except (KeyError, AttributeError) as e: # Catch missing node data or attribute access error
             raise ValueError(f"Failed to access lat/lon for node {node}. Ensure nodes have data dict with 'lat'/'lon'. Error: {e}") from e


        # 2. Iterative refinement
        while True:
            nodes_removed_in_iteration = set()
            # Evaluate nodes other than source and target
            nodes_to_evaluate = potentially_valid_nodes - {source, target}

            for v in nodes_to_evaluate:
                predecessors = {u for u in Gm.predecessors(v) if u in potentially_valid_nodes}
                successors = {w for w in Gm.successors(v) if w in potentially_valid_nodes}
                if not predecessors or not successors:
                    nodes_removed_in_iteration.add(v)
                    continue

                has_valid_turn = False
                pos2 = coords[v]
                for u in predecessors:
                    pos1 = coords[u]
                    for w in successors:
                        pos3 = coords[w]
                        angle = get_turn_angle(pos1, pos2, pos3)
                        # --- Disallow U-turns (angle very close to 0°) ---
                        if angle >= 0.0 and angle <= min_turn_angle:
                            has_valid_turn = True
                            break
                    if has_valid_turn:
                        break

                if not has_valid_turn:
                    nodes_removed_in_iteration.add(v)
                    

            # Update potentially_valid_nodes or break if converged
            if not nodes_removed_in_iteration:
                break
            else:
                potentially_valid_nodes -= nodes_removed_in_iteration
                
            # print(f'1predecessor: {list(Gm.predecessors('BAMES'))}, successor: {list(Gm.successors('BAMES'))}')

        valid_nodes = potentially_valid_nodes # Final set after iteration

    else:
        raise Exception('Error: Angle checking on undirected graph requires all_simple_paths, which can be slow.')

    # print(f'2predecessor: {list(Gm.predecessors('BAMES'))}, successor: {list(Gm.successors('BAMES'))}')
    # --- Final Pruning Step (applies to all cases) ---
    nodes_to_remove = [n for n in Gm.nodes() if n not in valid_nodes]
    # print(f'nodes_to_remove: {nodes_to_remove}')
    if nodes_to_remove:
        Gm.remove_nodes_from(nodes_to_remove)
        
    # print(f'3predecessor: {list(Gm.predecessors('BAMES'))}, successor: {list(Gm.successors('BAMES'))}')
        

    # Optional: Check if graph changed (can be useful for debugging)
    # final_node_count = Gm.number_of_nodes()
    # if initial_node_count != final_node_count:
    #     print(f"Pruning removed {initial_node_count - final_node_count} nodes.")

# --- Example Usage ---
# (Same as before, ensure Gmxx exists with 'lat'/'lon')
# try:
#     Gmxx_copy = Gm.copy() # Work on a copy
#     print(f'Before: {Gmxx_copy.number_of_nodes()} nodes, {Gmxx_copy.number_of_edges()} edges.')
#     prune_dead_ends_and_sharp_turns(Gmxx_copy, 'LEMD', 'EGLL', min_turn_angle=45)
#     print(f'After: {Gmxx_copy.number_of_nodes()} nodes, {Gmxx_copy.number_of_edges()} edges.')
# except (ValueError, KeyError, NameError, Exception) as e:
#      print(f"An error occurred: {e}")
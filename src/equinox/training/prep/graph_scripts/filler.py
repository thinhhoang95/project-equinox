import math
import networkx as nx
from sklearn.neighbors import BallTree # Or KDTree
import numpy as np
from tqdm import tqdm

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great-circle distance between two points on the earth (specified in decimal degrees)."""
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r

def calculate_bearing(lat1, lon1, lat2, lon2):
    """Calculate the initial bearing (direction) from point 1 to point 2."""
    lat1, lon1, lat2, lon2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    initial_bearing = math.atan2(y, x)
    initial_bearing = math.degrees(initial_bearing)
    # Normalize bearing to 0-360
    bearing = (initial_bearing + 360) % 360
    return bearing

def angle_difference(bearing1, bearing2):
    """Calculate the smallest difference between two bearings (0-360 degrees)."""
    diff = abs(bearing1 - bearing2)
    return min(diff, 360 - diff)

def get_node_coordinates(G, node):
    """Helper to get lat, lon from graph node."""
    # Handle potential missing nodes during lookup if graph is modified concurrently elsewhere
    if node not in G.nodes:
        raise KeyError(f"Node {node} not found in graph.")
    if 'lat' not in G.nodes[node] or 'lon' not in G.nodes[node]:
         raise KeyError(f"Missing 'lat' or 'lon' for node {node}.")
    return G.nodes[node]['lat'], G.nodes[node]['lon']

def fill_graph_gaps(G, lemd_coords, egll_coords, T=1.0, max_neighbor_dist_km=100, k_neighbors=10,
                    max_edges_to_process=None):
    """
    Identifies gaps between flight segments and adds plausible connecting edges to the graph G.

    Starts from potentially good existing edges (long, aligned LEMD->EGLL) and tries
    to extend them forwards and backwards by finding nearby, directionally
    consistent waypoints, adding edges to G for these connections.

    Args:
        G (nx.DiGraph): Graph where nodes have 'lat', 'lon' attributes. Edges represent flight segments.
                        THIS GRAPH WILL BE MODIFIED IN PLACE.
        lemd_coords (tuple): (latitude, longitude) of LEMD.
        egll_coords (tuple): (latitude, longitude) of EGLL.
        T (float): Temperature parameter for Boltzmann probability (currently unused, selects best).
        max_neighbor_dist_km (float): Max distance (km) to search for neighbors for connection.
        k_neighbors (int): Number of nearest neighbors to consider if radius search yields few results.
        max_edges_to_process (int, optional): Limit the number of initial 'seed' edges to process. Defaults to None (process all).


    Returns:
        tuple: (nx.DiGraph, set): The modified graph G and a set containing the newly added edges (u, v).
    """
    lemd_lat, lemd_lon = lemd_coords
    egll_lat, egll_lon = egll_coords

    # 1. Calculate overall direction
    overall_bearing = calculate_bearing(lemd_lat, lemd_lon, egll_lat, egll_lon)
    overall_vec = np.array([egll_lon - lemd_lon, egll_lat - lemd_lat])
    # Normalize, handle potential zero vector if coords are identical
    norm = np.linalg.norm(overall_vec)
    if norm > 1e-9:
        overall_vec /= norm
    else:
        # Handle case where start and end points are the same
        # In this scenario, directionality checks become less meaningful
        # We could return early, or default to a different behavior.
        # For now, let's proceed but direction checks will likely fail/behave strangely.
        print("Warning: LEMD and EGLL coordinates are very close or identical.")
        # Default vector, e.g., North, or handle as error? Let's default to North.
        overall_vec = np.array([0, 1]) # Arbitrary direction if points coincide


    # 2. Score and sort initial edges existing in G
    edges_with_scores = []
    initial_edges = list(G.edges()) # Get a static list of edges at the start
    print(f"Scoring {len(initial_edges)} initial edges...")
    for u, v in tqdm(initial_edges):
        try:
            lat1, lon1 = get_node_coordinates(G, u)
            lat2, lon2 = get_node_coordinates(G, v)
        except KeyError as e:
            # print(f"Warning: Skipping edge ({u}, {v}) due to missing node data: {e}")
            continue # Skip edges if nodes or their data are missing

        length = haversine_distance(lat1, lon1, lat2, lon2)
        # Avoid division by zero or nonsensical bearings for zero-length segments
        if length < 1e-3: # e.g., less than 1 meter
            continue

        bearing = calculate_bearing(lat1, lon1, lat2, lon2)
        align_diff = angle_difference(bearing, overall_bearing)
        alignment_score = max(0, 1 - (align_diff / 90.0)) # Penalize > 90 deg difference heavily
        score = length * (alignment_score ** 2)
        edges_with_scores.append(((u, v), score))

    # Sort edges by score, highest first
    sorted_edges = sorted(edges_with_scores, key=lambda item: item[1], reverse=True)

    # Limit number of edges to process if requested
    if max_edges_to_process is not None and len(sorted_edges) > max_edges_to_process:
        print(f"Processing top {max_edges_to_process} scored edges out of {len(sorted_edges)}.")
        sorted_edges = sorted_edges[:max_edges_to_process]
    else:
        print(f"Processing all {len(sorted_edges)} scored edges.")


    # 3. Prepare for neighbor search using all nodes currently in G
    # It's crucial to rebuild this if nodes are added mid-process, but
    # this version adds edges between existing nodes, so it should be stable.
    all_nodes = list(G.nodes())
    # Filter nodes to only include those with valid coordinates for BallTree
    valid_nodes_for_tree = []
    coords_list_rad = []
    node_to_idx_map = {}
    idx_to_node_map = {}

    print("Building neighbor search tree...")
    for i, node in enumerate(tqdm(all_nodes)):
         try:
             lat, lon = get_node_coordinates(G, node)
             coords_list_rad.append([math.radians(lat), math.radians(lon)])
             valid_nodes_for_tree.append(node)
             current_idx = len(valid_nodes_for_tree) - 1
             node_to_idx_map[node] = current_idx
             idx_to_node_map[current_idx] = node
         except KeyError:
             # print(f"Warning: Node {node} skipped in neighbor search tree due to missing coordinates.")
             pass # Skip nodes without coordinates

    if not valid_nodes_for_tree:
        print("Error: No nodes with valid coordinates found in the graph. Cannot perform neighbor search.")
        return G, set()

    coords_list_rad_np = np.array(coords_list_rad)
    tree = BallTree(coords_list_rad_np, metric='haversine')


    # 4. Iterative Path Growing and Edge Addition
    processed_seed_edges = set() # Tracks initial edges used as seeds
    added_edges = set()         # Tracks newly added edges (u, v)

    print(f"Growing paths and adding edges...")
    for edge_data, score in tqdm(sorted_edges):
        seed_edge = edge_data
        if seed_edge in processed_seed_edges:
            continue

        u, v = seed_edge
        processed_seed_edges.add(seed_edge) # Mark initial seed edge as processed

        # --- Grow Forward ---
        last_node = v
        path_nodes_forward = [u, v] # Keep track of nodes in this specific growth sequence
        while True:
            if last_node not in node_to_idx_map: # Check if last_node has valid coords & is in tree
                # print(f"Stopping forward growth from {last_node}: Node not in search tree (missing coords?).")
                break

            try:
                last_lat, last_lon = get_node_coordinates(G, last_node)
                last_coords_rad = np.radians([[last_lat, last_lon]])
            except KeyError:
                # print(f"Stopping forward growth from {last_node}: Missing coordinates.")
                break # Stop growing if node data is missing

            # Find k nearest neighbors within max distance
            dist_limit_rad = max_neighbor_dist_km / 6371.0
            # Query neighbors: first by radius, then k nearest if radius is empty
            indices = tree.query_radius(last_coords_rad, r=dist_limit_rad, return_distance=False)[0]

            # If radius search empty/only self, try k-NN
            if len(indices) <= (1 if node_to_idx_map[last_node] in indices else 0) and k_neighbors > 0:
                 distances, indices_k = tree.query(last_coords_rad, k=min(k_neighbors + 1, len(valid_nodes_for_tree)), return_distance=True)
                 # indices from query are shape (1, k), need [0]
                 # Filter out self explicitly if present
                 indices = [idx for idx in indices_k[0] if idx != node_to_idx_map[last_node]]


            candidates = []
            for idx in indices:
                neighbor_node = idx_to_node_map[idx]

                # Skip nodes already part of this specific forward growth sequence
                if neighbor_node in path_nodes_forward:
                    continue

                # Check if edge already exists
                if G.has_edge(last_node, neighbor_node):
                    continue

                try:
                    n_lat, n_lon = get_node_coordinates(G, neighbor_node)
                except KeyError:
                    continue # Skip neighbors without coords

                # Check directionality
                step_vec = np.array([n_lon - last_lon, n_lat - last_lat])
                norm_step = np.linalg.norm(step_vec)
                if norm_step > 1e-6:
                    step_vec /= norm_step
                    # Allow some deviation, but prevent backtracking (dot product > threshold)
                    if np.dot(step_vec, overall_vec) < 0.1:
                        continue

                dist_ln = haversine_distance(last_lat, last_lon, n_lat, n_lon)
                # Ensure neighbor is actually within distance limit if using k-NN fallback
                if dist_ln > max_neighbor_dist_km:
                     continue

                bearing_ln = calculate_bearing(last_lat, last_lon, n_lat, n_lon)
                align_diff_ln = angle_difference(bearing_ln, overall_bearing)
                energy = dist_ln * (1 + (align_diff_ln / 90.0)) # Lower is better

                candidates.append((neighbor_node, energy))

            if not candidates:
                break # No suitable forward neighbor found

            # Select best candidate (lowest energy)
            candidates.sort(key=lambda item: item[1])
            best_w, best_energy = candidates[0]

            # **** ADD EDGE TO GRAPH ****
            new_edge = (last_node, best_w)
            if not G.has_edge(last_node, best_w):
                # print(f"  Adding forward edge: {new_edge}")
                G.add_edge(last_node, best_w)
                added_edges.add(new_edge)

            path_nodes_forward.append(best_w) # Track sequence
            last_node = best_w # Continue growing from the new node


        # --- Grow Backward --- (Similar logic)
        first_node = u
        path_nodes_backward = [u] # Keep track of nodes added backward from u
        while True:
            if first_node not in node_to_idx_map:
                # print(f"Stopping backward growth from {first_node}: Node not in search tree.")
                break
            try:
                first_lat, first_lon = get_node_coordinates(G, first_node)
                first_coords_rad = np.radians([[first_lat, first_lon]])
            except KeyError:
                # print(f"Stopping backward growth from {first_node}: Missing coordinates.")
                break

            dist_limit_rad = max_neighbor_dist_km / 6371.0
            indices = tree.query_radius(first_coords_rad, r=dist_limit_rad, return_distance=False)[0]

            if len(indices) <= (1 if node_to_idx_map[first_node] in indices else 0) and k_neighbors > 0:
                distances, indices_k = tree.query(first_coords_rad, k=min(k_neighbors + 1, len(valid_nodes_for_tree)), return_distance=True)
                indices = [idx for idx in indices_k[0] if idx != node_to_idx_map[first_node]]


            candidates = []
            for idx in indices:
                neighbor_node = idx_to_node_map[idx]

                # Avoid connecting back to the forward path or backward path already explored
                if neighbor_node in path_nodes_forward or neighbor_node in path_nodes_backward:
                     continue

                # Check if edge already exists
                if G.has_edge(neighbor_node, first_node):
                     continue

                try:
                    p_lat, p_lon = get_node_coordinates(G, neighbor_node)
                except KeyError:
                    continue

                # Check directionality: step from neighbor_node -> first_node should align with overall LEMD->EGLL
                step_vec = np.array([first_lon - p_lon, first_lat - p_lat])
                norm_step = np.linalg.norm(step_vec)
                if norm_step > 1e-6:
                    step_vec /= norm_step
                    if np.dot(step_vec, overall_vec) < 0.1:
                        continue

                dist_pf = haversine_distance(p_lat, p_lon, first_lat, first_lon)
                if dist_pf > max_neighbor_dist_km:
                     continue

                bearing_pf = calculate_bearing(p_lat, p_lon, first_lat, first_lon)
                align_diff_pf = angle_difference(bearing_pf, overall_bearing)
                energy = dist_pf * (1 + (align_diff_pf / 90.0))

                candidates.append((neighbor_node, energy))

            if not candidates:
                break # No suitable backward neighbor found

            candidates.sort(key=lambda item: item[1])
            best_p, best_energy = candidates[0]

            # **** ADD EDGE TO GRAPH ****
            new_edge = (best_p, first_node)
            if not G.has_edge(best_p, first_node):
                # print(f"  Adding backward edge: {new_edge}")
                G.add_edge(best_p, first_node)
                added_edges.add(new_edge)

            path_nodes_backward.append(best_p) # Track sequence
            first_node = best_p # Continue growing backward from this new node


    print(f"Finished processing. Added {len(added_edges)} new edges.")
    return G, added_edges

import os
import networkx as nx 
from equinox.helpers.haversine import haversine as haversine_distance
from tqdm import tqdm
from equinox.training.prep.graph_scripts.cycle_prune import make_acyclic_by_bearing_and_degree
import math

def filter_graph_by_detour(G: nx.Graph, origin, destination, threshold: float) -> nx.Graph:
    """
    Return the induced subgraph of G containing exactly those nodes 'n'
    for which:
        dist(origin → n) + dist(n → destination)
        ≤ (1 + threshold) x dist(origin → destination)

    Parameters
    ----------
    G : nx.Graph
        A graph whose nodes have 'lat' and 'lon' attributes (in degrees).
    origin : node
        The node ID in G of the origin waypoint.
    destination : node
        The node ID in G of the destination waypoint.
    threshold : float
        Allowed fractional detour (e.g. 0.1 for up to a 10% longer path).

    Returns
    -------
    nx.Graph
        The subgraph induced by the set of waypoints satisfying the detour criterion.
    """
    # 1. Precompute the direct (shortest possible) distance
    lat_o, lon_o = G.nodes[origin]['lat'], G.nodes[origin]['lon']
    lat_d, lon_d = G.nodes[destination]['lat'], G.nodes[destination]['lon']
    direct_dist = haversine_distance(lat_o, lon_o, lat_d, lon_d)
    max_allowed = (1.0 + threshold) * direct_dist

    # 2. Scan all nodes once, pruning early if origin→node alone already exceeds max_allowed
    keepers = []
    for n, data in G.nodes(data=True):
        lat_n, lon_n = data['lat'], data['lon']

        d_on = haversine_distance(lat_o, lon_o, lat_n, lon_n)
        if d_on > max_allowed:
            # even reaching n already is too far
            continue

        d_nd = haversine_distance(lat_n, lon_n, lat_d, lon_d)
        if d_on + d_nd <= max_allowed:
            keepers.append(n)

    # 3. Return the induced subgraph (make a copy if you need it disconnected from G)
    return G.subgraph(keepers).copy()

def add_edges_to_Gm(route_df, Gm, haversine_distance):
    """
    For each row in route_df, add edges to Gm for each consecutive pair in real_waypoints.
    
    Parameters
    ----------
    route_df : pandas.DataFrame
        Must contain a column 'real_waypoints' with space‑separated waypoint IDs.
    Gm : networkx.Graph (or DiGraph)
        Graph whose nodes are waypoint IDs, each with 'lat' and 'lon' attributes.
    haversine_distance : callable
        Function lat1, lon1, lat2, lon2 -> distance in nautical miles.
    """
    for idx, row in route_df.iterrows():
        # split into waypoint sequence
        wpts = row['real_waypoints'].split()
        # iterate over each segment
        for u, v in zip(wpts, wpts[1:]):
            if Gm.has_node(u) and Gm.has_node(v):
                lat1, lon1 = Gm.nodes[u]['lat'], Gm.nodes[u]['lon']
                lat2, lon2 = Gm.nodes[v]['lat'], Gm.nodes[v]['lon']
                # compute distance in nautical miles
                dist_nm = haversine_distance(lat1, lon1, lat2, lon2)
                # add edge with dist, tail‑wind, and preference
                Gm.add_edge(u, v,
                           dist=dist_nm,
                           t_wind=0.0,   # default tail‑wind (m/s)
                           pref=0)       # default preference
                
def add_edges_from_historical_data(directory: str, origin: str, dest: str, graph: nx.Graph):
    """
    Add edges to a graph based on historical route data from CSV files.
    
    This function scans all CSV files in a given directory for historical flight routes
    that match a specific origin-destination pair, extracts all consecutive waypoint
    segments from those routes, and adds them as edges to the provided graph.
    
    Parameters
    ----------
    directory : str
        Path to directory containing CSV files with historical route data.
        Each CSV file should contain a 'real_waypoints' column with space-separated
        waypoint identifiers.
    origin : str
        Origin waypoint identifier (e.g., 'LEMD'). Routes must start with this waypoint.
    dest : str
        Destination waypoint identifier (e.g., 'EGLL'). Routes must end with this waypoint.
    graph : networkx.Graph
        Graph to add edges to. Must contain nodes with 'lat' and 'lon' attributes.
        Only segments where both waypoints exist as nodes in the graph will be added.
        
    Returns
    -------
    None
        Function modifies the graph in-place.
        
    Notes
    -----
    - Only processes CSV files that contain a 'real_waypoints' column
    - Filters routes to only include those starting with origin and ending with dest
    - Uses haversine distance to calculate edge weights in nautical miles
    - Avoids adding duplicate edges by checking if edge already exists
    - Prints summary statistics about segments found and edges added
    """
    import os
    import pandas as pd
    
    segments = set()  # Use set to avoid duplicates
    
    # Get all CSV files in directory
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    # Remove all macOS ._ files
    csv_files = [f for f in csv_files if not f.startswith('.')]
    
    pbar = tqdm(csv_files, desc='Browsing historical data')
    for csv_file in pbar:
        file_path = os.path.join(directory, csv_file)
        try:
            # Read CSV file
            df = pd.read_csv(file_path)
            
            # Check if required column exists
            if 'real_waypoints' not in df.columns:
                continue
                
            # Filter routes that start with origin and end with dest
            filtered_routes = df[
                df['real_waypoints'].str.startswith(origin + ' ') & 
                df['real_waypoints'].str.endswith(' ' + dest)
            ]
            
            # Extract segments from each filtered route
            for _, row in filtered_routes.iterrows():
                waypoints = row['real_waypoints'].split()
                # Create consecutive pairs as segments
                for i in range(len(waypoints) - 1):
                    segments.add((waypoints[i], waypoints[i + 1]))
            
            # Update progress bar with current segment count
            pbar.set_postfix({'segments': len(segments)})

                    
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            continue

    print(f'Found {len(segments)} unique segments for {origin} → {dest}')

    # Add segments to graph if they don't exist
    edges_added = 0
    for u, v in segments:
        if graph.has_node(u) and graph.has_node(v):
            if not graph.has_edge(u, v):
                # Get coordinates for both nodes
                lat1, lon1 = graph.nodes[u]['lat'], graph.nodes[u]['lon']
                lat2, lon2 = graph.nodes[v]['lat'], graph.nodes[v]['lon']
                # Calculate distance in nautical miles
                dist_nm = haversine_distance(lat1, lon1, lat2, lon2)
                # Add edge with distance attribute
                graph.add_edge(u, v, dist=dist_nm)
                edges_added += 1
    
    print(f'Added {edges_added} new edges to the graph')
    
    return list(segments)

def nodes_that_cannot_reach_goal(G, goal):
    """
    Given a directed or undirected graph G and a goal node,
    return a list of nodes from which the goal node is NOT reachable.

    Parameters
    ----------
    G : networkx.Graph or networkx.DiGraph
        The route graph.
    goal : node
        The goal node.

    Returns
    -------
    list
        List of nodes from which goal is NOT reachable.
    """
    if not G.has_node(goal):
        raise KeyError(f"Goal node {goal} not in graph")

    all_nodes = set(G.nodes())
    if G.is_directed():
        # In a DiGraph, nodes that can reach goal are its ancestors plus itself
        reachable_nodes = set(nx.ancestors(G, goal))
        reachable_nodes.add(goal)
    else:
        # In undirected, all nodes in the same connected component as goal
        reachable_nodes = set(nx.node_connected_component(G, goal))
    unreachable_nodes = all_nodes - reachable_nodes
    return list(unreachable_nodes)
    
def nodes_that_source_cannot_reach(G, source):
    """
    Given a directed or undirected graph G and a source node,
    return a list of nodes that cannot be reached from the source node.

    Parameters
    ----------
    G : networkx.Graph or networkx.DiGraph
        The route graph.
    source : node
        The source node.

    Returns
    -------
    list
        List of nodes that cannot be reached from the source node.
    """
    if not G.has_node(source):
        raise KeyError(f"Source node {source} not in graph")

    all_nodes = set(G.nodes())
    if G.is_directed():
        # In a DiGraph, nodes reachable from source are its descendants plus itself
        reachable_nodes = set(nx.descendants(G, source))
        reachable_nodes.add(source)
    else:
        # In undirected, all nodes in the same connected component as source
        reachable_nodes = set(nx.node_connected_component(G, source))
    unreachable_nodes = all_nodes - reachable_nodes
    return list(unreachable_nodes)
    

def is_acyclic(G):
    """
    Checks whether the given graph is acyclic.

    Parameters
    ----------
    G : networkx.Graph or networkx.DiGraph
        The graph to check.

    Returns
    -------
    bool
        True if the graph is acyclic, False otherwise.
    """
    if G.is_directed():
        # For directed graphs, use networkx's is_directed_acyclic_graph
        return nx.is_directed_acyclic_graph(G)
    else:
        # For undirected graphs, acyclic means no cycles (i.e., a forest)
        # A connected undirected acyclic graph is a tree; in general, a forest
        try:
            cycles = nx.find_cycle(G)
            return False
        except nx.exception.NetworkXNoCycle:
            return True

# Bearing between source and goal 
def compute_bearing(lat1, lon1, lat2, lon2):
    """
    Compute the initial bearing (forward azimuth) from (lat1, lon1) to (lat2, lon2).
    All args in degrees. Returns bearing in degrees from North (0-360).
    """
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlon_rad = math.radians(lon2 - lon1)
    x = math.sin(dlon_rad) * math.cos(lat2_rad)
    y = math.cos(lat1_rad) * math.sin(lat2_rad) - \
        math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon_rad)
    bearing_rad = math.atan2(x, y)
    bearing_deg = (math.degrees(bearing_rad) + 360) % 360
    return bearing_deg


def prepare_base_graph(nodes_only_graph_path: str, source_id: str, destination_id: str, routes_dir: str,
                       delete_isolated_nodes: bool = True, minimum_detour_allowed: float = 0.035):
    Gno = nx.read_gml(nodes_only_graph_path)
    # Convert Gno to a directed graph
    Gno = Gno.to_directed()

    Gno = filter_graph_by_detour(Gno, source_id, destination_id, minimum_detour_allowed) # Only 7.5% of *minimum* detour is allowed, real path could be much longer
    print(f'The filtered graph has {Gno.number_of_nodes()} nodes and {Gno.number_of_edges()} edges')

    # Compute bearing between source and destination
    bearing = compute_bearing(Gno.nodes[source_id]['lat'], Gno.nodes[source_id]['lon'], Gno.nodes[destination_id]['lat'], Gno.nodes[destination_id]['lon'])
    print(f'Bearing between {source_id} and {destination_id}: {bearing} degrees')

    # Add edges from historical data
    add_edges_from_historical_data(
        directory=routes_dir,
        origin=source_id,
        dest=destination_id,
        graph=Gno
    )

    if delete_isolated_nodes:
        # Remove isolated nodes
        Gno = remove_isolated_nodes(Gno)
    else:
        print('Not removing isolated nodes per instruction')

    return Gno

def remove_collinear_edges(Gno):
    # Remove collinear edges
    from equinox.training.prep.graph_scripts.collinear import refine_graph
    for i in range(10):
        refine_graph(Gno, 1e-4, 1e-4)
        print(f'Collinearity check iteration {i}: {Gno.number_of_edges()} edges remaining.')
    return Gno

def remove_isolated_nodes(Gno):
    isolated_nodes = list(nx.isolates(Gno))
    Gno.remove_nodes_from(isolated_nodes)
    print(f"Removed {len(isolated_nodes)} isolated nodes from the graph.")
    print(f"The graph now has {Gno.number_of_nodes()} nodes and {Gno.number_of_edges()} edges.")
    return Gno

def make_graph_acyclic(Gno, source_id, destination_id):
    bearing = compute_bearing(Gno.nodes[source_id]['lat'], Gno.nodes[source_id]['lon'], Gno.nodes[destination_id]['lat'], Gno.nodes[destination_id]['lon'])
    Gno = make_acyclic_by_bearing_and_degree(Gno, bearing)
    return Gno

from equinox.training.prep.graph_scripts.improve_connectivity import improve_graph_connectivity

def improve_connectivity(Gno, source_id, destination_id, n_iter=20):
    bearing = compute_bearing(Gno.nodes[source_id]['lat'], Gno.nodes[source_id]['lon'], Gno.nodes[destination_id]['lat'], Gno.nodes[destination_id]['lon'])
    for i in range(n_iter):
        orphan_goal_nodes = nodes_that_cannot_reach_goal(Gno, destination_id)
        orphan_source_nodes = nodes_that_source_cannot_reach(Gno, source_id)

        print(f'Iteration {i}, there are {len(nodes_that_cannot_reach_goal(Gno, destination_id))} nodes that can not reach {destination_id}, {len(nodes_that_source_cannot_reach(Gno, source_id))} nodes that can not be reached from {source_id}')
        edge_before = Gno.number_of_edges()

        Gno = improve_graph_connectivity(Gno, orphan_goal_nodes, orphan_source_nodes,
                                        main_bearing=bearing,
                                        radius_nm=400,
                                        bearing_tolerance_deg=85,
                                        n_degree_connections=4,
                                        n_nearest_connections=4)
        
        edge_after = Gno.number_of_edges()
        print(f'Iteration {i}, {edge_after - edge_before} edges added')

def remove_backtracking_edges(G, source_id, destination_id, max_allowed_deviation_angle=90):
    """
    Remove edges from G that deviate significantly from the main bearing.

    An edge is removed if its bearing differs from the main source-to-destination
    bearing by more than `max_allowed_deviation_angle`. This helps eliminate
    routes that backtrack or take significant detours.
    """
    main_bearing = compute_bearing(G.nodes[source_id]['lat'], G.nodes[source_id]['lon'], G.nodes[destination_id]['lat'], G.nodes[destination_id]['lon'])

    def angle_diff(a, b):
        d = abs(a - b) % 360
        return min(d, 360 - d)

    edges_to_remove = []
    for u, v in G.edges():
        lat1 = G.nodes[u].get('lat')
        lon1 = G.nodes[u].get('lon')
        lat2 = G.nodes[v].get('lat')
        lon2 = G.nodes[v].get('lon')
        if None in (lat1, lon1, lat2, lon2):
            continue  # skip if missing coordinates
        edge_bearing = compute_bearing(lat1, lon1, lat2, lon2)
        diff = angle_diff(main_bearing, edge_bearing)
        if diff > max_allowed_deviation_angle:
            edges_to_remove.append((u, v))
    # Remove the edges
    G.remove_edges_from(edges_to_remove)
    if edges_to_remove:
        print(f"Removed {len(edges_to_remove)} backtracking edges (angle > {max_allowed_deviation_angle}° from main bearing).")
    return G

def remove_unreachable_nodes(G, source_id, destination_id):
    """
    Remove all nodes that are unreachable from the source OR cannot reach the destination.
    
    A node is kept only if:
    1. There is a path from source_id to the node, AND
    2. There is a path from the node to destination_id
    
    This ensures every remaining node lies on some path from source to destination.
    """
    import networkx as nx
    
    # Find nodes reachable from source
    try:
        reachable_from_source = set(nx.single_source_shortest_path_length(G, source_id).keys())
    except nx.NetworkXNoPath:
        reachable_from_source = {source_id}
    
    # Find nodes that can reach destination (reverse the graph)
    G_reversed = G.reverse() if G.is_directed() else G
    try:
        can_reach_destination = set(nx.single_source_shortest_path_length(G_reversed, destination_id).keys())
    except nx.NetworkXNoPath:
        can_reach_destination = {destination_id}
    
    # Keep only nodes that satisfy both conditions
    nodes_to_keep = reachable_from_source.intersection(can_reach_destination)
    nodes_to_remove = set(G.nodes()) - nodes_to_keep
    
    if nodes_to_remove:
        G.remove_nodes_from(nodes_to_remove)
        print(f"Removed {len(nodes_to_remove)} unreachable nodes. Graph now has {G.number_of_nodes()} nodes.")
    else:
        print("No unreachable nodes found.")
    
    return G


def process_graph(nodes_only_graph_path: str, source_id: str, destination_id: str, routes_dir: str,
                       delete_isolated_nodes: bool = False):
    Gno = prepare_base_graph(nodes_only_graph_path, source_id, destination_id, routes_dir, delete_isolated_nodes)
    improve_connectivity(Gno, source_id, destination_id, n_iter=10)
    # Remove collinear edges
    Gno = remove_collinear_edges(Gno)
    Gno = remove_backtracking_edges(Gno, source_id, destination_id, max_allowed_deviation_angle=90)
    Gno = remove_isolated_nodes(Gno)
    Gno = remove_unreachable_nodes(Gno, source_id, destination_id)
    Gno = make_graph_acyclic(Gno, source_id, destination_id)
    return Gno

def process_and_save_graph(nodes_only_graph_path: str, source_id: str, destination_id: str, routes_dir: str,
                           delete_isolated_nodes: bool = False, output_path: str = None):
    Gno = process_graph(nodes_only_graph_path, source_id, destination_id, routes_dir, delete_isolated_nodes)
    if output_path is not None:
        nx.write_gml(Gno, output_path)
    return Gno

if __name__ == '__main__':
    path_prefix = 'D:\\project-akrav\\'
    nodes_only_graph_path = os.path.join(path_prefix, 'data', 'graphs', 'ats_fra_nodes_only.gml')
    source_id = 'LEMD'
    destination_id = 'EGLL'
    routes_dir = os.path.join(path_prefix, 'matched_filtered_data')

    import time
    path_prefix_output = 'D:\\project-equinox\\'
    case_name = f'{source_id}_{destination_id}'
    start_time = time.time()
    output_path = os.path.join(path_prefix_output, 'data', 'cases', case_name, 'graphs', f'routes.gml')
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    Gno = process_and_save_graph(nodes_only_graph_path, source_id, destination_id, routes_dir,
    output_path=output_path)
    print(f'Graph processing completed in {time.time() - start_time} seconds')
    print(f'Final graph has {Gno.number_of_nodes()} nodes and {Gno.number_of_edges()} edges')

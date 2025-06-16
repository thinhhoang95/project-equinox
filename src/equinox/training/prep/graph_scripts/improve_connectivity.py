import math
import networkx as nx

# Earth radius in nautical miles
EARTH_RADIUS_NM = 3440.0  # Approximate Earth radius in nautical miles

def haversine_distance_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees) in nautical miles.
    """
    # Convert decimal degrees to radians
    lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
    lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)

    # Haversine formula
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance_nm = EARTH_RADIUS_NM * c
    return distance_nm

def calculate_initial_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculates the initial bearing (forward azimuth) from point 1 to point 2
    in degrees (0-360).
    """
    lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
    lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)

    dlon = lon2_rad - lon1_rad

    x = math.sin(dlon) * math.cos(lat2_rad)
    y = math.cos(lat1_rad) * math.sin(lat2_rad) - \
        math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon)

    initial_bearing_rad = math.atan2(x, y)
    initial_bearing_deg = math.degrees(initial_bearing_rad)
    
    # Normalize to 0-360 degrees
    return (initial_bearing_deg + 360) % 360

def bearing_difference(bearing1: float, bearing2: float) -> float:
    """
    Calculates the shortest angular difference between two bearings.
    The result is in the range [-180, 180].
    """
    diff = (bearing2 - bearing1 + 180) % 360 - 180
    return diff

def improve_graph_connectivity(
    G: nx.Graph, 
    goal_orphans: list, 
    source_orphans: list, 
    main_bearing: float, 
    radius_nm: float,
    bearing_tolerance_deg: float = 90.0,
    n_degree_connections: int = 2,
    n_nearest_connections: int = 1
) -> nx.Graph:
    """
    Attempts to improve connectivity of a route graph G by adding edges to/from orphan nodes.

    Args:
        G: The route graph (networkx.Graph or networkx.DiGraph). Nodes must have 'lat' and 'lon' attributes.
        goal_orphans: List of nodes from which the goal node cannot be reached.
        source_orphans: List of nodes that cannot reach the source node.
        main_bearing: The main bearing (in degrees) between the source and goal nodes.
        radius_nm: Search radius in nautical miles to find connection candidates.
        bearing_tolerance_deg: Allowable deviation (in degrees) from the main_bearing for new edges.
        n_degree_connections: The number of degree-based connections to attempt for each orphan node.
        n_nearest_connections: The number of nearest-based connections to attempt for each orphan node.

    Returns:
        The modified graph G.
    """
    all_nodes = set(G.nodes())
    goal_orphans_set = set(g for g in goal_orphans if g in G) # Ensure orphans are in G
    source_orphans_set = set(s for s in source_orphans if s in G) # Ensure orphans are in G

    # Cap the bearing tolerance to a maximum of 90 degrees to prevent backtracking.
    # This ensures new edges are always in the "forward" hemisphere relative to main_bearing.
    effective_tolerance_deg = min(bearing_tolerance_deg, 90.0)

    # Note: The problem implies non_orphans are derived from all_nodes.
    # If an orphan is connected, it's still processed based on the initial list.
    
    # 1. Derive non-orphan lists
    # These are potential connection targets/sources, not necessarily reachable/able to reach goal/source.
    non_goal_orphans = list(all_nodes - goal_orphans_set)
    non_source_orphans = list(all_nodes - source_orphans_set)

    # 2. Process goal-orphan nodes first
    for g_orphan in goal_orphans_set:
        try:
            g_orphan_attrs = G.nodes[g_orphan]
            g_orphan_lat, g_orphan_lon = g_orphan_attrs['lat'], g_orphan_attrs['lon']
        except KeyError:
            print(f"Warning: Goal orphan {g_orphan} missing lat/lon attributes or not in G. Skipping.")
            continue

        potential_connections = []
        for candidate_node in non_goal_orphans:
            if candidate_node == g_orphan:
                continue
            try:
                cand_attrs = G.nodes[candidate_node]
                cand_lat, cand_lon = cand_attrs['lat'], cand_attrs['lon']
            except KeyError:
                # Silently skip candidates without lat/lon, or log if necessary
                continue

            dist_nm = haversine_distance_nm(g_orphan_lat, g_orphan_lon, cand_lat, cand_lon)

            if dist_nm <= radius_nm:
                bearing_to_cand = calculate_initial_bearing(g_orphan_lat, g_orphan_lon, cand_lat, cand_lon)
                diff = bearing_difference(main_bearing, bearing_to_cand)
                
                if abs(diff) <= effective_tolerance_deg:
                    potential_connections.append((candidate_node, G.degree(candidate_node), dist_nm))
        
        # Degree-based connections
        # Sort by degree (desc), then distance (asc)
        sorted_by_degree = sorted(potential_connections, key=lambda x: (-x[1], x[2]))
        
        degree_connections_made = 0
        for cand_node, _, _ in sorted_by_degree:
            if degree_connections_made >= n_degree_connections:
                break
            if not G.has_edge(g_orphan, cand_node):
                G.add_edge(g_orphan, cand_node)
                if g_orphan == 'ULTIB':
                    raise Exception(f"ULTIB ({g_orphan}) is a goal orphan and an edge was added from it to {cand_node}")
                # print(f"Added edge (degree-based) from goal-orphan {g_orphan} to {cand_node}")
                degree_connections_made += 1
        
        # Nearest-based connections
        # Sort by distance (asc), then degree (desc)
        sorted_by_distance = sorted(potential_connections, key=lambda x: (x[2], -x[1]))
        
        nearest_connections_made = 0
        for cand_node, _, _ in sorted_by_distance:
            if nearest_connections_made >= n_nearest_connections:
                break
            if not G.has_edge(g_orphan, cand_node): # Check again, could have been added by degree
                G.add_edge(g_orphan, cand_node)
                if g_orphan == 'ULTIB':
                    raise Exception(f"ULTIB ({g_orphan}) is a goal orphan and an edge was added from it to {cand_node}")
                # print(f"Added edge (nearest-based) from goal-orphan {g_orphan} to {cand_node}")
                nearest_connections_made += 1
                
    # 3. Repeat the process for source-orphan nodes
    for s_orphan in source_orphans_set:
        try:
            s_orphan_attrs = G.nodes[s_orphan]
            s_orphan_lat, s_orphan_lon = s_orphan_attrs['lat'], s_orphan_attrs['lon']
        except KeyError:
            print(f"Warning: Source orphan {s_orphan} missing lat/lon attributes or not in G. Skipping.")
            continue

        potential_connections = []
        for candidate_node in non_source_orphans:
            if candidate_node == s_orphan:
                continue
            try:
                cand_attrs = G.nodes[candidate_node]
                cand_lat, cand_lon = cand_attrs['lat'], cand_attrs['lon']
            except KeyError:
                continue

            # Distance from candidate to source_orphan
            dist_nm = haversine_distance_nm(cand_lat, cand_lon, s_orphan_lat, s_orphan_lon)

            if dist_nm <= radius_nm:
                # Bearing from candidate to source_orphan
                bearing_from_cand = calculate_initial_bearing(cand_lat, cand_lon, s_orphan_lat, s_orphan_lon)
                diff = bearing_difference(main_bearing, bearing_from_cand)

                if abs(diff) <= effective_tolerance_deg:
                    potential_connections.append((candidate_node, G.degree(candidate_node), dist_nm))
        
        # Degree-based connections
        # Sort by degree (desc), then distance (asc)
        sorted_by_degree = sorted(potential_connections, key=lambda x: (-x[1], x[2]))
        
        degree_connections_made = 0
        for source_node, _, _ in sorted_by_degree: # source_node is candidate_node
            if degree_connections_made >= n_degree_connections:
                break
            if not G.has_edge(source_node, s_orphan):
                G.add_edge(source_node, s_orphan)
                degree_connections_made += 1
        
        # Nearest-based connections
        # Sort by distance (asc), then degree (desc)
        sorted_by_distance = sorted(potential_connections, key=lambda x: (x[2], -x[1]))
        
        nearest_connections_made = 0
        for source_node, _, _ in sorted_by_distance: # source_node is candidate_node
            if nearest_connections_made >= n_nearest_connections:
                break
            if not G.has_edge(source_node, s_orphan): # Check again
                G.add_edge(source_node, s_orphan)
                nearest_connections_made += 1
                
    return G

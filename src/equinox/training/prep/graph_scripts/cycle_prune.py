import networkx as nx
import math # For bearing calculations

# Helper function to calculate bearing between two lat/lon points
def calculate_bearing(lat1_deg, lon1_deg, lat2_deg, lon2_deg):
    """Calculates the bearing from point 1 to point 2.

    Args:
        lat1_deg (float): Latitude of point 1 in degrees.
        lon1_deg (float): Longitude of point 1 in degrees.
        lat2_deg (float): Latitude of point 2 in degrees.
        lon2_deg (float): Longitude of point 2 in degrees.

    Returns:
        float: Bearing in degrees from 0 to 360.
    """
    lat1_rad = math.radians(lat1_deg)
    lon1_rad = math.radians(lon1_deg)
    lat2_rad = math.radians(lat2_deg)
    lon2_rad = math.radians(lon2_deg)

    delta_lon_rad = lon2_rad - lon1_rad

    y = math.sin(delta_lon_rad) * math.cos(lat2_rad)
    x = math.cos(lat1_rad) * math.sin(lat2_rad) - \
        math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon_rad)
    
    bearing_rad = math.atan2(y, x)
    bearing_deg = (math.degrees(bearing_rad) + 360) % 360  # Normalize to 0-360
    return bearing_deg

# Helper function to check if an edge's bearing is reverse to the main bearing
def is_reverse_direction(edge_bearing, main_bearing_degrees):
    """Checks if the edge_bearing is in the reverse direction to main_bearing_degrees.
    Reverse is defined as an angular difference greater than 90 degrees and less than 270 degrees
    when comparing edge_bearing relative to main_bearing_degrees.

    Args:
        edge_bearing (float): Bearing of the edge in degrees.
        main_bearing_degrees (float): Main bearing in degrees.

    Returns:
        bool: True if the edge is in the reverse direction, False otherwise.
    """
    # Calculate the difference in bearing, normalized to [0, 360)
    delta_bearing = (edge_bearing - main_bearing_degrees + 360.0) % 360.0
    # An edge is "reverse" if its direction is more than 90 degrees away from the main bearing,
    # up to 270 degrees. (e.g. main=0, reverse is >90 and <270)
    return delta_bearing > 90.0 and delta_bearing < 270.0

def make_acyclic_by_bearing_and_degree(graph, main_bearing_degrees):
    """
    Makes a directed graph acyclic by iteratively finding cycles
    and removing an edge based on bearing and node degrees.
    Nodes must have 'lat' and 'lon' attributes for bearing calculations.

    Args:
        graph (nx.DiGraph): The input directed graph. Nodes are expected to have
                            'lat' and 'lon' attributes (float values in degrees).
        main_bearing_degrees (float): The overall desired bearing (e.g., from a
                                      global source to a global destination) in degrees.

    Returns:
        nx.DiGraph: An acyclic version of the graph.
    """
    acyclic_graph = graph.copy()
    iteration_count = 0 # Safety counter for debugging complex scenarios

    while True:
        iteration_count += 1
        if iteration_count > len(graph) * len(graph): # Basic safety break for very complex graphs
            print("Warning: Exceeded maximum iterations, potential complex issue or non-convergence. Breaking.")
            break
        try:
            # Find a cycle. orientation='original' gives list of (u,v) edges.
            cycle_edges_list = nx.find_cycle(acyclic_graph, orientation='original')
            if not cycle_edges_list: # Should be caught by NetworkXNoCycle, but defensive
                break

            edge_to_remove_by_rule1 = None
            first_reverse_edge_bearing = None # For logging

            # Rule 1: Check for edges in reverse direction
            for u, v, orient in cycle_edges_list:
                if not acyclic_graph.has_edge(u,v):
                    # This should ideally not happen if find_cycle is on the current graph state
                    print(f"Warning: Cycle edge ({u},{v}) from find_cycle not found in current graph. Skipping.")
                    continue

                # Check for lat/lon attributes
                node_u_data = acyclic_graph.nodes.get(u, {})
                node_v_data = acyclic_graph.nodes.get(v, {})

                if not (isinstance(node_u_data.get('lat'), (int, float)) and
                        isinstance(node_u_data.get('lon'), (int, float)) and
                        isinstance(node_v_data.get('lat'), (int, float)) and
                        isinstance(node_v_data.get('lon'), (int, float))):
                    print(f"Warning: Node(s) in edge ({u},{v}) from cycle missing valid lat/lon attributes. Cannot calculate bearing for Rule 1.")
                else:
                    u_lat, u_lon = node_u_data['lat'], node_u_data['lon']
                    v_lat, v_lon = node_v_data['lat'], node_v_data['lon']
                    current_edge_bearing = calculate_bearing(u_lat, u_lon, v_lat, v_lon)

                    if is_reverse_direction(current_edge_bearing, main_bearing_degrees):
                        if edge_to_remove_by_rule1 is None: # Pick the first reverse edge encountered
                            edge_to_remove_by_rule1 = (u, v)
                            first_reverse_edge_bearing = current_edge_bearing
                        # If we want to break immediately after finding the first reverse edge:
                        # break 
            
            final_edge_to_remove = None
            removal_reason = ""

            if edge_to_remove_by_rule1:
                final_edge_to_remove = edge_to_remove_by_rule1
                removal_reason = (f"reverse bearing (edge bearing {first_reverse_edge_bearing:.1f}° "
                                  f"vs main bearing {main_bearing_degrees:.1f}°)")
            else:
                # Rule 2: If no reverse edges, remove edge connecting nodes with smallest sum of degrees
                min_sum_degrees = float('inf')
                edge_for_min_sum_degrees = None

                if not cycle_edges_list: # Should not happen here
                    print("Warning: cycle_edges_list empty before Rule 2. Breaking.")
                    break
                
                for u, v, orient in cycle_edges_list:
                    if not acyclic_graph.has_edge(u,v): continue # Defensive

                    # Node degrees (sum of in and out for DiGraph)
                    if not (acyclic_graph.has_node(u) and acyclic_graph.has_node(v)):
                         print(f"Warning: Node(s) in cycle edge ({u},{v}) not in graph for degree calculation. Skipping.")
                         continue
                    
                    degree_u = acyclic_graph.degree[u]
                    degree_v = acyclic_graph.degree[v]
                    current_sum_degrees = degree_u + degree_v
                    
                    if edge_for_min_sum_degrees is None or current_sum_degrees < min_sum_degrees:
                        min_sum_degrees = current_sum_degrees
                        edge_for_min_sum_degrees = (u, v)
                    # If sums are equal, first one encountered in cycle_edges_list is kept.

                if edge_for_min_sum_degrees:
                    final_edge_to_remove = edge_for_min_sum_degrees
                    removal_reason = f"smallest sum of degrees ({min_sum_degrees})"
                else:
                    # Fallback: If Rule 1 & Rule 2 failed (e.g., cycle_edges_list empty or all edges had issues)
                    if cycle_edges_list:
                        print(f"Warning: Rule 1 and Rule 2 failed to select an edge for cycle {cycle_edges_list}. Using fallback (first edge).")
                        # Ensure the first edge still exists.
                        u_fb, v_fb = cycle_edges_list[0]
                        if acyclic_graph.has_edge(u_fb, v_fb):
                            final_edge_to_remove = (u_fb, v_fb)
                            removal_reason = "fallback (first edge of cycle)"
                        else:
                             print(f"Fallback failed: first edge {cycle_edges_list[0]} no longer in graph. Breaking.")
                             break # Cannot proceed with this cycle
                    else:
                        print("Warning: No cycle edges list available for fallback. Breaking.")
                        break # No cycle to process

            # Perform removal
            if final_edge_to_remove:
                u_rem, v_rem = final_edge_to_remove
                if acyclic_graph.has_edge(u_rem, v_rem):
                    print(f"Cycle found (length {len(cycle_edges_list)}). Path: {cycle_edges_list if len(cycle_edges_list) < 10 else str(cycle_edges_list[:5]) + '...'}")
                    print(f"Removing edge: ({u_rem},{v_rem}) due to: {removal_reason}")
                    acyclic_graph.remove_edge(u_rem, v_rem)
                else:
                    print(f"Error: Selected edge ({u_rem},{v_rem}) for removal not found in graph. State: {removal_reason}. Breaking.")
                    break 
            else:
                # No edge selected for removal, but a cycle was detected (cycle_edges_list was not empty).
                if cycle_edges_list: 
                     print(f"Critical: Cycle {cycle_edges_list if len(cycle_edges_list) < 10 else str(cycle_edges_list[:5]) + '...'} detected, but no edge was chosen for removal. Breaking.")
                break # Break from while loop

        except nx.NetworkXNoCycle:
            print("No more cycles found.")
            break # Normal termination
        
        if not acyclic_graph.edges():
            print("Graph has no edges left.")
            break

    return acyclic_graph

# Example Usage (commented out as it requires a graph with lat/lon attributes):
#
# G_cyclic = nx.DiGraph()
# # Add nodes with lat/lon attributes
# G_cyclic.add_node('A', lat=40.7128, lon=-74.0060) # New York
# G_cyclic.add_node('B', lat=34.0522, lon=-118.2437) # Los Angeles
# G_cyclic.add_node('C', lat=41.8781, lon=-87.6298) # Chicago
# G_cyclic.add_node('D', lat=30.2672, lon=-97.7431) # Austin

# # Add edges to create a cycle
# G_cyclic.add_edge('A', 'B') # NY to LA (West-ish)
# G_cyclic.add_edge('B', 'C') # LA to Chicago (NorthEast-ish)
# G_cyclic.add_edge('C', 'A') # Chicago to NY (East-SouthEast-ish)
# G_cyclic.add_edge('C', 'D') # Chicago to Austin (South-ish)


# print("Original graph degrees:", {n: G_cyclic.degree[n] for n in G_cyclic.nodes()})
# print("Original graph edges:", G_cyclic.edges())

# # Assume main bearing is East (90 degrees)
# main_bearing = 90.0 
# G_acyclic = make_acyclic_by_bearing_and_degree(G_cyclic, main_bearing)

# print("Acyclic graph edges:", G_acyclic.edges(data=True))
# try:
#     cycle = nx.find_cycle(G_acyclic, orientation='original')
#     print(f"Error: Graph still has cycles: {cycle}")
# except nx.NetworkXNoCycle:
#     print("Verified: The new graph is acyclic.")

# Another example:
# G_test = nx.DiGraph()
# G_test.add_node(1, lat=0, lon=0)
# G_test.add_node(2, lat=1, lon=0) # North of 1
# G_test.add_node(3, lat=0, lon=1) # East of 1
# G_test.add_node(4, lat=-1, lon=0) # South of 1

# G_test.add_edge(1,2) # N (0 deg)
# G_test.add_edge(2,3) # SE (135 deg)
# G_test.add_edge(3,1) # W (270 deg)

# print("Original Test Graph Edges:", G_test.edges())
# Bearing from 1 to 3 is East (90 deg). Let's set main_bearing to 90
# G_test_acyclic = make_acyclic_by_bearing_and_degree(G_test, 90.0)
# print("Acyclic Test Graph Edges:", G_test_acyclic.edges())
# try:
#     nx.find_cycle(G_test_acyclic, orientation='original')
#     print("Verified: Test graph is acyclic.")
# except nx.NetworkXException as e:
#     print(f"Error in test graph: {e}")
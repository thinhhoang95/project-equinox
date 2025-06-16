import math
# Assuming networkx is installed and imported as nx in your environment
# import networkx as nx

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points
    on the earth (specified in decimal degrees) using the Haversine formula.

    Args:
        lat1 (float): Latitude of the first point.
        lon1 (float): Longitude of the first point.
        lat2 (float): Latitude of the second point.
        lon2 (float): Longitude of the second point.

    Returns:
        float: Distance in kilometers.
    """
    # Convert decimal degrees to radians
    lon1_rad, lat1_rad, lon2_rad, lat2_rad = map(math.radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.asin(math.sqrt(a))
    # Radius of earth in kilometers. Use 6371 for kilometers
    r = 6371
    return c * r

def create_airport_connections(airport_code, airport_lat, airport_lon, graph, distance_threshold_km=100):
    """
    Adds edges between a specified airport node and all waypoint nodes
    in the graph that are within a given distance threshold.

    Assumes waypoint nodes in the graph have 'lat' and 'lon' attributes.
    The airport node is added if it doesn't exist. Edges are added with
    the calculated distance as the 'weight' attribute.

    Args:
        airport_code (str): The identifier for the airport node (e.g., 'KJFK').
        airport_lat (float): Latitude of the airport.
        airport_lon (float): Longitude of the airport.
        graph (nx.Graph): The NetworkX graph containing waypoint nodes.
                          This graph will be modified in place.
        distance_threshold_km (float, optional): The maximum distance (in km)
                                                 for creating an edge. Defaults to 100.

    Returns:
        None: The input graph is modified directly.
    """
    # Ensure the airport node exists in the graph, adding it if necessary
    if airport_code not in graph:
        # Add node with its coordinates and specify its type
        graph.add_node(airport_code, lat=airport_lat, lon=airport_lon, type='airport')
    elif 'lat' not in graph.nodes[airport_code] or 'lon' not in graph.nodes[airport_code]:
        # Update existing node if it's missing coordinates
        graph.nodes[airport_code]['lat'] = airport_lat
        graph.nodes[airport_code]['lon'] = airport_lon
        if 'type' not in graph.nodes[airport_code]:
             graph.nodes[airport_code]['type'] = 'airport'

    # Iterate through waypoint nodes in the graph
    for wp_node, data in graph.nodes(data=True):
        # Skip self-connections or connections to other nodes marked as airports
        if wp_node == airport_code or data.get('type') == 'airport':
            continue

        # Ensure waypoint node has coordinate data
        if 'lat' not in data or 'lon' not in data:
            # print(f"Warning: Node {wp_node} is missing 'lat' or 'lon' attribute. Skipping.")
            continue

        wp_lat = data['lat']
        wp_lon = data['lon']

        # Calculate distance from airport to waypoint
        distance = haversine(airport_lat, airport_lon, wp_lat, wp_lon)

        # Add an edge if the waypoint is within the threshold distance
        if distance <= distance_threshold_km:
            graph.add_edge(airport_code, wp_node, weight=distance)

# Example Usage (requires networkx library):
#
# import networkx as nx
#
# Create an example graph (like G or Gm from your context)
# G = nx.Graph()
# G.add_node('WP1', lat=40.7128, lon=-74.0060) # Waypoint near NYC
# G.add_node('WP2', lat=40.8128, lon=-74.1060) # Another waypoint near NYC
# G.add_node('WP3', lat=34.0522, lon=-118.2437) # Waypoint near LAX (far)
# G.add_node('WAYPOINT_NO_COORDS') # Node without coordinates
#
# Define airport details
# jfk_code = 'KJFK'
# jfk_lat, jfk_lon = 40.6413, -73.7781
#
# print("Graph before adding connections:")
# print(f"Nodes: {G.nodes(data=True)}")
# print(f"Edges: {G.edges()}")
#
# Call the function to add connections for JFK within 100km
# create_airport_connections(jfk_code, jfk_lat, jfk_lon, G, distance_threshold_km=100)
#
# print("\nGraph after adding connections for KJFK:")
# print(f"Nodes: {G.nodes(data=True)}") # Node KJFK should now exist with attributes
# print(f"Edges: {G.edges(data=True)}") # Edges between KJFK and WP1, WP2 should exist
#
# Example for another airport
# lax_code = 'KLAX'
# lax_lat, lax_lon = 33.9416, -118.4085
# create_airport_connections(lax_code, lax_lat, lax_lon, G, distance_threshold_km=100)
#
# print("\nGraph after adding connections for KLAX:")
# print(f"Nodes: {G.nodes(data=True)}")
# print(f"Edges: {G.edges(data=True)}") # Edge between KLAX and WP3 should exist
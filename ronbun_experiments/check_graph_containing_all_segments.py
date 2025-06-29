import networkx as nx
import pandas as pd
from typing import List, Tuple, Set

def parse_route_to_segments(route_waypoints: str) -> List[Tuple[str, str]]:
    """
    Parse a route string into consecutive waypoint segments.
    
    Args:
        route_waypoints: Space-separated waypoint string like "LGAV LGMG AMUGO SOMIG_73 DINOX"
        
    Returns:
        List of tuples representing segments: [("LGAV", "LGMG"), ("LGMG", "AMUGO"), ...]
    """
    waypoints = route_waypoints.strip().split()
    segments = []
    
    for i in range(len(waypoints) - 1):
        segments.append((waypoints[i], waypoints[i + 1]))
    
    return segments

def load_graph_edges(graph_path: str) -> Set[Tuple[str, str]]:
    """
    Load graph from GML file and extract all edges as (source, target) tuples.
    
    Args:
        graph_path: Path to the GML graph file
        
    Returns:
        Set of (source, target) tuples representing edges in the graph
    """
    G = nx.read_gml(graph_path)
    edges = set()
    
    for source, target in G.edges():
        # Get node labels instead of IDs
        source_label = G.nodes[source].get('label', str(source))
        target_label = G.nodes[target].get('label', str(target))
        edges.add((source_label, target_label))
    
    return edges

def load_graph_nodes(graph_path: str) -> Set[str]:
    """
    Load graph from GML file and extract all node labels.
    
    Args:
        graph_path: Path to the GML graph file
        
    Returns:
        Set of node labels in the graph
    """
    G = nx.read_gml(graph_path)
    nodes = set()
    
    for node_id in G.nodes():
        # Get node label instead of ID
        node_label = G.nodes[node_id].get('label', str(node_id))
        nodes.add(node_label)
    
    return nodes

def parse_route_to_nodes(route_waypoints: str) -> List[str]:
    """
    Parse a route string into individual waypoint nodes.
    
    Args:
        route_waypoints: Space-separated waypoint string like "LGAV LGMG AMUGO SOMIG_73 DINOX"
        
    Returns:
        List of waypoint nodes
    """
    return route_waypoints.strip().split()

def check_nodes_in_graph(nodes: List[str], graph_nodes: Set[str]) -> Tuple[List[str], List[str]]:
    """
    Check which nodes exist in the graph.
    
    Args:
        nodes: List of waypoint nodes to check
        graph_nodes: Set of node labels in the graph
        
    Returns:
        Tuple of (found_nodes, missing_nodes)
    """
    found_nodes = []
    missing_nodes = []
    
    for node in nodes:
        if node in graph_nodes:
            found_nodes.append(node)
        else:
            missing_nodes.append(node)
    
    return found_nodes, missing_nodes

def check_segments_in_graph(segments: List[Tuple[str, str]], graph_edges: Set[Tuple[str, str]]) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Check which segments exist in the graph.
    
    Args:
        segments: List of (source, target) segments to check
        graph_edges: Set of (source, target) edges in the graph
        
    Returns:
        Tuple of (found_segments, missing_segments)
    """
    found_segments = []
    missing_segments = []
    
    for segment in segments:
        if segment in graph_edges:
            found_segments.append(segment)
        else:
            missing_segments.append(segment)
    
    return found_segments, missing_segments

def check_all_routes_against_graph(csv_path: str, graph_path: str) -> dict:
    """
    Check all routes in CSV file against the graph.
    
    Args:
        csv_path: Path to the all_routes.csv file
        graph_path: Path to the routes.gml graph file
        
    Returns:
        Dictionary with results for each route
    """
    # Load the graph edges and nodes
    print(f"Loading graph from {graph_path}...")
    graph_edges = load_graph_edges(graph_path)
    graph_nodes = load_graph_nodes(graph_path)
    print(f"Loaded {len(graph_edges)} edges and {len(graph_nodes)} nodes from graph")
    
    # Load the routes CSV
    print(f"Loading routes from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    results = {}
    total_routes = len(df)
    
    for idx, row in df.iterrows():
        flight_id = row['flight_id']
        route_waypoints = row['real_waypoints']
        
        # Parse route into segments and nodes
        segments = parse_route_to_segments(route_waypoints)
        nodes = parse_route_to_nodes(route_waypoints)
        
        # Check segments and nodes against graph
        found_segments, missing_segments = check_segments_in_graph(segments, graph_edges)
        found_nodes, missing_nodes = check_nodes_in_graph(nodes, graph_nodes)
        
        results[flight_id] = {
            'route': route_waypoints,
            'total_segments': len(segments),
            'found_segments': found_segments,
            'missing_segments': missing_segments,
            'segments_found_count': len(found_segments),
            'segments_missing_count': len(missing_segments),
            'all_segments_found': len(missing_segments) == 0,
            'total_nodes': len(nodes),
            'found_nodes': found_nodes,
            'missing_nodes': missing_nodes,
            'nodes_found_count': len(found_nodes),
            'nodes_missing_count': len(missing_nodes),
            'all_nodes_found': len(missing_nodes) == 0
        }
        
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{total_routes} routes")
    
    return results

def print_summary(results: dict):
    """
    Print a summary of the results.
    """
    total_routes = len(results)
    routes_with_all_segments = sum(1 for r in results.values() if r['all_segments_found'])
    routes_with_missing_segments = total_routes - routes_with_all_segments
    routes_with_all_nodes = sum(1 for r in results.values() if r['all_nodes_found'])
    routes_with_missing_nodes = total_routes - routes_with_all_nodes
    
    print("\n=== SUMMARY ===")
    print(f"Total routes checked: {total_routes}")
    print(f"Routes with all segments found: {routes_with_all_segments} ({routes_with_all_segments/total_routes*100:.1f}%)")
    print(f"Routes with missing segments: {routes_with_missing_segments} ({routes_with_missing_segments/total_routes*100:.1f}%)")
    print(f"Routes with all nodes found: {routes_with_all_nodes} ({routes_with_all_nodes/total_routes*100:.1f}%)")
    print(f"Routes with missing nodes: {routes_with_missing_nodes} ({routes_with_missing_nodes/total_routes*100:.1f}%)")
    
    if routes_with_missing_segments > 0:
        print("\n=== ROUTES WITH MISSING SEGMENTS ===")
        for flight_id, result in results.items():
            if not result['all_segments_found']:
                print(f"\nFlight {flight_id}:")
                print(f"  Route: {result['route']}")
                print(f"  Missing segments ({result['segments_missing_count']}/{result['total_segments']}):")
                for segment in result['missing_segments']:
                    print(f"    {segment[0]} -> {segment[1]}")
    
    if routes_with_missing_nodes > 0:
        print("\n=== ROUTES WITH MISSING NODES ===")
        for flight_id, result in results.items():
            if not result['all_nodes_found']:
                print(f"\nFlight {flight_id}:")
                print(f"  Route: {result['route']}")
                print(f"  Missing nodes ({result['nodes_missing_count']}/{result['total_nodes']}):")
                for node in result['missing_nodes']:
                    print(f"    {node}")

if __name__ == "__main__":
    # Define file paths
    csv_path = "/Volumes/CrucialX/project-equinox/data/cases/LGAV_LFPG/all_routes.csv"
    graph_path = "/Volumes/CrucialX/project-equinox/data/cases/LGAV_LFPG/graphs/routes.gml"
    
    # Run the analysis
    results = check_all_routes_against_graph(csv_path, graph_path)
    
    # Print summary
    print_summary(results)
    
    # Example: Check a specific route
    example_route = "LGAV LGMG AMUGO SOMIG_73 DINOX"
    print(f"\n=== EXAMPLE: {example_route} ===")
    segments = parse_route_to_segments(example_route)
    nodes = parse_route_to_nodes(example_route)
    print(f"Segments: {segments}")
    print(f"Nodes: {nodes}")
    
    graph_edges = load_graph_edges(graph_path)
    graph_nodes = load_graph_nodes(graph_path)
    found_segments, missing_segments = check_segments_in_graph(segments, graph_edges)
    found_nodes, missing_nodes = check_nodes_in_graph(nodes, graph_nodes)
    print(f"Found segments: {found_segments}")
    print(f"Missing segments: {missing_segments}")
    print(f"Found nodes: {found_nodes}")
    print(f"Missing nodes: {missing_nodes}")

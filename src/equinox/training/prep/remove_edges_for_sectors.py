import os
import networkx as nx
import geopandas as gpd
from shapely.geometry import Point, LineString

def remove_edges_through_sectors(
    G: nx.DiGraph,
    sectors_to_exclude: list,
    sectors_geojson_path: str,
    output_dir: str = None,
) -> tuple[nx.DiGraph, gpd.GeoDataFrame]:
    """
    Removes edges from a graph that intersect with one or more of the specified sectors.

    An edge is removed if it crosses, is contained within, or has at least
    one endpoint inside an excluded sector. In other words, any edge that
    intersects with the geometry of an excluded sector is removed.

    Args:
        G (nx.DiGraph): The input graph. Nodes are expected to have 'lon' and 'lat'
                        attributes.
        sectors_to_exclude (list): A list of sector IDs to exclude.
        sectors_geojson_path (str): Path to the GeoJSON file with sector definitions.
        output_dir (str, optional): If provided, the modified graph is saved as GML
                                    to this directory. Defaults to None.

    Returns:
        tuple[nx.DiGraph, gpd.GeoDataFrame]: A tuple containing:
            - nx.DiGraph: A new graph with the specified edges removed.
            - gpd.GeoDataFrame: A GeoDataFrame with the polygons of the excluded sectors.
    """
    all_sectors = gpd.read_file(sectors_geojson_path)
    excluded_sectors_gdf = all_sectors[
        all_sectors["sector_id"].isin(sectors_to_exclude)
    ]

    if excluded_sectors_gdf.empty:
        print("Warning: No sectors to exclude were found in the GeoJSON file.")
        if output_dir:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            nx.write_gml(G, os.path.join(output_dir, "graph_with_no_removed_edges.gml"))
        return G.copy(), excluded_sectors_gdf

    excluded_polygons = excluded_sectors_gdf.union_all()

    edges_to_remove = []
    G_copy = G.copy()

    for u, v in G_copy.edges():
        try:
            pos_u = (G_copy.nodes[u]["lon"], G_copy.nodes[u]["lat"])
            pos_v = (G_copy.nodes[v]["lon"], G_copy.nodes[v]["lat"])
        except KeyError:
            print(f"Warning: Skipping edge ({u}, {v}) due to missing 'lon' or 'lat' attribute.")
            continue

        edge_line = LineString([pos_u, pos_v])

        # An edge is removed if it intersects with the excluded polygons in any way.
        if edge_line.intersects(excluded_polygons):
            edges_to_remove.append((u, v))

    G_copy.remove_edges_from(edges_to_remove)
    print(f"Removed {len(edges_to_remove)} edges.")

    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, "graph_after_edge_removal.gml")
        nx.write_gml(G_copy, output_path)
        print(f"Saved modified graph to {output_path}")

    return G_copy, excluded_sectors_gdf

if __name__ == "__main__":
    # Create a sample graph for demonstration
    G = nx.read_gml('scenarios/routes.gml')

    print(f"Original number of edges: {G.number_of_edges()}")

    # Define sectors to exclude and path to geojson
    sectors_to_exclude = ["LFBBZ3"]
    # Assuming the script is run from the root of project-collarfall
    geojson_path = "crida-data/Airspace/sectors.geojson"
    output_directory = "output"

    # Run the function
    if os.path.exists(geojson_path):
        modified_G, excluded_polygons_gdf = remove_edges_through_sectors(
            G, sectors_to_exclude, geojson_path, output_dir=output_directory
        )
        print(f"Number of edges after removal: {modified_G.number_of_edges()}")
        print(f"Returned {len(excluded_polygons_gdf)} excluded sector polygons.")
        print("\nEdges in the original graph:")
        for edge in G.edges():
            print(edge)
        
        print("\nEdges in the modified graph:")
        for edge in modified_G.edges():
            print(edge)
    else:
        print(f"Error: Could not find {geojson_path}.")
        print("Please ensure the path is correct and you are running the script from the project root.")

import matplotlib.pyplot as plt
import networkx as nx
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def plot_routes_on_map(graph, routes, ax=None, show_waypoints=True, route_alpha=1.0, thickness=2):
    """
    Plots waypoints and routes on a map using Cartopy.

    Args:
        graph (nx.Graph): A NetworkX graph where nodes have 'lat' and 'lon' attributes
                          and are identified by their names (e.g., waypoint IDs).
        routes (list[list[str]]): A list of routes, where each route is a list of waypoint names.
        ax (matplotlib.axes.Axes, optional): A Matplotlib Axes object to plot on.
                                             If None, a new figure and axes will be created.
        show_waypoints (bool, optional): Whether to plot waypoint markers and labels.
        route_alpha (float, optional): Opacity for route line plotting.
    """
    if ax is None:
        fig = plt.figure(figsize=(48, 16))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    # ax.stock_img()
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    # ax.add_feature(cfeature.LAND)
    # ax.add_feature(cfeature.OCEAN)
    # ax.add_feature(cfeature.LAKES)
    # ax.add_feature(cfeature.RIVERS)

    # Collect all coordinates for extent calculation
    all_lons = []
    all_lats = []

    # 1. Plot all the nodes on the map with text label of the waypoint name (small font)
    if show_waypoints:
        for node, data in graph.nodes(data=True):
            lon, lat = data['lon'], data['lat']
            all_lons.append(lon)
            all_lats.append(lat)
            ax.plot(lon, lat, 'o', color='blue', markersize=3, transform=ccrs.Geodetic())
            ax.text(lon + 0.01, lat + 0.01, str(node), fontsize=6, transform=ccrs.Geodetic())

    # 3. Plot the routes in thick lines
    for route_idx, route in enumerate(routes):
        route_lons = []
        route_lats = []
        for waypoint_name in route:
            if waypoint_name in graph.nodes:
                node_data = graph.nodes[waypoint_name]
                route_lons.append(node_data['lon'])
                route_lats.append(node_data['lat'])
                all_lons.append(node_data['lon']) # also consider route points for extent
                all_lats.append(node_data['lat'])
            else:
                print(f"Warning: Waypoint '{waypoint_name}' in route {route_idx} not found in graph.")
        
        if route_lons and route_lats: # Ensure there are points to plot
            ax.plot(
                route_lons,
                route_lats,
                '-',
                linewidth=thickness,
                alpha=route_alpha,
                transform=ccrs.Geodetic(),
                label=f'Route {route_idx+1}',
            )

    # Set map extent
    if all_lons and all_lats:
        buffer = 1.0 # Degree buffer around min/max coordinates
        min_lon, max_lon = min(all_lons) - buffer, max(all_lons) + buffer
        min_lat, max_lat = min(all_lats) - buffer, max(all_lats) + buffer
        ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
    
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    
    if ax is None: # Only call plt.show() if we created the figure
        plt.legend()
        plt.show()

def plot_waypoints(graph, highlighted_nodes=None, ax=None):
    """
    Plots waypoints on a map using Cartopy, with specified nodes highlighted in red.

    Args:
        graph (nx.Graph): A NetworkX graph where nodes have 'lat' and 'lon' attributes
                          and are identified by their names (e.g., waypoint IDs).
        highlighted_nodes (list[str], optional): A list of node names to highlight in red.
                                                If None, no nodes will be highlighted.
        ax (matplotlib.axes.Axes, optional): A Matplotlib Axes object to plot on.
                                             If None, a new figure and axes will be created.
    """
    if ax is None:
        fig = plt.figure(figsize=(48, 16))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    # ax.stock_img()
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.LAKES)
    ax.add_feature(cfeature.RIVERS)

    # Collect all coordinates for extent calculation
    all_lons = []
    all_lats = []

    # Set default highlighted_nodes to empty list if None
    if highlighted_nodes is None:
        highlighted_nodes = []

    # Plot all the nodes on the map with text label of the waypoint name (small font)
    for node, data in graph.nodes(data=True):
        lon, lat = data['lon'], data['lat']
        all_lons.append(lon)
        all_lats.append(lat)
        
        # Choose color based on whether node is highlighted
        color = 'red' if node in highlighted_nodes else 'blue'
        ax.plot(lon, lat, 'o', color=color, markersize=3, transform=ccrs.Geodetic())
        ax.text(lon + 0.01, lat + 0.01, str(node), fontsize=6, transform=ccrs.Geodetic())

    # Set map extent
    if all_lons and all_lats:
        buffer = 1.0 # Degree buffer around min/max coordinates
        min_lon, max_lon = min(all_lons) - buffer, max(all_lons) + buffer
        min_lat, max_lat = min(all_lats) - buffer, max(all_lats) + buffer
        ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
    
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    
    if ax is None: # Only call plt.show() if we created the figure
        plt.show()


import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature
import numpy as np

def plot_route_graph_pdf(Gm, show_label=False, highlighted_labels = [], output_path=None,
                         origin_node="LEMD", destination_node="EGLL"):
    # Assuming Gm is your NetworkX graph
    # Each node in Gm has 'lat' and 'lon' attributes

    # Extract latitudes and longitudes
    lats = []
    lons = []
    ids = []
    for node, data in Gm.nodes(data=True):
        lats.append(data['lat'])
        lons.append(data['lon'])
        ids.append(node)

    # Calculate the bounds with some padding
    min_lon, max_lon = min(lons) - 2, max(lons) + 2
    min_lat, max_lat = min(lats) - 2, max(lats) + 2

    # Set up the map
    if show_label:
        fig = plt.figure(figsize=(40, 56))
    else:
        fig = plt.figure(figsize=(10, 14))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.add_feature(cartopy.feature.LAND)
    ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())

    # Plot the waypoints
    ax.scatter(lons, lats, color='blue', s=1, transform=ccrs.PlateCarree(), label='Waypoints')
    # Add text labels for each waypoint
    if show_label:
        for lon, lat, id_text in zip(lons, lats, ids):
            if id_text in highlighted_labels:
                ax.text(lon, lat, id_text, transform=ccrs.PlateCarree(), fontsize=10, color='blue')
            else:
                ax.text(lon, lat, id_text, transform=ccrs.PlateCarree(), fontsize=10)
    # Plot the edges
    for u, v in Gm.edges():
        lon1, lat1 = Gm.nodes[u]['lon'], Gm.nodes[u]['lat']
        lon2, lat2 = Gm.nodes[v]['lon'], Gm.nodes[v]['lat']
        ax.plot([lon1, lon2], [lat1, lat2], color='gray', linewidth=0.5, 
                alpha=0.3, transform=ccrs.PlateCarree())

    # Add the LEMD and EGLL nodes as stars with text labels
    lemd_lon, lemd_lat = Gm.nodes[origin_node]['lon'], Gm.nodes[origin_node]['lat']
    egll_lon, egll_lat = Gm.nodes[destination_node]['lon'], Gm.nodes[destination_node]['lat']

    # Plot stars for origin and destination
    ax.scatter(lemd_lon, lemd_lat, color='red', s=100, marker='*', transform=ccrs.PlateCarree())
    ax.scatter(egll_lon, egll_lat, color='red', s=100, marker='*', transform=ccrs.PlateCarree())

    # Add text labels for the airports
    ax.text(lemd_lon+0.2, lemd_lat+0.2, origin_node, transform=ccrs.PlateCarree(), fontsize=8)
    ax.text(egll_lon+0.2, egll_lat+0.2, destination_node, transform=ccrs.PlateCarree(), fontsize=8)

    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False

    plt.title(f'Plausible Connections between {origin_node} and {destination_node}')
    if output_path is not None:
        plt.savefig(output_path, format="pdf", bbox_inches="tight")
    else:
        plt.show()


if __name__ == '__main__':
    # Example Usage:
    # Create a sample graph
    G = nx.Graph()
    G.add_node("A", lat=34.0522, lon=-118.2437)  # Los Angeles
    G.add_node("B", lat=36.1699, lon=-115.1398)  # Las Vegas
    G.add_node("C", lat=32.7157, lon=-117.1611)  # San Diego
    G.add_node("D", lat=37.7749, lon=-122.4194)  # San Francisco
    G.add_node("E", lat=40.7128, lon=-74.0060)   # New York (for extent testing)

    # Define sample routes (list of lists of waypoint names)
    example_routes = [
        ["A", "B", "D"],
        ["C", "A", "D"]
    ]
    
    # Create a new figure and axes specifically for the example
    fig_example = plt.figure(figsize=(12, 10))
    ax_example = fig_example.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    plot_routes_on_map(G, example_routes, ax=ax_example)
    
    ax_example.set_title("Route Map Example")
    plt.legend() # Ensure legend is shown for the example
    plt.show()

    # Example with a route containing an unknown waypoint
    G_small = nx.Graph()
    G_small.add_node("X", lat=45.0, lon=-90.0)
    G_small.add_node("Y", lat=46.0, lon=-91.0)
    routes_with_unknown = [
        ["X", "Z", "Y"] # "Z" is not in G_small
    ]
    fig_unknown = plt.figure(figsize=(8,8))
    ax_unknown = fig_unknown.add_subplot(1,1,1, projection=ccrs.PlateCarree())
    plot_routes_on_map(G_small, routes_with_unknown, ax=ax_unknown)
    ax_unknown.set_title("Map with Unknown Waypoint in Route")
    plt.legend()
    plt.show()

    # Example usage of plot_waypoints function
    print("\nExample usage of plot_waypoints function:")
    fig_waypoints = plt.figure(figsize=(12, 10))
    ax_waypoints = fig_waypoints.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    # Highlight nodes A and D in red
    highlighted = ["A", "D"]
    plot_waypoints(G, highlighted_nodes=highlighted, ax=ax_waypoints)
    
    ax_waypoints.set_title("Waypoints Map with Highlighted Nodes (A and D in red)")
    plt.show()


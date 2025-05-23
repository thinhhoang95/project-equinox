import matplotlib.pyplot as plt
import networkx as nx
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def plot_routes_on_map(graph, routes, ax=None):
    """
    Plots waypoints and routes on a map using Cartopy.

    Args:
        graph (nx.Graph): A NetworkX graph where nodes have 'lat' and 'lon' attributes
                          and are identified by their names (e.g., waypoint IDs).
        routes (list[list[str]]): A list of routes, where each route is a list of waypoint names.
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

    # 1. Plot all the nodes on the map with text label of the waypoint name (small font)
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
            ax.plot(route_lons, route_lats, '-', linewidth=2, transform=ccrs.Geodetic(), label=f'Route {route_idx+1}')

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

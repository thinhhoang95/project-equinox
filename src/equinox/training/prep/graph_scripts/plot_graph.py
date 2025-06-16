import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature
import numpy as np

def plot_graph(Gm, show_label=False, highlighted_labels = []):
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
    lemd_lon, lemd_lat = Gm.nodes['LEMD']['lon'], Gm.nodes['LEMD']['lat']
    egll_lon, egll_lat = Gm.nodes['EGLL']['lon'], Gm.nodes['EGLL']['lat']

    # Plot stars for origin and destination
    ax.scatter(lemd_lon, lemd_lat, color='red', s=100, marker='*', transform=ccrs.PlateCarree())
    ax.scatter(egll_lon, egll_lat, color='red', s=100, marker='*', transform=ccrs.PlateCarree())

    # Add text labels for the airports
    ax.text(lemd_lon+0.2, lemd_lat+0.2, 'LEMD', transform=ccrs.PlateCarree(), fontsize=8)
    ax.text(egll_lon+0.2, egll_lat+0.2, 'EGLL', transform=ccrs.PlateCarree(), fontsize=8)

    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False

    plt.title('Plausible Connections between LEMD and EGLL')
    plt.savefig("route_graph_plot.pdf", format="pdf", bbox_inches="tight")
    plt.show()
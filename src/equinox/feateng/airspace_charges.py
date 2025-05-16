import networkx as nx   
import numpy as np
import pandas as pd
import json
from shapely.geometry import LineString, Polygon
from geopy.distance import geodesic
from tqdm import tqdm
from equinox.feateng.laplace import enumerate_nodes, node_names_to_ids, node_ids_to_names

def compute_charges_for_graph(graph: nx.Graph, charges_df: pd.DataFrame) -> nx.Graph:
    
    def calculate_line_length_meters(coords_list_lon_lat):
        # Calculates length of a list of (lon, lat) coordinates in meters
        length_m = 0.0
        for i in range(len(coords_list_lon_lat) - 1):
            p1_lon_lat = coords_list_lon_lat[i]
            p2_lon_lat = coords_list_lon_lat[i+1]
            # geopy.distance.geodesic expects (lat, lon)
            length_m += geodesic((p1_lon_lat[1], p1_lon_lat[0]), (p2_lon_lat[1], p2_lon_lat[0])).m
        return length_m

    def get_geom_length_meters(geom):
        # geom has coordinates as (lon, lat)
        if geom.is_empty:
            return 0.0
        
        total_len_m = 0.0
        if geom.geom_type == 'LineString':
            total_len_m = calculate_line_length_meters(list(geom.coords))
        elif geom.geom_type == 'MultiLineString':
            for line in geom.geoms: # Corrected from geom.geoms() to geom.geoms
                total_len_m += calculate_line_length_meters(list(line.coords))
        elif geom.geom_type == 'GeometryCollection':
            for part_geom in geom.geoms: # Corrected from geom.geoms() to geom.geoms
                if part_geom.geom_type in ['LineString', 'MultiLineString']:
                    total_len_m += get_geom_length_meters(part_geom)
        return total_len_m

    try:
        with open("data/ufir/eurofirs.geojson", 'r') as f:
            fir_geo_data = json.load(f)
    except FileNotFoundError:
        # Consider logging this error
        # print("Error: data/eurofirs.geojson not found.")
        return graph # Or raise an exception
        
    firs_features = fir_geo_data.get('features', [])

    parsed_firs = []
    for fir_feature in firs_features:
        try:
            props = fir_feature['properties']
            geom_data = fir_feature['geometry']
            
            if geom_data['type'] != 'Polygon':
                continue

            exterior_coords_lon_lat = [(pt[0], pt[1]) for pt in geom_data['coordinates'][0]]
            interior_coords_list_lon_lat = []
            if len(geom_data['coordinates']) > 1:
                for interior_ring_lon_lat in geom_data['coordinates'][1:]:
                    interior_coords_list_lon_lat.append([(pt[0], pt[1]) for pt in interior_ring_lon_lat])
            
            fir_poly = Polygon(exterior_coords_lon_lat, holes=interior_coords_list_lon_lat if interior_coords_list_lon_lat else None)
            
            fir_lon1 = props.get('lon_1')
            fir_lat1 = props.get('lat_1')
            fir_lon2 = props.get('lon_2')
            fir_lat2 = props.get('lat_2')

            if not all(isinstance(val, (int, float)) for val in [fir_lon1, fir_lat1, fir_lon2, fir_lat2]):
                b = fir_poly.bounds 
                fir_bounds = b
            else:
                fir_bounds = (min(fir_lon1, fir_lon2), min(fir_lat1, fir_lat2), 
                              max(fir_lon1, fir_lon2), max(fir_lat1, fir_lat2))

            parsed_firs.append({
                'id': props.get('id'), # Use .get for safety
                'polygon': fir_poly,
                'bounds': fir_bounds
            })
        except Exception:
            # Consider logging this error for specific FIR
            # print(f"Error processing FIR feature {props.get('id', 'Unknown')}: {e}. Skipping.")
            continue
            
    charges_df_indexed = None
    if 'designator' in charges_df.columns:
        try:
            charges_df_indexed = charges_df.set_index('designator')
        except Exception: # Fallback if set_index fails (e.g. duplicate designators not allowed in index)
            charges_df_indexed = None # Will use boolean indexing

    for u, v, edge_data in tqdm(graph.edges(data=True), desc="Calculating airspace charges"):
        node_u_data = graph.nodes[u]
        node_v_data = graph.nodes[v]

        wp1_lon, wp1_lat = node_u_data.get('lon'), node_u_data.get('lat')
        wp2_lon, wp2_lat = node_v_data.get('lon'), node_v_data.get('lat')

        if None in [wp1_lon, wp1_lat, wp2_lon, wp2_lat]:
            edge_data['airspace_charge'] = 0.0
            continue
            
        wp1_coords_lon_lat = (wp1_lon, wp1_lat)
        wp2_coords_lon_lat = (wp2_lon, wp2_lat)

        if wp1_coords_lon_lat == wp2_coords_lon_lat:
            edge_data['airspace_charge'] = 0.0
            continue

        edge_linestring = LineString([wp1_coords_lon_lat, wp2_coords_lon_lat])
        total_segment_length_m = get_geom_length_meters(edge_linestring)

        if total_segment_length_m == 0:
            edge_data['airspace_charge'] = 0.0
            continue

        edge_min_lon, edge_min_lat = min(wp1_lon, wp2_lon), min(wp1_lat, wp2_lat)
        edge_max_lon, edge_max_lat = max(wp1_lon, wp2_lon), max(wp1_lat, wp2_lat)

        segment_weighted_charge = 0.0
        
        for fir_info in parsed_firs:
            fir_id = fir_info['id']
            if not fir_id: # Skip if FIR ID is missing
                continue

            fir_polygon = fir_info['polygon']
            fir_min_lon, fir_min_lat, fir_max_lon, fir_max_lat = fir_info['bounds']

            bbox_intersects = not (edge_max_lon < fir_min_lon or \
                                   edge_min_lon > fir_max_lon or \
                                   edge_max_lat < fir_min_lat or \
                                   edge_min_lat > fir_max_lat)

            if not bbox_intersects:
                continue

            if edge_linestring.intersects(fir_polygon):
                intersection_geom = edge_linestring.intersection(fir_polygon)
                length_in_fir_m = get_geom_length_meters(intersection_geom)

                if length_in_fir_m > 0:
                    fir_global_rate = None
                    try:
                        if charges_df_indexed is not None and fir_id in charges_df_indexed.index:
                             fir_global_rate = charges_df_indexed.loc[fir_id, 'global_rate']
                        else: # Fallback to boolean indexing if index not set or ID not found
                             rate_series = charges_df[charges_df['designator'] == fir_id]['global_rate']
                             if not rate_series.empty:
                                 fir_global_rate = rate_series.iloc[0]
                        
                        if fir_global_rate is not None and pd.notna(fir_global_rate):
                            segment_weighted_charge += (length_in_fir_m / total_segment_length_m) * float(fir_global_rate)
                        # else:
                            # print(f"Warning: No/NaN charge rate for FIR {fir_id}")
                    except (KeyError, IndexError):
                        # print(f"Warning: FIR ID {fir_id} not in charges_df or rate missing.")
                        pass # FIR not in charges_df or 'global_rate' column missing
                        
        edge_data['airspace_charge'] = segment_weighted_charge
        
    return graph

if __name__ == "__main__":
    charges_df = pd.read_csv("data/ufir/fir_charges.csv")
    # Example of charges_df:
    # designator,name,national_rate,global_rate
    # EBBU,BRUSSELS FIR,120.49,120.6
    # EDGG,LANGEN FIR,99.91,100.02
    # EDMM,MUNICH FIR,99.91,100.02

    graph_path = "data/graph/LEMD_EGLL_2023_04_01.gml"
    graph = nx.read_gml(graph_path)
    # Nodes: waypoints (with ID, lat, lon)
    # Load the fir_geo object here

    charge_graph = compute_charges_for_graph(graph, charges_df)

    # Print some edge airspace charges to inspect the results
    print("Sample edge airspace charges:")
    count = 0
    for u, v, data in charge_graph.edges(data=True):
        print(f"Edge {u} -> {v}: airspace_charge = {data.get('airspace_charge')}")
        count += 1
        if count >= 10:
            break
     
    # Create a cost matrix from the charge_graph
    node_mapping = enumerate_nodes(charge_graph)
    node_ids = node_names_to_ids(charge_graph, list(charge_graph.nodes()))
    cost_matrix = np.zeros((len(node_ids), len(node_ids)))
    for u, v, data in charge_graph.edges(data=True):
        cost_matrix[node_mapping[u], node_mapping[v]] = data.get('airspace_charge', 0)

    print(cost_matrix[:30, :30])

    # Save the cost matrix to a npy file
    np.save(f"data/graph/{graph_path.split('/')[-1]}_charges.npy", cost_matrix)
    print(f"Saved cost matrix to {graph_path.split('/')[-1]}_charges.npy")


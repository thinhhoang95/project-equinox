import os
import pandas as pd
import numpy as np
import networkx as nx
from equinox.training.prep.prep_graph import process_and_save_graph
from equinox.feateng.airspace_charges import compute_charges_for_graph
from equinox.feateng.laplace import enumerate_nodes, node_names_to_ids
from equinox.feateng.distance import haversine_distance_matrix

path_prefix = "D:\\project-akrav\\"
path_output = "D:\\project-equinox\\"
source_id = "LEMD"
destination_id = "EGLL"
routes_dir = os.path.join(path_prefix, "matched_filtered_data")

def prep_graph():
    nodes_only_graph_path = os.path.join(
        path_prefix, "data", "graphs", "ats_fra_nodes_only.gml"
    )
    routes_dir = os.path.join(path_prefix, "matched_filtered_data")

    import time

    path_prefix_output = path_output
    case_name = f"{source_id}_{destination_id}"
    start_time = time.time()
    output_path = os.path.join(
        path_prefix_output, "data", "cases", case_name, "graphs", f"routes.gml"
    )
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    Gno = process_and_save_graph(
        nodes_only_graph_path,
        source_id,
        destination_id,
        routes_dir,
        output_path=output_path,
        minimum_detour_allowed=0.025,
        n_iter=10,
        max_allowed_deviation_angle=60,
    )
    print(f"Graph processing completed in {time.time() - start_time} seconds")
    print(
        f"Final graph has {Gno.number_of_nodes()} nodes and {Gno.number_of_edges()} edges"
    )


from equinox.training.prep.prep_routes import filter_routes_by_origin_dest


def prep_routes_data():
    # Example: filter flights from Madrid to London Heathrow
    input_directory = routes_dir
    origin_airport = source_id
    destination_airport = destination_id
    output_directory = os.path.join(path_output, "data", "cases")

    filtered_flights = filter_routes_by_origin_dest(
        input_dir=input_directory,
        origin=origin_airport,
        dest=destination_airport,
        output_dir=output_directory,
    )

    print(f"Filtered {len(filtered_flights)} flights")

def prep_airspace_charges():
    charges_df = pd.read_csv("data/ufir/fir_charges.csv")
    # Example of charges_df:
    # designator,name,national_rate,global_rate
    # EBBU,BRUSSELS FIR,120.49,120.6
    # EDGG,LANGEN FIR,99.91,100.02
    # EDMM,MUNICH FIR,99.91,100.02

    case_name = f"{source_id}_{destination_id}"
    graph_path = os.path.join(
        path_output, "data", "cases", case_name, "graphs", "routes.gml"
    )
    graph = nx.read_gml(graph_path)
    # Nodes: waypoints (with ID, lat, lon)

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

    print(f"Cost matrix shape: {cost_matrix.shape}")
    print("Sample cost matrix values:")
    print(cost_matrix[:min(30, cost_matrix.shape[0]), :min(30, cost_matrix.shape[1])])

    # Save the cost matrix to a npy file
    graph_filename = os.path.basename(graph_path).replace('.gml', '')
    output_file = os.path.join(
        os.path.dirname(graph_path), f"{graph_filename}_charges.npy"
    )
    np.save(output_file, cost_matrix)
    print(f"Saved cost matrix to {output_file}")


def prep_distance_matrix():
    """
    Prepare and save the distance matrix for the route graph.
    
    This function loads the route graph, computes the Haversine distance matrix
    using the haversine_distance_matrix function from the distance module,
    and saves it as a .npy file.
    """
    case_name = f"{source_id}_{destination_id}"
    graph_path = os.path.join(
        path_output, "data", "cases", case_name, "graphs", "routes.gml"
    )
    
    print(f"Loading graph from: {graph_path}")
    graph = nx.read_gml(graph_path)
    
    print(f"Computing distance matrix for graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
    dist_matrix = haversine_distance_matrix(graph)
    
    print(f"Distance matrix shape: {dist_matrix.shape}")
    print("Sample distance matrix values (first 5x5):")
    print(dist_matrix[:min(5, dist_matrix.shape[0]), :min(5, dist_matrix.shape[1])])
    
    # Save the distance matrix to a npy file
    graph_filename = os.path.basename(graph_path).replace('.gml', '')
    output_file = os.path.join(
        os.path.dirname(graph_path), f"{graph_filename}_distances.npy"
    )
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    np.save(output_file, dist_matrix)
    print(f"Saved distance matrix to {output_file}")


def prep_default_yaml():
    """
    Writes a default.yaml configuration file for the case.
    """
    case_name = f"{source_id}_{destination_id}"
    case_dir = os.path.join(path_output, "data", "cases", case_name)

    graph_filename_base = "routes"
    graph_dir = os.path.join(case_dir, "graphs")
    graph_file_path = os.path.join(graph_dir, f"{graph_filename_base}.gml")
    charges_file_path = os.path.join(graph_dir, f"{graph_filename_base}_charges.npy")
    distances_file_path = os.path.join(
        graph_dir, f"{graph_filename_base}_distances.npy"
    )

    # Make paths relative to the project root (path_output)
    relative_graph_path = os.path.relpath(graph_file_path, path_output).replace(
        "\\", "/"
    )
    relative_charges_path = os.path.relpath(charges_file_path, path_output).replace(
        "\\", "/"
    )
    relative_distances_path = os.path.relpath(
        distances_file_path, path_output
    ).replace("\\", "/")

    # The output dir for transitions should be inside the case folder
    relative_output_dir = os.path.join(
        "data", "cases", case_name, "transitions"
    ).replace("\\", "/")

    # wind_data_dir is absolute
    wind_data_dir_path = os.path.join(path_output, "data", "era5")

    yaml_content = f"""aircraft_model: NARROW_BODY_JET
charges_file_path: {relative_charges_path}
climb_phase_switch_allowance_climb_time_bins: 10
cost_model_beta0: 0.0
cost_model_beta1: 2.0
cost_model_beta2: 0.0
cost_model_beta3: 1.0
cruise_altitude_ft: 35000.0
cruise_speed_kts: 450.0
delta_t_seconds: 600
device_preference: cuda
distances_file_path: {relative_distances_path}
etto_delta_t_seconds: 30
file_prefix: {case_name}
goal_elevation_ft: 0.0
goal_node: {destination_id}
graph_file_path: {relative_graph_path}
initial_alt_ft: 0.0
max_elapsed_time_since_takeoff_hours: 0.75
max_flight_duration_hours: 5.0
origin_node: {source_id}
output_dir: {relative_output_dir}
source_elevation_ft: 0.0
wind_data_dir: {wind_data_dir_path.replace('\\', '\\\\')}
disable_config_wind_model: true
gamma: 0.001
alpha_pref_reg: 1.0
"""

    output_yaml_path = os.path.join(case_dir, "default.yaml")
    os.makedirs(os.path.dirname(output_yaml_path), exist_ok=True)

    with open(output_yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"Saved default config to {output_yaml_path}")


if __name__ == "__main__":
    # print("Starting preparation...")
    # prep_graph()
    # print("Graph preparation completed")
    # prep_routes_data()
    # print("Routes preparation completed")
    # prep_airspace_charges()
    # print("Airspace charges preparation completed")
    # prep_distance_matrix()
    # print("Distance matrix preparation completed")
    prep_default_yaml()
    print("Default yaml preparation completed")
    print("Preparation completed")
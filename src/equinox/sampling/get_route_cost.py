import torch
import numpy as np
import networkx as nx
from typing import List, Tuple
import os # For checking file existence in example

# Equinox imports
from equinox.helpers.datetimeh import datestr_to_seconds_since_midnight, seconds_since_midnight_to_datetime
from equinox.cost.cost_rev1 import CostRev1
from equinox.wind.wind_date import WindDate
from equinox.vnav.vnav_performance import Performance, get_eta_and_distance_climb, get_eta_and_distance_descent
from equinox.vnav.vnav_profiles_rev1 import NARROW_BODY_JET_CLIMB_PROFILE, NARROW_BODY_JET_DESCENT_PROFILE, NARROW_BODY_JET_CLIMB_VS_PROFILE, NARROW_BODY_JET_DESCENT_VS_PROFILE
from equinox.route.forward_state import get_next_state_fw
from equinox.route.get_wind import get_wind
from equinox.helpers.haversine import haversine # For haversine distance calculation
from equinox.wind.wind_model import WindModel
from equinox.cost.cost_model_1 import cost_model_1

# Constants for flight phases
PHASE_CLIMB = 0
PHASE_CRUISE = 1
PHASE_DESCENT = 2

# Conversion factor
MPS_TO_KNOTS = 1.94384

def get_route_cost(
    route: List[str],
    takeoff_time_str: str,
    graph_path: str = "data/graph/LEMD_EGLL_2023_04_01.gml",
    dist_matrix_path: str = "data/graph/LEMD_EGLL_2023_04_01_distances.npy",
    ac_matrix_path: str = "data/graph/LEMD_EGLL_2023_04_01_charges.npy",
    wind_model: WindModel = None,
    source_elevation_ft: float = 0.0,
    destination_elevation_ft: float = 0.0, # Used for TOD logic
    cruise_alt_ft: float = 35000.0,
    cruise_speed_kts: float = 450.0,
    device_str: str = "cpu"
) -> float:

    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")
    print(f"Using device: {device}")

    if not route or len(route) < 2:
        print("Error: Route must contain at least two nodes.")
        return 0.0

    # 0. Load graph, distance and charge matrices.
    G = nx.read_gml(graph_path)
    node_list_for_matrix = list(G.nodes()) 
    node_to_idx_for_matrix = {nid: i for i, nid in enumerate(node_list_for_matrix)}

    for node_id in route:
        if node_id not in G:
            raise ValueError(f"Node {node_id} from route not found in graph {graph_path}")
        if node_id not in node_to_idx_for_matrix:
            raise ValueError(f"Node {node_id} from route not found in node_to_idx_for_matrix mapping.")

    dist_matrix_np = np.load(dist_matrix_path)
    ac_matrix_np = np.load(ac_matrix_path)

    # 1. Load the cost_model_instance CostRev1
    cost_model_instance = cost_model_1

    # wind_date_str = takeoff_time_str.split(" ")[0]
    # wind_model = WindDate(date_str=wind_date_str, data_dir=wind_data_dir) 

    performance_model = Performance(
        climb_speed_profile=NARROW_BODY_JET_CLIMB_PROFILE,
        descent_speed_profile=NARROW_BODY_JET_DESCENT_PROFILE,
        climb_vertical_speed_profile=NARROW_BODY_JET_CLIMB_VS_PROFILE,
        descent_vertical_speed_profile=NARROW_BODY_JET_DESCENT_VS_PROFILE,
        cruise_altitude_ft=cruise_alt_ft,
        cruise_speed_kts=cruise_speed_kts,
    )
    
    climb_perf_table = get_eta_and_distance_climb(performance_model, 1000)
    descent_perf_table = get_eta_and_distance_descent(performance_model, 1000)

    # 2. Get the takeoff time
    takeoff_ssm = datestr_to_seconds_since_midnight(takeoff_time_str)

    # 3. Perform a forward state prediction pass
    node_states = {} 

    current_node_id = route[0]
    current_alt_ft = torch.tensor([source_elevation_ft], dtype=torch.float32, device=device)
    current_eta_ssm = torch.tensor([takeoff_ssm], dtype=torch.float32, device=device)
    current_phase = torch.tensor([PHASE_CLIMB], dtype=torch.int64, device=device)

    node_states[current_node_id] = {
        "alt": current_alt_ft.item(), "eta": current_eta_ssm.item(), "phase": current_phase.item()
    }
    
    route_edges_params_for_cost = []

    print("\n--- Forward State Prediction ---")
    print(f"Takeoff from {current_node_id} at {seconds_since_midnight_to_datetime(takeoff_time_str, current_eta_ssm.item()).strftime('%Y-%m-%d %H:%M:%S')} alt {current_alt_ft.item():.0f} ft, phase {current_phase.item()}")

    for i in range(len(route) - 1):
        u_node_id = route[i]
        v_node_id = route[i+1]

        coords_src_list = (G.nodes[u_node_id]['lat'], G.nodes[u_node_id]['lon'])
        coords_tgt_list = (G.nodes[v_node_id]['lat'], G.nodes[v_node_id]['lon'])

        coords_src = torch.tensor([coords_src_list], dtype=torch.float32, device=device)
        coords_tgt = torch.tensor([coords_tgt_list], dtype=torch.float32, device=device)
        
        active_performance_table = climb_perf_table
        if current_phase.item() == PHASE_DESCENT:
            active_performance_table = descent_perf_table

        # 4. Compute wind values using state at the source of the edge (u_node_id)
        tailwind_at_u_mps = get_wind(
            coords_src, coords_tgt, current_alt_ft, current_eta_ssm, wind_model
        )
        tailwind_at_u_kts = tailwind_at_u_mps * MPS_TO_KNOTS

        u_idx = node_to_idx_for_matrix[u_node_id]
        v_idx = node_to_idx_for_matrix[v_node_id]
        route_edges_params_for_cost.append({
            "u_id": u_node_id, "v_id": v_node_id,
            "u_idx": u_idx, "v_idx": v_idx,
            "tailwind_kts": tailwind_at_u_kts.item()
        })

        next_alt_ft, next_eta_ssm, next_phase = get_next_state_fw(
            coords_src=coords_src, alts_src=current_alt_ft, eta_src=current_eta_ssm,
            phase_src=current_phase, coords_tgt=coords_tgt,
            climb_performance=active_performance_table, wind_model=wind_model
        )
        
        current_alt_ft = next_alt_ft
        current_eta_ssm = next_eta_ssm
        current_phase = next_phase

        node_states[v_node_id] = {
            "alt": current_alt_ft.item(), "eta": current_eta_ssm.item(), "phase": current_phase.item()
        }
        print(f"  Segment {u_node_id} -> {v_node_id}:")
        print(f"    Tailwind at {u_node_id} (used for cost): {tailwind_at_u_kts.item():.2f} kts")
        print(f"    Arrival at {v_node_id}: ETA {seconds_since_midnight_to_datetime(takeoff_time_str, current_eta_ssm.item()).strftime('%Y-%m-%d %H:%M:%S')}, Alt {current_alt_ft.item():.0f} ft, Phase {current_phase.item()}")

        # final_dest_node_id = route[-1]
        # if current_phase.item() != PHASE_DESCENT and v_node_id != final_dest_node_id:
        #     dist_remaining_nm_to_final = 0
        #     current_leg_idx_in_route = route.index(v_node_id)
            
        #     path_remaining_nodes = route[current_leg_idx_in_route:]
        #     if len(path_remaining_nodes) > 1 : # Check if there's at least one segment remaining
        #         for k_rem in range(len(path_remaining_nodes) - 1):
        #             node1_id = path_remaining_nodes[k_rem]
        #             node2_id = path_remaining_nodes[k_rem+1]
        #             pos1 = (G.nodes[node1_id]['lat'], G.nodes[node1_id]['lon'])
        #             pos2 = (G.nodes[node2_id]['lat'], G.nodes[node2_id]['lon'])
        #             dist_remaining_nm_to_final += haversine(torch.tensor(pos1[0]), torch.tensor(pos1[1]), torch.tensor(pos2[0]), torch.tensor(pos2[1])).item()

        #         # Approx dist needed for descent
        #         alt_diffs_current = torch.abs(descent_perf_table[:, 0] - current_alt_ft.item())
        #         idx_closest_current_alt = torch.argmin(alt_diffs_current)
        #         dist_needed_for_descent_nm = descent_perf_table[idx_closest_current_alt, 2].item()
        #         # This assumes descent_perf_table's distances are to a common low altitude (e.g., destination_elevation_ft or 0)
        #         # A more precise calculation would be: dist_to_descend(current_alt, dest_alt)

        #         DESCENT_INITIATION_BUFFER_NM = 10 
        #         if dist_remaining_nm_to_final < dist_needed_for_descent_nm + DESCENT_INITIATION_BUFFER_NM:
        #             print(f"    INFO: Switching to DESCENT phase for next segment. (Currently at {v_node_id})")
        #             print(f"      Dist remaining to {final_dest_node_id}: {dist_remaining_nm_to_final:.1f} nm. Approx dist for descent from {current_alt_ft.item():.0f} ft: {dist_needed_for_descent_nm:.1f} nm.")
        #             current_phase = torch.tensor([PHASE_DESCENT], dtype=torch.int64, device=device)
    
    # 5. Use the cost model to get the cost at each edge
    total_route_cost = 0.0
    print("\n--- Calculating Edge Costs ---")

    if not route_edges_params_for_cost:
        print("Warning: No edges processed for cost calculation.")
        if len(route) < 2: # No edges if route is too short
             return 0.0
        # If route has edges but params list is empty, something went wrong in the loop.
        # This might happen if route has only one node, handled by initial check.


    for edge_param in route_edges_params_for_cost:
        u_idx_tensor = torch.tensor([edge_param["u_idx"]], device=device, dtype=torch.long)
        v_idx_tensor = torch.tensor([edge_param["v_idx"]], device=device, dtype=torch.long)
        tailwind_tensor = torch.tensor([edge_param["tailwind_kts"]], dtype=torch.float32, device=device)

        edge_cost_tensor = cost_model_instance.forward(
            edge_indices=(u_idx_tensor, v_idx_tensor),
            distance_matrix_d=dist_matrix_np, 
            airspace_charge_matrix_ac=ac_matrix_np,
            tailwind_values_w=tailwind_tensor
        )
        
        cost_value = edge_cost_tensor.item()
        print(f"  Cost for edge {edge_param['u_id']} -> {edge_param['v_id']} (tailwind {edge_param['tailwind_kts']:.2f} kts): {cost_value:.6f}")
        total_route_cost += cost_value

    # 6. Return the total cost
    print(f"\nTotal calculated route cost: {total_route_cost:.2f}")
    return total_route_cost



if __name__ == '__main__':
    print("Running example for get_route_cost...")
    
    example_route = []
    graph_file = "data/graph/LEMD_EGLL_2023_04_01.gml"
    dist_matrix_file = "data/graph/LEMD_EGLL_2023_04_01_distances.npy"
    ac_matrix_file = "data/graph/LEMD_EGLL_2023_04_01_charges.npy"
    wind_dir = "data/era5"

    # Check if data files exist before running
    required_paths = [graph_file, dist_matrix_file, ac_matrix_file, wind_dir]
    files_ok = True
    for p in required_paths:
        if not os.path.exists(p):
            print(f"ERROR: Required data file/directory not found: {p}")
            files_ok = False
            
    if not files_ok:
        print("\nSkipping example run due to missing data files.")
    else:
        try:
            G_sample = nx.read_gml(graph_file)
            nodes = list(G_sample.nodes())
            
            # Define explicit start and end nodes for a more robust example path finding
            start_node_example = "LEMD"
            end_node_example = "EGLL" # Or another node known to be in your graph

            if start_node_example in nodes and end_node_example in nodes:
                try:
                    # Using 'weight=None' for unweighted shortest path (number of hops)
                    # Or a specific weight like 'dist_nm' if available and desired
                    path_nodes = nx.shortest_path(G_sample, source=start_node_example, target=end_node_example, weight=None)
                    example_route = path_nodes 
                    # To make example faster, take a sub-section, e.g., first 5-10 nodes
                    if len(example_route) > 10:
                         example_route = example_route[:10] 
                    elif len(example_route) < 2: # Path is too short or direct
                         if len(nodes) >=2: example_route = [start_node_example, list(G_sample.successors(start_node_example))[0]] if list(G_sample.successors(start_node_example)) else nodes[:2]
                         else: example_route = ["NODE_A", "NODE_B"] # Fallback
                    print(f"Using example route (max 10 segments): {example_route}")
                except nx.NetworkXNoPath:
                    print(f"No path found between {start_node_example} and {end_node_example}. Using a basic fallback route.")
                    if len(nodes) >= 2: example_route = nodes[:2]
                    else: example_route = ["NODE_A", "NODE_B"] 
                except Exception as e:
                    print(f"Error finding shortest path: {e}. Using basic fallback")
                    if len(nodes) >= 2: example_route = nodes[:2]
                    else: example_route = ["NODE_A", "NODE_B"]
            else:
                print(f"Start ({start_node_example}) or End ({end_node_example}) node not in graph. Using generic placeholder nodes.")
                if len(nodes) >=2: example_route = nodes[:2]
                else: example_route = ["NODE_A", "NODE_B"] # Absolute fallback

            if len(example_route) < 2 : # Final check if route is still too short
                print("Example route is too short. Please check graph and node names.")
            else:
                takeoff = "2023-04-01 12:00:00"
                total_cost = get_route_cost(
                    route=example_route,
                    takeoff_time_str=takeoff,
                    graph_path=graph_file,
                    dist_matrix_path=dist_matrix_file,
                    ac_matrix_path=ac_matrix_file,
                    wind_data_dir=wind_dir,
                    # Optional: Specify source/destination elevations if known and different from 0
                    # source_elevation_ft=2000, # Example: LEMD elevation approx 2000ft
                    # destination_elevation_ft=80 # Example: EGLL elevation approx 80ft
                )
                print(f"\nExample calculated total cost for route {example_route}: {total_cost}")

        except FileNotFoundError:
            print(f"ERROR: Graph file {graph_file} not found. Cannot run example.")
        except Exception as e:
            print(f"An error occurred during the example setup or run: {e}")
            import traceback
            traceback.print_exc() 
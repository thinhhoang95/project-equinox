import torch
import numpy as np
import networkx as nx
from datetime import datetime
from equinox.dp.pretoc.forward_soft_bellman import run_forward_soft_bellman
from equinox.helpers.datetimeh import datestr_to_seconds_since_midnight, seconds_since_midnight_to_datetime
from equinox.cost.cost_rev1 import CostRev1
from equinox.cost.cost_model_1 import cost_model_1
from equinox.wind.wind_date import WindDate
from equinox.vnav.vnav_performance import Performance
from equinox.vnav.vnav_profiles_rev1 import NARROW_BODY_JET_CLIMB_PROFILE, NARROW_BODY_JET_DESCENT_PROFILE, NARROW_BODY_JET_CLIMB_VS_PROFILE, NARROW_BODY_JET_DESCENT_VS_PROFILE

def test_forward_dp():
    
    print('Test: Forward Dynamic Programming (forward_dp_vec2)')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the route graph
    G = nx.read_gml("../data/graph/LEMD_EGLL_2023_04_01.gml")

    node_list_for_matrix = list(G.nodes()) # Consistent order for matrix indexing
    node_to_idx_for_matrix = {nid: i for i, nid in enumerate(node_list_for_matrix)}
    num_actual_nodes = len(node_list_for_matrix)

    # Load the distance matrix
    dist_matrix = np.load("../data/graph/LEMD_EGLL_2023_04_01_distances.npy")
    
    # Load the airspace charges matrix
    ac_matrix = np.load("../data/graph/LEMD_EGLL_2023_04_01_charges.npy")

    # Load the wind model
    wind_model = WindDate(date_str="2024-04-01", data_dir="../data/era5")

    # Load the performance model for a typical narrow body jet
    performance_model = Performance(
        climb_speed_profile=NARROW_BODY_JET_CLIMB_PROFILE,
        descent_speed_profile=NARROW_BODY_JET_DESCENT_PROFILE,
        climb_vertical_speed_profile=NARROW_BODY_JET_CLIMB_VS_PROFILE,
        descent_vertical_speed_profile=NARROW_BODY_JET_DESCENT_VS_PROFILE,
        cruise_altitude_ft=35000.0,
        cruise_speed_kts=450.0,
    )

    # Load the cost model
    cost_model_instance = cost_model_1
    
    # Takeoff time
    takeoff_time_str = "2023-04-01 12:00:00"

    try:
        V_final, eta_final, alt_final, phase_final, transitions_list = run_forward_soft_bellman(
            graph=G,
            source_node_id="LEMD",
            takeoff_time_str=takeoff_time_str,
            source_elevation_ft=0.0, # Assuming SFO at sea level
            goal_elevation_ft=0.0,   # Not used during forward DP
            cost_model=cost_model_instance,
            wind_model=wind_model,
            performance_model=performance_model,
            dist_matrix_np=dist_matrix,
            ac_matrix_np=ac_matrix,
            initial_alt_ft=0.0, # Start at 0ft AMSL if source_elevation_ft is 0 for profile alignment
            delta_t_seconds=600, # time window length
            max_flight_duration_hours=5, # max duration to consider for time bins
            etto_delta_t_seconds=30, # ETTO bin width
            max_elapsed_time_since_takeoff_hours=0.75, # max elapsed time to consider for ETTO bins
            device=device
        )
        # transitions_list = [(node_1, eps_1, alt_1, node_2, eps_2, alt_2)]
        # where eps_1 and eps_2 are ETTO values (not ETTO bins)
        # node_1 and node_2 are node IDs (integers, not node names)
        # alt_1 and alt_2 are altitudes (float, in feet)

        print("\n--- Results ---")
        print(f"V function shape: {V_final.shape}")
        print(f'Total number of unique nodes_from in transitions_list: {len(set([node_1 for node_1, _, _, node_2, _, _ in transitions_list]))}')
        print(f'Total number of unique nodes_to in transitions_list: {len(set([node_2 for _, _, _, node_2, _, _ in transitions_list]))}')
        print(f'Total number of unique transitions in transitions_list: {len(transitions_list)}')
        # base_output_time_str = f"{takeoff_time_str}"

        # for node_idx_res in range(num_actual_nodes):
        #     for time_idx_res in range(V_final.shape[1]):
        #         if not torch.isinf(V_final[node_idx_res, time_idx_res]):
        #             node_id_res = node_list_for_matrix[node_idx_res] # Use the consistent list
                    
        #             # Calculate the approximate start time of this bin for display
        #             # min_time_overall_seconds is takeoff_seconds_since_midnight from inside run_forward_dp
        #             # This was the min_time_overall_seconds used to calculate k_v
        #             takeoff_ssm_ref = datestr_to_seconds_since_midnight(takeoff_time_str) # ssm: seconds since midnight
        #             bin_start_time_ssm = takeoff_ssm_ref + time_idx_res * 600 # 600 is delta_t_seconds from example

        #             time_dt_display = seconds_since_midnight_to_datetime(base_output_time_str, bin_start_time_ssm)
                    
        #             print(f"Node {node_id_res} ({node_idx_res}), Time Bin {time_idx_res} (approx arrival by {time_dt_display.strftime('%Y-%m-%d %H:%M:%S')}):")
        #             print(f"  V = {V_final[node_idx_res, time_idx_res].item():.2f}")
        #             active_eta_val = eta_final[node_idx_res, time_idx_res].item()
        #             active_alt_val = alt_final[node_idx_res, time_idx_res].item()
        #             active_phase_val = phase_final[node_idx_res, time_idx_res].item()
        #             print(f"  Exact ETA: {seconds_since_midnight_to_datetime(base_output_time_str, active_eta_val).strftime('%Y-%m-%d %H:%M:%S') if not np.isnan(active_eta_val) else 'N/A'}")
        #             print(f"  Altitude (ft): {active_alt_val if not np.isnan(active_alt_val) else 'N/A'}")
        #             print(f"  Phase: {active_phase_val if active_phase_val != -1 else 'N/A'}")

        # Dump the value function V to a file
        import os

        output_dir = "../data/results/forward"
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, "V_final.npy"), V_final.cpu().detach().numpy())
        np.save(os.path.join(output_dir, "eta_final.npy"), eta_final.cpu().detach().numpy())
        np.save(os.path.join(output_dir, "alt_final.npy"), alt_final.cpu().detach().numpy())
        np.save(os.path.join(output_dir, "phase_final.npy"), phase_final.cpu().detach().numpy())

        return V_final, eta_final, alt_final, phase_final, transitions_list

    except Exception as e:
        print(f"An error occurred during the example run: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_forward_dp()
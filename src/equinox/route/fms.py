from typing import Tuple, List
import networkx as nx
from equinox.helpers.datetimeh import datestr_to_seconds_since_midnight
from equinox.route.forward_state import get_next_state_fw
from equinox.route.backward_state import get_next_state_bw
import torch
from equinox.vnav.vnav_performance import Performance, get_eta_and_distance_climb, get_eta_and_distance_descent
from equinox.vnav.vnav_profiles_rev1 import NARROW_BODY_JET_CLIMB_PROFILE, NARROW_BODY_JET_CLIMB_VS_PROFILE, NARROW_BODY_JET_DESCENT_PROFILE, NARROW_BODY_JET_DESCENT_VS_PROFILE
from equinox.wind.wind_model import WindModel
from equinox.wind.wind_date import WindDate
from datetime import datetime
from equinox.helpers.haversine import haversinet # torch version
from equinox.helpers.datetimeh import seconds_to_hhmmss

def parse_waypoint_string(flight_plan_string: str):
    return flight_plan_string.split(" ") # split on spaces

def get_4d_trajectory(waypoint_string: str, airac_graph: nx.Graph,
                      origin_elevation_ft: float, destination_elevation_ft: float,
                      take_off_time_s: float, # seconds since midnight
                      climb_performance: List[Tuple[float, float, float]], # [(altitude (ft), eta (s), along_track_distance (nm))]
                      descent_performance: List[Tuple[float, float, float]], # ADDED
                      wind_model: WindModel) -> Tuple[List[str], List[str], List[str], List[str]]:
    # Parse flight plan
    flight_plan_waypoints = parse_waypoint_string(waypoint_string) # e.g. ["EGLL", "EHAM", "LFPG", "LIRF"]
    
    # Lookup the coordinates of the waypoints
    flight_plan_coords = [(airac_graph.nodes[wp]['lat'], airac_graph.nodes[wp]['lon']) for wp in flight_plan_waypoints]

    # ======== Forward Pass ========    
    # State vector initialization: phase (0=climb, 1=cruise, 2=descent), eta (s), along_track_distance (nm)
    flight_plan_phase_fw = torch.zeros(len(flight_plan_coords), dtype=torch.int32) # (N)
    flight_plan_eta_fw = torch.zeros(len(flight_plan_coords), dtype=torch.float32) # (N)
    flight_plan_dist_fw = torch.zeros(len(flight_plan_coords), dtype=torch.float32) # (N)
    flight_plan_alt_ft_fw = torch.zeros(len(flight_plan_coords), dtype=torch.float32) # (N)
    
    # Set the origin and destination phases
    flight_plan_phase_fw[0] = 0 # origin is climb
    flight_plan_phase_fw[-1] = 2 # destination is descent

    # Set the altitude at the origin and destination
    flight_plan_alt_ft_fw[0] = origin_elevation_ft
    flight_plan_alt_ft_fw[-1] = destination_elevation_ft

    for i in range(len(flight_plan_coords) - 1):
        src_coords = torch.tensor([flight_plan_coords[i]], dtype=torch.float32) # (1, 2)
        alts_src = torch.tensor([flight_plan_alt_ft_fw[i]], dtype=torch.float32) # (1)
        eta_src = torch.tensor([flight_plan_eta_fw[i]], dtype=torch.float32) # (1)
        phase_src = torch.tensor([flight_plan_phase_fw[i]], dtype=torch.int32) # (1)
        tgt_coords = torch.tensor([flight_plan_coords[i + 1]], dtype=torch.float32) # (1, 2)
        dist_src = torch.tensor([flight_plan_dist_fw[i]], dtype=torch.float32) # (1)
        
        # Get haversine distance between src and tgt
        dist_tgt = haversinet(src_coords[0, 0], src_coords[0, 1], tgt_coords[0, 0], tgt_coords[0, 1])
        dist_tgt_np = dist_tgt.cpu().numpy().item() # relative distance from src to tgt

        # Get the next state
        alt_tgt, eta_tgt, phase_tgt = get_next_state_fw(src_coords, alts_src, eta_src, phase_src, tgt_coords, climb_performance, wind_model)

        # Convert to numpy on cpu before displaying
        alt_tgt_np = alt_tgt.cpu().numpy().item()
        eta_tgt_np = eta_tgt.cpu().numpy().item()
        phase_tgt_np = phase_tgt.cpu().numpy().item()

        # Update the state vectors
        flight_plan_alt_ft_fw[i + 1] = alt_tgt_np
        flight_plan_eta_fw[i + 1] = eta_tgt_np
        flight_plan_dist_fw[i + 1] = dist_tgt_np + dist_src # cumulative distance
        flight_plan_phase_fw[i + 1] = phase_tgt_np

    # ======== Backward Pass ========
    # Initialize _bw tensors for altitude, ETA, and phase
    flight_plan_phase_bw = torch.zeros_like(flight_plan_phase_fw)
    flight_plan_eta_bw = torch.zeros_like(flight_plan_eta_fw)
    flight_plan_alt_ft_bw = torch.zeros_like(flight_plan_alt_ft_fw)

    # Set initial conditions for the backward pass at the destination
    flight_plan_alt_ft_bw[-1] = destination_elevation_ft
    flight_plan_eta_bw[-1] = flight_plan_eta_fw[-1]  # Use arrival ETA from forward pass
    flight_plan_phase_bw[-1] = 2  # DESCENT phase

    # Iterate backwards from the second to last waypoint to the origin
    for i in range(len(flight_plan_coords) - 1, 0, -1):
        # Define coordinates for the current segment in backward pass
        # coords_src_bw is the waypoint whose state we are calculating (i-1)
        # coords_tgt_bw is the reference waypoint (i)
        coords_src_bw = torch.tensor([flight_plan_coords[i-1]], dtype=torch.float32)
        alts_tgt_bw = torch.tensor([flight_plan_alt_ft_bw[i]], dtype=torch.float32)
        eta_tgt_bw = torch.tensor([flight_plan_eta_bw[i]], dtype=torch.float32)
        phase_tgt_bw = torch.tensor([flight_plan_phase_bw[i]], dtype=torch.int32)
        coords_tgt_bw = torch.tensor([flight_plan_coords[i]], dtype=torch.float32)
        
        # Get the state (altitude, ETA, phase) of the preceding waypoint (i-1) using backward propagation
        alt_src_val, eta_src_val, phase_src_val = get_next_state_bw(
            coords_src_bw,
            alts_tgt_bw,
            eta_tgt_bw,
            phase_tgt_bw,
            coords_tgt_bw,
            descent_performance, # Use descent performance characteristics
            wind_model
        )

        # Update the backward pass state vectors for waypoint (i-1)
        flight_plan_alt_ft_bw[i-1] = alt_src_val.cpu().numpy().item()
        flight_plan_eta_bw[i-1] = eta_src_val.cpu().numpy().item()
        flight_plan_phase_bw[i-1] = phase_src_val.cpu().numpy().item()
        
    # ======== Combine Forward and Backward Pass Results ========
    # Update the final trajectory (_fw variables) using the descent profile from the backward pass (_bw variables).
    # The flight_plan_dist_fw remains from the forward pass.
    
    # Ensure destination values are consistent with backward pass (especially for phase and altitude)
    flight_plan_alt_ft_fw[-1] = flight_plan_alt_ft_bw[-1] 
    flight_plan_eta_fw[-1] = flight_plan_eta_bw[-1] # This should match by construction
    flight_plan_phase_fw[-1] = flight_plan_phase_bw[-1] # This ensures phase is 2 (DESCENT)

    # Iterate backwards from the second-to-last waypoint.
    # If the backward pass determined a point is in DESCENT, use its results for alt, eta, and phase.
    # Otherwise, retain the forward pass results (for CRUISE/CLIMB phases).
    for i in range(len(flight_plan_coords) - 2, -1, -1): # Iterate from N-2 down to 0
        if flight_plan_phase_bw[i].item() == 2:  # If waypoint 'i' is in DESCENT phase per backward pass
            flight_plan_alt_ft_fw[i] = flight_plan_alt_ft_bw[i]
            flight_plan_eta_fw[i] = flight_plan_eta_bw[i]
            flight_plan_phase_fw[i] = flight_plan_phase_bw[i] # This will set phase to 2
        else:
            # This point 'i' is not in DESCENT phase according to the backward pass (e.g., it's CRUISE or CLIMB).
            # This signifies the Top of Descent (or earlier phase) from the backward pass perspective.
            # We retain the forward pass calculations for this point and all earlier points.
            break
            
    return flight_plan_phase_fw, flight_plan_eta_fw, flight_plan_dist_fw, flight_plan_alt_ft_fw

import time

if __name__ == "__main__":
    # Load the airac graph
    time_start = time.time()
    airac_graph = nx.read_gml("data/wp/airac_2502.gml")
    print(f"Time taken to load the airac graph: {time.time() - time_start} seconds")

    # Prepare the performance tables
    performance = Performance(
        NARROW_BODY_JET_CLIMB_PROFILE,
        NARROW_BODY_JET_DESCENT_PROFILE,
        NARROW_BODY_JET_CLIMB_VS_PROFILE,
        NARROW_BODY_JET_DESCENT_VS_PROFILE,
        cruise_altitude_ft=35000,
        cruise_speed_kts=450,
    )

    # Get the climb_performance table (the time and distance profiles)
    climb_performance_table = get_eta_and_distance_climb(performance, 1000) # return [(altitude (ft), eta (s), along_track_distance (nm))]

    # Get the descent_performance table
    descent_performance_table = get_eta_and_distance_descent(performance, 1000) # ADDED

    # Load the wind_model
    wind_model = WindDate("2024-04-01", "data/era5")

    # Waypoint string
    fpl_str = "LFBO FISTO POI PEPAX NIMER KEPER LUMAN ROMGO NERKI BANOX LFPG"

    # Take-off time
    takeoff_time = "2024-04-01 06:00:00"
    takeoff_time_s = datestr_to_seconds_since_midnight(takeoff_time)

    time_start = time.time()
    # Get the 4D trajectory
    flight_plan_phase_fw, flight_plan_eta_fw, flight_plan_dist_fw, flight_plan_alt_ft_fw = get_4d_trajectory(
        fpl_str, airac_graph, 1000, 1000, takeoff_time_s, 
        climb_performance_table, 
        descent_performance_table, # ADDED
        wind_model
    )
    print(f"Time taken to get the 4D trajectory: {time.time() - time_start} seconds")
    # Format and print the results as a table
    import pandas as pd

    # Convert tensors to numpy arrays
    phase_np = flight_plan_phase_fw.cpu().numpy()
    eta_np = flight_plan_eta_fw.cpu().numpy()
    dist_np = flight_plan_dist_fw.cpu().numpy()
    alt_np = flight_plan_alt_ft_fw.cpu().numpy()

    # Map phase integers to names
    phase_map = {0: "CLIMB", 1: "CRUISE", 2: "DESCENT"}
    phase_str = [phase_map.get(int(p), str(int(p))) for p in phase_np]

    # Build DataFrame
    df = pd.DataFrame({
        "Waypoint #": range(len(phase_np)),
        "Phase": phase_str,
        # "ETA (s)": eta_np,
        "TTLT": [seconds_to_hhmmss(e) for e in eta_np],
        "Distance (nm)": dist_np,
        "Altitude (ft)": alt_np,
    })

    # Format and print
    print(df.to_string(index=False, float_format="%.2f"))

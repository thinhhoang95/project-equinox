from equinox.route.value_iteration_torch import get_next_state
from equinox.vnav.vnav_performance import Performance, get_eta_and_distance_climb
from equinox.wind.wind_free import WindFree 
from equinox.vnav.vnav_profiles_rev1 import NARROW_BODY_JET_CLIMB_PROFILE, NARROW_BODY_JET_CLIMB_VS_PROFILE, NARROW_BODY_JET_DESCENT_PROFILE, NARROW_BODY_JET_DESCENT_VS_PROFILE
import torch
from equinox.helpers.haversine import haversine

def test_one_climb_phase_wind_free():
    # Create a wind-free wind model
    wind_model = WindFree()

    # Create a performance model
    performance = Performance(
        NARROW_BODY_JET_CLIMB_PROFILE,
        NARROW_BODY_JET_DESCENT_PROFILE,
        NARROW_BODY_JET_CLIMB_VS_PROFILE,
        NARROW_BODY_JET_DESCENT_VS_PROFILE,
        cruise_altitude_ft=35000,
        cruise_speed_kts=450,
    )

    # Get the performance table for the climb phase
    performance_table = get_eta_and_distance_climb(performance, 1000) # altitude (ft), eta (s), along_track_distance (nm)

    # Print the performance table
    # Print the performance table as a formatted table
    print("Performance Table (altitude_ft | eta_sec | along_track_distance_nm):")
    print(f"{'Altitude (ft)':>15} | {'ETA (s)':>10} | {'Distance (nm)':>16}")
    print("-" * 48)
    for row in performance_table:
        alt, eta, dist = row
        print(f"{alt:15,.0f} | {eta:10.1f} | {dist:16.2f}")
    
    # Create source state
    coords_src = torch.tensor([[37.7749, -122.4194]]) # SFO
    alts_src = torch.tensor([1000]) # 1000 ft
    eta_src = torch.tensor([0]) # 0 seconds
    phase_src = torch.tensor([0]) # CLIMB

    # Create target state
    coords_tgt = torch.tensor([[38.123, -121.021]]) # TIPRE waypoint

    # Get the haversine distance between the source and target
    dist_nm = haversine(coords_src[0, 0], coords_src[0, 1], coords_tgt[0, 0], coords_tgt[0, 1])

    print(f"Distance between SFO and TIPRE: {dist_nm:.2f} nautical miles")

    # Get the next state
    alt_tgt, eta_tgt, phase_tgt = get_next_state(coords_src, alts_src, eta_src, phase_src, coords_tgt, performance_table, wind_model)

    # Convert to numpy on cpu before displaying
    alt_tgt_np = alt_tgt.cpu().numpy()
    eta_tgt_np = eta_tgt.cpu().numpy()
    phase_tgt_np = phase_tgt.cpu().numpy()

    # Print the next state
    print("Next state table:")
    print(f"{'Altitude (ft)':>15} | {'ETA (s)':>10} | {'Phase':>8}")
    print("-" * 40)
    for alt, eta, phase in zip(alt_tgt_np, eta_tgt_np, phase_tgt_np):
        print(f"{alt:15,.0f} | {eta:10.1f} | {phase:8d}")

def test_two_climb_phases_wind_free():
    # Create a wind-free wind model
    wind_model = WindFree()

    # Create a performance model
    performance = Performance(
        NARROW_BODY_JET_CLIMB_PROFILE,
        NARROW_BODY_JET_DESCENT_PROFILE,
        NARROW_BODY_JET_CLIMB_VS_PROFILE,
        NARROW_BODY_JET_DESCENT_VS_PROFILE,
        cruise_altitude_ft=35000,
        cruise_speed_kts=450,
    )

    # Get the performance table for the climb phase
    performance_table = get_eta_and_distance_climb(performance, 1000) # altitude (ft), eta (s), along_track_distance (nm)

    # Print the performance table
    # Print the performance table as a formatted table
    print("Performance Table (altitude_ft | eta_sec | along_track_distance_nm):")
    print(f"{'Altitude (ft)':>15} | {'ETA (s)':>10} | {'Distance (nm)':>16}")
    print("-" * 48)
    for row in performance_table:
        alt, eta, dist = row
        print(f"{alt:15,.0f} | {eta:10.1f} | {dist:16.2f}")
    
    # Create source state
    coords_src = torch.tensor([[37.7749, -122.4194]]) # SFO
    alts_src = torch.tensor([1000]) # 1000 ft
    eta_src = torch.tensor([0]) # 0 seconds
    phase_src = torch.tensor([0]) # CLIMB

    # Create target state
    coords_tgt = torch.tensor([[38.407, -117.179]]) # INSLO waypoint

    # Get the haversine distance between the source and target
    dist_nm = haversine(coords_src[0, 0], coords_src[0, 1], coords_tgt[0, 0], coords_tgt[0, 1])

    print(f"Distance between SFO and INSLO: {dist_nm:.2f} nautical miles")

    # Get the next state
    alt_tgt, eta_tgt, phase_tgt = get_next_state(coords_src, alts_src, eta_src, phase_src, coords_tgt, performance_table, wind_model)

    # Convert to numpy on cpu before displaying
    alt_tgt_np = alt_tgt.cpu().numpy()
    eta_tgt_np = eta_tgt.cpu().numpy()
    phase_tgt_np = phase_tgt.cpu().numpy()

    # Print the next state
    print("Next state table:")
    print(f"{'Altitude (ft)':>15} | {'ETA (s)':>10} | {'Phase':>8}")
    print("-" * 40)
    for alt, eta, phase in zip(alt_tgt_np, eta_tgt_np, phase_tgt_np):
        print(f"{alt:15,.0f} | {eta:10.1f} | {phase:8d}")

if __name__ == "__main__":
    # test_one_climb_phase_wind_free()
    test_two_climb_phases_wind_free()
from equinox.helpers.datetimeh import datestr_to_seconds_since_midnight
from equinox.route.forward_state import get_next_state_fw
from equinox.route.backward_state_toc import get_next_state_bw
from equinox.vnav.vnav_performance import Performance, get_eta_and_distance_climb, get_eta_and_distance_descent
from equinox.wind.wind_free import WindFree 
from equinox.wind.wind_model import WindModel
from equinox.wind.wind_date import WindDate
from equinox.vnav.vnav_profiles_rev1 import NARROW_BODY_JET_CLIMB_PROFILE, NARROW_BODY_JET_CLIMB_VS_PROFILE, NARROW_BODY_JET_DESCENT_PROFILE, NARROW_BODY_JET_DESCENT_VS_PROFILE
import torch
from equinox.helpers.haversine import haversine
from equinox.helpers.datetimeh import seconds_since_midnight_to_datestr

def convert_seconds_to_minutes_and_seconds(seconds: float) -> str:
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes}:{seconds:02d}"

def test_full_regime_flight():
    # wind_model = WindDate(
    #     date_str="2024-04-01",
    #     data_dir="data/era5"
    # )

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
    
    # Get the performance table for the descent phase
    descent_performance_table = get_eta_and_distance_descent(performance, 1000) # altitude (ft), eta (s), along_track_distance (nm)
    climb_performance_table = get_eta_and_distance_climb(performance, 1000) # altitude (ft), eta (s), along_track_distance (nm)

    # Print the performance table
    # Print the performance table as a formatted table
    print("Descent Performance Table (altitude_ft | eta_sec | along_track_distance_nm):")
    print(f"{'Altitude (ft)':>15} | {'ETATO (s)':>10} | {'Distance (nm)':>16}")
    print("-" * 48)
    for row in descent_performance_table:
        alt, eta, dist = row
        print(f"{alt:15,.0f} | {convert_seconds_to_minutes_and_seconds(eta):>10} | {dist:16.2f}")

    # Print the climb performance table
    print("Climb Performance Table (altitude_ft | eta_sec | along_track_distance_nm):")
    print(f"{'Altitude (ft)':>15} | {'ETATO (s)':>10} | {'Distance (nm)':>16}")
    print("-" * 48)
    for row in climb_performance_table:
        alt, eta, dist = row
        print(f"{alt:15,.0f} | {convert_seconds_to_minutes_and_seconds(eta):>10} | {dist:16.2f}")

    # Destination airport: KJFK
    coords_tgt = torch.tensor([[51.4680, -0.4551]]) # EGLL/London Heathrow
    alts_tgt = torch.tensor([35000]) # 0 ft, elevation at arrival
    
    # Arrival time
    time_at_arrival = "2024-04-01 12:00:00"
    seconds_since_midnight = datestr_to_seconds_since_midnight(time_at_arrival)

    # Takeoff time
    time_at_takeoff = "2024-04-01 10:25:17" # at Madrid-Barajas
    seconds_since_midnight_takeoff = datestr_to_seconds_since_midnight(time_at_takeoff)

    # Create source state
    eta_tgt = torch.tensor([seconds_since_midnight]) # seconds since midnight
    eta_takeoff = torch.tensor([seconds_since_midnight_takeoff]) # seconds since midnight
    phase_tgt = torch.tensor([1]) # CRUISE

    # The preceding node coordinates
    coords_src = torch.tensor([[40.4895, -3.5643]]) # LEMD

    # Get the haversine distance between the source and target
    dist_nm = haversine(coords_src[0, 0], coords_src[0, 1], coords_tgt[0, 0], coords_tgt[0, 1])

    print(f"Distance between LEMD and EGLL: {dist_nm:.2f} nautical miles")
    print(f"Time at takeoff: {time_at_takeoff}")
    print(f"Time at arrival: {time_at_arrival}")

    # Get the preceding state
    alt_src, eta_src, phase_src = get_next_state_bw(
        coords_src,
        alts_tgt,
        eta_tgt,
        phase_tgt,
        coords_tgt,
        descent_performance_table,
        climb_performance_table,
        eta_takeoff,
        wind_model)

    # Convert to numpy on cpu before displaying
    alt_src_np = alt_src.cpu().numpy()
    eta_src_np = eta_src.cpu().numpy()
    phase_src_np = phase_src.cpu().numpy()
    
    # Print the preceding state
    print("Preceding state table:")
    print(f"{'Altitude (ft)':>15} | {'ETA':>25} | {'Phase':>8}")
    print("-" * 54)
    for alt, eta, phase in zip(alt_src_np, eta_src_np, phase_src_np):
        print(f"{alt:15,.0f} | {seconds_since_midnight_to_datestr(time_at_arrival, eta):>25} | {phase:8d}")

if __name__ == "__main__":
    test_full_regime_flight()
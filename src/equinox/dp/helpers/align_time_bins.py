from typing import Dict, Union
from equinox.helpers.datetimeh import datestr_to_seconds_since_midnight

def calculate_aligned_time_parameters(
    takeoff_time_str: str,
    estimated_landing_time_str: str,
    global_window_duration_hours: float,
    delta_t_seconds: int
) -> Dict[str, Union[float, int]]:
    """
    Calculates parameters for a common time grid to align forward and backward DP passes.

    The goal is to ensure that a time bin index 'k' refers to the same absolute time
    interval in both forward (V_f[node, k]) and backward (V_b[node, k]) DP results.
    This is achieved by defining a common 'min_time_overall_seconds' and 'num_time_bins'
    to be used by both DP computations.

    The common time window, defined by 'global_window_duration_hours', is effectively
    centered around the actual flight duration (from takeoff to estimated landing)
    by adjusting the 'min_time_overall_seconds'.

    Args:
        takeoff_time_str: ISO format takeoff time string (e.g., "2023-04-01 12:00:00").
        estimated_landing_time_str: ISO format estimated landing time string.
        global_window_duration_hours: The total duration of the common time window in hours.
                                      This determines 'num_time_bins'. Must be non-negative
                                      and sufficient to cover the actual flight duration.
        delta_t_seconds: Duration of each time bin in seconds. Must be > 0.

    Returns:
        A dictionary containing:
        - "min_time_overall_seconds": The common reference start time (seconds since midnight)
                                      for the shared time grid. Can be a float.
        - "num_time_bins": The common number of time bins for DP arrays (integer).
        - "takeoff_bin_idx": The bin index (integer) in the common grid for the takeoff time.
        - "landing_bin_idx": The bin index (integer) in the common grid for the landing time.
        - "delta_t_seconds": Passed through for convenience (integer).
        - "takeoff_seconds_since_midnight": The takeoff time in seconds since midnight.
        - "landing_seconds_since_midnight": The landing time in seconds since midnight.

    Raises:
        ValueError: If inputs are invalid (e.g., landing before takeoff,
                    window duration too short, non-positive delta_t_seconds or
                    global_window_duration_hours).
    """
    if delta_t_seconds <= 0:
        raise ValueError("delta_t_seconds must be positive.")
    if global_window_duration_hours < 0: # Allow 0 duration for specific cases like instantaneous events
        raise ValueError("global_window_duration_hours must be non-negative.")

    takeoff_s = datestr_to_seconds_since_midnight(takeoff_time_str)
    landing_s = datestr_to_seconds_since_midnight(estimated_landing_time_str)

    if landing_s < takeoff_s:
        raise ValueError(f"Estimated landing time ({estimated_landing_time_str}) "
                         f"cannot be before takeoff time ({takeoff_time_str}).")

    actual_flight_duration_s = landing_s - takeoff_s
    # Ensure global_window_duration_s is float for precision with padding
    global_window_duration_s = float(global_window_duration_hours * 3600.0)


    if global_window_duration_s < actual_flight_duration_s:
        raise ValueError(
            f"Global window duration ({global_window_duration_hours:.2f} hrs) must be "
            f"at least the actual flight duration ({actual_flight_duration_s / 3600.0:.2f} hrs)."
        )

    # Calculate num_time_bins based on the global window duration.
    # This matches the logic in the DP codes: int(duration / delta) + 1
    if delta_t_seconds == 0: # Should have been caught, but defensive
         raise ValueError("delta_t_seconds cannot be zero.")
    num_time_bins = int(global_window_duration_s / delta_t_seconds) + 1
    
    # Center the actual flight duration within the global window.
    # padding_s is the total available slack time in the window.
    padding_s = global_window_duration_s - actual_flight_duration_s
    # Distribute half of the padding before the takeoff time.
    padding_before_s = padding_s / 2.0
    
    common_min_time_overall_seconds = float(takeoff_s - padding_before_s)

    # Calculate bin indices using this common reference start time.
    # round() behaves like torch.round() for .5 cases (rounds to nearest even integer).
    # Bin indices must be integers.
    takeoff_bin_idx = int(round((takeoff_s - common_min_time_overall_seconds) / delta_t_seconds))
    landing_bin_idx = int(round((landing_s - common_min_time_overall_seconds) / delta_t_seconds))
    
    # Ensure takeoff_bin_idx is not negative due to floating point nuances if padding_before_s is extremely small and takeoff_s is common_min_time_overall_seconds
    if takeoff_s == common_min_time_overall_seconds and takeoff_bin_idx < 0 : # pragma: no cover
        takeoff_bin_idx = 0

    # Validate that calculated bin indices are within the valid range [0, num_time_bins - 1]
    # These should hold if previous checks on durations are correct.
    if not (0 <= takeoff_bin_idx < num_time_bins): # pragma: no cover
        raise ValueError(f"Calculated takeoff_bin_idx {takeoff_bin_idx} is out of bounds [0, {num_time_bins - 1}]. "
                         f"Check input parameters. common_min_time: {common_min_time_overall_seconds}, takeoff_s: {takeoff_s}")
    if not (0 <= landing_bin_idx < num_time_bins): # pragma: no cover
        raise ValueError(f"Calculated landing_bin_idx {landing_bin_idx} is out of bounds [0, {num_time_bins - 1}]. "
                         f"Check input parameters. common_min_time: {common_min_time_overall_seconds}, landing_s: {landing_s}")


    return {
        "min_time_overall_seconds": common_min_time_overall_seconds,
        "num_time_bins": num_time_bins,
        "takeoff_bin_idx": takeoff_bin_idx,
        "landing_bin_idx": landing_bin_idx,
        "delta_t_seconds": delta_t_seconds,
        "takeoff_seconds_since_midnight": takeoff_s,
        "landing_seconds_since_midnight": landing_s
    }

# Example usage (for testing or demonstration):
if __name__ == '__main__': # pragma: no cover
    # Scenario 1: Exact match
    params1 = calculate_aligned_time_parameters(
        takeoff_time_str="2023-01-01 10:00:00",
        estimated_landing_time_str="2023-01-01 11:00:00",
        global_window_duration_hours=1.0,
        delta_t_seconds=600  # 10 minutes
    )
    print(f"Scenario 1: {params1}")
    # Expected: min_time around 10:00 (36000), num_bins=7 (for 1hr/10min), takeoff_idx=0, landing_idx=6

    # Scenario 2: Window larger than flight
    params2 = calculate_aligned_time_parameters(
        takeoff_time_str="2023-01-01 10:00:00",
        estimated_landing_time_str="2023-01-01 11:00:00", # 1 hour flight
        global_window_duration_hours=2.0, # 2 hour window
        delta_t_seconds=600
    )
    print(f"Scenario 2: {params2}")
    # Expected: min_time around 09:30 (34200), num_bins=13, takeoff_idx=3, landing_idx=9

    # Scenario 3: Zero duration flight, non-zero window
    params3 = calculate_aligned_time_parameters(
        takeoff_time_str="2023-01-01 10:00:00",
        estimated_landing_time_str="2023-01-01 10:00:00", # 0 hour flight
        global_window_duration_hours=1.0, # 1 hour window
        delta_t_seconds=600
    )
    print(f"Scenario 3: {params3}")
    # Expected: min_time around 09:30 (34200), num_bins=7, takeoff_idx=3, landing_idx=3

    # Scenario 4: Flight fits perfectly at end of a bin, window matches
    params4 = calculate_aligned_time_parameters(
        takeoff_time_str="2023-01-01 10:00:00",       # 36000s
        estimated_landing_time_str="2023-01-01 10:10:00", # 36600s (10 min flight)
        global_window_duration_hours=10.0/60.0,       # 10 min window
        delta_t_seconds=600                           # 10 min bins
    )
    print(f"Scenario 4: {params4}")
    # Expected: min_time=36000, num_bins=2 (0,1), takeoff_idx=0, landing_idx=1

    try:
        calculate_aligned_time_parameters(
            takeoff_time_str="2023-01-01 10:00:00",
            estimated_landing_time_str="2023-01-01 09:00:00", # Landing before takeoff
            global_window_duration_hours=1.0,
            delta_t_seconds=600
        )
    except ValueError as e:
        print(f"Error (expected): {e}")

    try:
        calculate_aligned_time_parameters(
            takeoff_time_str="2023-01-01 10:00:00",
            estimated_landing_time_str="2023-01-01 11:00:00",
            global_window_duration_hours=0.5, # Window too short
            delta_t_seconds=600
        )
    except ValueError as e:
        print(f"Error (expected): {e}")

    try:
        calculate_aligned_time_parameters(
            takeoff_time_str="2023-01-01 10:00:00",
            estimated_landing_time_str="2023-01-01 11:00:00",
            global_window_duration_hours=1.0, 
            delta_t_seconds=0 # delta_t zero
        )
    except ValueError as e:
        print(f"Error (expected): {e}")
        
    # Scenario 5: global_window_duration_hours is zero, flight duration is zero
    params5 = calculate_aligned_time_parameters(
        takeoff_time_str="2023-01-01 10:00:00",
        estimated_landing_time_str="2023-01-01 10:00:00", 
        global_window_duration_hours=0.0, 
        delta_t_seconds=600
    )
    print(f"Scenario 5: {params5}")
    # Expected: min_time=36000, num_bins=1, takeoff_idx=0, landing_idx=0
    
    # Scenario 6: Test with non-integer global_window_duration_hours
    params6 = calculate_aligned_time_parameters(
        takeoff_time_str="2023-01-01 10:00:00",
        estimated_landing_time_str="2023-01-01 10:30:00", # 30 min flight
        global_window_duration_hours=0.75, # 45 min window
        delta_t_seconds=300 # 5 min bins
    )
    print(f"Scenario 6: {params6}")
    # actual_flight_duration_s = 1800s
    # global_window_duration_s = 0.75 * 3600 = 2700s
    # padding_s = 2700 - 1800 = 900s
    # padding_before_s = 450s
    # takeoff_s = 36000s
    # common_min_time_overall_seconds = 36000 - 450 = 35550s
    # num_time_bins = int(2700 / 300) + 1 = 9 + 1 = 10
    # takeoff_bin_idx = round((36000 - 35550) / 300) = round(450 / 300) = round(1.5) = 2
    # landing_s = 36000 + 1800 = 37800s
    # landing_bin_idx = round((37800 - 35550) / 300) = round(2250 / 300) = round(7.5) = 8
    # Expected: min_time=35550.0, num_bins=10, takeoff_idx=2, landing_idx=8 
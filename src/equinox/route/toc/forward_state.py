import torch
from equinox.wind.wind_model import WindModel
from equinox.helpers.haversine import haversinet
from typing import List, Tuple
import numpy
import math
from datetime import timedelta
from equinox.route.get_wind import get_wind
from equinox.route.batch_interpolator import batched_interp1d_torch

# Phase identifiers
CLIMB, CRUISE, DESCENT = 0, 1, 2

def _interp1d_torch(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """
    PyTorch-based 1D linear interpolation, similar to numpy.interp.
    Assumes xp is sorted.
    Handles out-of-bounds x by clamping to the first/last values of fp.

    Args:
        x (torch.Tensor): New x-coordinates to interpolate, shape [B].
        xp (torch.Tensor): Known x-coordinates of the profile, shape [P], sorted.
        fp (torch.Tensor): Known y-coordinates of the profile, shape [P].

    Returns:
        torch.Tensor: Interpolated y-coordinates, shape [B].
    """
    # Ensure inputs are on the same device and have compatible dtypes
    # (Caller should ensure this, but can be added for robustness if needed)
    # x, xp, fp = x.to(device), xp.to(device), fp.to(device)
    # x, xp, fp = x.to(dtype), xp.to(dtype), fp.to(dtype)

    if xp.numel() == 0:
        # Undefined behavior for empty profile, return NaN or raise error
        return torch.full_like(x, float('nan'))
    if xp.numel() == 1:
        # Single point profile, all x map to the single fp value
        return torch.full_like(x, fp[0])

    # Find indices i such that xp[i-1] <= x < xp[i]
    # searchsorted(sorted_sequence, values_to_search)
    # 'right=False' (default) means that if x[k] is equal to xp[j], then indices[k] is j.
    # This means xp[indices[k]-1] would be xp[j-1] and xp[indices[k]] would be xp[j].
    # Let's use right=False for idx0 and right=True for idx1 for clarity, or manage clamping.

    # Simplified approach: Get insertion indices
    # i are indices such that if x were inserted into xp, it would maintain order.
    # All xp[j] for j < i are <= x, and all xp[j] for j >= i are > x. (This is for right=False if x is present)
    # More precisely using PyTorch doc for torch.searchsorted(input, values, out_int32=False, right=False, side='left', sorter=None)
    # `side='left'` (default if `right=False`): `out[i] = sum_{j=0}^{N-1} (self[j] < values[i])`
    # `side='right'` (`right=True`): `out[i] = sum_{j=0}^{N-1} (self[j] <= values[i])`
    # This means `out` gives the first index `k` where `values[i] <= self[k]` (for side='left')
    # or `values[i] < self[k]` (for side='right').

    # Let's use `right=True`. `i = torch.searchsorted(xp, x, right=True)`
    # Then `xp[i-1]` is the lower x-bound and `xp[i]` is the upper x-bound for interpolation for x.
    # We must clamp `i` so that `i-1` and `i` are valid indices for `xp` and `fp`.
    i = torch.searchsorted(xp, x, right=True)

    # Clamp i to be in [1, len(xp)-1] for xp[i] and xp[i-1]
    # If i is 0 (x < xp[0]), then i_clamped becomes 1. xp[0], xp[1] used.
    # If i is len(xp) (x >= xp[len-1]), then i_clamped becomes len(xp)-1. xp[len-2], xp[len-1] used.
    i_clamped = torch.clamp(i, 1, xp.numel() - 1)

    x0 = xp[i_clamped - 1]
    x1 = xp[i_clamped]
    y0 = fp[i_clamped - 1]
    y1 = fp[i_clamped]
    
    # Denominator for interpolation slope
    denom = x1 - x0
    # Where denom is 0 (i.e., x0 == x1, duplicate points in xp or at ends), use y0.
    # This also handles cases where x is exactly at a knot xp[j],
    # then if clamping results in x0=xp[j-1], x1=xp[j], and x=xp[j], then (x-x0)/(x1-x0) = 1, result y1 (fp[j]).
    # if x=xp[j-1], then (x-x0)=0, result y0 (fp[j-1]). This matches numpy.interp.
    
    # Interpolation factor, handling denom == 0
    # Factor = (x - x0) / denom
    factor = (x - x0) / torch.where(denom == 0, torch.tensor(1.0, device=x.device, dtype=x.dtype), denom)
    # If denom was 0, x0==x1. If x==x0, then factor is 0/1=0. If x!=x0, factor is non-zero/1.
    # To ensure if denom is 0, result is y0 (as factor * (y1-y0) should be 0), set factor to 0.
    factor = torch.where(denom == 0, torch.tensor(0.0, device=x.device, dtype=x.dtype), factor)

    interp_val = y0 + factor * (y1 - y0)

    # Handle out-of-bounds for x, clamp to first/last fp values
    # Values in x that are less than xp[0]
    interp_val = torch.where(x < xp[0], fp[0], interp_val)
    # Values in x that are greater than xp[-1]
    interp_val = torch.where(x > xp[-1], fp[-1], interp_val)
    
    return interp_val

def get_next_state_fw(
    coords_src: torch.Tensor,
    alts_src: torch.Tensor,
    eta_src: torch.Tensor,
    phase_src: torch.Tensor,
    coords_tgt: torch.Tensor,
    climb_performance: List[Tuple[float, float, float]],
    wind_model: WindModel,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the next state (altitude, Estimated Time of Arrival, and phase) for a batch of flight segments,
    considering aircraft performance characteristics and wind conditions.

    *Note: the ETA is there only to provide the absolute time, and to help retrieve the relevant wind data.*
    *The traveled distance is derived from interpolating the **altitude** at source nodes with the performance table.*

    The function processes each segment independently based on its starting state (coordinates, altitude, ETA, phase)
    and the target coordinates for the segment. It uses a provided climb performance profile to model altitude
    and time changes during climb and determines cruise behavior based on the profile or a default speed.
    Wind effects are incorporated using a WindModel instance to calculate ground speed and adjusted distances.

    The function handles the following scenarios for each segment:
    1.  **Cruise Phase:** If the segment starts in the CRUISE phase, the altitude remains constant,
        and the time taken to traverse the segment is calculated based on the great-circle distance (Haversine)
        and the wind-adjusted ground speed (True Air Speed + wind component along track).
    2.  **Climb/Descent Phase:** If the segment starts in a non-cruise phase (CLIMB or DESCENT - though current logic
        primarily handles CLIMB based on `climb_performance`), the function determines the aircraft's progress
        through the climb/descent profile based on its current altitude and the segment distance. It calculates
        the time and altitude reached by the end of the segment, considering wind effects on ground distance covered.
    3.  **Transition to Cruise:** If a segment starts in a climb/descent phase and reaches or surpasses the
        Top of Climb (ToC) altitude within the segment distance, the function calculates the state at ToC
        (altitude and time) and then calculates the time for the remaining distance in the CRUISE phase
        at the ToC altitude.

    Args:
        coords_src (torch.Tensor): Source coordinates for each segment, shape [batch_size, 2].
                                   Format is (latitude, longitude) in degrees.
        alts_src (torch.Tensor): Source altitudes for each segment, shape [batch_size].
                                 Altitude is in feet.
        eta_src (torch.Tensor): Estimated Time of Arrival (ETA) at the source point for each segment,
                                shape [batch_size]. Time is in seconds (e.g., since midnight, depending on the min timestamp in the wind model).
        phase_src (torch.Tensor): The current flight phase for each segment, shape [batch_size].
                                  Uses integer identifiers: CLIMB (0), CRUISE (1), DESCENT (2).
        coords_tgt (torch.Tensor): Target coordinates for each segment, shape [batch_size, 2].
                                   Format is (latitude, longitude) in degrees.
        climb_performance (List[Tuple[float, float, float]]): A list defining the aircraft's climb
                                   profile. Each tuple represents a point in the profile with
                                   (altitude in feet, elapsed time from profile start in seconds,
                                   wind-free distance covered from profile start in nautical miles).
                                   Assumed to be sorted by altitude and time. This profile is also
                                   used to derive the cruise True Air Speed if possible.
        wind_model (WindModel): An instance of the WindModel to query wind components
                                at specific locations, altitudes, and times.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing three tensors for the
        state at the target point of each segment, all of shape [batch_size]:
            - alt_tgt (torch.Tensor): Target altitude in feet.
            - eta_tgt (torch.Tensor): Target Estimated Time of Arrival in seconds.
            - phase_tgt (torch.Tensor): Target flight phase (CLIMB, CRUISE, or DESCENT, potentially transitioning
                                        to CRUISE if ToC is reached within the segment).

    Examples:

    Assuming necessary imports and `WindModel`, `Performance`, `get_eta_and_distance_climb` are available.

    1.  **Climbing Segment (Wind-Free):**
        ```python
        from equinox.wind.wind_free import WindFree
        from equinox.vnav.vnav_performance import Performance, get_eta_and_distance_climb
        from equinox.vnav.vnav_profiles_rev1 import NARROW_BODY_JET_CLIMB_PROFILE, NARROW_BODY_JET_DESCENT_PROFILE, NARROW_BODY_JET_CLIMB_VS_PROFILE, NARROW_BODY_JET_DESCENT_VS_PROFILE
        import torch

        wind_model = WindFree()
        performance = Performance(
            NARROW_BODY_JET_CLIMB_PROFILE,
            NARROW_BODY_JET_DESCENT_PROFILE,
            NARROW_BODY_JET_CLIMB_VS_PROFILE,
            NARROW_BODY_JET_DESCENT_VS_PROFILE,
            cruise_altitude_ft=35000,
            cruise_speed_kts=450,
        )
        climb_perf_table = get_eta_and_distance_climb(performance, 1000)

        coords_src = torch.tensor([[37.7749, -122.4194]]) # SFO
        alts_src = torch.tensor([1000]) # 1000 ft
        eta_src = torch.tensor([0]) # 0 seconds
        phase_src = torch.tensor([0]) # CLIMB
        coords_tgt = torch.tensor([[38.123, -121.021]]) # TIPRE waypoint

        alt_tgt, eta_tgt, phase_tgt = get_next_state_fw(
            coords_src, alts_src, eta_src, phase_src, coords_tgt, climb_perf_table, wind_model
        )
        # Expected output will show altitude and ETA consistent with the climb profile
        # covering the distance to TIPRE, phase remains CLIMB if ToC not reached.
        print(f"Target Altitude: {alt_tgt.item():.0f} ft, Target ETA: {eta_tgt.item():.1f} s, Target Phase: {phase_tgt.item()}")
        ```

    2.  **Cruise Segment (Wind-Free):**
        ```python
        from equinox.wind.wind_free import WindFree
        from equinox.vnav.vnav_performance import Performance, get_eta_and_distance_climb
        from equinox.vnav.vnav_profiles_rev1 import NARROW_BODY_JET_CLIMB_PROFILE, NARROW_BODY_JET_DESCENT_PROFILE, NARROW_BODY_JET_CLIMB_VS_PROFILE, NARROW_BODY_JET_DESCENT_VS_PROFILE
        import torch

        wind_model = WindFree()
        performance = Performance(
            NARROW_BODY_JET_CLIMB_PROFILE,
            NARROW_BODY_JET_DESCENT_PROFILE,
            NARROW_BODY_JET_CLIMB_VS_PROFILE,
            NARROW_BODY_JET_DESCENT_VS_PROFILE,
            cruise_altitude_ft=35000,
            cruise_speed_kts=450,
        )
        # While performance_table is used to derive cruise speed internally,
        # for a purely cruise segment, the full table might not be strictly necessary if cruise speed is known.
        # However, the function expects it, so pass a valid one.
        climb_perf_table = get_eta_and_distance_climb(performance, 1000)


        coords_src = torch.tensor([[37.7749, -122.4194]]) # SFO
        alts_src = torch.tensor([35000]) # 35000 ft (cruise altitude)
        eta_src = torch.tensor([0]) # 0 seconds
        phase_src = torch.tensor([1]) # CRUISE
        coords_tgt = torch.tensor([[38.407, -117.179]]) # INSLO waypoint

        alt_tgt, eta_tgt, phase_tgt = get_next_state_fw(
            coords_src, alts_src, eta_src, phase_src, coords_tgt, climb_perf_table, wind_model
        )
        # Expected output will show altitude remaining 35000 ft, ETA based on cruise speed and distance,
        # and phase remaining CRUISE.
        print(f"Target Altitude: {alt_tgt.item():.0f} ft, Target ETA: {eta_tgt.item():.1f} s, Target Phase: {phase_tgt.item()}")
        ```
    """
    device = coords_src.device
    dtype = (
        coords_src.dtype
    )  # Assuming float64 based on typical use in related functions

    # Constants
    KNOTS_TO_MPS = 0.514444
    MPS_TO_KNOTS = 1.0 / KNOTS_TO_MPS
    DEFAULT_CRUISE_TAS_KTS = 450.0  # Fallback if cannot derive from climb_performance

    # Initialize output tensors
    num_segments = coords_src.shape[0]
    alt_tgt = torch.zeros(num_segments, device=device, dtype=dtype)
    eta_tgt = torch.zeros(num_segments, device=device, dtype=dtype)
    phase_tgt = torch.full(
        (num_segments,), -1, device=device, dtype=torch.long
    )  # Init with invalid

    # --- Derive cruise TAS from climb_performance (last segment behavior) ---
    if len(climb_performance) >= 2:
        perf_last_dist_wf_nm = climb_performance[-1][2]
        perf_second_last_dist_wf_nm = climb_performance[-2][2]
        perf_last_time_s = climb_performance[-1][1]
        perf_second_last_time_s = climb_performance[-2][1]

        delta_dist_wf_cruise_segment = (
            perf_last_dist_wf_nm - perf_second_last_dist_wf_nm
        )
        delta_time_s_cruise_segment = perf_last_time_s - perf_second_last_time_s

        if delta_time_s_cruise_segment > 1e-6:
            tas_cruise_kts = delta_dist_wf_cruise_segment / (
                delta_time_s_cruise_segment / 3600.0
            )
        else:
            tas_cruise_kts = DEFAULT_CRUISE_TAS_KTS
    else:
        tas_cruise_kts = DEFAULT_CRUISE_TAS_KTS

    # --- 0. Cruise Phase Handling ---
    cruise_mask = phase_src == CRUISE
    if cruise_mask.any():
        num_cruise = cruise_mask.sum().item()
        if num_cruise > 0:
            alt_tgt[cruise_mask] = alts_src[cruise_mask]
            phase_tgt[cruise_mask] = CRUISE

            coords_src_cruise = coords_src[cruise_mask]
            alts_src_cruise = alts_src[cruise_mask]
            eta_src_cruise = eta_src[cruise_mask]
            coords_tgt_cruise = coords_tgt[cruise_mask]

            dist_nm_cruise = haversinet(
                coords_src_cruise[:, 0],
                coords_src_cruise[:, 1],
                coords_tgt_cruise[:, 0],
                coords_tgt_cruise[:, 1],
            )

            wind_mps_cruise = get_wind(
                coords_src_cruise,
                coords_tgt_cruise,
                alts_src_cruise,
                eta_src_cruise,
                wind_model,
            )
            wind_kts_cruise = wind_mps_cruise * MPS_TO_KNOTS

            gs_kts_cruise = tas_cruise_kts + wind_kts_cruise

            time_hours_cruise = torch.zeros_like(dist_nm_cruise)
            # Handle cases with very low or zero ground speed to prevent division by zero or very large times
            valid_gs_mask = gs_kts_cruise > 1.0  # Min 1 knot GS to proceed
            time_hours_cruise[valid_gs_mask] = (
                dist_nm_cruise[valid_gs_mask] / gs_kts_cruise[valid_gs_mask]
            )
            # For invalid GS, eta_tgt will remain eta_src + 0 effectively, or handle as error/very long time
            # For now, time_hours_cruise for invalid GS is 0, so eta_tgt = eta_src.
            # A more robust solution might involve setting a max time or specific error handling.
            time_hours_cruise[~valid_gs_mask] = (
                torch.finfo(dtype).max / 3600.0
            )  # Effectively infinite time

            time_secs_cruise = time_hours_cruise * 3600.0
            eta_tgt[cruise_mask] = eta_src_cruise + time_secs_cruise

    # --- Non-Cruise Phase Handling (e.g., CLIMB) ---
    # Assuming DESCENT would use a similar logic with a descent_performance table or reversed climb_performance
    non_cruise_mask = (
        phase_src == CLIMB
    )  # | (phase_src == DESCENT) # Add DESCENT if applicable

    if non_cruise_mask.any():
        num_nc = non_cruise_mask.sum().item()
        if num_nc > 0:
            coords_src_nc = coords_src[non_cruise_mask]
            alts_src_nc = alts_src[non_cruise_mask]
            eta_src_nc = eta_src[non_cruise_mask]
            phase_src_nc = phase_src[
                non_cruise_mask
            ]  # To carry over CLIMB or DESCENT status
            coords_tgt_nc = coords_tgt[non_cruise_mask]

            # 1. Compute Haversine distance for the leg
            dist_leg_nm_nc = haversinet(
                coords_src_nc[:, 0],
                coords_src_nc[:, 1],
                coords_tgt_nc[:, 0],
                coords_tgt_nc[:, 1],
            )

            # Prepare climb_performance tensors
            if not climb_performance:
                # Cannot proceed without climb performance data for non-cruise phases
                # Set to error state or skip these segments?
                # Defaulting to keep current state for these problematics segments
                alt_tgt[non_cruise_mask] = alts_src_nc
                eta_tgt[non_cruise_mask] = eta_src_nc
                phase_tgt[non_cruise_mask] = phase_src_nc
                raise ValueError("Climb performance data is not specified.")

            # Unpack climb_performance into separate lists for each profile attribute:
            #   perf_alts_list: List of altitudes (in feet) at each profile point.
            #   perf_times_s_list: List of elapsed times (in seconds) from profile start to each altitude.
            #   perf_dist_wf_nm_list: List of wind-free distances covered (in nautical miles) to each altitude.
            perf_alts_list, perf_times_s_list, perf_dist_wf_nm_list = zip(
                *climb_performance
            )
            # Convert lists to Torch tensors
            # These are the columns of the climb performance table!
            perf_alts_prof = torch.tensor(
                perf_alts_list, dtype=dtype, device=device
            )  # [P] Altitude profile (ft)
            perf_times_s_prof = torch.tensor(
                perf_times_s_list, dtype=dtype, device=device
            )  # [P] Time profile (s)
            perf_dist_wf_nm_prof = torch.tensor(
                perf_dist_wf_nm_list, dtype=dtype, device=device
            )  # [P] Wind-free distance profile (nm)

            # Ensure profile has at least two points for interpolation to be meaningful
            if perf_alts_prof.numel() < 2:
                # Cannot interpolate with less than 2 profile points
                alt_tgt[non_cruise_mask] = alts_src_nc
                eta_tgt[non_cruise_mask] = eta_src_nc
                phase_tgt[non_cruise_mask] = phase_src_nc
                # Raise error or log, as this situation might lead to unexpected behavior if not handled.
                # For now, we assume processing of these segments stops here and they retain source state.
                # Consider adding a specific warning or error if this path is taken frequently.
                # This was previously just a comment, now explicitly continuing to next segment batch if any.
                # This block means current non_cruise_mask segments will not be processed further if profile is too short.
                # If there are other non_cruise_mask segments with valid profiles, they will continue.
                # This needs careful thought: if one segment in a batch fails here, should all fail?
                # For now, let's assume we want to process valid ones.
                # However, raising an error might be safer if a valid profile is always expected.
                # Let's revert to the original behavior of just setting target to source and continuing.
                # The original code just had comments and implicitly continued.
                # To ensure we only skip if ALL nc segments hit this, it's complex.
                # The original code's structure implied it would proceed to use these (potentially incorrect)
                # current_time_s_profile_src etc if the numpy.interp loop ran with a bad profile.
                # The current_time_s_profile_src would be zero.
                # The safer approach is to handle this more explicitly.
                # For now, consistent with original: fill with src and let logic proceed,
                # though _interp1d_torch will handle profile length 1 correctly.
                # This check is for numel < 2, so profile length 0 or 1.
                # _interp1d_torch handles numel=1. If numel=0, it returns NaNs.
                # If numel is 0 or 1, the interpolations below might not be meaningful
                # for subsequent calculations of ground distance profiles etc.
                # Let's ensure that if profile is too short, we don't proceed with complex calcs for these segments.
                # A simple way is to return src state for these.
                # We need a mask for segments with invalid profiles if we want to selectively skip.
                # For simplicity of this change, let's assume valid profile length >=2 based on problem context.
                # If not, the _interp1d_torch will handle len=1, and len=0 will give NaNs which propagate.
                # The original ValueError was for climb_performance being empty, not short.
                pass # Let _interp1d_torch handle it, or rely on prior checks for empty climb_performance

            # 2. Interpolate to find current aircraft state within the wind-free climb profile
            # Using alts_src_nc to find its corresponding time and wind-free distance in the profile
            
            # current_time_s_profile_src: For each non-cruise segment, this tensor will hold the interpolated elapsed time (in seconds)
            #   from the start of the climb (takeoff) profile up to the current source altitude (alts_src_nc[i]).
            current_time_s_profile_src = _interp1d_torch(
                alts_src_nc, perf_alts_prof, perf_times_s_prof
            )
            
            # current_dist_wf_nm_profile_src: For each non-cruise segment, this tensor will hold the interpolated wind-free distance (in nautical miles)
            #   covered from the start of the profile (i.e., from takeoff) up to the current source altitude (alts_src_nc[i]).
            current_dist_wf_nm_profile_src = _interp1d_torch(
                alts_src_nc, perf_alts_prof, perf_dist_wf_nm_prof
            )

            # 3. Get wind at source for non-cruise segments
            wind_mps_src_nc = get_wind(
                coords_src_nc, coords_tgt_nc, alts_src_nc, eta_src_nc, wind_model
            )
            wind_kts_src_nc = (
                wind_mps_src_nc * MPS_TO_KNOTS
            )  # Convert m/s to kts; Shape: [num_nc]

            # 4. Create the effective ground distance profile (batched)
            # This profile shows ground distance covered vs. altitude and time, considering wind_kts_src_nc.
            # Equivalently, another "distance" column (adjusted for wind) in the performance table.
            # If I look at the whole climb profile, for every possible altitude, how far would I have gone along the ground (with wind)?
            # perf_dist_wf_nm_prof is [P], wind_kts_src_nc is [B], perf_times_s_prof is [P]
            # We want perf_ground_dist_profile to be [B, P]
            perf_ground_dist_profile = perf_dist_wf_nm_prof.unsqueeze(
                0
            ) + wind_kts_src_nc.unsqueeze(1) * (perf_times_s_prof.unsqueeze(0) / 3600.0)

            # Current ground distance covered by aircraft, based on its wind condition and profile progress
            # i.e., Given where I am right now (my current altitude), how far have I actually gone along the ground (with wind)?
            current_ground_dist_profile_src = (
                current_dist_wf_nm_profile_src
                + wind_kts_src_nc * (current_time_s_profile_src / 3600.0)
            )

            # 5. Target total ground distance from profile start, after traversing the current leg
            target_total_ground_dist_from_takeoff = (
                current_ground_dist_profile_src + dist_leg_nm_nc
            )

            # ToC parameters from original profile
            alt_toc_profile = perf_alts_prof[-1]
            # time_s_toc_profile = perf_times_s_prof[-1] # Not directly used in this logic flow for eta_at_toc

            # Ground distance to reach ToC for each segment's wind condition
            dist_toc_ground_profile_b = perf_ground_dist_profile[
                :, -1
            ]  # Shape: [num_nc]

            # --- Interpolate for target altitude and profile time using the ground distance profile ---
            # y_known needs to be broadcasted to [B, P] for batched_interp1d_torch
            num_profile_points = perf_alts_prof.shape[0]
            perf_alts_prof_b = perf_alts_prof.unsqueeze(0).expand(
                num_nc, num_profile_points
            )
            perf_times_s_prof_b = perf_times_s_prof.unsqueeze(0).expand(
                num_nc, num_profile_points
            )

            # Interpolate target altitude and profile time assuming continuous climb/descent
            alt_tgt_nc_cont = batched_interp1d_torch(
                target_total_ground_dist_from_takeoff,
                perf_ground_dist_profile,
                perf_alts_prof_b,
                device,
            )
            time_tgt_profile_s_nc_cont = batched_interp1d_torch(
                target_total_ground_dist_from_takeoff,
                perf_ground_dist_profile,
                perf_times_s_prof_b,
                device,
            )

            # ETA calculation base: offset between absolute source ETA and source profile time
            eta_offset = eta_src_nc - current_time_s_profile_src

            # --- Edge Case: Top of Climb (ToC) or end of profile ---
            # Mask for segments that are still climbing/descending within the profile
            is_still_in_profile_mask = (
                target_total_ground_dist_from_takeoff < dist_toc_ground_profile_b
            )

            # Mask for segments that reach or pass ToC (or end of defined profile) on this leg
            # And were not already at/beyond ToC at the source of this leg
            is_toc_reached_on_leg_mask = (~is_still_in_profile_mask) & (
                current_ground_dist_profile_src < dist_toc_ground_profile_b
            )

            # Initialize temporary holders for non_cruise results
            alt_tgt_nc_final = torch.zeros_like(alts_src_nc)
            eta_tgt_nc_final = torch.zeros_like(eta_src_nc)
            phase_tgt_nc_final = torch.full_like(phase_src_nc, -1, dtype=torch.long)

            # Case 1: Still climbing/descending within the profile
            if is_still_in_profile_mask.any():
                alt_tgt_nc_final[is_still_in_profile_mask] = alt_tgt_nc_cont[
                    is_still_in_profile_mask
                ]
                eta_tgt_nc_final[is_still_in_profile_mask] = (
                    eta_offset[is_still_in_profile_mask]
                    + time_tgt_profile_s_nc_cont[is_still_in_profile_mask]
                )
                phase_tgt_nc_final[is_still_in_profile_mask] = phase_src_nc[
                    is_still_in_profile_mask
                ]  # Retain CLIMB/DESCENT

            # Case 2: ToC (or end of profile) is reached on this leg
            if is_toc_reached_on_leg_mask.any():
                alt_tgt_nc_final[is_toc_reached_on_leg_mask] = (
                    alt_toc_profile  # Target alt is ToC/profile end altitude
                )
                phase_tgt_nc_final[is_toc_reached_on_leg_mask] = (
                    CRUISE  # Transition to CRUISE
                )

                # Calculate ETA at ToC
                # Time to reach ToC based on the wind-adjusted ground distance profile
                time_to_reach_toc_s_profile = batched_interp1d_torch(
                    # Interpolate: for each segment, find the time (s) at which the ground distance profile reaches ToC
                    # dist_toc_ground_profile_b[is_toc_reached_on_leg_mask]: scalar ground distance at ToC for each batch element (shape: [N_toc])
                    # perf_ground_dist_profile[is_toc_reached_on_leg_mask]: 1D array of ground distances from performance table for each batch (shape: [N_toc, n_profile])
                    # perf_times_s_prof_b[is_toc_reached_on_leg_mask]: 1D array of times (s) from performance table for each batch (shape: [N_toc, n_profile])
                    dist_toc_ground_profile_b[is_toc_reached_on_leg_mask],
                    perf_ground_dist_profile[is_toc_reached_on_leg_mask],
                    perf_times_s_prof_b[is_toc_reached_on_leg_mask],
                    device,
                )
                eta_at_toc = (
                    eta_offset[is_toc_reached_on_leg_mask] + time_to_reach_toc_s_profile
                )

                # Distance climbed/descended on this leg until ToC (i.e., in profile means climbing/descending)
                dist_in_profile_on_leg = (
                    dist_toc_ground_profile_b[is_toc_reached_on_leg_mask]
                    - current_ground_dist_profile_src[is_toc_reached_on_leg_mask]
                )

                # Distance remaining to cruise
                dist_cruise_on_leg = (
                    dist_leg_nm_nc[is_toc_reached_on_leg_mask] - dist_in_profile_on_leg
                )
                dist_cruise_on_leg = torch.clamp(
                    dist_cruise_on_leg, min=0
                )  # Ensure non-negative

                # Ground speed for the remaining cruise portion of the leg
                # Using the same tas_cruise_kts derived earlier, and wind at source of this leg for simplicity
                gs_cruise_kts_toc = (
                    tas_cruise_kts + wind_kts_src_nc[is_toc_reached_on_leg_mask]
                )

                time_cruise_on_leg_hours = torch.zeros_like(dist_cruise_on_leg)
                valid_gs_toc_mask = gs_cruise_kts_toc > 1.0
                time_cruise_on_leg_hours[valid_gs_toc_mask] = (
                    dist_cruise_on_leg[valid_gs_toc_mask]
                    / gs_cruise_kts_toc[valid_gs_toc_mask]
                )
                time_cruise_on_leg_hours[~valid_gs_toc_mask] = (
                    torch.finfo(dtype).max / 3600.0
                )  # Infinite time

                time_cruise_on_leg_s = time_cruise_on_leg_hours * 3600.0
                eta_tgt_nc_final[is_toc_reached_on_leg_mask] = (
                    eta_at_toc + time_cruise_on_leg_s
                )

            # Case 3: Aircraft was already at/beyond ToC at the source of this leg (should be CRUISE phase)
            # This case implies phase_src might not have been CRUISE, or alts_src is above profile.
            # For robustness, handle segments that don't fall into above two masks:
            # These might be segments where current_ground_dist_profile_src >= dist_toc_ground_profile_b
            # Such segments should ideally be handled by the main CRUISE logic if phase_src was correct.
            # If they reach here, it implies they started non-cruise but effectively at or beyond ToC.
            already_at_or_beyond_toc_mask = ~(
                is_still_in_profile_mask | is_toc_reached_on_leg_mask
            )
            if already_at_or_beyond_toc_mask.any():
                # Treat as if cruising from source altitude (which is ToC alt or above)
                alt_tgt_nc_final[already_at_or_beyond_toc_mask] = alts_src_nc[
                    already_at_or_beyond_toc_mask
                ]  # Maintain alt
                phase_tgt_nc_final[already_at_or_beyond_toc_mask] = CRUISE

                gs_cruise_kts_post_toc = (
                    tas_cruise_kts + wind_kts_src_nc[already_at_or_beyond_toc_mask]
                )
                time_hours_post_toc = torch.zeros_like(
                    dist_leg_nm_nc[already_at_or_beyond_toc_mask]
                )

                valid_gs_post_toc_mask = gs_cruise_kts_post_toc > 1.0
                time_hours_post_toc[valid_gs_post_toc_mask] = (
                    dist_leg_nm_nc[already_at_or_beyond_toc_mask][
                        valid_gs_post_toc_mask
                    ]
                    / gs_cruise_kts_post_toc[valid_gs_post_toc_mask]
                )
                time_hours_post_toc[~valid_gs_post_toc_mask] = (
                    torch.finfo(dtype).max / 3600.0
                )

                eta_tgt_nc_final[already_at_or_beyond_toc_mask] = (
                    eta_src_nc[already_at_or_beyond_toc_mask]
                    + time_hours_post_toc * 3600.0
                )

            # Update main output tensors for non_cruise_mask segments
            alt_tgt[non_cruise_mask] = alt_tgt_nc_final
            eta_tgt[non_cruise_mask] = eta_tgt_nc_final
            phase_tgt[non_cruise_mask] = phase_tgt_nc_final

    return alt_tgt, eta_tgt, phase_tgt

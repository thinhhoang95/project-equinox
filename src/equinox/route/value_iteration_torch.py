from typing import List, Tuple
import torch
from torch_scatter import scatter_logsumexp, scatter_add
from equinox.helpers.haversine import haversinet
from equinox.wind.wind_model import WindModel

import math
from datetime import timedelta
import numpy

# Phase identifiers
CLIMB, CRUISE, DESCENT = 0, 1, 2

# def get_heuristic_altitude(coords_src: torch.Tensor,
#                            alts_src: torch.Tensor,
#                            eta_src: torch.Tensor,
#                            phase_src: torch.Tensor,
#                            wind_src: torch.Tensor,
#                            coords_tgt: torch.Tensor,
#                            climb_performance: List[Tuple[float, float]]) -> torch.Tensor:

#     # Initialize target altitude tensor
#     alt_tgt = torch.zeros_like(alts_src)

#     # Handle cruise phase: retain source altitude
#     cruise_mask = phase_src == CRUISE
#     alt_tgt[cruise_mask] = alts_src[cruise_mask]

#     # Handle climb/descent: perform linear interpolation from climb_performance
#     non_cruise_mask = ~cruise_mask
#     if non_cruise_mask.any():
#         # Compute ground speed: convert wind from m/s to knots (1 m/s ≈ 1.94384 knots)
#         wind_knots = wind_src[non_cruise_mask] * 1.94384
#         gs = V_src[non_cruise_mask] + wind_knots

#         # Compute distance in nautical miles
#         lat_src = coords_src[non_cruise_mask, 0]
#         lon_src = coords_src[non_cruise_mask, 1]
#         lat_tgt = coords_tgt[non_cruise_mask, 0]
#         lon_tgt = coords_tgt[non_cruise_mask, 1]
#         dist_nm = haversine_distance(lat_src, lon_src, lat_tgt, lon_tgt)

#         # Compute time to target in seconds
#         time_hours = dist_nm / gs
#         time_seconds = time_hours * 3600.0

#         # Compute target ETA
#         eta_tgt = eta_src[non_cruise_mask] + time_seconds

#         # Prepare climb performance arrays
#         perf_alts, perf_etas = zip(*climb_performance)
#         perf_alts = torch.tensor(perf_alts, dtype=alts_src.dtype, device=alts_src.device)
#         perf_etas = torch.tensor(perf_etas, dtype=eta_src.dtype, device=alts_src.device)

#         # Find interpolation indices
#         idx = torch.bucketize(eta_tgt, perf_etas)
#         idx_lo = torch.clamp(idx - 1, 0, len(perf_etas) - 1)
#         idx_hi = torch.clamp(idx, 0, len(perf_etas) - 1)

#         eta_lo = perf_etas[idx_lo]
#         eta_hi = perf_etas[idx_hi]
#         alt_lo = perf_alts[idx_lo]
#         alt_hi = perf_alts[idx_hi]

#         # Avoid division by zero for identical eta points
#         denom = eta_hi - eta_lo
#         slope = torch.where(denom > 0, (alt_hi - alt_lo) / denom, torch.zeros_like(denom))

#         # Interpolate altitude
#         alt_interp = alt_lo + slope * (eta_tgt - eta_lo)

#         alt_tgt[non_cruise_mask] = alt_interp

#     return alt_tgt


def batched_interp1d_torch(
    x_new_batched: torch.Tensor,
    x_known_batched: torch.Tensor,
    y_known_batched: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Performs batched 1D linear interpolation.
    x_new_batched: [B], points to interpolate at for each batch item.
    x_known_batched: [B, P], sorted x-coordinates for each batch item.
    y_known_batched: [B, P], y-coordinates corresponding to x_known_batched.
    device: torch.device to use for new tensors.
    Returns y_new_batched: [B]
    """
    # Ensure x_new is clipped to the range of each row in x_known_batched
    min_x_known = x_known_batched[:, 0]
    max_x_known = x_known_batched[:, -1]
    # Ensure x_new_batched has same shape as min_x_known for clamp if it's a scalar
    if x_new_batched.ndim == 0:
        x_new_batched_expanded = x_new_batched.expand_as(min_x_known)
    else:
        x_new_batched_expanded = x_new_batched

    x_new_clipped = torch.clamp(x_new_batched_expanded, min_x_known, max_x_known)

    idx_right = torch.searchsorted(
        x_known_batched, x_new_clipped.unsqueeze(1), right=True
    )
    idx_right = torch.clamp(idx_right, 1, x_known_batched.shape[1] - 1)
    idx_left = idx_right - 1

    batch_indices = torch.arange(x_known_batched.shape[0], device=device).unsqueeze(1)

    x_left = torch.gather(x_known_batched, 1, idx_left)
    x_right = torch.gather(x_known_batched, 1, idx_right)
    y_left = torch.gather(y_known_batched, 1, idx_left)
    y_right = torch.gather(y_known_batched, 1, idx_right)

    denom = x_right - x_left
    # Avoid division by zero: if denom is very small, assume x_new_clipped is at x_left or x_right
    # If x_left == x_right, weight is 0 if x_new_clipped == x_left, or could be nan if not handled.
    # A simple safe way: if denom is zero, result is y_left.
    weight_right = torch.where(
        denom > 1e-9,
        (x_new_clipped.unsqueeze(1) - x_left) / denom,
        torch.zeros_like(denom),
    )

    y_new_interp = y_left + weight_right * (y_right - y_left)
    return y_new_interp.squeeze(1)


def get_next_state(
    coords_src: torch.Tensor,
    alts_src: torch.Tensor,
    eta_src: torch.Tensor,
    phase_src: torch.Tensor,
    coords_tgt: torch.Tensor,
    climb_performance: List[Tuple[float, float, float]],
    wind_model: WindModel,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the next state (altitude, ETA, phase) for a batch of flight segments.

    climb_performance: List of tuples [altitude (ft), time (s), wind_free_covered_distance (nm)]
                       Defines the aircraft's climb profile from takeoff (or a reference start).
                       Assumed to be sorted by altitude and time. Time is relative to the takeoff time.
    wind_model: Instance of WindModel for querying wind data.

    Returns:
        alt_tgt (torch.Tensor): Target altitude in feet for each segment.
        eta_tgt (torch.Tensor): Target Estimated Time of Arrival in seconds for each segment.
        phase_tgt (torch.Tensor): Target phase (CLIMB, CRUISE, DESCENT) for each segment.
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
            if len(perf_alts_list) < 2:
                # Cannot interpolate with less than 2 profile points
                alt_tgt[non_cruise_mask] = alts_src_nc
                eta_tgt[non_cruise_mask] = eta_src_nc
                phase_tgt[non_cruise_mask] = phase_src_nc
                # Again, consider logging or raising an error
                # This structure assumes we continue after this if-block, so let's structure to skip calculations
                # For the edit, we'll assume a valid profile length and proceed.

            # 2. Interpolate to find current aircraft state within the wind-free climb profile
            # Using alts_src_nc to find its corresponding time and wind-free distance in the profile
            # batched_interp1d_torch expects batched x_known and y_known.
            # Here, perf_alts_prof is 1D, alts_src_nc is [B]. Need to expand profile for batched call or use a loop/smarter 1D interp.
            # For simplicity, let's use a standard 1D interpolation, assuming a single profile for all nc segments.
            # This implies a loop for each nc segment if we were to use the current batched_interp1d_torch as is for this step.
            # Or, adapt interp1d for 1D x_known, batched x_new.

            # Corrected interpolation for step 2 (using a simpler 1D approach for this part)
            # This will use broadcasting if alts_src_nc is a tensor and perf_alts_prof is 1D.

            # current_time_s_profile_src: For each non-cruise segment, this tensor will hold the interpolated elapsed time (in seconds)
            #   from the start of the climb (takeoff) profile up to the current source altitude (alts_src_nc[i]).
            current_time_s_profile_src = torch.zeros_like(alts_src_nc)
            # current_dist_wf_nm_profile_src: For each non-cruise segment, this tensor will hold the interpolated wind-free distance (in nautical miles)
            #   covered from the start of the profile (i.e., from takeoff) up to the current source altitude (alts_src_nc[i]).
            current_dist_wf_nm_profile_src = torch.zeros_like(alts_src_nc)
            for i in range(num_nc):  # iterate over each non-cruise segment
                # Crude 1D interp for each item; ideally vectorize or use a more robust 1D PyTorch interp.
                # For now, using a PyTorch-idiomatic equivalent of np.interp:
                # torch.from_numpy fails if numpy.interp returns a scalar (np.float64), so wrap in float() and use torch.tensor directly
                current_time_s_profile_src[i] = torch.tensor(
                    float(
                        numpy.interp(
                            alts_src_nc[i].cpu().numpy(),
                            perf_alts_prof.cpu().numpy(),
                            perf_times_s_prof.cpu().numpy(),
                        )
                    ),
                    device=device,
                    dtype=dtype,
                )
                current_dist_wf_nm_profile_src[i] = torch.tensor(
                    float(
                        numpy.interp(
                            alts_src_nc[i].cpu().numpy(),
                            perf_alts_prof.cpu().numpy(),
                            perf_dist_wf_nm_prof.cpu().numpy(),
                        )
                    ),
                    device=device,
                    dtype=dtype,
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


def get_wind(
    coords_src: torch.Tensor,
    coords_tgt: torch.Tensor,
    altitude: torch.Tensor,
    eta_src: torch.Tensor,
    wind_model: WindModel,
) -> torch.Tensor:
    """
    coords_src: [E,2] (lat, lon) in degrees
    coords_tgt: [E,2] (lat, lon) in degrees
    altitude: [E] in feet
    eta_src: [E] arrival time at source in seconds
    wind_model: WindModel instance
    returns: [E] along-track wind speed in m/s (positive tailwind)
    """
    # Move data to CPU numpy arrays
    lat_src = coords_src[:, 0].cpu().numpy()
    lon_src = coords_src[:, 1].cpu().numpy()
    lat_tgt = coords_tgt[:, 0].cpu().numpy()
    lon_tgt = coords_tgt[:, 1].cpu().numpy()
    alt_ft = altitude.cpu().numpy()
    eta_sec = eta_src.cpu().numpy()

    wind_along = []
    for la, lo, la2, lo2, altv, eta in zip(
        lat_src, lon_src, lat_tgt, lon_tgt, alt_ft, eta_sec
    ):
        # compute query time relative to model's earliest time
        query_time = wind_model._time_min + timedelta(seconds=float(eta))
        u, v = wind_model.get_wind_components(
            float(la), float(lo), float(altv), query_time
        )
        # if data missing, assume zero wind
        if u is None or v is None or math.isnan(u) or math.isnan(v):
            wind_along.append(0.0)
            continue
        # bearing from source to target (radians, from north clockwise)
        phi1 = math.radians(la)
        phi2 = math.radians(la2)
        dlon = math.radians(lo2 - lo)
        x = math.sin(dlon) * math.cos(phi2)
        y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(
            phi2
        ) * math.cos(dlon)
        bearing = math.atan2(x, y)
        # unit vector components: east = sin(bearing), north = cos(bearing)
        e = math.sin(bearing)
        n = math.cos(bearing)
        # project wind vector onto track
        wind_along.append(u * e + v * n)
    # return as tensor on original device/dtype
    return torch.tensor(wind_along, dtype=altitude.dtype, device=altitude.device)


def compute_speed(distance: torch.Tensor, wind: torch.Tensor) -> torch.Tensor:
    """
    distance: [E]
    wind: [E]
    returns: [E] groundspeed
    """
    # Placeholder: true airspeed = 450 knots
    return torch.full_like(distance, 450.0) + wind


def infer_phase(
    eta: torch.Tensor, climb_performance: List[Tuple[float, float]]
) -> torch.Tensor:
    """
    eta: [N] expected arrival time
    returns: [N] phase id
    """
    # Placeholder: all cruise
    return torch.full_like(eta, CRUISE, dtype=torch.long)


class CostModel(torch.nn.Module):
    """
    A learnable cost model c(features; theta).
    For demonstration, we use a linear model on:
       [distance, speed, wind]
    """

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: [E, 3]
        # returns: [E] cost
        return self.linear(features).squeeze(-1)


def vectorized_forward_pass(
    edge_index: torch.LongTensor,
    edge_dist: torch.Tensor,
    node_coords: torch.Tensor,
    topo_layers: list,
    CTOT: float,
    cost_model: torch.nn.Module,
    wind_model: WindModel,
    climb_performance: List[Tuple[float, float]],
):
    """
    edge_index: [2, E] tensor of (src, tgt)
    edge_dist: [E] distances
    node_coords: [N, 2] (lat, lon)
    topo_layers: list of lists, each sublist is a set of node indices in topological order
    CTOT: scalar, take-off time
    cost_model: learnable cost function

    returns: V [N], EETA [N], phase [N]
    """
    device = edge_index.device
    N = node_coords.size(0)  # Num of nodes
    E = edge_dist.size(0)  # Num of edges

    # Initialize state [V_TAS (kts), EETA (seconds), phase = [0, 1, 2], alt (feet)
    V = torch.full((N,), float("inf"), dtype=torch.double, device=device)
    EETA = torch.zeros((N,), dtype=torch.double, device=device)
    phase = torch.zeros((N,), dtype=torch.long, device=device)
    alt = torch.zeros((N,), dtype=torch.double, device=device)

    # Source is assumed layer 0, single node
    s = topo_layers[0][0]
    V[s] = 0.0
    EETA[s] = CTOT
    phase[s] = CLIMB
    alt[s] = 0.0  # TODO: Update with the departure airport elevation (ft)

    # Pre-split edge_index
    src = edge_index[0]
    tgt = edge_index[1]

    for layer in topo_layers[1:]:
        layer_idx = torch.tensor(layer, dtype=torch.long, device=device)
        # Select edges entering this layer
        mask = torch.isin(tgt, layer_idx)
        src_e = src[mask]
        tgt_e = tgt[mask]
        dist_e = edge_dist[mask]

        # Gather source state
        EETA_src = EETA[src_e]
        V_src = V[src_e]
        phase_src = phase[src_e]
        coords_src = node_coords[src_e]
        coords_tgt = node_coords[tgt_e]
        alts_src = alt[src_e]

        # Get the wind at the source nodes
        wind_s = get_wind(coords_src, coords_tgt, alts_src, EETA_src, wind_model)

        # Altitude & wind, heuristically estimated for the targets
        alt_e = get_heuristic_altitude(
            coords_src,
            alts_src,
            EETA_src,
            phase_src,
            wind_s,
            V_src,
            coords_tgt,
            climb_performance,
        )
        wind_e = get_wind(coords_src, coords_tgt, alt_e, EETA_src, wind_model)

        # Speed, time, cost
        speed_e = compute_speed(dist_e, wind_e)
        time_e = dist_e / speed_e
        feats = torch.stack([dist_e, speed_e, wind_e], dim=1)
        cost_e = cost_model(feats)

        # Log-sum-exp for V
        M_e = -cost_e - V_src
        V_new = -scatter_logsumexp(M_e, tgt_e, dim=0, dim_size=N)

        # Expected ETA
        P_e = torch.exp(M_e + V_new[tgt_e])
        EETA_new = scatter_add(P_e * (EETA_src + time_e), tgt_e, dim=0, dim_size=N)

        # Update only this layer
        V[layer_idx] = V_new[layer_idx]
        EETA[layer_idx] = EETA_new[layer_idx]
        phase[layer_idx] = infer_phase(EETA[layer_idx])

    return V, EETA, phase


if __name__ == "__main__":
    # Example usage
    # N nodes, E edges
    N, E = 5, 7
    edge_index = torch.tensor(
        [[0, 0, 1, 1, 2, 2, 3], [1, 2, 2, 3, 3, 4, 4]], dtype=torch.long  # src  # tgt
    )
    edge_dist = torch.rand(E).double() * 500.0
    node_coords = torch.rand(N, 2).double() * 180 - 90
    topo_layers = [[0], [1, 2], [3], [4]]
    CTOT = 0.0
    cost_model = CostModel().double()

    V, EETA, phase = vectorized_forward_pass(
        edge_index, edge_dist, node_coords, topo_layers, CTOT, cost_model
    )
    print("V:", V)
    print("EETA:", EETA)
    print("phase:", phase)

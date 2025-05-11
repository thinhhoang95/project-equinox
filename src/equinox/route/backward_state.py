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

def get_backward_state(
    coords_src: torch.Tensor,
    alts_t: torch.Tensor,
    eta_t: torch.Tensor,
    phase_t: torch.Tensor,
    coords_tgt: torch.Tensor,
    descent_performance: List[Tuple[float, float, float]],
    wind_model: WindModel,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Args:
        coords_src (torch.Tensor): Source coordinates for each segment, shape [batch_size, 2].
                                   Format is (latitude, longitude) in degrees.
        alts_t (torch.Tensor): Source altitudes for each segment, shape [batch_size].
                                 Altitude is in feet.
        eta_t (torch.Tensor): Estimated Time of Arrival (ETA) at the source point for each segment,
                                shape [batch_size]. Time is in seconds (e.g., since midnight, depending on the min timestamp in the wind model).
        phase_t (torch.Tensor): The current flight phase for each segment, shape [batch_size].
                                  Uses integer identifiers: CLIMB (0), CRUISE (1), DESCENT (2).
        coords_tgt (torch.Tensor): Target coordinates for each segment, shape [batch_size, 2].
                                   Format is (latitude, longitude) in degrees.
        descent_performance (List[Tuple[float, float, float]]): A list defining the aircraft's climb
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
    """
    device = coords_src.device
    dtype = (
        coords_src.dtype
    )  # Assuming float64 based on typical use in related functions

    # Constants
    KNOTS_TO_MPS = 0.514444
    MPS_TO_KNOTS = 1.0 / KNOTS_TO_MPS
    DEFAULT_CRUISE_TAS_KTS = 450.0  # Fallback if cannot derive from climb_performance

    # Args mapping based on backward logic:
    # coords_src -> p_s (point for which state is being calculated)
    # alts_t -> alt_t_actual (known altitude at p_t)
    # eta_t -> eta_t_actual (known ETA at p_t, also proxy for wind lookup time at p_s)
    # phase_t -> phase_t_actual (known phase at p_t)
    # coords_tgt -> p_t (known target point)
    # descent_performance -> descent_profile_raw (the raw table to be processed)

    p_s_coords = coords_src # to be calculated
    alt_t_actual = alts_t
    eta_t_actual = eta_t # Also used as eta_s_for_wind_approx
    phase_t_actual = phase_t
    p_t_coords = coords_tgt # known, given
    descent_profile_raw = descent_performance # Interpret as descent profile

    # Initialize output tensors (state at p_s)
    num_segments = p_s_coords.shape[0]
    alt_s_out = torch.zeros(num_segments, device=device, dtype=dtype)
    eta_s_out = torch.zeros(num_segments, device=device, dtype=dtype)
    phase_s_out = torch.full((num_segments,), -1, device=device, dtype=torch.long)

    # --- Derive cruise TAS from descent_profile ---
    # Assumes descent_profile[0] and [1] can define cruise speed before descent starts.
    # E.g. first segment of descent profile is at cruise altitude/speed.
    if len(descent_profile_raw) >= 2:
        # (alt, time_from_ToD, dist_wf_from_ToD)
        # Using the first two points of the descent profile to infer TAS.
        # This assumes these points represent level flight at cruise before descent initiation.
        perf_alt0, perf_time0_from_tod, perf_dist0_wf_from_tod = descent_profile_raw[0]
        perf_alt1, perf_time1_from_tod, perf_dist1_wf_from_tod = descent_profile_raw[1]

        delta_dist_wf_cruise = perf_dist1_wf_from_tod - perf_dist0_wf_from_tod
        delta_time_s_cruise = perf_time1_from_tod - perf_time0_from_tod

        if delta_time_s_cruise > 1e-6: # Avoid division by zero
            tas_cruise_kts = delta_dist_wf_cruise / (delta_time_s_cruise / 3600.0)
        else:
            # If time delta is zero, check if dist delta is also zero (implies first point is ToD)
            # Or if alt0 == alt1, could be level cruise segment.
            # Fallback if ambiguous.
            tas_cruise_kts = DEFAULT_CRUISE_TAS_KTS
    else:
        tas_cruise_kts = DEFAULT_CRUISE_TAS_KTS


    # --- 1. Cruise Phase Handling (Backward) ---
    # If phase_t_actual is CRUISE, then phase_s_out is CRUISE, alt_s_out is alt_t_actual.
    # Calculate eta_s_out = eta_t_actual - time_taken_for_segment_S_to_T.
    cruise_mask = phase_t_actual == CRUISE
    if cruise_mask.any():
        num_cruise = cruise_mask.sum().item()
        if num_cruise > 0:
            alt_s_out[cruise_mask] = alt_t_actual[cruise_mask]
            phase_s_out[cruise_mask] = CRUISE

            p_s_cruise = p_s_coords[cruise_mask]
            p_t_cruise = p_t_coords[cruise_mask]
            alt_s_cruise_for_wind = alt_s_out[cruise_mask] # Altitude at S is same as T for cruise
            eta_s_approx_for_wind = eta_t_actual[cruise_mask] # Use eta_T as proxy for eta_S for wind

            dist_nm_leg_ST = haversinet(
                p_s_cruise[:, 0], p_s_cruise[:, 1],
                p_t_cruise[:, 0], p_t_cruise[:, 1]
            )

            # Wind along track S -> T, estimated using conditions at S (alt_S, eta_S_approx)
            wind_mps_at_s = get_wind(
                p_s_cruise, p_t_cruise,
                alt_s_cruise_for_wind,
                eta_s_approx_for_wind, # Using eta_t_actual as proxy for eta_s for wind
                wind_model
            )
            wind_kts_at_s = wind_mps_at_s * MPS_TO_KNOTS

            gs_kts_ST = tas_cruise_kts + wind_kts_at_s # Ground speed from S to T

            time_hours_ST = torch.zeros_like(dist_nm_leg_ST)
            valid_gs_mask = gs_kts_ST > 1.0
            time_hours_ST[valid_gs_mask] = dist_nm_leg_ST[valid_gs_mask] / gs_kts_ST[valid_gs_mask]
            time_hours_ST[~valid_gs_mask] = torch.finfo(dtype).max / 3600.0 # Effectively infinite time

            time_secs_ST = time_hours_ST * 3600.0
            eta_s_out[cruise_mask] = eta_t_actual[cruise_mask] - time_secs_ST

    # --- Non-Cruise Phase Handling (e.g., DESCENT from T back to S) ---
    # phase_t_actual == DESCENT (or CLIMB if profile is misused for climb with backward logic)
    # For robust handling, assume any non-cruise phase uses the descent_profile logic backward.
    non_cruise_mask = (phase_t_actual == DESCENT) | (phase_t_actual == CLIMB) # CLIMB here implies backward from a "climb state at T"
    # More accurate: non_cruise_mask = phase_t_actual != CRUISE, if profile is always descent.
    # For now, let's assume phase_t_actual tells us if T is in a profiled segment.

    if non_cruise_mask.any():
        num_nc = non_cruise_mask.sum().item()
        if num_nc > 0:
            p_s_nc = p_s_coords[non_cruise_mask]
            alt_t_nc = alt_t_actual[non_cruise_mask]
            eta_t_nc = eta_t_actual[non_cruise_mask] # Actual ETA at T
            # eta_s_for_wind_nc is eta_t_nc, used as proxy for eta at S for wind lookup
            phase_t_nc = phase_t_actual[non_cruise_mask]
            p_t_nc = p_t_coords[non_cruise_mask]

            if not descent_profile_raw:
                alt_s_out[non_cruise_mask] = alt_t_nc # Fallback: no change
                eta_s_out[non_cruise_mask] = eta_t_nc
                phase_s_out[non_cruise_mask] = phase_t_nc
                raise ValueError("Descent performance data is not specified for non-cruise backward calculation.")

            # Unpack descent_profile: (alt_ft, time_sec_from_ToD, dist_nm_wf_from_ToD)
            # P_alt decreases, P_time_from_ToD increases, P_dist_wf_from_ToD increases.
            prof_alts_list, prof_times_s_list, prof_dist_wf_nm_list = zip(*descent_profile_raw)
            
            P_alt_prof = torch.tensor(prof_alts_list, dtype=dtype, device=device)         # [ProfPoints]
            P_time_prof = torch.tensor(prof_times_s_list, dtype=dtype, device=device)     # [ProfPoints]
            P_dist_wf_prof = torch.tensor(prof_dist_wf_nm_list, dtype=dtype, device=device) # [ProfPoints]

            if len(prof_alts_list) < 2:
                alt_s_out[non_cruise_mask] = alt_t_nc
                eta_s_out[non_cruise_mask] = eta_t_nc
                phase_s_out[non_cruise_mask] = phase_t_nc
                # Consider logging an error, for now, just pass through state
                # This was handled as a raise in fwd, let's be consistent if critical
                raise ValueError("Descent profile must have at least two points.")

            # Leg distance S -> T
            dist_leg_ST_nm_nc = haversinet(
                p_s_nc[:, 0], p_s_nc[:, 1],
                p_t_nc[:, 0], p_t_nc[:, 1]
            )

            # Wind at S for S->T leg: Use alt_t_nc as proxy for alt_s_nc for wind calc.
            # This is an approximation as alt_s_nc is what we are solving for.
            wind_mps_at_s_nc = get_wind(
                p_s_nc, p_t_nc,
                alt_t_nc, # Using Target's altitude as proxy for Source's altitude for wind
                eta_t_nc, # Using Target's ETA as proxy for Source's ETA for wind
                wind_model
            )
            wind_kts_at_s_nc = wind_mps_at_s_nc * MPS_TO_KNOTS # Shape [num_nc]

            # Create effective ground distance profile from ToD, considering wind at S
            # P_ground_dist_from_ToD = P_dist_wf + W_s * (P_time_from_ToD / 3600.0)
            # P_ground_dist_from_ToD should be [num_nc, ProfPoints]
            P_ground_dist_prof_nc = P_dist_wf_prof.unsqueeze(0) + \
                                    wind_kts_at_s_nc.unsqueeze(1) * (P_time_prof.unsqueeze(0) / 3600.0)


            # Interpolate to find current state at T within the descent profile
            # We need to interpolate alt_t_nc against P_alt_prof to find time_T_profile and dist_wf_T_profile.
            # numpy.interp requires x-coordinates (P_alt_prof) to be sorted.
            # P_alt_prof is typically sorted descending for a descent profile.
            # We'll flip for interp if needed, or use searchsorted carefully.
            
            # For simplicity with batched_interp1d_torch, P_alt_prof needs to be sorted ascending for x_known.
            # Let's sort the profile by altitude (ascending) for interpolation.
            # This assumes P_alt_prof might not be sorted or is descending.
            sorted_indices = torch.argsort(P_alt_prof)
            P_alt_prof_sorted = P_alt_prof[sorted_indices]
            P_time_prof_sorted = P_time_prof[sorted_indices]
            P_dist_wf_prof_sorted = P_dist_wf_prof[sorted_indices]

            # Interpolate to find profile time and wind-free distance at alt_t_nc
            # This gives time from ToD (or profile start if sorted alt) and dist_wf from ToD (or profile start)
            time_T_profile_val = batched_interp1d_torch(alt_t_nc, P_alt_prof_sorted.unsqueeze(0).expand(num_nc, -1), P_time_prof_sorted.unsqueeze(0).expand(num_nc, -1), device)
            dist_wf_T_profile_val = batched_interp1d_torch(alt_t_nc, P_alt_prof_sorted.unsqueeze(0).expand(num_nc, -1), P_dist_wf_prof_sorted.unsqueeze(0).expand(num_nc, -1), device)
            
            # Ground distance covered from ToD to reach p_t (T)
            ground_dist_T_from_ToD_val = dist_wf_T_profile_val + wind_kts_at_s_nc * (time_T_profile_val / 3600.0)

            # Target total ground distance from ToD to reach p_s (S)
            # Moving backward from T to S, so subtract leg distance.
            target_ground_dist_S_from_ToD = ground_dist_T_from_ToD_val - dist_leg_ST_nm_nc
            
            # ToD parameters from original (potentially unsorted by alt for interp, but first entry is ToD)
            alt_ToD_prof = P_alt_prof[0] # Highest altitude in profile
            # Ground distance from ToD to ToD itself (should be 0 if P_dist_wf[0] and P_time[0] are 0)
            # Use the first point of the calculated P_ground_dist_prof_nc for consistency
            ground_dist_at_ToD_prof = P_ground_dist_prof_nc[:, 0] # Value for each batch item's wind

            # Interpolate for target altitude (alt_s_out) and profile time (time_S_profile) at S
            # using target_ground_dist_S_from_ToD on the P_ground_dist_prof_nc.
            # P_ground_dist_prof_nc should be sorted for batched_interp1d_torch's x_known.
            # P_dist_wf_prof and P_time_prof increase, so P_ground_dist_prof_nc should be sorted if wind is not excessively negative.
            # Assuming P_ground_dist_prof_nc is sorted as x_known.
            # y_known are P_alt_prof and P_time_prof (original order, co-sorted with P_ground_dist_prof_nc's P_dist_wf_prof part).

            alt_s_interp = batched_interp1d_torch(
                target_ground_dist_S_from_ToD, P_ground_dist_prof_nc, P_alt_prof.unsqueeze(0).expand(num_nc, -1), device
            )
            time_S_profile_interp = batched_interp1d_torch(
                target_ground_dist_S_from_ToD, P_ground_dist_prof_nc, P_time_prof.unsqueeze(0).expand(num_nc, -1), device
            )

            # --- Edge Case: Top of Descent (ToD) ---
            # Mask for segments where S is still descending (after ToD)
            is_S_in_descent_mask = target_ground_dist_S_from_ToD > ground_dist_at_ToD_prof
            
            # Mask for segments where ToD is crossed on leg S->T
            # (S is at or before ToD, T was after ToD)
            is_ToD_crossed_mask = (target_ground_dist_S_from_ToD <= ground_dist_at_ToD_prof) & \
                                  (ground_dist_T_from_ToD_val > ground_dist_at_ToD_prof)

            # Initialize with fallback (e.g. if no mask matches, though one should)
            alt_s_out_nc_final = torch.full_like(alt_t_nc, -1.0) # alt_t_nc.clone() 
            eta_s_out_nc_final = torch.full_like(eta_t_nc, -1.0) # eta_t_nc.clone()
            phase_s_out_nc_final = torch.full_like(phase_t_nc, -1) # phase_t_nc.clone()


            # Case 1: S is still in descent (after ToD)
            if is_S_in_descent_mask.any():
                mask = is_S_in_descent_mask
                alt_s_out_nc_final[mask] = alt_s_interp[mask]
                phase_s_out_nc_final[mask] = DESCENT # Or phase_t_nc[mask] if it could be CLIMB backward
                
                time_taken_ST_descent = time_T_profile_val[mask] - time_S_profile_interp[mask]
                # Ensure time_taken is non-negative; if S is "before" T in profile time, something is wrong.
                time_taken_ST_descent = torch.clamp(time_taken_ST_descent, min=0)
                eta_s_out_nc_final[mask] = eta_t_nc[mask] - time_taken_ST_descent
            
            # Case 2: ToD is crossed on leg S->T (S is at/before ToD, T is after ToD)
            if is_ToD_crossed_mask.any():
                mask = is_ToD_crossed_mask
                alt_s_out_nc_final[mask] = alt_ToD_prof # S is at ToD altitude (cruise alt)
                phase_s_out_nc_final[mask] = CRUISE

                # Time spent in descent (from ToD to T)
                time_desc_ToD_to_T = time_T_profile_val[mask] - P_time_prof[0] # P_time_prof[0] is time at ToD (0)
                time_desc_ToD_to_T = torch.clamp(time_desc_ToD_to_T, min=0)

                # Ground distance covered in descent (from ToD to T)
                dist_desc_ToD_to_T_ground = ground_dist_T_from_ToD_val[mask] - ground_dist_at_ToD_prof[mask]
                dist_desc_ToD_to_T_ground = torch.clamp(dist_desc_ToD_to_T_ground, min=0)
                
                # Remaining distance for cruise part (S to ToD)
                dist_cruise_S_to_ToD = dist_leg_ST_nm_nc[mask] - dist_desc_ToD_to_T_ground
                dist_cruise_S_to_ToD = torch.clamp(dist_cruise_S_to_ToD, min=0)

                # GS for cruise part (S to ToD) using tas_cruise_kts and wind_kts_at_s_nc
                # (wind_kts_at_s_nc was based on alt_t_nc, an approximation for this cruise part at alt_ToD_prof)
                gs_cruise_S_to_ToD = tas_cruise_kts + wind_kts_at_s_nc[mask]

                time_cruise_S_to_ToD_hours = torch.zeros_like(dist_cruise_S_to_ToD)
                valid_gs_cruise_mask = gs_cruise_S_to_ToD > 1.0
                time_cruise_S_to_ToD_hours[valid_gs_cruise_mask] = \
                    dist_cruise_S_to_ToD[valid_gs_cruise_mask] / gs_cruise_S_to_ToD[valid_gs_cruise_mask]
                time_cruise_S_to_ToD_hours[~valid_gs_cruise_mask] = torch.finfo(dtype).max / 3600.0
                
                time_cruise_S_to_ToD_secs = time_cruise_S_to_ToD_hours * 3600.0
                
                eta_s_out_nc_final[mask] = eta_t_nc[mask] - time_desc_ToD_to_T - time_cruise_S_to_ToD_secs

            # Case 3: S is before ToD, and T was also at/before ToD (i.e. entire leg S-T is cruise)
            # This should have been caught by the main CRUISE block if phase_t_actual was CRUISE.
            # If phase_t_actual was DESCENT but alt_t_actual was above or at ToD_alt in profile,
            # then ground_dist_T_from_ToD_val might be <= ground_dist_at_ToD_prof.
            # These are segments that effectively are fully cruise backward from T.
            already_cruise_mask = ~(is_S_in_descent_mask | is_ToD_crossed_mask)
            if already_cruise_mask.any():
                # Treat as full cruise segment S->T
                mask = already_cruise_mask
                alt_s_out_nc_final[mask] = alt_t_nc[mask] # Maintain altitude from T (cruise)
                phase_s_out_nc_final[mask] = CRUISE

                # Recalculate time for cruise using dist_leg_ST_nm_nc[mask]
                # Wind is wind_kts_at_s_nc[mask]
                gs_kts_ST_case3 = tas_cruise_kts + wind_kts_at_s_nc[mask]
                
                time_hours_ST_case3 = torch.zeros_like(dist_leg_ST_nm_nc[mask])
                valid_gs_mask_c3 = gs_kts_ST_case3 > 1.0
                time_hours_ST_case3[valid_gs_mask_c3] = dist_leg_ST_nm_nc[mask][valid_gs_mask_c3] / gs_kts_ST_case3[valid_gs_mask_c3]
                time_hours_ST_case3[~valid_gs_mask_c3] = torch.finfo(dtype).max / 3600.0
                
                time_secs_ST_case3 = time_hours_ST_case3 * 3600.0
                eta_s_out_nc_final[mask] = eta_t_nc[mask] - time_secs_ST_case3


            # Update main output tensors for non_cruise_mask segments
            alt_s_out[non_cruise_mask] = alt_s_out_nc_final
            eta_s_out[non_cruise_mask] = eta_s_out_nc_final
            phase_s_out[non_cruise_mask] = phase_s_out_nc_final
            
    # The function is defined to return (alt_tgt, eta_tgt, phase_tgt)
    # In our backward context, this means (alt_s_out, eta_s_out, phase_s_out)
    return alt_s_out, eta_s_out, phase_s_out

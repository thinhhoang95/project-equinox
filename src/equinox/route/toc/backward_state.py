import torch
from equinox.wind.wind_model import WindModel
from equinox.helpers.haversine import haversinet
from typing import List, Tuple
import numpy
import math
from datetime import timedelta
from equinox.route.get_wind import get_wind
from equinox.route.batch_interpolator import batched_interp1d_torch, batched_interp1d_torch_anyorder

# Phase identifiers
CLIMB, CRUISE, DESCENT = 0, 1, 2

def get_next_state_bw(
    coords_src: torch.Tensor,
    alts_t: torch.Tensor,
    eta_t: torch.Tensor,
    phase_t: torch.Tensor,
    coords_tgt: torch.Tensor,
    descent_performance: List[Tuple[float, float, float]],
    climb_performance: List[Tuple[float, float, float]],
    takeoff_eta: float,
    origin_elevation_ft: float,
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
        descent_performance (List[Tuple[float, float, float]]): A list defining the aircraft's descent
                                   profile. Each tuple represents a point in the profile with
                                   (altitude in feet, elapsed time from profile start (ToD) in seconds,
                                   wind-free distance covered from profile start (ToD) in nautical miles).
                                   Assumed to be sorted by altitude (descending) and time (increasing).
        climb_performance (List[Tuple[float, float, float]]): A list defining the aircraft's climb
                                   profile. Each tuple represents a point in the profile with
                                   (altitude in feet, elapsed time from takeoff in seconds,
                                   wind-free distance covered from takeoff in nautical miles).
                                   Assumed to be sorted by altitude (ascending) and time (increasing).
        takeoff_eta (float): The Estimated Time of Departure from the origin airport in seconds.
        wind_model (WindModel): An instance of the WindModel to query wind components
                                at specific locations, altitudes, and times.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing three tensors for the
        state at the target point of each segment, all of shape [batch_size]:
            - alt_tgt (torch.Tensor): Target altitude in feet.
            - eta_tgt (torch.Tensor): Target Estimated Time of Arrival in seconds.
            - phase_tgt (torch.Tensor): Target flight phase (CLIMB, CRUISE, or DESCENT).
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

    # --- Process Climb Performance Data ---
    if not climb_performance:
        raise ValueError("Climb performance data must be provided.")
    if len(climb_performance) < 2:
        raise ValueError("Climb performance profile must have at least two points.")

    climb_prof_alts_list, climb_prof_times_s_list, climb_prof_dist_wf_nm_list = zip(*climb_performance)
    C_alt_prof = torch.tensor(climb_prof_alts_list, dtype=dtype, device=device)         # [ClimbProfPoints]
    C_time_prof = torch.tensor(climb_prof_times_s_list, dtype=dtype, device=device)     # [ClimbProfPoints]
    C_dist_wf_prof = torch.tensor(climb_prof_dist_wf_nm_list, dtype=dtype, device=device) # [ClimbProfPoints]

    # Assuming climb_performance is sorted by altitude (ascending)
    # Last point in climb_performance is cruise altitude (Top of Climb - ToC)
    alt_toc_prof = C_alt_prof[-1]
    time_to_toc_prof = C_time_prof[-1] # Time from takeoff to ToC
    dist_wf_to_toc_prof = C_dist_wf_prof[-1] # Wind-free distance from takeoff to ToC
    eta_toc = takeoff_eta + time_to_toc_prof


    # --- Derive cruise TAS from descent_profile (or climb if descent not suitable) ---
    # Assumes descent_profile[0] and [1] can define cruise speed before descent starts.
    # E.g. first segment of descent profile is at cruise altitude/speed.
    # If descent profile is short, try to use climb profile's top for cruise TAS (approx.)
    if len(descent_profile_raw) >= 2:
        perf_alt0, perf_time0_from_tod, perf_dist0_wf_from_tod = descent_profile_raw[0]
        perf_alt1, perf_time1_from_tod, perf_dist1_wf_from_tod = descent_profile_raw[1]

        delta_dist_wf_cruise = perf_dist1_wf_from_tod - perf_dist0_wf_from_tod
        delta_time_s_cruise = perf_time1_from_tod - perf_time0_from_tod

        if delta_time_s_cruise > 1e-6:
            tas_cruise_kts = delta_dist_wf_cruise / (delta_time_s_cruise / 3600.0)
            if tas_cruise_kts <= 0: # TAS must be positive
                tas_cruise_kts = DEFAULT_CRUISE_TAS_KTS
        else:
            tas_cruise_kts = DEFAULT_CRUISE_TAS_KTS
    elif len(climb_performance) >=2 : # Try to infer from climb profile if descent is too short
        # This assumes the last two points of climb profile represent cruise or near-cruise
        # Or, more simply, that the aircraft reaches its cruise speed at ToC.
        # A better approach might be to have an explicit cruise_tas parameter or derive from full climb performance.
        # For now, using a default or expecting it from descent.
        # Let's use the speed between the last two points of the climb profile as an approximation of cruise TAS
        # if it's not level flight, this is not accurate, but better than default in some cases.
        if C_time_prof[-1] - C_time_prof[-2] > 1e-6:
            climb_tas_approx = (C_dist_wf_prof[-1] - C_dist_wf_prof[-2]) / ((C_time_prof[-1] - C_time_prof[-2]) / 3600.0)
            if climb_tas_approx > 0 :
                tas_cruise_kts = climb_tas_approx
            else:
                tas_cruise_kts = DEFAULT_CRUISE_TAS_KTS
        else:
            tas_cruise_kts = DEFAULT_CRUISE_TAS_KTS

    else:
        tas_cruise_kts = DEFAULT_CRUISE_TAS_KTS
    
    # Ensure tas_cruise_kts is reasonable
    if tas_cruise_kts <= 0:
        tas_cruise_kts = DEFAULT_CRUISE_TAS_KTS


    # --- 1. Cruise Phase Handling (Backward) ---
    cruise_mask = phase_t_actual == CRUISE
    if cruise_mask.any():
        num_cruise_initial = cruise_mask.sum().item()
        if num_cruise_initial > 0:
            p_s_cruise_initial = p_s_coords[cruise_mask]
            p_t_cruise_initial = p_t_coords[cruise_mask]
            alt_t_cruise_initial = alt_t_actual[cruise_mask]
            eta_t_cruise_initial = eta_t_actual[cruise_mask]

            # Initially assume S is also in cruise
            alt_s_out[cruise_mask] = alt_t_cruise_initial
            phase_s_out[cruise_mask] = CRUISE

            dist_nm_leg_ST_cruise_initial = haversinet(
                p_s_cruise_initial[:, 0], p_s_cruise_initial[:, 1],
                p_t_cruise_initial[:, 0], p_t_cruise_initial[:, 1]
            )

            wind_mps_at_s_cruise_initial = get_wind(
                p_s_cruise_initial, p_t_cruise_initial,
                alt_s_out[cruise_mask], # Use alt_S (which is alt_T here)
                eta_t_cruise_initial,   # Use eta_T as proxy for eta_S for wind
                wind_model
            )
            wind_kts_at_s_cruise_initial = wind_mps_at_s_cruise_initial * MPS_TO_KNOTS
            gs_kts_ST_cruise_initial = tas_cruise_kts + wind_kts_at_s_cruise_initial

            time_hours_ST_cruise_initial = torch.zeros_like(dist_nm_leg_ST_cruise_initial)
            valid_gs_mask_cruise_initial = gs_kts_ST_cruise_initial > 1.0
            time_hours_ST_cruise_initial[valid_gs_mask_cruise_initial] = dist_nm_leg_ST_cruise_initial[valid_gs_mask_cruise_initial] / gs_kts_ST_cruise_initial[valid_gs_mask_cruise_initial]
            time_hours_ST_cruise_initial[~valid_gs_mask_cruise_initial] = torch.finfo(dtype).max / 3600.0
            
            time_secs_ST_cruise_initial = time_hours_ST_cruise_initial * 3600.0
            eta_s_out_cruise_tentative = eta_t_cruise_initial - time_secs_ST_cruise_initial
            eta_s_out[cruise_mask] = eta_s_out_cruise_tentative

            # --- Check for transition from Cruise at T to Climb at S ---
            # This occurs if the tentatively calculated eta_s (assuming cruise) is before eta_toc
            # and alt_t_actual for these segments is at or very near cruise altitude (alt_toc_prof)
            # We also need to ensure T itself was "after" or "at" ToC (eta_t >= eta_toc)
            # For simplicity, we assume if phase_t is CRUISE, it's at cruise_alt.
            
            # Identify segments that were initially CRUISE but S might be in CLIMB
            # Condition: eta_s_out[cruise_mask] < eta_toc AND eta_t_actual[cruise_mask] >= eta_toc (approx, phase_t is CRUISE)
            # And alt_t_actual should be at cruise altitude (alt_toc_prof)
            # alt_t_cruise_initial should be alt_toc_prof for these cases.
            
            # Mask for cruise segments where the calculated source ETA is before ToC ETA
            # Ensure both tensors are of the same dtype (Float) for torch.isclose
            alt_t_cruise_initial_float = alt_t_cruise_initial.to(alt_toc_prof.dtype)
            alt_toc_prof_expanded = alt_toc_prof.expand_as(alt_t_cruise_initial_float)
            s_is_climb_candidate_mask = (eta_s_out_cruise_tentative < eta_toc) & (torch.isclose(alt_t_cruise_initial_float, alt_toc_prof_expanded))

            if s_is_climb_candidate_mask.any():
                num_climb_trans = s_is_climb_candidate_mask.sum().item()
                
                # S (source, e.g., S2 in sanity check) is in climb.
                # T (target, e.g., S1 in sanity check) is in cruise at alt_toc_prof.
                # Condition was: eta_S_tentative < eta_toc <= eta_T_actual
                # Flight path for this segment S-T is: S (climb) -> ToC -> T (cruise)
                
                p_s_trans = p_s_cruise_initial[s_is_climb_candidate_mask] # Coords of S 
                p_t_trans = p_t_cruise_initial[s_is_climb_candidate_mask] # Coords of T
                eta_T_actual_trans = eta_t_cruise_initial[s_is_climb_candidate_mask] # ETA at T

                # Wind for the overall S-T leg (from initial cruise calc), used for cruise part ToC->T
                wind_kts_ST_leg_trans = wind_kts_at_s_cruise_initial[s_is_climb_candidate_mask]
                gs_cruise_ToC_T_part = tas_cruise_kts + wind_kts_ST_leg_trans # GS for cruise part ToC -> T

                # Time and distance for the cruise part of the leg (ToC -> T)
                time_cruise_ToC_to_T = eta_T_actual_trans - eta_toc 
                time_cruise_ToC_to_T = torch.clamp(time_cruise_ToC_to_T, min=0)

                dist_cruise_ToC_to_T_ground = (time_cruise_ToC_to_T / 3600.0) * gs_cruise_ToC_T_part
                dist_cruise_ToC_to_T_ground = torch.clamp(dist_cruise_ToC_to_T_ground, min=0)

                # Total ground distance of the S-T leg (from initial cruise calculation)
                total_dist_S_T_ground = dist_nm_leg_ST_cruise_initial[s_is_climb_candidate_mask]

                # Distance for the climb part of the leg (S -> ToC)
                dist_climb_S_to_ToC_ground = total_dist_S_T_ground - dist_cruise_ToC_to_T_ground
                dist_climb_S_to_ToC_ground = torch.clamp(dist_climb_S_to_ToC_ground, min=0)
                
                # Wind for climb segment S->ToC. 
                # Using conditions at ToC (alt_toc_prof, eta_toc).
                # Direction from p_s_trans (S) to p_t_trans (T), as ToC lies on this path.
                wind_mps_climb_S_ToC = get_wind(
                    p_s_trans, 
                    p_t_trans, 
                    alt_toc_prof.expand(num_climb_trans), 
                    eta_toc.expand(num_climb_trans),    
                    wind_model
                )
                wind_kts_climb_S_ToC = wind_mps_climb_S_ToC * MPS_TO_KNOTS

                # Climb profile adjusted for wind during S->ToC
                C_ground_dist_prof_climb_seg = C_dist_wf_prof.unsqueeze(0) + \
                                           wind_kts_climb_S_ToC.unsqueeze(1) * (C_time_prof.unsqueeze(0) / 3600.0)
                # Ground distance from takeoff to ToC based on this wind-adjusted profile
                ground_dist_takeoff_to_ToC_val = C_ground_dist_prof_climb_seg[:, -1]

                # Target ground distance of S from takeoff:
                # (Dist from Takeoff to ToC) - (Dist from S to ToC)
                target_ground_dist_S_from_takeoff = ground_dist_takeoff_to_ToC_val - dist_climb_S_to_ToC_ground
                target_ground_dist_S_from_takeoff = torch.clamp(target_ground_dist_S_from_takeoff, min=0)
                
                # Make C_alt_prof and C_time_prof broadcastable for batched_interp
                C_alt_prof_exp = C_alt_prof.unsqueeze(0).expand(num_climb_trans, -1)
                C_time_prof_exp = C_time_prof.unsqueeze(0).expand(num_climb_trans, -1)

                # Interpolate on the climb ground distance profile to find alt_s and time_s_from_takeoff at S
                alt_s_val = batched_interp1d_torch_anyorder(
                    target_ground_dist_S_from_takeoff, C_ground_dist_prof_climb_seg, C_alt_prof_exp, device
                )
                time_s_from_takeoff_val = batched_interp1d_torch_anyorder(
                    target_ground_dist_S_from_takeoff, C_ground_dist_prof_climb_seg, C_time_prof_exp, device
                )
                
                # Update the main output tensors for these transition segments
                original_indices_cruise = cruise_mask.nonzero(as_tuple=True)[0]
                transition_indices_in_original = original_indices_cruise[s_is_climb_candidate_mask]

                alt_s_out[transition_indices_in_original] = alt_s_val
                eta_s_out[transition_indices_in_original] = takeoff_eta + time_s_from_takeoff_val
                phase_s_out[transition_indices_in_original] = CLIMB
    
    # --- 2. Climb Phase Handling (Backward) ---
    # This handles cases where phase_t_actual is CLIMB.
    # S is before T, both are in climb phase.
    climb_phase_mask = phase_t_actual == CLIMB
    if climb_phase_mask.any():
        num_climb = climb_phase_mask.sum().item()
        if num_climb > 0:
            p_s_climb = p_s_coords[climb_phase_mask]
            p_t_climb = p_t_coords[climb_phase_mask]
            alt_t_climb = alt_t_actual[climb_phase_mask] # Known altitude at T (in climb)
            eta_t_climb = eta_t_actual[climb_phase_mask] # Known ETA at T

            # Ensure T is not above ToC altitude if in CLIMB phase (data consistency)
            # alt_t_climb = torch.min(alt_t_climb, alt_toc_prof.expand_as(alt_t_climb)) # Not strictly necessary if input is good

            dist_leg_ST_nm_climb = haversinet(
                p_s_climb[:, 0], p_s_climb[:, 1],
                p_t_climb[:, 0], p_t_climb[:, 1]
            )

            # Wind for S->T leg (both in climb).
            # Use average altitude of S and T for wind. S alt is unknown.
            # Approximation: use alt_t_climb for wind lookup, and eta_t_climb.
            # This is wind at T, applied for S->T segment.
            wind_mps_at_t_climb = get_wind(
                p_s_climb, p_t_climb, # For direction
                alt_t_climb,          # Alt at T
                eta_t_climb,          # ETA at T
                wind_model
            )
            wind_kts_at_t_climb = wind_mps_at_t_climb * MPS_TO_KNOTS # Shape [num_climb]

            # Create batched ground distance climb profile: C_dist_wf_prof + wind * (C_time_prof / 3600)
            # Wind is specific to each segment in the batch.
            C_ground_dist_prof_climb = C_dist_wf_prof.unsqueeze(0) + \
                                       wind_kts_at_t_climb.unsqueeze(1) * (C_time_prof.unsqueeze(0) / 3600.0)
            # C_ground_dist_prof_climb has shape [num_climb, ClimbProfPoints]

            # Interpolate to find T's state (time_from_takeoff, ground_dist_from_takeoff) in the climb profile.
            # x_known is C_alt_prof (sorted ascending).
            # Need to expand C_alt_prof, C_time_prof, C_ground_dist_prof_climb for batched interp if not already.
            C_alt_prof_exp_climb = C_alt_prof.unsqueeze(0).expand(num_climb, -1)
            C_time_prof_exp_climb = C_time_prof.unsqueeze(0).expand(num_climb, -1)
            # Note: C_ground_dist_prof_climb is already [num_climb, ProfPoints]

            # Find ground distance from takeoff to T (alt_t_climb)
            # Interpolate alt_t_climb on C_alt_prof_exp_climb (x) to get values from C_ground_dist_prof_climb (y)
            ground_dist_T_from_takeoff = batched_interp1d_torch(
                alt_t_climb, C_alt_prof_exp_climb, C_ground_dist_prof_climb, device
            )
            # batched_interp1d_torch expects x_known to be sorted, C_alt_prof is.

            # Target ground distance for S from Takeoff:
            # Moving backward from T to S, so subtract leg distance.
            target_ground_dist_S_from_takeoff_climb = ground_dist_T_from_takeoff - dist_leg_ST_nm_climb
            target_ground_dist_S_from_takeoff_climb = torch.clamp(target_ground_dist_S_from_takeoff_climb, min=0) # Cannot be before takeoff

            # Interpolate on the climb ground distance profile (C_ground_dist_prof_climb as x)
            # to find alt_s (y1=C_alt_prof_exp_climb) and time_s_from_takeoff (y2=C_time_prof_exp_climb)
            alt_s_climb = batched_interp1d_torch_anyorder(
                target_ground_dist_S_from_takeoff_climb, C_ground_dist_prof_climb, C_alt_prof_exp_climb, device
            )
            time_s_from_takeoff_climb = batched_interp1d_torch_anyorder(
                target_ground_dist_S_from_takeoff_climb, C_ground_dist_prof_climb, C_time_prof_exp_climb, device
            )

            # Update main output tensors
            alt_s_out[climb_phase_mask] = alt_s_climb
            eta_s_out[climb_phase_mask] = takeoff_eta + time_s_from_takeoff_climb
            phase_s_out[climb_phase_mask] = CLIMB

            # Handle cases where S is effectively at takeoff (e.g. target_ground_dist_S_from_takeoff_climb is 0)
            # target_ground_dist_S_from_takeoff_climb is the ground distance from takeoff to S
            # it could be clamped to 0 if calculation reveals that it is negative.

            at_takeoff_mask = target_ground_dist_S_from_takeoff_climb <= 1e-3 # Small epsilon
            if at_takeoff_mask.any():
                # For these, alt_s should be C_alt_prof[0], eta_s should be takeoff_eta
                # Get original indices for climb_phase_mask
                original_indices_climb = climb_phase_mask.nonzero(as_tuple=True)[0]
                takeoff_indices_in_original = original_indices_climb[at_takeoff_mask]

                alt_s_out[takeoff_indices_in_original] = C_alt_prof[0]
                eta_s_out[takeoff_indices_in_original] = takeoff_eta
                # Phase remains CLIMB (or could be a separate PRE_FLIGHT if needed)


    # --- 3. Descent Phase Handling (Backward) ---
    # (Previously non_cruise_mask, now specifically descent_mask)
    # phase_t_actual == DESCENT
    descent_mask = phase_t_actual == DESCENT
    # The original code had non_cruise_mask = (phase_t_actual == DESCENT) | (phase_t_actual == CLIMB)
    # Since CLIMB is handled above, this section is now only for DESCENT.

    if descent_mask.any():
        num_desc = descent_mask.sum().item()
        if num_desc > 0:
            p_s_desc = p_s_coords[descent_mask]
            alt_t_desc = alt_t_actual[descent_mask]
            eta_t_desc = eta_t_actual[descent_mask]
            phase_t_desc = phase_t_actual[descent_mask] # Should all be DESCENT
            p_t_desc = p_t_coords[descent_mask]

            if not descent_profile_raw:
                # This should ideally not happen if routing logic ensures performance data
                alt_s_out[descent_mask] = alt_t_desc # Fallback
                eta_s_out[descent_mask] = eta_t_desc
                phase_s_out[descent_mask] = phase_t_desc # Keep as DESCENT
                # Consider raising an error or logging a warning
                raise ValueError("Descent performance data is not specified for DESCENT phase backward calculation.")

            # Unpack descent_profile: (alt_ft, time_sec_from_ToD, dist_nm_wf_from_ToD)
            # P_alt decreases, P_time_from_ToD increases, P_dist_wf_from_ToD increases.
            # These are typically "time to landing" and "dist_wf to landing" if profile is structured that way.
            # Let's call them D_ for Descent profile.
            D_prof_alts_list, D_prof_times_s_list, D_prof_dist_wf_nm_list = zip(*descent_profile_raw)
            
            D_alt_prof = torch.tensor(D_prof_alts_list, dtype=dtype, device=device)         # [DescProfPoints]
            D_time_prof = torch.tensor(D_prof_times_s_list, dtype=dtype, device=device)     # [DescProfPoints]
            D_dist_wf_prof = torch.tensor(D_prof_dist_wf_nm_list, dtype=dtype, device=device) # [DescProfPoints]

            if len(D_prof_alts_list) < 2:
                alt_s_out[descent_mask] = alt_t_desc
                eta_s_out[descent_mask] = eta_t_desc
                phase_s_out[descent_mask] = phase_t_desc
                raise ValueError("Descent profile must have at least two points for DESCENT phase.")

            dist_leg_ST_nm_desc = haversinet(
                p_s_desc[:, 0], p_s_desc[:, 1],
                p_t_desc[:, 0], p_t_desc[:, 1]
            )

            # Wind at S for S->T leg (during descent). Use alt_t_desc as proxy for alt_s_desc for wind.
            wind_mps_at_s_desc = get_wind(
                p_s_desc, p_t_desc,
                alt_t_desc, 
                eta_t_desc, 
                wind_model
            )
            wind_kts_at_s_desc = wind_mps_at_s_desc * MPS_TO_KNOTS # Shape [num_desc]

            # Effective ground distance profile from ToD (or relative to landing if profile is structured as such)
            # D_ground_dist_prof = D_dist_wf + W_s * (D_time_prof / 3600.0)
            # D_time_prof is time FROM ToD (increases). D_dist_wf_prof is dist FROM ToD (increases).
            # If profile is "to landing", then time and dist are "to go", signs might need adjustment.
            # Assuming profile is (alt, time_from_tod, dist_wf_from_tod)
            # where alt decreases, time_from_tod increases, dist_wf_from_tod increases.
            D_ground_dist_prof_desc = D_dist_wf_prof.unsqueeze(0) + \
                                      wind_kts_at_s_desc.unsqueeze(1) * (D_time_prof.unsqueeze(0) / 3600.0)
            # D_ground_dist_prof_desc shape: [num_desc, DescProfPoints]

            # Interpolate to find current state at T (alt_t_desc) within the descent profile.
            # D_alt_prof is typically sorted descending. batched_interp1d_torch_anyorder handles this.
            # We need (time_at_T_from_ToD, ground_dist_at_T_from_ToD)
            
            # Expand D_alt_prof, D_time_prof, D_dist_wf_prof for batched interpolation
            D_alt_prof_exp = D_alt_prof.unsqueeze(0).expand(num_desc, -1)
            D_time_prof_exp = D_time_prof.unsqueeze(0).expand(num_desc, -1)
            # D_ground_dist_prof_desc is already [num_desc, DescProfPoints]
            
            # Interpolate alt_t_desc on D_alt_prof_exp (x) to get values from D_time_prof_exp (y1) and D_ground_dist_prof_desc (y2)
            # Note: The original code sorted D_alt_prof to ascending for batched_interp1d_torch.
            # With _anyorder, this is not strictly needed, but the profile itself should be monotonic.
            # Let's assume D_alt_prof, D_time_prof, D_dist_wf_prof correspond.
            
            time_T_from_ToD_val = batched_interp1d_torch_anyorder(alt_t_desc, D_alt_prof_exp, D_time_prof_exp, device)
            # ground_dist_T_from_ToD_val = batched_interp1d_torch_anyorder(alt_t_desc, D_alt_prof_exp, D_ground_dist_prof_desc, device)
            # Correction: We need to interpolate for ground_dist_T_from_ToD using the D_alt_prof vs D_ground_dist_prof_desc relationship.
            # The D_ground_dist_prof_desc is already wind-adjusted.
            # We need ground distance at T using alt_T on the wind-adjusted ground distance profile points that correspond to D_alt_prof
            # This needs careful handling if D_alt_prof is x-axis for D_ground_dist_prof_desc's y-values.
            # The current D_ground_dist_prof_desc is [batch, points]. It's the Y values. X values are D_alt_prof.
            
            # Simpler: interpolate alt_t_desc on D_alt_prof to get D_dist_wf_prof at T and D_time_prof at T
            dist_wf_T_from_ToD_val = batched_interp1d_torch_anyorder(alt_t_desc, D_alt_prof_exp, D_dist_wf_prof.unsqueeze(0).expand(num_desc, -1), device)
            # Then calculate ground_dist_T_from_ToD_val
            ground_dist_T_from_ToD_val = dist_wf_T_from_ToD_val + wind_kts_at_s_desc * (time_T_from_ToD_val / 3600.0)


            # Target total ground distance for S from ToD.
            # Moving backward from T to S, so subtract leg distance from T's distance from ToD.
            # This means S is "earlier" in the descent profile (closer to ToD).
            target_ground_dist_S_from_ToD = ground_dist_T_from_ToD_val - dist_leg_ST_nm_desc
            
            # Parameters from ToD point in profile (first point, highest altitude)
            alt_ToD_prof = D_alt_prof[0]
            time_ToD_prof = D_time_prof[0] # Should be 0 if profile starts at ToD
            # Ground distance at ToD itself (value for each batch item based on its wind)
            # ground_dist_at_ToD_prof_val = D_dist_wf_prof[0] + wind_kts_at_s_desc * (D_time_prof[0] / 3600.0)
            # This is effectively D_ground_dist_prof_desc[:, 0]
            ground_dist_at_ToD_prof_val = D_ground_dist_prof_desc[:, 0]


            # Interpolate for alt_s_out and time_S_from_ToD using target_ground_dist_S_from_ToD
            # on the D_ground_dist_prof_desc (x-values) vs D_alt_prof (y1) and D_time_prof (y2).
            alt_s_interp_desc = batched_interp1d_torch_anyorder(
                target_ground_dist_S_from_ToD, D_ground_dist_prof_desc, D_alt_prof_exp, device
            )
            time_S_from_ToD_interp = batched_interp1d_torch_anyorder(
                target_ground_dist_S_from_ToD, D_ground_dist_prof_desc, D_time_prof_exp, device
            )

            # --- Edge Cases for Descent ---
            # Mask for segments where S is still in descent (i.e., target_ground_dist_S_from_ToD >= ground_dist_at_ToD_prof_val)
            # (S is after or at ToD)
            is_S_in_descent_mask = target_ground_dist_S_from_ToD >= ground_dist_at_ToD_prof_val
            
            # Mask for segments where S is before ToD (i.e. S-ToD part is cruise)
            # (target_ground_dist_S_from_ToD < ground_dist_at_ToD_prof_val)
            # This implies T was in descent, but S is in cruise.
            is_S_cruise_before_ToD_mask = target_ground_dist_S_from_ToD < ground_dist_at_ToD_prof_val
            
            # Initialize with fallback (should not be needed if masks cover all)
            alt_s_out_desc_final = torch.full_like(alt_t_desc, -1.0)
            eta_s_out_desc_final = torch.full_like(eta_t_desc, -1.0)
            phase_s_out_desc_final = torch.full_like(phase_t_desc, -1, dtype=torch.long)

            # Case 1: S is still in descent (S is between ToD and T)
            if is_S_in_descent_mask.any():
                mask = is_S_in_descent_mask
                alt_s_out_desc_final[mask] = alt_s_interp_desc[mask]
                phase_s_out_desc_final[mask] = DESCENT
                
                # Time taken for S->T segment = time_T_from_ToD_val - time_S_from_ToD_interp
                time_taken_ST_desc = time_T_from_ToD_val[mask] - time_S_from_ToD_interp[mask]
                time_taken_ST_desc = torch.clamp(time_taken_ST_desc, min=0) # Should be positive
                eta_s_out_desc_final[mask] = eta_t_desc[mask] - time_taken_ST_desc
            
            # Case 2: S is in cruise, T is in descent (ToD is crossed on leg S->T)
            if is_S_cruise_before_ToD_mask.any():
                mask = is_S_cruise_before_ToD_mask
                alt_s_out_desc_final[mask] = alt_ToD_prof # S is at ToD altitude (cruise alt)
                phase_s_out_desc_final[mask] = CRUISE

                # Time spent in descent part (from ToD to T)
                # time_T_from_ToD_val is already calculated for these segments.
                # time_ToD_prof is D_time_prof[0] (usually 0).
                time_desc_ToD_to_T = time_T_from_ToD_val[mask] - time_ToD_prof # time_ToD_prof is scalar
                time_desc_ToD_to_T = torch.clamp(time_desc_ToD_to_T, min=0)

                # Ground distance covered in descent part (from ToD to T)
                # ground_dist_T_from_ToD_val is calculated. ground_dist_at_ToD_prof_val is calculated.
                dist_desc_ToD_to_T_ground = ground_dist_T_from_ToD_val[mask] - ground_dist_at_ToD_prof_val[mask]
                dist_desc_ToD_to_T_ground = torch.clamp(dist_desc_ToD_to_T_ground, min=0)
                
                # Remaining distance for cruise part (S to ToD)
                dist_cruise_S_to_ToD = dist_leg_ST_nm_desc[mask] - dist_desc_ToD_to_T_ground
                dist_cruise_S_to_ToD = torch.clamp(dist_cruise_S_to_ToD, min=0)

                # GS for cruise part (S to ToD) using tas_cruise_kts and wind_kts_at_s_desc[mask]
                # Wind approx: wind_kts_at_s_desc was based on alt_t_desc (in descent).
                # For S->ToD (cruise part), wind should ideally be at alt_ToD_prof.
                # Re-calculate wind for this cruise segment at alt_ToD_prof.
                # p_s for this part is p_s_desc[mask], p_t is ToD (coords not directly known, use p_t_desc[mask] as proxy for direction)
                # For simplicity, we can use wind_kts_at_s_desc[mask] as an approximation, or re-fetch wind at ToD alt.
                # Let's use existing wind_kts_at_s_desc[mask] for now, acknowledging it's an approx.
                gs_cruise_S_to_ToD = tas_cruise_kts + wind_kts_at_s_desc[mask]

                time_cruise_S_to_ToD_hours = torch.zeros_like(dist_cruise_S_to_ToD)
                valid_gs_cruise_mask_desc = gs_cruise_S_to_ToD > 1.0
                time_cruise_S_to_ToD_hours[valid_gs_cruise_mask_desc] = \
                    dist_cruise_S_to_ToD[valid_gs_cruise_mask_desc] / gs_cruise_S_to_ToD[valid_gs_cruise_mask_desc]
                time_cruise_S_to_ToD_hours[~valid_gs_cruise_mask_desc] = torch.finfo(dtype).max / 3600.0
                
                time_cruise_S_to_ToD_secs = time_cruise_S_to_ToD_hours * 3600.0
                
                # ETA at S = ETA at T - time_desc_ToD_to_T - time_cruise_S_to_ToD_secs
                eta_s_out_desc_final[mask] = eta_t_desc[mask] - time_desc_ToD_to_T - time_cruise_S_to_ToD_secs

            # Update main output tensors for descent_mask segments
            alt_s_out[descent_mask] = alt_s_out_desc_final
            eta_s_out[descent_mask] = eta_s_out_desc_final
            phase_s_out[descent_mask] = phase_s_out_desc_final
            
    # Final check for any segments not processed (should not happen if logic is complete)
    unprocessed_mask = phase_s_out == -1
    if unprocessed_mask.any():
        # Fallback or error for any segments that didn't get handled by cruise, climb, or descent logic
        # This might indicate an unknown phase_t_actual or a gap in logic.
        # For now, pass through T's state to S for these.
        alt_s_out[unprocessed_mask] = alt_t_actual[unprocessed_mask]
        eta_s_out[unprocessed_mask] = eta_t_actual[unprocessed_mask]
        phase_s_out[unprocessed_mask] = phase_t_actual[unprocessed_mask] # Or set to an ERROR/UNKNOWN phase
        # Consider logging a warning here.
        # print(f"Warning: {unprocessed_mask.sum().item()} segments were not processed and fell through.")

    return alt_s_out, eta_s_out, phase_s_out
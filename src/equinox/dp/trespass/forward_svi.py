# *****************************************************************************
# ATTENTION: This implementation does not use the graph topological sort.
# So it is INCORRECT. Consult `forward_svi_log.py` for the correct implementation
# (and improved performance).
# *****************************************************************************

import torch
import math

from equinox.route.get_wind import get_wind
# For type hinting, actual instances are passed as arguments
from equinox.cost.cost_rev1 import CostRev1 
from equinox.wind.wind_model import WindModel
from equinox.dp.trespass.transition_utils import get_base_transition

# Conversion factor from meters per second to knots
MPS_TO_KNOTS = 1.94384

def forward_soft_value_iteration(
    state_transitions: list[tuple],
    # The tuple contains:
    # u_idx: index of the source waypoint
    # k_u_idx: wall-clock time bin index at u_idx
    # rho_u_idx: remaining climb time bin index at u_idx
    # phase_u: flight phase at u_idx (0:CLIMB, 1:CRUISE, 2:DESCENT)
    # u_alt_ft: altitude in feet at u_idx
    # v_idx: index of the target waypoint
    # k_v_idx: wall-clock time bin index at v_idx
    # rho_v_idx: remaining climb time bin index at v_idx
    # phase_v: flight phase at v_idx
    # v_alt_ft: altitude in feet at v_idx (defines state s_v, not directly used for cost(u,v))
    origin_node_idx: int,
    cost_model: CostRev1,
    num_nodes: int,
    num_time_bins_wall_clock: int,
    num_rho_bins: int,
    num_phases: int,
    distance_matrix_d: torch.Tensor,
    airspace_charge_matrix_ac: torch.Tensor,
    node_coords_deg: torch.Tensor, # Shape [num_nodes, 2] (latitude, longitude) in degrees
    wind_model: WindModel,
    min_wall_clock_time_sec: float, # Absolute start time for k_idx=0 in seconds
    delta_t_wall_clock_sec: float, # Duration of each wall-clock time bin in seconds
    device: torch.device,
    verbose: bool = False
) -> torch.Tensor:
    """
    Computes the soft forward value function Z(s) and V(s) using dynamic programming.
    Z(s) = sum_{xi_source->s} exp(-c(xi_source->s))
    V(s) = -log Z(s)
    """

    # Initialize Z_val tensor with zeros.
    # Z_val stores the sum of probabilities (or unnormalized mass) reaching each state.
    Z_val = torch.zeros(
        (num_nodes, num_time_bins_wall_clock, num_rho_bins, num_phases),
        dtype=torch.float64,  # Use float64 for precision
        device=device
    )

    # Identify unique origin states from the transitions list and initialize them.
    # An origin state is (origin_node_idx, k_u_idx, rho_u_idx, phase_u).
    # We assume a uniform distribution over these valid starting states.
    actual_origin_states = set()
    for st_tuple in state_transitions:
        u_idx, k_u, rho_u, _, phase_u, _, _, _, _, _ = get_base_transition(st_tuple)
        if u_idx == origin_node_idx:
            actual_origin_states.add((k_u, rho_u, phase_u))

    if not actual_origin_states:
        if verbose:
            print(f"Warning: No transitions found starting from origin_node_idx {origin_node_idx}. "
                  f"All Z values will remain 0, and V values will be inf.")
    else:
        num_actual_origin_states = len(actual_origin_states)
        initial_Z_at_origin = 1.0 / num_actual_origin_states
        for k_u_orig, rho_u_orig, phase_u_orig in actual_origin_states:
            Z_val[origin_node_idx, k_u_orig, rho_u_orig, phase_u_orig] = initial_Z_at_origin
            if verbose:
                print(f"Initialized origin state: Z[{origin_node_idx}, {k_u_orig}, {rho_u_orig}, {phase_u_orig}] = {initial_Z_at_origin:.4e}")
    
    # Sort transitions to process states in a topological order.
    # Primary sort key is the time index of the source state (k_u_idx).
    # This ensures that when processing a transition u->v, Z(u) has been finalized.
    sorted_transitions = sorted(state_transitions, key=lambda x: (x[1], x[0], x[2], x[3]))

    if verbose:
        print(f"Processing {len(sorted_transitions)} state transitions...")

    for i, transition_elements in enumerate(sorted_transitions):
        u_idx, k_u, rho_u, u_alt_ft, phase_u, \
        v_idx, k_v, rho_v, _, phase_v = get_base_transition(transition_elements)
        # v_alt_ft is part of the state s_v definition but not directly used for cost(u,v) calculation here.

        Z_s_u = Z_val[u_idx, k_u, rho_u, phase_u].item() # .item() to get scalar

        # If the source state s_u has zero probability mass, it cannot contribute to s_v
        if Z_s_u == 0.0:
            continue

        # --- Calculate edge cost c(s_u, s_v) ---
        # 1. Prepare inputs for cost_model
        edge_u_indices = torch.tensor([u_idx], device=device, dtype=torch.long)
        edge_v_indices = torch.tensor([v_idx], device=device, dtype=torch.long)
        
        # 2. Calculate tailwind for the edge (u,v)
        # Wind is calculated at the source waypoint u_idx, source altitude u_alt_ft, and source time k_u.
        coords_src_edge = node_coords_deg[u_idx].unsqueeze(0).to(device=device, dtype=torch.float32)
        coords_tgt_edge = node_coords_deg[v_idx].unsqueeze(0).to(device=device, dtype=torch.float32)
        altitude_for_wind_ft = torch.tensor([u_alt_ft], device=device, dtype=torch.float32)
        
        eta_src_sec_edge = min_wall_clock_time_sec + k_u * delta_t_wall_clock_sec
        eta_src_tensor_edge = torch.tensor([eta_src_sec_edge], device=device, dtype=torch.float32)
        
        tailwind_mps = get_wind(
            coords_src_edge, 
            coords_tgt_edge, 
            altitude_for_wind_ft, 
            eta_src_tensor_edge, 
            wind_model
        )
        tailwind_knots = tailwind_mps * MPS_TO_KNOTS
        
        # 3. Call cost_model to get the cost of the edge
        cost_uv_tensor = cost_model(
            (edge_u_indices, edge_v_indices),
            distance_matrix_d,
            airspace_charge_matrix_ac,
            tailwind_knots.to(device=device) # Ensure tailwind is on the same device as model
        )
        cost_uv = cost_uv_tensor.item() # Get scalar cost

        # --- Update Z_val for the target state s_v ---
        # Z(s_v) = sum_{s_u -> s_v} Z(s_u) * exp(-cost_uv)
        # This is an accumulation over all incoming edges to s_v.
        # Note: cost_uv should be float64 for precision with Z_s_u (float64)
        exp_term = torch.exp(torch.tensor(-cost_uv, dtype=torch.float64, device=device))
        Z_val[v_idx, k_v, rho_v, phase_v] += Z_s_u * exp_term
        
        if verbose and (i % (len(sorted_transitions)//100 + 1) == 0 or i == len(sorted_transitions)-1) : # Log progress
             if Z_s_u * exp_term > 1e-300: # Log only if contribution is not extremely tiny
                print(f"  Transition {i+1}/{len(sorted_transitions)}: "
                      f"Z[{u_idx},{k_u},{rho_u},{phase_u}]={Z_s_u:.3e} -> "
                      f"Z[{v_idx},{k_v},{rho_v},{phase_v}] "
                      f"Cost={cost_uv:.3f}, ExpTerm={exp_term:.3e}, "
                      f"AddMass={(Z_s_u * exp_term):.3e}, NewZ={Z_val[v_idx, k_v, rho_v, phase_v]:.3e}")


    # Calculate V_soft from Z_val: V_soft(s) = -log Z(s)
    # torch.log(0) results in -inf. So -torch.log(0) results in +inf, which is correct for unreachable states.
    # Clamp Z_val to be non-negative just in case of tiny numerical errors leading to negative zero.
    # However, with exp terms, Z_val should remain non-negative.
    V_soft = -torch.log(Z_val) # Z_val is already float64
    
    # Ensure that any state s where Z(s) was 0 (hence V(s) is inf) is represented by float('inf')
    # This is typically handled correctly by -log(0.0) = inf in PyTorch.
    # V_soft[Z_val == 0] = float('inf') # Redundant if using torch.log on non-negative Z_val

    if verbose:
        num_finite_v = torch.isfinite(V_soft).sum().item()
        total_states = V_soft.numel()
        print(f"Forward SVI complete. {num_finite_v}/{total_states} states have finite soft values.")
        # Example of checking a few values if needed for debugging:
        # print(V_soft[origin_node_idx, :, :, :])

    return V_soft

# Example usage structure (for testing, not part of the final library function)
if __name__ == '__main__':
    # This block would require setting up all the mock inputs:
    # cost_model, state_transitions, num_nodes, num_time_bins_wall_clock, etc.
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("forward_svi.py main block reached - for testing, actual usage is by importing the function.")
    
    # --- Mock Inputs Setup (Simplified) ---
    # This is a very basic setup and needs to be expanded significantly for a real test.
    # device = torch.device("cpu")
    # num_nodes_test = 3
    # num_time_bins_test = 5
    # num_rho_bins_test = 2
    # num_phases_test = 3
    # origin_node_test = 0
    
    # # Mock Cost Model (just returns 1.0 for any edge)
    # class MockCostModel(torch.nn.Module):
    #     def forward(self, edge_indices, dist_matrix, ac_matrix, tailwind):
    #         return torch.ones(edge_indices[0].shape[0], device=device, dtype=torch.float32)
    # cost_model_test = MockCostModel()

    # # Mock Wind Model
    # class MockWindModel(WindModel):
    #     def __init__(self): pass
    #     def get_wind_components_batched(self, lats, lons, alts, etas):
    #         return torch.zeros_like(lats), torch.zeros_like(lats) # Zero wind
    # wind_model_test = MockWindModel()
        
    # state_transitions_test = [
    #     # u_idx, k_u, rho_u, ph_u, alt_u,  v_idx, k_v, rho_v, ph_v, alt_v
    #     (0, 0, 1, 0, 1000.0,  1, 1, 1, 0, 1000.0), # Origin (0,0,1,0) -> (1,1,1,0)
    #     (0, 0, 1, 0, 1000.0,  1, 2, 0, 1, 2000.0), # Origin (0,0,1,0) -> (1,2,0,1) (another path to node 1, different state)
    #     (1, 1, 1, 0, 1000.0,  2, 2, 1, 0, 1000.0), # (1,1,1,0) -> Goal (2,2,1,0)
    #     (1, 2, 0, 1, 2000.0,  2, 3, 0, 1, 2000.0), # (1,2,0,1) -> Goal (2,3,0,1)
    # ]
    # node_coords_test = torch.tensor([[0.0,0.0],[1.0,1.0],[2.0,2.0]], device=device, dtype=torch.float32) # lat,lon
    # d_matrix_test = torch.ones((num_nodes_test, num_nodes_test), device=device)
    # ac_matrix_test = torch.zeros((num_nodes_test, num_nodes_test), device=device)

    # V_soft_output = forward_soft_value_iteration(
    #     state_transitions=state_transitions_test,
    #     origin_node_idx=origin_node_test,
    #     cost_model=cost_model_test,
    #     num_nodes=num_nodes_test,
    #     num_time_bins_wall_clock=num_time_bins_test,
    #     num_rho_bins=num_rho_bins_test,
    #     num_phases=num_phases_test,
    #     distance_matrix_d=d_matrix_test,
    #     airspace_charge_matrix_ac=ac_matrix_test,
    #     node_coords_deg=node_coords_test,
    #     wind_model=wind_model_test,
    #     min_wall_clock_time_sec=0.0,
    #     delta_t_wall_clock_sec=300.0, # 5 minutes
    #     device=device,
    #     verbose=True
    # )
    # print("\n--- V_soft Output (example values) ---")
    # # We expect Z[0,0,1,0] = 1.0 (if it's the only origin state from transitions)
    # # V[0,0,1,0] = -log(1.0) = 0.0
    # print(f"V_soft at a specific origin state (e.g., V[{origin_node_test},0,1,0]): {V_soft_output[origin_node_test,0,1,0].item()}")
    #
    # # Z[1,1,1,0] should be Z[0,0,1,0]*exp(-cost(0->1)) = 1.0 * exp(-1) approx 0.367
    # # V[1,1,1,0] = -log(0.367) approx 1.0
    # print(f"V_soft at (1,1,1,0): {V_soft_output[1,1,1,0].item()}")
    # print(f"V_soft at (1,2,0,1): {V_soft_output[1,2,0,1].item()}") # Should also be approx 1.0
    #
    # # Z[2,2,1,0] = Z[1,1,1,0]*exp(-cost(1->2)) = exp(-1)*exp(-1) = exp(-2) approx 0.135
    # # V[2,2,1,0] = -log(exp(-2)) = 2.0
    # print(f"V_soft at (2,2,1,0): {V_soft_output[2,2,1,0].item()}")
    # # Z[2,3,0,1] = Z[1,2,0,1]*exp(-cost(1->2)) = exp(-1)*exp(-1) = exp(-2)
    # # V[2,3,0,1] = 2.0
    # print(f"V_soft at (2,3,0,1): {V_soft_output[2,3,0,1].item()}")

    # print("\n--- Full V_soft tensor (origin node slice example) ---")
    # print(V_soft_output[origin_node_test, :, :, :])
    pass

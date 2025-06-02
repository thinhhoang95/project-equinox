import torch
import math

from equinox.route.get_wind import get_wind
# For type hinting, actual instances are passed as arguments
from equinox.cost.cost_rev1 import CostRev1 
from equinox.wind.wind_model import WindModel

# Conversion factor from meters per second to knots
MPS_TO_KNOTS = 1.94384

def forward_soft_value_iteration(
    state_transitions: list[tuple[int, int, int, int, float, int, int, int, int, float]],
    # The tuple contains, in this exact order:
    # (u_idx, k_u_idx, rho_u_idx, phase_u, u_alt_ft,
    #  v_idx, k_v_idx, rho_v_idx, phase_v, v_alt_ft)
    origin_node_idx: int,
    cost_model: CostRev1,
    num_nodes: int,
    num_time_bins_wall_clock: int,
    num_rho_bins: int,
    num_phases: int,
    distance_matrix_d: torch.Tensor,
    airspace_charge_matrix_ac: torch.Tensor,
    node_coords_deg: torch.Tensor,  # Shape [num_nodes, 2] (latitude, longitude) in degrees
    wind_model: WindModel,
    min_wall_clock_time_sec: float,   # Absolute start time for k_idx=0 in seconds
    delta_t_wall_clock_sec: float,    # Duration of each wall-clock time bin in seconds
    device: torch.device,
    verbose: bool = False
) -> torch.Tensor:
    """
    Computes the soft forward value function V(s) via a log‐space “sum of exponentials.”
    We store L(s) := log Z(s), where Z(s) = sum_{u→s} exp( - cost(u→s) ) * Z(u).
    At the end, V(s) = -L(s).  Any unreachable state remains at +inf.

    Algorithmic changes from the original “linear‐Z” version:
      • We keep everything in float64 (double) and in log‐space.
      • L_val is initialized to -∞ for all states, except origin states get log(1/|origins|).
      • On each transition u→v, we compute:   a_u := L(u) – cost(u→v).
      • Then we do:  L(v) ← logaddexp( old L(v),  a_u ).
      • This avoids underflow, because we never directly compute exp(–large_value)
        in isolation; we combine “log‐masses” via logaddexp, which is numerically stable.
    """

    # 1. Create L_val and fill with -∞ (unreachable)
    L_val = torch.full(
        (num_nodes, num_time_bins_wall_clock, num_rho_bins, num_phases),
        fill_value=float('-inf'),
        dtype=torch.float64,
        device=device
    )

    # 2. Identify which (k_u, rho_u, phase_u) on the origin_node are actually used.
    actual_origin_states = set()
    for st in state_transitions:
        u_idx, k_u, rho_u, phase_u, u_alt_ft, v_idx, k_v, rho_v, phase_v, v_alt_ft = st
        if u_idx == origin_node_idx:
            actual_origin_states.add((k_u, rho_u, phase_u))

    # 3. Initialize each origin‐state to log(1 / num_actual_origin_states)
    if not actual_origin_states:
        if verbose:
            print(
                f"Warning: No transitions found starting from origin_node_idx {origin_node_idx}. "
                "All L_val entries remain -inf, so V(s)=+inf for every state."
            )
    else:
        num_actual = len(actual_origin_states)
        initial_logmass = math.log(1.0 / num_actual)  # double‐precision log
        for (k_u, rho_u, phase_u) in actual_origin_states:
            L_val[origin_node_idx, k_u, rho_u, phase_u] = initial_logmass
            if verbose:
                print(
                    f"Initialized origin state: "
                    f"L[{origin_node_idx},{k_u},{rho_u},{phase_u}] = {initial_logmass:.4e}"
                )

    # 4. Sort transitions so that when we visit (u→v), L(u) is already finalized.
    #    Primary key: k_u (source time), then u_idx, then rho_u, then phase_u.
    sorted_transitions = sorted(
        state_transitions,
        key=lambda x: (x[1], x[0], x[2], x[3])
    )

    if verbose:
        print(f"Processing {len(sorted_transitions)} state transitions in log‐space...")

    # 5. Main loop: for each transition, update L(v) = logaddexp( L(v),  L(u) - cost(u→v) ).
    for i, trans in enumerate(sorted_transitions):
        # Unpack exactly according to our documented order:
        u_idx, k_u, rho_u, u_alt_ft, phase_u, \
        v_idx, k_v, rho_v, v_alt_ft, phase_v = trans

        # a) Pull the current log‐mass at u
        L_s_u = L_val[u_idx, k_u, rho_u, phase_u]
        if torch.isneginf(L_s_u):
            # If state u is unreachable (L = -inf), skip
            continue

        # b) Compute tailwind at the source (u_idx, u_alt_ft, k_u):
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
        tailwind_knots = tailwind_mps * MPS_TO_KNOTS  # still float32 on `device`

        # c) Compute cost_uv in float64
        #   - We force distance_matrix_d, airspace_charge_matrix_ac, tailwind_knots into float64,
        #     so that cost_model runs internally in double precision (if it supports it).
        edge_u_indices = torch.tensor([u_idx], device=device, dtype=torch.long)
        edge_v_indices = torch.tensor([v_idx], device=device, dtype=torch.long)

        cost_uv_tensor = cost_model(
            (edge_u_indices, edge_v_indices),
            distance_matrix_d.to(dtype=torch.float64),
            airspace_charge_matrix_ac.to(dtype=torch.float64),
            tailwind_knots.to(dtype=torch.float64)
        )
        # cost_uv_tensor.shape == [1], dtype=float64.  Extract as Python float.
        cost_uv = float(cost_uv_tensor.item())

        # d) Form the “incoming log‐mass” from u→v:  a_u = L(u) - cost_uv
        #    Since L_s_u is a torch.float64 tensor, and cost_uv is a Python float,
        #    (L_s_u - cost_uv) remains a torch.float64 scalar on `device`.
        a_u = L_s_u - cost_uv

        # e) Merge into L_val[v] using logaddexp:
        old_L_v = L_val[v_idx, k_v, rho_v, phase_v]  # current log‐mass at v (maybe -inf)
        # torch.logaddexp handles two scalars stably:
        new_L_v = torch.logaddexp(old_L_v, a_u)
        L_val[v_idx, k_v, rho_v, phase_v] = new_L_v

        if verbose and (i % (len(sorted_transitions)//100 + 1) == 0 or i == len(sorted_transitions)-1):
            # Only print if this transition’s contribution is not extremely far below
            # the current L(v).  (We could check e^{ a_u - new_L_v } > threshold, etc.)
            print(
                f"  Transition {i+1}/{len(sorted_transitions)}:  "
                f"u=({u_idx},{k_u},{rho_u},{phase_u}) L(u)={L_s_u:.3f}, "
                f"cost={cost_uv:.3f},  a_u={a_u:.3f}  ->  "
                f"new L(v={v_idx},{k_v},{rho_v},{phase_v})={new_L_v:.3f}"
            )

    # 6. Convert back:  V_soft(s) = -L_val(s).  If L_val(s) = -inf (unreachable), V_soft(s) = +inf.
    V_soft = -L_val

    if verbose:
        num_finite = torch.isfinite(V_soft).sum().item()
        total_states = V_soft.numel()
        print(f"Forward SVI (log‐space) complete. {num_finite}/{total_states} states have finite V(s).")

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

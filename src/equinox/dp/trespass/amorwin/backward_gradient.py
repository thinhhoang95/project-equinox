import torch
from typing import Optional, Tuple, Union
import os
import numpy as np
from collections import defaultdict


def backward_gradient_pass(
    state_transitions: list[tuple[int, int, int, float, int, int, int, float, int, int]],
    avg_tailwind_knots_per_transition: torch.Tensor,
    V_f: torch.Tensor,
    V_b: torch.Tensor,
    cost_model: torch.nn.Module,
    empirical_counts: torch.Tensor,
    origin_node_idx: int,
    num_nodes: int,
    distance_matrix_d: torch.Tensor,
    airspace_charge_matrix_ac: torch.Tensor,
    device: torch.device,
    gamma: float = 1.0,
    verbose: bool = False,
    return_edge_time_sums: bool = False,
    cruise_speed_kts: Optional[float] = None,
) -> tuple[torch.Tensor, ...]:
    """
    Performs a memory-efficient backward pass to compute gradients for Maximum Entropy Inverse Learning.

    This function avoids creating a large intermediate tensor for link gradients by using a two-pass
    approach with a running sum for the final gradient.

    Pass 1: Computes the expected traversal likelihood (N_expected) for each link by summing the
            probabilities of all state transitions within that link.
    Pass 2: Computes the final gradient ∇_θ ℓ(θ) by iterating through each unique link, calculating
            its cost gradient, and immediately applying it to a running sum.
    
    The gradient of the negative log-likelihood is given by:
    ∇_θ ℓ(θ) = Σ_e [ (Σ_{s_u→s_v ∈ e} ∇_θ c(s_u,s_v;θ)) * ( N_empirical(e) - N_expected(e) ) ]
    where 'e' is a link between two waypoints.

    Args:
        state_transitions: List of all state transitions in the graph.
        avg_tailwind_knots_per_transition: Pre-computed tailwind for each transition.
        V_f: Forward soft value function (cost-from-origin).
        V_b: Backward soft value function (cost-to-go).
        cost_model: The neural network model for edge costs.
        empirical_counts: A tensor of shape (num_nodes, num_nodes) with the
                          empirical traversal counts for each link.
        origin_node_idx: The index of the origin waypoint.
        num_nodes: Total number of waypoints.
        distance_matrix_d: Matrix of distances between waypoints.
        airspace_charge_matrix_ac: Matrix of airspace charges.
        device: The PyTorch device to use for computations.
        gamma: Temperature parameter for soft operations.
        verbose: If True, prints progress information.

    Returns:
        A tuple containing:
        - link_traversal_likelihoods (torch.Tensor): Tensor of shape (num_nodes, num_nodes) with
          the expected traversal likelihood (N_expected) for each link.
        - total_log_likelihood_grad (torch.Tensor): The final gradient of the log-likelihood
          with respect to the cost model parameters.
        - log_partition_z (torch.Tensor): The log of the partition function Z.
        - edge_time_sums (torch.Tensor, optional): If `return_edge_time_sums=True`, returns an
          additional dense tensor of shape (num_nodes, num_nodes) where each entry is:
            sum_{transitions u->v} p_transition * time(transition),
          with time computed as dist / (60 * (cruise_speed_kts + tailwind_knots)).
    """
    cost_model.train()
    num_cost_params = sum(p.numel() for p in cost_model.parameters() if p.requires_grad)

    # Calculate the log partition function Z as the soft minimum of the backward values
    # at the origin node. This aggregates the cost-to-go over all possible start states.
    v_origin = V_b[origin_node_idx]
    non_inf_v = v_origin[v_origin != torch.inf]
    log_partition_z = -gamma * torch.logsumexp(-non_inf_v / gamma, dim=0) if non_inf_v.numel() > 0 else torch.tensor(float('inf'))

    link_traversal_likelihoods = torch.zeros((num_nodes, num_nodes), dtype=torch.float64, device=device)

    edge_time_sums = None
    if return_edge_time_sums:
        if cruise_speed_kts is None:
            cruise_speed_kts = float(getattr(cost_model, "cruise_speed_kts", 0.0))
        if cruise_speed_kts <= 0:
            raise ValueError("cruise_speed_kts must be positive to compute edge_time_sums.")
        edge_time_sums = torch.zeros((num_nodes, num_nodes), dtype=torch.float64, device=device)

    if torch.isinf(log_partition_z):
        if verbose:
            print("Warning: Partition function is infinite. No paths from start to goal. Gradients will be zero.")
        if return_edge_time_sums:
            return (
                link_traversal_likelihoods,
                torch.zeros(num_cost_params, dtype=torch.float64, device=device),
                log_partition_z,
                edge_time_sums,
            )
        return (
            link_traversal_likelihoods,
            torch.zeros(num_cost_params, dtype=torch.float64, device=device),
            log_partition_z,
        )

    # --- Pass 1: Compute Link Traversal Likelihoods (N_expected) ---
    if verbose:
        print("Pass 1: Computing link traversal likelihoods...")
    
    with torch.no_grad(): # No gradients needed for this pass
        for i, trans in enumerate(state_transitions):
            u_idx, k_u, rho_u, _, phase_u, v_idx, k_v, rho_v, _, phase_v = trans

            v_f_u = V_f[u_idx, k_u, rho_u, phase_u]
            v_b_v = V_b[v_idx, k_v, rho_v, phase_v]

            if torch.isinf(v_f_u) or torch.isinf(v_b_v):
                continue

            tailwind_knots = avg_tailwind_knots_per_transition[i].to(device)
            edge_u_indices = torch.tensor([u_idx], device=device, dtype=torch.long)
            edge_v_indices = torch.tensor([v_idx], device=device, dtype=torch.long)
            cost_uv = cost_model(
                (edge_u_indices, edge_v_indices),
                distance_matrix_d, airspace_charge_matrix_ac, tailwind_knots
            ).squeeze()

            log_p_transition = (-v_f_u - cost_uv - v_b_v + log_partition_z) / gamma
            p_transition = torch.exp(log_p_transition)
            if p_transition.item() == torch.inf:
                print(f"Warning: p_transition is infinite for transition {u_idx} -> {v_idx}")
            link_traversal_likelihoods[u_idx, v_idx] += p_transition #.item() not needed to prevent CPU/GPU sync

            if edge_time_sums is not None:
                dist_uv = distance_matrix_d[u_idx, v_idx].to(dtype=torch.float64)
                tailwind_knots_f = tailwind_knots.to(dtype=torch.float64)
                time_transition = dist_uv / (60.0 * (float(cruise_speed_kts) + tailwind_knots_f))
                edge_time_sums[u_idx, v_idx] += p_transition * time_transition
    
    if verbose:
        print("Pass 1 complete.")

    # --- Pass 2: Compute Final Gradient using a running sum ---
    if verbose:
        print("Pass 2: Computing final gradient with a running sum...")

    total_log_likelihood_grad = torch.zeros(num_cost_params, dtype=torch.float64, device=device)

    # Group transitions by their parent link (u_idx, v_idx)
    transitions_by_link = defaultdict(list)
    for i, t in enumerate(state_transitions):
        transitions_by_link[(t[0], t[5])].append(i)

    # Consider all links that either have transitions or appear in empirical data
    empirical_links = set(tuple(x) for x in empirical_counts.nonzero(as_tuple=False).tolist())
    all_unique_links = set(transitions_by_link.keys()).union(empirical_links)

    if verbose:
        print(f"Processing {len(all_unique_links)} unique links for gradient calculation.")

    for link_idx, (u_idx, v_idx) in enumerate(all_unique_links):
        link_grad = torch.zeros(num_cost_params, dtype=torch.float64, device=device)
        
        # We can only get a gradient for links that exist in our model (i.e., have state transitions).
        if (u_idx, v_idx) in transitions_by_link:
            
            transition_indices_for_link = transitions_by_link[(u_idx, v_idx)]
            num_transitions_in_link = len(transition_indices_for_link)
            
            tailwind_batch = avg_tailwind_knots_per_transition[transition_indices_for_link].to(device)
            
            # Prepare batch inputs for the cost model
            edge_u_indices = torch.full((num_transitions_in_link,), u_idx, device=device, dtype=torch.long)
            edge_v_indices = torch.full((num_transitions_in_link,), v_idx, device=device, dtype=torch.long)

            # Get probabilities for each state transition in this link
            with torch.no_grad():
                p_transitions_in_link = torch.zeros(num_transitions_in_link, dtype=torch.float64, device=device)
                
                # We need the costs to calculate the probabilities
                all_costs_in_link_no_grad = cost_model(
                    (edge_u_indices, edge_v_indices),
                    distance_matrix_d, airspace_charge_matrix_ac, tailwind_batch
                )

                for i, trans_master_idx in enumerate(transition_indices_for_link):
                    trans = state_transitions[trans_master_idx]
                    _u_idx, k_u, rho_u, _, phase_u, _v_idx, k_v, rho_v, _, phase_v = trans
                    v_f_u = V_f[_u_idx, k_u, rho_u, phase_u]
                    v_b_v = V_b[_v_idx, k_v, rho_v, phase_v]

                    if torch.isinf(v_f_u) or torch.isinf(v_b_v):
                        p_transitions_in_link[i] = 0.0
                        continue
                    
                    cost_uv = all_costs_in_link_no_grad[i].item() # Use already computed cost
                    log_p = (-v_f_u - cost_uv - v_b_v + log_partition_z) / gamma
                    p_transitions_in_link[i] = torch.exp(log_p)

                # Normalize probabilities to get conditional P(s_u->s_v | e)
                total_p_for_link = p_transitions_in_link.sum()
                if total_p_for_link > 1e-9:
                    weights = p_transitions_in_link / total_p_for_link
                else:
                    # weights = torch.zeros_like(p_transitions_in_link)
                    # If total probability is negligible, attribute all of it to the
                    # single most likely state transition to provide a gradient signal.
                    weights = torch.zeros_like(p_transitions_in_link)
                    if p_transitions_in_link.numel() > 0:
                        max_prob_idx = torch.argmax(p_transitions_in_link)
                        weights[max_prob_idx] = 1.0
            
            # Calculate cost gradient for each state transition and perform a weighted sum
            for i in range(num_transitions_in_link):
                if weights[i].item() > 1e-9: # Only compute gradient if weight is non-trivial
                    cost_model.zero_grad()
                    
                    # Recalculate cost for this specific transition to build the graph
                    single_cost = cost_model(
                        (edge_u_indices[i].unsqueeze(0), edge_v_indices[i].unsqueeze(0)),
                        distance_matrix_d, airspace_charge_matrix_ac, tailwind_batch[i].unsqueeze(0)
                    )
                    
                    single_cost.backward()
                    
                    grad_vector = []
                    for param in cost_model.parameters():
                        if param.requires_grad:
                            if param.grad is None:
                                grad_vector.append(torch.zeros_like(param.detach()).flatten())
                            else:
                                grad_vector.append(param.grad.detach().flatten())
                    
                    if grad_vector:
                        single_transition_grad = torch.cat(grad_vector)
                        link_grad += weights[i] * single_transition_grad


        # Update the total gradient using the gradient of the link's cost
        n_empirical = empirical_counts[u_idx, v_idx]
        n_expected = link_traversal_likelihoods[u_idx, v_idx]
        
        # Gradient of negative log-likelihood is E_empirical - E_model
        total_log_likelihood_grad += (link_grad / gamma) * (n_empirical - n_expected)

        if verbose and (link_idx > 0 and link_idx % 1000 == 0 or link_idx == len(all_unique_links) - 1):
            print(f"\r  Processed {link_idx + 1}/{len(all_unique_links)} links...", end="", flush=True)

    if verbose:
        print("\nPass 2 complete. Final gradient computed.")

    if return_edge_time_sums:
        return link_traversal_likelihoods, total_log_likelihood_grad, log_partition_z, edge_time_sums
    return link_traversal_likelihoods, total_log_likelihood_grad, log_partition_z


# if __name__ == '__main__':
#     # This is an example of how to use the backward_gradient_pass function.
#     # It requires mock data or loading real data from the project.
#     print("Running example for backward_gradient_pass...")

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     # --- Mock Configuration ---
#     # In a real scenario, this would come from a config file or object
#     class MockConfig:
#         file_prefix = "LFOO_LFPB_1633"
#         num_nodes = 800  # Example value
#         num_time_bins_wall_clock = 24
#         num_rho_bins = 10
#         num_phases = 3
#         origin_node_idx = 0 # Example
#         goal_node_idx = 100 # Example

#     config = MockConfig()
    
#     # Define shapes
#     v_shape = (config.num_nodes, config.num_time_bins_wall_clock, config.num_rho_bins, config.num_phases)
    
#     # --- Create Mock Data / Placeholders ---
#     # In a real run, these would be loaded from files
#     print("Loading mock data and value functions...")
#     # These paths are placeholders and will likely fail unless the data exists
#     v_fwd_path = f"data/graph/V_soft/{config.file_prefix}_V_FWD_SPRSE_WIND.pt"
#     v_bwd_path = f"data/graph/V_soft/{config.file_prefix}_V_BWD_SPRSE_WIND.pt"

#     try:
#         # Attempt to load real data if available
#         V_f = load_value_function(v_fwd_path, v_shape, device)
#         V_b = load_value_function(v_bwd_path, v_shape, device)
#         state_transitions = torch.load(f"data/graph/{config.file_prefix}_state_transitions.pt")
#         avg_tailwind = torch.load(f"data/graph/{config.file_prefix}_avg_tailwind_knots_per_transition.pt")
#         distance_matrix_d = torch.load("data/distance_matrix_d.pt")
#         airspace_charge_matrix_ac = torch.load("data/airspace_charge_matrix_ac.pt")
#         empirical_counts = torch.load(f"data/empirical_counts/{config.file_prefix}_empirical_counts.pt").to(device, dtype=torch.float64)
#         print("Successfully loaded data files.")
#     except FileNotFoundError as e:
#         print(f"Could not load data file: {e}. Using dummy data instead.")
#         # Create dummy data if files are not found
#         num_transitions = 10000
#         V_f = torch.rand(v_shape, device=device, dtype=torch.float64) * 10
#         V_b = torch.rand(v_shape, device=device, dtype=torch.float64) * 10
#         V_b[config.goal_node_idx, :, :, :] = 0 # Goal has 0 cost-to-go
#         state_transitions = [
#             (
#                 np.random.randint(0, config.num_nodes), np.random.randint(0, config.num_time_bins_wall_clock), np.random.randint(0, config.num_rho_bins), 0.0, np.random.randint(0, config.num_phases),
#                 np.random.randint(0, config.num_nodes), np.random.randint(0, config.num_time_bins_wall_clock), np.random.randint(0, config.num_rho_bins), 0.0, np.random.randint(0, config.num_phases)
#             ) for _ in range(num_transitions)
#         ]
#         avg_tailwind = torch.randn(num_transitions, device=device)
#         distance_matrix_d = torch.rand((config.num_nodes, config.num_nodes), device=device)
#         airspace_charge_matrix_ac = torch.rand((config.num_nodes, config.num_nodes), device=device)
#         empirical_counts = torch.zeros((config.num_nodes, config.num_nodes), device=device, dtype=torch.float64)
#         empirical_counts[0, 1] = 1.0 # Mock one empirical trajectory

#     # --- Initialize Cost Model ---
#     cost_model = CostRev2(
#         beta0=0.0, beta1=1.0, beta2=1.0, beta3=1.0,
#         num_waypoints=config.num_nodes,
#         device=device
#     )
#     # Ensure preference matrix requires gradients
#     cost_model.preference_matrix_p.requires_grad = True

#     # --- Run Gradient Pass ---
#     print("Executing backward_gradient_pass...")
    
#     likelihoods, grad, log_partition_z = backward_gradient_pass(
#         state_transitions=state_transitions,
#         avg_tailwind_knots_per_transition=avg_tailwind,
#         V_f=V_f,
#         V_b=V_b,
#         cost_model=cost_model,
#         empirical_counts=empirical_counts,
#         origin_node_idx=config.origin_node_idx,
#         num_nodes=config.num_nodes,
#         distance_matrix_d=distance_matrix_d,
#         airspace_charge_matrix_ac=airspace_charge_matrix_ac,
#         device=device,
#         verbose=True
#     )

#     print("\n--- Results ---")
#     print(f"Shape of traversal likelihoods: {likelihoods.shape}")
#     print(f"Sum of traversal likelihoods: {likelihoods.sum().item():.4f} (should be close to 1.0)")
#     print(f"Shape of final gradient: {grad.shape}")
#     print(f"Gradient values (first 10): {grad[:10].tolist()}")

#     # --- Perform a Gradient Descent Step (Example) ---
#     print("\n--- Example SGD Step ---")
#     optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, cost_model.parameters()), lr=0.01)
    
#     # Manually assign gradients
#     param_idx = 0
#     for param in cost_model.parameters():
#         if param.requires_grad:
#             num_param_elements = param.numel()
#             # Reshape grad to match param shape
#             param.grad = grad[param_idx : param_idx + num_param_elements].view(param.shape).to(param.dtype)
#             param_idx += num_param_elements

#     # Check preference matrix before step
#     pref_matrix_before = cost_model.preference_matrix_p.detach().clone()
    
#     optimizer.step()

#     # Check preference matrix after step
#     pref_matrix_after = cost_model.preference_matrix_p.detach().clone()
    
#     change = torch.norm(pref_matrix_after - pref_matrix_before).item()
#     print(f"Norm of change in preference matrix after one SGD step: {change:.6f}")
#     if change > 1e-9:
#         print("OK: Parameters were updated.")
#     else:
#         print("WARNING: Parameters did not update. Check gradient calculation or optimizer setup.")

#     print("\nExample run finished.")

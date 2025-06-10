import torch
import numpy as np
import networkx as nx
from equinox.dp.trespass.sparse_io_utils import load_sparse_coo_tensor_with_convention

def sample_tres_trajectory(
    G: nx.DiGraph,
    node_to_idx: dict,
    idx_to_node: dict,
    origin_node_id: str,
    goal_node_id: str,
    initial_rho: int,
    initial_phase: int,
    backward_values: torch.Tensor, # V_soft_bwd_np from test_tres.py (num_nodes, num_time_bins_wall_clock, num_rho_bins, num_phases)
    edge_costs_uv: torch.Tensor, # edge_costs from backward_svi (sparse COO)
    max_steps: int = 1000 # Max steps to prevent infinite loops
):
    """
    Samples a single trajectory using the TResPASS algorithm.
    The initial wall-clock time bin (k) is sampled uniformly from all valid starting k bins
    for the given origin_node_id, initial_rho, and initial_phase, based on finite and positive
    values in `backward_values`.

    Args:
        G: The route graph.
        node_to_idx: Mapping from node ID to an integer index.
        idx_to_node: Mapping from integer index to node ID.
        origin_node_id: The starting node ID (e.g., "LEMD").
        goal_node_id: The target node ID (e.g., "EGLL").
        initial_rho: Initial climb time remaining bin index.
        initial_phase: Initial flight phase (0: CLIMB, 1: CRUISE, 2: DESCENT).
        backward_values: The precomputed backward soft value function (Z_b in pseudo-code).
                         Shape: (num_nodes, num_k_bins, num_rho_bins, num_phase_bins)
        edge_costs_uv: Sparse COO tensor of edge costs.
                       Indices: (u_idx, k_u, rho_u, phase_u, v_idx, k_v, rho_v, phase_v)
        max_steps: Maximum number of steps in a trajectory to prevent infinite loops.

    Returns:
        A list of state tuples representing the sampled trajectory.
        Each state is (node_id, k, rho, phase).
        Raises exceptions if sampling fails for debugging purposes.
    """
    current_node_idx = node_to_idx[origin_node_id]
    
    num_k_bins = backward_values.shape[1]
    
    # Find valid initial k values
    valid_initial_ks = []
    for k_candidate in range(num_k_bins):
        val = backward_values[current_node_idx, k_candidate, initial_rho, initial_phase].item()
        if np.isfinite(val) and val > 0: # Z_b must be finite and positive
            valid_initial_ks.append(k_candidate)

    if not valid_initial_ks:
        raise ValueError(
            f"No valid initial k found for origin_node_id='{origin_node_id}' (idx={current_node_idx}), "
            f"initial_rho={initial_rho}, initial_phase={initial_phase}. "
            "All Z_b values are non-positive or non-finite."
        )

    # Uniformly sample initial_k from the valid ones
    current_k = np.random.choice(valid_initial_ks)
    current_rho = initial_rho
    current_phase = initial_phase

    goal_node_idx = node_to_idx[goal_node_id]
    
    trajectory = [(idx_to_node[current_node_idx], current_k, current_rho, current_phase)]

    # The backward_values tensor is Z_b
    # Z_b[i] in pseudo-code corresponds to backward_values[node_idx, k, rho, phase]
    # c_ij in pseudo-code corresponds to edge_costs_uv for the transition from state i to state j

    # Convert sparse edge_costs_uv to a dense representation for easier lookup if it's not too large,
    # or filter edges efficiently. For now, assume we can iterate through relevant outgoing edges.
    # The edge_costs_uv is a sparse COO tensor.
    # We need to find all (v_idx, k_v, rho_v, phase_v) reachable from (current_node_idx, current_k, current_rho, current_phase)

    num_nodes = backward_values.shape[0]
    num_k_bins = backward_values.shape[1]
    num_rho_bins = backward_values.shape[2]
    num_phase_bins = backward_values.shape[3]

    for step in range(max_steps):
        if current_node_idx == goal_node_idx:
            # Potentially add a condition for k, rho, phase if goal is more specific
            # For now, reaching the goal node is sufficient
            print(f"Goal {goal_node_id} reached at step {step}.")
            return trajectory

        current_state_tuple = (current_node_idx, current_k, current_rho, current_phase)
        
        # Z_b[i]
        Z_b_i = backward_values[current_node_idx, current_k, current_rho, current_phase].item()

        if not np.isfinite(Z_b_i) or Z_b_i == 0:
            raise ValueError(f"Z_b[i] is {Z_b_i} at state {idx_to_node[current_node_idx], current_k, current_rho, current_phase}. State is unreachable or has no path to goal.")

        possible_next_states = []
        probabilities = []

        # Find relevant edges from edge_costs_uv
        # edge_costs_uv.indices() are [8, num_edges]
        # [u_idx, k_u, rho_u, phase_u, v_idx, k_v, rho_v, phase_v]
        
        # Filter edges starting from the current state (u_idx, k_u, rho_u, phase_u)
        mask = (edge_costs_uv.indices()[0] == current_node_idx) & \
               (edge_costs_uv.indices()[1] == current_k) & \
               (edge_costs_uv.indices()[2] == current_rho) & \
               (edge_costs_uv.indices()[3] == current_phase)
        
        relevant_edges_indices = edge_costs_uv.indices()[:, mask] # Shape [8, num_relevant_edges]
        relevant_costs = edge_costs_uv.values()[mask]       # Shape [num_relevant_edges]

        if relevant_edges_indices.shape[1] == 0:
            raise RuntimeError(f"No outgoing edges found from state {idx_to_node[current_node_idx], current_k, current_rho, current_phase}. State is terminal or graph is incomplete.")

        for edge_idx in range(relevant_edges_indices.shape[1]):
            next_node_idx = relevant_edges_indices[4, edge_idx].item()
            next_k = relevant_edges_indices[5, edge_idx].item()
            next_rho = relevant_edges_indices[6, edge_idx].item()
            next_phase = relevant_edges_indices[7, edge_idx].item()
            
            cost_ij = relevant_costs[edge_idx].item()
            
            # Z_b[j]
            Z_b_j = backward_values[next_node_idx, next_k, next_rho, next_phase].item()

            if not np.isfinite(Z_b_j):
                print(f"Warning: Z_b[j] is {Z_b_j} for next state {idx_to_node[next_node_idx], next_k, next_rho, next_phase}. Skipping this transition.")
                continue

            # π(j | i) = e^{-c_{ij}} Z_b[j] / Z_b[i]
            # Note: The prompt uses e^{-c_ij}. If edge_costs_uv already stores log-probabilities or similar, this might change.
            # Assuming c_ij is the direct cost here.
            # The backward SVI calculates V_bwd(i) = log Σ_j exp( -cost(i,j) + V_bwd(j) ).
            # So Z_b[i] = exp(V_bwd[i]).
            # Then prob = exp(-cost(i,j)) * exp(V_bwd[j]) / exp(V_bwd[i])
            #           = exp( -cost(i,j) + V_bwd[j] - V_bwd[i] )
            # This is consistent with the prompt's π(j | i) = e^{-c_{ij}} Z_b[j] / Z_b[i]
            # where Z_b[i] is from the backward pass (our backward_values).

            print(f"{idx_to_node[current_node_idx], current_k, current_rho, current_phase} → {idx_to_node[next_node_idx], next_k, next_rho, next_phase}  Cost: {cost_ij}, Z_b[j]: {Z_b_j}, Z_b[i]: {Z_b_i}")
            
            prob = np.exp(-cost_ij) * Z_b_j / Z_b_i
            
            if prob > 0 and np.isfinite(prob): # Ensure probability is valid
                possible_next_states.append((next_node_idx, next_k, next_rho, next_phase))
                probabilities.append(prob)
            # else:
                # print(f"Warning: Invalid probability {prob} for transition to {idx_to_node[next_node_idx], next_k, next_rho, next_phase}. Skipping.")


        if not possible_next_states:
            raise RuntimeError(f"No valid next states with finite probabilities from {idx_to_node[current_node_idx], current_k, current_rho, current_phase}. All transitions have invalid probabilities.")

        # Normalize probabilities
        probabilities = np.array(probabilities)
        if np.sum(probabilities) == 0: # Should not happen if we checked prob > 0
            raise ValueError(f"Sum of probabilities is zero at state: {idx_to_node[current_node_idx], current_k, current_rho, current_phase}. All probabilities are invalid.")
            
        probabilities /= np.sum(probabilities)

        # Sample next state
        try:
            choice_idx = np.random.choice(len(possible_next_states), p=probabilities)
        except ValueError as e:
            raise ValueError(f"Error during np.random.choice at state {idx_to_node[current_node_idx], current_k, current_rho, current_phase}: {e}. Probabilities: {probabilities}, Sum: {np.sum(probabilities)}")

        next_state = possible_next_states[choice_idx]
        
        current_node_idx, current_k, current_rho, current_phase = next_state
        trajectory.append((idx_to_node[current_node_idx], current_k, current_rho, current_phase))

    raise RuntimeError(f"Max steps {max_steps} reached before finding goal {goal_node_id}. Trajectory may be stuck in a loop or goal is unreachable.")

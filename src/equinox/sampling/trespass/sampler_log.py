from venv import logger
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
    initial_rho: int, # should be equal to the max number of climb time bins 
    initial_phase: int,
    soft_cost_to_go: torch.Tensor, # V_bwd (num_nodes, num_time_bins_wall_clock, num_rho_bins, num_phases)
    edge_costs_uv: torch.Tensor, # edge_costs from backward_svi (sparse COO)
    gamma: float = 0.1, # this temperature should also be the same as the temperature used in the backward_svi
    max_steps: int = 1000 # Max steps to prevent infinite loops
):
    """
    Samples a single trajectory using the TResPASS algorithm.
    The initial wall-clock time bin (k) is sampled uniformly from all valid starting k bins
    for the given origin_node_id, initial_rho, and initial_phase, based on finite
    values in `soft_cost_to_go`.

    Args:
        G: The route graph.
        node_to_idx: Mapping from node ID to an integer index.
        idx_to_node: Mapping from integer index to node ID.
        origin_node_id: The starting node ID (e.g., "LEMD").
        goal_node_id: The target node ID (e.g., "EGLL").
        initial_rho: Initial climb time remaining bin index.
        initial_phase: Initial flight phase (0: CLIMB, 1: CRUISE, 2: DESCENT).
        soft_cost_to_go: The precomputed soft cost-to-go, V_bwd. This tensor is used for
                         sampling probabilities.
                         Shape: (num_nodes, num_k_bins, num_rho_bins, num_phase_bins).
        edge_costs_uv: Sparse COO tensor of edge costs.
                       Indices: (u_idx, k_u, rho_u, phase_u, v_idx, k_v, rho_v, phase_v)
        gamma: The sampling temperature. Defaults to 1.0.
        max_steps: Maximum number of steps in a trajectory to prevent infinite loops.

    Returns:
        A tuple containing:
        - A list of state tuples representing the sampled trajectory.
          Each state is (node_id, k, rho, phase).
        - A list of floats representing the cost of each transition in the trajectory.
        Raises exceptions if sampling fails for debugging purposes.
    """
    current_node_idx = node_to_idx[origin_node_id]
    
    num_k_bins = soft_cost_to_go.shape[1]
    
    # Find valid initial k values
    valid_initial_ks = []
    for k_candidate in range(num_k_bins):
        # V_bwd is the soft cost-to-go. Z_b = exp(-V_bwd).
        # We need Z_b to be finite and positive, which means V_bwd must be finite.
        v_bwd = soft_cost_to_go[current_node_idx, k_candidate, initial_rho, initial_phase].item()
        if np.isfinite(v_bwd):
            valid_initial_ks.append(k_candidate)

    if not valid_initial_ks:
        raise ValueError(
            f"No valid initial k found for origin_node_id='{origin_node_id}' (idx={current_node_idx}), "
            f"initial_rho={initial_rho}, initial_phase={initial_phase}. "
            "All V_bwd values are non-finite."
        )

    # Uniformly sample initial_k from the valid ones
    current_k = np.random.choice(valid_initial_ks)
    current_rho = initial_rho
    current_phase = initial_phase

    # FOR DEBUGGING
    print("WARNING: DEBUGGING MODE, setting current_k, current_rho, current_phase to 39, 36, 0")
    print("*" * 100)
    # Find the last non-inf value of soft_cost_to_go[current_node_idx, :, current_rho, current_phase]
    cost_slice = soft_cost_to_go[current_node_idx, :, current_rho, current_phase]
    finite_mask = torch.isfinite(cost_slice)
    if finite_mask.any():
        # Find the last (highest index) finite value
        finite_indices = torch.where(finite_mask)[0]
        current_k = finite_indices[-1].item()
        logger.warning(f"Setting current_k to {current_k} because it is the last non-inf value of soft_cost_to_go[current_node_idx, :, current_rho, current_phase]")
    else:
        # Fallback if no finite values found
        raise ValueError(f"No finite values found for state {idx_to_node[current_node_idx], current_rho, current_phase}. All values are non-finite.")
    

    goal_node_idx = node_to_idx[goal_node_id]
    
    trajectory = [(idx_to_node[current_node_idx], current_k, current_rho, current_phase)]
    trajectory_costs = []

    # The soft_cost_to_go tensor is V_bwd
    # V_bwd[i] in pseudo-code corresponds to soft_cost_to_go[node_idx, k, rho, phase]
    # c_ij in pseudo-code corresponds to edge_costs_uv for the transition from state i to state j

    # Convert sparse edge_costs_uv to a dense representation for easier lookup if it's not too large,
    # or filter edges efficiently. For now, assume we can iterate through relevant outgoing edges.
    # The edge_costs_uv is a sparse COO tensor.
    # We need to find all (v_idx, k_v, rho_v, phase_v) reachable from (current_node_idx, current_k, current_rho, current_phase)

    num_nodes = soft_cost_to_go.shape[0]
    num_k_bins = soft_cost_to_go.shape[1]
    num_rho_bins = soft_cost_to_go.shape[2]
    num_phase_bins = soft_cost_to_go.shape[3]

    for step in range(max_steps):
        if current_node_idx == goal_node_idx:
            # Potentially add a condition for k, rho, phase if goal is more specific
            # For now, reaching the goal node is sufficient
            print(f"Goal {goal_node_id} reached at step {step}.")
            return trajectory, trajectory_costs

        current_state_tuple = (current_node_idx, current_k, current_rho, current_phase)
        
        # V_bwd[i]
        V_bwd_i = soft_cost_to_go[current_node_idx, current_k, current_rho, current_phase].item()

        if not np.isfinite(V_bwd_i):
            raise ValueError(f"V_bwd[i] is {V_bwd_i} at state {idx_to_node[current_node_idx], current_k, current_rho, current_phase}. State is unreachable or has no path to goal.")

        possible_next_transitions = []
        probabilities = []
        components = []

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
            
            # V_bwd[j]
            V_bwd_j = soft_cost_to_go[next_node_idx, next_k, next_rho, next_phase].item()

            if not np.isfinite(V_bwd_j):
                print(f"Warning: V_bwd[j] is {V_bwd_j} for next state {idx_to_node[next_node_idx], next_k, next_rho, next_phase}. Skipping this transition.")
                continue

            # The transition probability π(j | i) is derived from the soft Bellman equation.
            # With a temperature parameter γ, the un-normalized probability is:
            # π(j | i) ∝ exp(-(cost_ij + V_bwd_j - V_bwd_i) / γ)
            
            # print(f"{idx_to_node[current_node_idx], current_k, current_rho, current_phase} → {idx_to_node[next_node_idx], next_k, next_rho, next_phase}  Cost: {cost_ij}, V_bwd[j]: {V_bwd_j}, V_bwd[i]: {V_bwd_i}")
            
            prob = np.exp(-(cost_ij + V_bwd_j - V_bwd_i) / gamma)
            
            if prob > 0 and np.isfinite(prob): # Ensure probability is valid
                possible_next_transitions.append(
                    ((next_node_idx, next_k, next_rho, next_phase), cost_ij)
                )
                probabilities.append(prob)
                components.append((V_bwd_i, V_bwd_j, cost_ij, prob))
            # else:
                # print(f"Warning: Invalid probability {prob} for transition to {idx_to_node[next_node_idx], next_k, next_rho, next_phase}. Skipping.")


        if not possible_next_transitions:
            raise RuntimeError(f"No valid next states with finite probabilities from {idx_to_node[current_node_idx], current_k, current_rho, current_phase}. All transitions have invalid probabilities.")

        # Normalize probabilities
        probabilities = np.array(probabilities)
        probabilities_sum = np.sum(probabilities)
        if probabilities_sum == 0: # Should not happen if we checked prob > 0
            raise ValueError(f"Sum of probabilities is zero at state: {idx_to_node[current_node_idx], current_k, current_rho, current_phase}. All probabilities are invalid.")
        else:
            if abs(probabilities_sum - 1) > 1e-1:
                # print(f"WARNING: Probabilities sum: {probabilities_sum} at state: {idx_to_node[current_node_idx], current_k, current_rho, current_phase}. All probabilities are invalid.")
                raise ValueError(f"Probabilities sum: {probabilities_sum} at state: {idx_to_node[current_node_idx], current_k, current_rho, current_phase}. All probabilities are invalid.")

        probabilities /= probabilities_sum # SHOULD NOT BE NECESSARY???

        # DEBUGGING
        # for i in range(len(possible_next_transitions)):
        #     print(
        #         f"From {idx_to_node[current_node_idx], current_k, current_rho, current_phase} "
        #         f"to {idx_to_node[possible_next_transitions[i][0][0]], possible_next_transitions[i][0][1], possible_next_transitions[i][0][2], possible_next_transitions[i][0][3]} "
        #         f"with probability {probabilities[i]} "
        #         f"(V_bwd[i]: {components[i][0]:.3e}, V_bwd[j]: {components[i][1]:.3e}, cost_ij: {components[i][2]})"
        #     )

        # Sample next state
        try:
            choice_idx = np.random.choice(len(possible_next_transitions), p=probabilities)
            # Greedy choice: pick the transition with the highest probability
            # print("WARNING: DEBUGGING MODE, USING GREEDY CHOICE")
            # choice_idx = int(np.argmax(probabilities))
        except ValueError as e:
            raise ValueError(f"Error during np.random.choice at state {idx_to_node[current_node_idx], current_k, current_rho, current_phase}: {e}. Probabilities: {probabilities}, Sum: {np.sum(probabilities)}")

        next_state, transition_cost = possible_next_transitions[choice_idx]
        
        current_node_idx, current_k, current_rho, current_phase = next_state
        trajectory.append((idx_to_node[current_node_idx], current_k, current_rho, current_phase))
        trajectory_costs.append(transition_cost)

    raise RuntimeError(f"Max steps {max_steps} reached before finding goal {goal_node_id}. Trajectory may be stuck in a loop or goal is unreachable.")

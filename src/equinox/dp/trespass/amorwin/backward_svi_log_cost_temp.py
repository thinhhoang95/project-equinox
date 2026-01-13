import torch
import math

import networkx as nx

def backward_soft_value_iteration(
    state_transitions: list[tuple[int, int, int, float, int, int, int, float, int, int]], # Note: types for alt and phase might be swapped in usage
    avg_tailwind_knots_per_transition: torch.Tensor,
    # The tuple, based on apparent usage in forward_svi (unpacking on L262 fwd_svi):
    # (u_idx, k_u, rho_u, u_alt_ft, phase_u,
    #  v_idx, k_v, rho_v, v_alt_ft, phase_v)
    # Original docstring for fwd_svi had phase_u at index 3, u_alt_ft at index 4.
    # We follow the variable names from unpacking in fwd_svi:
    # trans[0]=u_idx, trans[1]=k_u, trans[2]=rho_u, trans[3]=u_alt_ft, trans[4]=phase_u
    # trans[5]=v_idx, trans[6]=k_v, trans[7]=rho_v, trans[8]=v_alt_ft, trans[9]=phase_v
    G: nx.DiGraph,
    idx_to_node: dict[int, str],
    goal_node_idx: int, # Changed from origin_node_idx
    cost_model: torch.nn.Module,
    num_nodes: int,
    num_time_bins_wall_clock: int,
    num_rho_bins: int,
    num_phases: int,
    distance_matrix_d: torch.Tensor,
    airspace_charge_matrix_ac: torch.Tensor,
    device: torch.device,
    gamma: float = 0.1, # this temperature should also be the same as the temperature used in the sampler
    verbose: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the soft backward value function (cost-to-go) V(s) via soft value iteration.
    It also returns a tensor containing the costs of all processed edges.

    This function implements backward soft value iteration to calculate the soft-optimal
    cost-to-go from each state to a designated goal state.

    The algorithm computes V(s), the soft-minimum cost to go from state s to the goal.
    The soft-Bellman equation for backward value iteration is:
    V(s) = softmin_{s→v} (cost(s→v) + V(v))
    where softmin is the log-sum-exp operator with temperature γ:
    softmin({x_i}) = -γ * log(sum_i(exp(-x_i / γ))).
    A smaller γ biases the process towards the shortest path (hard-min).

    ## Parameters

    - **state_transitions** (`list[tuple[int, int, int, float, int, int, int, float, int, int]]`):
      List of state transitions. Based on usage in the forward pass, we assume the tuple elements are:
      `(u_idx, k_u, rho_u, u_alt_ft, phase_u, v_idx, k_v, rho_v, v_alt_ft, phase_v)`.
      `u_alt_ft` is altitude at u (used for wind), `phase_u` is phase at u (used for V_val indexing).
      Similarly for `v_alt_ft` and `phase_v`.
      The types in the signature `(..., float, int, ..., float, int)` reflect this `alt, phase` order.

    - **avg_tailwind_knots_per_transition** (`torch.Tensor`): A 1D tensor of pre-computed
      average tailwind in knots for each transition, matching the order of `state_transitions`.
    - **G** (`nx.DiGraph`): NetworkX directed graph.
    - **idx_to_node** (`dict[int, str]`): Mapping from integer node indices to string node IDs.
    - **goal_node_idx** (`int`): Index of the goal node.
    - **cost_model** (`CostRev1`): Cost model instance.
    - **num_nodes** (`int`): Total number of nodes.
    - **num_time_bins_wall_clock** (`int`): Number of wall-clock time bins.
    - **num_rho_bins** (`int`): Number of climb/profile time bins.
    - **num_phases** (`int`): Number of flight phases.
    - **distance_matrix_d** (`torch.Tensor`): Pairwise distances between nodes.
    - **airspace_charge_matrix_ac** (`torch.Tensor`): Airspace charges.
    - **device** (`torch.device`): PyTorch device.
    - **gamma** (`float`): Temperature parameter for the soft-min operator. A smaller `gamma`
      makes the soft-min approximate the hard-min more closely.
    - **verbose** (`bool`, optional): If True, prints progress. Defaults to False.

    ## Returns

    **tuple[torch.Tensor, torch.Tensor]**:
    - **V_soft_cost_to_go** (`torch.Tensor`): Soft value function V (cost-to-go) with shape `(num_nodes,
      num_time_bins_wall_clock, num_rho_bins, num_phases)`. Values represent soft cost-to-go,
      with +inf for states from which the goal is unreachable.
    - **edge_costs_uv** (`torch.Tensor`): Sparse COO tensor storing the computed cost for each processed transition (u,v).
      Shape: `(num_nodes, num_time_bins_wall_clock, num_rho_bins, num_phases,  # u state
               num_nodes, num_time_bins_wall_clock, num_rho_bins, num_phases)`. # v state
      Values are `float64`. Accessing an element corresponding to an unprocessed transition
      (i.e., one not in `state_transitions` or skipped) will yield 0. This 0 should typically
      be interpreted as `float('inf')` (representing an unreachable or non-existent path).
      The cost of a processed transition from state `(u_idx, k_u, rho_u, phase_u)` to state
      `(v_idx, k_v, rho_v, phase_v)` is stored.

    ## Algorithm Details

    - Computations are in float64.
    - The value function `V_val` is initialized to +∞. For actual goal states (g_node, k, rho, phase)
      that are destinations of transitions, V(g_state) = 0.
    - Transitions `u→v` are processed in an order ensuring V(v) is finalized before updating V(u).
      This typically means iterating nodes in reverse topological order.
    - For each transition u→v: compute `val_from_v := V(v) + cost(u→v)`.
    - Update: V(u) ← softmin(old V(u), val_from_v).
      This is computed as: `V(u) = -γ * log(exp(-V_old(u)/γ) + exp(-val_from_v/γ))`.
    - Final result: The function returns the computed V_val tensor.
    """

    distance_matrix_d = distance_matrix_d.to(dtype=torch.float64)
    airspace_charge_matrix_ac = airspace_charge_matrix_ac.to(dtype=torch.float64)
    avg_tailwind_knots_per_transition = avg_tailwind_knots_per_transition.to(dtype=torch.float64)

    # 1. Create V_val and fill with +∞
    V_val = torch.full(
        (num_nodes, num_time_bins_wall_clock, num_rho_bins, num_phases),
        fill_value=float('inf'),
        dtype=torch.float64,
        device=device
    )

    # Initialize lists to store indices and values for the sparse edge_costs_uv tensor
    processed_cost_indices_list = []
    processed_cost_values_list = []
    # Define the full 8D shape for the sparse tensor
    full_shape_8d = (num_nodes, num_time_bins_wall_clock, num_rho_bins, num_phases,
                     num_nodes, num_time_bins_wall_clock, num_rho_bins, num_phases)

    # 2. Identify actual goal states (v_idx, k_v, rho_v, phase_v)
    #    These are states at goal_node_idx that are targets of any transition.
    #    Unpacking: trans[5]=v_idx, trans[6]=k_v, trans[7]=rho_v, trans[9]=phase_v
    actual_goal_states = set()
    for st in state_transitions:
        # st = (u_idx, k_u, rho_u, u_alt_ft, phase_u,
        #       v_idx, k_v, rho_v, v_alt_ft, phase_v)
        if st[5] == goal_node_idx: # v_idx == goal_node_idx
            actual_goal_states.add((st[6], st[7], st[9])) # (k_v, rho_v, phase_v)

    # 3. Initialize V_val for actual_goal_states to 0.0
    if not actual_goal_states:
        if verbose:
            print(
                f"Warning: No transitions found ending at goal_node_idx {goal_node_idx}. "
                "All V_val entries will remain +inf."
            )
    else:
        for (k_g, rho_g, phase_g) in actual_goal_states:
            # Ensure indices are within bounds before assignment
            if (0 <= k_g < num_time_bins_wall_clock and
                0 <= rho_g < num_rho_bins and
                0 <= phase_g < num_phases):
                V_val[goal_node_idx, k_g, rho_g, phase_g] = 0.0
                if verbose:
                    print(
                        f"Initialized goal state: "
                        f"V[{goal_node_idx},{k_g},{rho_g},{phase_g}] = 0.0"
                    )
            elif verbose: # Print warning if an identified goal state is out of bounds for V_val
                 print(
                    f"Warning: Identified goal state ({goal_node_idx},{k_g},{rho_g},{phase_g}) "
                    f"is out of bounds for V_val dimensions "
                    f"({num_time_bins_wall_clock},{num_rho_bins},{num_phases}). Skipping initialization."
                )

    # 4. Sort transitions so that when we visit (u→v) to update V(u), V(v) is already finalized.
    #    Primary key: topological sort order of v_idx (destination), descending.
    #    Then k_v, phase_v, rho_v (ascending as tie-breakers).
    #    Indices for v: trans[5]=v_idx, trans[6]=k_v, trans[7]=rho_v, trans[9]=phase_v
    try:
        topo_order_str_nodes = list(nx.topological_sort(G))
        node_str_to_topo_rank = {node_name_str: rank for rank, node_name_str in enumerate(topo_order_str_nodes)}
    except nx.NetworkXUnfeasible:
        raise ValueError("The graph G must be a DAG for topological sorting.")
    except Exception as e:
        raise ValueError(f"Error during topological sort: {e}. Ensure G is a DAG.")

    # Sort by v_idx's topological rank (desc), then k_v, phase_v, rho_v
    # To sort by rank descending, we can use `rank` and `reverse=True` on the whole sort,
    # or sort by `-rank` ascending.
    # Let's sort by rank (ascending) and then iterate the sorted_transitions list in reverse.
    # Or, more directly, use reverse=True on the primary key's contribution if possible,
    # or sort by negative rank.
    
    # Sorting key:
    # - Topological rank of v_idx (descending).
    # - k_v (ascending)
    # - phase_v (ascending)
    # - rho_v (ascending)
    # trans[5] = v_idx, trans[6] = k_v, trans[9] = phase_v, trans[7] = rho_v
    
    # Associate original indices with transitions before sorting
    indexed_transitions = list(enumerate(state_transitions))
    
    # Create a reverse map for topological rank (higher value for earlier nodes in topo sort)
    # No, we need lower value for nodes that are "later" in topological sort (closer to typical sinks)
    # so that when sorted ascending, these appear first, and their predecessors later.
    # For backward pass, we process states v in reverse topological order.
    # This means states "closer" to the goal (higher topological rank) are processed first.
    # Their V(v) values are used to update V(u) for u -> v.
    # So, we need transitions sorted such that V(v) is known.
    # Iterating transitions sorted by v_idx (reverse topo), k_v, phase_v, rho_v.

    sorted_indexed_transitions = sorted(
        indexed_transitions,
        key=lambda x: (
            node_str_to_topo_rank.get(idx_to_node.get(x[1][5]), float('-inf')), # v_idx topo rank
            x[1][6],  # k_v
            x[1][9],  # phase_v
            x[1][7]   # rho_v
        ),
        reverse=True # Process v with higher topo rank first (effectively reverse topo order)
    )

    if verbose:
        print(f"Processing {len(sorted_indexed_transitions)} state transitions in soft-min backward pass...")

    # 5. Main loop: for each transition u→v, update V(u) = softmin( V(u), V(v) + cost(u→v) ).
    for i, (original_index, trans) in enumerate(sorted_indexed_transitions):
        # Unpack based on assumed structure from forward_svi's usage:
        # trans[0]=u_idx, trans[1]=k_u, trans[2]=rho_u, trans[3]=u_alt_ft, trans[4]=phase_u
        # trans[5]=v_idx, trans[6]=k_v, trans[7]=rho_v, trans[8]=v_alt_ft, trans[9]=phase_v
        u_idx, k_u, rho_u, u_alt_ft, phase_u, \
        v_idx, k_v, rho_v, v_alt_ft, phase_v = trans

        # a) Pull the current value at v (the "successor" state in backward pass)
        V_s_v = V_val[v_idx, k_v, rho_v, phase_v]
        if torch.isinf(V_s_v):
            # If state v has an infinite cost-to-go (i.e., goal is unreachable from v), skip
            continue

        # b) Retrieve pre-computed tailwind for this transition
        tailwind_knots = avg_tailwind_knots_per_transition[original_index]

        # c) Compute cost_uv for the transition u → v
        edge_u_indices = torch.tensor([u_idx], device=device, dtype=torch.long)
        edge_v_indices = torch.tensor([v_idx], device=device, dtype=torch.long)

        cost_uv_tensor = cost_model(
            (edge_u_indices, edge_v_indices),
            distance_matrix_d,
            airspace_charge_matrix_ac,
            tailwind_knots
        )
        cost_uv = float(cost_uv_tensor.item())

        # d) Form the value flowing backward from v to u:  V(v) + cost(u→v)
        val_from_v = V_s_v + cost_uv # V_s_v is float64 tensor element, cost_uv is float

        # Store the computed cost by appending indices and value
        indices_key = (u_idx, k_u, rho_u, phase_u, v_idx, k_v, rho_v, phase_v)
        processed_cost_indices_list.append(indices_key)
        processed_cost_values_list.append(cost_uv)

        # e) Merge into V_val[u] using softmin:
        #    V(u) references phase_u (trans[4]) and rho_u (trans[2])
        old_V_u = V_val[u_idx, k_u, rho_u, phase_u]
        # softmin_gamma(a,b) = -gamma * log(exp(-a/gamma) + exp(-b/gamma))
        new_V_u = -gamma * torch.logaddexp(-old_V_u / gamma, -val_from_v / gamma)
        V_val[u_idx, k_u, rho_u, phase_u] = new_V_u

        if verbose and (i % (len(sorted_indexed_transitions)//100 + 1) == 0 or i == len(sorted_indexed_transitions)-1):
            print(
                f"\r  Bwd Transition {i+1}/{len(sorted_indexed_transitions)}: "
                f"u=({u_idx},{k_u},{rho_u},{phase_u}), V(v={v_idx},{k_v},{rho_v},{phase_v})={V_s_v:.3f}, "
                f"cost(u→v)={cost_uv:.3f}, V(v)+cost={val_from_v:.3f}  ->  "
                f"new V(u)={new_V_u:.3f}",
                end="", flush=True
            )

    # After the loop, construct the sparse edge_costs_uv tensor
    if processed_cost_indices_list:
        # Convert lists to tensors
        # Indices need to be shape (num_dimensions, num_elements)
        indices_tensor = torch.tensor(processed_cost_indices_list, dtype=torch.long, device=device).t()
        values_tensor = torch.tensor(processed_cost_values_list, dtype=torch.float64, device=device)
        edge_costs_uv = torch.sparse_coo_tensor(
            indices_tensor,
            values_tensor,
            full_shape_8d,
            dtype=torch.float64,
            device=device
        ).coalesce() # coalesce sums duplicates (if any) and sorts indices
    else:
        # No transitions processed, create an empty sparse tensor
        indices_tensor = torch.empty((8, 0), dtype=torch.long, device=device) # 8 dimensions
        values_tensor = torch.empty((0,), dtype=torch.float64, device=device)
        edge_costs_uv = torch.sparse_coo_tensor(
            indices_tensor,
            values_tensor,
            full_shape_8d,
            dtype=torch.float64,
            device=device
        )

    # 6. Convert back: V_soft(s) = -L_val(s). Represents soft cost-to-go.
    V_soft_cost_to_go = V_val

    if verbose:
        num_finite = torch.isfinite(V_soft_cost_to_go).sum().item()
        total_states = V_soft_cost_to_go.numel()
        print(f"\nBackward SVI (soft-min) complete. {num_finite}/{total_states} states have finite V(s) (cost-to-go).")

    return V_soft_cost_to_go, edge_costs_uv

# Example usage structure (for testing, adapt from forward_svi if needed)
if __name__ == '__main__':
    print("backward_svi_log.py main block reached - for testing, actual usage is by importing the function.")
    # To test this, you would need to set up:
    # G, idx_to_node, goal_node_idx, cost_model, num_*, distance_matrix_d, etc.
    # state_transitions (careful with tuple element order)
    # device = torch.device("cpu")

    # Example:
    # If goal_node_idx = 2, k=2, rho=1, phase=0 is a goal state (V=0).
    # If trans (1,1,1,0) -> (2,2,1,0) with cost 1.
    # V(1,1,1,0) = softmin(inf, V(2,2,1,0) + 1) = softmin(inf, 0 + 1) = 1.
    pass

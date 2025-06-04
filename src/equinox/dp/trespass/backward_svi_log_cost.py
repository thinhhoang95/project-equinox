import torch
import math

from equinox.route.get_wind import get_wind
# For type hinting, actual instances are passed as arguments
from equinox.cost.cost_rev1 import CostRev1
from equinox.wind.wind_model import WindModel
import networkx as nx
# Conversion factor from meters per second to knots
MPS_TO_KNOTS = 1.94384

def backward_soft_value_iteration(
    state_transitions: list[tuple[int, int, int, float, int, int, int, float, int, int]], # Note: types for alt and phase might be swapped in usage
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
    cost_model: CostRev1,
    num_nodes: int,
    num_time_bins_wall_clock: int,
    num_rho_bins: int,
    num_phases: int,
    distance_matrix_d: torch.Tensor,
    airspace_charge_matrix_ac: torch.Tensor,
    wind_model: WindModel,
    min_wall_clock_time_sec: float,
    delta_t_wall_clock_sec: float,
    device: torch.device,
    verbose: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the soft backward value function (cost-to-go) V(s) via log-space soft value iteration.
    It also returns a tensor containing the costs of all processed edges.

    This function implements a numerically stable version of backward soft value iteration
    using log-space computations. It calculates the soft optimal cost-to-go from each state
    to a designated goal state.

    The algorithm maintains L(s) := log Z(s), where Z(s) represents the partition function
    for paths from state s to the goal. For backward SVI,
    Z(s) = sum_{s→v} exp(-cost(s→v)) * Z(v).
    The final soft value function (cost-to-go) is V(s) = -L(s).

    ## Parameters

    - **state_transitions** (`list[tuple[int, int, int, float, int, int, int, float, int, int]]`):
      List of state transitions. Based on usage in the forward pass, we assume the tuple elements are:
      `(u_idx, k_u, rho_u, u_alt_ft, phase_u, v_idx, k_v, rho_v, v_alt_ft, phase_v)`.
      `u_alt_ft` is altitude at u (used for wind), `phase_u` is phase at u (used for L_val indexing).
      Similarly for `v_alt_ft` and `phase_v`.
      The types in the signature `(..., float, int, ..., float, int)` reflect this `alt, phase` order.

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
    - **wind_model** (`WindModel`): Wind model instance.
    - **min_wall_clock_time_sec** (`float`): Absolute start time for k_idx=0.
    - **delta_t_wall_clock_sec** (`float`): Duration of each wall-clock time bin.
    - **device** (`torch.device`): PyTorch device.
    - **verbose** (`bool`, optional): If True, prints progress. Defaults to False.

    ## Returns

    **tuple[torch.Tensor, torch.Tensor]**:
    - **V_soft_cost_to_go** (`torch.Tensor`): Soft value function V (cost-to-go) with shape `(num_nodes,
      num_time_bins_wall_clock, num_rho_bins, num_phases)`. Values represent -log Z(s),
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

    - Computations are in float64 log-space.
    - L_val is initialized to -∞. For actual goal states (g_node, k, rho, phase)
      that are destinations of transitions, L(g_state) = log(1) = 0.
    - Transitions `u→v` are processed in an order ensuring L(v) is finalized before updating L(u).
      This typically means iterating nodes in reverse topological order.
    - For each transition u→v: compute a_v := L(v) - cost(u→v).
    - Update: L(u) ← logaddexp(old L(u), a_v).
    - Final result: V(s) = -L(s).
    """

    # 0. Extract node coordinates
    node_coords_deg = torch.zeros((num_nodes, 2), dtype=torch.float32, device=device)
    for i in range(num_nodes):
        if i not in idx_to_node:
            if verbose:
                print(f"Warning: Index {i} not in idx_to_node. Cannot fetch coordinates.")
            continue
        node_name_str = idx_to_node[i]
        if node_name_str not in G.nodes:
            raise ValueError(f"Node name \'{node_name_str}\' (for index {i}) not found in graph G.")
        node_data = G.nodes[node_name_str]
        try:
            lat = float(node_data['lat'])
            lon = float(node_data['lon'])
        except KeyError as e:
            raise ValueError(f"Node \'{node_name_str}\' in G is missing \'lat\' or \'lon\': {e}")
        except ValueError as e:
            raise ValueError(f"Could not convert lat/lon for node \'{node_name_str}\': {e}")
        node_coords_deg[i, 0] = lat
        node_coords_deg[i, 1] = lon

    # 1. Create L_val and fill with -∞
    L_val = torch.full(
        (num_nodes, num_time_bins_wall_clock, num_rho_bins, num_phases),
        fill_value=float('-inf'),
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

    # 3. Initialize L_val for actual_goal_states to 0.0 (log(1))
    if not actual_goal_states:
        if verbose:
            print(
                f"Warning: No transitions found ending at goal_node_idx {goal_node_idx}. "
                "All L_val entries will remain -inf."
            )
    else:
        for (k_g, rho_g, phase_g) in actual_goal_states:
            # Ensure indices are within bounds before assignment
            if (0 <= k_g < num_time_bins_wall_clock and
                0 <= rho_g < num_rho_bins and
                0 <= phase_g < num_phases):
                L_val[goal_node_idx, k_g, rho_g, phase_g] = 0.0
                if verbose:
                    print(
                        f"Initialized goal state: "
                        f"L[{goal_node_idx},{k_g},{rho_g},{phase_g}] = 0.0"
                    )
            elif verbose: # Print warning if an identified goal state is out of bounds for L_val
                 print(
                    f"Warning: Identified goal state ({goal_node_idx},{k_g},{rho_g},{phase_g}) "
                    f"is out of bounds for L_val dimensions "
                    f"({num_time_bins_wall_clock},{num_rho_bins},{num_phases}). Skipping initialization."
                )

    # 4. Sort transitions so that when we visit (u→v) to update L(u), L(v) is already finalized.
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
    
    # Create a reverse map for topological rank (higher value for earlier nodes in topo sort)
    # No, we need lower value for nodes that are "later" in topological sort (closer to typical sinks)
    # so that when sorted ascending, these appear first, and their predecessors later.
    # For backward pass, we process states v in reverse topological order.
    # This means states "closer" to the goal (higher topological rank) are processed first.
    # Their L(v) values are used to update L(u) for u -> v.
    # So, we need transitions sorted such that L(v) is known.
    # Iterating transitions sorted by v_idx (reverse topo), k_v, phase_v, rho_v.

    sorted_transitions = sorted(
        state_transitions,
        key=lambda x: (
            node_str_to_topo_rank.get(idx_to_node.get(x[5]), float('-inf')), # v_idx topo rank
            x[6],  # k_v
            x[9],  # phase_v
            x[7]   # rho_v
        ),
        reverse=True # Process v with higher topo rank first (effectively reverse topo order)
    )

    if verbose:
        print(f"Processing {len(sorted_transitions)} state transitions in log-space (backward pass)...")

    # 5. Main loop: for each transition u→v, update L(u) = logaddexp( L(u), L(v) - cost(u→v) ).
    for i, trans in enumerate(sorted_transitions):
        # Unpack based on assumed structure from forward_svi's usage:
        # trans[0]=u_idx, trans[1]=k_u, trans[2]=rho_u, trans[3]=u_alt_ft, trans[4]=phase_u
        # trans[5]=v_idx, trans[6]=k_v, trans[7]=rho_v, trans[8]=v_alt_ft, trans[9]=phase_v
        u_idx, k_u, rho_u, u_alt_ft, phase_u, \
        v_idx, k_v, rho_v, v_alt_ft, phase_v = trans

        # a) Pull the current log‐mass at v (the "successor" state in backward pass)
        L_s_v = L_val[v_idx, k_v, rho_v, phase_v]
        if torch.isneginf(L_s_v):
            # If state v has an infinite cost-to-go (i.e., goal is unreachable from v), skip
            continue

        # b) Compute tailwind for segment u->v. Wind is experienced when flying from u.
        #    Use u's altitude (u_alt_ft = trans[3]) and u's departure time (k_u = trans[1]).
        coords_src_edge = node_coords_deg[u_idx].unsqueeze(0).to(device=device, dtype=torch.float32)
        coords_tgt_edge = node_coords_deg[v_idx].unsqueeze(0).to(device=device, dtype=torch.float32)
        altitude_for_wind_ft = torch.tensor([u_alt_ft], device=device, dtype=torch.float32) # u_alt_ft from trans[3]
        eta_src_sec_edge = min_wall_clock_time_sec + k_u * delta_t_wall_clock_sec # k_u from trans[1]
        eta_src_tensor_edge = torch.tensor([eta_src_sec_edge], device=device, dtype=torch.float32)

        tailwind_mps = get_wind(
            coords_src_edge,
            coords_tgt_edge,
            altitude_for_wind_ft,
            eta_src_tensor_edge,
            wind_model
        )
        tailwind_knots = tailwind_mps * MPS_TO_KNOTS

        # c) Compute cost_uv for the transition u → v
        edge_u_indices = torch.tensor([u_idx], device=device, dtype=torch.long)
        edge_v_indices = torch.tensor([v_idx], device=device, dtype=torch.long)

        cost_uv_tensor = cost_model(
            (edge_u_indices, edge_v_indices),
            distance_matrix_d.to(dtype=torch.float64),
            airspace_charge_matrix_ac.to(dtype=torch.float64),
            tailwind_knots.to(dtype=torch.float64)
        )
        cost_uv = float(cost_uv_tensor.item())

        # d) Form the "log‐mass flowing backward" from v to u:  a_v = L(v) - cost(u→v)
        a_v = L_s_v - cost_uv # L_s_v is float64, cost_uv is float

        # Store the computed cost by appending indices and value
        indices_key = (u_idx, k_u, rho_u, phase_u, v_idx, k_v, rho_v, phase_v)
        processed_cost_indices_list.append(indices_key)
        processed_cost_values_list.append(cost_uv)

        # e) Merge into L_val[u] using logaddexp:
        #    L(u) references phase_u (trans[4]) and rho_u (trans[2])
        old_L_u = L_val[u_idx, k_u, rho_u, phase_u]
        new_L_u = torch.logaddexp(old_L_u, torch.tensor(a_v, device=device, dtype=torch.float64)) # Ensure a_v is tensor for logaddexp
        L_val[u_idx, k_u, rho_u, phase_u] = new_L_u

        if verbose and (i % (len(sorted_transitions)//100 + 1) == 0 or i == len(sorted_transitions)-1):
            print(
                f"  Bwd Transition {i+1}/{len(sorted_transitions)}: "
                f"u=({u_idx},{k_u},{rho_u},{phase_u}), L(v={v_idx},{k_v},{rho_v},{phase_v})={L_s_v:.3f}, "
                f"cost(u→v)={cost_uv:.3f}, a_v={a_v:.3f}  ->  "
                f"new L(u)={new_L_u:.3f}"
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
    V_soft_cost_to_go = -L_val

    if verbose:
        num_finite = torch.isfinite(V_soft_cost_to_go).sum().item()
        total_states = V_soft_cost_to_go.numel()
        print(f"Backward SVI (log‐space) complete. {num_finite}/{total_states} states have finite V(s) (cost-to-go).")

    return V_soft_cost_to_go, edge_costs_uv

# Example usage structure (for testing, adapt from forward_svi if needed)
if __name__ == '__main__':
    print("backward_svi_log.py main block reached - for testing, actual usage is by importing the function.")
    # To test this, you would need to set up:
    # G, idx_to_node, goal_node_idx, cost_model, num_*, distance_matrix_d, etc.
    # state_transitions (careful with tuple element order)
    # device = torch.device("cpu")

    # Example:
    # If goal_node_idx = 2, k=2, rho=1, phase=0 is a goal state (L=0).
    # If trans (1,1,1,0) -> (2,2,1,0) with cost 1.
    # L(1,1,1,0) = logaddexp(-inf, L(2,2,1,0) - 1) = logaddexp(-inf, 0 - 1) = -1.
    # V(1,1,1,0) = 1.
    pass

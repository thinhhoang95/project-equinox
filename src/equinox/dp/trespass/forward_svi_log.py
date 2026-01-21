# *****************************************************************************
# * CAUTION: We hard-coded the initial log-mass values for the origin states. *
# * This is not a general solution and should be fixed in the future.         *
# * (line 235)                                                                *
# *****************************************************************************

import torch
import math

from equinox.route.get_wind import get_wind
# For type hinting, actual instances are passed as arguments
from equinox.wind.wind_model import WindModel
import networkx as nx

from equinox.dp.trespass.transition_utils import get_base_transition
# Conversion factor from meters per second to knots
MPS_TO_KNOTS = 1.94384

def forward_soft_value_iteration(
    state_transitions: list[tuple],
    # The tuple contains, in this exact order:
    # (u_idx, k_u_idx, rho_u_idx, u_alt_ft, phase_u,
    #  v_idx, k_v_idx, rho_v_idx, v_alt_ft, phase_v)
    G: nx.DiGraph,
    idx_to_node: dict[int, str], # Added: mapping from integer index to string node ID in G
    origin_node_idx: int,
    cost_model: torch.nn.Module,
    num_nodes: int,
    num_time_bins_wall_clock: int,
    num_rho_bins: int,
    num_phases: int,
    distance_matrix_d: torch.Tensor,
    airspace_charge_matrix_ac: torch.Tensor,
    wind_model: WindModel,
    min_wall_clock_time_sec: float,   # Absolute start time for k_idx=0 in seconds
    delta_t_wall_clock_sec: float,    # Duration of each wall-clock time bin in seconds
    device: torch.device,
    verbose: bool = False
) -> torch.Tensor:
    """
    Computes the soft forward value function V(s) via log-space soft value iteration.
    
    This function implements a numerically stable version of forward soft value iteration
    using log-space computations to avoid numerical underflow issues that can occur
    with exponential operations on large negative values.
    
    The algorithm maintains L(s) := log Z(s), where Z(s) represents the partition function
    Z(s) = sum_{u→s} exp(-cost(u→s)) * Z(u) for all transitions u→s leading to state s.
    The final soft value function is V(s) = -L(s).
    
    ## Parameters
    
    - **state_transitions** (`list[tuple[int, int, int, int, float, int, int, int, int, float]]`):
      List of state transitions, where each transition is a tuple:
      `(u_idx, k_u_idx, rho_u_idx, u_alt_ft, phase_u,
       v_idx, k_v_idx, rho_v_idx, v_alt_ft, phase_v)`
      representing a transition from state u to state v with their respective indices,
      time bins, climb time bins, phases, and altitudes.
    
    - **G** (`nx.DiGraph`): NetworkX directed graph representing the route network with node coordinates.
    
    - **idx_to_node** (`dict[int, str]`): Mapping from integer node indices to string node IDs in the graph.
    
    - **origin_node_idx** (`int`): Index of the origin node where the journey begins.
    
    - **cost_model** (`CostRev1`): Cost model instance implementing the CostRev1 interface for computing
      transition costs including fuel, time, and airspace charges.
    
    - **num_nodes** (`int`): Total number of nodes in the network.
    
    - **num_time_bins_wall_clock** (`int`): Number of wall-clock time bins for discretization.
    
    - **num_rho_bins** (`int`): Number of climb time bins for vertical profile discretization.
    
    - **num_phases** (`int`): Number of flight phases (typically 3: CLIMB, CRUISE, DESCENT).
    
    - **distance_matrix_d** (`torch.Tensor`): Tensor of pairwise distances between nodes in meters.
    
    - **airspace_charge_matrix_ac** (`torch.Tensor`): Tensor of airspace charges between node pairs.
    
    - **wind_model** (`WindModel`): Wind model instance for computing wind effects on flight segments.
    
    - **min_wall_clock_time_sec** (`float`): Absolute start time for k_idx=0 in seconds since midnight.
    
    - **delta_t_wall_clock_sec** (`float`): Duration of each wall-clock time bin in seconds.
    
    - **device** (`torch.device`): PyTorch device (CPU or CUDA) for tensor computations.
    
    - **verbose** (`bool`, optional): If True, prints detailed progress information during computation.
      Defaults to False.
    
    ## Returns
    
    **torch.Tensor**: Soft value function V with shape `(num_nodes, num_time_bins_wall_clock,
    num_rho_bins, num_phases)`. Values represent the negative log partition function,
    with +inf for unreachable states.
    
    ## Algorithm Details
    
    - All computations are performed in float64 precision and log-space for numerical stability
    - L_val is initialized to -∞ for all states except origin states which get log(1/|origins|)
    - For each transition u→v: compute a_u := L(u) - cost(u→v)
    - Update: L(v) ← logaddexp(old L(v), a_u) using numerically stable log-sum-exp
    - Final result: V(s) = -L(s), with unreachable states remaining at +∞
    
    ## Example
    
    ```python
    import torch
    import networkx as nx
    from equinox.cost.cost_model_1 import cost_model_1
    from equinox.wind.wind_free import WindFree
    
    # Load route graph and setup
    G = nx.read_gml("data/graph/LEMD_EGLL_2023_04_01.gml")
    node_to_idx = {node: i for i, node in enumerate(G.nodes())}
    idx_to_node = {i: node for i, node in enumerate(G.nodes())}
    
    # Load precomputed transitions from forward pass
    import pickle
    transitions = pickle.load(open("data/graph/transitions/LEMD_EGLL_2023_04_01_REACHABLE.pkl", "rb"))
    
    # Setup parameters
    origin_node_idx = node_to_idx["LEMD"]
    num_nodes = len(G.nodes())
    num_time_bins_wall_clock = 61  # 5 hours at 5-minute intervals
    num_rho_bins = 37  # climb time discretization
    num_phases = 3  # CLIMB, CRUISE, DESCENT
    
    # Load distance and airspace charge matrices
    import numpy as np
    dist_matrix = np.load("data/graph/LEMD_EGLL_2023_04_01_distances.npy")
    ac_matrix = np.load("data/graph/LEMD_EGLL_2023_04_01_charges.npy")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    distance_matrix_d = torch.tensor(dist_matrix, dtype=torch.float32, device=device)
    airspace_charge_matrix_ac = torch.tensor(ac_matrix, dtype=torch.float32, device=device)
    
    # Time parameters
    from equinox.helpers.datetimeh import datestr_to_seconds_since_midnight
    takeoff_time_str = "2023-04-01 10:15:00"
    min_wall_clock_time_sec = float(datestr_to_seconds_since_midnight(takeoff_time_str))
    delta_t_wall_clock_sec = 300.0  # 5 minutes
    
    # Wind model
    wind_model = WindFree()  # No wind for simplicity
    
    # Compute soft value function
    V_soft = forward_soft_value_iteration(
        state_transitions=transitions,
        G=G,
        idx_to_node=idx_to_node,
        origin_node_idx=origin_node_idx,
        cost_model=cost_model_1,
        num_nodes=num_nodes,
        num_time_bins_wall_clock=num_time_bins_wall_clock,
        num_rho_bins=num_rho_bins,
        num_phases=num_phases,
        distance_matrix_d=distance_matrix_d,
        airspace_charge_matrix_ac=airspace_charge_matrix_ac,
        wind_model=wind_model,
        min_wall_clock_time_sec=min_wall_clock_time_sec,
        delta_t_wall_clock_sec=delta_t_wall_clock_sec,
        device=device,
        verbose=True
    )
    
    print(f"V_soft shape: {V_soft.shape}")
    print(f"Finite values: {torch.isfinite(V_soft).sum().item()}")
    # Save results
    np.save("V_soft_results.npy", V_soft.cpu().numpy())
    ```
    
    ## Note
    
    This log-space implementation avoids numerical underflow that can occur when
    computing exp(-large_cost) directly, making it suitable for problems with
    large cost values or many transitions. The function requires precomputed
    state transitions from a forward reachability analysis.
    """

    # 0. Extract node coordinates from G using idx_to_node mapping
    node_coords_deg = torch.zeros((num_nodes, 2), dtype=torch.float32, device=device)
    for i in range(num_nodes):
        if i not in idx_to_node:
            # This case should ideally not happen if idx_to_node is complete for 0..num_nodes-1
            # If it can, we might need to decide how to handle missing nodes (e.g., skip, error, default coords)
            # For now, assume idx_to_node covers all relevant indices.
            if verbose:
                print(f"Warning: Index {i} not in idx_to_node. Cannot fetch coordinates for this index.")
            # Depending on strictness, could raise ValueError here or fill with NaN/default.
            # Let's assume for now that all nodes 0..num_nodes-1 are in G and idx_to_node.
            # If a node_idx appears in transitions, it MUST be in idx_to_node and G.
            # If num_nodes is just a maximum and not all indices are used, this is fine.
            continue # Skip if node index is not in the map; it implies it might not be in G under this ID.

        node_name_str = idx_to_node[i]
        if node_name_str not in G.nodes:
            raise ValueError(f"Node name '{node_name_str}' (for index {i}) not found in graph G. Nodes: {list(G.nodes)[:5]}...")

        node_data = G.nodes[node_name_str]
        try:
            lat = float(node_data['lat'])
            lon = float(node_data['lon'])
        except KeyError as e:
            raise ValueError(f"Node '{node_name_str}' in G is missing 'lat' or 'lon' attribute: {e}")
        except ValueError as e:
            raise ValueError(f"Could not convert lat/lon to float for node '{node_name_str}': {e}")

        node_coords_deg[i, 0] = lat
        node_coords_deg[i, 1] = lon

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
        u_idx, k_u, rho_u, u_alt_ft, phase_u, v_idx, k_v, rho_v, v_alt_ft, phase_v = get_base_transition(st)
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
        # original_states_probs = {39: 0.0, 40: 0.0, 41: 0.0, 42: 0.0}
        # original_states_probs = {39:-1.16657401, 40:-0.75997401, 41:-1.52747401, 42:-5.57117401} # these are log-probs, not values (values = -log-probs!)
        for (k_u, rho_u, phase_u) in actual_origin_states:
            # L_val[origin_node_idx, k_u, rho_u, phase_u] = original_states_probs[k_u]
            L_val[origin_node_idx, k_u, rho_u, phase_u] = initial_logmass
            if verbose:
                print(
                    f"Initialized origin state: "
                    # f"Log likelihood@[{origin_node_idx},{k_u},{rho_u},{phase_u}] = {original_states_probs[k_u]:.4e}"
                    f"Log likelihood@[{origin_node_idx},{k_u},{rho_u},{phase_u}] = {initial_logmass:.4e}"
                )

    # 4. Sort transitions so that when we visit (u→v), L(u) is already finalized.
    #    Primary key: topological sort order of u_idx (via its string name),
    #    then k_u (source time), then rho_u, then phase_u.
    try:
        # G contains string node IDs. topological_sort works on these directly.
        topo_order_str_nodes = list(nx.topological_sort(G))
        # node_to_topo_rank will map string node names to their rank.
        node_str_to_topo_rank = {node_name_str: rank for rank, node_name_str in enumerate(topo_order_str_nodes)}
    except nx.NetworkXUnfeasible: # Not a DAG
        raise ValueError("The graph G must be a Directed Acyclic Graph (DAG) for topological sorting.")
    except Exception as e:
        raise ValueError(f"Error during topological sort: {e}. Ensure G is a DAG.")

    sorted_transitions = sorted(
        state_transitions,
        key=lambda x: (
            node_str_to_topo_rank.get(idx_to_node.get(x[0]), float('inf')),
            x[1], # k_u
            x[4], # phase_u
            x[2]  # rho_u
        )
        # x[0] is u_idx (integer). idx_to_node[x[0]] gives string name.
        # .get on idx_to_node for safety, though u_idx should always be in it if transitions are valid.
        # .get on node_str_to_topo_rank for safety, though all nodes from valid transitions should be in G.
    )

    if verbose:
        print(f"Processing {len(sorted_transitions)} state transitions in log-space...")

    # 5. Main loop: for each transition, update L(v) = logaddexp( L(v),  L(u) - cost(u→v) ).
    for i, trans in enumerate(sorted_transitions):
        # Unpack exactly according to our documented order:
        u_idx, k_u, rho_u, u_alt_ft, phase_u, \
        v_idx, k_v, rho_v, v_alt_ft, phase_v = get_base_transition(trans)

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

        # d) Form the "incoming log‐mass" from u→v:  a_u = L(u) - cost_uv
        #    Since L_s_u is a torch.float64 tensor, and cost_uv is a Python float,
        #    (L_s_u - cost_uv) remains a torch.float64 scalar on `device`.
        a_u = L_s_u - cost_uv

        # e) Merge into L_val[v] using logaddexp:
        old_L_v = L_val[v_idx, k_v, rho_v, phase_v]  # current log‐mass at v (maybe -inf)
        # torch.logaddexp handles two scalars stably:
        new_L_v = torch.logaddexp(old_L_v, a_u)
        L_val[v_idx, k_v, rho_v, phase_v] = new_L_v

        if verbose and (i % (len(sorted_transitions)//100 + 1) == 0 or i == len(sorted_transitions)-1):
            # Only print if this transition's contribution is not extremely far below
            # the current L(v).  (We could check e^{ a_u - new_L_v } > threshold, etc.)
            print(f"  Transition {i+1}/{len(sorted_transitions)}:  u=({u_idx},{k_u},{rho_u},{phase_u}) L(u)={L_s_u:.3f}, cost={cost_uv:.3f},  a_u={a_u:.3f}  ->  new L(v={v_idx},{k_v},{rho_v},{phase_v})={new_L_v:.3f}", end='\r', flush=True)

    # 6. Convert back:  V_soft(s) = -L_val(s).  If L_val(s) = -inf (unreachable), V_soft(s) = +inf.
    V_soft = -L_val

    if verbose:
        num_finite = torch.isfinite(V_soft).sum().item()
        total_states = V_soft.numel()
        print(f"Forward SVI (log‐space) complete. {num_finite}/{total_states} states have finite V(s).")

    return V_soft


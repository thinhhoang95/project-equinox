import numpy as np
from collections import deque

#----------------------------------------
# Placeholder helper functions (to be implemented)
#----------------------------------------
def get_heuristic_altitude(lat_src, lon_src, lat_tgt, lon_tgt, EETA_src, phase_src):
    """
    Compute a heuristic cruise altitude for the segment from src->tgt.
    Inputs are numpy arrays of same shape:
      lat_src, lon_src, lat_tgt, lon_tgt: float arrays of coordinates
      EETA_src: expected arrival time at source node
      phase_src: integer phase at source node (0=climb,1=cruise,2=descent)
    Returns:
      alt_e: numpy array of altitudes for each edge
    """
    raise NotImplementedError


def get_wind(alt_e, EETA_src):
    """
    Query a wind model to get vector wind components for each edge.
    Inputs:
      alt_e: vector of altitudes
      EETA_src: vector of times
    Returns:
      wind_e: array of same shape, or tuple of components
    """
    raise NotImplementedError


def compute_speed(distances, wind):
    """
    Compute ground speed over each edge given distance and wind.
    """
    raise NotImplementedError


def compute_cost(speed, wind):
    """
    Compute cost c_{j->i} for each edge given speed and wind.
    """
    raise NotImplementedError


def heuristic_altitude_from_EETA(EETA):
    """
    Simple mapping from expected arrival time to desired altitude.
    """
    raise NotImplementedError


def infer_phase(EETA):
    """
    Infer flight phase (climb, cruise, descent) from expected arrival times.
    Returns integer array same shape as EETA.
    """
    raise NotImplementedError


#----------------------------------------
# Graph layering / topological decomposition
#----------------------------------------
def compute_layers(src, tgt, num_nodes):
    """
    Partition nodes into topological layers: each layer contains nodes
    whose predecessors have all appeared in earlier layers.

    Args:
      src: 1D array of source node indices for each edge
      tgt: 1D array of target node indices for each edge
      num_nodes: total number of nodes N

    Returns:
      layers: list of lists of node indices, in topological order
    """
    # Compute in-degrees
    indegree = np.zeros(num_nodes, dtype=int)
    for v in tgt:
        indegree[v] += 1

    # Build adjacency (successors)
    succ = [[] for _ in range(num_nodes)]
    for u, v in zip(src, tgt):
        succ[u].append(v)

    # Kahn's algorithm
    q = deque([i for i in range(num_nodes) if indegree[i] == 0])
    layers = []

    while q:
        layer = list(q)
        layers.append(layer)
        next_q = deque()
        while q:
            u = q.popleft()
            for v in succ[u]:
                indegree[v] -= 1
                if indegree[v] == 0:
                    next_q.append(v)
        q = next_q

    return layers


#----------------------------------------
# Forward pass through DAG (vectorized)
#----------------------------------------
def forward_pass(
    src,                # array of shape (E,) of source node indices
    tgt,                # array of shape (E,) of target node indices
    distances,          # array of shape (E,) of link distances
    lats, lons,         # arrays of shape (N,) of node coordinates
    CTOT,               # scalar: takeoff time at source
    climb_phase = 0     # integer code for 'climb'
):
    """
    Perform one forward pass on DAG to compute:
      - V[i]    = -log partition function for node i
      - EETA[i] = expected arrival time at node i
      - phase[i]
      - alt[i]

    Uses per-layer vectorized aggregation via numpy.

    Returns:
      V, EETA, phase, alt  (each numpy array of length N)
    """
    num_edges = src.shape[0]
    num_nodes = lats.shape[0]

    # Initialize state arrays
    V    = np.full(num_nodes, np.inf)
    EETA = np.zeros(num_nodes)
    phase = np.zeros(num_nodes, dtype=int)
    alt   = np.zeros(num_nodes)

    # Topological layering
    layers = compute_layers(src, tgt, num_nodes)

    # Source is assumed the only node in first layer
    source = layers[0][0]
    V[source]    = 0.0
    EETA[source] = CTOT
    phase[source] = climb_phase
    alt[source]   = heuristic_altitude_from_EETA(np.array([CTOT]))[0]

    # Process each subsequent layer
    for layer in layers[1:]:
        mask = np.isin(tgt, layer)

        src_e = src[mask]
        tgt_e = tgt[mask]
        D_e   = distances[mask]

        # Gather current-state per-edge
        V_src    = V[src_e]
        EETA_src = EETA[src_e]
        phase_src= phase[src_e]
        lat_src  = lats[src_e]; lon_src = lons[src_e]
        lat_tgt  = lats[tgt_e]; lon_tgt = lons[tgt_e]

        # Heuristic altitude + wind + speeds + costs
        alt_e  = get_heuristic_altitude(lat_src, lon_src, lat_tgt, lon_tgt,
                                        EETA_src, phase_src)
        wind_e = get_wind(alt_e, EETA_src)
        speed_e= compute_speed(D_e, wind_e)
        time_e = D_e / speed_e
        cost_e = compute_cost(speed_e, wind_e)

        # Compute M = -cost - V_src
        M_e = -cost_e - V_src

        # Segment log-sum-exp to update V for this layer
        # 1) compute per-node maximum
        max_M = np.full(num_nodes, -np.inf)
        np.maximum.at(max_M, tgt_e, M_e)
        # 2) accumulate exp(M - max)
        sum_exp = np.zeros(num_nodes)
        exp_term = np.exp(M_e - max_M[tgt_e])
        np.add.at(sum_exp, tgt_e, exp_term)
        # 3) finalize V
        V_new = -(np.log(sum_exp) + max_M)
        V[layer] = V_new[layer]

        # Compute edge-posterior weights P_e
        P_e = np.exp(M_e + V[tgt_e])

        # Update EETA via weighted sum
        EETA_tmp = np.zeros(num_nodes)
        X_e = P_e * (EETA_src + time_e)
        np.add.at(EETA_tmp, tgt_e, X_e)
        EETA[layer] = EETA_tmp[layer]

        # Recompute altitude & phase for this layer
        alt[layer]   = heuristic_altitude_from_EETA(EETA[layer])
        phase[layer] = infer_phase(EETA[layer])

    return V, EETA, phase, alt

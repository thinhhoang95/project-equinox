"""
map_match_route.py

Given
  • a NetworkX DiGraph `Gwf` (wire-frame airways, edges weighted with length_nm)
  • a list of raw waypoints `obs_points` as (lat, lon) pairs in the original order

return
  • best_path_nodes: list of node-ids in Gwf
  • best_path_edges: flattened edge sequence (useful for visualisation or saving)
 
The algorithm is a classic Hidden-Markov-Model map-matcher (Newson & Krumm 2009):
  1.  Build candidate sets: k nearest graph nodes for each observation.
  2.  Compute log-likelihoods for emissions (distance error) and transitions
      (shortest‐path length vs. chord distance).
  3.  Run Viterbi to obtain the most likely node sequence.
  4.  Unroll to a concrete edge path with NetworkX.
"""

from __future__ import annotations
from typing import List, Dict, Tuple
import math
import heapq

import networkx as nx
import pandas as pd
from sklearn.neighbors import KDTree
from tqdm import tqdm


# ---------------------------------------------------------------------------
# helper: great-circle distance  (nautical miles)
# ---------------------------------------------------------------------------
def haversine_nm(lat1, lon1, lat2, lon2) -> float:
    R_NM = 3440.065  # radius of earth in nautical miles
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1))
         * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return 2 * R_NM * math.asin(math.sqrt(a))


# ---------------------------------------------------------------------------
# 0. build a KD-Tree for fast nearest-neighbour search on the graph
# ---------------------------------------------------------------------------
def build_kdtree(G: nx.Graph):
    coords = [(G.nodes[n]["lat"], G.nodes[n]["lon"]) for n in G.nodes]
    return KDTree([[lat, lon] for lat, lon in coords]), list(G.nodes)


# ---------------------------------------------------------------------------
# 1. candidate generation (k nearest)
# ---------------------------------------------------------------------------
def candidate_sets(obs_pts: List[Tuple[float, float]],
                   G: nx.Graph,
                   k: int = 5):
    tree, idx2node = build_kdtree(G)
    cand_nodes = []
    cand_dists = []

    for lat, lon in obs_pts:
        dists, idx = tree.query([[lat, lon]], k=k)
        nodes = [idx2node[i] for i in idx[0]]
        cand_nodes.append(nodes)
        cand_dists.append(dists[0])   # kilometres here, we'll convert later
    # convert KDTree's default metres to NM
    cand_dists = [[d / 1852.0 for d in row] for row in cand_dists]
    return cand_nodes, cand_dists


# ---------------------------------------------------------------------------
# 2. build emission and transition scores (log-probs)
# ---------------------------------------------------------------------------
def log_emission(distance_nm: float, sigma: float = 0.3) -> float:
    """
    Gaussian error model, σ in NM.
    """
    return -0.5 * (distance_nm / sigma) ** 2 - math.log(sigma * math.sqrt(2 * math.pi))


def log_transition(G: nx.Graph,
                   n_from: str,
                   n_to: str,
                   obs_gap_nm: float,
                   beta: float = 1.5) -> float:
    """
    Transition likelihood from node_i to node_j given that the observed
    waypoints are obs_gap_nm NM apart.
    Following Newson & Krumm:
       p ~ exp( - | (network_len - obs_len) | / β )
    """
    try:
        network_len = nx.shortest_path_length(G, n_from, n_to, weight="length_nm")
    except nx.NetworkXNoPath:
        return -math.inf   # impossible transition

    diff = abs(network_len - obs_gap_nm)
    return -diff / beta


# ---------------------------------------------------------------------------
# 3. Viterbi dynamic programming
# ---------------------------------------------------------------------------
def viterbi_match(G: nx.Graph,
                  obs_pts: List[Tuple[float, float]],
                  k: int = 15,
                  sigma: float = 0.3,
                  beta: float = 1.5):
    cand_nodes, cand_dists = candidate_sets(obs_pts, G, k)

    # pre-compute great-circle gaps between consecutive observations
    obs_gaps = [haversine_nm(*obs_pts[i],
                             *obs_pts[i + 1])
                for i in range(len(obs_pts) - 1)]

    # DP tables
    logdelta: List[Dict[str, float]] = []
    backptr: List[Dict[str, str]] = []

    # t = 0
    first_scores = {n: log_emission(dist, sigma=sigma)
                    for n, dist in zip(cand_nodes[0], cand_dists[0])}
    logdelta.append(first_scores)
    backptr.append({})

    # t ≥ 1
    for t in range(1, len(obs_pts)):
        step_scores: Dict[str, float] = {}
        step_back: Dict[str, str] = {}

        obs_gap = obs_gaps[t - 1]
        for j, n_j in enumerate(cand_nodes[t]):
            best_prev = -math.inf
            best_state = None

            for n_i, prev_score in logdelta[t - 1].items():
                tr_score = log_transition(G, n_i, n_j, obs_gap, beta=beta)
                if tr_score == -math.inf:
                    continue
                score = prev_score + tr_score

                if score > best_prev:
                    best_prev = score
                    best_state = n_i

            if best_state is None:
                continue  # unreachable

            step_scores[n_j] = best_prev + log_emission(cand_dists[t][j], sigma=sigma)
            step_back[n_j] = best_state

        logdelta.append(step_scores)
        backptr.append(step_back)

    # Termination
    if not logdelta[-1]:
        raise RuntimeError("No feasible path after Viterbi!")

    last_state = max(logdelta[-1], key=logdelta[-1].get)

    # Back-trace
    best_path_nodes = [last_state]
    for t in reversed(range(1, len(obs_pts))):
        last_state = backptr[t][last_state]
        best_path_nodes.append(last_state)

    best_path_nodes.reverse()

    # Expand to concrete edge sequence
    if not best_path_nodes:
        return [], []

    # Start with the first node from the Viterbi result.
    full_path_nodes = [best_path_nodes[0]]
    for u, v in zip(best_path_nodes, best_path_nodes[1:]):
        # Find path between consecutive Viterbi nodes and append all but the first node.
        path_segment = nx.shortest_path(G, u, v, weight="length_nm", method="dijkstra")
        full_path_nodes.extend(path_segment[1:])

    # Collapse the full node path into edge tuples.
    edges_flat = [(full_path_nodes[i], full_path_nodes[i + 1])
                  for i in range(len(full_path_nodes) - 1)]

    return best_path_nodes, edges_flat


# ---------------------------------------------------------------------------
# Example glue
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # -- 0. read the graphs and the CSV --------------------------
    # Your Gwf should already contain attributes:
    #   node:  lat, lon
    #   edge:  length_nm  (pre-computed great-circle or airway length)

    print('Loading the wireframe graph...')
    Gwf: nx.Graph = nx.read_gml("data/cases/LEMD_EGLL/graphs/routes.gml")

    print('Adding the length_nm attribute to the graph...')
    # Adding the length_nm attribute to the graph using the great-circle distance
    for u, v in Gwf.edges():
        lat1, lon1 = Gwf.nodes[u]["lat"], Gwf.nodes[u]["lon"]
        lat2, lon2 = Gwf.nodes[v]["lat"], Gwf.nodes[v]["lon"]
        Gwf.edges[u, v]["length_nm"] = haversine_nm(lat1, lon1, lat2, lon2)

    
    print('Loading the CSV...')
    df = pd.read_csv("data/cases/LEMD_EGLL/all_routes.csv")
    
    sculpted_routes = []

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing flights"):
        # Assuming flight_id exists in the dataframe, if not, generate one.
        flight_id = row.get("flight_id", f"flight_{index}")
        
        real_waypoints = row.get("real_waypoints", "")
        if not real_waypoints or not isinstance(real_waypoints, str):
            print(f"WARNING: Flight {flight_id} - skipping, 'real_waypoints' is missing or not a string.")
            continue
            
        orig_wp_names = real_waypoints.split()
        
        # Convert names → lat/lon using Gwf attributes
        obs_pts = [(Gwf.nodes[n]["lat"], Gwf.nodes[n]["lon"])
                   for n in orig_wp_names
                   if n in Gwf.nodes]

        if len(obs_pts) < 2:
            print(f"WARNING: Flight {flight_id} - skipping, not enough waypoints found in graph ({len(obs_pts)}).")
            continue

        try:
            # -- 1. run the matcher -------------------------------------
            best_nodes, best_edges = viterbi_match(Gwf, obs_pts, k=6)

            # -- 2. store the result ---------------------------------------
            if not best_edges:
                print(f"WARNING: Flight {flight_id} - no edges found in sculpted route, skipping.")
                continue
                
            full_nodes = [best_edges[0][0]] + [edge[1] for edge in best_edges]
            sculpted_route_str = " ".join(full_nodes)

            sculpted_routes.append({
                "flight_id": flight_id,
                "route": sculpted_route_str,
                "takeoff_time": row.get("takeoff"),
                "landing_time": row.get("landing"),
                "cruise_altitude": max(map(float, row.get("alts", "0").split())) if row.get("alts") and isinstance(row.get("alts"), str) and row.get("alts").strip() else None,
            })

        except RuntimeError as e:
            print(f"ERROR: Flight {flight_id} - Viterbi matching failed: {e}")
        except Exception as e:
            print(f"ERROR: Flight {flight_id} - unexpected error: {e}")

    # -- 3. save to new CSV ---------------------------------------
    if sculpted_routes:
        output_df = pd.DataFrame(sculpted_routes)
        output_path = "data/cases/LEMD_EGLL/all_routes_sculpted.csv"
        output_df.to_csv(output_path, index=False)
        print(f"\nSaved {len(output_df)} sculpted routes to {output_path}")
    else:
        print("\nNo routes were processed successfully.")

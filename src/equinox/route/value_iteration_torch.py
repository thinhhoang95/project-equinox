from typing import List, Tuple
import torch
from torch_scatter import scatter_logsumexp, scatter_add
from equinox.helpers.haversine import haversinet
from equinox.wind.wind_model import WindModel
from equinox.route.forward_state import get_wind
import math
from datetime import timedelta
import numpy

# Phase identifiers
CLIMB, CRUISE, DESCENT = 0, 1, 2


def compute_speed(distance: torch.Tensor, wind: torch.Tensor) -> torch.Tensor:
    """
    distance: [E]
    wind: [E]
    returns: [E] groundspeed
    """
    # Placeholder: true airspeed = 450 knots
    return torch.full_like(distance, 450.0) + wind


def infer_phase(
    eta: torch.Tensor, climb_performance: List[Tuple[float, float]]
) -> torch.Tensor:
    """
    eta: [N] expected arrival time
    returns: [N] phase id
    """
    # Placeholder: all cruise
    return torch.full_like(eta, CRUISE, dtype=torch.long)


class CostModel(torch.nn.Module):
    """
    A learnable cost model c(features; theta).
    For demonstration, we use a linear model on:
       [distance, speed, wind]
    """

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: [E, 3]
        # returns: [E] cost
        return self.linear(features).squeeze(-1)


def vectorized_forward_pass(
    edge_index: torch.LongTensor,
    edge_dist: torch.Tensor,
    node_coords: torch.Tensor,
    topo_layers: list,
    CTOT: float,
    cost_model: torch.nn.Module,
    wind_model: WindModel,
    climb_performance: List[Tuple[float, float]],
):
    """
    edge_index: [2, E] tensor of (src, tgt)
    edge_dist: [E] distances
    node_coords: [N, 2] (lat, lon)
    topo_layers: list of lists, each sublist is a set of node indices in topological order
    CTOT: scalar, take-off time
    cost_model: learnable cost function

    returns: V [N], EETA [N], phase [N]
    """
    device = edge_index.device
    N = node_coords.size(0)  # Num of nodes
    E = edge_dist.size(0)  # Num of edges

    # Initialize state [V_TAS (kts), EETA (seconds), phase = [0, 1, 2], alt (feet)
    V = torch.full((N,), float("inf"), dtype=torch.double, device=device)
    EETA = torch.zeros((N,), dtype=torch.double, device=device)
    phase = torch.zeros((N,), dtype=torch.long, device=device)
    alt = torch.zeros((N,), dtype=torch.double, device=device)

    # Source is assumed layer 0, single node
    s = topo_layers[0][0]
    V[s] = 0.0
    EETA[s] = CTOT
    phase[s] = CLIMB
    alt[s] = 0.0  # TODO: Update with the departure airport elevation (ft)

    # Pre-split edge_index
    src = edge_index[0]
    tgt = edge_index[1]

    for layer in topo_layers[1:]:
        layer_idx = torch.tensor(layer, dtype=torch.long, device=device)
        # Select edges entering this layer
        mask = torch.isin(tgt, layer_idx)
        src_e = src[mask]
        tgt_e = tgt[mask]
        dist_e = edge_dist[mask]

        # Gather source state
        EETA_src = EETA[src_e]
        V_src = V[src_e]
        phase_src = phase[src_e]
        coords_src = node_coords[src_e]
        coords_tgt = node_coords[tgt_e]
        alts_src = alt[src_e]

        # Get the wind at the source nodes
        wind_s = get_wind(coords_src, coords_tgt, alts_src, EETA_src, wind_model)

        # Altitude & wind, heuristically estimated for the targets
        alt_e = get_heuristic_altitude(
            coords_src,
            alts_src,
            EETA_src,
            phase_src,
            wind_s,
            V_src,
            coords_tgt,
            climb_performance,
        )
        wind_e = get_wind(coords_src, coords_tgt, alt_e, EETA_src, wind_model)

        # Speed, time, cost
        speed_e = compute_speed(dist_e, wind_e)
        time_e = dist_e / speed_e
        feats = torch.stack([dist_e, speed_e, wind_e], dim=1)
        cost_e = cost_model(feats)

        # Log-sum-exp for V
        M_e = -cost_e - V_src
        V_new = -scatter_logsumexp(M_e, tgt_e, dim=0, dim_size=N)

        # Expected ETA
        P_e = torch.exp(M_e + V_new[tgt_e])
        EETA_new = scatter_add(P_e * (EETA_src + time_e), tgt_e, dim=0, dim_size=N)

        # Update only this layer
        V[layer_idx] = V_new[layer_idx]
        EETA[layer_idx] = EETA_new[layer_idx]
        phase[layer_idx] = infer_phase(EETA[layer_idx])

    return V, EETA, phase


if __name__ == "__main__":
    # Example usage
    # N nodes, E edges
    N, E = 5, 7
    edge_index = torch.tensor(
        [[0, 0, 1, 1, 2, 2, 3], [1, 2, 2, 3, 3, 4, 4]], dtype=torch.long  # src  # tgt
    )
    edge_dist = torch.rand(E).double() * 500.0
    node_coords = torch.rand(N, 2).double() * 180 - 90
    topo_layers = [[0], [1, 2], [3], [4]]
    CTOT = 0.0
    cost_model = CostModel().double()

    V, EETA, phase = vectorized_forward_pass(
        edge_index, edge_dist, node_coords, topo_layers, CTOT, cost_model
    )
    print("V:", V)
    print("EETA:", EETA)
    print("phase:", phase)

import pytest


try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


pytestmark = pytest.mark.skipif(torch is None, reason="torch is required for feature-matrix tests")


if torch is not None:
    from src.equinox.preferences.disentanglement import build_feature_matrix


def test_build_feature_matrix_uses_time_and_tailwind_matrix() -> None:
    device = torch.device("cpu")
    dtype = torch.float64
    dist = torch.tensor([[0.0, 60.0], [120.0, 0.0]], dtype=dtype)
    ac = torch.tensor([[0.0, 2.0], [3.0, 0.0]], dtype=dtype)
    tailwind = torch.tensor([[0.0, 10.0], [-5.0, 0.0]], dtype=dtype)

    edge_u = torch.tensor([0, 1], dtype=torch.long)
    edge_v = torch.tensor([1, 0], dtype=torch.long)
    cruise_speed_kts = 100.0

    X = build_feature_matrix(
        edge_u,
        edge_v,
        dist,
        ac,
        cruise_speed_kts=cruise_speed_kts,
        tailwind_values_w=tailwind,
        device=device,
        dtype=dtype,
    )

    dist_e = dist[edge_u, edge_v]
    ac_e = ac[edge_u, edge_v]
    ac_dist = ac_e * dist_e / 100.0
    time_e = dist_e / (60.0 * (cruise_speed_kts + tailwind[edge_u, edge_v]))
    expected = torch.stack([torch.ones_like(dist_e), ac_dist, time_e], dim=1)

    assert torch.allclose(X, expected)


def test_build_feature_matrix_defaults_tailwind_to_zero() -> None:
    device = torch.device("cpu")
    dtype = torch.float64
    dist = torch.tensor([[0.0, 60.0], [120.0, 0.0]], dtype=dtype)
    ac = torch.tensor([[0.0, 2.0], [3.0, 0.0]], dtype=dtype)

    edge_u = torch.tensor([0, 1], dtype=torch.long)
    edge_v = torch.tensor([1, 0], dtype=torch.long)
    cruise_speed_kts = 100.0

    X = build_feature_matrix(
        edge_u,
        edge_v,
        dist,
        ac,
        cruise_speed_kts=cruise_speed_kts,
        device=device,
        dtype=dtype,
    )

    dist_e = dist[edge_u, edge_v]
    ac_e = ac[edge_u, edge_v]
    ac_dist = ac_e * dist_e / 100.0
    time_e = dist_e / (60.0 * cruise_speed_kts)
    expected = torch.stack([torch.ones_like(dist_e), ac_dist, time_e], dim=1)

    assert torch.allclose(X, expected)


def test_build_feature_matrix_accepts_tailwind_vector() -> None:
    device = torch.device("cpu")
    dtype = torch.float64
    dist = torch.tensor([[0.0, 60.0], [120.0, 0.0]], dtype=dtype)
    ac = torch.tensor([[0.0, 2.0], [3.0, 0.0]], dtype=dtype)
    tailwind_e = torch.tensor([10.0, -5.0], dtype=dtype)

    edge_u = torch.tensor([0, 1], dtype=torch.long)
    edge_v = torch.tensor([1, 0], dtype=torch.long)
    cruise_speed_kts = 100.0

    X = build_feature_matrix(
        edge_u,
        edge_v,
        dist,
        ac,
        cruise_speed_kts=cruise_speed_kts,
        tailwind_values_w=tailwind_e,
        device=device,
        dtype=dtype,
    )

    dist_e = dist[edge_u, edge_v]
    ac_e = ac[edge_u, edge_v]
    ac_dist = ac_e * dist_e / 100.0
    time_e = dist_e / (60.0 * (cruise_speed_kts + tailwind_e))
    expected = torch.stack([torch.ones_like(dist_e), ac_dist, time_e], dim=1)

    assert torch.allclose(X, expected)

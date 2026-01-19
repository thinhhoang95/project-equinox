from __future__ import annotations

import math
from datetime import datetime

import pytest


try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


pytestmark = pytest.mark.skipif(torch is None, reason="torch is required for wind tests")


if torch is not None:
    from src.equinox.wind.wind_model import WindModel


class _FakeWind:
    def __init__(self, u_values_mps: "torch.Tensor", v_values_mps: "torch.Tensor") -> None:
        self._time_min = datetime(2024, 1, 1, 0, 0, 0)
        self._u_values_mps = u_values_mps
        self._v_values_mps = v_values_mps

    def get_wind_components_batched(
        self,
        lats_pt: "torch.Tensor",
        lons_pt: "torch.Tensor",
        alts_ft_pt: "torch.Tensor",
        etas_sec_pt: "torch.Tensor",
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        assert lats_pt.shape == self._u_values_mps.shape
        return self._u_values_mps.to(device=lats_pt.device, dtype=lats_pt.dtype), self._v_values_mps.to(
            device=lats_pt.device, dtype=lats_pt.dtype
        )


def test_average_tailwind_all_nan_stays_nan() -> None:
    device = torch.device("cpu")
    dtype = torch.float32
    num_integration_steps = 3
    fake = _FakeWind(
        u_values_mps=torch.full((num_integration_steps,), float("nan")),
        v_values_mps=torch.full((num_integration_steps,), float("nan")),
    )

    transitions = [(0, 0, 0, 35000.0, 0, 1, 1, 0, 35000.0, 0)]
    node_coords_deg = torch.tensor([[0.0, 0.0], [0.0, 1.0]], device=device, dtype=dtype)

    out = WindModel.get_average_tailwind_on_edges_knots(
        fake,
        transitions=transitions,
        node_coords_deg=node_coords_deg,
        min_wall_clock_time_sec=0.0,
        delta_t_wall_clock_sec=600.0,
        num_integration_steps=num_integration_steps,
    )

    assert out.shape == (1,)
    assert math.isnan(float(out.item()))


def test_average_tailwind_ignores_partial_nan_samples() -> None:
    device = torch.device("cpu")
    dtype = torch.float32
    num_integration_steps = 3

    # Eastbound track (0,0) -> (0,1): e_track≈1, n_track≈0, so tailwind ~= u_component.
    fake = _FakeWind(
        u_values_mps=torch.tensor([1.0, float("nan"), 3.0], dtype=dtype),
        v_values_mps=torch.zeros((num_integration_steps,), dtype=dtype),
    )

    transitions = [(0, 0, 0, 35000.0, 0, 1, 1, 0, 35000.0, 0)]
    node_coords_deg = torch.tensor([[0.0, 0.0], [0.0, 1.0]], device=device, dtype=dtype)

    out = WindModel.get_average_tailwind_on_edges_knots(
        fake,
        transitions=transitions,
        node_coords_deg=node_coords_deg,
        min_wall_clock_time_sec=0.0,
        delta_t_wall_clock_sec=600.0,
        num_integration_steps=num_integration_steps,
    )

    expected_mps = torch.tensor([2.0], dtype=dtype)  # nanmean([1, nan, 3])
    expected_kts = expected_mps * 1.94384
    assert torch.allclose(out, expected_kts, atol=1e-5, rtol=0.0)

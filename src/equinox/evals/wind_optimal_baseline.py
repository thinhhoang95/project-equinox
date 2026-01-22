from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

State = Tuple[str, int, int, int]  # (node_id, k, rho, phase)


@dataclass(frozen=True)
class WindOptimalChoice:
    sample_index: int
    elapsed_time_seconds: float


def elapsed_time_seconds(states: Sequence[State], delta_t_seconds: float) -> float:
    if not states:
        raise ValueError("states must not be empty.")
    if delta_t_seconds is None:
        raise ValueError("delta_t_seconds must be provided.")
    delta_t_seconds = float(delta_t_seconds)
    if delta_t_seconds <= 0 or not math.isfinite(delta_t_seconds):
        raise ValueError("delta_t_seconds must be a positive finite value.")

    ks = []
    for state in states:
        try:
            k_value = float(state[1])
        except (IndexError, TypeError, ValueError) as exc:
            raise ValueError("state must include a numeric k value.") from exc
        if not math.isfinite(k_value):
            raise ValueError("state k value must be finite.")
        ks.append(k_value)

    if len(ks) == 1:
        delta_k = 0.0
    elif all(k_next >= k_prev for k_prev, k_next in zip(ks, ks[1:])):
        delta_k = ks[-1] - ks[0]
    else:
        delta_k = max(ks) - min(ks)

    if delta_k < 0:
        raise ValueError("elapsed time cannot be negative.")

    return float(delta_k * delta_t_seconds)


def choose_min_time_sample(
    sample_states: Sequence[Sequence[State]],
    delta_t_seconds: float,
) -> WindOptimalChoice:
    if not sample_states:
        raise ValueError("sample_states must not be empty.")

    best_index = None
    best_time = None
    for idx, states in enumerate(sample_states):
        try:
            elapsed = elapsed_time_seconds(states, delta_t_seconds)
        except ValueError:
            continue
        if best_time is None or elapsed < best_time:
            best_time = elapsed
            best_index = idx

    if best_index is None or best_time is None:
        raise ValueError("No valid sample states available to choose from.")

    return WindOptimalChoice(sample_index=best_index, elapsed_time_seconds=best_time)

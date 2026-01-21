from __future__ import annotations

from typing import Optional, Sequence, Tuple

BASE_TRANSITION_LEN = 10


def get_base_transition(transition: Sequence) -> Tuple:
    """Return the 10-field transition tuple used for indexing and state identity."""
    if len(transition) < BASE_TRANSITION_LEN:
        raise ValueError(
            "Transition tuple is too short to contain the base 10 fields. "
            f"Expected length >= {BASE_TRANSITION_LEN}, got {len(transition)}."
        )
    return tuple(transition[:BASE_TRANSITION_LEN])


def get_transition_times(transition: Sequence) -> Tuple[Optional[float], Optional[float]]:
    """Return (eta_u_abs_s, eta_v_abs_s) if present, otherwise (None, None)."""
    eta_u = transition[BASE_TRANSITION_LEN] if len(transition) > BASE_TRANSITION_LEN else None
    eta_v = transition[BASE_TRANSITION_LEN + 1] if len(transition) > BASE_TRANSITION_LEN + 1 else None
    return eta_u, eta_v


def parse_transition(
    transition: Sequence,
) -> Tuple[Tuple, Optional[float], Optional[float]]:
    """Return (base_10_tuple, eta_u_abs_s, eta_v_abs_s)."""
    base = get_base_transition(transition)
    eta_u, eta_v = get_transition_times(transition)
    return base, eta_u, eta_v

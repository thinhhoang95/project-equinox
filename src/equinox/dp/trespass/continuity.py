from bisect import bisect_left, bisect_right
from collections import defaultdict


def _index_current_states(current_states):
    index = defaultdict(list)
    for u_idx, k_idx, rho, phase in current_states:
        index[(u_idx, rho, phase)].append(k_idx)
    for k_list in index.values():
        k_list.sort()
    return index


def _has_k_in_range(k_list, low, high, *, low_inclusive=True, high_inclusive=True):
    if low > high:
        return False
    if low_inclusive:
        left = bisect_left(k_list, low)
    else:
        left = bisect_right(k_list, low)
    if high_inclusive:
        right = bisect_right(k_list, high)
    else:
        right = bisect_left(k_list, high)
    return left < right


def match_next_states(opts, current_states, k_tolerance_bins):
    exact = {v_state for (u_state, v_state) in opts if u_state in current_states}
    if exact:
        return exact, "exact"
    if k_tolerance_bins <= 0:
        return set(), "none"

    index = _index_current_states(current_states)
    forward = set()
    backward = set()
    for u_state, v_state in opts:
        u_idx, k_req, rho, phase = u_state
        k_list = index.get((u_idx, rho, phase))
        if not k_list:
            continue
        if _has_k_in_range(
            k_list,
            k_req - k_tolerance_bins,
            k_req,
            low_inclusive=True,
            high_inclusive=True,
        ):
            forward.add(v_state)
        elif _has_k_in_range(
            k_list,
            k_req,
            k_req + k_tolerance_bins,
            low_inclusive=False,
            high_inclusive=True,
        ):
            backward.add(v_state)

    if forward:
        return forward, "forward"
    if backward:
        return backward, "backward"
    return set(), "none"

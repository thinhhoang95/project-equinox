"""
State continuity matching for TRESPASS dynamic programming.

This module provides functionality to match states across transitions in the TRESPASS
(Time-Resolved Efficient Search for Paths Across State Space) dynamic programming
algorithm. It handles exact state matching and approximate matching with tolerance
for wall-clock time bin indices (k_idx) when exact matches are not available.

The module is used to ensure continuity of state transitions in flight trajectory
planning, where states are represented as tuples of:
    (u_idx, k_idx, rho, phase)
where:
    - u_idx: Node/waypoint index in the route graph
    - k_idx: Wall-clock time bin index
    - rho: Remaining climb time bin index
    - phase: Flight phase (0=CLIMB, 1=CRUISE, 2=DESCENT)

Usage Example:
    >>> from equinox.dp.trespass.continuity import match_next_states
    >>> 
    >>> # Define state transitions (options)
    >>> # Each option is a tuple: (u_state, v_state)
    >>> opts = [
    ...     ((0, 10, 5, 0), (1, 12, 4, 0)),  # u_state -> v_state
    ...     ((0, 11, 5, 0), (1, 13, 4, 0)),
    ...     ((0, 12, 5, 0), (1, 14, 4, 0)),
    ... ]
    >>> 
    >>> # Current states we're trying to match from
    >>> current_states = {
    ...     (0, 10, 5, 0),  # Exact match available
    ...     (0, 15, 5, 0),  # No exact match, will use tolerance
    ... }
    >>> 
    >>> # Match with tolerance of 2 bins
    >>> next_states, match_kind = match_next_states(opts, current_states, k_tolerance_bins=2)
    >>> print(next_states)
    {(1, 12, 4, 0), (1, 14, 4, 0)}
    >>> print(match_kind)
    "exact"
    >>> 
    >>> # Example with forward tolerance matching
    >>> current_states = {(0, 9, 5, 0)}  # k_idx=9, but option requires k_idx=10
    >>> next_states, match_kind = match_next_states(opts, current_states, k_tolerance_bins=2)
    >>> print(next_states)
    {(1, 12, 4, 0)}  # Matched via forward tolerance (9 is within [8, 10])
    >>> print(match_kind)
    "forward"
    >>> 
    >>> # Example with backward tolerance matching
    >>> current_states = {(0, 13, 5, 0)}  # k_idx=13, but option requires k_idx=10
    >>> next_states, match_kind = match_next_states(opts, current_states, k_tolerance_bins=2)
    >>> print(next_states)
    {(1, 12, 4, 0)}  # Matched via backward tolerance (13 is within (10, 12])
    >>> print(match_kind)
    "backward"

Input/Output Examples:

    Example 1: Exact match
    ----------------------
    Input:
        opts = [((0, 10, 5, 0), (1, 12, 4, 0))]
        current_states = {(0, 10, 5, 0)}
        k_tolerance_bins = 2
    
    Output:
        ({(1, 12, 4, 0)}, "exact")
    
    Example 2: Forward tolerance match
    ----------------------
    Input:
        opts = [((0, 10, 5, 0), (1, 12, 4, 0))]
        current_states = {(0, 9, 5, 0)}  # k_idx=9, within [8, 10] of required k_idx=10
        k_tolerance_bins = 2
    
    Output:
        ({(1, 12, 4, 0)}, "forward")
    
    Example 3: Backward tolerance match
    ----------------------
    Input:
        opts = [((0, 10, 5, 0), (1, 12, 4, 0))]
        current_states = {(0, 11, 5, 0)}  # k_idx=11, within (10, 12] of required k_idx=10
        k_tolerance_bins = 2
    
    Output:
        ({(1, 12, 4, 0)}, "backward")
    
    Example 4: No match
    ----------------------
    Input:
        opts = [((0, 10, 5, 0), (1, 12, 4, 0))]
        current_states = {(0, 20, 5, 0)}  # k_idx=20, too far from required k_idx=10
        k_tolerance_bins = 2
    
    Output:
        (set(), "none")
    
    Example 5: Multiple matches
    ----------------------
    Input:
        opts = [
            ((0, 10, 5, 0), (1, 12, 4, 0)),
            ((0, 10, 5, 0), (1, 13, 4, 0)),
            ((0, 11, 5, 0), (1, 14, 4, 0)),
        ]
        current_states = {(0, 10, 5, 0), (0, 11, 5, 0)}
        k_tolerance_bins = 1
    
    Output:
        ({(1, 12, 4, 0), (1, 13, 4, 0), (1, 14, 4, 0)}, "exact")

Matching Strategy:
    The function uses a three-tier matching strategy:
    1. Exact match: If any u_state in opts exactly matches a current_state, return
       all corresponding v_states with match_kind="exact"
    2. Forward tolerance: If no exact match, look for states where the current k_idx
       is within [k_req - tolerance, k_req] (inclusive on both ends)
    3. Backward tolerance: If no forward match, look for states where the current
       k_idx is within (k_req, k_req + tolerance] (exclusive on left, inclusive on right)
    
    Forward matches are preferred over backward matches, as they represent states
    that are slightly ahead in time, which is generally more desirable in trajectory
    planning.

Typical Use Case:
    This module is used in flight trajectory planning to ensure that state transitions
    are continuous across route segments. When checking if a route can be traversed
    through a sequence of waypoints, this function verifies that states at each waypoint
    can be matched to valid transitions to the next waypoint, with some flexibility
    for minor timing discrepancies.
"""

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

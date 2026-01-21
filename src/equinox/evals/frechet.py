from __future__ import annotations

from typing import Callable, Sequence, Tuple

Point = Tuple[float, float]
PointDistanceFn = Callable[[Point, Point], float]


def discrete_frechet(
    points_a: Sequence[Point],
    points_b: Sequence[Point],
    point_dist_fn: PointDistanceFn,
) -> float:
    if not points_a or not points_b:
        raise ValueError("Frechet distance requires non-empty point sequences.")

    n = len(points_a)
    m = len(points_b)
    ca = [[-1.0 for _ in range(m)] for _ in range(n)]

    for i in range(n):
        for j in range(m):
            dist = point_dist_fn(points_a[i], points_b[j])
            if i == 0 and j == 0:
                ca[i][j] = dist
            elif i == 0:
                ca[i][j] = max(ca[i][j - 1], dist)
            elif j == 0:
                ca[i][j] = max(ca[i - 1][j], dist)
            else:
                ca[i][j] = max(
                    min(ca[i - 1][j], ca[i - 1][j - 1], ca[i][j - 1]),
                    dist,
                )

    return float(ca[n - 1][m - 1])

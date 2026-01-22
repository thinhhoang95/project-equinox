from src.equinox.evals.wind_optimal_baseline import (
    choose_min_time_sample,
    elapsed_time_seconds,
)


def test_elapsed_time_seconds_monotone() -> None:
    states = [("A", 10, 0, 0), ("B", 12, 0, 0), ("C", 15, 0, 0)]
    assert elapsed_time_seconds(states, 600) == 3000.0


def test_elapsed_time_seconds_non_monotone_fallback() -> None:
    states = [("A", 10, 0, 0), ("B", 5, 0, 0), ("C", 12, 0, 0)]
    assert elapsed_time_seconds(states, 60) == 420.0


def test_choose_min_time_sample_prefers_shorter() -> None:
    sample_states = [
        [("A", 0, 0, 0), ("B", 50, 0, 0)],
        [("A", 0, 0, 0), ("B", 40, 0, 0)],
    ]
    choice = choose_min_time_sample(sample_states, 60)
    assert choice.sample_index == 1
    assert choice.elapsed_time_seconds == 2400.0

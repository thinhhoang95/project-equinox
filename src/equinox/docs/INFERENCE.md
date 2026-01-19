# Inference Usage

This page shows how to run inference with the lin_disent cost model using the new
`compute_4d_path` entrypoint. It loads a trained checkpoint, runs backward SVI,
samples trajectories, and optionally reconstructs 4D outputs.

## Basic usage

```python
from equinox.sampling.pipeline import compute_4d_path

res = compute_4d_path(
    case_dir="data/cases/LGAV_LFPG",
    checkpoint_path="data/cases/LGAV_LFPG/batch_sgd_results/checkpoint_iter_200.pt",
    flight_id="392AECAFR98CQ",
    takeoff_timestamp=1682781206,
    n_samples=10,
    policy="sample",
    return_4d=True,
    seed=7,
)

print(res.samples[0].route)
```

## Auto-select flight and checkpoint

If you omit `checkpoint_path`, the newest `checkpoint_iter_*.pt` under
`case_dir/batch_sgd_results` is used (falling back to `final_results.pt`).
If you omit `flight_id` and `takeoff_timestamp`, the first available flight
in `case_dir/tres_runs` is chosen.

```python
from equinox.sampling.pipeline import compute_4d_path

res = compute_4d_path(
    case_dir="data/cases/LGAV_LFPG",
    n_samples=5,
    policy="greedy",
    return_4d=False,
)
```

## Persist outputs to disk

When `output_dir` is provided, the pipeline writes:
- `routes.txt` with one route per line (`total_cost,waypoint...`)
- `metadata.pt` with run metadata
- `trajectory_*.pt` for each sample (if `return_4d=True`)

```python
from equinox.sampling.pipeline import compute_4d_path

res = compute_4d_path(
    case_dir="data/cases/LGAV_LFPG",
    checkpoint_path="data/cases/LGAV_LFPG/batch_sgd_results/checkpoint_iter_200.pt",
    n_samples=100,
    policy="sample",
    return_4d=True,
    seed=7,
    output_dir="data/cases/LGAV_LFPG/inference_outputs/sample_100",
)
```

## Caching SVI results

Backward SVI is the most expensive step. By default, results are cached under
`case_dir/inference_cache` keyed by `(checkpoint_hash, flight_id, takeoff_ts, gamma)`.
Disable caching or change its location as needed:

```python
from equinox.sampling.pipeline import compute_4d_path

res = compute_4d_path(
    case_dir="data/cases/LGAV_LFPG",
    n_samples=3,
    policy="sample",
    cache_dir="data/cases/LGAV_LFPG/inference_cache_alt",
    use_cache=True,
)
```

## Notes and guardrails

- Inference requires case configs with `cost_model_version: lin_disent`. Legacy
  configs (`cost_model_beta*`) are rejected by default.
- Checkpoints created by recent training runs include config snapshots and a
  graph fingerprint. The inference metadata reports whether the checkpoint
  fingerprint matches the case graph.
- 4D reconstruction uses `get_4d_trajectory` and requires a takeoff timestamp
  for time anchoring.

# Inference Usage

This page shows how to run inference with the lin_disent cost model using
`compute_4d_path_for_dataset` (dataset-backed) or `compute_4d_path_for_flight`
(single flight intent). Both paths load a trained checkpoint, run backward SVI,
sample trajectories, and optionally reconstruct 4D outputs.

## Dataset-backed inference

```python
from equinox.sampling.pipeline import compute_4d_path_for_dataset

res = compute_4d_path_for_dataset(
    case_dir="data/cases/LGAV_LFPG",
    checkpoint_path="data/cases/LGAV_LFPG/batch_sgd_results/checkpoint_iter_200.pt",
    flight_id="392AECAFR98CQ",
    takeoff_timestamp=1682781206,
    n_samples=10,
    policy="sample",
    write_4d_csv=True,
    seed=7,
)

print(res.samples[0].route)
```

### Auto-select flight and checkpoint

If you omit `checkpoint_path`, the newest `checkpoint_iter_*.pt` under
`case_dir/batch_sgd_results` is used (falling back to `final_results.pt`).
If you omit `flight_id` and `takeoff_timestamp`, the first available flight
in `case_dir/tres_runs` is chosen.

```python
from equinox.sampling.pipeline import compute_4d_path_for_dataset

res = compute_4d_path_for_dataset(
    case_dir="data/cases/LGAV_LFPG",
    n_samples=100,
    policy="greedy",
    return_4d=True,
    write_4d_csv=True,
    output_dir="data/cases/LGAV_LFPG/inference100",
)
```

## Persist outputs to disk

When `output_dir` is provided, the pipeline writes:
- `routes.txt` with one route per line (`total_cost,waypoint...`)
- `metadata.pt` with run metadata
- `trajectory_*.pt` for each sample (if `return_4d=True`)

Set `write_4d_csv=True` (requires `return_4d=True` and `output_dir`) to also emit:
- `shortest_path_4d_waypoints.csv` (per-waypoint 4D points)
- `shortest_path_4d_trajectories.csv` (segment-based, Silverdrizzle-compatible schema)
- `shortest_path_4d_trajectories_tranched.csv` (vertical tranchification output)

```python
from equinox.sampling.pipeline import compute_4d_path_for_dataset

res = compute_4d_path_for_dataset(
    case_dir="data/cases/LGAV_LFPG",
    checkpoint_path="data/cases/LGAV_LFPG/batch_sgd_results/checkpoint_iter_200.pt",
    n_samples=100,
    policy="sample",
    return_4d=True,
    seed=7,
    output_dir="data/cases/LGAV_LFPG/inference_outputs/sample_100",
    write_4d_csv=True,
    tranche_altitudes_ft=[10000, 15000, 20000, 24000, 28000, 32000],
)
```

## Flight-specific inference

`compute_4d_path_for_flight` runs TResPASS forward/backward, thinning, wind
averaging, backward SVI, and sampling for a single flight intent. You must
provide a takeoff time (either `takeoff_timestamp` or `takeoff_time_str`) so
wind lookup and time anchoring are deterministic.

```python
from equinox.sampling.pipeline import compute_4d_path_for_flight

res = compute_4d_path_for_flight(
    case_dir="data/cases/LGAV_LFPG",
    checkpoint_path="data/cases/LGAV_LFPG/batch_sgd_results/checkpoint_iter_200.pt",
    origin_node="LGAV",
    goal_node="LFPG",
    takeoff_time_str="2023-04-29 17:13:26",
    n_samples=10,
    policy="sample",
    return_4d=True,
    output_dir="data/cases/LGAV_LFPG/inference_outputs/flight_intent",
    write_4d_csv=True,
)
```

## Caching SVI results

Backward SVI is the most expensive step. By default, results are cached under
`case_dir/inference_cache` keyed by `(checkpoint_hash, flight_id, takeoff_ts, gamma)`.
Disable caching or change its location as needed:

```python
from equinox.sampling.pipeline import compute_4d_path_for_dataset

res = compute_4d_path_for_dataset(
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

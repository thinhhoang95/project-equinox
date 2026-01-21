# Automated Evaluation Scripts

The goal is to automate evaluation of trained checkpoints produced by
`src/equinox/training/batch_sgd_pipeline_parallel_tsb_truellh.py`.
Because MaxEnt IRL treats links independently, we need a geometric
route comparison that captures geographic closeness rather than
per-link likelihood.

## Evaluation protocol (per flight)

Unit of evaluation: `(flight_id, takeoff_time)` from
`case_dir/tres_runs/all_routes_feasibly_snapped.csv`.

1. Sample routes with `compute_4d_path_for_dataset`:
   - `return_4d=False` (Frechet only needs 2D geometry)
   - `n_samples` per flight (default 200)
   - `policy="sample"` or `policy="greedy"`
   - optional `output_dir` to persist `routes.txt` for auditing
2. Reference route: `route` column from `all_routes_feasibly_snapped.csv`.
3. Canonicalize both reference and sampled routes:
   - remove consecutive duplicate waypoints
   - validate waypoint existence in the graph
     - strict: missing waypoint => mark flight invalid
     - drop: remove missing waypoints, track `dropped_waypoints_*`
4. Optional resampling:
   - resample route points to uniform spacing in NM before Frechet
   - note: resampling improves geometric stability when waypoint
     density varies by route
5. Convert waypoint lists to coordinates via graph node `lat/lon`.
6. Compute discrete Frechet distance with haversine (NM) point metric.

## Discrete Frechet definition

Let `P = (p0..pn-1)` and `Q = (q0..qm-1)` be polyline point sequences.
The discrete Frechet recurrence:

`c(i,j) = max(d(pi,qj), min(c(i-1,j), c(i-1,j-1), c(i,j-1)))`

Return `c(n-1,m-1)` in nautical miles.

## Per-flight metrics

Required (per existing doc):
- max Frechet distance across sampled routes
- variance of Frechet distances

Recommended additions:
- min Frechet distance (best sampled match)
- mean + median Frechet (duplicate-weighted)
- support size: `n_unique / n_samples`
- entropy of sampled route distribution (optional)
- coverage: for thresholds `{5,10,20,50,100}` NM
  - `P(d_F <= tau)` over samples (duplicate-weighted)
  - `I(min d_F <= tau)` (did any route get within tau?)

Report metrics on:
- all samples (duplicate-weighted)
- unique routes only (support perspective)

## Case-level aggregation

For each metric across flights:
- mean / median / p90 / p95

Coverage summary:
- average coverage per threshold
- hit rate per threshold (fraction of flights with `min <= tau`)

## Output layout

Use a run directory keyed by checkpoint + settings:

`case_dir/eval_temp/<run_id>/`
- `inference/<flight_id>_<takeoff_time>/routes.txt`
- `metrics_per_flight.csv`
- `summary.json`
- `closest_routes/<flight_id>_<takeoff_time>.json` (top-k closest routes)

Run id example:
`<checkpoint_hash>_n200_g0p1_psample_rs20`

## Parallelization and caching

Parallelize across flights (not within a flight):
- `ProcessPoolExecutor(max_workers=K)`
- each worker runs inference + Frechet + metrics

Device strategy:
- CPU recommended for high parallelism
- GPU: keep `max_workers` low to avoid CUDA contention

Caching:
- keep `use_cache=True` so backward SVI is reused from
  `case_dir/inference_cache`.

## Reproducibility

Deterministic per-flight sampling seed:
- global `--seed S`
- `seed_i = stable_hash(f"{flight_id}:{takeoff_time}:{S}")`

## Failure handling

Do not crash the whole run on a single flight:
- missing CLSR/WIND artifacts
- missing reference route
- missing waypoint coords
- empty sampled routes

Record `status` + `error_message` per flight.

## Optional sweeps

Evaluate multiple checkpoints:
- loop through `batch_sgd_results/checkpoint_iter_*.pt`
- run eval per checkpoint
- build learning curves for key metrics (e.g., median min Frechet)

## Implementation location

All eval code lives under `src/equinox/evals`.

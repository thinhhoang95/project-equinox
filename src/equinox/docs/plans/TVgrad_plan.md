Could you help me: (i) first verify carefully the formulas and the plan, and (ii) ONLY WHEN THE PLAN IS SOUND, proceed to implement the following plan. The goal is to make sure that the projection of the preference gradient stays consistent mathematically with the (time-changing) common dispatch features X in [batch_sgd_pipeline_parallel_tsb_truellh.py](src/equinox/training/batch_sgd_pipeline_parallel_tsb_truellh.py) and [disentanglement.py](src/equinox/preferences/disentanglement.py) .

---
# Plan

## What’s fixed today (and why it’s inconsistent with closure-level \(p(t)\))
In `batch_sgd_pipeline_parallel_tsb_truellh.py`, the projector is built **once at startup** using a **fixed per-edge tailwind mean**, then reused forever:

```1145:1207:src/equinox/training/batch_sgd_pipeline_parallel_tsb_truellh.py
pref_enabled = components["cost_model_version"] == "lin_disent"
pref_projector = None
...
if pref_enabled:
    ...
    d_e = global_counts[edge_u_cpu, edge_v_cpu].to(device=device, dtype=torch.float64)

    mean_tailwind_e = _compute_mean_tailwind_per_edge(...)
    ...
    X_raw = build_feature_matrix(
        pref_edge_u,
        pref_edge_v,
        components["dist_matrix"],
        components["ac_matrix"],
        cruise_speed_kts=float(components["cost_model"].cruise_speed_kts),
        tailwind_values_w=mean_tailwind_e,
        device=device,
        dtype=torch.float64,
    )
    X_norm, feature_means, feature_scales, manual_scales = d_weighted_normalize_features(...)
    pref_projector = PreferenceProjector(X_norm, d_e, ...)
```

And `disentanglement.py` explicitly encodes that assumption (“tailwind must be fixed”) because it was designed for precomputing \(X^\top D X\):

```74:109:src/equinox/preferences/disentanglement.py
- For disentanglement/projection, `tailwind_values_w` must be a fixed per-edge statistic
  (e.g., climatology mean tailwind per edge).
...
time_e = dist / (60.0 * (cruise_speed_kts + tailwind_e))
```

## Target behavior
Per batch, build the **time column** of \(X\) from the **current closure-graph distribution**:

$$
X_{\text{time}}(e)=\frac{T_e}{N_e},\quad
N_e=\sum_{t\in C(e)} p(t),\quad
T_e=\sum_{t\in C(e)} p(t)\,\text{time}(t).
$$

Then use that batch \(X\) to project the preference gradient/update (a moving projector).

## Where this plugs into the current pipeline
Preference updates happen here:

```1399:1449:src/equinox/training/batch_sgd_pipeline_parallel_tsb_truellh.py
if pref_enabled and pref_projector is not None and pref_grad_queue:
    ...
    pref_grad_proj = pref_projector.project(pref_grad_avg)
    p_e = p_e - batch_config.pref_learning_rate * pref_grad_proj
    p_e = pref_projector.project(p_e)
```

The plan is: **rebuild `pref_projector` inside the loop, per iteration/batch**, right before this block, using batch-aggregated \((N_e, T_e)\).

## Data plumbing you need (minimal + consistent)
### In workers (per flight)
You already compute `expected_counts` (dense \(N\times N\)) via `backward_gradient_pass`, and you already return `pref_grad_e`.

Add two more per-flight vectors along the canonical edge list `(edge_u, edge_v)`:

- **`n_expected_e`**: `expected_counts[edge_u, edge_v]`  (this is \(N_{f,e}\))
- **`t_expected_e`**: \(T_{f,e}\) gathered the same way

To get \(T_{f,e}\) **without redoing work**, the cleanest approach is to extend the DP/backward step that already computes \(p(t)\) to also accumulate `p_transition * time(t)` per base link.

Practical note: you’ll want to return **numerator + denominator**, not `X_time` directly, so the main process can do the correct batch aggregation:
\[
X_{\text{time,batch}}(e)=\frac{\sum_f T_{f,e}}{\sum_f N_{f,e}}.
\]

### In main (per batch)
Accumulate across successful flights:

- `N_batch_e = Σ_f n_expected_e`
- `T_batch_e = Σ_f t_expected_e`
- `X_time_batch_e = T_batch_e / N_batch_e` where `N_batch_e > eps`

Then rebuild `X` and `PreferenceProjector` and proceed with the projected preference update.

## File-by-file revision plan
### 1) `src/equinox/training/batch_sgd_pipeline_parallel_tsb_truellh.py`
- **Extend `FlightGradientResult`** to carry `n_expected_e` and `t_expected_e` (both optional tensors).
- **Worker path (`process_single_flight`)**:
  - After `backward_gradient_pass`, gather `n_expected_e`.
  - Also obtain `t_expected_e` (see “supporting change” below) and return both on CPU.
- **Main loop**:
  - While iterating `batch_results`, accumulate sums for:
    - `pref_grad_sum_e`
    - `N_batch_e`
    - `T_batch_e`
  - Compute `X_time_batch_e` just before the preference update.
  - Rebuild `X_raw = [1, ac_dist, X_time_batch]`, normalize, and re-instantiate `PreferenceProjector` for this iteration.
  - Use that **fresh** projector in the existing update block (project grad, SGD step, reproject \(p\)).

**Edge handling for `N_batch_e == 0`** (you need a deterministic fallback):
- **Recommended**: keep a persistent `x_time_fallback_e` (e.g., `dist/(60*cruise_speed)` or the existing climatology-based time) and fill missing entries from it.
- Optional stability knob: EMA smoothing `x_time ← (1-β)x_time + β x_time_batch` on edges with support.

### 2) `src/equinox/preferences/disentanglement.py`
Make it explicit that “projector X” can be rebuilt, and add a clean API for the new workflow:
- **Add a helper** that builds \(X\) when **time is already precomputed per edge**, e.g. `build_feature_matrix_from_time(...)` that stacks `[1, ac_dist, time_e]`.
- Keep the existing `build_feature_matrix(... tailwind_values_w=...)` for the old fixed-tailwind path (useful for initialization/fallback), but **relax the docstring** so it no longer claims tailwind must be fixed in general—only that it must be fixed *if you want a static projector*.

### Supporting change (required to be “correct”)
You need access to \(T_{f,e}=\sum p(t)\,\text{time}(t)\). The most consistent place to compute it is alongside the existing probability pass that forms `expected_counts` (since that’s where \(p(t)\) is already computed).

Concretely: extend the routine that currently does:

- `expected_counts[u,v] += p_transition`

to also do:

- `time_sum[u,v] += p_transition * time(t)`

and expose that to the worker.

(You can do this either by extending `backward_gradient_pass`’s return values, or by adding a small “expected edge stats” helper that shares the same \(p(t)\) computation.)

## Validation checklist (fast + high-signal)
- **Sanity on stats**:
  - fraction of edges with `N_batch_e>0`
  - `X_time_batch_e` min/max/mean (especially on `d_e>0` support)
- **Projector behavior**:
  - log `||X^T D p||` each iteration (it should stay ~0 after reprojection)
  - log condition number of \(X^\top D X\) (still 3×3, cheap)
- **Correctness smoke test for linear model** (`CostLinearDisentangled`):
  - For a small batch, compare the autograd common-weight gradient vs the analytic form using your reconstructed per-edge expected features; they should match closely (up to numerical tolerance).
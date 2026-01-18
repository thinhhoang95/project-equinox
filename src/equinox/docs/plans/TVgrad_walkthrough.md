# Walkthrough: “Time-Varying Projector” for Preference Gradients

This document explains the concrete code changes that implement the plan in `src/equinox/training/TVgrad_plan.md`.

The goal is to keep the **projection constraint** consistent with the **time-varying common-feature matrix** \(X\) when the *time* column is defined using the **current closure distribution** (rather than a fixed climatology tailwind statistic).

In short:

- We keep the identifiability constraint:
  \[
  X^\top D\,p = 0
  \]
  with fixed global empirical weights \(D=\mathrm{diag}(d)\).
- But we now rebuild \(X\) (and thus the projector) **every batch** using:
  \[
  X_{\text{time}}(e)=\frac{T_e}{N_e},\quad
  N_e=\sum_{t\in C(e)} p(t),\quad
  T_e=\sum_{t\in C(e)} p(t)\,\text{time}(t).
  \]

---

## 1) What changed conceptually

Previously, `batch_sgd_pipeline_parallel_tsb_truellh.py` built a single projector at startup using:

- fixed `d_e` from global empirical counts,
- fixed `X` computed from a **fixed per-edge mean tailwind** statistic.

That is consistent if your projector is intended to be **static**.

But if your interpretation is that the time feature should reflect the *current closure-state distribution* (i.e., the model’s current implied transition probabilities), then the time column is **not static** and the projector must move with it. This is what we implement.

---

## 2) Key math mapping to current code

### 2.1 What `expected_counts` already is

In `src/equinox/dp/trespass/amorwin/backward_gradient.py`, the function `backward_gradient_pass(...)` already computes:

- `link_traversal_likelihoods[u,v] += p_transition`

where `p_transition = exp(log_p_transition)` is the model probability for an individual *state transition* whose base waypoint edge is `(u,v)`.

Therefore, for a base edge \(e=(u,v)\):

- \(N_e\) is exactly:
  \[
  N_e = \sum_{t\in C(e)} p(t)
  \]
  and corresponds to `expected_counts[u,v]`.

### 2.2 What we add: `edge_time_sums`

We extend that same probability loop to also accumulate:

- `edge_time_sums[u,v] += p_transition * time_transition`

with:

- `time_transition = dist(u,v) / (60 * (cruise_speed_kts + tailwind_transition))`

This matches the **time definition used by the linear model** and the original feature builder.

Therefore:

- \(T_e\) is exactly:
  \[
  T_e = \sum_{t\in C(e)} p(t)\,\text{time}(t)
  \]
  and corresponds to `edge_time_sums[u,v]`.

Then per batch:

- \(X_{\text{time}}(e)\) is:
  \[
  X_{\text{time}}(e) = \frac{T_e}{N_e}
  \]
  implemented as `x_time_batch = t_expected_sum_e / n_expected_sum_e` with fallbacks on `N_e==0`.

---

## 3) File-by-file implementation details

### 3.1 `src/equinox/dp/trespass/amorwin/backward_gradient.py`

**Change:** Add an optional return that exposes the time moment per base edge.

- New parameters:
  - `return_edge_time_sums: bool = False`
  - `cruise_speed_kts: Optional[float] = None`
- When `return_edge_time_sums=True`, the function returns a 4-tuple:
  - `(expected_counts, gradient, log_partition_z, edge_time_sums)`
- `edge_time_sums` is a dense `(num_nodes, num_nodes)` tensor.

Why here:

- This is exactly where the code already computes `p_transition`.
- Computing `T_e` anywhere else would either recompute probabilities or risk inconsistency.

### 3.2 `src/equinox/training/batch_sgd_pipeline_parallel_tsb_truellh.py` (worker)

**Change:** Workers now return, per flight, the denominators and numerators needed for batch aggregation:

- `n_expected_e`: gathered along the canonical edge list:
  - `expected_counts[edge_u, edge_v]`
- `t_expected_e`: gathered similarly:
  - `edge_time_sums[edge_u, edge_v]`

These are added to `FlightGradientResult` as:

- `n_expected_e: Optional[torch.Tensor]`
- `t_expected_e: Optional[torch.Tensor]`

Importantly:

- We still compute `pref_grad_e` exactly as before:
  \[
  g_p(e) = \frac{n_\text{emp}(e) - n_\text{exp}(e)}{\gamma}
  \]
- The extra vectors are *only* for reconstructing the time feature used in the projector.

### 3.3 `src/equinox/training/batch_sgd_pipeline_parallel_tsb_truellh.py` (main process)

**Change:** The projector is rebuilt every iteration (every batch) when preferences are enabled.

#### Startup: persistent “feature parts” + fallback time

At startup we still build a “baseline” `X_raw` using `mean_tailwind_e` (same as before), but we keep:

- `pref_feature_bias_e = X_raw[:,0]`
- `pref_feature_ac_dist_e = X_raw[:,1]`
- `pref_time_fallback_e = X_raw[:,2]` (deterministic fallback time)

This makes batch projector rebuild cheap:

- only the third column changes; first two are fixed.

#### Per batch: aggregate and rebuild

While collecting results:

- accumulate:
  - `n_expected_sum_e = Σ_f n_expected_e`
  - `t_expected_sum_e = Σ_f t_expected_e`

Then just before the preference update:

- compute:
  - `x_time_batch = t_expected_sum_e / n_expected_sum_e` on supported edges
  - fallback to `pref_time_fallback_e` when `n_expected_sum_e` is ~0
- rebuild:
  - `X_raw_batch = [bias, ac_dist, x_time_batch]`
  - normalize under the same `d_e` via `d_weighted_normalize_features`
  - instantiate a fresh `PreferenceProjector`

Finally:

- project the preference gradient
- SGD update `p_e`
- reproject `p_e` (numerical drift removal)

#### Logging

Each batch can now log a high-signal snapshot of the moving time feature:

- time support fraction (how many edges have `N_batch_e > eps`, optionally restricted to empirical support `d_e>0`)
- min/mean/max of the time feature on support
- condition number of \(X^\top D X\) for the rebuilt projector (if available)

This is the quickest way to sanity-check the pipeline without debugging inside workers.

### 3.4 `src/equinox/preferences/disentanglement.py`

**Change:** Add a clean feature-building helper for workflows where time is already computed per edge:

- `build_feature_matrix_from_time(...)` stacks:
  - `[bias, ac_dist, time_e]`

And clarify in the `build_feature_matrix(...)` docstring:

- tailwind must be fixed only if you want a static projector,
- for moving projectors, supply per-edge time explicitly.

---

## 4) Runtime “trace map” (what happens each iteration)

1. Main process broadcasts `cost_model_state` (including current `p`) to workers.
2. Each worker:
   - loads CLSR/WIND files
   - runs forward+backward SVI to get \(V_f,V_b\)
   - calls `backward_gradient_pass(...)` to get:
     - `expected_counts` (base edge marginals)
     - `edge_time_sums` (time-weighted base edge marginals)
     - `gradient` for common model params
   - computes `pref_grad_e` and gathers `(n_expected_e, t_expected_e)` along the canonical edge list.
3. Main process aggregates per-flight results:
   - applies Adam update for common weights
   - rebuilds batch \(X\) using aggregated `T/N`
   - projects and updates preferences with the fresh projector

---

## 5) Edge cases and stability notes

### 5.1 `N_batch_e == 0`

If an edge has no support in the current batch’s closure distribution, `T/N` is undefined.

We use a deterministic fallback:

- `pref_time_fallback_e` from the startup (climatology-based) feature matrix.

This keeps:

- the feature matrix finite,
- the projector well-defined,
- behavior deterministic.

### 5.2 Numerical issues

The code explicitly repairs non-finite time values:

- if `~isfinite(x_time_batch)` then fall back to `pref_time_fallback_e`.

This prevents NaNs from poisoning the projector.

---

## 6) Suggested quick validation when you run training

Without changing code, you can inspect logs for:

- the “Preference projector (batch)” line:
  - `time_support` should be non-trivial for reasonable batch sizes,
  - time values should be within plausible ranges,
  - condition number should remain within a stable range (if it explodes, increase ridge or reduce pref LR).
- “Preference update” line:
  - constraint violation should stay small after reprojection.

---

## 7) Where to look in code (entry points)

- Moving projector rebuild: `src/equinox/training/batch_sgd_pipeline_parallel_tsb_truellh.py`
  - locate the preference update block and the batch projector rebuild just before it.
- Time-moment accumulation: `src/equinox/dp/trespass/amorwin/backward_gradient.py`
  - search for `return_edge_time_sums` and `edge_time_sums`.
- Feature builders: `src/equinox/preferences/disentanglement.py`
  - `build_feature_matrix(...)` and `build_feature_matrix_from_time(...)`.


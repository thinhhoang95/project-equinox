# Walkthrough: “Linear + Disentangled Preferences” refactor

This document explains the code changes made to implement the refactor described in `src/equinox/docs/plans/refact_revised.md`.

It’s written to help you (or a teammate) understand:

- what changed,
- where it lives in the repo,
- how it fits together at runtime (main process vs workers),
- how to enable and tune the new behavior,
- and what is intentionally **not** changed yet (Phase E, tests/benchmarks).

## High-level goal recap

We want a model where waypoint-edge costs decompose as:

> **Total cost** = **common linear cost** + **per-edge preference**
>
> \[
> c(e) = x(e)^\top w + p(e)
> \]

and we enforce the identifiability / “gauge-fixing” constraint:

> \[
> X^\top D\,p = 0
> \]

where:

- \(E\) is the base waypoint graph edge set,
- \(X\) stacks per-edge features \(x(e)\),
- \(D=\mathrm{diag}(d)\) uses **global empirical edge counts** \(d=\hat n_\mathcal{D}\) as fixed weights.

This is implemented in a **new cost model version**, so existing versions remain unchanged unless you explicitly select `"lin_disent"`.

## What changed (map of files)

### New cost model

- `src/equinox/cost/cost_linear_disentangled.py`
  - Adds `CostLinearDisentangled`, a cost model with:
    - trainable **linear weights** `common_weights` (autograd params),
    - **preferences** `preference_matrix_p` stored as a **buffer** (saved in `state_dict`, but not optimized by autograd).

### Config plumbing

- `src/equinox/config.py`
  - `get_cost_model_class()` recognizes `cost_model_version: "lin_disent"`.
- `default.yaml`
  - Documents the toggle as a commented line: `# cost_model_version: lin_disent`.

### SVI hot-loop dtype cleanup

These functions used to cast `distance_matrix_d` / `airspace_charge_matrix_ac` to float64 **inside** the transition loop.
That was expensive. The casts are now hoisted once before the loop.

- `src/equinox/dp/trespass/amorwin/forward_svi_log_temp.py`
- `src/equinox/dp/trespass/amorwin/backward_svi_log_cost_temp.py`

### Disentanglement utilities (new module)

- `src/equinox/preferences/disentanglement.py`
  - Builds deterministic waypoint-edge lists \(E\),
  - Computes global empirical weights \(d\),
  - Builds feature matrix \(X\),
  - Applies \(D\)-weighted normalization,
  - Implements the ridge-stabilized projector \(P_{\perp,D}\).

### Training loop integration (worker + main)

- `src/equinox/training/batch_sgd_pipeline_parallel_tsb_truellh.py`
  - Workers compute per-flight preference gradients (a vector over base edges).
  - Main process aggregates, projects, and applies SGD updates to `p`, then reprojects.
  - Adds CLI knobs for preference SGD + projection ridge.

## The new model: `CostLinearDisentangled`

File: `src/equinox/cost/cost_linear_disentangled.py`

### Model state

The model has two “parts”:

1. **Common linear weights** (trainable autograd parameters)
   - `common_weights`: shape `(3,)`
2. **Preferences** (buffer persisted in checkpoints)
   - `preference_matrix_p`: shape `(num_waypoints, num_waypoints)`
   - buffer means:
     - it appears in `model.state_dict()`,
     - it is *not* returned by `model.parameters()` and thus not updated by Adam.

### Feature definition: fixed per waypoint-edge

The feature map is fixed and time-independent:

- `bias`: \(1\)
- `ac_dist`: \(\mathrm{AC}(e)\cdot d(e) / 100\)
- `dist`: \(d(e)\)

This matches the “common cost is explicitly linear \(Xw\)” decision.

Tailwind is still passed through the shared cost-model call signature for compatibility, but the linear model **does not use it** (because we need \(x(e)\) fixed per edge to precompute the projector).

### Total cost used everywhere

`forward()` returns:

- `common_cost = X @ common_weights`
- plus `pref_cost = preference_matrix_p[u,v]`

SVI and expected-count computations call `cost_model(...)`, so they automatically incorporate preferences when using `"lin_disent"`.

## The disentanglement module: building \(E\), \(X\), \(d\), and \(P_{\perp,D}\)

File: `src/equinox/preferences/disentanglement.py`

### Edge universe \(E\)

We define \(E\) as the directed edges of the **base waypoint graph** (`nx` edges from the `.gml` graph).

- `build_edge_list(graph, node_to_idx)` returns `(edge_u, edge_v)` tensors in a deterministic order (`sorted()`).

This ordering is important: it defines the canonical correspondence between:

- “edge id” (implicitly the index in the edge list),
- entries of any vector in \(\mathbb{R}^m\) (like `p_e` or `pref_grad_e`),
- the rows of \(X\) and weights \(d_e\).

### Global empirical weights \(d\)

`compute_empirical_counts_from_routes(routes, node_to_idx, num_nodes)`:

- parses all demonstration routes (the `route` column in `all_routes_feasibly_snapped.csv`),
- accumulates dense counts `counts[u,v]`.

Then the per-edge weights are gathered along the edge list:

- `d_e = counts[edge_u, edge_v]`

This is a “global, fixed at training start” empirical weighting.

### Feature matrix \(X\)

`build_feature_matrix(edge_u, edge_v, dist_matrix, ac_matrix, ...)` constructs:

`X_raw[:, :] = [bias, ac_dist, dist]`.

### \(D\)-weighted normalization

The projection solve depends on \(M = X^\top D X\). If columns of \(X\) have wildly different scales or are correlated, the solve can be ill-conditioned.

We therefore normalize features using \(D\)-weighted stats:

- weighted mean \(\mu_j = \frac{\sum_e d_e X_{e,j}}{\sum_e d_e}\)
- weighted std \(\sigma_j\) similarly
- and return:
  - `X_norm = (X - mu) / sigma` (bias column is kept as-is)

This makes the projection numerically more stable without changing the conceptual constraint (it’s still “orthogonality to the chosen feature span”, just in a better-conditioned basis).

### Ridge-stabilized projector \(P_{\perp,D}\)

The core operation is:

- \(M = X^\top D X + \lambda I\)
- solve \(M\alpha = X^\top D v\)
- return \(P_{\perp,D}v = v - X\alpha\)

This is implemented by `PreferenceProjector`:

- it precomputes `M` once,
- tries Cholesky (`torch.linalg.cholesky`) and falls back to `torch.linalg.solve`,
- optionally reports a condition number estimate.

The ridge \(\lambda\) is configurable via CLI (`--pref-projection-ridge`) and defaults to `1e-8`.

## Training pipeline changes: how preference learning works

File: `src/equinox/training/batch_sgd_pipeline_parallel_tsb_truellh.py`

### Main process vs workers (what runs where)

- **Main process** owns the authoritative `components['cost_model']`.
  - It constructs global `E`, `d_e`, `X`, and `PreferenceProjector` once at startup when `"lin_disent"` is enabled.
  - It updates:
    - `common_weights` via Adam (existing behavior),
    - `preference_matrix_p` via **plain SGD + projection** (new).

- **Workers** receive `cost_model_state` each batch, reconstruct a cost model, then:
  - run forward/backward SVI (using costs that include current `p`),
  - compute expected counts `n_expected` (existing backward-gradient pass),
  - compute a **preference gradient vector** over base edges.

### Preference gradient: analytic (no autograd)

Because for preferences \( \partial c(e) / \partial p(e) = 1 \), the negative log-likelihood gradient for preferences is:

\[
g_p(e) = \frac{n_{\text{emp}}(e) - n_{\text{exp}}(e)}{\gamma}
\]

In worker code, when `"lin_disent"` is enabled:

- compute per-flight dense `empirical_counts[u,v]` (already existed),
- get dense `expected_counts[u,v]` from `backward_gradient_pass`,
- gather along base edges:
  - `pref_grad_e = (empirical_counts[edge_u, edge_v] - expected_counts[edge_u, edge_v]) / gamma`

This is returned to the main process as `FlightGradientResult.pref_grad_e`.

### Main-process update: projected SGD

In the main loop:

1. Aggregate per-flight `pref_grad_e` across the batch (mean).
2. (Optional) add L2 regularization term:
   - `pref_grad += 2 * alpha_pref_reg * p_e`
3. Project the gradient:
   - `pref_grad_proj = P_perp,D(pref_grad)`
4. SGD step (plain SGD, no momentum):
   - `p_e ← p_e - pref_lr * pref_grad_proj`
5. Reproject to remove numerical drift:
   - `p_e ← P_perp,D(p_e)`
6. Write back into the model buffer:
   - zero out the full `preference_matrix_p`,
   - assign only base edges `(edge_u, edge_v)`.

This ensures the constraint remains satisfied and that `p` persists via checkpoints (`state_dict`).

### Logging + TensorBoard

When preferences are enabled and updated, the loop logs:

- projected preference gradient norm,
- constraint violation `||X^T D p||`,
- min/max/mean of preferences on empirical support (`d_e > 0`).

These are also logged to TensorBoard under:

- `Preferences/Grad_Norm_L2`
- `Preferences/Constraint_Violation`
- `Preferences/Mean`, `Preferences/Min`, `Preferences/Max`

## How to enable the new model

### Config

Set:

```yaml
cost_model_version: lin_disent
```

The main entry points that use `RunConfiguration` will now instantiate `CostLinearDisentangled`.

### Batch training script knobs

In `src/equinox/training/batch_sgd_pipeline_parallel_tsb_truellh.py`:

- `--pref-learning-rate` (default `1e-2`)
- `--pref-projection-ridge` (default `1e-8`)

These control only preference updates. The common linear weights still use `--learning-rate` with Adam.

## SVI dtype changes (performance-only)

In both forward/backward SVI implementations we now cast:

- `distance_matrix_d` to float64 once,
- `airspace_charge_matrix_ac` to float64 once,
- `avg_tailwind_knots_per_transition` to float64 once,

before the per-transition loop.

This removes redundant `.to(dtype=torch.float64)` calls inside hot loops and keeps the existing “stability-first” semantics (still operating in float64 for these values inside SVI).

## What’s intentionally not done yet (follow-ups)

### Phase E: faster gradient derivation

The backward-gradient pass is still the expensive “semi-gradient via many backwards” implementation.
No analytic/common-gradient speedup or chunked vectorized backward has been implemented yet.

### Tests/benchmarks

No new unit tests or regression tests were added in this change set.
If you want them, the most valuable next additions are:

- projector correctness (`X^T D project(p) ≈ 0`),
- training smoke test (`p` changes and remains projected),
- small synthetic “analytic vs autograd” check for the linear model (if we implement Phase E1).

## Practical notes / gotchas

- **Tailwind is ignored** by `"lin_disent"` (by design). If wind effects are important in the “common” model, they must be represented either:
  - by extending the feature set with a fixed per-edge wind statistic, or
  - by moving wind into a different common-cost formulation (but then you lose the “fixed X per edge” property unless you’re careful).
- **Preference matrix is dense \(N\times N\)**: this is simple and fast to index, and acceptable for the current graph sizes, but it increases `state_dict` size.
- **Projection ridge**: if you see unstable or exploding `p`, increase `--pref-projection-ridge` and/or reduce `--pref-learning-rate`.
- **CPU/GPU device**: the projector currently runs on the same `device` as training (whatever `components['device']` ends up being). If you move training to CPU, the projector will be CPU too.

## Quick “sanity check” checklist

When running with `cost_model_version: lin_disent`, you should see:

- “Preference projector condition number” logged once at startup,
- “Preference update: … constraint=…” logged each iteration where a batch produced `pref_grad_e`,
- preference stats (min/max/mean) changing over iterations,
- constraint violation staying small (relative scale depends on ridge and normalization).

If `pref_grad_e` is always missing:

- confirm `components["cost_model_version"] == "lin_disent"`,
- confirm workers have `pref_enabled=True` in `_init_worker`,
- ensure flights are successful (failed flights don’t contribute gradients).

### Complete refactoring plan (engineering-ready, based on `plan_revision_decision.md`)

### 0) Goals / non-goals

- **Goals**
  - **Implement preference disentanglement** with the fixed constraint \(X^\top D p = 0\) (bias included; empirical-occupancy weights \(D=\mathrm{diag}(\hat n_\mathcal{D})\)).
  - **Introduce a new explicitly linear cost model** (do **not** modify `cost_rev4.py`), so “common” cost is truly \(Xw\).
  - **Store preferences as a model buffer** (in `state_dict`, not autograd parameters), and **update `p` with plain SGD** + projection.
  - **Speed up gradient derivation**, and **remove redundant float64 casts in SVI** (keep existing stability-first semantics; chunk where needed).

- **Non-goals (for this refactor)**
  - **Do not change existing model versions’ behavior** unless explicitly gated behind a new `cost_model_version`.
  - **Do not redesign data ingestion / TRES preprocessing** beyond what’s needed for \(D\) computation.

---

### 1) Decisions to encode (from `plan_revision_decision.md`)

- **Common cost becomes explicitly linear**; implement as a **new cost model file**.
- **Preferences `p` live as a buffer in the model** (persisted in `state_dict`, shipped to workers via `cost_model_state`).
- **Edge universe \(E\)** is **the base waypoint-to-waypoint graph edges** (`nx` edges from the `.gml` graph).
- **Dense matrices are acceptable** (graph size stable; dense `N×N` counts ok).
- **Projection stability improvements**: add **ridge** and **\(D\)-weighted feature normalization**, with room for **manual renormalization weights** later.
- **Preference optimizer**: **plain SGD** (not Adam/momentum).
- **Backward-gradient heuristics**: **keep current semantics** for now (tiny-weight threshold + argmax fallback), but small changes are OK if clearly documented and stability-first.
- **Remove redundant float64 casts** in SVI / related hot loops.

---

### 2) Target architecture (what we’re building)

- **New cost model**: `src/equinox/cost/<new_linear_model>.py`
  - **Common cost**: \(c_{\text{common}}(e)=x(e)^\top w\) where \(x(e)\in\mathbb{R}^d\) is **fixed per waypoint-edge**.
  - **Preferences**: \(p(e)\) stored as a **buffer** (recommend dense `preference_matrix_p[N,N]` buffer for simplicity; only entries on \(E\) are used/updated).
  - **Total cost used everywhere** (SVI + expected counts + likelihood): \(c(e)=c_{\text{common}}(e)+p(e)\).  
    This must be true in `forward_soft_value_iteration`, `backward_soft_value_iteration`, and the expected-count computation (or you train the wrong model).

- **Disentanglement utilities** (new module): `src/equinox/preferences/disentanglement.py` (or similar)
  - Build deterministic **edge list** from base graph: `edges = sorted([(u_idx,v_idx), ...])`.
  - Compute **global empirical weights** \(d=\hat n_\mathcal{D}\) (dense `N×N`, plus `d_e` gathered along edge list).
  - Compute **feature matrix** \(X\in\mathbb{R}^{m\times d}\) for the edge list.
  - Precompute/factorize \(M = X^\top D X + \lambda I\).
  - Provide `project(v_e) = v_e - X * solve(M, X^T D v_e)` and diagnostics `||X^T D p||`.

- **Training integration**
  - Update **common params (`w`)** with the existing optimizer (Adam is fine unless you choose otherwise).
  - Update **preferences (`p`)** in the **main process** with **SGD + projection**, using batch-aggregated preference gradients.

---

### 3) Work breakdown (phased, with clear deliverables)

### Phase A — New linear cost model + config plumbing (no disentanglement yet)
- **Add** new cost model file, e.g. `src/equinox/cost/cost_linear_disentangled.py`.
- **Update** `src/equinox/config.py:get_cost_model_class` to recognize a new `cost_model_version` string (e.g. `"lin_disent"`).
- **Update** any docs/config examples (e.g. case `default.yaml`) to show how to select the new version.
- **Acceptance criteria**
  - New model can be constructed by `RunConfiguration.initialize_cost_model(...)`.
  - End-to-end SVI runs with the new model (no preference learning yet; `p` initialized to zero).

### Phase B — Remove redundant `.to(dtype=torch.float64)` casts in SVI (safe perf win)
- **Edit**:
  - `src/equinox/dp/trespass/amorwin/forward_svi_log_temp.py`
  - `src/equinox/dp/trespass/amorwin/backward_svi_log_cost_temp.py`
- **Change**: stop calling `.to(dtype=torch.float64)` inside the per-transition loops; rely on the already-float64 matrices in worker context.
- **Acceptance criteria**
  - Numerical output matches previous within tight tolerance (expect bit-level diffs possible, but no functional regressions).
  - Wall-time reduction measurable on a representative flight.

### Phase C — Define \(E\), build \(X\), compute global \(d\), implement projection \(P_{\perp,D}\)
- **Add** a module `src/equinox/preferences/disentanglement.py` containing:
  - **Edge list builder** from base graph edges (deterministic ordering).
  - **Global `d` computation** over the training dataset:
    - Parse all routes once at training start.
    - Accumulate dense `d_uv` counts; gather `d_e` along the edge list.
  - **Feature definition** for \(X\):
    - Decide the initial fixed-per-edge feature set (must be fixed on waypoint edges; bias included).
    - Implement \(D\)-weighted column centering/scaling + optional manual per-feature scaling overrides.
  - **Projection operator** with ridge:
    - \(M=X^\top D X + \lambda I\)
    - Solve + project; expose diagnostics: conditioning, `||X^T D p||`.
- **Acceptance criteria**
  - For random `p_e`, `project(p_e)` satisfies `||X^T D p||` near 0 (within tolerance driven by ridge).
  - If ridge is needed (ill-conditioning), logs show it clearly.

### Phase D — Preference gradients + projected SGD update in the training loop
- **Update** `src/equinox/training/batch_sgd_pipeline_parallel_tsb_truellh.py`:
  - Extend `FlightGradientResult` to carry **preference gradient** (recommended: a vector `pref_grad_e[m]`).
  - In worker:
    - After computing `expected_counts`, compute per-flight `pref_grad_e = (n_emp_e - n_exp_e) / gamma` (plus optional L2 reg term).
    - Return it with the existing named-parameter gradients.
  - In main:
    - Aggregate `pref_grad_e` across flights (mean).
    - Apply projection: `pref_grad_e ← P_{\perp,D}(pref_grad_e)`.
    - **SGD step** on `p_e` (learning rate `pref_lr`), then **reproject** `p_e ← P_{\perp,D}(p_e)` to remove drift.
    - Write updated `p_e` back into the model buffer (dense `p[u,v]` on base edges; keep non-edges at 0).
  - Add logging:
    - `||X^T D p||`, `||pref_grad||`, `p` stats (min/max/mean on empirical support).
- **Acceptance criteria**
  - `p` changes across iterations (and survives checkpoint save/load because it’s a buffer).
  - Constraint violation `||X^T D p||` stays small and stable.
  - No worker/main mismatch: workers use the updated `p` each iteration (validate by logging a checksum/stats in worker debug mode).

### Phase E — Faster gradient derivation (stability-first; explicit behavior changes)
You have two viable paths; pick one as “v1”, keep the other as a follow-up.

- **Path E1 (recommended if the new model is truly linear \(Xw\) on waypoint edges): analytic gradients**
  - Compute `expected_counts` as now.
  - Compute **common weight gradients analytically**:
    - \(g_w = \sum_{e\in E} x(e)\,(n_\text{emp}(e)-n_\text{exp}(e))/\gamma\)
  - This can eliminate the expensive “Pass 2” autograd loop for the new model entirely.
  - Keep the old autograd path for non-linear models (behind an `isinstance` check or capability flag).
  - **Acceptance criteria**: on a small flight, analytic `g_w` matches autograd `g_w` (when autograd is enabled for the same linear model) within tolerance.

- **Path E2 (if common cost still depends on per-transition quantities you want to learn): vectorized chunked semi-gradient**
  - Implement the chunked “detached coefficient” approach you outlined, **preserving**:
    - argmax fallback for near-zero link normalizers,
    - the `weights[i] > 1e-9` threshold behavior.
  - Be explicit about differences (chunking order, numeric accumulation) and quantify impact.
  - **Acceptance criteria**: speedup vs baseline; gradient similarity within agreed tolerance on a fixed seed run.

### Phase F — Tests, benchmarks, rollout
- **Tests**
  - **Projection tests**: `project(p)` enforces `X^T D p ≈ 0`; ridge behavior; scaling overrides.
  - **Training smoke test**: 1 iteration, 1 batch, deterministic seed; verify `p` updates and is projected.
  - **Gradient correctness**: compare old/new gradient for the linear model on a tiny synthetic graph.
- **Benchmarks**
  - Per-flight runtime breakdown: SVI time, expected-count time, gradient time, IPC overhead.
  - Compare before/after for a representative batch.
- **Rollout**
  - Default remains existing cost model versions.
  - New behavior is activated only via `cost_model_version: "<new>"`.
  - Document “expected behavior changes” prominently (new cost model changes learned routes; projection constrains preferences).

---

### 4) “Stuff the team must decide up front” (to unblock Phase C/D)
- **Feature set for \(X\)** (fixed per waypoint-edge) and initial scaling:
  - Minimum: **bias + distance + charge×distance** (or whichever set you standardize on).
- **Ridge magnitude policy**:
  - fixed (e.g. `1e-8`) vs adaptive (based on condition number / min eigenvalue).
- **Preference learning rate** (`pref_lr`) and optional regularization (L2 on `p`).

# References
- The current implementation is documented in `TRAINING.md`
- The mathematical foundation of the method can be found in `preference_disentanglement.md`.

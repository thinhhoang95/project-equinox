### What the math guide requires (and what it means in your code)
The guide formalizes “common dispatch” vs “edge preference” as a **decomposition of waypoint-edge costs**:

- **Per-edge cost vector**: \(c = Xw + p\)
- **Preference identifiability constraint (recommended)**: \(X^\top D p = 0\), where \(D=\mathrm{diag}(\hat n_\mathcal{D})\) uses **empirical expected edge counts** as weights, and **\(X\) should include a bias column** so the preferences have zero mean on the empirically visited edges (hence zero covariance)  
  See:

```151:190:/Volumes/CrucialX/project-equinox/src/equinox/docs/plans/preference_disentanglement.md
## 5.2 Empirical-occupancy weighted orthogonality (recommended)
...
X^\top D\,p = 0. (12)
...
Efficient application ... solve (X^\top D X)α = X^\top D v, then P_{\perp,D}v = v - Xα.
```

The key practical point: **you never form an \(m\times m\) projector**; projection is “streamable” over edges in \(O(md)\) with a cheap \(d\times d\) solve.

### The most efficient edge-preference data structure (waypoint-to-waypoint edges)
Avoid “objects per edge” (Python dicts / dataclasses for each edge) — they’ll kill memory and CPU. Use a **single contiguous table** keyed by an integer edge id.

I’d recommend this two-level layout:

- **`EdgeIndex` (static, built once from the waypoint graph)**  
  - **`edge_u: int32[m]`, `edge_v: int32[m]`**: edge list in a fixed order (this *is* your \(E\), so \(p\in\mathbb{R}^m\) matches the guide).
  - **Fast lookup option (best runtime, moderate memory)**:  
    - **`edge_id_of_uv: int32[num_nodes, num_nodes]`**, filled with `-1` for non-edges, else `edge_id`.  
    This makes cost-time lookup as fast as your current distance/charge lookups (`matrix[u,v]`).
  - **Low-memory lookup option (best memory, slower runtime)**: CSR adjacency  
    - `indptr: int32[num_nodes+1]`, `dst: int32[m]`, `eid: int32[m]`  
    Lookup needs a search within `dst[indptr[u]:indptr[u+1]]` (OK only if outdegree is tiny).

- **`EdgePreferences` (learned state)**
  - **`p: float32[m]`** (or unconstrained `u: float32[m]` if you choose reparam; but see below)
  - Keep it as a **single contiguous `torch.Tensor`** (CPU or GPU depending on where you run SVI).

Why not `preference_matrix_p: (N,N)` like the commented code in `CostRev4`? Two reasons:
1. It’s dense even for sparse graphs.
2. More importantly, your gradient pipeline currently flattens **all** trainable params into a single vector:

```62:70:/Volumes/CrucialX/project-equinox/src/equinox/dp/trespass/amorwin/backward_gradient.py
cost_model.train()
num_cost_params = sum(p.numel() for p in cost_model.parameters() if p.requires_grad)
```

If you add a huge `(N,N)` preference parameter with `requires_grad=True`, `num_cost_params` explodes and the whole “vector-of-grads passed back from workers” design becomes infeasible.

### Refactor plan: make the pipeline follow the guide (efficiently)
Below is the refactor plan that matches your current architecture (SVI in workers, model update in main), while keeping performance sane.

#### 1) Introduce a “decomposed cost” API: common cost + waypoint-edge preference
Right now every DP call uses:

```95:104:/Volumes/CrucialX/project-equinox/src/equinox/dp/trespass/amorwin/backward_gradient.py
cost_uv = cost_model((edge_u_indices, edge_v_indices), distance_matrix_d, airspace_charge_matrix_ac, tailwind_knots)
```

Refactor conceptually to:

- `common_cost_uv = common_cost_model(...)` (your existing `CostRev4` logic)
- `pref_uv = p[ edge_id_of_uv[u,v] ]`
- `total_cost_uv = common_cost_uv + pref_uv`

Do **not** make `p` a huge autograd parameter; treat it as **externally updated state** (buffer) used in the forward cost.

#### 2) Define \(X\) (with bias) at the waypoint-edge level
You need an \(X\in\mathbb{R}^{m\times d}\) where columns represent “common dispatch factors” you want preferences to be orthogonal to, plus a bias.

Given your current model, a pragmatic starting point is:

- \(x_0(e)=1\) (bias)
- \(x_1(e)=\mathrm{AC}(e)\cdot d(e)\) (or scaled as you do in `cost_rev4`)
- \(x_2(e)=d(e)\) (or another “distance-like” term)

**Wind is time-dependent in your implementation**, so if you want wind inside \(X\) you must decide a fixed per-edge statistic, e.g.:
- empirical mean tailwind for that waypoint-edge over demonstrations, or
- climatology mean, or
- omit wind from \(X\) and treat wind as part of “common” but not part of the disentanglement gauge.

(If you make \(X\) depend on per-flight tailwind, the guide’s “precompute \(X^\top D X\) once” assumption breaks.)

#### 3) Compute \(d=\hat n_\mathcal{D}\) once (global) from demonstration routes
You already compute per-flight empirical counts:

```263:292:/Volumes/CrucialX/project-equinox/src/equinox/training/batch_sgd_pipeline_parallel_tsb_truellh.py
def compute_empirical_counts_for_flight(...)-> torch.Tensor:
    empirical_counts = torch.zeros((num_nodes, num_nodes), dtype=torch.float64)
    ...
    empirical_counts[from_idx, to_idx] += 1.0
```

Refactor to also support accumulating into the edge-id space:
- `d[eid] += 1` for each consecutive `(u,v)` in each demonstrated route.
- scale by `1/N` if you want “expected per-trajectory counts” (scaling doesn’t change the constraint subspace).

#### 4) Implement the weighted projector operator (no big matrices)
Create a small utility (conceptually) that can apply \(P_{\perp,D}\) to any vector \(v\in\mathbb{R}^m\) using:

- precomputed \(M := X^\top D X \in \mathbb{R}^{d\times d}\)
- solve \(M\alpha = X^\top D v\)
- return \(v - X\alpha\)

This is exactly the guide’s efficient recipe:

```182:189:/Volumes/CrucialX/project-equinox/src/equinox/docs/plans/preference_disentanglement.md
(X^\top D X)α = X^\top D v
P_{\perp,D}v = v - Xα
```

Implement it **without storing full \(X\)** if you want maximum memory efficiency: just stream over edges and compute the needed sums.

#### 5) Compute preference gradients analytically from counts mismatch (not via autograd)
Your current negative-loglik gradient accumulation uses mismatch:

```205:211:/Volumes/CrucialX/project-equinox/src/equinox/dp/trespass/amorwin/backward_gradient.py
n_empirical = empirical_counts[u_idx, v_idx]
n_expected = link_traversal_likelihoods[u_idx, v_idx]
total_log_likelihood_grad += (link_grad / gamma) * (n_empirical - n_expected)
```

For preferences, \( \partial c / \partial p_e = 1\). So the per-edge preference gradient is just:

- \(g_p(e) = (n_\text{emp}(e)-n_\text{exp}(e))/\gamma\)  
(+ L2 reg if you want: `+ 2*alpha_pref_reg*p[e]`)

This should be computed **per unique (u,v)** using the same `n_expected` you already computed in Pass 1.

#### 6) Enforce the constraint during updates
Use the guide’s “projected update” idea, but do it at the **batch update point** (main process), not inside DP loops:

- aggregate batch preference gradients into a length-\(m\) vector (dense) or sparse updates + dense projection of `p`
- update `p` (SGD/Adam)
- then **reproject**: `p ← P_{\perp,D} p`
- optionally log `||Xᵀ D p||` as a health metric

This gives you the intended “preferences live in an orthogonal space”.

#### 7) Multiprocessing integration (keep it efficient)
Today each worker returns only `gradient: Dict[str, Tensor]` (common params). Extend the per-flight result to also return one of:
- a **sparse preference update**: `(edge_id[], grad_value[])` for edges touched by that flight, or
- a compact dict `{edge_id: grad}`

Do **not** return a full `(num_nodes,num_nodes)` expected count matrix from each worker unless you’re sure it’s small — it’s unnecessary for preferences once you’ve aggregated by link.

### Practical caveats (worth deciding up front)
- **What exactly is “common feature space” \(X\)?**  
  The constraint only removes correlation with whatever you put in \(X\). If \(X\) excludes wind, preferences can still soak up “wind-like” effects.
- **Preference scale ambiguity vs intercept**: the guide’s bias feature is essential if you want “mean preference = 0 on empirical support” (otherwise `p` can drift by a constant).
- **DP cost calls are extremely hot**: make preference lookup O(1) (dense id matrix) unless memory forces CSR.
- **Your current backward gradient is very expensive** (per-link repeated `.backward()` calls). Preference learning shouldn’t make it worse (hence “analytic pref grad”), but long-term you may want to vectorize/accelerate the common-parameter gradient too.

---

### Why the current common-parameter gradient is slow
`backward_gradient_pass` does **many tiny autograd graphs**:

- It loops **per unique link**, then loops **per transition inside the link**, and for each non-trivial weight does `zero_grad()`, a single-edge forward, and `backward()`:

```127:203:/Volumes/CrucialX/project-equinox/src/equinox/dp/trespass/amorwin/backward_gradient.py
for link_idx, (u_idx, v_idx) in enumerate(all_unique_links):
    ...
    for i in range(num_transitions_in_link):
        if weights[i].item() > 1e-9:
            cost_model.zero_grad()
            single_cost = cost_model(...)
            single_cost.backward()
            ...
            link_grad += weights[i] * single_transition_grad
```

That’s worst-case **O(#transitions)** separate backwards, plus lots of Python overhead and device sync (`.item()`).

### High-level acceleration strategy (keep the same “semi-gradient” semantics)
Keep exactly what your code already does conceptually: treat the distribution over transitions (via `V_f`, `V_b`, `log_partition_z`) as **fixed** when differentiating w.r.t. cost-model parameters. Then you can compute the common-parameter gradient with **one weighted dot-product backward per chunk**.

Key identity (matches your current weighting logic):
- Define, for each waypoint-link \(e=(u,v)\), the conditional weights over its state transitions \(i\in e\):
  \[
  q_i = \frac{p_i}{\sum_{j\in e} p_j} \quad (\text{or your “argmax fallback” if } \sum p_j \approx 0)
  \]
- Your current gradient contribution is:
  \[
  \frac{n_{\text{emp}}(e)-n_{\text{exp}}(e)}{\gamma}\;\sum_{i\in e} q_i \,\nabla_\theta c_i
  \]
- So if we define a **detached scalar weight per transition**
  \[
  a_i := \frac{n_{\text{emp}}(e)-n_{\text{exp}}(e)}{\gamma}\;q_i \quad \text{(stop-grad)}
  \]
  then the whole gradient is just:
  \[
  \nabla_\theta \sum_i a_i \, c_i
  \]
  because \(a_i\) is treated constant.

### Concrete refactor plan for `backward_gradient_pass` (vectorized + chunked)
#### 1) Pre-tensorize transitions once
From `state_transitions` build GPU/CPU tensors for:
- `(u_idx, v_idx)` per transition (shape `[T]`)
- indexing tuples for `V_f[u,k,rho,phase]` and `V_b[v,k',rho',phase']` (shape `[T]`)
- `tailwind[T]` already aligned.

Filter out transitions where `V_f` or `V_b` is `inf` (exactly like current code).

#### 2) Batch compute costs and transition probabilities \(p_i\) (no grad)
Compute, in chunks:
- `cost_i = cost_model((u,v), dist, ac, tailwind_chunk)` (batched)
- `log_p_i = (-v_f_u - cost_i - v_b_v + log_partition_z) / gamma`
- `p_i = exp(log_p_i)`

All of this under `torch.no_grad()`.

#### 3) Compute expected link counts and per-link normalizers (vectorized)
Use a **flat link id**: `lid = u*num_nodes + v` (int64), shape `[T]`.

Then:
- `n_expected_flat = zeros(num_nodes*num_nodes).index_add_(0, lid, p_i)`  
  (this is your `link_traversal_likelihoods` but flattened)
- `n_expected(u,v)` is `n_expected_flat[lid]` for that link.

This replaces the per-transition accumulation loop in Pass 1.

#### 4) Compute conditional weights \(q_i\) (mostly vectorized, small fallback loop)
Base case:
- `q_i = p_i / (n_expected_flat[lid] + eps)`

Fallback (to mimic your current behavior exactly):
- Find links where `n_expected_flat[lid] <= eps` **and** the link has transitions; for each such link, set `q` to one-hot on the transition with max `p_i` (you can reuse a `transitions_by_link` dict, but only touch these rare links).

#### 5) Build detached per-transition coefficients \(a_i\)
Get empirical counts as a flat vector too:
- `n_emp_flat = empirical_counts.reshape(-1)` (already exists per flight)

Then per transition:
- `m_e = (n_emp_flat[lid] - n_expected_flat[lid]) / gamma`
- `a_i = (m_e * q_i).detach()`  (**critical**: stop-grad through \(p\), \(V\), normalization)

#### 6) One backward per chunk for all common parameters
Now do a second pass in chunks (grad enabled):
- recompute `cost_i` for the chunk (batched)
- `loss_chunk = (a_i_chunk * cost_i).sum()`
- `loss_chunk.backward()` (accumulates grads into model params)

Finally, flatten grads to your existing vector format and return.

This replaces **thousands+** of individual `single_cost.backward()` calls with **~T/chunk_size** backwards.

### Expected speedups (where you’ll feel it)
- **Pass 1** becomes a handful of batched `cost_model` calls instead of `T` calls of batch size 1.
- **Pass 2** becomes chunked batched backward instead of “per-link × per-transition” backwards.
- You also eliminate many `.item()` calls that force CPU/GPU sync.

### Optional extra wins (outside backward gradient, but very impactful)
- In both `forward_svi_log_temp.py` and `backward_svi_log_cost_temp.py`, you repeatedly do `distance_matrix_d.to(dtype=torch.float64)` / `airspace_charge_matrix_ac.to(dtype=torch.float64)` inside tight loops (e.g. forward SVI around `L281-L286`). Hoisting those casts **once** before the loop is typically a big wall-time reduction.
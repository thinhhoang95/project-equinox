### TRES state vector (what the “state” is)
In the TRES/TRESPASS artifacts that training consumes, a **state** is stored/identified as a 5-tuple
\[
s=(n,\;k,\;\rho,\;h,\;\phi)
\]
where \(n\) is waypoint index, \(k\) is a wall‑clock time bin, \(\rho\) is a *remaining climb time* bin, \(h\) is altitude (ft), and \(\phi\) is phase (CLIMB/CRUISE/DESCENT). A **closure/transition** stores two such states (source and destination), i.e. a 10‑tuple, optionally with absolute ETAs appended.

```12:19:src/equinox/dp/trespass/thinning.py
- **State**: A 5-tuple (waypoint_idx, k_idx, rho_idx, altitude, phase_idx) representing
  a flight configuration at a specific waypoint, time bin, remaining climb time bin,
  altitude, and flight phase.
  
- **Closure/Transition**: A tuple representing a state transition. The base format is a
  10-tuple: (u_idx, k_u, rho_u, alt_u, phase_u, v_idx, k_v, rho_v, alt_v, phase_v),
  where u and v are the source and destination states. Optionally, closures may include
  absolute ETA times as fields 11 and 12: (..., eta_u_abs_s, eta_v_abs_s).
```

**How discretization relates to “time remaining / time bin”:**
- **Wall‑clock bin** \(k\): conceptually \(k=\left\lfloor(\eta-\eta_{\min})/\Delta t_{\text{wall}}\right\rfloor\), where \(\eta\) is absolute time (seconds since midnight) and \(\Delta t_{\text{wall}}\) is the wall‑clock bin width.
- **Remaining climb bin** \(\rho\): in the backward closures it is treated as “how many climb bins remain until TOC”. In the forward climb pass you’ll see **elapsed-since-takeoff** bins \(\varepsilon\); the backward pass converts by \(\rho \approx \varepsilon_{\max}-\varepsilon\) (so at takeoff \(\rho\) is maximal; at/after TOC \(\rho\to 0\)).

---

### TRES forward loop (feasibility / climb reachability; `tres_forward.py`)
Methodologically, forward TRES is a **kinematic reachability DP**:
- **State tracked forward** is effectively \((n,k,\varepsilon)\) plus altitude+phase as metadata; \(\varepsilon\) is ETTO (“elapsed time since takeoff”) binned.
- It processes nodes in **topological generations** (requires the waypoint graph to be a DAG), batches successor expansions, and calls a physics/performance transition model (`get_next_state_fw`) to propagate \((\eta,h,\phi)\).
- It records **climb-related transitions** (CLIMB→CLIMB, CLIMB→CRUISE) as a compact “climb feasibility skeleton” that the backward pass later uses to enforce/permit climb structure.

---

### TRES backward loop (closure construction; `tres_backward.py`)
Backward TRES is another reachability/feasibility DP, but run from the destination:
- Initializes at the **goal** near landing: \(\rho=0\), \(\phi=\) DESCENT, and \(k\) corresponding to landing-time bin.
- Walks **reverse topological generations** and expands predecessors. It has two conceptual propagation mechanisms:
  - **Standard (CRUISE/DESCENT) propagation** via `get_next_state_bw`, producing predecessor states with \(\rho=0\).
  - **Climb-aware propagation** that uses the forward “climb transitions” to inject feasible CLIMB states and allow a controlled CRUISE→CLIMB switch near TOC (this is where the “remaining climb time” \(\rho\) is computed as roughly \(\varepsilon_{\max}-\varepsilon_u\)).
- Outputs closure tuples in the base 10‑field format (plus absolute ETA fields), which are later thinned and used as the training state-transition graph.

---

### Thinning / morphing (pruning to origin→goal paths; `thin_closures`)
Thinning takes the (possibly huge) closure set and keeps only transitions lying on **some** valid path from any valid origin state to any goal state:
- **Origin states**: source waypoint with \(\rho=\max\rho\) (takeoff-like “full remaining climb time”).
- **Goal states**: destination waypoint (any \(k,\rho,h,\phi\)).
- It computes:
  - forward reachability from origin states
  - backward reachability to goal states
  - keeps the intersection and filters transitions accordingly.
- Optional **k-morphing** merges adjacent \(k\)-bins when their implied/recorded ETA intervals overlap within a tolerance window (reducing state-space size without changing feasible connectivity much).

---

### Soft Bellman value iteration — forward “message” (`forward_soft_value_iteration`)
This is the first of the two *soft DP* loops used during learning. It computes a **forward partition / cost-to-come** in log-space.

Define the MaxEnt path model (for a fixed flight) as
\[
p(\xi)\propto \exp\!\left(-\frac{C(\xi)}{\gamma}\right),\qquad C(\xi)=\sum_{e\in\xi}c(e),
\]
with temperature \(\gamma>0\).

Let \(Z_{\to}(s)\) be the “sum of exponentiated negative costs” over all partial paths from start to state \(s\). Then
\[
Z_{\to}(v)=\sum_{u\to v} Z_{\to}(u)\,\exp\!\left(-\frac{c(u\!\to\! v)}{\gamma}\right).
\]
In log-space \(L(s)=\log Z_{\to}(s)\):
\[
L(v)\leftarrow \logaddexp\!\Big(L(v),\,L(u)-\frac{c(u\to v)}{\gamma}\Big).
\]
Finally \(V_f(s)=-\gamma\,L(s)\) (a “free-energy” / soft cost-to-come).

```196:320:src/equinox/dp/trespass/amorwin/forward_svi_log_temp.py
    # 1. Create L_val and fill with -∞ (unreachable)
    L_val = torch.full(
        (num_nodes, num_time_bins_wall_clock, num_rho_bins, num_phases),
        fill_value=float('-inf'),
        dtype=torch.float64,
        device=device
    )

    # 2. Identify which (k_u, rho_u, phase_u) on the origin_node are actually used.
    actual_origin_states = set()
    for st in state_transitions:
        u_idx, k_u, rho_u, u_alt_ft, phase_u, v_idx, k_v, rho_v, v_alt_ft, phase_v = get_base_transition(st)
        if u_idx == origin_node_idx:
            actual_origin_states.add((k_u, rho_u, phase_u))

    # 3. Initialize each origin‐state to log(1 / num_actual_origin_states)
    # ...
        initial_logmass = 0.0
        for (k_u, rho_u, phase_u) in actual_origin_states:
            L_val[origin_node_idx, k_u, rho_u, phase_u] = initial_logmass

    # ...
    # 5. Main loop: for each transition, update L(v) = logaddexp( L(v),  L(u) - cost(u→v) / gamma ).
    for i, (original_index, trans) in enumerate(sorted_indexed_transitions):
        u_idx, k_u, rho_u, u_alt_ft, phase_u, \
        v_idx, k_v, rho_v, v_alt_ft, phase_v = get_base_transition(trans)
        L_s_u = L_val[u_idx, k_u, rho_u, phase_u]
        if torch.isneginf(L_s_u):
            continue
        tailwind_knots = avg_tailwind_knots_per_transition[original_index]
        cost_uv_tensor = cost_model((edge_u_indices, edge_v_indices), distance_matrix_d, airspace_charge_matrix_ac, tailwind_knots)
        cost_uv = float(cost_uv_tensor.item())
        a_u = L_s_u - cost_uv / gamma
        old_L_v = L_val[v_idx, k_v, rho_v, phase_v]
        new_L_v = torch.logaddexp(old_L_v, a_u)
        L_val[v_idx, k_v, rho_v, phase_v] = new_L_v

    V_soft = -gamma * L_val
    return V_soft
```

**Key methodological points:**
- **State index used by SVI** is \((n,k,\rho,\phi)\). Altitude is carried in closures mainly for wind amortization/feasibility, but the SVI tensor does not add an altitude dimension.
- This is a **single pass** over a sorted transition list (not an iterative fixed-point solve), relying on the state-transition structure behaving like a DAG under the chosen ordering.

---

### Soft Bellman value iteration — backward cost-to-go (`backward_soft_value_iteration`)
The second soft DP loop computes the **soft cost-to-go** \(V_b(s)\) via the soft Bellman recursion:
\[
V_b(s)= -\gamma\log\sum_{s\to v}\exp\!\left(-\frac{c(s\to v)+V_b(v)}{\gamma}\right)
\]
(i.e., a soft-min over \(c+V\)).

```104:249:src/equinox/dp/trespass/amorwin/backward_svi_log_cost_temp.py
    # 1. Create V_val and fill with +∞
    V_val = torch.full(
        (num_nodes, num_time_bins_wall_clock, num_rho_bins, num_phases),
        fill_value=float('inf'),
        dtype=torch.float64,
        device=device
    )

    # 2-3. Initialize V(goal states)=0 for actual goal states found in transitions.
    # ...

    # 5. Main loop: for each transition u→v, update V(u) = softmin( V(u), V(v) + cost(u→v) ).
    for i, (original_index, trans) in enumerate(sorted_indexed_transitions):
        u_idx, k_u, rho_u, u_alt_ft, phase_u, \
        v_idx, k_v, rho_v, v_alt_ft, phase_v = get_base_transition(trans)

        V_s_v = V_val[v_idx, k_v, rho_v, phase_v]
        if torch.isinf(V_s_v):
            continue

        tailwind_knots = avg_tailwind_knots_per_transition[original_index]
        cost_uv_tensor = cost_model((edge_u_indices, edge_v_indices), distance_matrix_d, airspace_charge_matrix_ac, tailwind_knots)
        cost_uv = float(cost_uv_tensor.item())

        val_from_v = V_s_v + cost_uv
        old_V_u = V_val[u_idx, k_u, rho_u, phase_u]
        new_V_u = -gamma * torch.logaddexp(-old_V_u / gamma, -val_from_v / gamma)
        V_val[u_idx, k_u, rho_u, phase_u] = new_V_u
```

---

### Soft Bellman marginals (how expected link usage is computed)
With \(V_f\) and \(V_b\), the code constructs **state-transition probabilities** and aggregates them to **link marginals** (expected traversals per waypoint edge).

It first computes a scalar “start free energy”
\[
V_{\text{start}}=-\gamma\log\sum_{s\in S_{\text{origin}}}\exp\!\left(-\frac{V_b(s)}{\gamma}\right),
\]
then for each closure transition \(u\to v\),
\[
\log p(u\to v)=\frac{-V_f(u)-c(u\to v)-V_b(v)+V_{\text{start}}}{\gamma}.
\]

```72:136:src/equinox/dp/trespass/amorwin/backward_gradient.py
    v_origin = V_b[origin_node_idx]
    non_inf_v = v_origin[v_origin != torch.inf]
    log_partition_z = -gamma * torch.logsumexp(-non_inf_v / gamma, dim=0) if non_inf_v.numel() > 0 else torch.tensor(float('inf'))

    # ...
    with torch.no_grad():
        for i, trans in enumerate(state_transitions):
            u_idx, k_u, rho_u, _, phase_u, v_idx, k_v, rho_v, _, phase_v = get_base_transition(trans)
            v_f_u = V_f[u_idx, k_u, rho_u, phase_u]
            v_b_v = V_b[v_idx, k_v, rho_v, phase_v]
            # ...
            cost_uv = cost_model((edge_u_indices, edge_v_indices), distance_matrix_d, airspace_charge_matrix_ac, tailwind_knots).squeeze()

            log_p_transition = (-v_f_u - cost_uv - v_b_v + log_partition_z) / gamma
            p_transition = torch.exp(log_p_transition)
            link_traversal_likelihoods[u_idx, v_idx] += p_transition
```

Then the **expected link count** for edge \(e=(i,j)\) is
\[
N_{\text{exp}}(i,j)=\sum_{\text{closures }u\to v\text{ with }(u_n,v_n)=(i,j)} p(u\to v).
\]

---

### MaxEnt inverse learning gradient (dispatch/common weights + preferences)
The optimization target per flight is the MaxEnt negative log-likelihood:
\[
\text{NLL}(\theta)=\frac{1}{\gamma}\Big(C_\theta(\xi_{\text{emp}})+V_{\text{start}}(\theta)\Big),
\]
which yields the classic “empirical minus expected counts” signal.

- **Common (dispatch) parameters**: the gradient pass computes a vector gradient w.r.t. trainable cost-model parameters by combining link marginals with \(\nabla_\theta c\).
- **Per-edge preference offsets** \(p(e)\): because \(p(e)\) enters additively per traversal, the per-edge gradient is simply
\[
\frac{\partial \text{NLL}}{\partial p(e)}=\frac{N_{\text{emp}}(e)-N_{\text{exp}}(e)}{\gamma}.
\]

In the training worker, this is exactly what’s computed:

```919:933:src/equinox/training/batch_sgd_pipeline_parallel_tsb_truellh.py
        if ctx.get("pref_enabled"):
            edge_u = ctx.get("edge_u")
            edge_v = ctx.get("edge_v")
            # ...
            expected_e = expected_counts[edge_u, edge_v]
            # Gradient of the negative log-likelihood w.r.t. the per-edge cost offset p(e):
            #   ∂NLL/∂p(e) = (N_empirical(e) - N_expected(e)) / gamma
            pref_grad_e = (empirical_counts[edge_u, edge_v] - expected_e) / gamma
```

---

### Cost model: “dispatch/common” vs “preference” (and what’s being disentangled)
Training uses `lin_disent`, implemented by `CostLinearDisentangled`, with the decomposition
\[
c(e;w,p)=x(e)^\top w + p(e).
\]

```16:31:src/equinox/cost/cost_linear_disentangled.py
    Explicitly linear common cost with additive per-edge preferences.

    Common cost: c_common(e) = x(e)^T w
    Total cost: c(e) = c_common(e) + p(e)

    Sign convention:
      - ``preference_matrix_p[u, v]`` is an additive *cost offset* (a penalty term).
      - Higher values increase the edge cost and therefore make the edge less likely under
        MaxEnt policies of the form ``p(path) ∝ exp(-cost(path)/gamma)``.
      - If you want a "preference score" where larger means *more chosen*, use ``-p(e)``.

    Feature definition (fixed per waypoint-edge):
      - bias = 1
      - ac_dist = AC(e) * d(e) / 100.0
      - time = 60.0 * dist / (cruise_speed_kts + tailwind_e)
```

So “dispatch/common” is \(x^\top w\) (bias + charges×distance + time), while “preference” is an edge-specific residual \(p(e)\).

---

### Disentanglement & gauge fixing (why projection is needed, and what it enforces)
Without constraints, \(p(e)\) is **not identifiable**:
- It can absorb any component in the span of the common features \(X\) (making \(w\) ambiguous).
- It can absorb “node potential” differences \(B^\top\phi\) (path costs shift by constants for fixed OD), a gauge freedom.

The code uses a **projector** to enforce both:
\[
X^\top W\,p = 0\qquad\text{and}\qquad B\,W\,p = 0,
\]
i.e. preference offsets live in the **cycle space** and are **orthogonal** (under \(W\)) to the common-feature subspace.

```330:333:src/equinox/preferences/disentanglement.py
class GaugeFixedPreferenceProjector:
    """Projects onto X^T W p = 0 and B W p = 0 using a stable Schur complement."""
```

The projection is applied as a structured subtraction of a feature component and a node-potential component:
\[
p \leftarrow p - X\alpha - B^\top\psi,
\]
with \(\alpha,\psi\) solved via a Schur-complement system (no dense projection matrix is formed):

```566:583:src/equinox/preferences/disentanglement.py
    def project(self, v_e: torch.Tensor) -> torch.Tensor:
        rhs1 = self.X.t().matmul(self.w_e * v_e)
        rhs2 = self._bw_apply(v_e)
        z2 = self._solve_laplacian(rhs2)
        rhs1_eff = rhs1 - self._C.t().matmul(z2)
        alpha = self._solve_aeff(rhs1_eff)
        psi = z2 - self._Z.matmul(alpha)
        return v_e - self.X.matmul(alpha) - self._bt_apply(psi)
```

**How this connects to “dispatch vs preference disentanglement” in the pipeline:**
- The pipeline builds \(X\) from the common-feature columns (bias, ac_dist, time), normalizes it, and uses the projector to keep \(p\) orthogonal to those columns.
- It also maintains cycle-gauge node potentials (stored in the cost model) so the *common* part can be “cycle-projected” consistently (removing unidentifiable potential components from the common features).

---

### Preference learning update (projected gradient step)
In the main training loop, per-flight preference gradients are averaged across the batch, optionally regularized, then projected and applied:
\[
p \leftarrow \Pi\Big(p - \eta_p\,\Pi(\nabla_p \text{NLL})\Big),
\]
where \(\Pi\) is the projector enforcing the constraints above.

```1691:1734:src/equinox/training/batch_sgd_pipeline_parallel_tsb_truellh.py
                pref_grad_avg = pref_grad_sum / float(len(pref_grad_queue))
                # ...
                if alpha_pref_reg != 0.0:
                    pref_grad_avg = pref_grad_avg + 2.0 * alpha_pref_reg * p_e

                pref_grad_proj = pref_project(pref_grad_avg)
                p_e = p_e - batch_config.pref_learning_rate * pref_grad_proj
                p_e = pref_project(p_e)

                with torch.no_grad():
                    pref_matrix.zero_()
                    pref_matrix[pref_edge_u, pref_edge_v] = p_e.to(dtype=pref_matrix.dtype)

                pref_violation_features = float(pref_projector.violation_features(p_e).item())
                pref_violation_cycle = float(pref_projector.violation_cycle(p_e).item())
```

---

### Where the soft DP loops sit in the overall training pipeline
Per iteration and per flight, `batch_sgd_pipeline_parallel_tsb_truellh.py` does:
- load thinned closures + amortized wind (`CLSR_*.pkl`, `WIND_*.pt`)
- infer tensor sizes \((K,\rho,\phi)\) from closures
- compute \(V_f\) (forward soft DP) and \(V_b\) (backward soft DP)
- compute expected link counts and gradients (MaxEnt IRL)
- aggregate across flights:
  - update common weights \(w\) (Adam on autograd parameters)
  - update preferences \(p\) (manual projected step)

That’s the complete methodological “spine”: **feasible state-transition graph → soft Bellman DP → marginals → expected-vs-empirical gradient → disentangled parameter updates**.
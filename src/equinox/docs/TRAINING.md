### Big picture (what this training code is doing)
You’re learning a **routing cost model** \(c_\theta(e, t)\) so that **observed flight routes** become likely under a **maximum-entropy (Boltzmann) distribution** over *all feasible routes*:

\[
p_\theta(\xi) \propto \exp\!\left(-\frac{C_\theta(\xi)}{\gamma}\right),\quad C_\theta(\xi)=\sum_{e\in \xi} c_\theta(e,t_e)
\]

- **\(\gamma\)** (temperature) controls how “peaky” the route distribution is:
  - small \(\gamma\) → almost “shortest-path” (hard min)
  - large \(\gamma\) → more exploration / entropy

The key trick is: **we never enumerate routes**. Instead we:
- precompute a **feasible state-transition graph** (TRES forward/backward + thinning)
- run **Bellman-soft dynamic programming** (soft value iteration) to get:
  - the “soft cost-to-come” \(V_f\)
  - the “soft cost-to-go” \(V_b\)
- combine those to compute **edge marginals** (expected link traversals) and a **gradient signal** that pushes expected usage toward empirical usage.

---

### 0) What files are “the pipeline”
There are two stages in the repo:

#### Pre-Training: deriving the base route graph for a city pair from historical data
- Pre-training utilities live under `src/equinox/training/prep/`.
- There is also a convenience runner in `src/equinox/training/prep_all.py` (note: it uses hard-coded paths and is meant as a local script rather than a general CLI).

#### Stage A — precompute *feasible state transitions* and *wind*
This is done by `src/equinox/training/tres_batch.py` using:
- forward TRES: `src/equinox/dp/trespass/tres_forward.py`
- backward TRES: `src/equinox/dp/trespass/tres_backward.py`
- thinning: `src/equinox/dp/trespass/thinning.py`
- wind amortization: wind model’s `get_average_tailwind_on_edges_knots`

This stage writes, per flight, files like:
- `FW_<flight_id>_<takeoff_ts>.pkl` (forward)
- `BW_<flight_id>_<takeoff_ts>.pkl` (backward closures)
- `CLSR_<flight_id>_<takeoff_ts>.pkl` (thinned closures; this is what training uses)
- `WIND_<flight_id>_<takeoff_ts>.pt` (tailwind per closure transition)

…and a dataset file:
- `tres_runs/all_routes_feasibly_snapped.csv` (snapped routes used as supervision)

The exact filenames the **training loop expects** are loaded here:

```369:407:/Volumes/CrucialX/project-equinox/src/equinox/training/batch_sgd_pipeline_parallel_tsb_truellh.py
def load_flight_tres_results(case_dir: str, flight_id: str, takeoff_timestamp: int) -> Tuple[List, List, List, torch.Tensor]:
    tres_dir = Path(case_dir) / "tres_runs"
    # ...
    for batch_dir in tres_dir.glob("batch*"):
        fw_file = batch_dir / f"FW_{flight_id}_{takeoff_timestamp}.pkl"
        bw_file = batch_dir / f"BW_{flight_id}_{takeoff_timestamp}.pkl"
        thinned_file = batch_dir / f"CLSR_{flight_id}_{takeoff_timestamp}.pkl"
        wind_file = batch_dir / f"WIND_{flight_id}_{takeoff_timestamp}.pt"
```

#### Stage B — learn cost model parameters (your requested training loop)
This is `src/equinox/training/batch_sgd_pipeline_parallel_tsb_truellh.py`:
- loads the per-flight precomputed artifacts above
- runs forward/backward **soft value iteration**
- runs a **gradient pass** (expected vs empirical counts)
- averages gradients over a batch of flights and updates model (Adam)

---

### 1) Stage A in detail: forward TRES → backward TRES → thinning → wind amortization

#### 1.1 Forward TRES: “what climb/cycle-time states can we reach?”
Forward TRES (`tres_forward`) walks forward in time from takeoff and records *reachable* climb states and *climb-related transitions*.

- **State (forward)** is effectively:
  - **waypoint** index \(n\)
  - **wall-clock bin** \(k\) (discretized absolute time)
  - **elapsed-since-takeoff bin** \(\varepsilon\) (called ETTO)
  - plus “continuous” metadata stored alongside: altitude and phase

Key points from `tres_forward`:
- builds time bins from takeoff time and `max_flight_duration_hours`
- builds ETTO bins from climb performance and `max_elapsed_time_since_takeoff_hours`
- iterates nodes in topological generations to enable batching (needs a DAG)
- calls `get_next_state_fw(...)` to propagate altitude/eta/phase with wind + performance
- stores transitions only for **CLIMB→CLIMB** and **CLIMB→CRUISE** (because the backward pass later uses these climb transitions as constraints)

```37:352:/Volumes/CrucialX/project-equinox/src/equinox/dp/trespass/tres_forward.py
def tres_forward(...):
    # ...
    topo_generations_node_ids = list(nx.topological_generations(graph))
    for generation_node_ids in tqdm(topo_generations_node_ids, desc="Topological Generations"):
        # batch inputs...
        alt_v_new_batch, eta_v_new_batch, phase_v_new_batch = get_next_state_fw(...)
        # ...
        if phase_u == CLIMB and phase_v in (CLIMB, CRUISE):
            transitions_list.append((u_node_idx_for_trans, eps_u_idx_for_trans, ..., v_node_idx, eps_v_idx, ...))
```

**Why do it this way?**
- **Physics first**: `get_next_state_fw` uses aircraft climb performance + wind to ensure reachability is kinematically plausible.
- **Discretized time**: wind is time-varying; without time bins you can’t associate a segment with “the wind at that time.”
- **DAG ordering**: With a DAG + time increasing, reachability is a one-pass DP over a sorted order rather than iterative search.

#### 1.2 Backward TRES: “from the destination, what can plausibly connect backwards?”
Backward TRES (`tres_backward`) starts from the destination and walks *backward* to identify **feasible state transitions** that could reach the goal, including descent/cruise and integrating climb feasibility via the forward transitions.

- **State (backward)** is explicitly documented as 4D:
  \[
  s=(\text{waypoint},\; k\_\text{wallclock},\; \rho,\; \phi)
  \]
  where **\(\rho\)** is a “remaining climb time bin” and **\(\phi\)** is phase.

- It initializes at the goal state (destination at landing time, descent, \(\rho=0\)), then processes nodes in reverse topological generations.

- It has *two propagation paths*:
  - **Path 1 (standard)** uses `get_next_state_bw(...)` to step back through CRUISE/DESCENT.
  - **Path 2 (climb-aware)** uses the **forward transitions_list** to inject climb feasibility and allow “CRUISE→CLIMB” switching near top-of-climb.

```34:427:/Volumes/CrucialX/project-equinox/src/equinox/dp/trespass/tres_backward.py
def tres_backward(...):
    # state is (waypoint, wall-clock bin, remaining climb bin, phase)
    # ...
    alt_u_std_batch, eta_u_std_ssm_batch, phase_u_std_batch = get_next_state_bw(...)
    # Path 1: standard propagation (CRZ/DES)
    # Path 2: transitions-based propagation (CLIMB focus)
```

**Why the forward+backward combination?**
- Forward alone gives you “can I get there from takeoff?” but doesn’t guarantee “and can I still reach the destination.”
- Backward alone gives you “can I get to the destination from here?” but needs extra structure to represent climb-phase feasibility.
- The combination yields a closure of **globally feasible** state transitions across climb/cruise/descent.

#### 1.3 Thinning: keep only transitions on *some path* origin → goal
Even after backward TRES, you may have a lot of dead-end state transitions. `thin_closures(...)` removes transitions whose states are not:
- reachable from any valid origin state, and
- able to reach any goal state

```6:93:/Volumes/CrucialX/project-equinox/src/equinox/dp/trespass/thinning.py
def thin_closures(source_node_idx: int, goal_node_idx: int, max_rho: int, G: nx.DiGraph, closures: list[tuple[...] ]):
    # build state-graph, compute descendants from origins and from goals (on reversed graph),
    # take intersection, and filter closures to those whose endpoints are both valid.
```

**Why this pruning matters**
- Soft value iteration and gradient computation cost scale with “number of transitions”.
- Thinning typically reduces that by removing unreachable branches or branches that can’t reach the goal.

#### 1.4 Wind amortization: precompute tailwind per *state transition*
This is what makes training fast: instead of re-interpolating ERA5 wind during learning, you store a tensor `avg_tailwind_knots` aligned with the thinned transitions order, then pass it into SVI + gradient.

A subtle but important detail: the backward TRES uses a **landing-time anchored wall-clock window**, so wind-time bin conversion must match that reference:

```258:279:/Volumes/CrucialX/project-equinox/src/equinox/training/tres_batch.py
# CRITICAL FIX: The 'k' time bins in thinned_transitions are from the backward pass,
# which uses estimated_landing_ssm - max_flight_duration_hours * 3600 as the time reference.
estimated_landing_ssm = datestr_to_seconds_since_midnight(flight_config.estimated_landing_time_str)
min_wall_clock_time_sec = float(estimated_landing_ssm - flight_config.max_flight_duration_hours * 3600)

avg_tailwind_knots = flight_components['wind_model'].get_average_tailwind_on_edges_knots(
    transitions=thinned_transitions,
    node_coords_deg=node_coords_deg,
    min_wall_clock_time_sec=min_wall_clock_time_sec,
    delta_t_wall_clock_sec=flight_config.delta_t_seconds,
    num_integration_steps=3
)
```

---

### 2) Stage B in detail: the training loop in `batch_sgd_pipeline_parallel_tsb_truellh.py`

#### 2.1 What is a “training iteration”?
One iteration:
- picks a **batch of flights** (fixed / random / sequential)
- for each flight:
  - load thinned transitions + wind
  - run forward SVI → get \(V_f\)
  - run backward SVI → get \(V_b\)
  - compute expected counts + gradient
- average gradients across the batch and update the global cost model

This is the core loop:

```975:1084:/Volumes/CrucialX/project-equinox/src/equinox/training/batch_sgd_pipeline_parallel_tsb_truellh.py
with ProcessPoolExecutor(max_workers=batch_config.num_workers) as executor:
    while iteration <= batch_config.max_iterations and not converged:
        # pick batch_idx (fixed/random/sequential)
        cost_model_state = components['cost_model'].state_dict()
        tasks = []
        for _, flight_data in batch_flights.iterrows():
            tasks.append((flight_data, components, batch_config, case_dir, cost_model_state, debug_this_flight))

        future_results = executor.map(_process_flight_wrapper, tasks)
        batch_results = list(tqdm(future_results, total=len(tasks), desc=f"Processing batch {batch_idx+1}"))

        # collect gradients, average by param name, optimizer.step()
```

#### 2.2 Per-flight work: “SVI + gradient”
Inside `process_single_flight` the flow is exactly:

```473:568:/Volumes/CrucialX/project-equinox/src/equinox/training/batch_sgd_pipeline_parallel_tsb_truellh.py
# 1. Load TRES results (needs CLSR + WIND)
forward_transitions, backward_transitions, thinned_transitions, avg_tailwind_knots = load_flight_tres_results(...)

# 2. Infer state tensor dimensions from thinned_transitions
num_time_bins_wall_clock = max_k_val + 1
num_rho_bins = max_rho_val + 1
num_phases = max_phase_val + 1

# 4. origin/goal indices from flight_data
origin_node_idx = components['node_to_idx'][flight_data['origin']]
goal_node_idx = components['node_to_idx'][flight_data['destination']]

# 6. forward + backward SVI (sequential inside the worker)
v_f = forward_soft_value_iteration(...)
v_b = backward_soft_value_iteration(...)

# 7. empirical_counts from snapped route string
empirical_counts = compute_empirical_counts_for_flight(...)

# 8. gradient pass
expected_counts, gradient, log_partition_z_tensor = backward_gradient_pass(...)
```

**Why compute \(V_f\) and \(V_b\) every iteration?**
- Because as \(\theta\) changes, the **costs change**, so the induced distribution over routes changes, so the partition function / marginals change.

**Why do it per flight?**
- Each flight has its own thinned transitions (time bins and feasible states depend on schedule and wind), so the DP is per-flight.

#### 2.3 Applying gradients safely across processes
The gradient is computed as a **flat vector** in the worker, then mapped to a **name→tensor** dict so that the main process can average and apply without relying on parameter ordering:

```300:330:/Volumes/CrucialX/project-equinox/src/equinox/training/batch_sgd_pipeline_parallel_tsb_truellh.py
def _vector_to_named_grads(grad_vector: torch.Tensor, model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        grads[name] = grad_vector[idx: idx + n].view_as(param).detach()
```

Then the main loop averages per-name and assigns `param.grad`:

```1040:1077:/Volumes/CrucialX/project-equinox/src/equinox/training/batch_sgd_pipeline_parallel_tsb_truellh.py
grad_sums = {name: torch.zeros_like(param, device=device, dtype=torch.float32) for name, param in trainable_named_params}
for grad_dict in gradient_queue:
    grad_sums[name] += grad_dict[name].to(device=device, dtype=torch.float32)

avg_grads = {name: g / denom for name, g in grad_sums.items()}
optimizer.zero_grad()
for name, param in trainable_named_params:
    param.grad = avg_grads[name].to(dtype=param.dtype)
optimizer.step()
```

**Why this design?**
- Multiprocessing + PyTorch models can easily drift in parameter ordering across refactors; name-keyed grads make updates robust.

#### 2.4 Edge preference update (lin_disent only)
When edge preferences are enabled, the worker also returns per-edge gradients, and the main process applies a **projected** update to `preference_matrix_p`:

```1687:1716:/Volumes/CrucialX/project-equinox/src/equinox/training/batch_sgd_pipeline_parallel_tsb_truellh.py
pref_grad_avg = pref_grad_sum / float(len(pref_grad_queue))
pref_matrix = components["cost_model"].preference_matrix_p
p_e = pref_matrix[pref_edge_u, pref_edge_v].to(device=device, dtype=pref_projector.X.dtype)
pref_grad_proj = pref_projector.project(pref_grad_avg)
p_e = p_e - batch_config.pref_learning_rate * pref_grad_proj
p_e = pref_projector.project(p_e)
pref_matrix.zero_()
pref_matrix[pref_edge_u, pref_edge_v] = p_e.to(dtype=pref_matrix.dtype)
```

The projector is built once from a deterministic edge list and D-weighted features (using dataset-level empirical counts), so preference updates stay disentangled from the common linear feature space.

---

### 3) Bellman-soft value iteration: what it computes and why it works

#### 3.1 Backward soft value iteration: soft “cost-to-go”
Backward SVI computes \(V_b(s)\): the soft minimum expected cost from state \(s\) to the goal:

\[
V_b(s) = -\gamma \log \sum_{s\to s'} \exp\!\left(-\frac{c(s,s') + V_b(s')}{\gamma}\right)
\]

In code, the soft-min merge is exactly the log-sum-exp form:

```201:244:/Volumes/CrucialX/project-equinox/src/equinox/dp/trespass/amorwin/backward_svi_log_cost_temp.py
val_from_v = V_s_v + cost_uv
old_V_u = V_val[u_idx, k_u, rho_u, phase_u]
new_V_u = -gamma * torch.logaddexp(-old_V_u / gamma, -val_from_v / gamma)
V_val[u_idx, k_u, rho_u, phase_u] = new_V_u
```

**Intuition**
- If a state has two possible next steps, one cheap and one expensive:
  - hard shortest path picks the cheapest
  - soft-min blends them, but exponentially downweights the expensive one
- That soft-min is exactly what you need for maximum-entropy route distributions.

#### 3.2 Forward soft value iteration: soft “cost-to-come”
Forward SVI computes the forward log-partition to reach each state:

- It maintains \(L(s)=\log Z(s)\) where
  \[
  Z(s)=\sum_{\text{paths to }s}\exp\!\left(-\frac{C(\text{path})}{\gamma}\right)
  \]
- Then returns \(V_f(s)=-\gamma L(s)\), which is the soft-min cost-to-come.

Core update:

```260:307:/Volumes/CrucialX/project-equinox/src/equinox/dp/trespass/amorwin/forward_svi_log_temp.py
L_s_u = L_val[u_idx, k_u, rho_u, phase_u]
# ...
cost_uv = float(cost_model(...).item())
a_u = L_s_u - cost_uv / gamma
old_L_v = L_val[v_idx, k_v, rho_v, phase_v]
new_L_v = torch.logaddexp(old_L_v, a_u)
L_val[v_idx, k_v, rho_v, phase_v] = new_L_v
V_soft = -gamma * L_val
```

**Why log-space?**
- Directly summing \(\exp(-C/\gamma)\) underflows quickly when costs are large.
- logaddexp is stable and makes this scalable.

---

### 4) Expected counts + gradient: how MaxEnt learning is implemented

#### 4.1 “Partition function” and edge probabilities
Given the soft values, the code forms a per-transition probability (marginal) roughly like:

\[
p(s\to s') \propto \exp\!\left(-\frac{V_f(s) + c(s,s') + V_b(s') - V_Z}{\gamma}\right)
\]

where \(V_Z\) is the soft cost from origin to goal (a cost-domain analogue of \(\log Z\)).

You can see the probability computation here:

```65:105:/Volumes/CrucialX/project-equinox/src/equinox/dp/trespass/amorwin/backward_gradient.py
v_origin = V_b[origin_node_idx]
non_inf_v = v_origin[v_origin != torch.inf]
log_partition_z = -gamma * torch.logsumexp(-non_inf_v / gamma, dim=0)

log_p_transition = (-v_f_u - cost_uv - v_b_v + log_partition_z) / gamma
p_transition = torch.exp(log_p_transition)
link_traversal_likelihoods[u_idx, v_idx] += p_transition
```

**Intuition**
- \(V_f(s)\) says “how hard is it (softly) to get to \(s\)”
- \(V_b(s')\) says “how hard is it (softly) to finish from \(s'\)”
- add the local edge cost in between
- normalize by the global “soft best” origin→goal score

#### 4.2 Expected vs empirical link usage
- **Empirical counts**: counts the edges in the snapped route string:

```219:248:/Volumes/CrucialX/project-equinox/src/equinox/training/batch_sgd_pipeline_parallel_tsb_truellh.py
waypoints = waypoints_str.split()
for i in range(len(waypoints) - 1):
    from_idx = node_to_idx[from_wp]
    to_idx = node_to_idx[to_wp]
    empirical_counts[from_idx, to_idx] += 1.0
```

- **Expected counts**: sum of probabilities across all state transitions corresponding to the same waypoint link \((u,v)\) (because many time/phase bins can map to the same \((u,v)\)).

#### 4.3 Gradient shape: “push expected toward empirical”
For each waypoint link \(e=(u,v)\), you get:
- a scalar mismatch \((N_\text{emp}(e) - N_\text{exp}(e))\)
- times a cost gradient \(\nabla_\theta c_\theta(e)\)

The implementation does it link-by-link (memory efficient) and accumulates:

```205:211:/Volumes/CrucialX/project-equinox/src/equinox/dp/trespass/amorwin/backward_gradient.py
n_empirical = empirical_counts[u_idx, v_idx]
n_expected = link_traversal_likelihoods[u_idx, v_idx]
total_log_likelihood_grad += (link_grad / gamma) * (n_empirical - n_expected)
```

**Why a 2-pass + link grouping design?**
- The “state transition graph” is huge (time × climb bins × phases).
- Building an 8D tensor of all transition gradients would explode memory.
- Grouping by link gives the core MaxEnt gradient signal while keeping memory bounded.

---

### 5) Cost model + learnable parameters (what the optimizer is updating)
The current training pipeline expects `cost_model_version: "lin_disent"` and uses `CostLinearDisentangled`.

It splits cost into:

\[
c(e,t) = \underbrace{w^\top x(e,t)}_{\text{common linear cost}} \;+\; \underbrace{p(e)}_{\text{per-edge preference}}
\]

Where the per-edge feature vector matches the model implementation:

- **bias**: \(1\)
- **ac\_dist**: \(AC(e)\cdot d(e) / 100\)
- **time**: \(60 \cdot d(e) / (\text{cruise\_speed\_kts} + \text{tailwind}_e)\)

See the cost model:

```14:76:/Volumes/CrucialX/project-equinox/src/equinox/cost/cost_linear_disentangled.py
class CostLinearDisentangled(nn.Module):
    """
    Common cost: c_common(e) = x(e)^T w
    Total cost: c(e) = c_common(e) + p(e)
    """
    # Feature definition (fixed per waypoint-edge):
    # - bias = 1
    # - ac_dist = AC(e) * d(e) / 100.0
    # - time = 60.0 * dist / (cruise_speed_kts + tailwind_e)
```

**What actually gets updated**
- **Common weights** `w` are updated by SGD/Adam from the MaxEnt gradient (main optimizer).
- **Preferences** `p(e)` live in `preference_matrix_p` and are updated **separately** via a projected gradient step (see `GaugeFixedPreferenceProjector` in `preferences/disentanglement.py`).
- `alpha_pref_reg` (if non-zero) adds L2 regularization to the preference update.

**Why the projector exists**
- Preferences are disentangled from the linear feature space by enforcing D-weighted orthogonality to the common feature columns.
- The projector also fixes a cycle-gauge so per-edge offsets remain identifiable (prevents adding arbitrary node potentials).

---

### 6) “How the moving parts are glued together”: config locations + key settings

#### 6.1 Where configuration comes from
- **Case YAML** (graph + aircraft + discretization + cost model):
  - training defaults to `--config <case_dir>/default.yaml`
  - the YAML is expected to live in the `--case-dir` you pass to the training script

- **RunConfiguration** loader and component initializer:
  - loads graph/matrices, builds node mappings, initializes cost model:

```11:127:/Volumes/CrucialX/project-equinox/src/equinox/config.py
class RunConfiguration:
    graph_file_path: str = None
    distances_file_path: str = None
    charges_file_path: str = None
    # ...
    cost_model_version: str = None
    gamma: float = None
    # ...
    def initialize_all_components(self, cost_model_version: str = None, ...):
        G, node_to_idx, idx_to_node, node_coords_deg = self.load_graph()
        cost_model = self.initialize_cost_model(len(G.nodes()), cost_model_version)
        dist_matrix = self.load_distance_matrix()
        ac_matrix = self.load_charges_matrix()
        return {...}
```

- **Training hyperparameters** come from the case YAML and are then passed into `BatchLearningConfig` (CLI flags can override). The training loop uses **`batch_config.gamma`** for SVI/gradient.

#### 6.2 Brief meaning of the most important YAML fields (case config)
From your `<case_dir>/default.yaml`:
- **`graph_file_path`**: DAG route graph (`.gml`) used for reachability and DP ordering.
- **`distances_file_path`**: NxN waypoint distance matrix used in the cost model.
- **`charges_file_path`**: NxN airspace charges matrix used in the cost model.
- **`aircraft_model`, `cruise_altitude_ft`, `cruise_speed_kts`**: used to build `Performance(...)` tables for climb/descent timing.
- **`delta_t_seconds`**: wall-clock bin size (e.g., 600s) → time resolution of states and wind sampling.
- **`etto_delta_t_seconds`**: “elapsed-since-takeoff” bin size (e.g., 30s) → climb-time resolution.
- **`max_flight_duration_hours`**: time window length for backward pass / bins.
- **`max_elapsed_time_since_takeoff_hours`**: cap on elapsed-time bins in forward pass.
- **`climb_phase_switch_allowance_climb_time_bins`**: tolerance window around TOC where cruise→climb switching is allowed in backward TRES.
- **`cost_model_version`**: should be `"lin_disent"` for the current training pipeline.
- **`common_weights`, `preference_weight`, `alpha_pref_reg`**: parameters used by the linear disentangled cost model.
- **`disable_config_wind_model`**: if true, `RunConfiguration.initialize_all_components` won’t load a wind model (training uses precomputed WIND files anyway).

#### 6.3 Brief meaning of the key training fields and CLI overrides
From `BatchLearningConfig` + CLI:
- **`training_batch_size`** (YAML) → `batch_size`: flights per SGD update.
- **`common_features_learning_rate`** (YAML) → `learning_rate`: optimizer LR for common weights.
- **`preference_feature_learning_rate`** (YAML) → `pref_learning_rate`: LR for per-edge preferences.
- **`preference_projection_ridge`** (YAML): ridge term for the preference projector.
- **`gamma`** (YAML) → `batch_config.gamma`: temperature for SVI + gradient.
- **`max_iters`** (YAML) / `--max-iters`: SGD iteration cap.
- **`convergence_threshold`**: stops when gradient L2 norm < threshold.
- **`checkpoint_interval`**: save model/optimizer/training history periodically.
- **`num_workers`**: parallel workers for per-flight processing.
- **`--debug-single-process`**: run without multiprocessing (easier debugging).
- **`--randomize`, `--batch-shuffling`, `--random-seed`, `--fixed-batch-index`**: batch selection strategy.
- **`--disable-edge-preference`**: freeze `preference_matrix_p` at zero (lin_disent only).

---

### 7) A tiny intuitive example (no time bins, just to understand the learning signal)
Imagine two candidate routes from O→G:
- Route A: O→A→G with cost 2
- Route B: O→B→G with cost 3  
With \(\gamma=1\):
\[
\frac{p(A)}{p(B)}=\exp(-(2-3))=\exp(1)\approx 2.7
\]
So the model strongly prefers route A.

If your data (empirical counts) repeatedly uses edges on route B, then \(N_\text{emp}(O,B)\) and \(N_\text{emp}(B,G)\) are high, but the expected counts from the model \(N_\text{exp}\) will be low. The gradient term \((N_\text{emp}-N_\text{exp})\) becomes positive, and SGD will push **down the costs** on those B-edges (or push up competing edges), making route B more likely next iteration.

That’s exactly what `backward_gradient_pass` implements at scale, with TRES ensuring feasibility and SVI supplying marginals.

---

### 8) End-to-end “trace map” (one flight, one iteration)
- **Precompute** (`tres_batch.py`):
  - forward reachability + climb transitions → `FW_*.pkl`
  - backward closures → `BW_*.pkl`
  - thinning → `CLSR_*.pkl`
  - wind amortization aligned to the backward time window → `WIND_*.pt`
  - snap observed route onto feasible edges → `all_routes_feasibly_snapped.csv`
- **Train** (`batch_sgd_pipeline_parallel_tsb_truellh.py`):
  - load `CLSR_*` + `WIND_*`
  - compute \(V_f\) (forward SVI) and \(V_b\) (backward SVI)
  - compute expected link counts + gradient via MaxEnt
  - average gradients across flights → Adam step → checkpoint/logs

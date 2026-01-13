# Plan Revision

### 1) Review of `refact.md` (holes / flaws + concrete fixes)

### Preference disentanglement refactor

- **Missing: “preferences must be in *all* DP cost calls” (not just gradients)**  
  Your plan sketches `total_cost = common_cost + pref`, but it doesn’t explicitly call out that **`p` must be used inside `forward_soft_value_iteration`, `backward_soft_value_iteration`, and `backward_gradient_pass`**; otherwise \(V_f,V_b\) and \(n_\text{exp}\) won’t correspond to the same model you’re training.  
  Evidence: SVI currently calls `cost_model(...)` directly inside the per-transition loop:

```272:286:/Volumes/CrucialX/project-equinox/src/equinox/dp/trespass/amorwin/forward_svi_log_temp.py
        # b) Retrieve pre-computed tailwind for this transition
        tailwind_knots = avg_tailwind_knots_per_transition[original_index]
 
        # c) Compute cost_uv in float64
        #   - We force distance_matrix_d, airspace_charge_matrix_ac, tailwind_knots into float64,
        #     so that cost_model runs internally in double precision (if it supports it).
        edge_u_indices = torch.tensor([u_idx], device=device, dtype=torch.long)
        edge_v_indices = torch.tensor([v_idx], device=device, dtype=torch.long)
 
        cost_uv_tensor = cost_model(
            (edge_u_indices, edge_v_indices),
            distance_matrix_d.to(dtype=torch.float64),
            airspace_charge_matrix_ac.to(dtype=torch.float64),
            tailwind_knots.to(dtype=torch.float64)
        )
```

- **Big conceptual gap: the math guide assumes “common cost = \(Xw\)”, but your code’s common cost is a nonlinear PLM**  
  `preference_disentanglement.md`’s identifiability argument is for \(c = Xw + p\). If you keep `CostRev4` / PLMs as the “common model”, then projecting \(p\) against some handcrafted \(X\) is still a useful *interpretability gauge*, but it’s **not the same identifiability guarantee** unless you define what “\(Xw\)” means in your actual parameterization.  
  “Plug”: decide (and document in the plan) which of these you’re doing:
  - **Option 1 (strictly matches guide)**: make common cost explicitly linear \(Xw\) (or at least linear-in-parameters with a fixed known feature map), then disentangle.
  - **Option 2 (pragmatic)**: keep PLM common model, and treat the projection as “preferences have zero covariance with chosen features” (not full identifiability vs PLM params).

> Decision: make the cost explicitly linear, but create a new cost model file instead (do not override the cost_rev4.py). This keeps thing simple, and we can revise it if needed.

- **Where does `p` live, and how do workers get the up-to-date `p` each iteration?**  
  The plan says “don’t make `p` a giant autograd parameter” (agree), but doesn’t fully specify the mechanics for multiprocessing. Right now **every flight task receives a pickled `cost_model_state`** each iteration:

```1044:1066:/Volumes/CrucialX/project-equinox/src/equinox/training/batch_sgd_pipeline_parallel_tsb_truellh.py
            # Get current model state to be used by all flights in this batch
            cost_model_state = {k: v.detach().cpu() for k, v in components['cost_model'].state_dict().items()}
 
            # Process flights in the batch in parallel
            batch_results = []
 
            # Prepare arguments for each flight in the batch
            tasks = []
            flight_list = list(batch_flights.iterrows())
            for i, (_, flight_data) in enumerate(flight_list):
                # Set debug flag for only the last flight in the batch
                debug_this_flight = (i == len(flight_list) - 1)
                flight_payload = {
                    "flight_id": flight_data["flight_id"],
                    "takeoff_timestamp": int(flight_data["takeoff_time"]),
                    "origin": flight_data["origin"],
                    "destination": flight_data["destination"],
                    "route": flight_data["route"],
                }
                tasks.append((flight_payload, cost_model_state, debug_this_flight))
 
            # Use the executor to run flight processing in parallel
            future_results = executor.map(_process_flight_wrapper, tasks)
```

  “Plug”: pick one and bake it into the plan:
  - **Buffer-in-model approach**: register `p` as a **buffer** (so it’s in `state_dict` but not in `parameters()`), and accept the serialization cost; or
  - **Worker-context approach**: keep `p` in `_WORKER_CONTEXT` (like `dist_matrix`, `ac_matrix`) and update once per iteration/batch (to avoid re-sending it per flight); or
  - **Shared-memory approach** if `p` becomes big.

> Decision: we will follow the buffer-in-model approach (first one).

- **Projection numerics are underspecified (rank/conditioning + scaling)**  
  You already note “solve a \(d\times d\) system”; the missing bits are:
  - **What if \(X^\top D X\) is singular / ill-conditioned** on empirical support? (very plausible if features are highly correlated or biased)  
    “Plug”: plan should explicitly include **ridge** (or pinv) and a diagnostic/alert when conditioning is bad.

> Agree, let's include it.

  - **Feature scaling**: if columns of \(X\) have very different magnitudes, the projection can be unstable.  
    “Plug”: normalize columns of \(X\) under the \(D\)-weighted inner product (mean/variance under \(d\)).

> Agree, let's implement this too. But leave the room for custom renormalization weights manual entry later as we might need them.

- **Projected updates + Adam/momentum**  
  The plan says “SGD/Adam” for `p` and then “reproject”. With Adam (or momentum), projecting only `p` but not the optimizer’s moments typically produces odd dynamics.  
  “Plug”: either commit to **plain SGD** for `p`, or move to the **re-parameterization \(p=P_{\perp,D}u\)** so the optimizer lives in unconstrained `u`, or explicitly project the update direction/moments too.

> Let's switch to plain SGD.

- **Edge universe \(E\) isn’t nailed down**  
  You should explicitly define \(E\) as one of:
  - edges of the base waypoint graph (`nx` edges),
  - all finite entries of distance matrix,
  - union of links that appear in thinned transitions across all flights.
  This affects **`m`**, `d`, the mapping, and whether “missing in thinned transitions” edges are learnable.  
  For the default graph in `data/graph/LEMD_EGLL_2023_04_01.gml`, it’s **566 nodes, 5028 directed edges**, so the dense `edge_id_of_uv` lookup is totally fine memory-wise; but the plan should still define what “edge” means.

> Agree, let's choose the edges of the base waypoint graph.

- **Sign consistency looks OK, but make it explicit for preferences too**  
  Your plan’s \(g_p(e)=(n_\text{emp}-n_\text{exp})/\gamma\) matches the code’s “negative log-likelihood gradient is empirical minus model” convention:

```205:211:/Volumes/CrucialX/project-equinox/src/equinox/dp/trespass/amorwin/backward_gradient.py
        # Update the total gradient using the gradient of the link's cost
        n_empirical = empirical_counts[u_idx, v_idx]
        n_expected = link_traversal_likelihoods[u_idx, v_idx]
        
        # Gradient of negative log-likelihood is E_empirical - E_model
        total_log_likelihood_grad += (link_grad / gamma) * (n_empirical - n_expected)
```

### Gradient-derivation speedup section

- **Core idea is correct and matches current “semi-gradient” semantics** (detach weights, backprop only through costs). Biggest hole is *fidelity details*:
  - **You currently drop tiny weights** (`if weights[i].item() > 1e-9`) and have an **argmax fallback** when normalizer is tiny. The vectorized plan mentions the fallback, but not the weight-threshold behavior. Decide if you want exact equivalence or a smoother “use all weights” version.

> Let's just keep things as they are for now.

  - **Chunking should be treated as mandatory**, not optional: avoid allocating `T`-sized float64 tensors for everything at once if `T` is large.

> Agree. 

- **The SVI dtype-cast is a bigger “free win” than the plan implies**  
  Your optional note is correct, but there’s an extra nuance: `CostRev4` forces tailwind back to float32, so repeated float64 casting is both costly and may not even accomplish “double precision cost”:

```169:180:/Volumes/CrucialX/project-equinox/src/equinox/cost/cost_rev4.py
        ac_dist_product_batch = ac_e_batch * dist_e_batch / 100.0
        # ac_dist_product_batch = dist_e_batch # for testing, independent of airspace charge
        # Ensure tailwind_values_w is on the correct device and dtype for PLM input
        tailwind_tensor_batch = tailwind_values_w.to(device=self.device, dtype=torch.float32)
 
        cost_component_ac_dist = self.plm_ac_dist(ac_dist_product_batch) # PLM (distance x airspace charge / 100.0)
        component_dist = dist_e_batch # distance only, in nm
        distance_due_to_tailwind = component_dist / (450.0) * tailwind_tensor_batch # >0 if tailwind, <0 if headwind; nautical miles
        cost_component_wind = self.plm_wind(distance_due_to_tailwind) # PLM (distance due to tailwind)
```

  “Plug”: either (a) stop doing float64 casts in SVI/gradient passes and keep value iteration in float64 only where needed, or (b) refactor cost models to *actually* run float64 end-to-end if that’s the intent.

  > Agree, let's stop doing float64 casts.

---

### 2) Questions / assumptions I’d want to clarify

- **Scope of models**: which `cost_model_version` are you targeting for the refactor (3 vs 4 vs 4lite vs 4rb1)? Do you need it to work across versions, or is it OK to introduce a new “decomposed” cost model class?

> Just introduce a new cost model since we decided to change the features to linear as mentioned above.

- **Definition of an “edge” for preferences**: is \(E\) the base waypoint graph edges, or “any (u,v) that appears in a thinned transition”, or “any finite distance entry”?

> The waypoint-to-waypoint graph.

- **How big can graphs get in your real runs?** (Default is 566 nodes / 5028 edges.) This decides whether dense `edge_id_of_uv` and/or dense count matrices are always fine. 

> Yes, roughly the same size every time. So I guess we can go with dense count matrices.

- **“Exact match” requirement for the accelerated gradient**: do you want the refactor to replicate current heuristics (tiny-weight skip + argmax fallback) exactly, or are small behavioral changes acceptable in exchange for speed/smoother gradients?

> Small behavioral changes are acceptable, but please be explicit about the changes, and make sure the results are not affected (too much). But stability is most important.
Got it. I’ll (a) read `GaugeFix.md` plus the current projection/gauge-fix code in `batch_sgd_pipeline_parallel_tsb_truellh.py` and `preferences/disentanglement.py`, (b) map the exact linear system you’re solving, then (c) recommend the numerically safest solve strategy for \(m\!\approx\!5000, d\!=\!3\) and lay out a concrete refactor plan with target functions, data flow, and solver choices (all projection living in `disentanglement`).

[13 tools called]

### (1) Most stable projection solve (m≈5000, d=3)

- **Recommendation (most stable in practice)**: **Schur complement that reduces to an SPD Laplacian solve + a tiny \(d\times d\) solve**, i.e. the “numerically nicer” weighted projection in `GaugeFix.md` (§3.2). It avoids the **symmetric-indefinite** KKT system and keeps the hard part as an **SPD** solve.  
  This is exactly the scheme described here:

```120:190:/Volumes/CrucialX/project-equinox/GaugeFix.md
## 3) How to enforce both constraints: projection onto the intersection
...
### 3.2 A numerically nicer special case: take $W=D_\epsilon$ and project in the same metric
...
\operatorname{Proj}(v)= v - X\alpha - B^\top\psi
...
\begin{bmatrix}
X^\top W X & X^\top W B^\top\\
B W X & B W B^\top
\end{bmatrix}
...
> This is what we are going to implement: $D_\epsilon$.
```

- **Why it’s more stable than full KKT**:
  - **Full KKT factorization** is a **saddle-point / indefinite** system (needs LDLᵀ with pivoting to be robust). In Python stacks it usually ends up as generic sparse LU, which is typically fine but **less numerically forgiving** when near-singularities appear (nullspaces, weakly connected graphs, tiny weights).
  - **Schur/Laplacian route** makes the big solve **SPD** (weighted Laplacian \(L=BWB^\top\) with a gauge pin), so you can use **Cholesky/LU on SPD** or **PCG** reliably.
  - With **\(d=3\)**, you can structure the solve so each projection costs **only \((d+1)\)** Laplacian solves plus a **\(3\times3\)** dense solve (very stable).

- **One crucial stability condition**: \(L=BWB^\top\) must be **nonsingular**.
  - Do this by **pinning a potential gauge**: remove one node row (“\(\psi(\text{ref})=0\)”) *per connected component* (or add a tiny ridge to \(L\) as a fallback). `GaugeFix.md` already notes removing a row (§3.1, “goal row removed”), but if the graph has >1 undirected connected component you need **one pinned node per component** for SPD.

### (2) Concrete refactor plan (all projection in `disentanglement.py`)

#### A) What exists today (feature-only projection)
`PreferenceProjector` currently enforces only \(X^\top Dp=0\) via \(p \leftarrow p - X(X^\top D X)^{-1}X^\top Dp\):

```192:252:/Volumes/CrucialX/project-equinox/src/equinox/preferences/disentanglement.py
class PreferenceProjector:
...
    def project(self, v_e: torch.Tensor) -> torch.Tensor:
...
        rhs = self.X.t().matmul(self.d_e * v_e)
...
        return v_e - self.X.matmul(alpha)
```

and the training pipeline constructs/uses it here:

```1198:1256:/Volumes/CrucialX/project-equinox/src/equinox/training/batch_sgd_pipeline_parallel_tsb_truellh.py
...
        X_norm, feature_means, feature_scales, manual_scales = d_weighted_normalize_features(
            X_raw,
            d_e,
            bias_index=0,
        )
        pref_projector = PreferenceProjector(
            X_norm,
            d_e,
            ridge=batch_config.pref_projection_ridge,
...
            p_e = pref_projector.project(p_e)
...
```

```1484:1567:/Volumes/CrucialX/project-equinox/src/equinox/training/batch_sgd_pipeline_parallel_tsb_truellh.py
...
                pref_projector = PreferenceProjector(
                    X_norm,
                    d_e,
                    ridge=batch_config.pref_projection_ridge,
...
                pref_grad_proj = pref_projector.project(pref_grad_avg)
                p_e = p_e - batch_config.pref_learning_rate * pref_grad_proj
                p_e = pref_projector.project(p_e)
...
```

#### B) Target design (GaugeFix.md intersection projection)
Implement **one projector** in `src/equinox/preferences/disentanglement.py` that projects onto:

\[
X^\top W p = 0,\qquad BWp=0,\qquad W=\mathrm{diag}(d_e+\varepsilon)
\]

using the §3.2 weighted formulation (`GaugeFix.md`).

**Proposed API (in `disentanglement.py`)**
- **`class GaugeFixedPreferenceProjector`** (new; this becomes the only projector the pipeline uses)
  - **`__init__(edge_u, edge_v, num_nodes, w_e, *, ref_nodes=None, laplacian_ridge=0.0, feature_ridge=...)`**
    - Builds the incidence operator \(B\) for the fixed edge ordering.
    - Builds and factorizes \(L = BWB^\top\) once (after removing pinned rows / one per component).
  - **`update_features(X: torch.Tensor)`**
    - Recomputes the small blocks that depend on \(X\) (because your pipeline rebuilds the time feature each iter):
      - \(A = X^\top W X\) (size \(3\times3\))
      - \(C = BWX\) (size \((n-1)\times 3\))
      - and precomputes \(Z = L^{-1}C\) via **3 Laplacian solves**
      - forms \(A_{\text{eff}} = A - C^\top Z\) (still \(3\times3\)) and factorizes it.
  - **`project(v_e: torch.Tensor) -> torch.Tensor`**
    - Uses the cached Laplacian factorization + \(A_{\text{eff}}\) to compute \((\alpha,\psi)\) and returns:
      - \(p = v - X\alpha - B^\top\psi\).
  - **Diagnostics**:
    - **`violation_features(p)`** = \(\|X^\top W p\|\)
    - **`violation_cycle(p)`** = \(\|BWp\|\)

**Core solve strategy inside `project` (stable Schur form)**
- Factorize \(L\) once.
- Each time:
  - Solve \(z_2 = L^{-1}(BWv)\) (1 Laplacian solve)
  - Solve \(\alpha\) from \(A_{\text{eff}}\alpha = X^\top W v - C^\top z_2\) (tiny \(3\times3\))
  - Set \(\psi = z_2 - Z\alpha\)
  - Return \(p=v - X\alpha - B^\top\psi\)

This is the stable “Laplacian solver + tiny dense solve” approach.

#### C) Pipeline changes (`batch_sgd_pipeline_parallel_tsb_truellh.py`)
Keep the pipeline responsible for **building features** \(X\) and weights \(d_e\), but move **all projection math** to the new projector.

- **Initialization block (current lines ~1198–1256)**:
  - Compute **weights** \(w_e = d_e + \varepsilon\) (choose \(\varepsilon>0\) as in `GaugeFix.md`).
  - Build `GaugeFixedPreferenceProjector(edge_u_cpu, edge_v_cpu, num_nodes, w_e, ...)`.
  - Call `pref_projector.update_features(X_norm)` once.
  - Replace `PreferenceProjector.project(p_e)` with `pref_projector.project(p_e)`.

- **Per-iteration preference update (current lines ~1484–1567)**:
  - Replace “rebuild `PreferenceProjector(...)`” with **`pref_projector.update_features(X_norm)`** (so the Laplacian factorization is reused).
  - Keep the same training logic:
    - `pref_grad_proj = pref_projector.project(pref_grad_avg)`
    - `p_e = p_e - lr * pref_grad_proj`
    - `p_e = pref_projector.project(p_e)`
  - Extend logging:
    - log **both** constraint violations (features + cycle), not just `X^T D p`.

That’s the minimal refactor that matches your constraint: **all projection implemented in `disentanglement.py`**, while the pipeline remains the orchestrator that supplies \(X\), \(d_e\), and the edge ordering.
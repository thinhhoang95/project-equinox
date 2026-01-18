# Hodge decomposition Gauge
Calculus on MDPs: Potential Shaping as a Gradient
Erik Jenner,1, 2 Herke van Hoof,1 Adam Gleave2

HODGE LAPLACIANS ON GRAPHS
LEK-HENG LIM†
https://arxiv.org/pdf/1507.05379
## 1) What is well-defined in the document (given $c$)

### Decomposition identifiability
You correctly note the gauge:
$$
Xw+p = X(w+\Delta) + (p - X\Delta).
$$
Imposing
$$
X^\top p = 0
$$
*does* fix this gauge **provided** $X$ has full column rank.

**Claim (correct):** If $\operatorname{rank}(X)=d$, then for a fixed $c$ there is a **unique** pair $(w,p)$ satisfying $c=Xw+p$ and $X^\top p=0$.

### If $X$ is rank-deficient
Let $\operatorname{rank}(X)=r<d$. Then $p$ is still forced into $\operatorname{col}(X)^\perp$ and remains unique **given $c$**, but $w$ is not:

- $p$ is unique
- $w$ is identifiable only up to $\operatorname{null}(X)$:
  $$
  w \sim w + \Delta,\quad \Delta\in \operatorname{null}(X).
  $$

So the document’s “formal identifiability guarantee” really needs the explicit assumption $\operatorname{rank}(X)=d$ if you care about $w$ as a parameter (not just $Xw$).

---

## 2) What remains ambiguous: $c$ is not identifiable in MaxEnt IRL (policy-equivalent gauges)

Even if $(w,p)$ is unique **given** $c$, MaxEnt IRL generally cannot identify $c$ uniquely from demonstrations because multiple cost functions induce the **same policy / same trajectory distribution**.

### 2.1 Potential-based shaping gauge (the big missing one)
In your setting (directed graph, actions = edges, fixed goal $g$), define any potential function $\psi$ on states with $\psi(g)=0$ and transform edge costs by
$$
c'_e \;=\; c_e + \psi(\text{tail}(e)) - \psi(\text{head}(e)).
$$

Then if $V$ satisfies your soft Bellman equation, define
$$
V'(s)=V(s)+\psi(s).
$$

You can verify:
- $V'$ satisfies the same soft Bellman recursion with $c'$
- and the induced Boltzmann policy is unchanged:
  $$
  \pi'(e\mid s)=\pi(e\mid s).
  $$

**Consequence:** the MaxEnt likelihood is identical under $(c,V)$ and $(c',V')$. Therefore, **$c$ is only identifiable up to addition of “gradient fields” on the graph** (range of the incidence transpose).

This is the analogue of the standard IRL non-identifiability: rewards/costs are identifiable only up to shaping (and constants in some cases).

> This ambiguity is independent of your $X^\top p=0$ constraint; it acts on $c$ itself.


# Big Plan

Your current $X^\top Dp=0$ construction is fixing the **feature–residual gauge** (how much of a *given* $c$ is attributed to $Xw$ versus $p$). The potential-shaping ambiguity is a different gauge acting on $c$ itself, and the clean way to stop the residual from “soaking up” shaping is exactly what you wrote: enforce a **cycle-space constraint** like $BWp=0$.

The reconciliation is:

- $X^\top Dp=0$  ⟹  **“$p$ contains no component explainable by $X$ (on empirical support)”**
- $BWp=0$        ⟹  **“$p$ contains no potential/gradient component (shaping)”**

So you want
$$
p \in \ker\!\big(X^\top D\big)\ \cap\ \ker\!\big(BW\big).
$$

Below is an end-to-end algorithm that learns $w$ and $p$ under both constraints, without needing $m\times m$ matrices.

---

## 1) Model + objective

Let $m=|E|$, $n=|S|$. You model edge costs as
$$
c \;=\; Xw + p,
$$
with constraints
$$
X^\top Dp=0,\qquad BWp=0.
$$

For MaxEnt over $o\to g$ paths, the negative log-likelihood per demo can be written as
$$
J(c)= c^\top \hat n_{\mathcal D} + \log Z(c),
$$
and the core gradient is the standard “empirical minus expected” occupancy:
$$
\nabla_c J(c)=\hat n_{\mathcal D}-\mathbb{E}_{c}[n(\tau)].
$$

Add regularization (strongly recommended because $p\in\mathbb{R}^m$ is high-dimensional), e.g.
$$
J(w,p)= J(Xw+p) + \frac{\lambda_w}{2}\|w-w_0\|_2^2 + \frac{\lambda_p}{2}\|p\|_{R}^2,
$$
where $R$ could be $I$, or $D$, or $(D+\epsilon I)$.

---

## 2) Choosing $W$ (important practical detail)

If you want the shaping-gauge removal to also be “empirically supported”, a very consistent choice is
$$
W := D_\epsilon := \operatorname{diag}(\hat n_{\mathcal D}+\epsilon),
$$
with a small $\epsilon>0$ so that:

- $BWp=0$ is well-posed (avoids singular weighted Laplacians when some edges have $0$ counts),
- you still emphasize edges actually present in data.

If you instead want a *global* gauge fix (not support-weighted), use $W=I$.

---

## 3) How to enforce both constraints: projection onto the intersection

The simplest end-to-end implementation is **projected gradient** in $(w,p)$:

- update $w$ with an unconstrained step
- update $p$ with a step **then project** back onto
  $$\{p:\ X^\top Dp=0,\ BWp=0\}.$$

That projection can be done by solving a sparse linear system of size $(d+n-1)$, not $m$.

### 3.1 A general “two-constraint projection” (works for any $D,W$)

Define the linear constraint operator
$$
A :=
\begin{bmatrix}
X^\top D\\
BW
\end{bmatrix}.
$$

The Euclidean projection of a vector $v\in\mathbb{R}^m$ onto $\ker(A)$ is
$$
\operatorname{Proj}(v)= v - A^\top \lambda,\qquad (AA^\top)\lambda = Av.
$$

Writing $\lambda=\begin{bmatrix}\alpha\\ \psi\end{bmatrix}$ with $\alpha\in\mathbb{R}^d$ and $\psi\in\mathbb{R}^{n-1}$, you get the explicit sparse block system
$$
\begin{bmatrix}
X^\top D^2 X & X^\top DWB^\top\\
BWD X & BW^2B^\top
\end{bmatrix}
\begin{bmatrix}\alpha\\ \psi\end{bmatrix}
=
\begin{bmatrix}
X^\top D v\\
BW v
\end{bmatrix},
$$
and then
$$
\operatorname{Proj}(v)= v - DX\,\alpha - WB^\top \psi.
$$

Notes:

- $B$ should be the incidence matrix with the $g$-row removed (equivalent to fixing $\psi(g)=0$).
- The bottom-right block is Laplacian-like; with $\epsilon>0$ and goal row removed it is typically SPD on the relevant component.

### 3.2 A numerically nicer special case: take $W=D_\epsilon$ and project in the same metric

If you set $W=D_\epsilon$ and (optionally) replace your first constraint by $X^\top Wp=0$ instead of $X^\top Dp=0$, you avoid the “squared weights” in the solve and the correction becomes very clean:
$$
\operatorname{Proj}(v)= v - X\alpha - B^\top\psi
$$
with
$$
\begin{bmatrix}
X^\top W X & X^\top W B^\top\\
B W X & B W B^\top
\end{bmatrix}
\begin{bmatrix}\alpha\\ \psi\end{bmatrix}
=
\begin{bmatrix}
X^\top W v\\
B W v
\end{bmatrix}.
$$
If you keep $X^\top Dp=0$ exactly as-is, use the general version above.

> This is what we are going to implement: $D_\epsilon$. 

---

## 4) End-to-end algorithm (projected-gradient training)

Below is the full loop. The only “IRL-specific” primitive you need is computing the model’s expected edge counts $\bar n(c)=\mathbb{E}_c[n(\tau)]$.

```code
Inputs:
  Graph G=(S,E), origin o, goal g
  Edge feature matrix X ∈ R^{m×d}
  Demonstrations {τ_i}; empirical edge-count mean n_hat ∈ R^m
  Incidence matrix B ∈ R^{(n-1)×m}   # remove goal row
  D = diag(n_hat + eps_D)             # eps_D > 0 recommended
  W = diag(weights + eps_W)           # e.g. W = D or W = I
  Regularization λ_w, λ_p; optional prior w0
Initialize:
  w ← w0 (or zeros)
  p ← 0
  p ← Proj(p)                         # ensure feasibility

Precompute:
  Factorize the projection KKT matrix (size d+n-1) if using direct solves

Repeat for t=1..T:
  c ← X w + p

  # 1) Soft DP to get policy π under costs c
  V ← soft_value_iteration(c, G, goal=g, with V(g)=0)
  π(e|s) ← exp( V(s) - c_e - V(head(e)) )

  # 2) Expected edge counts under π (absorbing at goal)
  n_model ← expected_edge_counts(G, π, origin=o, goal=g)

  # 3) Gradients of negative log-likelihood
  g_c ← n_hat - n_model               # = ∇_c J
  g_w ← Xᵀ g_c + λ_w (w - w0)
  g_p ← g_c + λ_p * grad_R(p)         # e.g. grad_R(p)=p, or D p, etc.

  # 4) Gradient steps
  w ← w - η_w * g_w                   # optionally project/parametrize for w≥0
  p_tmp ← p - η_p * g_p
  p ← Proj(p_tmp)                     # enforces Xᵀ D p=0 and B W p=0

Until convergence

Return w, p, c = Xw + p
```

---

<details>
<summary>How to compute expected edge counts $\mathbb{E}_c[n(\tau)]$ on a cyclic graph</summary>

Given the induced Markov policy $\pi(e\mid s)$ with absorbing goal $g$:

1) Build the state-to-state transition matrix on non-goal states:
$$
Q_{s,s'} = \sum_{e:s\to s'} \pi(e\mid s).
$$

2) Solve for expected state visit counts before absorption:
$$
(I - Q^\top)\,\mu = \mathbf{1}_o,
$$
where $\mathbf{1}_o$ is $1$ at the origin and $0$ elsewhere (in the non-goal indexing).

3) Then expected edge counts are:
$$
n_{\text{model},e}=\mu(\mathrm{tail}(e))\,\pi(e\mid \mathrm{tail}(e)).
$$

This is the standard “fundamental matrix” calculation for absorbing Markov chains and works even with cycles as long as absorption occurs with probability $1$ (which is typically true if the goal is reachable and costs make looping unattractive).
</details>

---

<details>
<summary>Remaining identifiability caveat (when learning $w$ too)</summary>

Even with $p$ constrained by $X^\top Dp=0$ and $BWp=0$, you can still have non-identifiability in $w$ **if** the feature space overlaps the shaping space, i.e. if there exist $\Delta w$ and $\psi$ such that
$$
X\Delta w = B^\top \psi.
$$

Then $w$ can move along $\Delta w$ without changing the policy/likelihood.

Practically: add a prior $w_0$ (from engineering knowledge), constrain signs/ranges of $w$, and/or fix one coefficient to set a scale.
</details>

---

## 5) What you already implemented vs. what to add

- Your $p(u)=P_{\perp,D}u$ is a good mechanism for enforcing $X^\top Dp=0$.
- To add the shaping gauge fix, you need **either**:
  - replace that parameterization by a single projector onto $\ker(X^\top D)\cap\ker(BW)$ (as above), **or**
  - keep your $P_{\perp,D}$ but do **true** intersection projection (not just one extra projection step unless you iterate alternating projections).

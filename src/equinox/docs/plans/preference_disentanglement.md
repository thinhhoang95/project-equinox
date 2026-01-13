## 1 Model formulation

Let $E$ be the set of edges of the routing graph and $m=|E|$.  
For each edge $e\in E$ we have a CDF‑feature vector  

\[
x(e)\in\mathbb{R}^{d}\qquad(d\ge 1),
\]

e.g. $d=3$ for *time*, *fuel* and *charge*.  
Stacking these vectors yields the matrix  

\[
X\in\mathbb{R}^{m\times d},\qquad X_{e,:}=x(e)^{\top}. \tag{1}
\]

Preferences are collected in a vector $p\in\mathbb{R}^{m}$ with $p_{e}= \text{pref}(e)$.  
Edge costs are then  

\[
c = Xw + p, \tag{2}
\]

where $w\in\mathbb{R}^{d}$ are the CDF weights.

## 2 Why a constraint is needed

Without any restriction on $p$, the pair $(w,p)$ is not identifiable. Indeed, for any $\Delta\in\mathbb{R}^{d}$,

\[
Xw + p = X(w+\Delta) + (p - X\Delta). \tag{3}
\]

Thus we must “fix a gauge”. The most direct choice is the orthogonality condition  

\[
X^{\top}p = 0, \tag{4}
\]

i.e. $p$ lies in the orthogonal complement of $\operatorname{col}(X)$, the column space of $X$.

## 3 Identifiability guarantee (formal)

Assume $X$ has full column rank, $\operatorname{rank}(X)=d$.  
Consider two decompositions of the same edge‑cost vector $c$:

\[
c = Xw + p = Xw^{\prime}+p^{\prime},\qquad 
X^{\top}p=0,\;X^{\top}p^{\prime}=0. \tag{5}
\]

Subtracting gives  

\[
X(w-w^{\prime}) = p^{\prime}-p. \tag{6}
\]

The left‑hand side is in $\operatorname{col}(X)$, the right‑hand side is in $\operatorname{col}(X)^{\perp}$ because  

\[
X^{\top}(p^{\prime}-p)=0.
\]

The only vector belonging to both subspaces is the zero vector; hence  

\[
X(w-w^{\prime})=0 \;\Longrightarrow\; w=w^{\prime}\quad(\text{full rank of }X),\qquad p=p^{\prime}.
\]

Thus the decomposition $(w,p)$ is unique under (4).

## 4 Enforcing $X^{\top}p=0$ concretely

### 4.1 Option A (recommended): re‑parameterise $p$

Let $\Pi$ be the orthogonal projector onto $\operatorname{col}(X)$:

\[
\Pi = X\,(X^{\top}X)^{-1}X^{\top}. \tag{7}
\]

The complementary projector is  

\[
P_{\perp}=I-\Pi. \tag{8}
\]

Introduce an unconstrained vector $u\in\mathbb{R}^{m}$ and define  

\[
p(u)=P_{\perp}u. \tag{9}
\]

By construction $X^{\top}p(u)=0$. One can now learn $(w,u)$ with any optimiser; $p$ is never an optimisation variable.

### 4.2 Option B: projected gradient

If you prefer to optimise $p$ directly, after each gradient step project back onto the feasible set:

1. Compute the unconstrained gradient $g_{p}$.
2. Project it:  

   \[
   g_{p}^{\text{proj}} = (I-\Pi) g_{p}.
   \]
3. Update $p \leftarrow p + \eta\, g_{p}^{\text{proj}}$.
4. (Optional) Re‑project $p \leftarrow (I-\Pi)p$ to remove numerical drift.

### 4.3 Option C: soft penalty

Add a quadratic penalty to the objective  

\[
\frac{\lambda}{2}\,\|X^{\top}p\|^{2}, \tag{10}
\]

which drives $X^{\top}p\to0$ as $\lambda\to\infty$. This is simple but does **not** enforce strict identifiability.

## 5 Efficient implementation of $P_{\perp}$

Forming the $m\times m$ matrix $P_{\perp}$ is unnecessary. For any $v\in\mathbb{R}^{m}$,

\[
\alpha = (X^{\top}X)^{-1}X^{\top}v,\qquad 
P_{\perp}v = v - X\alpha. \tag{11}
\]

The required $d\times d$ system is cheap because $d$ is small.

### 5.1 Python helper

```python
import numpy as np

def project_orthogonal(u, X, ridge=1e-12):
    """
    Returns p = (I - X (X^T X)^{-1} X^T) u.
    X: shape (m, d), u: shape (m,)
    """
    XtX = X.T @ X
    XtX = XtX + ridge * np.eye(XtX.shape[0])  # ridge for stability
    alpha = np.linalg.solve(XtX, X.T @ u)      # (d,)
    p = u - X @ alpha                         # (m,)
    return p
```

*Listing 1*: Projection onto the orthogonal complement of $\operatorname{col}(X)$.

If $X$ is rank‑deficient, replace the solve with a pseudoinverse or increase the ridge term.

## 5.2 Empirical-occupancy weighted orthogonality (recommended)

Sometimes orthogonality should reflect which edges are actually supported by the demonstrations. Let $\mathcal{D}=\{\tau_i\}_{i=1}^N$ be the demonstration set, and let $n(\tau)\in\mathbb{R}^m$ be the edge-count vector for a trajectory $\tau$ (so $n_e(\tau)$ is the number of times $\tau$ traverses edge $e$).

Define the **empirical expected edge counts**
$$
\hat{n}_{\mathcal{D}} \;:=\; \hat{\mathbb{E}}_{\mathcal{D}}[n(\tau)] \;=\; \frac{1}{N}\sum_{i=1}^N n(\tau_i)\in\mathbb{R}^m_{\ge 0}.
$$
Use these as fixed edge-importance weights:
$$
d \;:=\; \hat{n}_{\mathcal{D}},\qquad D \;:=\; \operatorname{diag}(d).
$$

We then enforce the **empirically weighted orthogonality constraint**
$$
X^\top D\,p = 0. \tag{12}
$$
Equivalently, $p$ lies in the $D$-orthogonal complement of $\operatorname{col}(X)$ under the weighted inner product $\langle u,v\rangle_D := u^\top Dv$.

The corresponding $D$-orthogonal projector onto $\operatorname{col}(X)$ is
$$
\Pi_D \;=\; X\,(X^\top D X)^{-1}X^\top D,\qquad P_{\perp,D} \;=\; I-\Pi_D,
$$
assuming $X^\top D X$ is invertible (i.e. $X$ has full column rank “on the empirical support” of $d$).

A convenient way to ensure feasibility is to parameterise
$$
p(u) \;=\; P_{\perp,D}u,
$$
with an unconstrained $u\in\mathbb{R}^m$.

**Efficient application of $P_{\perp,D}$ (no $m\times m$ matrices).** For any $v\in\mathbb{R}^m$, solve
$$
(X^\top D X)\alpha = X^\top D v
$$
and then compute
$$
P_{\perp,D}v = v - X\alpha.
$$

---

## 6 MaxEnt IRL with edge-wise preferences (fixed empirical weighting)

### 6.1 Standard notation

- $E$ — set of edges, $m=|E|$.
- $x(e)\in\mathbb{R}^d$ — CDF features; stacked in $X\in\mathbb{R}^{m\times d}$ as $X_{e,:}=x(e)^\top$.
- $w\in\mathbb{R}^d$ — CDF weights.
- $p\in\mathbb{R}^m$ — per-edge preferences.
- Edge-cost vector:
$$
c = Xw + p.
$$

For a trajectory $\tau$, let $n(\tau)\in\mathbb{R}^m$ be the edge-count vector, and define the trajectory feature sum
$$
F(\tau) = X^\top n(\tau).
$$
Trajectory cost:
$$
C_{w,p}(\tau) = w^\top F(\tau) + p^\top n(\tau). \tag{13}
$$

Let $\mathcal{D}=\{\tau_i\}_{i=1}^N$ be the demonstration set. Define empirical averages
$$
\hat{n}_{\mathcal{D}} := \frac{1}{N}\sum_{i=1}^N n(\tau_i),\qquad
\hat{F}_{\mathcal{D}} := \frac{1}{N}\sum_{i=1}^N F(\tau_i) = X^\top \hat{n}_{\mathcal{D}}.
$$

**Empirical weighting for identifiability (fixed):**
$$
d := \hat{n}_{\mathcal{D}},\qquad D := \operatorname{diag}(d).
$$

### 6.2 MaxEnt policy and gradients

MaxEnt trajectory distribution:
$$
\pi_{w,p}(\tau)\propto \mu(\tau)\exp\bigl(-C_{w,p}(\tau)\bigr). \tag{14}
$$

Let $J(w,p)=-\log L(w,p)$ be the negative log-likelihood. The (unconstrained) gradients are
$$
\nabla_w J(w,p) = - N\Bigl(\mathbb{E}_{\pi_{w,p}}[F(\tau)]-\hat{\mathbb{E}}_{\mathcal{D}}[F(\tau)]\Bigr), \tag{15}
$$
$$
\nabla_p J(w,p) = - N\Bigl(\mathbb{E}_{\pi_{w,p}}[n(\tau)]-\hat{\mathbb{E}}_{\mathcal{D}}[n(\tau)]\Bigr). \tag{16}
$$

i.e., should be data minus model, not model minus data - be careful about the sign of the gradients!

### 6.3 Identifiability via empirical weighted orthogonality

Because $c=Xw+p$ is invariant under
$$
Xw+p = X(w+\Delta) + (p-X\Delta),
$$
the pair $(w,p)$ is not identifiable without a gauge-fixing condition.

We fix the gauge using the **empirically weighted** constraint
$$
X^\top Dp = 0,\qquad D=\operatorname{diag}(\hat{n}_{\mathcal{D}}).
$$
This makes the decomposition identifiable under the condition that $X^\top D X$ is invertible (full rank of $X$ on the empirically visited edges).

### 6.4 Enforcing $X^\top Dp=0$ during optimisation

Define
$$
\Pi_D := X (X^\top D X)^{-1} X^\top D,\qquad P_{\perp,D} := I-\Pi_D.
$$

Two standard ways to enforce feasibility:

- **Re-parameterise:** optimise over $(w,u)$ with $p(u)=P_{\perp,D}u$.
- **Project:** if updating $p$ directly, maintain feasibility with
  $$
  p \leftarrow P_{\perp,D}p,
  $$
  and restrict preference updates to the feasible subspace by mapping the $p$-gradient (or update direction) through $P_{\perp,D}$.

Since $D$ is fixed from the data, you can precompute / factorise $X^\top D X$ once.

### 6.5 Interpretation

At a stationary point of the constrained optimisation:
$$
\mathbb{E}_{\pi_{w,p}}[F(\tau)] = \hat{\mathbb{E}}_{\mathcal{D}}[F(\tau)], \tag{17}
$$
and the edge-count residual has no component in directions explainable by $X$ under the empirical weighting:
$$
P_{\perp,D}\Bigl(\mathbb{E}_{\pi_{w,p}}[n(\tau)]-\hat{\mathbb{E}}_{\mathcal{D}}[n(\tau)]\Bigr)=0. \tag{18}
$$

---

## 7 Soft Bellman equations and policy extraction

Consider a directed graph where states are nodes and actions are outgoing edges. For a fixed destination $g$ (absorbing with $V(g)=0$), define the soft value function
$$
V(s) = -\log \sum_{e\in \operatorname{Out}(s)} \exp\bigl(-Q(s,e)\bigr),
\qquad
Q(s,e)=c_e + V(\operatorname{head}(e)). \tag{19}
$$

The induced Boltzmann policy is
$$
\pi(e\mid s)
=
\frac{\exp\bigl(-Q(s,e)\bigr)}{\sum_{e'\in \operatorname{Out}(s)}\exp\bigl(-Q(s,e')\bigr)}
=
\exp\bigl(V(s)-Q(s,e)\bigr). \tag{20}
$$

This policy computation uses the current costs $c=Xw+p$ (with $p$ constrained by $X^\top Dp=0$). The empirical weighting matrix $D=\operatorname{diag}(\hat{n}_{\mathcal{D}})$ is fixed and does **not** depend on $\pi$.

When the graph contains cycles, solve (19) by fixed-point iteration.

---

## 8 Expected edge counts under the soft policy

Given the soft policy $\pi$, compute expected edge traversals to form model expectations.

Let $\rho_0$ be the initial state distribution and $T$ a finite horizon. Let $\nu_t(s)$ denote the probability of being in state $s$ at time $t$. Initialise
$$
\nu_0=\rho_0.
$$
For $t=0,\dots,T-1$, propagate
$$
\nu_{t+1}(s') = \sum_{s}\nu_t(s)\sum_{e:s\to s'} \pi(e\mid s),
$$
(with the usual absorbing modification if the destination $g$ is terminal).

The expected number of traversals of edge $e$ under $\pi$ is
$$
n_{\text{model}}(e) = \sum_{t=0}^{T-1} \nu_t(\operatorname{tail}(e))\,\pi\bigl(e\mid \operatorname{tail}(e)\bigr).
$$

The corresponding expected CDF counts are
$$
F_{\text{model}} = X^\top n_{\text{model}}.
$$

From demonstrations, the empirical counterparts are
$$
\hat{n}_{\mathcal{D}}=\frac{1}{N}\sum_{i=1}^N n(\tau_i),\qquad \hat{F}_{\mathcal{D}}=X^\top \hat{n}_{\mathcal{D}}.
$$

These $\hat{n}_{\mathcal{D}}$ are used both as the MaxEnt matching target and (via $D=\operatorname{diag}(\hat{n}_{\mathcal{D}})$) as the fixed empirical weighting for the identifiability constraint.

---

<details>
<summary><strong>Optional: covariance interpretation under the empirical edge distribution</strong></summary>

Define the empirical edge distribution
$$
q(e) := \frac{d_e}{\sum_{e'} d_{e'}},\qquad d=\hat{n}_{\mathcal{D}}.
$$
If $E\sim q$ and you consider $A=(Xw)_E$ and $B=p_E$, then $X^\top Dp=0$ implies $\mathbb{E}_q[AB]=0$. If $X$ includes a constant feature $x_0(e)=1$, then the same constraint also implies $\mathbb{E}_q[B]=0$, and therefore
$$
\operatorname{Cov}_q(A,B)=0.
$$
</details>
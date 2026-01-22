# Disentangled Cost Learning for Air-Traffic Routing Under Potential-Shaping Invariance

Thinh Hoang

###### Abstract

We present a principled framework for learning edge-cost models in directed air-traffic networks while explicitly handling two sources of ambiguity: (1) the attribution ambiguity between feature-based costs and a residual preference, and (2) the potential-shaping invariance of routing policies. By projecting both features and residuals onto the cycle space of the graph, we obtain a gauge-fixed formulation

$$
c = X_{\text{cyc}}\,w + p,\qquad BW\,p = 0,\;X_{\text{cyc}}^{\top}Dp = 0,
$$

which yields identifiable, policy-relevant cost components. The paper details the mathematical derivation, implementation recipes, and practical considerations.

## 1 Setup and notation

### 1.1 Graph and costs

- Directed graph $G=(V,E)$ with $|V|=n$ nodes and $|E|=m$ edges.  
- Edge costs $c\in\mathbb{R}^{m}$, one scalar per edge.  
- Edge-feature matrix $X\in\mathbb{R}^{m\times d}$, with $d$ features per edge.  
- Feature weights $w\in\mathbb{R}^{d}$.  
- Residual / preference vector $p\in\mathbb{R}^{m}$.

### 1.2 Incidence matrix and potentials

Choose an orientation consistent with the directed edges and define the *full* node–edge incidence matrix $B_{\text{full}}\in\mathbb{R}^{n\times m}$ by  

\[
(B_{\text{full}})_{u,e}=
\begin{cases}
+1 & \text{if $e$ leaves node $u$,}\\
-1 & \text{if $e$ enters node $u$,}\\
0  & \text{otherwise.}
\end{cases}
\]

A node potential $\phi\in\mathbb{R}^{n}$ induces an edge “gradient”

\[
B_{\text{full}}^{\top}\phi\in\mathbb{R}^{m},\qquad
\left(B_{\text{full}}^{\top}\phi\right)_{e}=\phi(u)-\phi(v)
\]

for $e:u\to v$.

To remove the constant-potential nullspace we anchor one node (typically a goal) by dropping its row, obtaining the *reduced* incidence matrix  

\[
B\in\mathbb{R}^{(n-1)\times m}.
\]

Equivalently, we fix $\phi(\text{anchor})=0$ and represent $\phi$ only on the remaining $n-1$ nodes.

### 1.3 Weight matrices

We employ two diagonal edge-weight matrices:

- $W\in\mathbb{R}^{m\times m}$, $W\succ 0$, to define the gauge-fixing / cycle-space notion.  
- $D\in\mathbb{R}^{m\times m}$, $D\succeq 0$, to define the disentanglement notion.

Typical choices are $W=I$ (global gauge fix) or $W=D_{\epsilon}=\operatorname{diag}(\hat{n}+\epsilon)$ with a small $\epsilon>0$, and an analogous choice for $D$.

## 2 Why costs are not identifiable: potential-shaping invariance

### 2.1 Deterministic shortest path

For any origin–goal pair $o\to g$ and a path $\tau=(e_{1},\ldots,e_{T})$, the path cost is $\sum_{t=1}^{T}c_{e_{t}}$. If we add a potential gradient to edge costs,

\[
c^{\prime}=c+B_{\text{full}}^{\top}\phi,
\]

then  

\[
\sum_{t=1}^{T}c^{\prime}_{e_{t}}
= \sum_{t=1}^{T}c_{e_{t}}+\sum_{t=1}^{T}\bigl(\phi(s_{t})-\phi(s_{t+1})\bigr)
= \sum_{t=1}^{T}c_{e_{t}}+\phi(o)-\phi(g). \tag{1}
\]

The shift $\phi(o)-\phi(g)$ is the same for all $o\to g$ paths, so the argmin path(s) are unchanged.

### 2.2 Maximum-entropy (soft) path distributions

In a MaxEnt model,

\[
P(\tau\mid o,g,c)=\frac{\exp\!\bigl(-c^{\top}n(\tau)\bigr)}{Z_{o,g}(c)},\qquad
n(\tau)\in\mathbb{R}^{m}\text{ counts edge usages}.
\]

Adding the same potential gradient yields  

\[
(c+B_{\text{full}}^{\top}\phi)^{\top}n(\tau)
= c^{\top}n(\tau)+\phi(o)-\phi(g),
\]

so the constant cancels between numerator and partition function $Z_{o,g}$; the distribution is unchanged.

### 2.3 Consequence

The policy (or likelihood) identifies only an equivalence class  

\[
c\sim c+B_{\text{full}}^{\top}\phi.
\]

Thus, writing $c=Xw+p$ without a gauge-fix allows the residual $p$ to accidentally absorb an arbitrary $B^{\top}\phi$ component.

## 3 Two distinct ambiguities (“gauges”) we must fix

### 3.1 Gauge A: feature–residual attribution ambiguity

Even with a fixed $c$, the decomposition $c=Xw+p$ is not unique unless we constrain $p$. Our proposal  

\[
X^{\top}Dp=0 \tag{1}
\]

forces $p$ to be orthogonal to the feature span under the $D$‑weighted inner product. In words, “$p$ contains no component explainable by the features (on the weighted support).”

### 3.2 Gauge B: potential-shaping ambiguity (policy invariance)

Even after fixing the split, $c$ itself is ambiguous up to $B^{\top}\phi$. To ensure that $p$ is a *policy-identifiable* preference we forbid $p$ from containing any gradient component. This is expressed by the *cycle‑space constraint*

\[
BW\,p=0. \tag{2}
\]

Equivalently, $Wp$ is a circulation (zero divergence at every node).

### 3.3 Combined requirement

Collecting (1) and (2) we require  

\[
p\in\ker(X^{\top}D)\ \cap\ \ker(BW). \tag{3}
\]

## 4 Why $BWp=0$ is the right “no potential component” condition

Define a $W$‑weighted inner product on edge vectors:

\[
\langle a,b\rangle_{W}:=a^{\top}Wb.
\]

A vector $p$ has no gradient component (in the $W$‑orthogonal sense) iff it is orthogonal to every gradient $B^{\top}\phi$:

\[
\langle p,B^{\top}\phi\rangle_{W}=0\quad\forall\;\phi.
\]

Indeed,

\[
\langle p,B^{\top}\phi\rangle_{W}=p^{\top}WB^{\top}\phi=(BWp)^{\top}\phi,
\]

which vanishes for all $\phi$ exactly when $BWp=0$. Hence (2) is precisely the desired condition.

## 5 The cycle projector $P_{\text{cyc}}$ (“remove potential component”)

### 5.1 Derivation via weighted least squares

Given any edge vector $v\in\mathbb{R}^{m}$ we seek the best‑fitting potential gradient in the $W$‑metric:

\[
\phi^{*}=\arg\min_{\phi}\bigl\|v-B^{\top}\phi\bigr\|_{W}^{2},\qquad
\|x\|_{W}^{2}:=x^{\top}Wx.
\]

The first‑order optimality condition reads  

\[
B\,W\bigl(v-B^{\top}\phi^{*}\bigr)=0
\;\Longleftrightarrow\;
(BWB^{\top})\phi^{*}=BWv.
\]

Let the reduced weighted Laplacian be  

\[
L:=BWB^{\top}\in\mathbb{R}^{(n-1)\times(n-1)}.
\]

Assuming $W\succ 0$ and a proper anchor, $L$ is symmetric positive definite. Thus  

\[
\phi^{*}=L^{-1}(BWv),\qquad
P_{\text{cyc}}v:=v-B^{\top}\phi^{*}. \tag{4}
\]

Equation (4) can be written as a matrix operator (without forming it explicitly):

\[
P_{\text{cyc}}=I-B^{\top}(BWB^{\top})^{-1}BW. \tag{5}
\]

### 5.2 Key properties

1. **Gauge removal:** $P_{\text{cyc}}(v+B^{\top}\phi)=P_{\text{cyc}}v.$  
2. **Cycle constraint:** $BW\bigl(P_{\text{cyc}}v\bigr)=0.$  

## 6 Project features into the cycle space once and for all

### 6.1 Why we must also project $X$

Even if $p$ satisfies (2), the feature term $Xw$ can hide shaping: if there exist $\Delta w$ and $\psi$ such that  

\[
X\Delta w=B^{\top}\psi,
\]

then replacing $w\leftarrow w+\Delta w$ changes $c$ only by a gradient, leaving the policy unchanged. Hence $w$ is not identifiable.

### 6.2 Cycle‑projected features

Define the *cycle‑projected* feature matrix  

\[
X_{\text{cyc}}:=P_{\text{cyc}}X. \tag{6}
\]

By construction $BWX_{\text{cyc}}=0$, so every column of $X_{\text{cyc}}$ already lies in the cycle space.

### 6.3 Eliminating the shaping ambiguity in $w$

Suppose $X_{\text{cyc}}\Delta w=B^{\top}\psi$. Applying $BW$ to both sides yields  

\[
0=BWX_{\text{cyc}}\Delta w=L\psi,
\]

and because $L$ is invertible on the anchored component we obtain $\psi=0$, hence $X_{\text{cyc}}\Delta w=0$. Therefore, after (6) the only remaining non‑identifiability of $w$ is the usual linear‑model nullspace (collinearity).

## 7 The recommended model

Putting the pieces together we adopt the gauge‑fixed formulation  

\[
c=X_{\text{cyc}}\,w+p,\qquad
\underbrace{BWp=0}_{\text{no potential component}},\qquad
\underbrace{X_{\text{cyc}}^{\top}Dp=0}_{\text{feature disentanglement}}. \tag{7}
\]

Equation (7) yields a cost vector whose residual part $p$ is

- orthogonal to the engineered features (weighted by $D$), and  
- free of any gradient that could be absorbed by a potential‑shaping transformation.

Consequently $p$ captures a *policy‑identifiable preference* over cycles, while $X_{\text{cyc}}w$ accounts for the known feature‑based effects.

## 8 Practical end‑to‑end recipe

### 8.1 Precomputation

1. **Inputs**  
   - Graph $G = (V, E)$.  
   - Feature matrix $X \in \mathbb{R}^{m \times d}$.  
   - Anchor node (often a goal).  
   - Diagonal weight matrices $W \succ 0$, $D \succeq 0$.

2. **Build reduced incidence $B$** by dropping the anchor row.

3. **Form the weighted Laplacian**  

   \[
   L := B W B^{\top}.
   \]

   Factorize $L$ (e.g., Cholesky) for repeated solves.

4. **Cycle‑project the features**

   (a) Compute $R \coloneqq BWX \in \mathbb{R}^{(n-1) \times d}$.  

   (b) Solve $L\Phi_{X}=R$ for $\Phi_{X}\in\mathbb{R}^{(n-1)\times d}$.  

   (c) Set $X_{\mathrm{cyc}} := X - B^{\top}\Phi_{X}$.

5. **Pre‑compute the $D$‑weighted Gram matrix**  

   \[
   G := X_{\mathrm{cyc}}^{\top} D X_{\mathrm{cyc}} + \gamma I,
   \]

   with a small ridge $\gamma>0$; factorize $G$ as well.

### 8.2 Training loop (projected gradient)

The following pseudocode assumes we can obtain the gradient of the MaxEnt log‑likelihood with respect to the edge costs $c$ (denoted $g_c$).

```python
# Precomputed objects:
# B, W, D
# L = B @ W @ B.T  (factorized)
# X_cyc
# G = X_cyc.T @ D @ X_cyc + gamma * np.eye(d)  (factorized)
# hyper‑parameters
eta_w = 1e-3
eta_p = 1e-3
lambda_w = 1e-4
lambda_p = 1e-4
# initialization
w = np.zeros(d)
p = np.zeros(m)  # already satisfies BWp=0 and X_cyc.T @ D @ p = 0

for t in range(T):
    # 1) current costs
    c = X_cyc @ w + p

    # 2) obtain model expected edge counts under current costs
    # (implementation depends on the IRL algorithm)
    n_model = expected_edge_counts(c, demonstrations)

    # 3) gradient of negative log‑likelihood w.r.t. c
    g_c = n_hat - n_model   # empirical minus expected

    # 4) gradient step for w
    w -= eta_w * (X_cyc.T @ g_c + lambda_w * (w - w0))

    # 5) tentative update of p
    p_tmp = p - eta_p * (g_c + lambda_p * p)

    # 6) --- project p onto the intersection ---
    # (a) cycle projection: p1 = P_cyc(p_tmp)
    rhs = B @ (W @ p_tmp)
    phi = solve(L, rhs)                     # L^{-1} rhs
    p1 = p_tmp - B.T @ phi                  # now BW p1 = 0 (up to numerical error)

    # (b) D‑weighted feature orthogonalisation:
    b = X_cyc.T @ (D @ p1)
    alpha = solve(G, b)                     # G^{-1} b
    p = p1 - X_cyc @ alpha                  # preserves BW p = 0 and enforces X_cyc^T D p = 0
```

**Listing 1:** Projected gradient for $w$ and $p$

After convergence we return $(w,p)$ and the final cost $c = X_{\mathrm{cyc}}w + p$.

### 8.3 Why the two‑step projection is exact

Because $BWX_{\mathrm{cyc}} = 0$, any vector in the column span of $X_{\mathrm{cyc}}$ already satisfies the cycle constraint. Hence after step (a) we have $BWp_{1}=0$, and subtracting $X_{\mathrm{cyc}}\alpha$ in step (b) leaves the constraint unchanged:

\[
BW(p_{1} - X_{\mathrm{cyc}}\alpha)
= BWp_{1} - BWX_{\mathrm{cyc}}\alpha
= 0.
\]

## 9 What if we do not pre‑project $X$?

If $X$ is used directly, the two‑step projection may re‑introduce a gradient component when we subtract $X\alpha$ (because $BWX$ need not be zero). In that case we must project onto the intersection $\ker(X^{\top}D) \cap \ker(BW)$ in a single solve:

\[
\operatorname{Proj}(v) = v - A^{\top}\lambda,
\qquad
A = \begin{bmatrix}
X^{\top} D \\[2pt] BW
\end{bmatrix},
\qquad
(A A^{\top}) \lambda = A v.
\]

While mathematically correct, this approach is computationally heavier than the pre‑projected strategy described above.

## 10 Interpretation of each constraint

| Component | Constraint | Meaning |
|-----------|------------|---------|
| Disentanglement | $X_{\text{cyc}}^{\top} D p = 0$ | $p$ contains no component explainable by the engineered features |
| Gauge‑fix (residual) | $BWp = 0$ | $p$ contains no potential‑shaping (gradient) component |
| Feature gauge‑fix | $X_{\text{cyc}} = P_{\text{cyc}}X$ | Features themselves are already free of gradient components |

**Table 1:** Summary of the three core constraints.

## 11 Remaining identifiability caveats

1. **Feature collinearity.** If $X_{\mathrm{cyc}}\Delta w = 0$, $w$ is not unique. Regularisation (ridge, sparsity) or explicit constraints can mitigate this.  
2. **Pure‑gradient features.** Any column of $X$ that is exactly a gradient $B^{\top}\phi$ vanishes after projection $P_{\mathrm{cyc}}$, indicating that the feature is intrinsically unidentifiable from routing choices.  
3. **Disconnected subgraphs.** If some nodes are unreachable from the anchor, the reduced Laplacian $L$ is singular on those components. One must either restrict attention to the reachable subgraph or anchor each component separately.  
4. **Scaling / temperature.** In MaxEnt models, multiplying $c$ by a constant changes the effective temperature; this is independent of the potential‑shaping gauge and must be handled (e.g., by fixing a temperature hyper‑parameter).

## 12 Recommended defaults (practical)

- Choose $D = \operatorname{diag}(\hat{n} + \epsilon_{D})$ and $W = \operatorname{diag}(\hat{n} + \epsilon_{W})$ with a small $\epsilon>0$ to avoid division by zero.  
- Anchor a single, globally relevant node (e.g., a common destination).  
- Add $\ell_{2}$ regularisation on $w$ and $p$, possibly weighted by $D$.  
- Use a modest ridge $\gamma$ when forming $G$ to guarantee numerical stability.

## 13 Summary checklist

1. Build reduced incidence $B$ and select a positive‑definite $W$.  
2. Factorize $L = BWB^{\top}$.  
3. Compute the cycle‑projected features $X_{\mathrm{cyc}} = X - B^{\top}(BWB^{\top})^{-1}BWX$.  
4. Train the model $c = X_{\mathrm{cyc}}w + p$ using projected gradient.  
5. After each $p$ update enforce:  
   (i) $p \gets P_{\mathrm{cyc}}p$ (cycle/gauge projection);  
   (ii) $p \gets p - X_{\mathrm{cyc}}\bigl(X_{\mathrm{cyc}}^{\top} D X_{\mathrm{cyc}}\bigr)^{-1} X_{\mathrm{cyc}}^{\top} D p$ (feature disentanglement).  
6. Interpret the final $p$ as a policy‑identifiable preference over cycles, orthogonal to the engineered features.

## 14 Response 1

### 14.1 Cycle/potential projector: algebra check

#### 14.1.1 Dimensions

- $B \in \mathbb{R}^{(n-1) \times m}$, $W \in \mathbb{R}^{m \times m}$ diagonal with $W \succ 0$.  
- $v \in \mathbb{R}^{m}$, $\phi \in \mathbb{R}^{n-1}$.  
- $BWv \in \mathbb{R}^{n-1}$, $L := BWB^{\top} \in \mathbb{R}^{(n-1) \times (n-1)}$.

All products in  

\[
P_{\mathrm{cyc}} = I - B^{\top}(BWB^{\top})^{-1}BW \tag{2}
\]

have consistent shape: $B^{\top}(\cdot)BW$ is $m \times m$.

#### 14.1.2 Weighted least‑squares $\rightarrow$ normal equations

The objective is  

\[
\min_{\phi}\ (v-B^{\top}\phi)^{\top}W\,(v-B^{\top}\phi). \tag{3}
\]

The gradient w.r.t. $\phi$ is  

\[
\nabla_{\phi} = -2\,B\,W\,(v-B^{\top}\phi), \tag{4}
\]

so the first‑order optimality gives  

\[
BW(v-B^{\top}\phi^{\star})=0
\;\Longleftrightarrow\;
(BWB^{\top})\phi^{\star}=BWv. \tag{5}
\]

Hence  

\[
\phi^{\star}=(BWB^{\top})^{-1}BWv,\qquad
P_{\rm cyc}v=v-B^{\top}\phi^{\star}. \tag{6}
\]

These expressions are correct.

#### 14.1.3 Projector properties (quick sanity)

Let $L:=BWB^{\top}$. The following identities hold.

1. **Cycle constraint:** $BW(P_{\bf cyc}v)=0$  

   \[
   BW\bigl(I-B^{\top}L^{-1}BW\bigr)v
   = BWv-(BWB^{\top})L^{-1}BWv
   = BWv-BWv=0.
   \]

2. **Gauge removal:** $P_{\bf cyc}(v+B^{\top}\psi)=P_{\bf cyc}v$ (because $P_{\rm cyc}B^{\top}=0$):  

   \[
   \bigl(I-B^{\top}L^{-1}BW\bigr)B^{\top}\psi
   = B^{\top}\psi-B^{\top}L^{-1}(BWB^{\top})\psi
   = B^{\top}\psi-B^{\top}\psi=0.
   \]

3. **Idempotence:** $P_{\bf cyc}^{2}=P_{\bf cyc}$ (verified using $BWB^{\top}=L$).

Thus $P_{\bf cyc}$ removes the potential component provided $L$ is invertible (reduced/anchored case, connected graph, $W\succ 0$).

### 14.2 Weighted cycle space definition

The authors define  

\[
\mathcal{C}_{W}:=\ker(BW). \tag{11}
\]

Since  

\[
\langle B^{\top}\phi,\,z\rangle_{W}=(B^{\top}\phi)^{\top}Wz=\phi^{\top}BWz,
\]

$\mathcal{C}_{W}$ is exactly the $W$‑orthogonal complement of $\operatorname{im}(B^{\top})$. Because $W$ is invertible, $\ker(BW)=W^{-1}\ker(B)$, i.e. a scaled version of the usual cycle space.

### 14.3 Pre‑projecting $X$: algebra check

Define $X_{\rm cyc}:=P_{\rm cyc}X$. Then  

\[
BWX_{\rm cyc}=BW\bigl(I-B^{\top}L^{-1}BW\bigr)X
= BWX-(BWB^{\top})L^{-1}BWX
= BWX-BWX=0.
\]

Hence every column of $X_{\rm cyc}$ lies in $\ker(BW)$, and for any $\alpha$

\[
BW\bigl(X_{\rm cyc}\alpha\bigr)=0.
\]

Thus if $p\in\ker(BW)$ then the update $p\leftarrow p-X_{\rm cyc}\alpha$ stays in $\ker(BW)$.

### 14.4 Feature‑orthogonal projection step

The authors enforce $X_{\rm cyc}^{\top}Dp=0$ by  

\[
p\leftarrow p-X_{\rm cyc}\alpha,\qquad
\bigl(X_{\rm cyc}^{\top}DX_{\rm cyc}\bigr)\alpha = X_{\rm cyc}^{\top}Dp. \tag{16}
\]

This is the standard $D$‑orthogonal projection onto $\ker(X_{\rm cyc}^{\top}D)$ along $\operatorname{span}(X_{\rm cyc})$, assuming $X_{\rm cyc}^{\top}DX_{\rm cyc}$ is invertible on that subspace.

If a ridge term $G:=X_{\rm cyc}^{\top}DX_{\rm cyc}+\gamma I$ is used, solving  

\[
G\alpha = X_{\rm cyc}^{\top}Dp
\]

yields  

\[
X_{\rm cyc}^{\top}D\bigl(p-X_{\rm cyc}\alpha\bigr)=\gamma\alpha,
\]

so the constraint $X_{\rm cyc}^{\top}Dp=0$ holds exactly only when $\gamma=0$. With $\gamma>0$ the step provides a *regularized approximate* projection.

### 14.5 Two‑step projection onto the intersection

The two‑step procedure reaches a point in the intersection because:

1. Step 1 maps to $\ker(BW)$.  
2. Step 2 subtracts a vector in $\operatorname{span}(X_{\rm cyc})\subseteq\ker(BW)$, preserving membership in $\ker(BW)$, and is constructed to enforce $X_{\rm cyc}^{\top}Dp=0$.

Thus feasibility in one pass is correct. The resulting operator is a linear projector onto the intersection, but not necessarily the *orthogonal* projector for a single global metric unless that metric is chosen appropriately.

### 14.6 Clarifying the $BW\phi=0$ confusion

1. The equation $BW\phi=0$ is dimensionally inconsistent: $BW$ maps edge vectors $\in\mathbb{R}^{m}$ to node‑balance vectors $\in\mathbb{R}^{n-1}$, whereas $\phi\in\mathbb{R}^{n-1}$.  
2. One usually does *not* constrain $\phi$ at all. Instead, fix a gauge on the edge‑cost vector by applying $P_{\rm cyc}$.

A clean formulation is  

\[
\tilde{c}:=P_{\rm cyc}c,\qquad
\tilde{c}=X_{\rm cyc}w+p,\;
p\in\ker(BW),\;
X_{\rm cyc}^{\top}Dp=0.
\]

Then $\phi$ disappears automatically after projection, and no extra coupling constraint between $p$ and $\phi$ is required.

### 14.7 Bottom line

- The projector $P_{\rm cyc}=I-B^{\top}(BWB^{\top})^{-1}BW$ and the associated algebra are correct.  
- Pre‑projecting $X$ indeed makes the two constraints compatible, so the two‑step “make $p$ cyclic, then make it feature‑orthogonal” yields a feasible $p$ in one pass.  
- Avoid writing $BW\phi=0$; the gauge should be imposed on $c$ or $p$, not on $\phi$.  
- Adding a ridge term destroys the exactness of the $X_{\rm cyc}^{\top}Dp=0$ constraint; set $\gamma=0$ for exact feasibility or accept a regularized approximation.  
- The two‑step method is a valid linear projection onto the intersection, but it is not necessarily the orthogonal projection for a single quadratic metric.

## 15 Response 2

### 15.1 Cheapest monitor: cycle‑feasibility residual

For any edge vector $v$ (e.g., the current $p$ or a temporary update $p_{\rm tmp}$), define  

\[
r_{\rm cyc}(v):=\|BWv\|_{2}. \tag{17}
\]

- $r_{\rm cyc}(v)=0$ iff $v\in\ker(BW)$.  
- In practice one monitors this quantity *before* applying the cycle projection, because after projection it will be (up to round‑off) zero.

### 15.2 Potential energy via a Laplacian solve

Let $L:=BWB^{\top}$ and compute the potential fit  

\[
\phi^{\star}(v):=L^{-1}BWv. \tag{18}
\]

The removed potential component is $g(v)=B^{\top}\phi^{\star}(v)$, whose $W$‑energy is  

\[
E_{\rm pot}(v) := \|g(v)\|_{W}^{2}
= \|B^{\top}\phi^{\star}(v)\|_{W}^{2}
= \phi^{\star}(v)^{\top}L\,\phi^{\star}(v)
= (BWv)^{\top}L^{-1}(BWv). \tag{22}
\]

Since $P_{\rm cyc}$ is the $W$‑orthogonal projector, the energy splits exactly:

\[
\|v\|_{W}^{2}
= \|P_{\rm cyc}v\|_{W}^{2}+E_{\rm pot}(v). \tag{23}
\]

A convenient normalized “cycle fraction” is  

\[
s_{\rm cyc}(v):=
\frac{\|P_{\rm cyc}v\|_{W}^{2}}{\|v\|_{W}^{2}}
=1-\frac{(BWv)^{\top}L^{-1}(BWv)}{v^{\top}Wv}. \tag{24}
\]

### 15.3 Graph/weight “health” monitor: conditioning of $L$

The numerical stability of the previous quantities is governed by the spectrum of $L=BWB^{\top}$.

- $\lambda_{\min}(L)$: a very small value indicates an ill‑conditioned Laplacian solve.  
- $\kappa(L)=\lambda_{\max}(L)/\lambda_{\min}(L)$: a large condition number signals potential sensitivity in the potential estimates and projections.

### 15.4 Number of cyclic degrees of freedom

The dimension of the weighted cycle space is  

\[
\dim\ker(BW)=m-\operatorname{rank}(BW). \tag{25}
\]

For a connected graph with one anchored node and $W\succ 0$, $\operatorname{rank}(BW)=n-1$, so  

\[
\dim\ker(BW)=m-n+1. \tag{26}
\]

## Acknowledgments

We thank the anonymous reviewers for insightful comments that helped improve the presentation of this work.

## References

- Ziebart, B. D., Maas, A., Bagnell, J. A., & Dey, A. K. (2008). *Maximum Entropy Inverse Reinforcement Learning*. Proceedings of the 23rd AAAI Conference on Artificial Intelligence.  
- Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*. Cambridge University Press.  
</file>

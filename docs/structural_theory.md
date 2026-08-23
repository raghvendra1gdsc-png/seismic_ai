# Structural Dynamics Theory & Numerical Formulations

## 1. Governing Equation of Motion
For an $N$-degree-of-freedom multi-storey shear building subjected to horizontal unidirectional earthquake base motion $a_g(t)$, dynamic equilibrium is governed by the second-order matrix differential equation:

$$\mathbf{M} \mathbf{\ddot{u}}(t) + \mathbf{C} \mathbf{\dot{u}}(t) + \mathbf{K} \mathbf{u}(t) = -\mathbf{M} \mathbf{r} a_g(t)$$

where:
- $\mathbf{M} \in \mathbb{R}^{N \times N}$ is the diagonal lumped mass matrix.
- $\mathbf{C} \in \mathbb{R}^{N \times N}$ is the classical Rayleigh proportional damping matrix.
- $\mathbf{K} \in \mathbb{R}^{N \times N}$ is the tridiagonal lateral shear stiffness matrix.
- $\mathbf{u}(t) = [u_1(t), u_2(t), \dots, u_N(t)]^T \in \mathbb{R}^{N \times 1}$ is the vector of relative horizontal floor displacements with respect to the moving ground.
- $\mathbf{r} = [1, 1, \dots, 1]^T \in \mathbb{R}^{N \times 1}$ is the ground motion influence coefficient vector.
- $a_g(t)$ is the horizontal ground acceleration time-history in $\text{m/s}^2$.

---

## 2. Structural Matrices Assembly

### A. Lumped Mass Matrix ($\mathbf{M}$)
Under the rigid floor diaphragm idealization, the total structural mass of floor $i$ (including slab, dead load, and code-specified live load participation) is lumped at each storey level:

$$\mathbf{M} = \begin{bmatrix}
m_1 & 0 & 0 & \dots & 0 \\
0 & m_2 & 0 & \dots & 0 \\
0 & 0 & m_3 & \dots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
0 & 0 & 0 & \dots & m_N
\end{bmatrix}$$

### B. Lateral Shear Stiffness Matrix ($\mathbf{K}$)
Assuming flexurally rigid floor girders relative to columns (pure shear building idealization), columns deform in double curvature. The lateral shear stiffness of storey $i$ with $n_{\text{col}}$ identical columns of height $h_i$, modulus of elasticity $E$, and moment of inertia $I$ is:

$$k_i = \sum_{j=1}^{n_{\text{col}}} \frac{12 E I_j}{h_i^3}$$

The global assembled stiffness matrix $\mathbf{K}$ is symmetric, tridiagonal, and positive definite:

$$\mathbf{K} = \begin{bmatrix}
k_1 + k_2 & -k_2 & 0 & \dots & 0 \\
-k_2 & k_2 + k_3 & -k_3 & \dots & 0 \\
0 & -k_3 & k_3 + k_4 & \dots & 0 \\
\vdots & \vdots & \ddots & \ddots & -k_N \\
0 & 0 & \dots & -k_N & k_N
\end{bmatrix}$$

### C. Rayleigh Proportional Damping Matrix ($\mathbf{C}$)
Classical proportional damping is formulated as a linear combination of mass and stiffness:

$$\mathbf{C} = \alpha \mathbf{M} + \beta \mathbf{K}$$

The modal damping ratio $\zeta_n$ for mode $n$ with undamped circular frequency $\omega_n$ satisfies:

$$\zeta_n = \frac{\alpha}{2 \omega_n} + \frac{\beta \omega_n}{2}$$

Given target damping ratios $\zeta_i$ and $\zeta_j$ specified at modes $i$ and $j$ ($\omega_i < \omega_j$):

$$\begin{bmatrix} \alpha \\ \beta \end{bmatrix} = \frac{2 \omega_i \omega_j}{\omega_j^2 - \omega_i^2} \begin{bmatrix} \omega_j & -\omega_i \\ -1/\omega_j & 1/\omega_i \end{bmatrix} \begin{bmatrix} \zeta_i \\ \zeta_j \end{bmatrix}$$

---

## 3. Generalized Eigenvalue Analysis
Free undamped vibration satisfies:

$$\mathbf{K} \boldsymbol{\phi}_n = \omega_n^2 \mathbf{M} \boldsymbol{\phi}_n$$

- Circular natural frequencies: $\omega_1 \le \omega_2 \le \dots \le \omega_N$ (rad/s)
- Natural periods: $T_n = \frac{2\pi}{\omega_n}$ (s)
- Mode shapes: $\boldsymbol{\Phi} = [\boldsymbol{\phi}_1, \dots, \boldsymbol{\phi}_N]$

### Normalization & Orthogonality
Mass-orthonormality enforces:

$$\boldsymbol{\Phi}^T \mathbf{M} \boldsymbol{\Phi} = \mathbf{I}, \quad \boldsymbol{\Phi}^T \mathbf{K} \boldsymbol{\Phi} = \operatorname{diag}(\omega_1^2, \dots, \omega_N^2)$$

Modal participation factor: $\Gamma_n = \boldsymbol{\phi}_n^T \mathbf{M} \mathbf{r}$
Effective modal mass: $M_n^* = \Gamma_n^2$, with $\sum_{n=1}^N M_n^* = \sum_{i=1}^N m_i$.

---

## 4. Newmark-$\beta$ Direct Numerical Integration
The system is integrated step-by-step using the Newmark-$\beta$ family:

$$\mathbf{u}_{t+\Delta t} = \mathbf{u}_t + \Delta t \mathbf{\dot{u}}_t + \left(\frac{1}{2} - \beta\right)\Delta t^2 \mathbf{\ddot{u}}_t + \beta \Delta t^2 \mathbf{\ddot{u}}_{t+\Delta t}$$

$$\mathbf{\dot{u}}_{t+\Delta t} = \mathbf{\dot{u}}_t + (1 - \gamma)\Delta t \mathbf{\ddot{u}}_t + \gamma \Delta t \mathbf{\ddot{u}}_{t+\Delta t}$$

### Average Acceleration Method ($\gamma = 0.5, \beta = 0.25$)
- **Stability**: Unconditionally stable for all $\Delta t > 0$ in linear elastic regimes.
- **Effective Dynamic Stiffness**:
  $$\mathbf{\hat{K}} = \mathbf{K} + \frac{1}{\beta \Delta t^2} \mathbf{M} + \frac{\gamma}{\beta \Delta t} \mathbf{C}$$
- **Effective Load Vector**:
  $$\mathbf{\hat{P}}_{t+\Delta t} = \mathbf{P}_{t+\Delta t} + \mathbf{M}\left(a_0 \mathbf{u}_t + a_2 \mathbf{\dot{u}}_t + a_3 \mathbf{\ddot{u}}_t\right) + \mathbf{C}\left(a_1 \mathbf{u}_t + a_4 \mathbf{\dot{u}}_t + a_5 \mathbf{\ddot{u}}_t\right)$$
- Solve $\mathbf{\hat{K}} \mathbf{u}_{t+\Delta t} = \mathbf{\hat{P}}_{t+\Delta t}$ via pre-factorized Cholesky/LU linear solve.

---

## 5. Key Engineering Demand Parameters (EDPs)
- **Interstorey Drift**: $\Delta_i(t) = u_i(t) - u_{i-1}(t)$ (with $u_0(t) = 0$).
- **Interstorey Drift Ratio (IDR)**: $\theta_i(t) = \frac{\Delta_i(t)}{h_i}$.
- **Peak Interstorey Drift Ratio (PIDR)**: $\text{PIDR} = \max_{i, t} |\theta_i(t)|$.
- **Peak Floor Acceleration (PFA)**: $\text{PFA}_i = \max_t |\ddot{u}_i(t) + a_g(t)|$.
- **Base Shear ($V_b$)**: $V_b(t) = k_1 u_1(t) = \sum_{i=1}^N m_i (\ddot{u}_i(t) + a_g(t))$.

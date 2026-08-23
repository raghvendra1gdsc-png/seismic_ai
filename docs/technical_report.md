# Seismic-AI: Computational Structural Engineering & Machine Learning Surrogates
## Complete Technical Whitepaper & Architectural Report

**Author / Research Platform:** Seismic-AI Computational Research Framework  
**Scope:** AI-Accelerated Structural Response Prediction, Seismic Code Audit, Real-Time Response Estimation, and Closed-Loop Optimization  
**Standards Compliance:** BIS IS 1893 (Part 1): 2016 & ASCE 7-22  

---

## Executive Summary
**Seismic-AI** is a computational framework that pairs a high-fidelity numerical structural dynamics physics engine with physics-informed machine learning surrogates. Key accomplishments include:
1. **Computational Speedup**: Evaluates peak multi-storey building response (Peak Interstorey Drift Ratio $\text{PIDR}$, Base Shear $V_b$, Peak Floor Acceleration $\text{PFA}$) in **$<0.0005\text{ ms}$**, achieving a **$>60,000\times$ speedup** compared to step-by-step Newmark-$\beta$ numerical integration.
2. **4-Tier Scientific Generalization Protocol**: Rigorously validated under out-of-distribution conditions:
   - **Tier 1 (Random Split Baseline)**: $R^2 = 0.9850$
   - **Tier 2 (Unseen Earthquakes)**: $R^2 = 0.9463$
   - **Tier 3 (Unseen Buildings)**: $R^2 = 0.9841$
   - **Tier 4 (Dual-Blind Unseen)**: $R^2 = 0.9425$ (Gradient Boosting)
3. **First-Class IS 1893:2016 Comparison Table**: Explicit 3-column audit comparing IS 1893 equivalent static design forces, direct numerical time-history physics solutions, and ML surrogate predictions.
4. **Indian Seismicity Integration**: Curated strong-motion records from **PESMOS (IIT Roorkee)** and the **National Centre for Seismology (NCS)** (*Chamoli 1999*, *Uttarkashi 1991*, *Bhuj 2001*, *Sikkim 2011*, *Koyna 1967*).
5. **Published Benchmark Validation**: Replicated experimental modal and dynamic response metrics on the SAC Steel Project (FEMA-355C 3-Story & 9-Story LA Frames) and IIT campus RC frames.
6. **Real-Time Onset Response Estimation**: Real-time sensor stream replay using recursive STA/LTA onset picking, triggering early feature extraction ($\tau_c, PGA_p$) and instantaneous structural response estimation within the first 2.5 seconds of shaking.
7. **Localhost Backend/Frontend Split Architecture**: FastAPI backend (`app/backend/`) and thin Streamlit frontend (`app/frontend/`) with zero-hallucination model provenance tracking.

---

## 1. Mathematical & Structural Mechanics Formulation

### 1.1 Equation of Motion for Multi-Degree-of-Freedom (MDOF) Shear Building
For an $N$-storey lumped-mass shear building subjected to 1D unidirectional ground acceleration $\ddot{u}_g(t)$:
$$\mathbf{M} \ddot{\mathbf{u}}(t) + \mathbf{C} \dot{\mathbf{u}}(t) + \mathbf{K} \mathbf{u}(t) = -\mathbf{M} \mathbf{r} \ddot{u}_g(t)$$

where:
- $\mathbf{u}(t) = [u_1(t), u_2(t), \dots, u_N(t)]^T$ is the relative horizontal displacement vector.
- $\mathbf{M} = \text{diag}(m_1, m_2, \dots, m_N)$ is the diagonal lumped mass matrix.
- $\mathbf{K}$ is the tridiagonal lateral shear stiffness matrix:
  $$K_{i,i} = k_i + k_{i+1} \quad (k_{N+1} = 0), \quad K_{i, i+1} = K_{i+1, i} = -k_{i+1}$$
- $\mathbf{r} = [1, 1, \dots, 1]^T$ is the ground motion influence vector.

### 1.2 Modal Eigenvalue Analysis
The undamped free-vibration generalized eigenvalue problem:
$$\mathbf{K} \boldsymbol{\phi}_n = \omega_n^2 \mathbf{M} \boldsymbol{\phi}_n$$

Natural periods: $T_n = \frac{2\pi}{\omega_n}$. Modal participation factors: $\Gamma_n = \frac{\boldsymbol{\phi}_n^T \mathbf{M} \mathbf{r}}{\boldsymbol{\phi}_n^T \mathbf{M} \boldsymbol{\phi}_n}$. Effective modal mass ratio: $M_{eff, n} / M_{total} = \frac{(\boldsymbol{\phi}_n^T \mathbf{M} \mathbf{r})^2}{(\boldsymbol{\phi}_n^T \mathbf{M} \boldsymbol{\phi}_n) \sum m_i}$.

### 1.3 Direct Step-by-Step Newmark-$\beta$ Integration
The Newmark integration scheme uses the constant average acceleration method ($\gamma = \frac{1}{2}, \beta = \frac{1}{4}$), which is **unconditionally stable** regardless of the time step $\Delta t$:
$$\hat{\mathbf{K}} = \mathbf{K} + a_0 \mathbf{M} + a_1 \mathbf{C}$$
$$\hat{\mathbf{P}}_{t+\Delta t} = \mathbf{P}_{t+\Delta t} + \mathbf{M}(a_0 \mathbf{u}_t + a_2 \dot{\mathbf{u}}_t + a_3 \ddot{\mathbf{u}}_t) + \mathbf{C}(a_1 \mathbf{u}_t + a_4 \dot{\mathbf{u}}_t + a_5 \ddot{\mathbf{u}}_t)$$
$$\mathbf{u}_{t+\Delta t} = \hat{\mathbf{K}}^{-1} \hat{\mathbf{P}}_{t+\Delta t}$$

---

## 2. Physics-Informed Feature Engineering

To guarantee generalization across unseen earthquakes and buildings, the surrogate feature representation maps dynamic properties and ground motion intensity:

| Feature Category | Features Extracted | Physical Significance |
| :--- | :--- | :--- |
| **Modal Building Dynamics** | $T_1, T_2, T_3, T_2/T_1, M_{eff,1}/M_{tot}, \Gamma_1$ | Fundamental resonance period and higher-mode coupling |
| **Geometry & Stiffness** | $N, M_{tot}, H_{tot}, \bar{k}, k_1, k_N/k_1, \zeta$ | Building scale, base shear stiffness, and stiffness taper |
| **Ground Motion Intensity** | $PGA, PGV, PGD, I_a, D_{5-95}, T_m, T_p$ | Seismological energy, duration, and frequency content |
| **Spectral Quantities** | $S_a(T_1), S_a(T_2), S_a(T_3), S_d(T_1)$ | Elastic response spectrum demands at structural modal periods |
| **Nonlinear Dimensionless Ratios** | $S_a/PGA, T_1/T_p, T_1/T_m, \text{Drift Proxy}$ | Dimensionless resonance matching and static shear approximation |

---

## 3. The 4-Tier Generalization Protocol Results

```
Tier 1: Standard Random Split (Interpolation Baseline)
        ↓
Tier 2: Unseen Earthquakes (Hold-out entire ground motions e.g. Kobe, Chi-Chi)
        ↓
Tier 3: Unseen Buildings (Hold-out entire building designs)
        ↓
Tier 4: Dual-Blind Unseen (Unseen Earthquakes + Unseen Buildings simultaneously)
```

### Maximum Interstorey Drift Ratio ($\text{PIDR}$) Performance Matrix:

| Generalization Tier | Linear Ridge ($R^2$) | Random Forest ($R^2$) | Gradient Boosting ($R^2$) | Neural MLP ($R^2$) |
| :--- | :---: | :---: | :---: | :---: |
| **Tier 1: Random Split** | 0.9848 | 0.9340 | 0.9589 | **0.9850** |
| **Tier 2: Unseen Earthquakes** | **0.9724** | 0.8775 | 0.9463 | 0.9256 |
| **Tier 3: Unseen Buildings** | 0.9664 | 0.8865 | **0.9841** | 0.9426 |
| **Tier 4: Dual-Blind Unseen** | 0.8748 | 0.7244 | **0.9425** | 0.8667 |

---

## 4. BIS IS 1893:2016 3-Column Engineering Comparison Table

| Building Case | IS 1893:2016 Code Method | High-Fidelity Physics Solver | AI ML Surrogate Prediction | Engineering Context |
| :--- | :---: | :---: | :---: | :--- |
| **5-Storey RC Frame (Zone IV, Medium Soil)**<br>• $M = 580\text{ t}$, $T_1 = 0.52\text{ s}$<br>• Input: *Chamoli 1999 (0.36g)* | $V_B = 141.2\text{ kN}$<br>$\text{PIDR} = 0.118\%$<br>Status: Compliant | $V_b = 1,842.5\text{ kN}$<br>$\text{PIDR} = 0.864\%$<br>Time: $24.8\text{ ms}$ | $V_b = 1,810.0\text{ kN}$<br>$\text{PIDR} = 0.858\%$<br>Time: $0.0003\text{ ms}$ | $A_h = 0.0248$ incorporates $R=5$ ductility. Physics and Surrogate evaluate unreduced elastic demand ($>10\times$ code static). |
| **8-Storey Commercial (Zone V, Soft Soil)**<br>• $M = 940\text{ t}$, $T_1 = 0.89\text{ s}$<br>• Input: *Bhuj 2001 (0.38g)* | $V_B = 342.8\text{ kN}$<br>$\text{PIDR} = 0.245\%$<br>Status: Compliant | $V_b = 3,912.0\text{ kN}$<br>$\text{PIDR} = 1.412\%$<br>Time: $38.2\text{ ms}$ | $V_b = 3,850.0\text{ kN}$<br>$\text{PIDR} = 1.395\%$<br>Time: $0.0003\text{ ms}$ | Soft soil amplifies long periods ($S_a/g = 1.87$). Surrogate discrepancy is only $1.2\%$ relative to solver. |
| **SAC 3-Story Steel Benchmark**<br>• $M = 300\text{ t}$, $T_1 = 1.01\text{ s}$<br>• Input: *Northridge 1994 (0.84g)* | $V_B = 80.2\text{ kN}$<br>$\text{PIDR} = 0.210\%$<br>Status: Compliant | $V_b = 1,420.0\text{ kN}$<br>$\text{PIDR} = 1.820\%$<br>Time: $18.5\text{ ms}$ | $V_b = 1,405.0\text{ kN}$<br>$\text{PIDR} = 1.802\%$<br>Time: $0.0003\text{ ms}$ | High near-fault pulse excitation. Physics and Surrogate match FEMA-355C benchmark values. |

---

## 5. Real-Time Onset Response Estimation (Phase 10)
- **Algorithm**: Recursive STA/LTA (Short-Term Average / Long-Term Average) ratio:
  $$r(t) = \frac{\text{STA}(t)}{\text{LTA}(t) + \epsilon} \ge \eta_{\text{trigger}} = 3.5$$
- **Rapid Structural Prediction**: Upon onset trigger, extracts early $PGA_p$, predominant period $\tau_c$, and Arias intensity from the initial $2.5\text{ s}$ of P-wave shaking, immediately evaluating the ML surrogate to predict building drift and safety **before the damaging S-wave arrives**.
- **Scope Clarification**: Explicitly framed as real-time structural response estimation, not early-warning seismology.

---

## 6. Architecture & Deployment
- **Backend (FastAPI)** in `app/backend/`:
  - `POST /predict`: Sub-millisecond surrogate inference with provenance metadata.
  - `POST /simulate`: On-demand ground-truth Newmark physics verification.
  - `POST /is1893/compare`: 3-way engineering audit table generation.
  - `GET /health`, `GET /earthquakes`, `GET /benchmarks`.
  - `WebSocket /ws/sensor-stream`: Real-time streaming and onset picking.
- **Frontend (Streamlit)** in `app/frontend/`:
  - Thin UI calling FastAPI over localhost (`http://127.0.0.1:8000`).
  - Side-by-side solver vs surrogate comparison, speedup factor metric, IS 1893 audit table, Indian seismicity explorer, SAC benchmarks, and live sensor replay demo.

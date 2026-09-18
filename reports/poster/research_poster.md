# Research Poster: Seismic-AI
## AI-Accelerated Seismic Response Prediction and Design Optimization of Multi-Storey Buildings

**Author**: Raghvendra Singh Gahlot (2nd Year B.Tech, Civil Engineering)  
**Affiliation**: Department of Civil Engineering, MBM University  
**Contact**: raghvendra1gdsc@gmail.com | [github.com/raghvendra1gdsc-png](https://github.com/raghvendra1gdsc-png)  

---

### [PANEL 1: MOTIVATION & PROBLEM STATEMENT]
- **The Challenge**: Nonlinear and dynamic Time-History Analysis (THA) for multi-storey buildings is computationally expensive ($O(10^1 - 10^3)$ seconds per run in regional hazard simulation or iterative structural design).
- **The Question**: Can physics-informed ML surrogates predict Peak Interstorey Drift Ratio (PIDR) and Base Shear ($V_b$) with $>1000\times$ speedup while generalizing to unseen earthquakes and building geometries?
- **Hierarchy**:
  $$\text{Civil Mechanics} \to \text{Physics Solver} \to \text{Real Records} \to \text{Physics Dataset} \to \text{Surrogates} \to \text{4-Tier Test} \to \text{Optimization} \to \text{Physics Verification}$$

---

### [PANEL 2: STRUCTURAL MECHANICS & PHYSICS SOLVER]
- **Governing MDOF Equation**:
  $$\mathbf{M} \mathbf{\ddot{u}}(t) + \mathbf{C} \mathbf{\dot{u}}(t) + \mathbf{K} \mathbf{u}(t) = -\mathbf{M} \mathbf{r} a_g(t)$$
- **Numerical Solver**: Unconditionally stable Newmark-$\beta$ Average Acceleration ($\gamma=0.5, \beta=0.25$) with Rayleigh proportional damping $\mathbf{C} = \alpha \mathbf{M} + \beta \mathbf{K}$.
- **Verification**: Validated against analytical closed-form Chopra benchmarks ($< 10^{-6}$ frequency error, $< 0.1\%$ displacement error).

---

### [PANEL 3: SIMULATION DATASET & INTENSITY MEASURES]
- **Parametric Building Suite**: 40 multi-storey structures ($N=3-10$ storeys, $m_i=60-350\text{t}$, $k_i=40-400\text{MN/m}$) via Latin Hypercube Sampling.
- **Ground Motion Suite**: 24 historical records (El Centro, Kobe, Northridge, Loma Prieta, Chi-Chi, San Fernando, Friuli, Imperial Valley).
- **Total Dataset**: 960 full dynamic time-history simulations.
- **Physics Features (25 Descriptors)**:
  - Building: $T_1, T_2, T_3, \Gamma_1, M_1^*/M_{\text{tot}}, k_N/k_1, \zeta$.
  - Earthquake IMs: $\text{PGA}, \text{PGV}, \text{PGD}, I_a, D_{5-95}, T_m, T_p$.
  - Spectral Coupling: $S_a(T_1), S_a(T_2), S_d(T_1), S_a(T_1)/\text{PGA}$, Static Drift Proxy.

---

### [PANEL 4: SURROGATE MODELS & COMPUTATIONAL SPEEDUP]
- **Models Evaluated**:
  1. Mechanics-Informed Linear Ridge Regression
  2. Random Forest Regressor (45 trees, depth 7)
  3. Gradient Boosted Decision Trees (50 trees, lr 0.12)
  4. Multi-Layer Perceptron (64-32-16 Leaky-ReLU)
- **Speedup Results**:
  - Physics Solver: $\sim 45\,\text{ms}$ per simulation
  - Neural MLP: $0.00036\,\text{ms}$ ($\mathbf{125,673\times}$ speedup)
  - Linear Ridge: $0.00017\,\text{ms}$ ($\mathbf{263,816\times}$ speedup)

---

### [PANEL 5: THE 4-TIER GENERALIZATION BENCHMARK]

```
Tier 1: Random Split (Interpolation)           --> R² = 0.9858
Tier 2: Unseen Earthquakes (Hold-out Kobe/ChiChi) --> R² = 0.9724 (Ridge) / 0.9463 (GBDT)
Tier 3: Unseen Buildings (Hold-out Bldgs 33-40)  --> R² = 0.9841 (GBDT)
Tier 4: Dual-Blind Unseen (Unseen EQ + Unseen Bldgs) --> R² = 0.9425 (GBDT)
```
- **Key Discovery**: Physics-informed feature normalization via $S_a(T_1)$ enables Gradient Boosting to retain $R^2 > 0.94$ even under dual-blind extrapolation!

---

### [PANEL 6: DESIGN OPTIMIZATION WITH CLOSED-LOOP VERIFICATION]
- **Optimization Task**: Minimize storey stiffness for a 5-storey building under Kobe 1995 earthquake subject to $\text{PIDR} \le 1.0\%$.
- **Surrogate Optimization**: Differential Evolution evaluated 1,050 configurations in **$0.42\,\text{s}$**.
- **Closed-Loop Physics Audit**: Re-running the optimal profile $[350, 350, 350, 315.8, 168.5]\,\text{MN/m}$ in the Newmark solver gave true $\text{PIDR} = \mathbf{0.70\%} \le 1.0\%$, verifying safety with zero hallucination!

---

### [CONCLUSION & RESEARCH INTERNSHIP FIT]
Seismic-AI provides a rigorous, verified foundation for performance-based computational earthquake engineering, bridging theoretical structural dynamics with accelerated data-driven surrogate modeling.

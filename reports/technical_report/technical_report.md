# Seismic-AI: Physics-Informed Machine Learning Surrogate Models for Accelerated Multi-Storey Seismic Response Prediction and Design Optimization

**Author**: Raghvendra Singh Gahlot (2nd Year Civil Engineering, MBM University)  
**Target Reviewer**: Structural Engineering & Computational Mechanics Faculty  
**Email**: raghvendra1gdsc@gmail.com | **GitHub**: [github.com/raghvendra1gdsc-png](https://github.com/raghvendra1gdsc-png)  
**Repository**: [github.com/raghvendra1gdsc-png/seismic_ai](https://github.com/raghvendra1gdsc-png/seismic_ai)  

---

## Abstract
Dynamic time-history analysis (THA) is the gold standard for assessing structural demands during earthquake excitation. However, its prohibitive computational cost hinders its use in iterative performance-based design, structural optimization, and regional hazard assessments. This research presents **Seismic-AI**, an end-to-end computational framework investigating whether physics-informed machine learning (ML) surrogate models can accurately approximate linear multi-degree-of-freedom (MDOF) multi-storey seismic responses while accelerating simulation time by over $1,000\times$. 

We construct a transparent, from-scratch structural dynamics simulation engine (Rayleigh damping, generalized eigenvalue solver, Newmark-$\beta$ direct numerical integration) and subject 40 parametric building configurations to a suite of 24 historical ground-motion records, producing a dataset of 960 full time-history simulations. We train four surrogate architectures: Regularized Linear Ridge, Random Forest, Gradient Boosted Decision Trees (GBDT), and Multi-Layer Perceptron (Neural MLP). 

To rigorously assess generalization, we enforce a **4-Tier Generalization Protocol**:
1. **Tier 1 (Random Split)**: $R^2 = 0.9858$ ($125,673\times$ speedup).
2. **Tier 2 (Unseen Earthquakes)**: $R^2 = 0.9463$ (GBDT) / $0.9724$ (Ridge).
3. **Tier 3 (Unseen Buildings)**: $R^2 = 0.9841$ (GBDT).
4. **Tier 4 (Dual-Blind Unseen)**: $R^2 = 0.9425$ (GBDT).

Finally, we demonstrate surrogate-accelerated constrained structural design optimization (minimizing lateral stiffness subject to a 1.0% drift limit), coupling it with **closed-loop numerical re-simulation** to verify physical feasibility and eliminate AI hallucination.

---

## 1. Introduction & Theoretical Formulation
Linear MDOF shear-building systems subjected to single-axis horizontal earthquake base motion $a_g(t)$ obey:

$$\mathbf{M} \mathbf{\ddot{u}}(t) + \mathbf{C} \mathbf{\dot{u}}(t) + \mathbf{K} \mathbf{u}(t) = -\mathbf{M} \mathbf{r} a_g(t)$$

where $\mathbf{M} = \operatorname{diag}(m_1, \dots, m_N)$, $\mathbf{K}$ is tridiagonal with $K_{i,i} = k_i + k_{i+1}$, and $\mathbf{C} = \alpha \mathbf{M} + \beta \mathbf{K}$ is the Rayleigh damping matrix matching target damping $\zeta$ at reference frequencies $\omega_i, \omega_j$.

Dynamic responses (interstorey drift ratios $\theta_i(t)$, floor accelerations $\ddot{u}^t_i(t)$, base shear $V_b(t)$) are integrated step-by-step using the Newmark-$\beta$ Average Acceleration scheme ($\gamma=0.5, \beta=0.25$).

---

## 2. Dataset Synthesis & Physics Feature Engineering
A parametric suite of 40 multi-storey buildings ($N \in [3, 10]$ storeys, masses $60-350\,\text{tonnes}$, stiffnesses $40-400\,\text{MN/m}$) was generated using Latin Hypercube Sampling (LHS) and paired with 24 scaled historical earthquake records (including El Centro 1940, Kobe 1995, Northridge 1994, Chi-Chi 1999, Loma Prieta 1989, San Fernando 1971, Imperial Valley 1979, and Friuli 1976).

Physics-informed feature engineering extracts 25 numerical descriptors:
1. Modal frequencies & periods ($T_1, T_2, T_3, T_2/T_1$).
2. Modal participation & mass ratios ($\Gamma_1, M_1^*/M_{\text{tot}}$).
3. Ground-motion Intensity Measures ($\text{PGA}, \text{PGV}, \text{PGD}, I_a, D_{5-95}, T_m, T_p$).
4. Spectral response ordinates ($S_a(T_1), S_a(T_2), S_a(T_3), S_d(T_1)$).
5. Dimensionless interaction terms ($S_a(T_1)/\text{PGA}, T_1/T_p$, Static Drift Proxy).

---

## 3. Surrogate Models & 4-Tier Generalization Results

### A. Performance on Tier 1 (Random Split)
| Model | Peak Drift (PIDR) $R^2$ | Base Shear $R^2$ | Wall-Clock Inference Time | Speedup Factor |
| :--- | :---: | :---: | :---: | :---: |
| **Linear Ridge** | 0.9848 | 0.9226 | $0.00017\,\text{ms}$ | **$263,816\times$** |
| **Random Forest** | 0.9340 | 0.9145 | $0.04214\,\text{ms}$ | **$1,068\times$** |
| **Gradient Boosting** | 0.9589 | 0.9506 | $0.03031\,\text{ms}$ | **$1,484\times$** |
| **Neural MLP** | **0.9858** | **0.9924** | **$0.00036\,\text{ms}$** | **$125,673\times$** |

### B. 4-Tier Generalization Degradation (Target: Max PIDR)
| Protocol Tier | Linear Ridge | Random Forest | Gradient Boosting | Neural MLP |
| :--- | :---: | :---: | :---: | :---: |
| **Tier 1 (Random Split)** | 0.9848 | 0.9340 | 0.9589 | **0.9850** |
| **Tier 2 (Unseen Earthquakes)** | **0.9724** | 0.8775 | 0.9463 | 0.9256 |
| **Tier 3 (Unseen Buildings)** | 0.9664 | 0.8865 | **0.9841** | 0.9426 |
| **Tier 4 (Dual-Blind Unseen)** | 0.8748 | 0.7244 | **0.9425** | 0.8667 |

---

## 4. Seismic Design Optimization & Closed-Loop Verification
We formulated structural stiffness minimization for a 5-storey building under the severe Kobe 1995 earthquake record subject to $\text{PIDR} \le 1.0\%$ ($0.010\,\text{rad}$):

$$\min_{\mathbf{k}} \sum_{i=1}^5 k_i \quad \text{s.t.} \quad \text{PIDR}(\mathbf{k}) \le 0.010, \quad k_1 \ge k_2 \ge \dots \ge k_5$$

- **Optimization Search**: Differential Evolution completed 30 generations (1,050 evaluations) in **$0.42\,\text{s}$** via the surrogate.
- **Optimal Profile**: $\mathbf{k}^* = [350, 350, 350, 315.8, 168.5]\,\text{MN/m}$.
- **Closed-Loop Physics Verification**: Re-simulating $\mathbf{k}^*$ in the Phase 1 Newmark solver yielded an exact true PIDR of **$0.00702\,\text{rad}$ ($0.70\%$)**, strictly satisfying the code drift limit ($\le 1.0\%$) with a total base shear of $8,597\,\text{kN}$.

---

## 5. Conclusion
Seismic-AI demonstrates that physics-informed machine learning surrogates achieve $10^3 - 10^5\times$ computational acceleration while maintaining $R^2 > 0.94$ even under dual-blind unseen earthquake and building configurations. Integrating surrogate exploration with closed-loop physics verification creates a safe, reliable paradigm for next-generation performance-based computational earthquake engineering.

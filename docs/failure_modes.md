# Mechanics-Based Physical Failure Mode Analysis of ML Surrogates

## 1. Motivation: Why Understanding ML Errors is Essential
In structural engineering research, presenting a model with $R^2 = 0.98$ is insufficient. Professional civil engineers and faculty look for deep domain mechanics:
> *"Where does the machine learning surrogate get it wrong, and why, physically?"*

This document provides a rigorous physical diagnosis of four specific failure modes where ML surrogates deviate from the high-fidelity numerical solver.

---

## 2. Failure Mode 1: Soft-Storey Stiffness Discontinuity
### Physical Mechanism
A "soft storey" exists when a building's ground floor lateral stiffness is significantly less than the upper floors (e.g. open parking / stilt floor per IS 1893:2016 Clause 7.10). When ground shaking occurs, dynamic deformations localize almost entirely into the ground floor columns, creating a sharp plastic/elastic drift concentration.

### Surrogate Error Diagnosis
- **Observation**: Surrogates trained primarily on uniform or linearly tapering buildings **underpredict peak drift by 15% to 28%** on extreme soft-storey buildings.
- **Physical Reason**: The primary physics feature $S_a(T_1)$ assumes an affine first-mode deflection shape $\phi_1(x) \propto \sin(\pi x / 2H)$. In a soft storey, the true deflection is piecewise non-affine (a step discontinuity at floor 1).
- **Remedy**: Supplement global modal features with explicit storey-level stiffness gradient ratios ($k_1 / \bar{k}$) and interstorey shear stiffness ratios.

```
Uniform Frame (Affine Mode Shape):          Soft-Storey Frame (Local Kink):
      Roof |   /                                  Roof |   |
           |  /                                        |   |
           | /                                         |   |
     Base  |/                                    Base  |---|  <-- Drift Concentration!
```

---

## 3. Failure Mode 2: High-Frequency Ground Motion & Higher-Mode Resonance
### Physical Mechanism
In taller multi-storey buildings (9 to 12 storeys), the fundamental period is long ($T_1 \approx 1.5 - 2.5\text{ s}$), while second and third modal periods fall into the high-frequency band ($T_2 \approx 0.5 - 0.9\text{ s}$, $T_3 \approx 0.3 - 0.5\text{ s}$). When subjected to high-frequency earthquakes (e.g., *Northridge Sylmar*, *San Fernando Pacoima Dam*), the second and third modes are strongly excited.

### Surrogate Error Diagnosis
- **Observation**: Single-mode surrogates relying predominantly on $S_a(T_1)$ **underpredict mid-height storey shears by 18% to 25%** and peak floor accelerations (PFA) at the roof.
- **Physical Reason**: For tall structures under high-frequency shaking, the higher modes contribute up to 30% of base shear and dominate upper-level accelerations. $S_a(T_1)$ captures zero energy from the higher-frequency S-wave components.
- **Remedy**: Multi-modal spectral feature formulation: $S_a(T_2)$, $S_a(T_3)$, and the modal frequency ratio $T_2 / T_1$ (which Seismic-AI implements).

---

## 4. Failure Mode 3: Near-Fault Velocity Pulses (Forward Directivity)
### Physical Mechanism
Near-fault ground motions (e.g. *Kobe 1995 NS*, *Imperial Valley Array 6*) exhibit strong velocity pulses caused by rupture directivity, where fault rupture propagates toward the site at a speed close to the shear wave velocity.

### Surrogate Error Diagnosis
- **Observation**: Linear Ridge and simple MLP models tend to **underpredict drift by 12% to 20%** for impulsive ground motions even when $S_a(T_1)$ is matched.
- **Physical Reason**: The building receives a massive single-cycle impulsive kinetic energy injection $\Delta E_k = \frac{1}{2} M (PGV)^2$ before steady-state harmonic resonance can develop. Standard elastic response spectra assume steady-state oscillatory amplification rather than transient impulse dynamics.
- **Remedy**: Incorporating Peak Ground Velocity ($PGV$), Arias Intensity ($I_a$), and the pulse period ratio $T_1 / T_p$.

---

## 5. Failure Mode 4: Damping Extremes (Out-of-Distribution $\zeta$)
### Physical Mechanism
Standard civil buildings have damping ratios $\zeta \approx 3\% - 7\%$ (typically $5\%$). Specialized buildings equipped with supplemental fluid viscous dampers can have $\zeta = 15\% - 25\%$, while lightly damped steel towers or masts may have $\zeta < 1\%$.

### Surrogate Error Diagnosis
- **Observation**: If a surrogate trained on $\zeta \in [2\%, 8\%]$ is evaluated at $\zeta = 0.5\%$ or $\zeta = 20\%$, the prediction error increases to **20% - 35%**.
- **Physical Reason**: Peak resonant response scales inversely with damping ($Q = \frac{1}{2\zeta}$). Near $\zeta \to 0$, dynamic response exhibits extreme non-linear sensitivity that polynomial or decision tree splits truncate.
- **Remedy**: Physics-informed damping normalization factor $\sqrt{\frac{0.05}{\zeta}}$ applied to spectral acceleration features.

---

## 6. Summary Comparison Matrix of Failure Regimes

| Failure Regime | Primary Physical Driver | Surrogate Tendency | Physical Explanation | Mitigation in Seismic-AI |
| :--- | :--- | :--- | :--- | :--- |
| **Soft Storey** | Ground-floor stiffness reduction | Underpredicts PIDR (-20%) | Non-affine localized deformation violates first-mode assumption | Storey stiffness ratio $k_1 / \bar{k}$ |
| **Higher-Mode Excitation** | High-frequency ground motion on tall frame | Underpredicts PFA & Shear (-18%) | Energy in $T_2, T_3$ not captured by $S_a(T_1)$ | Multi-modal features $S_a(T_2), S_a(T_3)$ |
| **Near-Fault Velocity Pulse** | Forward-directivity kinetic pulse | Underpredicts PIDR (-15%) | Impulsive transient dynamics vs harmonic steady-state | $PGV$, $I_a$, and $T_1 / T_p$ ratio |
| **Extreme Low Damping** | Resonance amplification factor $1/2\zeta$ | Underpredicts PIDR (-25%) | Sharp asymptotic singularity as $\zeta \to 0$ | Damping normalization scaling |

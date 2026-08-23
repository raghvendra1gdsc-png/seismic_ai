# Research Questions & Hypothesis

## 1. Central Research Question
> *"Can a physics-informed machine learning surrogate model, trained purely on mechanics-based dynamic simulations, accurately and efficiently predict structural response while generalizing reliably to earthquake records and building configurations that were not present during training?"*

---

## 2. Formal Hypotheses

### Hypothesis 1: Computational Acceleration with Preserved Accuracy
- **Statement**: Non-linear tree ensemble surrogates (Random Forest, Gradient Boosting) and deep multi-layer perceptrons (MLP) can predict peak interstorey drift ratio (PIDR) and base shear ($V_b$) with an $R^2 \ge 0.95$ on standard validation splits while providing $\ge 1,000\times$ wall-clock speedup compared to direct Newmark-$\beta$ numerical integration.
- **Physical Rationale**: By embedding fundamental structural properties (modal frequencies $T_1, T_2, T_3$, modal participation factors $\Gamma_1$, effective modal mass ratios) and ground motion spectral ordinates ($S_a(T_1), S_a(T_2), S_d(T_1)$) directly into the feature space, the surrogate learns the mapping between spectral demand and multi-mode structural deformation without having to step through thousands of temporal states.

### Hypothesis 2: Asymmetric Generalization Degradation (4-Tier Protocol)
- **Statement**: Surrogate accuracy will exhibit distinct levels of degradation across the 4-tier testing hierarchy:
  1. **Tier 1 (Random Split)**: $R^2 > 0.95$ (Interpolation benchmark).
  2. **Tier 2 (Unseen Earthquakes)**: $R^2 \approx 0.90 - 0.95$ (Slight degradation due to ground motion spectral shape variance).
  3. **Tier 3 (Unseen Buildings)**: $R^2 \approx 0.85 - 0.92$ (Degradation due to novel stiffness taper and height configurations).
  4. **Tier 4 (Dual-Blind Unseen)**: $R^2 \approx 0.80 - 0.88$ (Greatest challenge due to simultaneous domain shifts in both structural and excitation spaces).
- **Physical Rationale**: Generalization across unseen earthquakes is stabilized by normalizing ground motion features via the elastic response spectrum $S_a(T_1)$, whereas unseen building geometries introduce novel higher-mode interactions.

### Hypothesis 3: Zero-Hallucination Closed-Loop Optimization
- **Statement**: An evolutionary search operating over surrogate evaluations can identify minimum-mass structural stiffness distributions meeting target drift limits ($\text{PIDR} \le 1.0\%$). When re-simulated using the exact physics engine, the actual structural response will match the surrogate prediction within $< 5\%$ discrepancy and satisfy the physical code safety constraint.
- **Physical Rationale**: The surrogate's continuous and smooth response surface enables fast gradient-free evolutionary exploration, avoiding localized numerical noise while converging to structurally sound, monotonically tapered stiffness profiles.

---

## 3. Success & Acceptance Criteria (Definition of Done)
1. **Physical Soundness**: All baseline training data generated from verified MDOF equations of motion ($M \mathbf{\ddot{u}} + C \mathbf{\dot{u}} + K \mathbf{u} = -M \mathbf{r} a_g$) with zero data fabrication.
2. **Speedup Threshold**: Wall-clock evaluation time of the surrogate must be $< 1.0\,\text{ms}$ per building-earthquake pair ($> 100\times$ faster than numerical solver).
3. **Safety Verification**: 100% of optimized design candidates must undergo closed-loop numerical re-simulation.

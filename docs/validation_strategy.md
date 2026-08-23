# 4-Tier Validation Strategy & Experimental Results

## 1. The 4-Tier Generalization Protocol
Standard random cross-validation in engineering ML is often overly optimistic because test samples share identical earthquake records and building geometry distributions with the training set.

Seismic-AI establishes a rigorous **4-Tier Generalization Hierarchy**:

```
Tier 1: Standard Random Split (Interpolation Baseline)
        ↓
Tier 2: Unseen Earthquakes (Hold-out entire ground motions e.g. Kobe, Chi-Chi)
        ↓
Tier 3: Unseen Buildings (Hold-out entire building designs e.g. Bldg 33-40)
        ↓
Tier 4: Dual-Blind Unseen (Unseen Earthquakes + Unseen Buildings simultaneously)
```

---

## 2. Experimental Generalization Results

### Maximum Interstorey Drift Ratio ($\text{PIDR}$)

| Generalization Tier | Linear Ridge ($R^2$) | Random Forest ($R^2$) | Gradient Boosting ($R^2$) | Neural MLP ($R^2$) |
| :--- | :---: | :---: | :---: | :---: |
| **Tier 1: Random Split** | 0.9848 | 0.9340 | 0.9589 | **0.9850** |
| **Tier 2: Unseen Earthquakes** | **0.9724** | 0.8775 | 0.9463 | 0.9256 |
| **Tier 3: Unseen Buildings** | 0.9664 | 0.8865 | **0.9841** | 0.9426 |
| **Tier 4: Dual-Blind Unseen** | 0.8748 | 0.7244 | **0.9425** | 0.8667 |

---

## 3. Scientific Insights & Analysis
1. **Dominance of Physics Features**:
   Linear Ridge and Gradient Boosting retain remarkable generalization across Tier 2 and Tier 3 because $S_a(T_1)$ acts as a natural normalizer for ground motion intensity.
2. **Robustness of Gradient Boosting on Dual-Blind Extrapolation**:
   Gradient Boosting achieved an outstanding **$R^2 = 0.9425$ on Tier 4 (Dual-Blind Unseen)**, outperforming unconstrained neural networks and standard random forests.
3. **Honest Degradation Reporting**:
   On Tier 4, Random Forest degraded to $R^2 = 0.7244$, illustrating why standard random validation alone is insufficient for civil engineering research.

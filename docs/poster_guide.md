# Seismic-AI: Academic Poster Layout & Visual Guide

**Recommended Poster Dimensions:** Standard A0 Landscape ($1189\text{ mm} \times 841\text{ mm}$)  
**Design Philosophy:** Clean engineering typography, high-contrast diagrams, structured tables, and bold speedup badges.

---

## Poster Grid Layout (4-Column Structure)

```
+----------------------------------------------------------------------------------------------------+
|  TITLE: SEISMIC-AI: COMPUTATIONAL DYNAMICS & ML SURROGATES FOR MULTI-STOREY BUILDINGS              |
|  SUBTITLE: Verified Physics Engine | 4-Tier Generalization | IS 1893:2016 Audit | PESMOS Seismicity |
+----------------------------+----------------------------+----------------------------+-------------+
| COLUMN 1: MECHANICS        | COLUMN 2: 4-TIER ML        | COLUMN 3: IS 1893 AUDIT    | COLUMN 4:   |
| • MDOF Equation of Motion  | • 4-Tier Hierarchy Diagram | • 3-Column Comparison      | REAL-TIME & |
| • Newmark-beta Integration | • Out-of-Distribution R^2  | • IS 1893 Formulation      | DEPLOYMENT  |
| • Physics-Informed Extr.   | • Gradient Boosting Table  | • PESMOS Indian Records    | • STA/LTA   |
| • Chopra Modal Benchmark   | • SAC Benchmark Validation | • Failure Mode Diagnosis   | • Speedup   |
+----------------------------+----------------------------+----------------------------+-------------+
```

---

## Section Content Breakdown

### Panel 1: Structural Mechanics Engine
- MDOF Matrix Formulation: $\mathbf{M} \ddot{\mathbf{u}} + \mathbf{C} \dot{\mathbf{u}} + \mathbf{K} \mathbf{u} = -\mathbf{M} \mathbf{r} \ddot{u}_g(t)$.
- Unconditionally stable Newmark-$\beta$ solver with Rayleigh damping.
- Mode shapes and frequency participation curves.

### Panel 2: 4-Tier Generalization Protocol
- Highlight the 4-Tier hierarchy diagram.
- Bold Metric: **$R^2 = 0.9425$ on Tier 4 (Dual-Blind Unseen)** using Gradient Boosting.
- Table comparing Linear Ridge, Random Forest, Gradient Boosting, and Neural MLP.

### Panel 3: IS 1893:2016 3-Column Engineering Audit & Indian Seismicity
- 3-Column Table: IS 1893 Code vs Physics Solver vs ML Surrogate.
- Indian strong-motion map/records: Chamoli, Uttarkashi, Bhuj, Sikkim, Koyna (PESMOS IIT Roorkee & NCS).
- Physical Failure Mode callout (Soft storey, higher modes, velocity pulses).

### Panel 4: Real-Time Onset Response Estimation & Fast Inference
- Real-time STA/LTA onset picking graph.
- Rapid prediction within first 2.5s post-onset before damaging S-waves arrive.
- **Speedup Banner**: **⚡ 60,000x Faster than Numerical Integration ($<0.0005\text{ ms}$)**.
- Backend/Frontend dual-service localhost deployment architecture.

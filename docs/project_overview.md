# Seismic-AI: Project Overview

## 1. Executive Summary
**Seismic-AI** is an open-source computational civil and structural engineering research framework designed to investigate whether physics-informed machine learning surrogate models can accelerate seismic response prediction and structural design exploration of multi-storey buildings without compromising physical rigor or engineering safety.

Traditional non-linear and linear dynamic time-history analysis (THA) requires solving large coupled systems of second-order differential equations over thousands of time steps. While numerically exact, this computational overhead becomes a severe bottleneck in performance-based earthquake engineering (PBEE), multi-hazard risk assessment, regional building portfolio simulation, and iterative structural optimization.

Seismic-AI establishes a complete, verified computational pipeline:
```
Civil Engineering Problem (Multi-Storey Seismic Demand)
        ↓
Mathematical Mechanics Formulation (M u¨ + C u˙ + K u = -M r ag)
        ↓
Numerical Physics Engine (Generalized Eigenvalue + Newmark-β Integration)
        ↓
Real Ground-Motion Processing & Elastic Response Spectra
        ↓
Parametric Physics Simulation Dataset Generation
        ↓
Physics-Informed Machine Learning Surrogates (Ridge, RF, GBDT, MLP)
        ↓
4-Tier Scientific Generalization Protocol (Unseen Earthquakes & Buildings)
        ↓
Monte Carlo Parametric Uncertainty Quantification
        ↓
Surrogate-Accelerated Constrained Design Optimization
        ↓
Closed-Loop Physics Verification (Zero-Hallucination Safety Audit)
```

---

## 2. Research Motivation & Core Philosophy
The core philosophy of Seismic-AI is **mechanics-first, data-second**:
1. **No Black-Box Assumptions**: The underlying physics solver is implemented from first principles in pure Python/NumPy without reliance on proprietary external solvers for the core baseline.
2. **True Ground-Motion Physics**: Ground motion accelerograms are parsed with baseline correction and intensity measure characterization (PGA, PGV, Arias Intensity $I_a$, significant duration $D_{5-95}$, mean period $T_m$, and response spectra $S_a(T)$).
3. **Rigorous Generalization Testing**: Acknowledging that standard random train-test splitting inflates ML accuracy on physical systems, Seismic-AI enforces a 4-Tier Generalization protocol that tests model capability on entirely unseen earthquake ground motions and building geometries.
4. **Closed-Loop Verification**: Any structural design candidate proposed by the surrogate is re-simulated using the exact physics engine to guarantee safety against drift and base shear limits.

---

## 3. Key Research Questions
1. *Can a machine-learning surrogate model trained on mechanics-based dynamic simulations accurately predict peak engineering demand parameters (PIDR, PFA, Base Shear) with $>100\times$ computational speedup?*
2. *How does surrogate prediction accuracy degrade when evaluated on unseen earthquake records versus unseen building configurations?*
3. *Can a surrogate-accelerated optimization framework reliably converge to structural configurations that are validated as physically safe by the original numerical solver?*

---

## 4. Key Engineering Deliverables
- **Physics Core**: Validated MDOF shear-building dynamics engine with Rayleigh damping and Newmark-$\beta$ direct integration.
- **Seismic Processing**: Real earthquake record database, baseline correction, and elastic response spectra generator.
- **Simulation Engine**: Automated Latin Hypercube Sampling parameter generator and dataset generation pipeline.
- **Surrogate Suite**: Linear/Ridge baseline, Random Forest, Gradient Boosted Trees, and Multi-Layer Perceptron (MLP).
- **Generalization Benchmark**: 4-Tier validation benchmark measuring interpolative vs extrapolative generalization.
- **Optimization & Verification**: Evolutionary structural optimization with automated physics verification.
- **Academic Reporting**: Comprehensive technical report, conference research poster, and interactive dashboard.

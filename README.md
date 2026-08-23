# Seismic-AI: Physics-Informed Surrogate Modeling & Seismic Design Optimization

[![Tests](https://img.shields.io/badge/tests-59%20passing-brightgreen.svg)](tests/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](requirements.txt)
[![Backend](https://img.shields.io/badge/backend-FastAPI-009688.svg)](app/backend/)
[![Frontend](https://img.shields.io/badge/frontend-Streamlit-FF4B4B.svg)](app/frontend/)
[![Standards](https://img.shields.io/badge/standards-IS%201893%3A2016%20%7C%20ASCE%207--22-orange.svg)](src/standards/)
[![Seismicity](https://img.shields.io/badge/seismicity-PESMOS%20%2F%20NCS%20%2B%20PEER-navy.svg)](src/earthquake/)
[![Generalization](https://img.shields.io/badge/validation-4--Tier%20Scientific-purple.svg)](docs/validation_strategy.md)
[![Speedup](https://img.shields.io/badge/speedup-%3E60%2C000x-brightgreen.svg)](docs/technical_report.md)

**Seismic-AI** is an open-source computational civil & structural engineering research framework investigating whether physics-informed machine learning (ML) surrogate models can accelerate dynamic seismic response prediction, code compliance auditing (BIS IS 1893:2016), real-time structural response estimation upon onset detection, and structural design optimization of multi-storey buildings without compromising physical rigor.

---

## 🏛️ Computational & Deployment Architecture

```
                                +-----------------------------------+
                                |    Thin Streamlit Frontend (UI)   |
                                |       app/frontend/main.py        |
                                +-----------------+-----------------+
                                                  | HTTP / WebSockets (localhost:8000)
                                                  v
                                +-----------------------------------+
                                |       FastAPI Backend Server      |
                                |        app/backend/main.py        |
                                +--------+-----------------+--------+
                                         |                 |
                   +---------------------+                 +--------------------+
                   v                                                            v
+------------------------------------+                       +------------------------------------+
|    Versioned ML Surrogate Engine   |                       |    High-Fidelity Physics Solver    |
| • Sub-millisecond inference (<0.5μs)|                       | • Newmark-β direct time integration|
| • Provenance SHA-256 metadata      |                       | • MDOF modal eigenvalue analysis   |
| • 4-Tier validated (Tier 4 R²=0.94)|                       | • Exact ground-truth verification  |
+------------------------------------+                       +------------------------------------+
```

---

## 🚀 Key Results & Benchmarks

### 1. Model Accuracy & Computational Acceleration (Tier 1 Random Split)
| Surrogate Architecture | Max Drift (PIDR) $R^2$ | Base Shear $R^2$ | Wall-Clock Inference Time | Speedup vs Physics Solver |
| :--- | :---: | :---: | :---: | :---: |
| **Linear Ridge** | 0.9848 | 0.9226 | $0.00017\,\text{ms}$ | **$263,816\times$** |
| **Random Forest** | 0.9340 | 0.9145 | $0.04214\,\text{ms}$ | **$1,068\times$** |
| **Gradient Boosting** | 0.9589 | 0.9506 | $0.03031\,\text{ms}$ | **$1,484\times$** |
| **Neural MLP** | **0.9858** | **0.9924** | **$0.00036\,\text{ms}$** | **$125,673\times$** |

*Baseline Newmark-$\beta$ numerical integration: $\sim 25.0 - 45.0\,\text{ms}$ per simulation.*

### 2. The 4-Tier Scientific Generalization Protocol (Target: Max PIDR)
| Protocol Tier | Linear Ridge ($R^2$) | Random Forest ($R^2$) | Gradient Boosting ($R^2$) | Neural MLP ($R^2$) |
| :--- | :---: | :---: | :---: | :---: |
| **Tier 1: Random Split** | 0.9848 | 0.9340 | 0.9589 | **0.9850** |
| **Tier 2: Unseen Earthquakes** | **0.9724** | 0.8775 | 0.9463 | 0.9256 |
| **Tier 3: Unseen Buildings** | 0.9664 | 0.8865 | **0.9841** | 0.9426 |
| **Tier 4: Dual-Blind Unseen** | 0.8748 | 0.7244 | **0.9425** | 0.8667 |

### 3. First-Class IS 1893:2016 3-Column Engineering Audit Table
| Building Case & Parameters | Column 1: IS 1893:2016 Code | Column 2: Physics Solver | Column 3: ML Surrogate | Engineering Context |
| :--- | :---: | :---: | :---: | :--- |
| **5-Storey Residential (Zone IV)**<br>• $M = 580\text{ t}$, $T_1 = 0.524\text{ s}$<br>• Input: *Chamoli 1999 (0.36g)* | $V_B = 141.2\text{ kN}$<br>$\text{PIDR} = 0.118\%$ | $V_b = 1,842.5\text{ kN}$<br>$\text{PIDR} = 0.864\%$ | $V_b = 1,810.0\text{ kN}$<br>$\text{PIDR} = 0.858\%$ | $A_h = 0.0248$ ($R=5$). Physics & Surrogate capture unreduced elastic MCE demand. |
| **8-Storey Commercial (Zone V)**<br>• $M = 940\text{ t}$, $T_1 = 0.892\text{ s}$<br>• Input: *Bhuj 2001 (0.38g)* | $V_B = 342.8\text{ kN}$<br>$\text{PIDR} = 0.245\%$ | $V_b = 3,912.0\text{ kN}$<br>$\text{PIDR} = 1.412\%$ | $V_b = 3,850.0\text{ kN}$<br>$\text{PIDR} = 1.395\%$ | Soft soil amplifies long periods ($S_a/g = 1.87$). Surrogate discrepancy: $1.2\%$. |
| **SAC 3-Story Steel Benchmark**<br>• $M = 300\text{ t}$, $T_1 = 1.012\text{ s}$<br>• Input: *Northridge 1994 (0.84g)* | $V_B = 80.2\text{ kN}$<br>$\text{PIDR} = 0.210\%$ | $V_b = 1,420.0\text{ kN}$<br>$\text{PIDR} = 1.820\%$ | $V_b = 1,405.0\text{ kN}$<br>$\text{PIDR} = 1.802\%$ | Near-fault pulse excitation. Matches published FEMA-355C benchmark values. |

---

## 📡 Real-Time Onset Response Estimation (Phase 10)
- **Contribution**: Uses recursive STA/LTA (Short-Term Average / Long-Term Average) energy ratios on incoming continuous accelerogram streams. Upon P-wave onset detection ($r \ge 3.5$), early features ($\tau_c, PGA_p$) are extracted from the first $2.5\text{ s}$ of shaking, immediately triggering the ML surrogate to predict multi-storey drift and safety **before damaging S-wave arrivals**.
- **Scope**: Explicitly framed as real-time structural response estimation triggered by onset detection (not early warning seismology or unvalidated hardware).

---

## 📂 Repository Structure

```
seismic-ai/
├── app/
│   ├── backend/           # FastAPI backend (POST /predict, POST /simulate, /is1893/compare, /ws)
│   │   ├── main.py        # API server routing and websocket streaming
│   │   ├── schemas.py     # Pydantic data schemas
│   │   └── metadata.py    # Model provenance registry & SHA-256 traceability
│   └── frontend/          # Thin Streamlit dashboard calling FastAPI
│       ├── main.py        # 7-module interactive UI (Physics vs Surrogate, IS 1893, PESMOS)
│       └── client.py      # HTTP and WebSocket API client
├── src/
│   ├── standards/         # BIS IS 1893:2016 code calculation & 3-way table generator
│   ├── structural/        # MDOF shear building & SAC/IIT published benchmark frames
│   ├── dynamics/          # Modal analysis, Rayleigh damping, Newmark-β solver, DynamicResponse
│   ├── earthquake/        # PESMOS (IIT Roorkee), NCS & PEER ground-motion database
│   ├── sensors/           # ReplaySensor, LiveSensor stub, STA/LTA detector, early features
│   ├── features/          # Physics-informed feature extractor & EDP target labels
│   ├── ml/                # Linear Ridge, Random Forest, Gradient Boosting, Neural MLP
│   └── optimization/      # Differential Evolution optimizer & Closed-Loop Verifier
├── tests/                 # Full automated test suite (59 passing tests, 100% pass)
├── docs/                  # In-depth theoretical, standard, and presentation documents
│   ├── technical_report.md# Full academic whitepaper
│   ├── is1893_comparison.md# Dedicated IS 1893:2016 3-column engineering audit
│   ├── failure_modes.md   # Mechanics-based failure mode diagnosis
│   ├── benchmark_validation.md # SAC Steel & IIT benchmark validation
│   ├── presentation_guide.md # Faculty interview slide deck & defense guide
│   └── poster_guide.md    # Conference poster layout
├── data/                  # Simulation datasets and real earthquake accelerograms
├── models/                # Serialized trained surrogates and metadata
└── results/               # Model comparison, generalization, and optimization JSONs
```

---

## 🛠️ Quickstart & Localhost Execution

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run All 59 Automated Unit Tests
```bash
pytest -v
```

### 3. Launch Backend & Frontend Services (Dual Process)

**Terminal 1 — FastAPI Backend:**
```bash
uvicorn app.backend.main:app --host 127.0.0.1 --port 8000 --reload
```

**Terminal 2 — Streamlit Frontend:**
```bash
streamlit run app/frontend/main.py --server.port 8501
```

Access the dashboard at `http://localhost:8501` and interactive API docs at `http://localhost:8000/docs`.

---

## 🎓 Academic Documents & Interview Resources
- 📑 [Full Academic Technical Report](docs/technical_report.md)
- 📋 [IS 1893:2016 3-Column Comparison Whitepaper](docs/is1893_comparison.md)
- 🔬 [Mechanics-Based Failure Mode Analysis](docs/failure_modes.md)
- 🏢 [SAC Steel & IIT Benchmark Validation](docs/benchmark_validation.md)
- 🎤 [Faculty Interview Slide Deck & Defense Guide](docs/presentation_guide.md)
- 🖼️ [Conference Poster Layout Guide](docs/poster_guide.md)
- 📚 [Structural Dynamics Theory Derivations](docs/structural_theory.md)
- 🔬 [4-Tier Validation Strategy](docs/validation_strategy.md)
- 📖 [Academic Bibliography](docs/references.md)

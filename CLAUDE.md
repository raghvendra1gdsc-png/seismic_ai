# CLAUDE.md — Seismic-AI Agent Guidelines & Architecture Rules

## 🏛️ Project Summary
**Seismic-AI** is a high-performance cyber-physical structural dynamics and AI surrogate framework (IIT Delhi / Stanford Blume Center Grade) combining nonlinear mechanics (Bouc-Wen hysteresis), Physics-Informed Neural Networks (PINNs), real-time sensor Hardware Abstraction Layer (HAL), and BIS IS 1893:2016 code compliance auditing.

---

## 🚀 Key Commands

### Environment & Testing
```bash
# Run full automated test suite (74 tests)
pytest -v

# Run specific test module
pytest tests/test_nonlinear_solver.py -v
pytest tests/test_pinn.py -v
pytest tests/test_live_alarm.py -v
```

### Local Services & Deployment
```bash
# Launch FastAPI Backend (bound to localhost or LAN 0.0.0.0)
uvicorn app.backend.main:app --host 0.0.0.0 --port 8000 --reload

# Launch Streamlit Frontend (13 interactive modules)
streamlit run app/frontend/main.py --server.port 8501
```

### gstack Workflow Automation
```bash
# Run gstack multi-persona code review
python scripts/gstack.py review

# Run gstack QA verification suite
python scripts/gstack.py qa

# Run gstack pre-release ship checklist
python scripts/gstack.py ship

# Run gstack product strategy alignment
python scripts/gstack.py office-hours
```

---

## 📐 Architecture & Coding Standards

1. **Physical Grounding First**:
   - Every surrogate must be verifiable against the true Newmark-beta / Newton-Raphson physics solver (`src/dynamics/solver.py`, `src/dynamics/nonlinear_solver.py`).
   - Use standard SI units internally: Displacement ($m$), Velocity ($m/s$), Acceleration ($m/s^2$ or $g$), Mass ($kg$), Stiffness ($N/m$), Force ($N$).
2. **Deterministic Model Versioning**:
   - Never load unversioned pickle files. Every surrogate model in `models/trained/` must have a corresponding `metadata_*.json` containing feature columns, training date, and validation metrics.
3. **Backend & Frontend Separation**:
   - `app/backend/`: Pure FastAPI REST & WebSocket APIs.
   - `app/frontend/`: Pure Streamlit UI consuming backend endpoints over HTTP/WebSocket.
4. **Sensor Stream Resiliency**:
   - Digital conditioning: Butterworth 4th-order bandpass ($0.1–25\text{ Hz}$) + recursive baseline detrending.
   - STA/LTA onset picking with ring buffers to prevent memory leaks during continuous streaming.
5. **Quality Gates**:
   - 100% test pass rate across all unit tests before committing or pushing changes.

---

## 👥 gstack Personas & Virtual Team
- **CEO / Product Lead**: Ruthless prioritization, user experience, life-saving lead-time focus.
- **Engineering Manager**: Clean architecture, sub-millisecond inference, strict typing.
- **Academic Research Lead**: Bouc-Wen hysteresis, PINN equilibrium loss, FEMA P-58 fragility.
- **QA Lead**: Zero regressions, fast boundary condition test execution.
- **Security Officer**: Sensor buffer safety, webhook timeouts, air-gapped offline fallbacks.

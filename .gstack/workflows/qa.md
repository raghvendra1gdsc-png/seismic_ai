# Workflow: Quality Assurance & Test Verification (`/qa` or `python scripts/gstack.py qa`)

## Execution Protocol
1. **Automated Unit & Integration Tests**:
   - Run `pytest -v` across all test files (`tests/test_*.py`).
   - Require 100% test pass rate across all 74+ tests.
2. **Backend Health & Endpoint Validation**:
   - Verify FastAPI schema contracts (`/predict`, `/simulate`, `/is1893/compare`, `/earthquakes/live-global`, `/alarm/broadcast`).
3. **Frontend Sanity Check**:
   - Validate Streamlit application syntax and client proxy imports.
4. **Generalization Metrics Check**:
   - Confirm Tier 4 dual-blind unseen $R^2 \ge 0.90$.

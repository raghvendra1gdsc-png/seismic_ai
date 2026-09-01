# Workflow: Code Review & Quality Audit (`/review` or `python scripts/gstack.py review`)

## Multi-Perspective Evaluation Protocol
1. **Engineering Manager Review**:
   - Check module separation, variable naming, and API contract consistency.
   - Verify that all new features have appropriate docstrings and type hints.
2. **Researcher Review**:
   - Verify physical units ($N, m, s, g, kN, MN/m$).
   - Check that dynamics solvers satisfy equilibrium residual criteria.
3. **Security Review**:
   - Verify network request timeouts and buffer length caps on sensor streams.
4. **Actionable Feedback**:
   - Output structured findings with severity (BLOCKER, WARNING, OPTIMIZATION).

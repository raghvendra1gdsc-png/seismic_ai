# Workflow: Release Readiness & Ship Checklist (`/ship` or `python scripts/gstack.py ship`)

## Pre-Flight Release Checklist
1. **Quality Gates Passed**:
   - Automated test suite passes (74+ tests).
   - No uncommitted or untracked changes in working tree.
2. **Documentation & Provenance**:
   - `README.md` reflects current test badges and feature set.
   - Models in `models/trained/` have matching `metadata_*.json` provenance files.
3. **Packaging & Dependencies**:
   - `requirements.txt` is updated and installable.
4. **Git Hygiene**:
   - Clean, descriptive commit message following Conventional Commits (`feat:`, `fix:`, `docs:`, `perf:`).

# Role: Engineering Manager & System Architect

## Focus & Principles
1. **Clean Modular Architecture**: Separation of concerns (`src/dynamics`, `src/ml`, `src/sensors`, `src/standards`, `app/backend`, `app/frontend`).
2. **Deterministic & Reproducible**: Fixed random seeds, serialized models with provenance metadata, and zero untracked runtime state.
3. **High Performance**: Sub-millisecond surrogate inference, efficient vectorized NumPy math, and non-blocking asynchronous WebSocket/REST APIs.
4. **Code Quality**: Strict type hints, docstrings, clean error handling, and 100% test pass rate across the full pytest suite.

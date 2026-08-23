# Engineering Limitations & Scope Boundaries

## 1. Structural Modeling Assumptions & Boundaries
While Seismic-AI implements a rigorous numerical physics engine, several structural mechanics simplifications are currently operative:

1. **Shear-Building Idealization**:
   - Assumes floor diaphragms and beams are infinitely rigid in flexure relative to columns.
   - Column joint rotations and beam-column joint panel deformations are neglected.
   - Suitable for low- to mid-rise shear-wall or braced-frame buildings; less accurate for slender moment-resisting frames where flexural cantilevering dominates.

2. **Linear Elastic Material Behavior**:
   - Assumes restoring forces follow $F_s = K u$.
   - Material yielding, concrete cracking, steel rebar strain hardening, and strength/stiffness degradation under cyclic hysteretic loops are not captured in Phase 1-6.
   - Future extensions will incorporate non-linear Bouc-Wen or elastoplastic fiber models.

3. **Small Displacement Kinematics**:
   - Geometric non-linearities, including geometric stiffness softening ($K_G$) and global $P$-$\Delta$ overturning moments, are omitted.

4. **1D Unidirectional Horizontal Ground Motion**:
   - Earthquake excitation is applied along a single horizontal axis.
   - Bi-directional orthogonal coupling and vertical ground motion components are not included.

5. **Fixed Base Boundary Condition**:
   - Soil-Structure Interaction (SSI) and foundation flexibility/rocking are neglected.

---

## 2. Regulatory & Code Compliance Disclaimer
- **Not a Code-Compliant Design Software**: Seismic-AI is an academic research platform intended to explore the viability of ML surrogates in computational mechanics. It is **not** a certified commercial structural engineering software (e.g. ETABS, SAP2000, OpenSees) and should not be used as the sole basis for building design or code certification without independent engineering peer review.

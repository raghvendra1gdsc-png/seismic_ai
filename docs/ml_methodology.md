# Machine Learning Methodology & Surrogate Models

## 1. Surrogate Architectures

### A. Mechanics-Informed Linear Ridge Regression
- **Formulation**: Regularized linear model with $L_2$ penalty:
  $$\min_{\mathbf{w}} \|\mathbf{y} - \mathbf{X}\mathbf{w}\|_2^2 + \alpha \|\mathbf{w}\|_2^2$$
  Closed-form solution: $\mathbf{w}^* = (\mathbf{X}^T \mathbf{X} + \alpha \mathbf{I})^{-1} \mathbf{X}^T \mathbf{y}$.
- **Role**: Serves as the transparent, interpretable mechanics baseline.

### B. Random Forest Regressor (Nonlinear Bagging Ensemble)
- **Formulation**: Ensemble of $M=45$ bootstrap-aggregated randomized decision trees of max depth $d=7$:
  $$\hat{y}(\mathbf{x}) = \frac{1}{M} \sum_{m=1}^M T_m(\mathbf{x})$$
- **Role**: Captures nonlinear thresholds and interactions between higher modal periods and ground motion frequency content.

### C. Gradient Boosted Decision Trees (GBDT)
- **Formulation**: Additive sequence of shallow trees ($M=50$, learning rate $\eta=0.12$, depth $d=4$) fitted to residual gradients:
  $$F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \eta h_m(\mathbf{x})$$
- **Role**: High-precision surrogate capturing subtle boundary interactions with low variance.

### D. Multi-Layer Perceptron (Neural Surrogate)
- **Architecture**: Fully-connected network: $\text{Input}(D=25) \to \text{FC}(64, \text{Leaky-ReLU}) \to \text{FC}(32, \text{Leaky-ReLU}) \to \text{FC}(16, \text{Leaky-ReLU}) \to \text{Output}(1)$.
- **Optimization**: Mini-batch stochastic gradient descent with Adam optimizer ($\beta_1=0.9, \beta_2=0.999$, learning rate $\eta=0.006$, weight decay $\lambda=10^{-4}$).

---

## 2. Model Performance Comparison

| Model Architecture | Validation $R^2$ | Validation RMSE | Test $R^2$ (Tier 1) | Inference Time | Speedup vs Physics Solver |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Linear Ridge** | 0.9833 | 0.00233 | 0.9848 | $0.00017\,\text{ms}$ | **$263,816\times$** |
| **Random Forest** | 0.9707 | 0.00310 | 0.9340 | $0.04214\,\text{ms}$ | **$1,068\times$** |
| **Gradient Boosting** | 0.9717 | 0.00304 | 0.9589 | $0.03031\,\text{ms}$ | **$1,484\times$** |
| **Neural MLP** | **0.9884** | **0.00195** | **0.9858** | **$0.00036\,\text{ms}$** | **$125,673\times$** |

*Note: Baseline physics time per simulation $\approx 45.0\,\text{ms}$.*

---

## 3. Base Shear Prediction Accuracy

| Model Architecture | Base Shear $R^2$ | Base Shear RMSE (kN) | Speedup Factor |
| :--- | :---: | :---: | :---: |
| **Linear Ridge** | 0.9226 | $1,669\,\text{kN}$ | $510,759\times$ |
| **Random Forest** | 0.9145 | $1,754\,\text{kN}$ | $1,050\times$ |
| **Gradient Boosting** | 0.9506 | $1,333\,\text{kN}$ | $1,436\times$ |
| **Neural MLP** | **0.9924** | **$522\,\text{kN}$** | **$128,955\times$** |

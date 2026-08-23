"""Machine learning surrogate models for seismic response prediction.

Implements a hierarchy of surrogate architectures:
1. LinearRidgeSurrogate (Regularized Linear Baseline)
2. RandomForestSurrogate (Nonlinear Bagging Ensemble)
3. GradientBoostingSurrogate (Gradient Boosted Decision Trees)
4. NeuralSurrogate (Physics-Informed Multi-Layer Perceptron)
"""

from typing import Optional, Dict, Any, List, Union
import pickle
import numpy as np


class StandardScaler:
    """Feature standardizer (z-score normalization: z = (x - mu) / sigma)."""

    def __init__(self) -> None:
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        x_arr = np.asarray(X, dtype=np.float64)
        self.mean_ = np.mean(x_arr, axis=0)
        scale = np.std(x_arr, axis=0)
        # Avoid division by zero
        self.scale_ = np.where(scale < 1e-8, 1.0, scale)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardScaler is not fitted yet.")
        x_arr = np.asarray(X, dtype=np.float64)
        return (x_arr - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardScaler is not fitted yet.")
        x_arr = np.asarray(X, dtype=np.float64)
        return x_arr * self.scale_ + self.mean_


class BaseSurrogate:
    """Abstract base class for all surrogate regression models."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseSurrogate":
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute R^2 (coefficient of determination)."""
        y_true = np.asarray(y, dtype=np.float64).flatten()
        y_pred = self.predict(X).flatten()
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot < 1e-12:
            return 1.0 if ss_res < 1e-12 else 0.0
        return float(1.0 - (ss_res / ss_tot))

    def save(self, filepath: str) -> None:
        """Serialize model to disk."""
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> "BaseSurrogate":
        """Deserialize model from disk."""
        with open(filepath, "rb") as f:
            return pickle.load(f)


class LinearRidgeSurrogate(BaseSurrogate):
    """Mechanics-informed Ridge regularized linear regression surrogate."""

    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__(name="LinearRidge")
        self.alpha = float(alpha)
        self.weights_: Optional[np.ndarray] = None
        self.bias_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRidgeSurrogate":
        x_norm = self.scaler_x.fit_transform(X)
        y_norm = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

        n_samples, n_features = x_norm.shape

        # Closed form: w = (X^T X + alpha * I)^{-1} X^T y
        reg_matrix = self.alpha * np.eye(n_features)
        xtx = x_norm.T @ x_norm + reg_matrix
        xty = x_norm.T @ y_norm

        self.weights_ = np.linalg.solve(xtx, xty)
        self.bias_ = 0.0  # Zero since both X and y are standardized
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")
        x_norm = self.scaler_x.transform(X)
        y_norm_pred = x_norm @ self.weights_ + self.bias_
        y_pred = self.scaler_y.inverse_transform(y_norm_pred.reshape(-1, 1)).flatten()
        return y_pred


class DecisionTreeRegressorNode:
    """Lightweight single decision tree regressor for random forest & boosting."""

    def __init__(
        self,
        max_depth: int = 6,
        min_samples_split: int = 5,
        max_features: Optional[int] = None,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.tree_: Optional[Dict] = None

    def fit(self, X: np.ndarray, y: np.ndarray, rng: Optional[np.random.RandomState] = None) -> "DecisionTreeRegressorNode":
        if rng is None:
            rng = np.random.RandomState()
        self.tree_ = self._build_tree(X, y, depth=0, rng=rng)
        return self

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int, rng: np.random.RandomState) -> Dict:
        n_samples, n_features = X.shape

        # Base conditions: max depth, min samples, or zero variance
        if depth >= self.max_depth or n_samples < self.min_samples_split or np.var(y) < 1e-9:
            return {"leaf": True, "value": float(np.mean(y))}

        # Feature subset
        if self.max_features is None:
            feat_indices = np.arange(n_features)
        else:
            n_sub = min(n_features, max(1, self.max_features))
            feat_indices = rng.choice(n_features, size=n_sub, replace=False)

        best_feat = None
        best_thresh = None
        best_var_reduction = 0.0
        current_var = np.var(y) * n_samples

        for f_idx in feat_indices:
            vals = np.unique(X[:, f_idx])
            if len(vals) <= 1:
                continue

            # Check percentile thresholds to accelerate split search
            if len(vals) > 10:
                thresholds = np.percentile(vals, np.linspace(10, 90, 9))
            else:
                thresholds = 0.5 * (vals[:-1] + vals[1:])

            for thresh in thresholds:
                left_mask = X[:, f_idx] <= thresh
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                y_l = y[left_mask]
                y_r = y[right_mask]

                var_reduction = current_var - (np.var(y_l) * len(y_l) + np.var(y_r) * len(y_r))
                if var_reduction > best_var_reduction:
                    best_var_reduction = var_reduction
                    best_feat = f_idx
                    best_thresh = thresh

        if best_feat is None:
            return {"leaf": True, "value": float(np.mean(y))}

        left_mask = X[:, best_feat] <= best_thresh
        right_mask = ~left_mask

        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1, rng)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1, rng)

        return {
            "leaf": False,
            "feature": best_feat,
            "threshold": best_thresh,
            "left": left_child,
            "right": right_child,
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._predict_row(row, self.tree_) for row in X], dtype=np.float64)

    def _predict_row(self, row: np.ndarray, node: Dict) -> float:
        if node["leaf"]:
            return node["value"]
        if row[node["feature"]] <= node["threshold"]:
            return self._predict_row(row, node["left"])
        return self._predict_row(row, node["right"])


class RandomForestSurrogate(BaseSurrogate):
    """Random Forest Ensemble surrogate model with bootstrap aggregation."""

    def __init__(
        self,
        n_estimators: int = 50,
        max_depth: int = 8,
        min_samples_split: int = 4,
        seed: int = 42,
    ) -> None:
        super().__init__(name="RandomForest")
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.seed = int(seed)
        self.trees_: List[DecisionTreeRegressorNode] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestSurrogate":
        x_norm = self.scaler_x.fit_transform(X)
        y_norm = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

        n_samples, n_features = x_norm.shape
        max_feat = max(1, int(np.sqrt(n_features)))
        rng = np.random.RandomState(self.seed)

        self.trees_ = []
        for _ in range(self.n_estimators):
            # Bootstrap sample
            boot_idx = rng.randint(0, n_samples, size=n_samples)
            tree = DecisionTreeRegressorNode(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=max_feat,
            )
            tree.fit(x_norm[boot_idx], y_norm[boot_idx], rng=rng)
            self.trees_.append(tree)

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")
        x_norm = self.scaler_x.transform(X)
        preds = np.column_stack([tree.predict(x_norm) for tree in self.trees_])
        y_norm_pred = np.mean(preds, axis=1)
        return self.scaler_y.inverse_transform(y_norm_pred.reshape(-1, 1)).flatten()


class GradientBoostingSurrogate(BaseSurrogate):
    """Gradient Boosted Decision Trees surrogate model."""

    def __init__(
        self,
        n_estimators: int = 60,
        learning_rate: float = 0.1,
        max_depth: int = 4,
        seed: int = 42,
    ) -> None:
        super().__init__(name="GradientBoosting")
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self.seed = int(seed)
        self.trees_: List[DecisionTreeRegressorNode] = []
        self.init_val_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostingSurrogate":
        x_norm = self.scaler_x.fit_transform(X)
        y_norm = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

        rng = np.random.RandomState(self.seed)
        self.init_val_ = float(np.mean(y_norm))
        residuals = y_norm - self.init_val_

        self.trees_ = []
        for _ in range(self.n_estimators):
            tree = DecisionTreeRegressorNode(
                max_depth=self.max_depth,
                min_samples_split=4,
                max_features=None,
            )
            tree.fit(x_norm, residuals, rng=rng)
            preds = tree.predict(x_norm)
            residuals = residuals - self.learning_rate * preds
            self.trees_.append(tree)

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")
        x_norm = self.scaler_x.transform(X)
        y_norm_pred = np.full(len(x_norm), self.init_val_, dtype=np.float64)
        for tree in self.trees_:
            y_norm_pred += self.learning_rate * tree.predict(x_norm)
        return self.scaler_y.inverse_transform(y_norm_pred.reshape(-1, 1)).flatten()


class NeuralSurrogate(BaseSurrogate):
    """Multi-Layer Perceptron (MLP) Neural Network surrogate model with Adam optimizer."""

    def __init__(
        self,
        hidden_layers: Tuple[int, ...] = (64, 32, 16),
        learning_rate: float = 0.005,
        max_epochs: int = 150,
        batch_size: int = 32,
        weight_decay: float = 1e-4,
        seed: int = 42,
    ) -> None:
        super().__init__(name="NeuralMLP")
        self.hidden_layers = hidden_layers
        self.learning_rate = float(learning_rate)
        self.max_epochs = int(max_epochs)
        self.batch_size = int(batch_size)
        self.weight_decay = float(weight_decay)
        self.seed = int(seed)
        self.params_: Dict[str, np.ndarray] = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NeuralSurrogate":
        x_norm = self.scaler_x.fit_transform(X)
        y_norm = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

        rng = np.random.RandomState(self.seed)
        n_samples, n_features = x_norm.shape

        # Initialize network architecture: [n_features, h1, h2, h3, 1]
        layer_dims = [n_features] + list(self.hidden_layers) + [1]
        num_layers = len(layer_dims) - 1

        self.params_ = {}
        # He / Kaiming normal initialization
        for l in range(1, len(layer_dims)):
            in_dim = layer_dims[l - 1]
            out_dim = layer_dims[l]
            self.params_[f"W{l}"] = rng.randn(in_dim, out_dim) * np.sqrt(2.0 / in_dim)
            self.params_[f"b{l}"] = np.zeros((1, out_dim))

        # Adam optimizer state
        m_dict = {k: np.zeros_like(v) for k, v in self.params_.items()}
        v_dict = {k: np.zeros_like(v) for k, v in self.params_.items()}
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        t_step = 0

        # Mini-batch SGD with Adam
        indices = np.arange(n_samples)
        for epoch in range(self.max_epochs):
            rng.shuffle(indices)
            for start_idx in range(0, n_samples, self.batch_size):
                t_step += 1
                batch_idx = indices[start_idx : start_idx + self.batch_size]
                xb = x_norm[batch_idx]
                yb = y_norm[batch_idx, np.newaxis]

                # Forward pass
                activations = {"A0": xb}
                current_a = xb
                for l in range(1, num_layers):
                    z = current_a @ self.params_[f"W{l}"] + self.params_[f"b{l}"]
                    current_a = np.maximum(0.01 * z, z)  # Leaky ReLU
                    activations[f"Z{l}"] = z
                    activations[f"A{l}"] = current_a

                # Output linear layer
                zl = current_a @ self.params_[f"W{num_layers}"] + self.params_[f"b{num_layers}"]
                activations[f"Z{num_layers}"] = zl
                activations[f"A{num_layers}"] = zl

                # Backward pass
                grads = {}
                d_out = 2.0 * (zl - yb) / len(batch_idx)  # MSE gradient

                dz = d_out
                for l in range(num_layers, 0, -1):
                    a_prev = activations[f"A{l-1}"]
                    grads[f"W{l}"] = a_prev.T @ dz + self.weight_decay * self.params_[f"W{l}"]
                    grads[f"b{l}"] = np.sum(dz, axis=0, keepdims=True)

                    if l > 1:
                        da = dz @ self.params_[f"W{l}"].T
                        z_prev = activations[f"Z{l-1}"]
                        # Leaky ReLU derivative
                        dz = da * np.where(z_prev > 0, 1.0, 0.01)

                # Adam parameter update
                for k in self.params_:
                    m_dict[k] = beta1 * m_dict[k] + (1 - beta1) * grads[k]
                    v_dict[k] = beta2 * v_dict[k] + (1 - beta2) * (grads[k] ** 2)
                    m_hat = m_dict[k] / (1 - beta1 ** t_step)
                    v_hat = v_dict[k] / (1 - beta2 ** t_step)
                    self.params_[k] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + eps)

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")
        x_norm = self.scaler_x.transform(X)
        current_a = x_norm
        num_layers = len(self.hidden_layers) + 1

        for l in range(1, num_layers):
            z = current_a @ self.params_[f"W{l}"] + self.params_[f"b{l}"]
            current_a = np.maximum(0.01 * z, z)  # Leaky ReLU

        y_norm_pred = (current_a @ self.params_[f"W{num_layers}"] + self.params_[f"b{num_layers}"]).flatten()
        return self.scaler_y.inverse_transform(y_norm_pred.reshape(-1, 1)).flatten()

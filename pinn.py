import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib, os, warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.metrics       import mean_absolute_error, mean_squared_error, r2_score


class PINN:
    """
    Physics-Informed Neural Network — pure NumPy, no external DL library.

    Architecture : Input -> [hidden_layers] -> Output(1), ReLU activations
    Optimizer    : Adam with gradient clipping
    Physics loss : Prandtl drag polar C_D = C_D0 + k*C_L^2
    """

    def __init__(self, layer_sizes: tuple, lr: float = 0.001,
                 lambda_physics: float = 0.5, epochs: int = 500,
                 batch_size: int = 256, random_state: int = 42):
        """
        Parameters
        ----------
        layer_sizes    : tuple of layer widths, e.g. (8, 128, 64, 32, 1)
        lr             : Adam learning rate
        lambda_physics : weight on physics loss term
        epochs         : maximum training epochs
        batch_size     : mini-batch size
        random_state   : numpy random seed
        """
        np.random.seed(random_state)
        self.layer_sizes     = layer_sizes
        self.lr              = lr
        self.lambda_physics  = lambda_physics
        self.epochs          = epochs
        self.batch_size      = batch_size

        self.weights = []
        self.biases  = []
        for i in range(len(layer_sizes) - 1):
            scale = np.sqrt(2.0 / layer_sizes[i])
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]) * scale)
            self.biases.append(np.zeros((1, layer_sizes[i+1])))

        self.m_w = [np.zeros_like(w) for w in self.weights]
        self.v_w = [np.zeros_like(w) for w in self.weights]
        self.m_b = [np.zeros_like(b) for b in self.biases]
        self.v_b = [np.zeros_like(b) for b in self.biases]
        self.t   = 0

        self.scaler         = StandardScaler()
        self.train_loss_log = []
        self.data_loss_log  = []
        self.phys_loss_log  = []

    @staticmethod
    def _relu(x):   return np.maximum(0.0, x)

    @staticmethod
    def _relu_d(x): return (x > 0).astype(float)

    def _forward(self, X):
        acts    = [X]
        pre_acts = []
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = acts[-1] @ W + b
            pre_acts.append(z)
            acts.append(self._relu(z) if i < len(self.weights) - 1 else z)
        return acts[-1], acts, pre_acts

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(X)
        out, _, _ = self._forward(Xs)
        return out.flatten()

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return r2_score(y, self.predict(X))

    def _physics_loss(self, X_orig: np.ndarray, y_pred: np.ndarray) -> tuple:
        """
        Enforce drag polar: y_pred ≈ C_D0 + k * C_L^2
        C_L^2 is feature index 6 in the original (unscaled) feature space.
        Returns (loss_value, Cd0_estimate, k_estimate).
        """
        cl_sq = X_orig[:, 6:7]
        A     = np.hstack([np.ones_like(cl_sq), cl_sq])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, y_pred, rcond=None)
            Cd0 = float(coeffs[0])
            k   = max(float(coeffs[1]), 0.0)
        except Exception:
            return 0.0, 0.0, 0.0
        polar    = Cd0 + k * cl_sq
        residual = y_pred - polar
        return float(np.mean(residual ** 2)), Cd0, k

    def _backward(self, Xs: np.ndarray, y: np.ndarray,
                  X_orig: np.ndarray) -> tuple:
        N = Xs.shape[0]
        y_pred, acts, pre_acts = self._forward(Xs)

        delta = 2.0 * (y_pred - y) / N

        cl_sq = X_orig[:, 6:7]
        A     = np.hstack([np.ones_like(cl_sq), cl_sq])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, y_pred, rcond=None)
            Cd0  = float(coeffs[0])
            k    = max(float(coeffs[1]), 0.0)
            phys_delta = (2.0 * self.lambda_physics / N) * (y_pred - (Cd0 + k * cl_sq))
        except Exception:
            phys_delta = np.zeros_like(delta)

        delta = delta + phys_delta

        grads_w, grads_b = [], []
        for i in reversed(range(len(self.weights))):
            grads_w.insert(0, acts[i].T @ delta)
            grads_b.insert(0, np.sum(delta, axis=0, keepdims=True))
            if i > 0:
                delta = (delta @ self.weights[i].T) * self._relu_d(pre_acts[i-1])
        return grads_w, grads_b

    def _adam(self, gw: list, gb: list,
              beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.t += 1
        for i in range(len(self.weights)):
            w = np.clip(gw[i], -1.0, 1.0)
            b = np.clip(gb[i], -1.0, 1.0)
            self.m_w[i] = beta1 * self.m_w[i] + (1 - beta1) * w
            self.v_w[i] = beta2 * self.v_w[i] + (1 - beta2) * w**2
            self.m_b[i] = beta1 * self.m_b[i] + (1 - beta1) * b
            self.v_b[i] = beta2 * self.v_b[i] + (1 - beta2) * b**2
            mw = self.m_w[i] / (1 - beta1**self.t)
            vw = self.v_w[i] / (1 - beta2**self.t)
            mb = self.m_b[i] / (1 - beta1**self.t)
            vb = self.v_b[i] / (1 - beta2**self.t)
            self.weights[i] -= self.lr * mw / (np.sqrt(vw) + eps)
            self.biases[i]  -= self.lr * mb / (np.sqrt(vb) + eps)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PINN":
        Xs = self.scaler.fit_transform(X)
        y  = y.reshape(-1, 1)
        N  = Xs.shape[0]

        best_loss      = np.inf
        patience_count = 0
        patience       = 30

        print(f"  PINN  arch={self.layer_sizes}  "
              f"lr={self.lr}  λ={self.lambda_physics}  "
              f"epochs={self.epochs}  batch={self.batch_size}")

        for epoch in range(self.epochs):
            idx  = np.random.permutation(N)
            Xs_s = Xs[idx]; Xo_s = X[idx]; y_s = y[idx]

            ep_loss = 0.0; n_b = 0
            for s in range(0, N, self.batch_size):
                Xb  = Xs_s[s:s+self.batch_size]
                Xo  = Xo_s[s:s+self.batch_size]
                yb  = y_s[s:s+self.batch_size]
                yp, _, _ = self._forward(Xb)
                d_l  = float(np.mean((yp - yb)**2))
                p_l, _, _ = self._physics_loss(Xo, yp)
                ep_loss += d_l + self.lambda_physics * p_l
                n_b     += 1
                gw, gb  = self._backward(Xb, yb, Xo)
                self._adam(gw, gb)

            ep_loss /= n_b
            self.train_loss_log.append(ep_loss)
            yp_s, _, _ = self._forward(Xs[:2000])
            dl = float(np.mean((yp_s - y[:2000])**2))
            pl, _, _   = self._physics_loss(X[:2000], yp_s)
            self.data_loss_log.append(dl)
            self.phys_loss_log.append(pl)

            if ep_loss < best_loss - 1e-7:
                best_loss = ep_loss; patience_count = 0
            else:
                patience_count += 1

            if (epoch + 1) % 50 == 0:
                print(f"    epoch {epoch+1:>4}/{self.epochs}  "
                      f"total={ep_loss:.6f}  data={dl:.6f}  physics={pl:.6f}")

            if patience_count >= patience:
                print(f"    Early stopping at epoch {epoch+1}")
                break

        return self

if __name__ == "__main__":
    from config import (FEATURES, TARGET, TRAIN_CSV, TEST_CSV,
                        MODELS_DIR, RESULTS_DIR, PLOTS_DIR,
                        MODEL_PARAMS, PLOT_DPI)

    os.makedirs(MODELS_DIR,  exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR,   exist_ok=True)

    train = pd.read_csv(TRAIN_CSV)
    test  = pd.read_csv(TEST_CSV)
    X_train = train[FEATURES].values
    y_train = train[TARGET].values
    X_test  = test[FEATURES].values
    y_test  = test[TARGET].values

    print("=" * 60)
    print("PINN TRAINING (standalone)")
    print("=" * 60)

    pinn = PINN(**MODEL_PARAMS["pinn"])
    pinn.fit(X_train, y_train)
    pred = pinn.predict(X_test)

    mae  = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2   = r2_score(y_test, pred)
    print(f"\n  MAE={mae:.5f}  RMSE={rmse:.5f}  R²={r2:.4f}")

    joblib.dump(pinn, os.path.join(MODELS_DIR, "pinn.pkl"))

    fig, ax = plt.subplots(figsize=(8, 4))
    ep = range(1, len(pinn.train_loss_log) + 1)
    ax.plot(ep, pinn.train_loss_log, color="#1D9E75", label="Total loss")
    ax.plot(ep, pinn.data_loss_log,  color="#3B8BD4", linestyle="--", label="Data loss")
    ax.plot(ep, pinn.phys_loss_log,  color="#EF9F27", linestyle=":",  label="Physics loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (MSE log scale)")
    ax.set_title("PINN Training Loss", fontweight="bold")
    ax.set_yscale("log"); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "09_pinn_loss.png")
    plt.savefig(out, dpi=PLOT_DPI, bbox_inches="tight"); plt.close()
    print(f"  Loss plot: {out}")
    print("Done.")

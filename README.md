# Physics-Based Aerodynamic Model & Physics-Informed Neural Network (PINN)

This document provides the complete, mathematically rigorous specification and production-ready Python implementation for the physics-based aerodynamic data generator, Physics-Informed Neural Network (PINN) architecture, and model evaluation pipeline.

---

## 1. Physics-Based Aerodynamic Model Formulation

The total drag coefficient $C_D$ is synthesized via component buildup:

$$
C_D = C_{D,f} + C_{D,p} + C_{D,i} + C_{D,s} + C_{D,w}
$$

where:
- $C_{D,f}$: Skin-friction drag coefficient
- $C_{D,p}$: Form/pressure drag coefficient
- $C_{D,i}$: Induced drag coefficient
- $C_{D,s}$: Post-stall separation drag coefficient
- $C_{D,w}$: Compressibility/wave drag rise

### 1.1 Skin-Friction Drag ($C_{D,f}$)
Evaluated across laminar and turbulent regimes weighted by the laminar fraction $\gamma_{	ext{lam}}$:

- **Laminar Flow (Blasius boundary layer):**
  $$C_{f,	ext{lam}} = \frac{1.328}{\sqrt{Re}}$$

- **Turbulent Flow (Schlichting correlation):**
  $$C_{f,	ext{turb}} = \frac{0.455}{(\log_{10} Re)^{2.58}}$$

- **Composite Skin Friction:**
  $$C_f = \gamma_{\text{lam}} C_{f,\text{lam}} + (1 - \gamma_{\text{lam}}) C_{f,\text{turb}}$$

### 1.2 Pressure / Form Drag ($C_{D,p}$)
Estimated using Hoerner's empirical thickness-correction relation:

$$
C_{D,p} = C_f \left[ 2\left(\frac{t}{c}\right) + 60\left(\frac{t}{c}\right)^4 \right]
$$

### 1.3 Lift Coefficient ($C_L$) & Induced Drag ($C_{D,i}$)
Linear thin-airfoil theory with camber correction and stall saturation:

$$
C_L = \text{clip}\left( 2\pi (\alpha - \alpha_{L0}),\, C_{L,\min},\, C_{L,\max} \right), \quad \alpha_{L0} \approx -2\,\text{camber}
$$

Induced drag follows the Prandtl lifting-line formulation:

$$
C_{D,i} = \frac{C_L^2}{\pi e AR}
$$

### 1.4 Post-Stall Drag Rise ($C_{D,s}$)
A smooth sigmoid transition models flow separation beyond critical angle of attack $\alpha_{\text{stall}}$:

$$
C_{D,s} = \frac{C_{D,s,\max}}{1 + \exp\left( -k_{\text{stall}} (|\alpha| - \alpha_{\text{stall}}) \right)}
$$

### 1.5 Compressibility & Wave Drag ($C_{D,w}$)
Subsonic Prandtl-Glauert transformation scaling profile components:

$$
\beta = \sqrt{\max(10^{-4}, 1 - M^2)}, \quad C_{D,\text{profile}} = \frac{C_{D,f} + C_{D,p}}{\beta}
$$

Transonic drag-rise above critical Mach number $M_{\text{crit}}$:

$$
M_{\text{crit}} = M_0 - k_{tc}\left(\frac{t}{c}\right) - k_{cl}|C_L|
$$
$$
C_{D,w} = k_w \max(0, M - M_{\text{crit}})^4
$$

---

## 2. Feature Vector Definition

The feature vector $X \in \mathbb{R}^8$ supplied to the machine learning model is defined as:

$$
X = \left[ \alpha,\, \log_{10}(Re),\, M,\, \frac{t}{c},\, \text{camber},\, C_L,\, C_L^2,\, Re_{\text{Mach}} \right]
$$

where:
- $Re_{\text{Mach}} = \frac{Re \times M}{10^6}$

Target variable:
$$y = C_D$$

---

## 3. Physics-Informed Neural Network (PINN)

The PINN leverages a composite loss function integrating MSE data loss with empirical parabolic drag-polar physics $C_D = C_{D0} + k C_L^2$:

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \lambda_{\text{physics}} \mathcal{L}_{\text{physics}}
$$

$$
\mathcal{L}_{\text{data}} = \frac{1}{B} \sum_{i=1}^B (y_i - \hat{y}_i)^2
$$

$$
\mathcal{L}_{\text{physics}} = \frac{1}{B} \sum_{i=1}^B \left( \hat{y}_i - (C_{D0} + k C_{L,i}^2) \right)^2
$$

---

## 4. Evaluation Metrics

1. **Mean Absolute Error (MAE):**
   $$\text{MAE} = \frac{1}{N}\sum_{i=1}^N |y_i - \hat{y}_i|$$

2. **Root Mean Squared Error (RMSE):**
   $$\text{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^N (y_i - \hat{y}_i)^2}$$

3. **Coefficient of Determination ($R^2$):**
   $$R^2 = 1 - \frac{\sum_{i=1}^N (y_i - \hat{y}_i)^2}{\sum_{i=1}^N (y_i - \bar{y})^2}$$

4. **Maximum Absolute Error (MaxError):**
   $$\text{MaxError} = \max_{i} |y_i - \hat{y}_i|$$

---

## 5. Complete Python Implementation

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ----------------------------------------------------------------------
# 1. Physics-Based Aerodynamic Data Generator
# ----------------------------------------------------------------------
class AerodynamicDataGenerator:
    """
    Generates synthetic aerodynamic training data using component buildup
    and empirical aero relations.
    """
    def __init__(self, AR=8.0, e=0.85, laminar_fraction=0.1):
        self.AR = AR
        self.e = e
        self.laminar_fraction = laminar_fraction

    def compute_skin_friction(self, Re: np.ndarray) -> np.ndarray:
        cf_lam = 1.328 / np.sqrt(np.maximum(Re, 1e2))
        cf_turb = 0.455 / (np.log10(np.maximum(Re, 1e2)) ** 2.58)
        return self.laminar_fraction * cf_lam + (1.0 - self.laminar_fraction) * cf_turb

    def compute_lift(self, alpha_deg: np.ndarray, camber: np.ndarray,
                     cl_max=1.5, cl_min=-1.2) -> np.ndarray:
        alpha_rad = np.radians(alpha_deg)
        alpha_l0 = np.radians(-2.0 * camber * 100.0)
        cl_linear = 2.0 * np.pi * (alpha_rad - alpha_l0)
        return np.clip(cl_linear, cl_min, cl_max)

    def generate_dataset(self, num_samples=5000, random_seed=42):
        np.random.seed(random_seed)
        
        # Physical parameter distributions
        alpha = np.random.uniform(-5.0, 18.0, num_samples)
        Re = 10 ** np.random.uniform(5.0, 7.2, num_samples)
        M = np.random.uniform(0.1, 0.75, num_samples)
        tc_ratio = np.random.uniform(0.06, 0.18, num_samples)
        camber = np.random.uniform(0.0, 0.06, num_samples)

        # 1. Lift Coefficient
        cl = self.compute_lift(alpha, camber)

        # 2. Skin Friction & Form Drag
        cf = self.compute_skin_friction(Re)
        cd_f = cf
        cd_p = cf * (2.0 * tc_ratio + 60.0 * (tc_ratio ** 4))

        # 3. Induced Drag
        cd_i = (cl ** 2) / (np.pi * self.e * self.AR)

        # 4. Stall Drag
        alpha_stall = 12.0 + 20.0 * tc_ratio + 10.0 * camber
        stall_steepness = 0.6
        cd_s = 0.15 / (1.0 + np.exp(-stall_steepness * (np.abs(alpha) - alpha_stall)))

        # 5. Mach / Compressibility & Wave Drag
        beta = np.sqrt(np.maximum(1e-4, 1.0 - M**2))
        profile_drag = (cd_f + cd_p) / beta
        m_crit = 0.85 - 0.1 * tc_ratio - 0.05 * np.abs(cl)
        wave_drag = 20.0 * (np.maximum(0.0, M - m_crit) ** 4)

        cd_total = profile_drag + cd_i + cd_s + wave_drag

        # Engineering Feature Matrix: [alpha, log10(Re), M, t/c, camber, CL, CL^2, Re_Mach]
        log10_re = np.log10(Re)
        cl_sq = cl ** 2
        re_mach = (Re * M) / 1e6

        X = np.column_stack([alpha, log10_re, M, tc_ratio, camber, cl, cl_sq, re_mach]).astype(np.float32)
        y = cd_total.reshape(-1, 1).astype(np.float32)

        return X, y

# ----------------------------------------------------------------------
# 2. Physics-Informed Neural Network (PINN)
# ----------------------------------------------------------------------
class AerodynamicPINN(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
        # Learnable physical polar coefficients: CD = CD0 + k * CL^2
        self.cd0 = nn.Parameter(torch.tensor([0.015], dtype=torch.float32))
        self.k = nn.Parameter(torch.tensor([0.045], dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def physics_loss(self, x_raw: torch.Tensor, y_pred: torch.Tensor, cl_sq_idx: int = 6) -> torch.Tensor:
        # Extract CL^2 feature
        cl_sq = x_raw[:, cl_sq_idx:cl_sq_idx+1]
        phys_target = torch.clamp(self.cd0, min=1e-4) + torch.clamp(self.k, min=1e-4) * cl_sq
        return torch.mean((y_pred - phys_target) ** 2)

# ----------------------------------------------------------------------
# 3. Model Training & Pipeline
# ----------------------------------------------------------------------
def train_pinn(model, dataloader, raw_train_x, epochs=40, lr=2e-3, lambda_physics=0.05):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse_loss_fn = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_epoch_loss = 0.0
        
        for batch_x_norm, batch_y, batch_x_raw in dataloader:
            optimizer.zero_grad()
            y_pred = model(batch_x_norm)
            
            loss_data = mse_loss_fn(y_pred, batch_y)
            loss_phys = model.physics_loss(batch_x_raw, y_pred)
            
            loss = loss_data + lambda_physics * loss_phys
            loss.backward()
            optimizer.step()
            
            total_epoch_loss += loss.item() * batch_x_norm.size(0)

        if epoch % 10 == 0 or epoch == 1:
            avg_loss = total_epoch_loss / len(dataloader.dataset)
            print(f"Epoch {epoch:02d}/{epochs} | Total Loss: {avg_loss:.6f} | CD0: {model.cd0.item():.4f} | k: {model.k.item():.4f}")

# ----------------------------------------------------------------------
# 4. Evaluation Metrics
# ----------------------------------------------------------------------
def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot)
    max_err = np.max(np.abs(y_true - y_pred))

    print("\n" + "=" * 50)
    print("               MODEL EVALUATION")
    print("=" * 50)
    print(f"Mean Absolute Error (MAE):        {mae:.6e}")
    print(f"Root Mean Squared Error (RMSE):   {rmse:.6e}")
    print(f"R² Score:                         {r2:.5f}")
    print(f"Maximum Absolute Error:           {max_err:.6e}")
    print("=" * 50)

# ----------------------------------------------------------------------
# 5. Main Execution Entry Point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Generate dataset
    generator = AerodynamicDataGenerator(AR=8.0, e=0.85)
    X, y = generator.generate_dataset(num_samples=5000, random_seed=42)

    # Train / Test split
    split_idx = int(0.8 * len(X))
    X_train_raw, X_test_raw = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Standardization
    mean_val, std_val = X_train_raw.mean(axis=0), X_train_raw.std(axis=0) + 1e-8
    X_train_norm = (X_train_raw - mean_val) / std_val
    X_test_norm = (X_test_raw - mean_val) / std_val

    # DataLoader
    train_dataset = TensorDataset(
        torch.tensor(X_train_norm),
        torch.tensor(y_train),
        torch.tensor(X_train_raw)
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Instantiate and Train
    model = AerodynamicPINN(input_dim=8, hidden_dim=64)
    train_pinn(model, train_loader, X_train_raw, epochs=40, lr=2e-3, lambda_physics=0.05)

    # Test Evaluation
    model.eval()
    with torch.no_grad():
        y_test_pred = model(torch.tensor(X_test_norm)).numpy()

    evaluate_model(y_test, y_test_pred)
```

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

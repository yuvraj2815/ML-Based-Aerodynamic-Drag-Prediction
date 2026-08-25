# Physics-Based Aerodynamic Model

The total drag coefficient is constructed from several components:

$$
C_D = C_{D,f} + C_{D,p} + C_{D,i} + C_{D,s}
$$

where:

- $C_{D,f}$ = skin-friction drag
- $C_{D,p}$ = pressure/form drag
- $C_{D,i}$ = induced drag
- $C_{D,s}$ = stall-related drag

A Mach/compressibility correction is also applied to the profile-related drag components.

---

## 1. Skin-Friction Drag

The implementation uses different correlations for laminar and turbulent flow.

### Laminar flow

$$
C_f = \frac{1.328}{\sqrt{Re}}
$$

### Turbulent flow

$$
C_f =
\frac{0.455}{(\log_{10} Re)^{2.58}}
$$

The final skin-friction coefficient combines the laminar and turbulent contributions according to the configured laminar-flow fraction.

---

## 2. Pressure/Form Drag

The pressure/form drag is calculated using a Hoerner-type empirical relation:

$$
C_{D,p}
=
C_f
\left[
2\left(\frac{t}{c}\right)
+
60\left(\frac{t}{c}\right)^4
\right]
$$

where:

- $t/c$ = thickness-to-chord ratio
- $C_f$ = skin-friction coefficient

---

## 3. Lift Coefficient

The lift coefficient is calculated using a thin-airfoil-based formulation:

$$
C_L = 2\pi(\alpha - \alpha_{L0})
$$

where:

- $\alpha$ = angle of attack
- $\alpha_{L0}$ = zero-lift angle

The calculated lift coefficient is clipped using configured limits to represent stall behavior.

---

## 4. Induced Drag

Induced drag follows the Prandtl lifting-line relation:

$$
C_{D,i}
=
\frac{C_L^2}{\pi e AR}
$$

where:

- $C_L$ = lift coefficient
- $e$ = Oswald efficiency factor
- $AR$ = aspect ratio

This relationship also motivates the use of $C_L^2$ as an input feature for the machine-learning models.

---

## 5. Stall Drag

A smooth sigmoid-based drag increase is introduced beyond the estimated stall angle.

The stall behavior depends on:

- camber
- thickness ratio
- angle of attack

This allows the generated dataset to represent the rapid increase in drag associated with post-stall behavior.

---

## 6. Mach Correction

The model applies a compressibility correction at higher Mach numbers.

For the subsonic regime, a Prandtl-Glauert-style correction is applied:

$$
\beta = \sqrt{1-M^2}
$$

and the corresponding correction is applied to the relevant aerodynamic quantities.

At higher Mach numbers, an additional wave-drag-rise term is introduced to represent the increase in drag approaching the transonic regime.

---

# Physics-Informed Neural Network

The PINN combines the conventional data-fitting objective with a physics-based constraint.

The aerodynamic drag-polar relationship used by the PINN is:

$$
C_D = C_{D0} + kC_L^2
$$

where:

- $C_{D0}$ = zero-lift drag coefficient
- $k$ = induced-drag factor
- $C_L$ = lift coefficient

The total PINN loss is:

$$
L =
L_{\text{data}}
+
\lambda_{\text{physics}}
L_{\text{physics}}
$$

where:

- $L_{\text{data}}$ = error between predicted and target drag
- $L_{\text{physics}}$ = violation of the aerodynamic drag relationship
- $\lambda_{\text{physics}}$ = physics-loss weighting factor

This encourages the neural network to fit the training data while remaining consistent with the aerodynamic relationship.

---

# Evaluation Metrics

The models are evaluated using four metrics.

## Mean Absolute Error

$$
MAE =
\frac{1}{N}
\sum_{i=1}^{N}
\left|y_i-\hat{y}_i\right|
$$

## Root Mean Squared Error

$$
RMSE =
\sqrt{
\frac{1}{N}
\sum_{i=1}^{N}
\left(y_i-\hat{y}_i\right)^2
}
$$

## Coefficient of Determination

$$
R^2 =
1-
\frac{
\sum_{i=1}^{N}
\left(y_i-\hat{y}_i\right)^2
}{
\sum_{i=1}^{N}
\left(y_i-\bar{y}\right)^2
}
$$

## Maximum Absolute Error

$$
MaxError =
\max_i
\left|y_i-\hat{y}_i\right|
$$

---

# Feature Engineering

The ML models use physics-informed features including:

$$
\log_{10}(Re)
$$

$$
C_L^2
$$

and

$$
Re_{\text{Mach}}
=
\frac{Re \times M}{10^6}
$$

The complete feature vector is:

$$
X =
[
\alpha,\,
\log_{10}(Re),\,
M,\,
t/c,\,
\text{camber},\,
C_L,\,
C_L^2,\,
Re_{\text{Mach}}
]
$$

The prediction target is:

$$
\boxed{C_D}
$$

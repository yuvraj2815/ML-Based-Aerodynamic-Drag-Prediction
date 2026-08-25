# ML-Based Aerodynamic Drag Prediction

A physics-informed machine learning framework for predicting the aerodynamic drag coefficient (C_D) of NACA airfoils across varying angle of attack, Reynolds number, Mach number, camber, and thickness ratio.

The project combines **classical aerodynamic correlations, engineered physics-based features, conventional machine-learning regressors, and a Physics-Informed Neural Network (PINN)** to study and predict aerodynamic drag.

---

## Overview

Accurate aerodynamic drag prediction is important for aircraft and aerodynamic-component design, but high-fidelity CFD simulations can be computationally expensive when evaluating large design spaces.

This project explores a lightweight alternative:

```text
Aerodynamic Parameters
        │
        ▼
Physics-Based Model
        │
        ├── Lift coefficient
        ├── Skin friction
        ├── Pressure drag
        ├── Induced drag
        ├── Stall drag
        └── Mach correction
        │
        ▼
Synthetic Aerodynamic Dataset
        │
        ▼
Feature Engineering
        │
        ▼
┌────────────┬────────────┬────────────┬────────────┐
│ Polynomial │   Random   │    SVR     │    ANN     │
│   Ridge    │   Forest   │   (RBF)    │            │
└────────────┴────────────┴────────────┴────────────┘
        │
        └──────────────┬───────────────┐
                       ▼               ▼
                    PINN          Model Comparison
                       │               │
                       └───────┬───────┘
                               ▼
                     Aerodynamic Prediction
```

The repository contains the complete workflow from dataset generation through training, evaluation, prediction, visualization, and material comparison.

---

# Objectives

* Develop a machine-learning model for aerodynamic drag prediction.
* Generate a controlled aerodynamic dataset using physics-based correlations.
* Compare multiple regression algorithms.
* Incorporate aerodynamic quantities such as (C_L) and (C_L^2) as ML features.
* Develop a Physics-Informed Neural Network using aerodynamic drag physics.
* Evaluate models using MAE, RMSE, (R^2), and maximum error.
* Analyze feature importance.
* Visualize predicted versus true drag coefficients.
* Investigate the effect of surface/material parameters on aerodynamic drag.

---

# Features

The ML models use the following features:

| Feature    | Description              |
| ---------- | ------------------------ |
| `alpha`    | Angle of attack          |
| `log_re`   | (\log_{10}(Re))          |
| `mach`     | Mach number              |
| `tc_ratio` | Thickness-to-chord ratio |
| `camber`   | Airfoil camber           |
| `cl`       | Lift coefficient         |
| `cl_sq`    | (C_L^2)                  |
| `re_mach`  | (Re \times Mach / 10^6)  |

These features are generated directly in the dataset pipeline and physics module.

The prediction target is:

[
\boxed{C_D}
]

---

# Physics-Based Aerodynamic Model

The dataset is generated using a physics-based drag model implemented in `physics.py`.

The total drag coefficient is constructed from several components:

[
C_D =
C_{D,f}
+
C_{D,p}
+
C_{D,i}
+
C_{D,s}
]

with Mach/compressibility correction applied to the profile-related components.

---

## 1. Skin-Friction Drag

The implementation uses:

### Laminar flow

[
C_f = \frac{1.328}{\sqrt{Re}}
]

based on the Blasius flat-plate correlation.

### Turbulent flow

[
C_f =
\frac{0.455}{(\log_{10}Re)^{2.58}}
]

using a Prandtl-Schlichting correlation.

The final skin-friction coefficient blends laminar and turbulent contributions according to the configured laminar fraction.

---

## 2. Pressure/Form Drag

The implementation uses an empirical Hoerner-type relation:

[
C_{D,p}
=======

C_f
\left[
2\left(\frac{t}{c}\right)
+
60\left(\frac{t}{c}\right)^4
\right]
]

where (t/c) is the airfoil thickness-to-chord ratio.

---

## 3. Lift Coefficient

The lift coefficient is obtained from a thin-airfoil-based formulation:

[
C_L =
2\pi(\alpha-\alpha_{L0})
]

with the result clipped using configured limits to represent stall behavior. Camber and thickness influence the effective limits.

---

## 4. Induced Drag

Induced drag follows the Prandtl lifting-line relation:

[
C_{D,i}
=======

\frac{C_L^2}{\pi e AR}
]

where:

* (AR) = aspect ratio
* (e) = Oswald efficiency factor

This is also why (C_L^2) is included as an ML feature.

---

## 5. Stall Drag

A smooth sigmoid-based drag increase is introduced beyond the estimated stall angle.

The stall angle depends on:

* camber
* thickness ratio

This allows the generated dataset to represent a rapid increase in drag around stall rather than relying only on a linear lift model.

---

## 6. Mach Correction

The model applies a subsonic compressibility correction for higher Mach numbers and introduces an additional wave-drag-rise term at higher Mach values.

The implementation uses a Prandtl-Glauert-style correction below (M=0.70) and adds a transonic wave-drag term above that threshold.

---

# Dataset Generation

`generate_dataset.py` creates the aerodynamic dataset by sweeping across:

* NACA airfoil profiles
* Reynolds numbers
* Mach numbers
* angles of attack

For every combination, the physics model calculates:

* (C_D)
* (C_L)

and additional derived features are stored in the dataset.

The generated records include:

```text
naca
alpha
reynolds
log_re
mach
tc_ratio
camber
cl
cd
cl_sq
re_mach
```

Before saving, the dataset is filtered using configured (C_D) and (C_L) ranges, duplicate records are removed, and the resulting data is split into training and testing sets.

---

# Machine-Learning Models

The training pipeline supports five models:

1. Polynomial Ridge Regression
2. Random Forest Regression
3. Support Vector Regression
4. Artificial Neural Network
5. Physics-Informed Neural Network

The model-selection interface is implemented in `train_models.py`.

---

## 1. Polynomial Ridge Regression

The model first expands the input features using polynomial features and then applies standardization followed by Ridge regression.

```text
Features
   ↓
PolynomialFeatures
   ↓
StandardScaler
   ↓
Ridge Regression
   ↓
Predicted CD
```

This provides a regularized nonlinear baseline.

---

## 2. Random Forest

A Random Forest regressor is trained directly on the engineered aerodynamic features.

The model also provides feature-importance values, which are saved to:

```text
results/feature_importance.csv
```

This allows the contribution of variables such as Mach number, (C_L), camber, and (C_L^2) to be examined.

---

## 3. Support Vector Regression

SVR uses:

```text
StandardScaler
      ↓
SVR (RBF kernel)
```

The RBF kernel allows nonlinear relationships between aerodynamic parameters and (C_D).

---

## 4. Artificial Neural Network

The ANN uses an `MLPRegressor` with feature standardization.

```text
Input Features
      ↓
StandardScaler
      ↓
MLPRegressor
      ↓
Predicted CD
```

---

# Physics-Informed Neural Network

The project also implements a PINN without an external deep-learning framework.

The PINN is written using **NumPy** and implements:

* fully connected neural-network layers,
* ReLU activations,
* Adam optimization,
* gradient clipping,
* feature standardization,
* data loss,
* physics loss.

---

## PINN Architecture

```text
Input Features
      │
      ▼
 ┌───────────┐
 │ Dense     │
 │ ReLU      │
 └─────┬─────┘
       ▼
 ┌───────────┐
 │ Dense     │
 │ ReLU      │
 └─────┬─────┘
       ▼
 ┌───────────┐
 │ Dense     │
 │ ReLU      │
 └─────┬─────┘
       ▼
   Output CD
```

The architecture is configurable through `layer_sizes`.

---

## Physics Loss

The PINN incorporates the aerodynamic drag-polar relationship:

[
C_D = C_{D0} + kC_L^2
]

The training objective therefore combines data-fitting and physics consistency:

[
L =
L_{\text{data}}
+
\lambda_{\text{physics}}
L_{\text{physics}}
]

where `lambda_physics` controls the contribution of the physics loss.

This allows the network to learn from the dataset while also being encouraged to respect a known aerodynamic relationship.

---

# Model Evaluation

Every trained model is evaluated on the held-out test set using:

### Mean Absolute Error

[
MAE =
\frac{1}{N}
\sum_{i=1}^{N}
|y_i-\hat y_i|
]

### Root Mean Squared Error

[
RMSE =
\sqrt{
\frac{1}{N}
\sum_{i=1}^{N}
(y_i-\hat y_i)^2
}
]

### Coefficient of Determination

[
R^2 =
1-
\frac{
\sum(y_i-\hat y_i)^2
}{
\sum(y_i-\bar y)^2
}
]

### Maximum Absolute Error

[
MaxError =
\max_i |y_i-\hat y_i|
]

These metrics are calculated automatically by `train_models.py`.

---

# Generated Results

The training pipeline automatically produces:

```text
results/
├── metrics_all.csv
└── feature_importance.csv
```

and visualization outputs such as:

```text
plots/
├── 06_model_comparison.png
├── 07_feature_importance.png
└── 08_predictions_vs_true.png
```

The model comparison includes MAE, RMSE, and (R^2), while the prediction plot compares true and predicted (C_D).

---

# Exploratory Data Analysis

`eda.py` performs exploratory analysis of the generated aerodynamic dataset.

The repository includes a dedicated EDA script alongside the dataset-generation and training pipelines.

The EDA stage is intended to examine relationships among:

* angle of attack,
* Reynolds number,
* Mach number,
* camber,
* thickness ratio,
* lift coefficient,
* drag coefficient.

---

# Prediction

After training, the saved models can be used for individual aerodynamic predictions through:

```text
predict.py
```

The prediction pipeline uses the same engineered features used during model training, including:

```text
alpha
log_re
mach
tc_ratio
camber
cl
cl_sq
re_mach
```

The physics module provides a helper for constructing these feature rows automatically.

---

# Material Comparison

The repository also contains:

```text
material_comparison.py
```

This provides a separate workflow for comparing aerodynamic behavior under different material/surface-related assumptions.

---

# Airfoil Visualization

`airfoil_plot.py` provides airfoil visualization functionality.

The underlying `physics.py` module also contains a NACA 4-digit coordinate generator that computes:

* upper-surface coordinates,
* lower-surface coordinates,
* camber-line coordinates.

---

# Project Structure

```text
ML-Based-Aerodynamic-Drag-Prediction/
│
├── airfoil_plot.py
├── config.py
├── eda.py
├── generate_dataset.py
├── material_comparison.py
├── physics.py
├── pinn.py
├── predict.py
├── train_models.py
│
├── drag_dataset.csv
├── requirements.txt
│
├── Drag_Prediction_Presentation_v2.pptx
└── final report ME 644.pdf
```

These are the files currently present in the GitHub repository.

---

# Installation — Windows

## 1. Clone the repository

Open **PowerShell** or **Command Prompt**:

```powershell
git clone https://github.com/yuvraj2815/ML-Based-Aerodynamic-Drag-Prediction.git
cd ML-Based-Aerodynamic-Drag-Prediction
```

---

## 2. Create a virtual environment

### PowerShell

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then:

```powershell
.\venv\Scripts\Activate.ps1
```

### Command Prompt

```cmd
python -m venv venv
venv\Scripts\activate
```

---

## 3. Install dependencies

```powershell
pip install -r requirements.txt
```

The repository includes a dedicated `requirements.txt` file.

---

# Running the Project

## Step 1 — Generate the dataset

```powershell
python generate_dataset.py
```

This generates the aerodynamic dataset and train/test CSV files according to the configuration in `config.py`.

---

## Step 2 — Run EDA

```powershell
python eda.py
```

---

## Step 3 — Train all models

```powershell
python train_models.py
```

By default, the training script runs:

```text
Polynomial Ridge
Random Forest
SVR
ANN
PINN
```

---

## Train selected models

For example:

```powershell
python train_models.py --models random_forest svr ann
```

To skip the PINN:

```powershell
python train_models.py --skip-pinn
```

The available model names are:

```text
poly_ridge
random_forest
svr
ann
pinn
```

---

## Step 4 — Generate predictions

```powershell
python predict.py
```

---

## Step 5 — Plot an airfoil

```powershell
python airfoil_plot.py
```

---

## Step 6 — Compare materials

```powershell
python material_comparison.py
```

---

# Reproducibility

The project uses a configured random seed for dataset generation and model training.

The dataset-generation pipeline:

1. defines the aerodynamic parameter space,
2. computes physics-based (C_D) and (C_L),
3. filters invalid/outlier ranges,
4. removes duplicates,
5. performs a train/test split,
6. saves the resulting CSV files.

This makes the experimental workflow reproducible from the configuration and source code.

---

# Configuration

Model and dataset settings are centralized in:

```text
config.py
```

The configuration controls items including:

* airfoil definitions,
* Reynolds-number values,
* Mach-number values,
* angle-of-attack range,
* (C_D) limits,
* (C_L) limits,
* test-set size,
* random seed,
* noise level,
* aspect ratio,
* Oswald efficiency,
* model parameters,
* output directories.

The dataset-generation and training scripts import these settings instead of hard-coding the experimental configuration independently.

---

# End-to-End Workflow

```text
                 ┌─────────────────────┐
                 │  config.py          │
                 │ Experimental Setup  │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ generate_dataset.py │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │   physics.py        │
                 │                     │
                 │ Skin friction       │
                 │ Pressure drag       │
                 │ Induced drag        │
                 │ Stall drag          │
                 │ Mach correction     │
                 └──────────┬──────────┘
                            │
                            ▼
                    Aerodynamic Dataset
                            │
                            ▼
                    ┌───────────────┐
                    │     EDA       │
                    └───────┬───────┘
                            │
                            ▼
                  Feature Engineering
                            │
                            ▼
              ┌─────────────┴─────────────┐
              │                           │
              ▼                           ▼
       Conventional ML                  PINN
              │                           │
       ┌──────┼──────┐                    │
       ▼      ▼      ▼                    │
     Ridge    RF    SVR                   │
                     │                    │
                     ▼                    ▼
                    ANN             Physics Loss
                     │                    │
                     └────────┬───────────┘
                              ▼
                       Model Evaluation
                              │
                              ▼
                  Predictions + Metrics
                              │
                              ▼
                   Engineering Analysis
```

---

# Why Physics-Informed ML?

A purely data-driven model can learn correlations from training data but may not inherently respect known aerodynamic relationships.

This project therefore includes physics-derived quantities such as:

[
C_L,\quad C_L^2,\quad \log_{10}(Re),\quad Re \times Mach
]

and a PINN that explicitly incorporates the drag-polar relationship:

[
C_D=C_{D0}+kC_L^2
]

The goal is to combine the flexibility of machine learning with aerodynamic structure already known from physical modeling.

---

# Technologies

* **Python**
* **NumPy**
* **Pandas**
* **scikit-learn**
* **Matplotlib**
* **Joblib**
* **Physics-based aerodynamic correlations**
* **Polynomial Regression**
* **Random Forest**
* **Support Vector Regression**
* **Artificial Neural Networks**
* **Physics-Informed Neural Networks**

The conventional ML models are implemented using scikit-learn, while the PINN is implemented directly in NumPy rather than using PyTorch or TensorFlow.

---

# Limitations

The aerodynamic dataset is generated using analytical and empirical correlations rather than high-fidelity CFD or experimental measurements.

Consequently, the ML models learn the behavior represented by the underlying physics model. They should therefore be interpreted as surrogate models for this controlled aerodynamic dataset rather than as replacements for validated CFD or wind-tunnel data.

The PINN's physics constraint is also based on the simplified drag-polar relationship implemented in the repository.

---

# Future Improvements

Potential extensions include:

* validating predictions against experimental wind-tunnel data,
* incorporating higher-fidelity CFD datasets,
* expanding the airfoil library,
* including surface roughness directly in the ML feature space,
* extending the model to additional flow regimes,
* comparing additional ensemble and deep-learning architectures,
* performing uncertainty quantification,
* optimizing airfoil geometry using the trained surrogate model,
* investigating transfer learning between airfoil families.

---

# Results and Documentation

The repository includes:

* the final ME 644 project report,
* a project presentation,
* the generated aerodynamic dataset,
* source code for dataset generation,
* physics modeling,
* ML training,
* PINN implementation,
* prediction,
* visualization,
* and material comparison.

---

# Citation

If you use this project, please reference the repository:

**ML-Based Aerodynamic Drag Prediction**
GitHub: `https://github.com/yuvraj2815/ML-Based-Aerodynamic-Drag-Prediction`

---


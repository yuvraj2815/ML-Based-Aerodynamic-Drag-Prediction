import os

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
PLOTS_DIR   = os.path.join(BASE_DIR, "plots")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

DATASET_CSV = os.path.join(DATA_DIR, "drag_dataset.csv")
TRAIN_CSV   = os.path.join(DATA_DIR, "drag_train.csv")
TEST_CSV    = os.path.join(DATA_DIR, "drag_test.csv")

AIRFOILS = {
    "0009": (0.00, 0.0, 0.09),
    "0012": (0.00, 0.0, 0.12),
    "0015": (0.00, 0.0, 0.15),
    "0018": (0.00, 0.0, 0.18),
    "2409": (0.02, 0.4, 0.09),
    "2412": (0.02, 0.4, 0.12),
    "2415": (0.02, 0.4, 0.15),
    "4409": (0.04, 0.4, 0.09),
    "4412": (0.04, 0.4, 0.12),
    "4415": (0.04, 0.4, 0.15),
}

REYNOLDS_LIST = [1e5, 2e5, 5e5, 1e6, 2e6, 5e6]
MACH_LIST     = [0.05, 0.10, 0.20, 0.30, 0.50, 0.70]
ALPHA_START   = -12.0    
ALPHA_END     =  22.0    
ALPHA_STEP    =   0.5   

ASPECT_RATIO    = 8.0   
OSWALD_EFF      = 0.85   
LAM_FRAC        = 0.30  


NOISE_STD_FRAC  = 0.015 
RANDOM_SEED     = 42

CD_MIN = 0.001
CD_MAX = 0.800
CL_MIN = -3.0
CL_MAX =  3.5

TEST_SIZE = 0.20  

FEATURES = ["alpha", "log_re", "mach", "tc_ratio", "camber", "cl", "cl_sq", "re_mach"]
TARGET   = "cd"

MATERIALS = {
    "CFRP":      {"ks": 0.3,  "color": "#1D9E75", "density": 1600},
    "Aluminium": {"ks": 1.0,  "color": "#3B8BD4", "density": 2780},
    "Titanium":  {"ks": 1.4,  "color": "#EF9F27", "density": 4430},
    "Steel":     {"ks": 3.0,  "color": "#E24B4A", "density": 7900},
}

MODEL_PARAMS = {
    "poly_ridge": {
        "degree": 3,
        "alpha":  1.0,
    },
    "random_forest": {
        "n_estimators":    200,
        "max_depth":       20,
        "min_samples_leaf": 2,
        "n_jobs":          -1,
        "random_state":    RANDOM_SEED,
    },
    "svr": {
        "kernel":  "rbf",
        "C":       10.0,
        "epsilon": 0.001,
        "gamma":   "scale",
    },
    "ann": {
        "hidden_layer_sizes":  (128, 64, 32),
        "activation":          "relu",
        "solver":              "adam",
        "learning_rate_init":  0.001,
        "max_iter":            1000,
        "early_stopping":      True,
        "validation_fraction": 0.10,
        "n_iter_no_change":    20,
        "random_state":        RANDOM_SEED,
    },
    "pinn": {
        "layer_sizes":    (len(FEATURES), 128, 64, 32, 1),
        "lr":             0.001,
        "lambda_physics": 0.5,
        "epochs":         500,
        "batch_size":     256,
        "random_state":   RANDOM_SEED,
    },
}

PLOT_DPI    = 150
MODEL_COLORS = {
    "Polynomial Ridge": "#B5D4F4",
    "Random Forest":    "#1D9E75",
    "SVR (RBF)":        "#7F77DD",
    "ANN":              "#EF9F27",
    "PINN":             "#E24B4A",
}
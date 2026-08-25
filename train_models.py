import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib, os, warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model   import Ridge
from sklearn.preprocessing  import StandardScaler, PolynomialFeatures
from sklearn.pipeline       import Pipeline
from sklearn.ensemble       import RandomForestRegressor
from sklearn.svm            import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics        import mean_absolute_error, mean_squared_error, r2_score

from config import (FEATURES, TARGET, MODEL_PARAMS, MODEL_COLORS,
                    TRAIN_CSV, TEST_CSV, MODELS_DIR, RESULTS_DIR, PLOTS_DIR, PLOT_DPI)

os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,   exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--models", nargs="+",
    choices=["poly_ridge", "random_forest", "svr", "ann", "pinn"],
    default=["poly_ridge", "random_forest", "svr", "ann", "pinn"])
parser.add_argument("--skip-pinn", action="store_true")
args = parser.parse_args()

models_to_run = args.models
if args.skip_pinn and "pinn" in models_to_run:
    models_to_run.remove("pinn")

DISPLAY = {
    "poly_ridge": "Polynomial Ridge",
    "random_forest":    "Random Forest",
    "svr":              "SVR (RBF)",
    "ann":              "ANN",
    "pinn":             "PINN",
}

train = pd.read_csv(TRAIN_CSV)
test  = pd.read_csv(TEST_CSV)
X_train = train[FEATURES].values
y_train = train[TARGET].values
X_test  = test[FEATURES].values
y_test  = test[TARGET].values

print("=" * 62)
print("AERODYNAMIC DRAG PREDICTION — TRAINING PIPELINE")
print("=" * 62)
print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")
print(f"  Features: {FEATURES}")
print(f"  Models  : {models_to_run}")

def build_model(name):
    p = MODEL_PARAMS[name]
    if name == "poly_ridge":
        return Pipeline([
            ("poly",   PolynomialFeatures(degree=p["degree"], include_bias=False)),
            ("scaler", StandardScaler()),
            ("model",  Ridge(alpha=p["alpha"])),
        ])
    if name == "random_forest":
        return RandomForestRegressor(**p)
    if name == "svr":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model",  SVR(**p)),
        ])
    if name == "ann":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model",  MLPRegressor(**p)),
        ])
    if name == "pinn":
        from pinn import PINN
        return PINN(**p)
    raise ValueError(f"Unknown model: {name}")

print(f"\n{'─'*62}")
print(f"{'Model':<20}  {'MAE':>8}  {'RMSE':>8}  {'R²':>8}  {'MaxErr':>8}")
print(f"{'─'*20}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")

metrics_list = []
predictions  = {}

for name in models_to_run:
    disp = DISPLAY[name]
    print(f"  Training {disp} ...", end="", flush=True)
    model = build_model(name)
    model.fit(X_train, y_train)
    pred    = model.predict(X_test)
    mae     = mean_absolute_error(y_test, pred)
    rmse    = np.sqrt(mean_squared_error(y_test, pred))
    r2      = r2_score(y_test, pred)
    max_err = float(np.max(np.abs(y_test - pred)))
    print(f"\r  {disp:<20}  {mae:>8.5f}  {rmse:>8.5f}  {r2:>8.4f}  {max_err:>8.5f}")
    metrics_list.append({"Model": disp, "MAE": mae, "RMSE": rmse,
                          "R2": r2, "MaxError": max_err})
    predictions[disp] = pred
    joblib.dump(model, os.path.join(MODELS_DIR, f"{name}.pkl"))

out_m = os.path.join(RESULTS_DIR, "metrics_all.csv")
pd.DataFrame(metrics_list).to_csv(out_m, index=False)
print(f"\n  Metrics: {out_m}")

imp_series = None
if "random_forest" in models_to_run:
    rf = joblib.load(os.path.join(MODELS_DIR, "random_forest.pkl"))
    imp_series = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
    out_i = os.path.join(RESULTS_DIR, "feature_importance.csv")
    imp_series.reset_index().rename(columns={"index":"feature",0:"importance"}).to_csv(out_i, index=False)
    print(f"  Feature importance: {out_i}")
    print("\n  RF Feature Importance:")
    for f, v in imp_series.items():
        print(f"    {f:<12}  {v:.4f}  {'█' * int(v*50)}")

if len(metrics_list) >= 1:
    names  = [m["Model"] for m in metrics_list]
    maes   = [m["MAE"]   for m in metrics_list]
    rmses  = [m["RMSE"]  for m in metrics_list]
    r2s    = [m["R2"]    for m in metrics_list]
    colors = [MODEL_COLORS.get(n, "#888780") for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    x = np.arange(len(names)); w = 0.35
    b1 = ax1.bar(x-w/2, maes,  w, color=colors, alpha=0.9,  label="MAE",  edgecolor="white")
    b2 = ax1.bar(x+w/2, rmses, w, color=colors, alpha=0.55, label="RMSE", edgecolor="white", hatch="//")
    for b in list(b1)+list(b2):
        ax1.text(b.get_x()+b.get_width()/2, b.get_height()+0.0003,
                 f"{b.get_height():.4f}", ha="center", va="bottom", fontsize=8)
    ax1.set_xticks(x); ax1.set_xticklabels(names, fontsize=9, rotation=10)
    ax1.set_ylabel("Error ($C_D$)"); ax1.set_title("MAE and RMSE", fontweight="bold")
    ax1.legend(); ax1.set_ylim(0, max(rmses)*1.35); ax1.grid(axis="y", alpha=0.3)

    bars = ax2.bar(names, r2s, color=colors, alpha=0.9, edgecolor="white")
    for b in bars:
        ax2.text(b.get_x()+b.get_width()/2, b.get_height()-0.012,
                 f"{b.get_height():.4f}", ha="center", va="top", fontsize=9,
                 color="white", fontweight="bold")
    ax2.axhline(0.99, color="red", linestyle="--", linewidth=1.2, alpha=0.7, label="$R^2=0.99$")
    ax2.set_ylabel("$R^2$"); ax2.set_title("R² Score", fontweight="bold")
    ax2.set_ylim(max(0, min(r2s)-0.05), 1.002)
    ax2.legend(); ax2.grid(axis="y", alpha=0.3)
    ax2.set_xticklabels(names, fontsize=9, rotation=10)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "06_model_comparison.png")
    plt.savefig(out, dpi=PLOT_DPI, bbox_inches="tight"); plt.close()
    print(f"\n  Plot: {out}")

if imp_series is not None:
    fc = ["#1D9E75" if v>0.10 else "#3B8BD4" if v>0.01 else "#B4B2A9" for v in imp_series]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.barh(imp_series.index[::-1], imp_series.values[::-1],
                   color=fc[::-1], edgecolor="white", height=0.6)
    for b in bars:
        ax.text(b.get_width()+0.004, b.get_y()+b.get_height()/2,
                f"{b.get_width()*100:.2f}%", va="center", fontsize=9)
    ax.set_xlabel("Importance (MDI)"); ax.set_title("Feature Importance", fontweight="bold")
    ax.set_xlim(0, imp_series.max()*1.15); ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "07_feature_importance.png")
    plt.savefig(out, dpi=PLOT_DPI, bbox_inches="tight"); plt.close()
    print(f"  Plot: {out}")

if predictions:
    n = len(predictions); cols = min(n, 3); rows = (n+cols-1)//cols
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4.5*rows), squeeze=False)
    for i, (nm, pred) in enumerate(predictions.items()):
        ax = axes.flatten()[i]
        col = MODEL_COLORS.get(nm, "#888780")
        ax.scatter(y_test, pred, s=4, alpha=0.2, color=col)
        lims = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
        ax.plot(lims, lims, "k--", linewidth=1.2)
        ax.set_xlabel("True $C_D$"); ax.set_ylabel("Pred $C_D$")
        ax.set_title(f"{nm}  ($R^2={r2_score(y_test,pred):.4f}$)", fontsize=10, fontweight="bold")
        ax.grid(alpha=0.3)
    for j in range(i+1, len(axes.flatten())): axes.flatten()[j].set_visible(False)
    plt.suptitle("Predicted vs True $C_D$", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "08_predictions_vs_true.png")
    plt.savefig(out, dpi=PLOT_DPI, bbox_inches="tight"); plt.close()
    print(f"  Plot: {out}")

print(f"\n{'─'*62}\nDone.  Models → {MODELS_DIR}/")

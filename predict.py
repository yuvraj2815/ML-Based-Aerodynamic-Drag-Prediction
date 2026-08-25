import argparse
import sys
import numpy as np
import pandas as pd
import joblib
import os

from pinn    import PINN 
from config  import (FEATURES, MODELS_DIR, AIRFOILS, ASPECT_RATIO,
                     OSWALD_EFF, LAM_FRAC, REYNOLDS_LIST, MACH_LIST)
from physics import build_feature_row, lift_coefficient


AVAILABLE_MODELS = ["poly_ridge", "random_forest", "svr", "ann", "pinn"]
DISPLAY = {
    "poly_ridge": "Polynomial Ridge",
    "random_forest":    "Random Forest",
    "svr":              "SVR (RBF)",
    "ann":              "ANN",
    "pinn":             "PINN",
}

def load_model(name: str):
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found: {path}\n"
            f"Run 'python train_models.py' first."
        )
    return joblib.load(path)


def feature_row_df(alpha, Re, Mach, tc, camber) -> pd.DataFrame:
    """Build a single-row DataFrame in the exact column order expected by models."""
    row = build_feature_row(alpha, Re, Mach, tc, camber,
                            lam_frac=LAM_FRAC, AR=ASPECT_RATIO, e=OSWALD_EFF)
    return pd.DataFrame([row])[FEATURES]


def predict_single(model, alpha, Re, Mach, tc, camber) -> float:
    X = feature_row_df(alpha, Re, Mach, tc, camber)
    return float(model.predict(X.values)[0])


def validate_alpha(v: float):
    if not -30.0 <= v <= 45.0:
        raise ValueError(f"alpha {v}° is outside typical range [-30, 45]")
    return v

def validate_re(v: float):
    if v < 1e3 or v > 1e9:
        raise ValueError(f"Reynolds number {v:.2e} is outside typical range [1e3, 1e9]")
    return v

def validate_mach(v: float):
    if not 0.0 < v < 1.0:
        raise ValueError(f"Mach {v} must be in range (0, 1) — subsonic only")
    return v

def validate_tc(v: float):
    if not 0.01 <= v <= 0.40:
        raise ValueError(f"t/c={v} outside typical range [0.01, 0.40]")
    return v

def validate_camber(v: float):
    if not 0.0 <= v <= 0.20:
        raise ValueError(f"camber={v} outside typical range [0.0, 0.20]")
    return v


def prompt_float(name: str, unit: str, lo: float, hi: float,
                 example: float, validator) -> float:
    while True:
        try:
            raw = input(f"  {name} [{unit}]  (range {lo} – {hi}, e.g. {example}): ").strip()
            if raw == "":
                raise ValueError("Empty input")
            val = float(raw)
            return validator(val)
        except ValueError as err:
            print(f"    ✗  {err}  — please try again.")


def interactive_input() -> dict:
    print("\n" + "─" * 55)
    print("  Enter airfoil and flow conditions")
    print("─" * 55)
    print("  (Hint: NACA 2412 → alpha=4, Re=1e6, Mach=0.10, tc=0.12, camber=0.02)")
    print()

    print("  Available NACA profiles in training data:")
    for naca, (cam, _, tc) in AIRFOILS.items():
        print(f"    NACA {naca}  →  tc={tc:.2f}  camber={cam:.2f}")
    print()

    alpha  = prompt_float("Angle of attack (α)",  "degrees", -30, 45,   4.0,  validate_alpha)
    Re     = prompt_float("Reynolds number (Re)", "—",       1e3, 1e9, 1e6,  validate_re)
    Mach   = prompt_float("Mach number",          "—",       0.01, 0.99, 0.1, validate_mach)
    tc     = prompt_float("Thickness/chord (t/c)", "—",      0.01, 0.40, 0.12, validate_tc)
    camber = prompt_float("Max camber fraction",   "—",      0.0,  0.20, 0.02, validate_camber)
    return dict(alpha=alpha, Re=Re, Mach=Mach, tc=tc, camber=camber)

def print_result(model_name: str, alpha, Re, Mach, tc, camber, cd_pred: float):
    CL = lift_coefficient(alpha, camber, tc)
    LD = CL / cd_pred if cd_pred > 0 else 0.0
    print(f"\n┌──────────────────────────────────────────────────────────┐")
    print(f"  │  Model        : {DISPLAY.get(model_name, model_name):<27}│")
    print(f"  │  Alpha        : {alpha:>8.2f} °                          │")
    print(f"  │  Reynolds     : {Re:>12.3e}                              │")
    print(f"  │  Mach         : {Mach:>8.3f}                             │")
    print(f"  │  t/c          : {tc:>8.4f}                               │")
    print(f"  │  Camber       : {camber:>8.4f}                           │")
    print(f"  ├──────────────────────────────────────────────────────────┤")
    print(f"  │  C_L (calc)   : {CL:>+8.5f}                              │")
    print(f"  │  C_D (pred)   : {cd_pred:>8.6f}                          │")
    print(f"  │  L/D ratio    : {LD:>8.2f}                               │")
    print(f"  └──────────────────────────────────────────────────────────┘")


def print_comparison(alpha, Re, Mach, tc, camber):
    CL  = lift_coefficient(alpha, camber, tc)
    print(f"\n  Inputs  α={alpha}°  Re={Re:.2e}  Ma={Mach}  t/c={tc}  camber={camber}")
    print(f"  C_L (thin airfoil theory) = {CL:+.5f}")
    print(f"\n  {'Model':<22}  {'Cd':>10}  {'L/D':>8}")
    print(f"  {'─'*22}  {'─'*10}  {'─'*8}")
    for name in AVAILABLE_MODELS:
        path = os.path.join(MODELS_DIR, f"{name}.pkl")
        if not os.path.exists(path):
            print(f"  {DISPLAY[name]:<22}  {'(not trained)':>10}")
            continue
        model = joblib.load(path)
        cd = predict_single(model, alpha, Re, Mach, tc, camber)
        ld = CL / cd if cd > 0 else 0.0
        print(f"  {DISPLAY[name]:<22}  {cd:>10.6f}  {ld:>8.2f}")

def batch_predict(model, input_csv: str, output_csv: str):
    df = pd.read_csv(input_csv)
    required = {"alpha", "re", "mach", "tc", "camber"}
    missing  = required - set(df.columns.str.lower())
    if missing:
        raise ValueError(f"Input CSV missing columns: {missing}")

    df.columns = df.columns.str.lower()
    rows = []
    for _, row in df.iterrows():
        feat = feature_row_df(row["alpha"], row["re"], row["mach"],
                               row["tc"], row["camber"])
        cd = float(model.predict(feat.values)[0])
        rows.append({**row.to_dict(), "cd_predicted": round(cd, 6)})

    out = pd.DataFrame(rows)
    out.to_csv(output_csv, index=False)
    print(f"\n  Predicted {len(out)} rows  →  {output_csv}")
    print(out[["alpha", "re", "mach", "tc", "camber", "cd_predicted"]].to_string(index=False))

def main():
    parser = argparse.ArgumentParser(
        description="Predict aerodynamic drag coefficient Cd",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model",   default="random_forest",
                        choices=AVAILABLE_MODELS, help="Model to use")
    parser.add_argument("--alpha",   type=float, help="Angle of attack (degrees)")
    parser.add_argument("--re",      type=float, help="Reynolds number (e.g. 1e6)")
    parser.add_argument("--mach",    type=float, help="Mach number (e.g. 0.1)")
    parser.add_argument("--tc",      type=float, help="Thickness/chord ratio (e.g. 0.12)")
    parser.add_argument("--camber",  type=float, help="Max camber fraction (e.g. 0.02)")
    parser.add_argument("--compare", action="store_true",
                        help="Compare all trained models for the same input")
    parser.add_argument("--batch",   type=str, help="CSV file with multiple inputs")
    parser.add_argument("--output",  type=str, default="predictions.csv",
                        help="Output CSV for batch predictions")
    args = parser.parse_args()

    print("=" * 55)
    print("  Aerodynamic Drag Coefficient Predictor")
    print("=" * 55)

    cli_vals = {k: getattr(args, k) for k in ("alpha", "re", "mach", "tc", "camber")}
    all_given = all(v is not None for v in cli_vals.values())

    if args.batch:
        model = load_model(args.model)
        batch_predict(model, args.batch, args.output)
        return

    if not all_given:
        inputs = interactive_input()
    else:
        inputs = {
            "alpha":  validate_alpha(args.alpha),
            "Re":     validate_re(args.re),
            "Mach":   validate_mach(args.mach),
            "tc":     validate_tc(args.tc),
            "camber": validate_camber(args.camber),
        }
        # Normalise key names
        if "re" in inputs:
            inputs["Re"] = inputs.pop("re")
        if "mach" in inputs:
            inputs["Mach"] = inputs.pop("mach")

    alpha  = inputs.get("alpha")
    Re     = inputs.get("Re") or inputs.get("re")
    Mach   = inputs.get("Mach") or inputs.get("mach")
    tc     = inputs.get("tc")
    camber = inputs.get("camber")

    if args.compare:
        print_comparison(alpha, Re, Mach, tc, camber)
    else:
        model = load_model(args.model)
        cd    = predict_single(model, alpha, Re, Mach, tc, camber)
        print_result(args.model, alpha, Re, Mach, tc, camber, cd)


if __name__ == "__main__":
    main()

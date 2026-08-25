import numpy as np
import pandas as pd
import os

from config  import (AIRFOILS, REYNOLDS_LIST, MACH_LIST,
                     ALPHA_START, ALPHA_END, ALPHA_STEP,
                     CD_MIN, CD_MAX, CL_MIN, CL_MAX,
                     TEST_SIZE, RANDOM_SEED, NOISE_STD_FRAC,
                     LAM_FRAC, ASPECT_RATIO, OSWALD_EFF,
                     DATA_DIR, DATASET_CSV, TRAIN_CSV, TEST_CSV)
from physics import compute_cd

from sklearn.model_selection import train_test_split

np.random.seed(RANDOM_SEED)
os.makedirs(DATA_DIR, exist_ok=True)

alpha_range = np.arange(ALPHA_START, ALPHA_END, ALPHA_STEP)

print("=" * 60)
print("Generating aerodynamic drag dataset")
print("=" * 60)
print(f"  Profiles   : {len(AIRFOILS)}")
print(f"  Re values  : {len(REYNOLDS_LIST)}")
print(f"  Mach values: {len(MACH_LIST)}")
print(f"  Alpha steps: {len(alpha_range)}  ({ALPHA_START}° to {ALPHA_END - ALPHA_STEP}°, step {ALPHA_STEP}°)")
print(f"  Expected   : {len(AIRFOILS)*len(REYNOLDS_LIST)*len(MACH_LIST)*len(alpha_range):,} rows")

records = []
for naca, (camber, _, tc) in AIRFOILS.items():
    for Re in REYNOLDS_LIST:
        for Mach in MACH_LIST:
            for alpha in alpha_range:
                Cd, CL = compute_cd(
                    alpha_deg=float(alpha),
                    Re=Re, Mach=Mach,
                    camber=camber, tc=tc,
                    ks_um=0.0,
                    lam_frac=LAM_FRAC,
                    AR=ASPECT_RATIO, e=OSWALD_EFF,
                    noise_std_frac=NOISE_STD_FRAC,
                )
                records.append({
                    "naca":     naca,
                    "alpha":    round(float(alpha), 1),
                    "reynolds": Re,
                    "log_re":   round(np.log10(Re), 6),
                    "mach":     Mach,
                    "tc_ratio": tc,
                    "camber":   camber,
                    "cl":       CL,
                    "cd":       Cd,
                    "cl_sq":    round(CL ** 2, 6),
                    "re_mach":  round(Re * Mach / 1e6, 6),
                })

df = pd.DataFrame(records)
before = len(df)
df = df[(df["cd"] >= CD_MIN) & (df["cd"] <= CD_MAX)]
df = df[(df["cl"] >= CL_MIN) & (df["cl"] <= CL_MAX)]
df = df.drop_duplicates().reset_index(drop=True)

print(f"\n  Total rows : {len(df):,}  (removed {before - len(df)} outliers)")
print(f"  Cd range   : {df['cd'].min():.4f} – {df['cd'].max():.4f}")
print(f"  CL range   : {df['cl'].min():.3f} – {df['cl'].max():.3f}")

train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_SEED)

df.to_csv(DATASET_CSV, index=False)
train_df.to_csv(TRAIN_CSV, index=False)
test_df.to_csv(TEST_CSV, index=False)

print(f"\n  Train rows : {len(train_df):,}")
print(f"  Test rows  : {len(test_df):,}")
print(f"\n  Saved: {DATASET_CSV}")
print(f"  Saved: {TRAIN_CSV}")
print(f"  Saved: {TEST_CSV}")
print("\nDone.")

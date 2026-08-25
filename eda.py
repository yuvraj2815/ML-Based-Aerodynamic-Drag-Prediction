import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

from config  import (DATASET_CSV, PLOTS_DIR, FEATURES, TARGET,
                     AIRFOILS, PLOT_DPI, REYNOLDS_LIST, MACH_LIST)
from physics import naca4_coords

os.makedirs(PLOTS_DIR, exist_ok=True)

df = pd.read_csv(DATASET_CSV)
nacas    = sorted(df["naca"].unique())
re_vals  = sorted(df["reynolds"].unique())
ma_vals  = sorted(df["mach"].unique())

print(f"Loaded {len(df):,} rows")
print(f"  NACA profiles : {nacas}")
print(f"  Re values     : {[f'{r:.0e}' for r in re_vals]}")
print(f"  Mach values   : {ma_vals}")
print(f"\nBasic Cd statistics:")
print(df[["alpha","reynolds","mach","tc_ratio","camber","cl","cd"]].describe().round(4).to_string())

def pick_profiles(naca_list, n=3):
    symmetric = [x for x in naca_list if str(x).zfill(4)[0] == "0"]
    cambered  = [x for x in naca_list if str(x).zfill(4)[0] != "0"]
    chosen = []
    if symmetric: chosen.append(symmetric[0])
    if len(cambered) >= 2:
        chosen.append(cambered[0])
        chosen.append(cambered[-1])
    elif cambered:
        chosen.append(cambered[0])
    return chosen[:n]

plot_profiles = pick_profiles(nacas)

def pick_re(re_list, n=3):
    if len(re_list) <= n: return re_list
    indices = [0, len(re_list)//2, len(re_list)-1]
    return [re_list[i] for i in indices]

plot_re = pick_re(re_vals)
re_colors = ["#B5D4F4", "#3B8BD4", "#0C447C"]

mach_ref = 0.1 if 0.1 in ma_vals else ma_vals[1] if len(ma_vals) > 1 else ma_vals[0]
re_ref   = 1e6 if 1e6 in re_vals else re_vals[len(re_vals)//2]

fig, axes = plt.subplots(1, len(plot_profiles), figsize=(5*len(plot_profiles), 4.5))
if len(plot_profiles) == 1: axes = [axes]

for ax, naca in zip(axes, plot_profiles):
    for re_val, col, lbl in zip(plot_re, re_colors, [f"Re={r:.0e}" for r in plot_re]):
        sub = df[(df["naca"]==naca) & (df["reynolds"]==re_val) & (df["mach"]==mach_ref)]
        sub = sub.sort_values("alpha")
        if sub.empty: continue
        ax.plot(sub["alpha"], sub["cd"], color=col, linewidth=1.8, label=lbl)
    ax.set_title(f"NACA {str(naca).zfill(4)}", fontsize=11, fontweight="bold")
    ax.set_xlabel("Angle of attack (°)"); ax.set_ylabel("$C_D$")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0)

fig.suptitle(f"$C_D$ vs $\\alpha$  (Ma={mach_ref})", fontsize=12, fontweight="bold")
plt.tight_layout()
out = os.path.join(PLOTS_DIR, "01_cd_vs_alpha.png")
plt.savefig(out, dpi=PLOT_DPI, bbox_inches="tight"); plt.close()
print(f"\nPlot 1: {out}")

colors_polar = ["#3B8BD4", "#1D9E75", "#EF9F27", "#E24B4A"]
fig, ax = plt.subplots(figsize=(7, 5))
alpha_lo, alpha_hi = df["alpha"].min(), 16.0

for naca, col in zip(plot_profiles, colors_polar):
    sub = df[(df["naca"]==naca) & (df["reynolds"]==re_ref) &
             (df["mach"]==mach_ref) & (df["alpha"].between(alpha_lo, alpha_hi))]
    if sub.empty: continue
    ax.scatter(sub["cl_sq"], sub["cd"], s=10, color=col, alpha=0.6,
               label=f"NACA {str(naca).zfill(4)}")
    m, b = np.polyfit(sub["cl_sq"], sub["cd"], 1)
    xl = np.linspace(0, sub["cl_sq"].max(), 100)
    ax.plot(xl, m*xl+b, color=col, linewidth=1.8, linestyle="--",
            label=f"Fit: $C_D={b:.4f}+{m:.4f}C_L^2$")

ax.set_xlabel("$C_L^2$"); ax.set_ylabel("$C_D$")
ax.set_title(f"Drag Polar  (Re={re_ref:.0e}, Ma={mach_ref})", fontweight="bold")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout()
out = os.path.join(PLOTS_DIR, "02_drag_polar.png")
plt.savefig(out, dpi=PLOT_DPI, bbox_inches="tight"); plt.close()
print(f"Plot 2: {out}")

corr_cols = FEATURES + [TARGET]
corr = df[corr_cols].corr()

fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr_cols))); ax.set_xticklabels(corr_cols, rotation=35, fontsize=9)
ax.set_yticks(range(len(corr_cols))); ax.set_yticklabels(corr_cols, fontsize=9)
for i in range(len(corr_cols)):
    for j in range(len(corr_cols)):
        ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center",
                fontsize=7.5, color="white" if abs(corr.iloc[i,j]) > 0.6 else "black")
plt.colorbar(im, ax=ax, shrink=0.8)
ax.set_title("Feature Correlation Matrix", fontweight="bold")
plt.tight_layout()
out = os.path.join(PLOTS_DIR, "03_correlation_heatmap.png")
plt.savefig(out, dpi=PLOT_DPI, bbox_inches="tight"); plt.close()
print(f"Plot 3: {out}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].hist(df[TARGET], bins=60, color="#3B8BD4", edgecolor="white", alpha=0.85)
axes[0].set_xlabel("$C_D$"); axes[0].set_ylabel("Count")
axes[0].set_title(f"$C_D$ Distribution  (n={len(df):,})", fontweight="bold")
axes[0].grid(alpha=0.3)

for naca, col in zip(plot_profiles, colors_polar):
    sub = df[df["naca"] == naca]
    axes[1].hist(sub[TARGET], bins=40, color=col, alpha=0.6, edgecolor="white",
                 label=f"NACA {str(naca).zfill(4)}")
axes[1].set_xlabel("$C_D$"); axes[1].set_ylabel("Count")
axes[1].set_title("$C_D$ by Profile", fontweight="bold")
axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)
plt.tight_layout()
out = os.path.join(PLOTS_DIR, "04_cd_distribution.png")
plt.savefig(out, dpi=PLOT_DPI, bbox_inches="tight"); plt.close()
print(f"Plot 4: {out}")

draw_list = list(AIRFOILS.keys())[:6] 
draw_colors = ["#3B8BD4","#1D9E75","#EF9F27","#E24B4A","#7F77DD","#0C447C"]

fig, axes = plt.subplots(len(draw_list), 1,
                          figsize=(10, 1.4 * len(draw_list) + 0.5), sharex=True)
if len(draw_list) == 1: axes = [axes]

for ax, naca, col in zip(axes, draw_list, draw_colors):
    camber, _, tc = AIRFOILS[naca]
    xu, yu, xl, yl, yc = naca4_coords(naca)
    ax.fill(np.concatenate([xu, xl[::-1]]), np.concatenate([yu, yl[::-1]]),
            color=col, alpha=0.22)
    ax.plot(np.concatenate([xu, xl[::-1]]), np.concatenate([yu, yl[::-1]]),
            color=col, linewidth=1.8)
    ax.plot(xu, yc, color=col, linewidth=0.9, linestyle="--", alpha=0.7)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle=":", alpha=0.5)
    ax.set_ylabel(f"NACA\n{naca}", fontsize=8.5, rotation=0, labelpad=44)
    ax.set_ylim(-0.17, 0.17); ax.set_yticks([]); ax.grid(False)
    for sp in ax.spines.values(): sp.set_visible(False)

axes[-1].set_xlabel("x/c  (chord fraction)", fontsize=10)
fig.suptitle("NACA Airfoil Profiles  (dashed = camber line)",
             fontsize=11, fontweight="bold")
plt.tight_layout()
out = os.path.join(PLOTS_DIR, "05_airfoil_profiles.png")
plt.savefig(out, dpi=PLOT_DPI, bbox_inches="tight"); plt.close()
print(f"Plot 5: {out}")

print(f"\nEDA complete. All plots in {PLOTS_DIR}/")

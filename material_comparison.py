import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

from config  import (MATERIALS, AIRFOILS, RESULTS_DIR, PLOTS_DIR,
                     PLOT_DPI, ASPECT_RATIO, OSWALD_EFF, LAM_FRAC,
                     REYNOLDS_LIST, MACH_LIST)
from physics import compute_cd

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,   exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--naca",  default="2412",
                    choices=list(AIRFOILS.keys()),
                    help="NACA profile to analyse")
parser.add_argument("--re",    type=float, default=1e6,
                    help="Reference Reynolds number")
parser.add_argument("--mach",  type=float, default=0.1,
                    help="Reference Mach number")
args = parser.parse_args()

naca_key = args.naca
if naca_key not in AIRFOILS:
    raise ValueError(f"NACA {naca_key} not in config.AIRFOILS: {list(AIRFOILS.keys())}")

camber, _, tc = AIRFOILS[naca_key]
Re_ref    = args.re
Mach_ref  = args.mach

# Auto-build alpha sweep from config values
alpha_range = np.arange(-10, 22, 0.5)

# Condition matrix — auto from config
re_sweep   = sorted(set([REYNOLDS_LIST[0], Re_ref, REYNOLDS_LIST[-1]]))
mach_sweep = sorted(set([MACH_LIST[0], Mach_ref, MACH_LIST[-1]]))

alpha_points = [-2.0, 0.0, 4.0, 8.0, 12.0]
conditions = (
    [(f"α={a}°", a, Re_ref, Mach_ref) for a in alpha_points] +
    [(f"Ma={m}",  4.0, Re_ref, m)    for m in mach_sweep if m != Mach_ref] +
    [(f"Re={r:.0e}", 4.0, r, Mach_ref) for r in re_sweep  if r != Re_ref]
)

print("=" * 70)
print(f"MATERIAL COMPARISON — NACA {naca_key}  (Re_ref={Re_ref:.0e}, Ma_ref={Mach_ref})")
print("=" * 70)

mat_names = list(MATERIALS.keys())
baseline  = mat_names[0]
header = f"{'Condition':<16}" + "".join(f"  {m:<14}" for m in mat_names)
print(f"\n{header}")
print("─" * len(header))

rows = []
for label, alpha, Re, Mach in conditions:
    row = {"Condition": label, "alpha": alpha, "Re": Re, "Mach": Mach}
    base_cd = compute_cd(alpha, Re, Mach, camber, tc,
                          ks_um=MATERIALS[baseline]["ks"],
                          lam_frac=LAM_FRAC, AR=ASPECT_RATIO, e=OSWALD_EFF)[0]
    line = f"  {label:<14}"
    for mat, props in MATERIALS.items():
        cd = compute_cd(alpha, Re, Mach, camber, tc,
                        ks_um=props["ks"],
                        lam_frac=LAM_FRAC, AR=ASPECT_RATIO, e=OSWALD_EFF)[0]
        pct = (cd / base_cd - 1) * 100 if base_cd > 0 else 0.0
        row[mat] = round(cd, 6)
        row[f"{mat}_pct"] = round(pct, 2)
        sign = "+" if pct >= 0 else ""
        line += f"  {cd:.5f} ({sign}{pct:.1f}%)"
    print(line)
    rows.append(row)

df_out = pd.DataFrame(rows)
out_csv = os.path.join(RESULTS_DIR, "material_comparison.csv")
df_out.to_csv(out_csv, index=False)
print(f"\n  Saved: {out_csv}")
print(f"  (Percentages relative to {baseline} baseline)")

cds_ref = {
    mat: compute_cd(4.0, Re_ref, Mach_ref, camber, tc,
                    ks_um=props["ks"],
                    lam_frac=LAM_FRAC, AR=ASPECT_RATIO, e=OSWALD_EFF)[0]
    for mat, props in MATERIALS.items()
}
base_cd_ref = cds_ref[baseline]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2))
colors = [p["color"] for p in MATERIALS.values()]

bars = ax1.bar(cds_ref.keys(), cds_ref.values(),
               color=colors, edgecolor="white", width=0.5, alpha=0.9)
for b, (mat, cd) in zip(bars, cds_ref.items()):
    pct = (cd / base_cd_ref - 1) * 100
    sign = "+" if pct >= 0 else ""
    ax1.text(b.get_x()+b.get_width()/2, b.get_height()+0.0002,
             f"{cd:.5f}\n({sign}{pct:.1f}%)", ha="center", va="bottom",
             fontsize=9, fontweight="bold")
ax1.set_ylabel("$C_D$")
ax1.set_title(f"$C_D$ by Material\n"
              f"NACA {naca_key}  α=4°  Re={Re_ref:.0e}  Ma={Mach_ref}",
              fontweight="bold")
ax1.set_ylim(0, max(cds_ref.values()) * 1.28)
ax1.axhline(base_cd_ref, color="gray", linestyle="--", linewidth=1, alpha=0.5)
ax1.grid(axis="y", alpha=0.3)

penalties = {mat: (cd/base_cd_ref-1)*100
             for mat, cd in cds_ref.items() if mat != baseline}
ax2.bar(penalties.keys(), penalties.values(),
        color=[MATERIALS[m]["color"] for m in penalties],
        edgecolor="white", width=0.5, alpha=0.9)
for i, (mat, pct) in enumerate(penalties.items()):
    ax2.text(i, pct + 0.01, f"+{pct:.2f}%", ha="center", va="bottom",
             fontsize=10, fontweight="bold")
ax2.set_ylabel(f"$C_D$ penalty vs {baseline} (%)")
ax2.set_title(f"Drag Penalty vs {baseline} Baseline", fontweight="bold")
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
out = os.path.join(PLOTS_DIR, "11_material_comparison.png")
plt.savefig(out, dpi=PLOT_DPI, bbox_inches="tight"); plt.close()
print(f"  Plot: {out}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

for mat, props in MATERIALS.items():
    cds_a = [compute_cd(a, Re_ref, Mach_ref, camber, tc,
                        ks_um=props["ks"],
                        lam_frac=LAM_FRAC, AR=ASPECT_RATIO, e=OSWALD_EFF)[0]
             for a in alpha_range]
    ax1.plot(alpha_range, cds_a, color=props["color"], linewidth=2, label=mat)

ax1.set_xlabel("Angle of attack (°)"); ax1.set_ylabel("$C_D$")
ax1.set_title(f"$C_D$ vs α by Material\nNACA {naca_key}  Re={Re_ref:.0e}  Ma={Mach_ref}",
              fontweight="bold")
ax1.legend(fontsize=9); ax1.grid(alpha=0.3)

re_sweep_plot = np.logspace(np.log10(min(REYNOLDS_LIST)),
                             np.log10(max(REYNOLDS_LIST)), 60)
for mat, props in MATERIALS.items():
    cds_r = [compute_cd(4.0, Re, Mach_ref, camber, tc,
                        ks_um=props["ks"],
                        lam_frac=LAM_FRAC, AR=ASPECT_RATIO, e=OSWALD_EFF)[0]
             for Re in re_sweep_plot]
    ax2.plot(re_sweep_plot, cds_r, color=props["color"], linewidth=2, label=mat)
ax2.set_xlabel("Reynolds number (Re)"); ax2.set_ylabel("$C_D$")
ax2.set_title(f"$C_D$ vs Re by Material\nNACA {naca_key}  α=4°  Ma={Mach_ref}",
              fontweight="bold")
ax2.set_xscale("log"); ax2.legend(fontsize=9); ax2.grid(alpha=0.3, which="both")

plt.tight_layout()
out = os.path.join(PLOTS_DIR, "12_material_cd_vs_re.png")
plt.savefig(out, dpi=PLOT_DPI, bbox_inches="tight"); plt.close()
print(f"  Plot: {out}")
print("\nMaterial comparison complete.")

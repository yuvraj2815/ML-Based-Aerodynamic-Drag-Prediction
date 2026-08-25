import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

from config  import AIRFOILS, PLOTS_DIR, PLOT_DPI
from physics import naca4_coords

os.makedirs(PLOTS_DIR, exist_ok=True)

PROFILE_COLORS = [
    "#3B8BD4", "#1D9E75", "#EF9F27", "#E24B4A",
    "#7F77DD", "#0C447C", "#085041", "#BA7517",
]


def parse_naca_custom(m: int, p: int, tt: int) -> str:
    """Build a 4-digit NACA code string from M, P, TT integers."""
    return f"{m}{p}{tt:02d}"


def plot_single(naca: str, annotate: bool = True, save: bool = True) -> str:
    code = str(naca).zfill(4)
    m_frac = int(code[0]) / 100
    p_frac = int(code[1]) / 10
    t_frac = int(code[2:]) / 100

    xu, yu, xl, yl, yc = naca4_coords(code, n=500)

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.fill(
        np.concatenate([xu, xl[::-1]]),
        np.concatenate([yu, yl[::-1]]),
        color="#3B8BD4", alpha=0.18, zorder=1
    )
    ax.plot(np.concatenate([xu, xl[::-1]]),
            np.concatenate([yu, yl[::-1]]),
            color="#3B8BD4", linewidth=2.2, zorder=2, label="Airfoil surface")

    ax.plot([0, 1], [0, 0], color="#888780", linewidth=1.0,
            linestyle="--", zorder=1, label="Chord line")

    if m_frac > 0:
        ax.plot(xu, yc, color="#EF9F27", linewidth=1.4,
                linestyle="-.", zorder=3, label="Camber line")

    if annotate:
        idx_t = np.argmax(yu - yl)
        x_t   = (xu[idx_t] + xl[idx_t]) / 2
        y_top = yu[idx_t]
        y_bot = yl[idx_t]
        ax.annotate("", xy=(x_t, y_top), xytext=(x_t, y_bot),
                    arrowprops=dict(arrowstyle="<->", color="#E24B4A", lw=1.5))
        ax.text(x_t + 0.015, (y_top + y_bot) / 2,
                f"t/c = {t_frac:.2f}", color="#E24B4A", fontsize=9,
                va="center", fontweight="bold")

        if m_frac > 0:
            idx_c = np.argmax(yc)
            ax.annotate("", xy=(xu[idx_c], yc[idx_c]), xytext=(xu[idx_c], 0),
                        arrowprops=dict(arrowstyle="<->", color="#EF9F27", lw=1.5))
            ax.text(xu[idx_c] + 0.015, yc[idx_c] / 2,
                    f"camber = {m_frac:.2f}", color="#EF9F27", fontsize=9,
                    va="center", fontweight="bold")

        ax.plot(0, 0, "o", color="#1D9E75", markersize=7, zorder=5)
        ax.text(0.01, -0.035, "Leading edge", fontsize=8.5, color="#1D9E75", fontweight="bold")

        ax.plot(1, 0, "o", color="#E24B4A", markersize=7, zorder=5)
        ax.text(0.88, -0.035, "Trailing edge", fontsize=8.5, color="#E24B4A", fontweight="bold")

        ax.annotate("", xy=(1.0, -0.06), xytext=(0.0, -0.06),
                    arrowprops=dict(arrowstyle="<->", color="#595959", lw=1.2))
        ax.text(0.5, -0.075, "Chord c = 1.0", ha="center",
                fontsize=9, color="#595959")

    ax.set_xlim(-0.05, 1.12)
    ax.set_ylim(-0.14, 0.22)
    ax.set_xlabel("x / c  (normalised chord)", fontsize=11)
    ax.set_ylabel("y / c", fontsize=11)
    ax.set_title(
        f"NACA {code}  |  "
        f"Max camber = {m_frac*100:.0f}%  "
        f"at {p_frac*100:.0f}% chord  |  "
        f"Max thickness = {t_frac*100:.0f}%",
        fontsize=12, fontweight="bold"
    )
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, f"airfoil_NACA{code}.png")
    if save:
        plt.savefig(out, dpi=PLOT_DPI, bbox_inches="tight")
        plt.close()
    return out

def plot_overlay(naca_list: list, save: bool = True) -> str:

    fig, ax = plt.subplots(figsize=(13, 5))

    for i, naca in enumerate(naca_list):
        code  = str(naca).zfill(4)
        col   = PROFILE_COLORS[i % len(PROFILE_COLORS)]
        xu, yu, xl, yl, yc = naca4_coords(code, n=500)

        xs = np.concatenate([xu, xl[::-1]])
        ys = np.concatenate([yu, yl[::-1]])
        ax.fill(xs, ys, color=col, alpha=0.12)
        ax.plot(xs, ys, color=col, linewidth=2.0, label=f"NACA {code}")

        # Camber line (only if cambered)
        m_frac = int(code[0]) / 100
        if m_frac > 0:
            ax.plot(xu, yc, color=col, linewidth=0.9, linestyle="-.", alpha=0.7)

    ax.plot([0, 1], [0, 0], color="#888780", linewidth=0.8,
            linestyle="--", label="Chord line")

    ax.set_xlim(-0.04, 1.08)
    ax.set_xlabel("x / c", fontsize=11)
    ax.set_ylabel("y / c", fontsize=11)
    ax.set_title("NACA Airfoil Profile Comparison", fontsize=12, fontweight="bold")
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=9, ncol=max(1, len(naca_list)//4))

    plt.tight_layout()
    names = "_".join([str(n).zfill(4) for n in naca_list])
    out = os.path.join(PLOTS_DIR, f"airfoil_overlay_{names}.png")
    if save:
        plt.savefig(out, dpi=PLOT_DPI, bbox_inches="tight")
        plt.close()
    return out


def plot_all_stacked(save: bool = True) -> str:
    profiles = list(AIRFOILS.keys())
    n_profiles = len(profiles)
    fig, axes = plt.subplots(n_profiles, 1,
                              figsize=(12, 1.5 * n_profiles + 0.8),
                              sharex=True)
    if n_profiles == 1:
        axes = [axes]

    for ax, naca, col in zip(axes, profiles, PROFILE_COLORS * 4):
        code = str(naca).zfill(4)
        xu, yu, xl, yl, yc = naca4_coords(code, n=500)

        ax.fill(np.concatenate([xu, xl[::-1]]),
                np.concatenate([yu, yl[::-1]]),
                color=col, alpha=0.20)
        ax.plot(np.concatenate([xu, xl[::-1]]),
                np.concatenate([yu, yl[::-1]]),
                color=col, linewidth=1.8)

        m_frac = int(code[0]) / 100
        if m_frac > 0:
            ax.plot(xu, yc, color=col, linewidth=0.9,
                    linestyle="-.", alpha=0.75)

        ax.axhline(0, color="#BBBBBB", linewidth=0.5, linestyle=":")
        ax.set_ylabel(f"NACA {code}", fontsize=9,
                      rotation=0, labelpad=52, va="center")
        ax.set_ylim(-0.16, 0.18)
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    axes[-1].set_xlabel("x / c  (chord fraction)", fontsize=10)
    fig.suptitle("All NACA Profiles  —  dashed = camber line",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()

    out = os.path.join(PLOTS_DIR, "airfoil_all_profiles.png")
    if save:
        plt.savefig(out, dpi=PLOT_DPI, bbox_inches="tight")
        plt.close()
    return out


def print_stats(naca: str):
    code = str(naca).zfill(4)
    m = int(code[0]) / 100
    p = int(code[1]) / 10
    t = int(code[2:]) / 100

    xu, yu, xl, yl, yc = naca4_coords(code, n=500)
    max_t_frac  = np.max(yu - yl)
    max_t_x     = (xu + xl)[np.argmax(yu - yl)] / 2
    max_cam     = np.max(yc)
    max_cam_x   = xu[np.argmax(yc)] if m > 0 else 0.0

    print(f"\n  NACA {code} — Profile Statistics")
    print(f"  {'─'*38}")
    print(f"  Max camber           : {m*100:.1f}%  at {p*100:.0f}% chord")
    print(f"  Max thickness        : {t*100:.1f}%  (nominal)")
    print(f"  Computed max t/c     : {max_t_frac:.4f}  at x/c = {max_t_x:.3f}")
    print(f"  Computed max camber  : {max_cam:.4f}  at x/c = {max_cam_x:.3f}")
    print(f"  Zero-lift angle (est): {-2*np.pi*m * 180/np.pi:.2f}°")
    print(f"  Stall angle (est)    : {12 + 30*m + 20*t:.1f}°")
    print(f"  CL_max (est)         : {1.0 + 5*m + 0.5*t:.3f}")


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Generate NACA airfoil profile plots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--naca", nargs="+", default=None,
                        help="NACA 4-digit code(s), e.g. 2412 or 0012 2412 4415")
    parser.add_argument("--custom", nargs=3, type=int, metavar=("M","P","TT"),
                        help="Custom NACA: M P TT  (e.g. --custom 3 4 18 → NACA 3418)")
    parser.add_argument("--annotate", action="store_true",
                        help="Add dimension annotations to single-profile plots")
    parser.add_argument("--overlay", action="store_true",
                        help="Force overlay mode even for a single profile")
    parser.add_argument("--all", action="store_true",
                        help="Plot all profiles defined in config.py")
    args = parser.parse_args()

    saved = []

    if args.custom:
        code = parse_naca_custom(*args.custom)
        naca_list = [code]
        print(f"  Custom profile: NACA {code.zfill(4)}")
    elif args.naca:
        naca_list = [str(n).zfill(4) for n in args.naca]
    else:
        naca_list = list(AIRFOILS.keys())
        args.all  = True

    print("=" * 55)
    print("  NACA Airfoil Profile Generator")
    print("=" * 55)

    if args.all:
        out = plot_all_stacked()
        print(f"\n  Stacked plot  →  {out}")
        saved.append(out)
        out2 = plot_overlay(list(AIRFOILS.keys()))
        print(f"  Overlay plot  →  {out2}")
        saved.append(out2)
        for naca in AIRFOILS:
            print_stats(naca)

    elif len(naca_list) == 1 and not args.overlay:
        naca = naca_list[0]
        print_stats(naca)
        out = plot_single(naca, annotate=args.annotate)
        print(f"\n  Profile plot  →  {out}")
        saved.append(out)

    else:
        for naca in naca_list:
            print_stats(naca)
        out = plot_overlay(naca_list)
        print(f"\n  Overlay plot  →  {out}")
        saved.append(out)
        for naca in naca_list:
            s = plot_single(naca, annotate=args.annotate)
            print(f"  Single plot   →  {s}")
            saved.append(s)

    print(f"\n  {len(saved)} plot(s) saved to {PLOTS_DIR}/")
    return saved


if __name__ == "__main__":
    main()

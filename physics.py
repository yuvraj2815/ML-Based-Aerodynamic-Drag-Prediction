import numpy as np
from config import ASPECT_RATIO, OSWALD_EFF, LAM_FRAC

def skin_friction_blasius(Re: float) -> float:
    """Laminar skin friction — Blasius flat plate solution."""
    if Re <= 0:
        return 0.02
    return 1.328 / np.sqrt(Re)


def skin_friction_prandtl(Re: float) -> float:
    """Turbulent skin friction — Prandtl-Schlichting correlation."""
    if Re <= 1e4:
        return 0.02
    return 0.455 / (np.log10(Re) ** 2.58)


def skin_friction_blend(Re: float, lam_frac: float = LAM_FRAC) -> float:
    """
    Blend of laminar (Blasius) and turbulent (Prandtl-Schlichting) Cf,
    weighted by laminar fraction of chord. Multiplied by 2 for both surfaces.
    """
    Cf_lam  = skin_friction_blasius(Re)
    Cf_turb = skin_friction_prandtl(Re)
    Cf      = lam_frac * Cf_lam + (1.0 - lam_frac) * Cf_turb
    return 2.0 * Cf


def roughness_correction(Cf_smooth: float, Re: float, ks_um: float,
                          chord: float = 1.0) -> float:
    """
    Moody-type roughness correction to skin friction.
    Cf_rough = Cf_smooth * (1 + 0.03 * log10(Re * ks/c))
    ks_um : surface roughness in micrometres
    chord : chord length in metres (default 1.0 for normalised)
    """
    if ks_um <= 0 or Re <= 0:
        return Cf_smooth
    ks_m   = ks_um * 1e-6
    arg    = max(Re * ks_m / chord, 1e-12)
    factor = 1.0 + 0.03 * np.log10(arg)
    return Cf_smooth * max(factor, 1.0)

def pressure_drag(Cf: float, tc: float) -> float:
    """
    Hoerner's empirical pressure / form drag formula.
    Cdp = Cf * (2*(t/c) + 60*(t/c)^4)
    """
    return Cf * (2.0 * tc + 60.0 * tc ** 4)

def lift_coefficient(alpha_deg: float, camber: float, tc: float) -> float:
    """
    Thin airfoil theory: CL = 2*pi*(alpha - alpha_L0)
    alpha_L0 (zero-lift angle) = -2*pi*camber  [radians]
    Clipped to [CL_min, CL_max] to represent stall.
    """
    alpha_rad = np.radians(alpha_deg)
    alpha_L0  = -2.0 * camber                         # rad
    CL_linear = 2.0 * np.pi * (alpha_rad - alpha_L0)
    CL_max    = 1.0 + 5.0 * camber + 0.5 * tc
    CL_min    = -(0.8 + 3.0 * camber)
    return float(np.clip(CL_linear, CL_min, CL_max))


def induced_drag(CL: float, AR: float = ASPECT_RATIO,
                 e: float = OSWALD_EFF) -> float:
    """
    Prandtl lifting line induced drag.
    Cdi = CL^2 / (pi * e * AR)
    """
    return CL ** 2 / (np.pi * e * AR)

def stall_angle(camber: float, tc: float) -> float:
    """
    Empirical stall angle (degrees).
    alpha_stall = 12 + 30*camber + 20*(t/c)
    """
    return 12.0 + 30.0 * camber + 20.0 * tc


def stall_drag(alpha_deg: float, camber: float, tc: float) -> float:
    """
    Smooth sigmoid Cd spike beyond stall angle.
    Positive and negative stall modelled separately.
    """
    a_stall_pos =  stall_angle(camber, tc)
    a_stall_neg = -(8.0 + 10.0 * camber + 15.0 * tc)

    spike_pos = 0.15 / (1.0 + np.exp(-2.5 * (alpha_deg - a_stall_pos - 2.0)))
    spike_neg = 0.08 / (1.0 + np.exp(-2.5 * (-(alpha_deg - a_stall_neg) - 2.0)))
    return spike_pos + spike_neg

def mach_correction(Cd0: float, Mach: float) -> float:
    """
    Prandtl-Glauert compressibility correction (subsonic: Ma < 0.70).
    Transonic wave drag rise added for Ma >= 0.70.
    """
    if Mach < 0.30:
        return Cd0
    elif Mach < 0.70:
        beta = np.sqrt(max(1.0 - Mach ** 2, 1e-6))
        return Cd0 / beta
    else:
        beta      = np.sqrt(abs(1.0 - Mach ** 2) + 1e-6)
        wave_drag = max(0.0, ((Mach - 0.70) / 0.30) ** 3) * 0.10
        return Cd0 / beta + wave_drag

def compute_cd(alpha_deg: float, Re: float, Mach: float,
               camber: float, tc: float,
               ks_um: float = 0.0,
               lam_frac: float = LAM_FRAC,
               AR: float = ASPECT_RATIO,
               e: float = OSWALD_EFF,
               noise_std_frac: float = 0.0) -> tuple:
    Cf_smooth = skin_friction_blend(Re, lam_frac)
    Cf        = roughness_correction(Cf_smooth, Re, ks_um) if ks_um > 0 else Cf_smooth

    Cd_f  = Cf
    Cd_p  = pressure_drag(Cf, tc)
    CL    = lift_coefficient(alpha_deg, camber, tc)
    Cd_i  = induced_drag(CL, AR, e)
    Cd_s  = stall_drag(alpha_deg, camber, tc)

    Cd0       = Cd_f + Cd_p + Cd_s
    Cd_total  = mach_correction(Cd0, Mach) + Cd_i

    if noise_std_frac > 0:
        noise    = np.random.normal(0, noise_std_frac * Cd_total)
        Cd_total = max(Cd_total + noise, 0.001)

    return round(float(Cd_total), 6), round(float(CL), 5)

def build_feature_row(alpha: float, Re: float, Mach: float,
                      tc: float, camber: float,
                      lam_frac: float = LAM_FRAC,
                      AR: float = ASPECT_RATIO,
                      e: float = OSWALD_EFF) -> dict:
    """
    Build a feature dictionary for ML model input.
    Computes derived features (log_re, cl, cl_sq, re_mach) automatically.
    """
    CL = lift_coefficient(alpha, camber, tc)
    return {
        "alpha":    alpha,
        "log_re":   round(np.log10(Re), 6),
        "mach":     Mach,
        "tc_ratio": tc,
        "camber":   camber,
        "cl":       round(CL, 6),
        "cl_sq":    round(CL ** 2, 6),
        "re_mach":  round(Re * Mach / 1e6, 6),
    }

def naca4_coords(naca_code: str, n: int = 300) -> tuple:
    code = str(naca_code).zfill(4)
    m = int(code[0]) / 100.0  
    p = int(code[1]) / 10.0   
    t = int(code[2:]) / 100.0  

    x = np.linspace(0, 1, n)

    yt = 5 * t * (0.2969 * np.sqrt(x)
                  - 0.1260 * x
                  - 0.3516 * x**2
                  + 0.2843 * x**3
                  - 0.1015 * x**4)
    if m == 0.0 or p == 0.0:
        yc  = np.zeros_like(x)
        dyc = np.zeros_like(x)
    else:
        yc  = np.where(x <= p,
                       m / p**2 * (2*p*x - x**2),
                       m / (1-p)**2 * (1 - 2*p + 2*p*x - x**2))
        dyc = np.where(x <= p,
                       2*m / p**2 * (p - x),
                       2*m / (1-p)**2 * (p - x))

    theta = np.arctan(dyc)

    xu = x  - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x  + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    return xu, yu, xl, yl, yc

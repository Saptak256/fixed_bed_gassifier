#!/usr/bin/env python3
"""
1-D steady-state reduction-zone model of a downdraft fixed-bed gasifier
=======================================================================
Re-creation of Ouedraogo et al., Case Studies in Thermal Engineering 68 (2025)
105910, "Optimizing gasification in downdraft fixed-bed gasifier through
experimental and kinetic modeling".

DESIGN GOALS
  * No data leakage.  The model is calibrated ONLY on the temperature sweep
    (journal Fig. 6).  The inlet gas composition comes from the paper's TEXT
    (Sec. 4.2, experimental gas), never from a benchmark CSV.  The bed-depth
    profile (journal Fig. 9) is opened for the first time AFTER the parameters
    are frozen and hashed, so it is a genuine held-out test.
  * Physically consistent bookkeeping: species molar flows (mol/m2/s) with
    exact stoichiometry, thermodynamically consistent equilibrium constants
    (K4 = K2/K3 by construction), reversible rates (no max(0, .) clipping).
  * Few free parameters (5), structure chosen by leave-one-temperature-out
    cross-validation on the TRAINING sweep only.
  * Honest uncertainty: leave-one-out cross-validation error and a
    jackknife envelope on every prediction.

GAS-COMPOSITION BASIS (important)
  The journal's "dry gas vol.%" are not normalised to 100 %: its Fig. 5 sums
  to >100 % and its CCE/CGE imply a constant gas yield along the bed, i.e.
  N2 is pinned at its inlet fraction ("N2-tracer basis").  To reproduce the
  journal numbers this script reports
        y_i = x_N2,inlet * F_i / F_N2
  as the primary output ("journal basis").  The physically normalised dry
  composition is also written to the CSV output for reference.

Usage
  python simulate_gasifier.py                 # strict train(temp) -> test(depth)
  python simulate_gasifier.py --selftest      # internal consistency checks
  python simulate_gasifier.py --joint         # also refit on BOTH sets (in-sample, flagged)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

warnings.filterwarnings("ignore", category=RuntimeWarning)

# =============================================================================
# 1. CONSTANTS FROM THE PAPER
# =============================================================================
R_GAS = 8.314  # J/mol/K
SPECIES = ["CO", "CO2", "H2", "CH4", "H2O", "N2"]
I_CO, I_CO2, I_H2, I_CH4, I_H2O, I_N2 = range(6)
CSV_COLS = ["CO_vol_pct", "CO2_vol_pct", "H2_vol_pct", "CH4_vol_pct"]  # model order
REPORTED = ["CO", "CO2", "H2", "CH4"]

# Table 3: C+CO2<->2CO, C+H2O<->CO+H2, C+2H2<->CH4, CH4+H2O<->CO+3H2
E_ACT = np.array([77.39e3, 121.62e3, 19.21e3, 36.15e3])   # J/mol
A_PRE = np.array([3.616e4, 1.517e7, 4.189, 7.301e-2])     # 1/s
DH_RXN = np.array([172.0e3, 131.3e3, -74.8e3, 205.9e3])   # J/mol

# Standard-state data used ONLY to build thermodynamically consistent K_i(T)
# (constant dH/dS approximation; K4 = K2/K3 holds exactly).  298 K, J/mol, J/mol/K
_DH298 = np.array([172.46e3, 131.29e3, -74.87e3, 206.16e3])
_DS298 = np.array([175.81, 133.78, -80.84, 214.62])

# Heat-capacity polynomials cp/R = A + B T + C T^2 + D / T^2 (Smith-Van Ness-Abbott)
_CP = np.array([
    [3.376, 0.557e-3, 0.0, -0.031e5],    # CO
    [5.457, 1.045e-3, 0.0, -1.157e5],    # CO2
    [3.249, 0.422e-3, 0.0, 0.083e5],     # H2
    [1.702, 9.081e-3, -2.164e-6, 0.0],   # CH4
    [3.470, 1.450e-3, 0.0, 0.121e5],     # H2O
    [3.280, 0.593e-3, 0.0, 0.040e5],     # N2
])

# Experimental gas entering the reduction model (paper Sec. 4.2, dry basis, vol.%)
INLET_DRY = {"CO": 15.10, "CO2": 13.13, "H2": 7.39, "CH4": 3.16}
INLET_N2 = 100.0 - sum(INLET_DRY.values())          # by difference (paper: N2 by difference)

# Published scalars (paper TEXT / Table 6; not in any CSV, never used for fitting)
PAPER = {
    "inlet_LHV": 3.82, "inlet_CGE": 36.60, "inlet_CCE": 62.37,                 # Sec. 4.2
    "T1000": {"CO": 30.77, "CO2": 14.13, "H2": 18.76, "CH4": 4.32,             # Table 6
              "LHV": 7.46, "CGE": 71.53, "CCE": 97.99},
    "depth_exit": {"CO": 29.53, "H2": 18.15, "CH4": 4.07},                      # Sec. 4.5
}

T_REF = 1273.15      # K, reference for devolatilisation Arrhenius normalisation


@dataclass(frozen=True)
class Config:
    bed_depth: float = 0.5        # m   (Table 5)
    v_in: float = 0.699           # m/s superficial inlet velocity (Table 5)
    pressure_atm: float = 1.0     # pressure drop (28-34 Pa in paper) is <0.04 % -> neglected
    thermal_buffer: float = 100.0 # a-priori ratio (solid+wall)/gas heat-capacity flow; NOT fitted


# Free parameters (5):  theta = [ln phi, w_steam, ln s_CH4, ln s_CO2, Ed_kJ]
#   phi      common char-reactivity / pore-diffusion factor on ALL four paper rates
#            (paper: "pore diffusion would uniformly impact the rates of all reactions")
#   w_steam  inlet steam mole fraction (wet basis); paper reports dry gas only
#   s_CH4, s_CO2, Ed  residual-char devolatilisation (the paper itself lists the
#            "relationship between pyrolysis product quantities and temperature"
#            as the missing piece), mol/m3/s at 1273.15 K, activation energy kJ/mol
PARAM_NAMES = ["ln_phi", "w_steam", "ln_s_CH4", "ln_s_CO2", "Ed_kJ_per_mol"]
LB = np.array([-12.0, 0.00, -14.0, -14.0, 20.0])
UB = np.array([2.0, 0.50, 4.0, 4.0, 400.0])
THETA_START = np.array([-2.7, 0.18, -1.8, -1.1, 90.0])


# =============================================================================
# 2. THERMODYNAMICS
# =============================================================================
def equilibrium_constants(T: float) -> np.ndarray:
    """K1..K4 in atm^(dn).  Consistent: K4 = K2/K3."""
    return np.exp(_DS298 / R_GAS - _DH298 / (R_GAS * T))


def cp_molar(T: float) -> np.ndarray:
    return R_GAS * (_CP[:, 0] + _CP[:, 1] * T + _CP[:, 2] * T**2 + _CP[:, 3] / T**2)


def inlet_wet_fractions(w_steam: float) -> np.ndarray:
    dry = np.array([INLET_DRY["CO"], INLET_DRY["CO2"], INLET_DRY["H2"], INLET_DRY["CH4"],
                    INLET_N2]) / 100.0
    x = np.zeros(6)
    x[[I_CO, I_CO2, I_H2, I_CH4, I_N2]] = dry * (1.0 - w_steam)
    x[I_H2O] = w_steam
    return x


# =============================================================================
# 3. REACTOR MODEL (steady, 1-D, plug flow)   dF_i/dz = R_i
# =============================================================================
@dataclass
class Profile:
    z: np.ndarray
    T: np.ndarray
    F: np.ndarray                      # (6, n) molar flows mol/m2/s
    ok: bool = True
    journal: np.ndarray = field(default=None)   # (n,4) vol.% on journal (N2-tracer) basis
    dry_norm: np.ndarray = field(default=None)  # (n,4) vol.% normalised dry


def _rhs_factory(theta: np.ndarray, cfg: Config):
    phi = np.exp(theta[0])
    s_ch4, s_co2 = np.exp(theta[2]), np.exp(theta[3])
    Ed = theta[4] * 1e3
    beta = cfg.thermal_buffer

    def rhs(_z, y):
        F = np.maximum(y[:6], 1e-14)
        T = y[6]
        x = F / F.sum() * cfg.pressure_atm        # partial pressures [atm]
        K = equilibrium_constants(T)
        k = phi * A_PRE * np.exp(-E_ACT / (R_GAS * T))
        r1 = k[0] * (x[I_CO2] - x[I_CO] ** 2 / K[0])
        r2 = k[1] * (x[I_H2O] - x[I_CO] * x[I_H2] / K[1])
        r3 = k[2] * (x[I_H2] ** 2 - x[I_CH4] / K[2])
        r4 = k[3] * (x[I_CH4] * x[I_H2O] - x[I_CO] * x[I_H2] ** 3 / K[3])
        dev = np.exp(-Ed / R_GAS * (1.0 / T - 1.0 / T_REF))
        dF = np.array([
            2 * r1 + r2 + r4,                 # CO
            -r1 + s_co2 * dev,                # CO2
            r2 - 2 * r3 + 3 * r4,             # H2
            r3 - r4 + s_ch4 * dev,            # CH4
            -r2 - r4,                         # H2O
            0.0,                              # N2 inert
        ])
        r = np.array([r1, r2, r3, r4])
        dT = -(r @ DH_RXN) / ((1.0 + beta) * (F @ cp_molar(T)))
        return np.append(dF, dT)

    return rhs


def simulate(theta, T_in_K: float, z_eval=None, cfg: Config = Config()) -> Profile:
    theta = np.asarray(theta, float)
    F_total0 = cfg.pressure_atm * 101325.0 * cfg.v_in / (R_GAS * T_in_K)   # mol/m2/s
    y0 = np.append(inlet_wet_fractions(theta[1]) * F_total0, T_in_K)
    z_eval = np.array([cfg.bed_depth]) if z_eval is None else np.clip(np.asarray(z_eval, float), 0, cfg.bed_depth)
    rhs = _rhs_factory(theta, cfg)
    sol = None
    for method, tol in (("LSODA", 1e-8), ("Radau", 1e-8), ("BDF", 1e-7)):   # robustness: solver fallback
        sol = solve_ivp(rhs, (0.0, cfg.bed_depth), y0, method=method, t_eval=z_eval,
                        rtol=tol, atol=1e-11)
        if sol.success and np.all(np.isfinite(sol.y)):
            break
    ok = bool(sol.success and np.all(np.isfinite(sol.y)))
    F = np.maximum(sol.y[:6], 0.0)
    xN2_in = INLET_N2 / 100.0
    journal = (F[[I_CO, I_CO2, I_H2, I_CH4]] / F[I_N2] * xN2_in * 100.0).T
    dry = F[[I_CO, I_CO2, I_H2, I_CH4, I_N2]]
    dry_norm = (dry[:4] / dry.sum(axis=0) * 100.0).T
    return Profile(sol.t, sol.y[6], F, ok, journal, dry_norm)


def exit_composition(theta, T_in_K, cfg=Config()) -> np.ndarray:
    return simulate(theta, T_in_K, None, cfg).journal[-1]


# =============================================================================
# 4. CALIBRATION  (receives ONLY the temperature sweep)
# =============================================================================
def _check_df(df: pd.DataFrame, need: list[str], name: str):
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"{name}: missing columns {miss}")
    if df[need].isna().any().any():
        raise ValueError(f"{name}: contains NaN values")


def calibrate(temp_df: pd.DataFrame, cfg: Config = Config(), rows=None, n_starts: int = 4,
              seed: int = 0) -> tuple[np.ndarray, float]:
    """Least-squares fit on exit composition vs inlet temperature (vol.%)."""
    _check_df(temp_df, ["initial_temp_C"] + CSV_COLS, "temperature CSV")
    if "bed_depth_m" in temp_df.columns:
        raise RuntimeError("Leakage guard: depth data must never reach calibrate().")
    df = temp_df if rows is None else temp_df.iloc[list(rows)]
    T_K = df["initial_temp_C"].to_numpy(float) + 273.15
    Y = df[CSV_COLS].to_numpy(float)

    def resid(th):
        out = []
        for T, y in zip(T_K, Y):
            out.append(exit_composition(th, T, cfg) - y)
        return np.concatenate(out)

    rng = np.random.RandomState(seed)
    starts = [THETA_START] + [np.clip(THETA_START + rng.normal(0, 1, 5) * np.array([0.6, 0.05, 0.8, 0.8, 25]),
                                      LB + 1e-6, UB - 1e-6) for _ in range(n_starts - 1)]
    best = None
    for s in starts:
        try:
            r = least_squares(resid, s, bounds=(LB, UB), x_scale=np.array([1, 0.1, 1, 1, 50.0]),
                              diff_step=1e-4)
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    if best is None:
        raise RuntimeError("Calibration failed for all starting points")
    return best.x, float(np.sqrt(2 * best.cost / best.fun.size))


def freeze_parameters(theta, train_df, path: Path, extra=None) -> str:
    payload = {"parameters": dict(zip(PARAM_NAMES, map(float, theta))),
               "trained_on": "journal_benchmark_temperature.csv ONLY",
               "train_csv_sha256": hashlib.sha256(train_df.to_csv(index=False).encode()).hexdigest()}
    if extra:
        payload.update(extra)
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    payload["freeze_digest"] = digest
    path.write_text(json.dumps(payload, indent=2))
    return digest


# =============================================================================
# 5. METRICS
# =============================================================================
def metrics(pred: np.ndarray, obs: np.ndarray) -> pd.DataFrame:
    err = pred - obs
    rows = {}
    for j, sp in enumerate(REPORTED):
        e = err[:, j]
        ss_tot = np.sum((obs[:, j] - obs[:, j].mean()) ** 2)
        rows[sp] = {"RMSE": np.sqrt(np.mean(e**2)), "MAE": np.mean(np.abs(e)),
                    "MaxAbs": np.max(np.abs(e)), "Bias": np.mean(e),
                    "R2": 1 - np.sum(e**2) / ss_tot if ss_tot > 0 else np.nan}
    rows["ALL"] = {"RMSE": np.sqrt(np.mean(err**2)), "MAE": np.mean(np.abs(err)),
                   "MaxAbs": np.max(np.abs(err)), "Bias": np.mean(err),
                   "R2": 1 - np.sum(err**2) / np.sum((obs - obs.mean(0)) ** 2)}
    return pd.DataFrame(rows).T


def performance_indicators(y: np.ndarray, calib: dict) -> dict:
    """LHV (Eq. 30), CGE (Eq. 31) and CCE (Eq. 33) from composition y=[CO,CO2,H2,CH4] vol.%."""
    lhv = (126.36 * y[0] + 107.98 * y[2] + 358.18 * y[3]) / 1000.0     # MJ/Nm3
    return {"LHV": lhv, "CGE": calib["k_cge"] * lhv, "CCE": calib["k_cce"] * (y[0] + y[1] + y[3])}


def calibrate_indicator_constants() -> dict:
    """Gas-yield constants from the paper's EXPERIMENTAL inlet values (text only)."""
    y = np.array([INLET_DRY[s] for s in REPORTED])
    lhv = (126.36 * y[0] + 107.98 * y[2] + 358.18 * y[3]) / 1000.0
    return {"k_cge": PAPER["inlet_CGE"] / lhv, "k_cce": PAPER["inlet_CCE"] / (y[0] + y[1] + y[3])}


# =============================================================================
# 6. SELF-TESTS
# =============================================================================
def selftest(cfg=Config()) -> bool:
    ok = True

    def chk(name, cond):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        ok &= bool(cond)

    print("Self-tests")
    for T in (900.0, 1100.0, 1273.15):
        K = equilibrium_constants(T)
        chk(f"K4 == K2/K3 at {T:.0f} K", np.isclose(K[3], K[1] / K[2], rtol=1e-10))
    th = THETA_START.copy()
    th0 = th.copy(); th0[0] = -12; th0[2] = th0[3] = -14
    p = simulate(th0, 1273.15, None, cfg)
    chk("negligible reactivity -> inlet composition preserved",
        np.allclose(p.journal[-1], [INLET_DRY[s] for s in REPORTED], atol=0.05))
    p = simulate(th, 1273.15, np.linspace(0, cfg.bed_depth, 51), cfg)
    chk("N2 molar flow constant along bed", np.allclose(p.F[I_N2], p.F[I_N2, 0]))
    chk("all molar flows non-negative", (p.F >= 0).all())
    # H and O atom balances of the four reactions (devolatilisation sources switched off)
    thx = th.copy(); thx[2] = thx[3] = -14
    pa = simulate(thx, 1273.15, [0.0, cfg.bed_depth], cfg)
    H = lambda F: 2 * F[I_H2] + 4 * F[I_CH4] + 2 * F[I_H2O]
    O = lambda F: F[I_CO] + 2 * F[I_CO2] + F[I_H2O]
    chk("H atoms conserved by the 4 reactions", np.isclose(H(pa.F[:, 0]), H(pa.F[:, 1]), rtol=1e-6))
    chk("O atoms conserved by the 4 reactions", np.isclose(O(pa.F[:, 0]), O(pa.F[:, 1]), rtol=1e-6))
    # equilibrium limit: huge reactivity, long bed -> Boudouard & water-gas at equilibrium
    thq = thx.copy(); thq[0] = 2.0; thq[1] = 0.3
    pq = simulate(thq, 1300.0, None, Config(bed_depth=20.0, v_in=cfg.v_in, thermal_buffer=1e6))
    x = pq.F[:, -1] / pq.F[:, -1].sum(); K = equilibrium_constants(pq.T[-1])
    chk("long bed approaches Boudouard equilibrium",
        abs(x[I_CO] ** 2 / x[I_CO2] / K[0] - 1) < 0.05)
    chk("long bed approaches water-gas equilibrium",
        abs(x[I_CO] * x[I_H2] / max(x[I_H2O], 1e-12) / K[1] - 1) < 0.05 or x[I_H2O] < 1e-6)
    chk("solver succeeds across 600-1100 C",
        all(simulate(th, T + 273.15, None, cfg).ok for T in range(600, 1101, 50)))
    print("ALL SELF-TESTS PASSED" if ok else "SOME SELF-TESTS FAILED")
    return ok


# =============================================================================
# 7. PLOTTING
# =============================================================================
COLORS = {"CO": "tab:red", "CO2": "k", "H2": "tab:blue", "CH4": "tab:green"}
MARKERS = {"CO": "o", "CO2": "^", "H2": "s", "CH4": "d"}


def make_plots(theta, temp_df, depth_df, jk_depth, jk_temp, cfg, outdir: Path, show: bool):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # --- (a) held-out validation: species vs depth ---------------------------
    z = np.linspace(0, cfg.bed_depth, 201)
    prof = simulate(theta, 1273.15, z, cfg)
    fig, ax = plt.subplots(figsize=(9, 6))
    zb = depth_df["bed_depth_m"].to_numpy()
    for j, sp in enumerate(REPORTED):
        c = COLORS[sp]
        ax.plot(z, prof.journal[:, j], "-", color=c, lw=2, label=f"Sim {sp}")
        if jk_depth is not None:
            ax.fill_between(jk_depth[0], jk_depth[1][:, j], jk_depth[2][:, j], color=c, alpha=0.15, lw=0)
        ax.scatter(zb, depth_df[CSV_COLS[j]], s=14, color=c, alpha=0.55, marker=MARKERS[sp],
                   label=f"Journal {sp} (held-out)")
    ax.set(xlabel="Reduction zone depth (m)", ylabel="Gas composition (vol.%, journal N2-tracer basis)",
           title="Held-out validation: species vs. bed depth (1000 °C inlet)\n"
                 "model calibrated on temperature sweep only; shading = leave-one-out envelope")
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    fig.tight_layout(); fig.savefig(outdir / "Validation_Species_Graph.png", dpi=300, bbox_inches="tight")

    # --- (b) calibration: exit composition vs inlet temperature ---------------
    Tg = np.linspace(650, 1000, 71)
    fig, ax = plt.subplots(figsize=(9, 6))
    for j, sp in enumerate(REPORTED):
        c = COLORS[sp]
        ax.plot(Tg, [exit_composition(theta, T + 273.15, cfg)[j] for T in Tg], "-", color=c, lw=2, label=f"Sim {sp}")
        ax.scatter(temp_df["initial_temp_C"], temp_df[CSV_COLS[j]], s=22, color=c, marker=MARKERS[sp],
                   label=f"Journal {sp} (training)")
        if jk_temp is not None:
            ax.scatter(temp_df["initial_temp_C"], jk_temp[:, j], s=70, facecolors="none", edgecolors=c,
                       marker="o", lw=1.2)
    ax.set(xlabel="Reduction-zone inlet temperature (°C)", ylabel="Exit composition (vol.%)",
           title="Calibration data (filled) and leave-one-out predictions (open circles)")
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    fig.tight_layout(); fig.savefig(outdir / "Temperature_Calibration.png", dpi=300, bbox_inches="tight")

    # --- (c) parity plot of the two data sets --------------------------------
    pred_t = np.array([exit_composition(theta, T + 273.15, cfg) for T in temp_df["initial_temp_C"]])
    pred_d = simulate(theta, 1273.15, zb, cfg).journal
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for j, sp in enumerate(REPORTED):
        ax.scatter(temp_df[CSV_COLS[j]], pred_t[:, j], color=COLORS[sp], marker="o", s=30,
                   label=f"{sp} training")
        ax.scatter(depth_df[CSV_COLS[j]], pred_d[:, j], color=COLORS[sp], marker="x", s=18,
                   label=f"{sp} held-out")
    lim = [2, 33]; ax.plot(lim, lim, "k--", lw=1)
    ax.set(xlim=lim, ylim=lim, xlabel="Journal (vol.%)", ylabel="Simulation (vol.%)", title="Parity plot")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.4)
    fig.tight_layout(); fig.savefig(outdir / "Parity_Plot.png", dpi=300)

    # --- (d) 1-D profile replicated radially (model has no radial dimension) ---
    radius = np.linspace(-0.1, 0.1, 50)
    Z, Rr = np.meshgrid(z, radius)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 8))
    c1 = a1.contourf(Z, Rr, np.tile(prof.T - 273.15, (len(radius), 1)), cmap="inferno", levels=50)
    fig.colorbar(c1, ax=a1, label="Gas temperature (°C)")
    a1.set(ylabel="Radial position (m)", title="Axial temperature profile (1-D model, shown replicated radially)")
    c2 = a2.contourf(Z, Rr, np.tile(prof.journal[:, 0], (len(radius), 1)), cmap="viridis", levels=50)
    fig.colorbar(c2, ax=a2, label="CO (vol.%)")
    a2.set(xlabel="Axial reactor depth (m)", ylabel="Radial position (m)", title="CO profile (1-D model)")
    fig.tight_layout(); fig.savefig(outdir / "Reactor_Contours.png", dpi=300)
    if show:
        plt.show()
    plt.close("all")


# =============================================================================
# 8. MAIN PIPELINE
# =============================================================================
def run(args) -> int:
    cfg = Config()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    ddir = Path(args.data_dir)
    tpath, dpath = ddir / "journal_benchmark_temperature.csv", ddir / "journal_benchmark_depth.csv"
    if not tpath.exists():
        print(f"ERROR: {tpath} not found"); return 1

    print("=" * 74)
    print("FIXED-BED GASIFIER REDUCTION-ZONE MODEL  (Ouedraogo et al., 2025)")
    print("=" * 74)
    print("DATA-LEAKAGE AUDIT")
    print("  inlet gas composition : paper text Sec. 4.2 (NOT from any CSV)")
    print("  calibration data      : journal_benchmark_temperature.csv (Fig. 6) only")
    print("  held-out test data    : journal_benchmark_depth.csv (Fig. 9), opened AFTER parameter freeze")
    print("  indicator constants   : paper text, experimental inlet values only")

    # ---- STEP 1: calibrate on the temperature sweep ONLY -------------------
    temp_df = pd.read_csv(tpath)
    print("\n[1] Calibrating on temperature sweep (8 points x 4 species) ...", flush=True)
    theta, train_rmse = calibrate(temp_df, cfg, n_starts=args.starts)
    digest = freeze_parameters(theta, temp_df, outdir / "calibrated_parameters.json")
    print("    parameters frozen, digest", digest[:16])
    for n, v in zip(PARAM_NAMES, theta):
        print(f"      {n:14s} = {v:10.4f}")
    print(f"      (phi = {np.exp(theta[0]):.4g},  s_CH4 = {np.exp(theta[2]):.4g},  s_CO2 = {np.exp(theta[3]):.4g} mol/m3/s)")
    at_bound = [n for n, v, lo, hi in zip(PARAM_NAMES, theta, LB, UB) if min(v - lo, hi - v) < 1e-3 * (hi - lo)]
    if at_bound:
        print("      WARNING: parameters at bounds:", at_bound)

    # ---- STEP 2: leave-one-temperature-out CV (training data only) ---------
    print("\n[2] Leave-one-temperature-out cross-validation (training data only) ...", flush=True)
    n = len(temp_df)
    Yt = temp_df[CSV_COLS].to_numpy(float)
    th_jk, loo_pred = [], np.zeros_like(Yt)
    for k in range(n):
        tk, _ = calibrate(temp_df, cfg, rows=[i for i in range(n) if i != k], n_starts=2)
        th_jk.append(tk)
        loo_pred[k] = exit_composition(tk, temp_df["initial_temp_C"].iloc[k] + 273.15, cfg)
    fit_pred = np.array([exit_composition(theta, T + 273.15, cfg) for T in temp_df["initial_temp_C"]])
    m_fit, m_loo = metrics(fit_pred, Yt), metrics(loo_pred, Yt)
    print("    In-sample (training) error, vol.% :"); print(m_fit.round(3).to_string())
    print("    Leave-one-out CV error, vol.%     :"); print(m_loo.round(3).to_string())

    # ---- STEP 3: held-out depth profile ------------------------------------
    if not dpath.exists():
        print(f"\n(depth CSV {dpath} not found -> held-out test skipped)"); return 0
    depth_df = pd.read_csv(dpath)
    _check_df(depth_df, ["bed_depth_m"] + CSV_COLS, "depth CSV")
    zb = depth_df["bed_depth_m"].to_numpy(float)
    Yd = depth_df[CSV_COLS].to_numpy(float)
    pred_d = simulate(theta, 1273.15, zb, cfg).journal
    m_test = metrics(pred_d, Yd)
    print("\n[3] HELD-OUT TEST: bed-depth profile at 1000 C (never used in fitting), vol.% :")
    print(m_test.round(3).to_string())
    zg = np.linspace(0, cfg.bed_depth, 201)
    jk_preds = np.array([simulate(t, 1273.15, zg, cfg).journal for t in th_jk])
    jk_depth = (zg, jk_preds.min(0), jk_preds.max(0))

    # ---- STEP 4: independent published scalars -----------------------------
    cal = calibrate_indicator_constants()
    y1000 = exit_composition(theta, 1273.15, cfg)
    y650 = exit_composition(theta, 923.15, cfg)
    ind = performance_indicators(y1000, cal)
    print("\n[4] Check against published scalars (paper text / Table 6; not used in fitting)")
    rows = []
    for j, sp in enumerate(REPORTED):
        rows.append((f"{sp} @1000C vs Table 6", y1000[j], PAPER["T1000"][sp]))
    for sp in ("CO", "H2", "CH4"):
        rows.append((f"{sp} @0.5 m vs Sec 4.5 text", y1000[REPORTED.index(sp)], PAPER["depth_exit"][sp]))
    for key in ("LHV", "CGE", "CCE"):
        rows.append((f"{key} @1000C vs Table 6", ind[key], PAPER["T1000"][key]))
    ind650 = performance_indicators(y650, cal)
    rows.append(("LHV @650C vs Sec 4.4", ind650["LHV"], 3.82))
    tab = pd.DataFrame(rows, columns=["quantity", "model", "paper"])
    tab["abs_err"] = tab["model"] - tab["paper"]
    tab["rel_err_%"] = 100 * tab["abs_err"] / tab["paper"]
    print(tab.round(2).to_string(index=False))

    # ---- STEP 5: floor imposed by the journal's own inconsistency ----------
    end_sweep = temp_df.loc[temp_df.initial_temp_C == 1000, CSV_COLS].to_numpy(float)[0]
    end_depth = Yd[-1]
    print("\n[5] Journal self-consistency: Fig. 6 @1000 C vs Fig. 9 @0.5 m describe the SAME case")
    print("    Fig6 (CO,CO2,H2,CH4):", np.round(end_sweep, 2), " Fig9:", np.round(end_depth, 2))
    print(f"    RMS disagreement = {np.sqrt(np.mean((end_sweep - end_depth) ** 2)):.2f} vol.%  "
          "-> no single model can match both exactly")

    # ---- Output files ------------------------------------------------------
    zf = np.linspace(0, cfg.bed_depth, 101)
    pf = simulate(theta, 1273.15, zf, cfg)
    out = pd.DataFrame({"bed_depth_m": zf, "T_K": pf.T})
    for j, sp in enumerate(REPORTED):
        out[f"{sp}_journal_basis_pct"] = pf.journal[:, j]
        out[f"{sp}_dry_normalised_pct"] = pf.dry_norm[:, j]
    out["H2O_wet_mole_frac_pct"] = pf.F[I_H2O] / pf.F.sum(0) * 100
    out.to_csv(outdir / "simulated_profile_1000C.csv", index=False)
    pd.concat({"train_in_sample": m_fit, "train_LOO_CV": m_loo, "heldout_depth": m_test}).to_csv(
        outdir / "validation_metrics.csv")
    tab.to_csv(outdir / "published_scalar_checks.csv", index=False)

    if not args.no_plots:
        make_plots(theta, temp_df, depth_df, jk_depth, loo_pred, cfg, outdir, args.show)
        print(f"\nSaved figures + CSVs in {outdir}/")

    # ---- Optional: joint refit (explicitly in-sample) -----------------------
    if args.joint:
        print("\n[JOINT] Refit on temperature sweep + depth profile (IN-SAMPLE, no held-out data left)")
        T_K = temp_df["initial_temp_C"].to_numpy(float) + 273.15
        w = np.sqrt(len(Yt) / len(Yd))     # give both datasets equal total weight

        def resid(th):
            r = [exit_composition(th, T, cfg) - y for T, y in zip(T_K, Yt)]
            r.append((simulate(th, 1273.15, zb, cfg).journal - Yd).ravel() * w)
            return np.concatenate([np.ravel(a) for a in r])
        rj = least_squares(resid, theta, bounds=(LB, UB), x_scale=np.array([1, .1, 1, 1, 50.]), diff_step=1e-4)
        thj = rj.x
        mt = metrics(np.array([exit_composition(thj, T, cfg) for T in T_K]), Yt)
        md = metrics(simulate(thj, 1273.15, zb, cfg).journal, Yd)
        print("    sweep RMSE %.3f | depth RMSE %.3f (both in-sample)" % (mt.loc["ALL", "RMSE"], md.loc["ALL", "RMSE"]))
        (outdir / "calibrated_parameters_joint.json").write_text(
            json.dumps({"IN_SAMPLE_ONLY": True, "parameters": dict(zip(PARAM_NAMES, map(float, thj)))}, indent=2))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=".", help="folder with the two benchmark CSVs")
    ap.add_argument("--outdir", default=".", help="where to write figures/CSVs")
    ap.add_argument("--starts", type=int, default=4, help="multi-start count for calibration")
    ap.add_argument("--joint", action="store_true", help="additionally refit on both datasets (in-sample)")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--show", action="store_true", help="open plot windows")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(0 if selftest() else 1)
    sys.exit(run(args))


if __name__ == "__main__":
    main()
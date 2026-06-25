#!/usr/bin/env python3
"""Plot CR3BP trajectory correction: reference vs uncorrected (perturbed) vs
thrml-corrected, using the actual maneuver vector the thrml solver finds."""
import os
os.environ.setdefault("JAX_PLATFORMS", "cuda")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from cr3bp_ising import (
    rhs_planar, load_jpl_orbit_row, build_linearized_model, build_qubo,
    decode_u_from_bits, objective_batch, objective_continuous,
)
from cr3bp_thrml import bqm_to_spin_float, ThrmlIsingSolver, auto_beta_range

JSON, ROW, K, BITS = "em_L1_lyapunov.json", 1593, 16, 8
mu, units, orbit = load_jpl_orbit_row(JSON, ROW)
s_ref0 = np.array([orbit["x"], orbit["y"], orbit["vx"], orbit["vy"]], dtype=float)
T = float(orbit["period"])
dx = 1e-6
delta0 = np.array([dx, 0.0, 0.0, 0.0])

model = build_linearized_model(mu, s_ref0, T, delta0, K, "period_scaled", 1e-2, 1e-11, 1e-13)
bqm, enc = build_qubo(model, K, BITS, 2e-3)

# --- solve with thrml (GPU) and take the best maneuver vector ---
h, J, var_order = bqm_to_spin_float(bqm, negate_for_thrml=True)
bs, be = auto_beta_range(h, J)
solver = ThrmlIsingSolver(h, J, iters=20000, beta_start=bs, beta_end=be)
Z = solver.solve_batch(num_reads=2000, seed=7)
perm = np.array([var_order.index(n) for n in enc.names], dtype=int)
U = decode_u_from_bits(Z[:, perm].astype(np.float64), enc)
objs = objective_batch(U, model)
ibest = int(np.argmin(objs))
u_best = U[ibest]
J_base = objective_continuous(np.zeros(2 * K), model)
print(f"best J = {objs[ibest]:.4e}  (baseline {J_base:.4e})")

tks = (np.arange(1, K + 1, dtype=float) * T) / (K + 1)
dv = [(u_best[2 * k], u_best[2 * k + 1]) for k in range(K)]


def propagate(s0, dv_per_tk):
    seg = [0.0] + list(tks) + [T]
    ts, xs, ys = [], [], []
    s = s0.astype(float).copy()
    for i in range(len(seg) - 1):
        sol = solve_ivp(lambda t, y: rhs_planar(t, y, mu), (seg[i], seg[i + 1]), s,
                        rtol=1e-11, atol=1e-13, dense_output=True, method="DOP853")
        tt = np.linspace(seg[i], seg[i + 1], 300)
        yy = sol.sol(tt)
        ts.append(tt); xs.append(yy[0]); ys.append(yy[1])
        s = sol.y[:, -1].copy()
        if i < len(tks) and dv_per_tk is not None:
            s[2] += dv_per_tk[i][0]; s[3] += dv_per_tk[i][1]
    return np.concatenate(ts), np.concatenate(xs), np.concatenate(ys)


t_ref, x_ref, y_ref = propagate(s_ref0, None)
_, x_unc, y_unc = propagate(s_ref0 + delta0, None)
_, x_cor, y_cor = propagate(s_ref0 + delta0, dv)

dev_unc = np.hypot(x_unc - x_ref, y_unc - y_ref) * units.lunit_km
dev_cor = np.hypot(x_cor - x_ref, y_cor - y_ref) * units.lunit_km
tdays = t_ref * units.tunit_s / 86400.0
dvtot_ms = sum(np.hypot(a, b) for a, b in dv) * units.vunit_m_s
print(f"terminal dev: uncorrected {dev_unc[-1]:.1f} km -> corrected {dev_cor[-1]:.1f} km")
print(f"total dv = {dvtot_ms:.3f} m/s")

# ---------------- figure ----------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

ax1.plot(x_ref, y_ref, "k-", lw=1.6, label="Reference L1 Lyapunov orbit")
ax1.plot(x_unc, y_unc, "r--", lw=1.4, label="Uncorrected (perturbed)")
ax1.plot(x_cor, y_cor, "g-", lw=1.2, label="thrml-corrected")
ax1.plot([1 - mu], [0], "o", color="0.45", ms=9, label="Moon")
ax1.plot(x_cor[0], y_cor[0], "k.", ms=10)
mt = np.searchsorted(t_ref, tks)
mt = np.clip(mt, 0, len(x_cor) - 1)
ax1.plot(x_cor[mt], y_cor[mt], "g^", ms=6, label="maneuvers (K=16)")
ax1.set_xlabel("x  [rotating frame, nondim]"); ax1.set_ylabel("y")
ax1.set_aspect("equal", adjustable="datalim")
ax1.legend(fontsize=8, loc="best"); ax1.set_title("Planar CR3BP trajectory (Earth–Moon, L1)")
ax1.grid(alpha=0.3)

ax2.semilogy(tdays, np.maximum(dev_unc, 1e-3), "r--", lw=1.6, label="Uncorrected")
ax2.semilogy(tdays, np.maximum(dev_cor, 1e-3), "g-", lw=1.6, label="thrml-corrected")
for tk in tks:
    ax2.axvline(tk * units.tunit_s / 86400.0, color="g", alpha=0.15, lw=1)
ax2.set_xlabel("time  [days]"); ax2.set_ylabel("position deviation from reference  [km]")
ax2.set_title("Terminal-error correction")
ax2.legend(fontsize=9, loc="best"); ax2.grid(alpha=0.3, which="both")
ax2.annotate(f"uncorrected: {dev_unc[-1]:.0f} km\ncorrected: {dev_cor[-1]:.1f} km\n"
             f"$\\Delta v$ = {dvtot_ms:.2f} m/s",
             xy=(0.03, 0.97), xycoords="axes fraction", va="top", fontsize=9,
             bbox=dict(boxstyle="round", fc="w", alpha=0.8))

fig.suptitle("CR3BP Trajectory-Correction Maneuver — solved as an Ising/QUBO problem (N=256 spins) with thrml",
             fontsize=11, y=1.0)
fig.tight_layout()
fig.savefig("traj_correction.png", dpi=150, bbox_inches="tight")
print("wrote traj_correction.png")

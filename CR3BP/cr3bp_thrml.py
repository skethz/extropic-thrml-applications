#!/usr/bin/env python3
"""
CR3BP benchmark (planar L1 Lyapunov) evaluated with Extropic's `thrml` on CPU --
a FAITHFUL replacement of the GPU Ising kernel: same architecture, same
parameters, only the per-site update is swapped from the hand-coded greedy flip
to thrml's `SpinGibbsConditional` heat-bath.

Architecture (mirrors ising_int.cu's ising_int_kernel)
- `num_reads` independent chains, each initialized to all-(-1) spins.
- Each chain runs `iters` single-site updates: pick a random site j, recompute
  its local field, and resample s_j. The GPU kernel does a zero-temperature
  GREEDY flip; here s_j is drawn by thrml's heat-bath conditional
  P(s_j=+1) = sigmoid(2*beta*L_j), with beta annealed hot->cold across the
  `iters` steps (so the cold tail behaves like the kernel's greedy descent).
- Implemented as `jax.lax.scan` over steps, `jax.vmap` over chains -- the scan
  body is a single site update, so it compiles fast even at large N (unlike a
  block-Gibbs sweep that unrolls all N sites).

This is the same thrml-native single-site incantation validated in the K2048
Max-Cut RSA project (thrml_rsa/thrml_update.py): SpinGibbsConditional computes
gamma = beta * L_j and draws bernoulli(sigmoid(2*gamma)), so the conditional is
computed by thrml, not hand-rolled.

Energy convention (verified against thrml/models/ising.py / discrete_ebm.py)
- thrml's heat-bath drives toward MINIMIZING -(sum b s + sum J s s), exactly what
  the GPU kernel minimizes. The GPU harness maps the standard dimod objective by
  negating (b=-h_dimod, J=-J_dimod); we apply the same negation (floats, no int
  quantization).
- The per-node bias h is injected via a CLAMPED ghost spin fixed at +1: an extra
  column J_aug[j, ghost] = h[j] makes the field sum include h[j]. The ghost is
  never selected for update, so it stays +1.

Run (same parameters as the GPU kernel run)
  source /scratch/hongse/venvs/thrml/bin/activate
  JAX_PLATFORMS=cpu python3 -u cr3bp_thrml.py --json em_L1_lyapunov.json --row 1593 \
    --K 16 --bits 8 --iters 20000 \
    --repeats 10 --num-reads 2000 --num-srt 8 \
    --target-mode frac --target-frac 0.1
"""

from __future__ import annotations

import argparse
import math
import os
import time
from typing import List, Tuple

# Keep thrml/JAX on CPU unless the caller overrode it explicitly.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

import dimod
import jax
import jax.numpy as jnp
from thrml.models.discrete_ebm import DiscreteEBMInteraction, SpinGibbsConditional

# Reuse the verified CR3BP / QUBO / TTS machinery unchanged.
from cr3bp_ising import (
    build_linearized_model,
    build_qubo,
    decode_u_from_bits,
    estimate_tts,
    load_jpl_orbit_row,
    objective_batch,
    objective_continuous,
    split_errors,
)


# -----------------------------
# thrml-native single-site heat-bath update
# (lifted from thrml_rsa/thrml_update.py -- the validated K2048 incantation)
# -----------------------------
_SAMP = SpinGibbsConditional()
_OUT_SD = jax.ShapeDtypeStruct((1,), jnp.bool_)


def gibbs_update_site(spins_bool, w_row, key):
    """thrml heat-bath for one site -> new bool spin (True=+1).

    spins_bool: (Naug,) bool. w_row: (Naug,) float = beta*J_aug[j,:] for site j.
    thrml's SpinGibbsConditional computes gamma = sum_i w_row[i]*s_i = beta*L_j and
    draws bernoulli(sigmoid(2*gamma)). Requires w_row[j] == 0 (zero diagonal)."""
    interaction = DiscreteEBMInteraction(n_spin=1, weights=w_row[None, :])
    active = jnp.ones((1, w_row.shape[0]), dtype=bool)
    states = [[spins_bool[None, :]]]
    new_val, _ = _SAMP.sample(key, [interaction], [active], states, None, _OUT_SD)
    return new_val[0]


# -----------------------------
# BQM -> float SPIN Ising coefficients for thrml
# (same negation as bqm_to_kernel_int_matrices, but float, no quantization)
# -----------------------------
def bqm_to_spin_float(
    bqm_binary: dimod.BinaryQuadraticModel,
    negate_for_thrml: bool,
) -> Tuple[np.ndarray, np.ndarray, List]:
    """Convert a BINARY BQM to a float SPIN Ising model (h, J, var_order)."""
    bqm_spin = bqm_binary.change_vartype(dimod.SPIN, inplace=False)

    var_order = list(bqm_spin.variables)
    idx = {v: i for i, v in enumerate(var_order)}
    N = len(var_order)

    h = np.zeros(N, dtype=np.float64)
    for v, bias in bqm_spin.linear.items():
        h[idx[v]] = float(bias)

    J = np.zeros((N, N), dtype=np.float64)
    for (u, v), bias in bqm_spin.quadratic.items():
        i, j = idx[u], idx[v]
        J[i, j] = float(bias)
        J[j, i] = float(bias)

    if negate_for_thrml:
        h = -h
        J = -J

    np.fill_diagonal(J, 0.0)
    return h, J, var_order


# -----------------------------
# thrml single-site solver (kernel-faithful architecture)
# -----------------------------
class ThrmlIsingSolver:
    """Mirrors the GPU kernel: `num_reads` chains, each `iters` single-site
    updates from an all-(-1) start, with thrml's heat-bath as the update rule
    and beta annealed hot->cold across the steps. solve_batch returns uint8 spins
    (num_reads, N) in var_order (node) order, 1 == spin up."""

    def __init__(self, h: np.ndarray, J: np.ndarray, iters: int,
                 beta_start: float, beta_end: float):
        self.N = int(h.shape[0])           # real spins
        self.iters = int(iters)

        # Augment with a ghost spin (index N, clamped +1) carrying the biases h.
        Naug = self.N + 1
        Jaug = np.zeros((Naug, Naug), dtype=np.float32)
        Jaug[: self.N, : self.N] = J.astype(np.float32)
        Jaug[: self.N, self.N] = h.astype(np.float32)
        Jaug[self.N, : self.N] = h.astype(np.float32)
        np.fill_diagonal(Jaug, 0.0)
        self.Jaug = jnp.asarray(Jaug)

        # all-(-1) real spins; ghost = +1.  (matches the kernel's init)
        init = np.zeros(Naug, dtype=bool)
        init[self.N] = True
        self.init_s_bool = jnp.asarray(init)

        # beta annealing schedule across the `iters` single-site steps.
        self.betas = jnp.asarray(
            np.geomspace(beta_start, beta_end, self.iters).astype(np.float32)
        )

        self._chain = jax.jit(jax.vmap(self._run_one, in_axes=(None, None, 0)))

    def _run_one(self, init_s_bool, betas, key):
        N = self.N
        Jaug = self.Jaug
        keys = jax.random.split(key, betas.shape[0])

        def step(s_bool, inp):
            beta_t, k = inp
            kj, ks = jax.random.split(k)
            j = jax.random.randint(kj, (), 0, N)        # real sites only
            new_b = gibbs_update_site(s_bool, beta_t * Jaug[j], ks)
            s_bool = s_bool.at[j].set(new_b)
            return s_bool, None

        s_bool, _ = jax.lax.scan(step, init_s_bool, (betas, keys))
        return s_bool

    def solve_batch(self, num_reads: int, seed: int) -> np.ndarray:
        keys = jax.random.split(jax.random.key(seed), num_reads)
        final = self._chain(self.init_s_bool, self.betas, keys)  # (num_reads, Naug)
        out = np.asarray(final[:, : self.N], dtype=bool).astype(np.uint8)
        return out


# -----------------------------
# Auto beta range from coupling magnitudes
# -----------------------------
def auto_beta_range(h: np.ndarray, J: np.ndarray) -> Tuple[float, float]:
    """beta_start/beta_end from the typical single-spin local-field magnitude
    f_i = sqrt(h_i^2 + sum_j J_ij^2).

    "warmstart-cold" schedule: start already moderately cold (2*beta*f ~ 16) so
    the chain descends greedily from the first steps (mirroring the kernel's
    zero-temperature greedy flip) instead of wandering hot, then quench deeply
    (2*beta*f ~ 600). Tuned on the N=256 CR3BP instance, where this beats a
    hot-start anneal by ~4x in best objective and is the difference between
    p_attempt 0 and 1 at fixed iters."""
    f = np.sqrt(h**2 + np.sum(J**2, axis=1))
    f_typ = float(np.median(f[f > 0])) if np.any(f > 0) else 1.0
    if f_typ <= 0.0:
        f_typ = 1.0
    return 8.0 / f_typ, 300.0 / f_typ


# -----------------------------
# Main: run attempts and compute TTS (mirrors cr3bp_ising.main)
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="JPL periodic_orbits JSON")
    ap.add_argument("--row", type=int, default=1593)

    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--bits", type=int, default=6)
    ap.add_argument("--vmax", type=float, default=2e-3)
    ap.add_argument("--lam", type=float, default=1e-2)
    ap.add_argument("--W", default="period_scaled", choices=["identity", "period_scaled", "vel_down_0p1"])
    ap.add_argument("--dx", type=float, default=1e-6)
    ap.add_argument("--dy", type=float, default=0.0)
    ap.add_argument("--dvx", type=float, default=0.0)
    ap.add_argument("--dvy", type=float, default=0.0)
    ap.add_argument("--rtol", type=float, default=1e-11)
    ap.add_argument("--atol", type=float, default=1e-13)

    # Same control knob as the GPU kernel: single-site updates per chain.
    ap.add_argument("--iters", type=int, default=20000, help="Single-site updates per chain")
    ap.add_argument("--beta-start", type=float, default=None, help="Hot inverse temp; auto if omitted")
    ap.add_argument("--beta-end", type=float, default=None, help="Cold inverse temp; auto if omitted")
    ap.add_argument("--no-negate", action="store_true", help="Do NOT negate h,J before thrml (debug).")

    ap.add_argument("--num-reads", type=int, default=2000, help="Parallel chains per solve")
    ap.add_argument("--num-srt", type=int, default=8, help="Solver calls per attempt; time sums.")
    ap.add_argument("--repeats", type=int, default=50, help="Number of attempts")
    ap.add_argument("--confidence", type=float, default=0.99)

    ap.add_argument("--target-mode", default="frac", choices=["frac", "absolute"])
    ap.add_argument("--target-frac", type=float, default=0.1)
    ap.add_argument("--target-abs", type=float, default=1e-6)

    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    print(f"[thrml] devices={jax.devices()}")

    mu, units, orbit = load_jpl_orbit_row(args.json, args.row)
    s_ref0 = np.array([orbit["x"], orbit["y"], orbit["vx"], orbit["vy"]], dtype=float)
    T = float(orbit["period"])
    delta0 = np.array([args.dx, args.dy, args.dvx, args.dvy], dtype=float)

    print("\n=== Reference orbit ===")
    print(f"row_index: {args.row}")
    print(f"mu: {mu:.16e}")
    print(f"T: {T:.15f} TU  ~ {T*units.tunit_s/86400.0:.3f} days")

    model = build_linearized_model(
        mu=mu, s_ref0=s_ref0, T=T, delta0=delta0,
        K=args.K, W_mode=args.W, lam=args.lam, rtol=args.rtol, atol=args.atol,
    )

    bqm, enc = build_qubo(model, args.K, args.bits, args.vmax)
    nb = len(bqm.variables)

    u0 = np.zeros(2 * args.K, dtype=float)
    J_baseline = objective_continuous(u0, model)
    if args.target_mode == "frac":
        J_target = args.target_frac * J_baseline
    else:
        J_target = args.target_abs

    err0 = -model.b
    pos0_km, vel0_m_s = split_errors(err0, units)

    print("\n=== Baseline (linearized, u=0) ===")
    print(f"pred terminal pos err: {pos0_km:.6f} km")
    print(f"pred terminal vel err: {vel0_m_s:.6f} m/s")
    print(f"J_baseline: {J_baseline:.12e}")
    print(f"J_target:   {J_target:.12e}")

    h, J, var_order = bqm_to_spin_float(bqm, negate_for_thrml=(not args.no_negate))

    beta_lo_auto, beta_hi_auto = auto_beta_range(h, J)
    beta_start = args.beta_start if args.beta_start is not None else beta_lo_auto
    beta_end = args.beta_end if args.beta_end is not None else beta_hi_auto

    print("\n=== thrml Ising encoding ===")
    print(f"N (variables): {nb}")
    print(f"max |J|: {float(np.max(np.abs(J))):.6e}")
    print(f"max |h|: {float(np.max(np.abs(h))):.6e}")
    print(f"iters (single-site updates/chain): {args.iters}")
    print(f"beta anneal: {beta_start:.4e} -> {beta_end:.4e}")

    if set(var_order) != set(enc.names):
        raise RuntimeError("Variable-label mismatch between thrml encoding and QUBO encoding.")
    perm = np.array([var_order.index(name) for name in enc.names], dtype=int)

    solver = ThrmlIsingSolver(
        h=h, J=J, iters=args.iters, beta_start=beta_start, beta_end=beta_end
    )

    print("\n=== Attempts ===")
    print(f"repeats: {args.repeats}")
    print(f"num_reads (chains per call): {args.num_reads}")
    print(f"num_srt (calls per attempt): {args.num_srt}")
    print(f"iters per call: {args.iters}")

    rng = np.random.default_rng(args.seed)

    successes: List[bool] = []
    times_s: List[float] = []
    best_objs: List[float] = []

    # Warm up JIT once (compile cost excluded from timed attempts).
    _ = solver.solve_batch(num_reads=min(args.num_reads, 8), seed=int(rng.integers(0, 2**31)))

    for r in range(args.repeats):
        t0 = time.perf_counter()
        best_obj_attempt = float("inf")

        for _k in range(max(1, args.num_srt)):
            seed_k = int(rng.integers(0, 2**63 - 1))
            Z = solver.solve_batch(num_reads=args.num_reads, seed=seed_k)  # (B,N) 0/1, node order
            Zp = Z[:, perm]
            U = decode_u_from_bits(Zp.astype(np.float64), enc)
            objs = objective_batch(U, model)
            m = float(np.min(objs))
            if m < best_obj_attempt:
                best_obj_attempt = m

        dt = time.perf_counter() - t0

        ok = best_obj_attempt <= J_target
        successes.append(ok)
        times_s.append(dt)
        best_objs.append(best_obj_attempt)

        if not args.quiet:
            print(f"  [{r+1:03d}/{args.repeats}] best J={best_obj_attempt:.3e} | success={ok} | time_s={dt:.6f}")

    res = estimate_tts(successes, times_s, args.confidence)

    print("\n=== thrml (CPU) Ising Performance Summary ===")
    print(f"p_attempt: {res['p_attempt']:.4f}")
    print(f"median time per attempt (s): {res['t_per_attempt_s']:.6f}")
    print(f"TTS@{args.confidence:.2f} (s): {res['tts_s']:.6f}")
    print(f"best objective (min over attempts): {float(np.min(best_objs)):.12e}")
    print(f"median objective (over attempts):  {float(np.median(best_objs)):.12e}")
    print(f"J_baseline: {J_baseline:.12e}")
    print(f"J_target:   {J_target:.12e}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
CR3BP benchmark (planar L1 Lyapunov) evaluated with YOUR CUDA Ising kernel.

Goal
- Use the same CR3BP-derived binary QUBO benchmark instance as the D-Wave QPU code.
- Replace the D-Wave sampler with your GPU Ising solver (libising_int.so).
- Compute TTS (time-to-solution) with the same success criterion: min J(u) <= J_target.

Your kernel interface (from ising_int.cu)
- init_gpu(N, max_batch_size)
- upload_J(J_host_int32 NxN)
- solve_batch(h_host_int32 (batch_size*N), s_host_uint8 (batch_size*N), batch_size, iters, seed)

Important convention note
Your kernel greedily aligns spins with local field L = h + sum(J*s), which corresponds to minimizing
E_kernel(s) = - sum_i h_i s_i - sum_{i<j} J_ij s_i s_j.
Standard Ising energy (dimod) is E(s)= sum_i h_i s_i + sum_{i<j} J_ij s_i s_j.
=> To make your kernel minimize the *standard* Ising objective, we negate coefficients:
   h_kernel = -h_dimod, J_kernel = -J_dimod

Install
  pip install numpy scipy dimod

Build the shared library (adjust arch if needed)
  nvcc -O3 --shared -Xcompiler -fPIC -arch=sm_90 ising_int.cu -o libising_int.so

Run (example)
  python3 cr3bp_ising.py \
    --json em_L1_lyapunov.json --row 1593 \
    --lib ./libising_int.so --iters 5000 \
    --repeats 50 --num-reads 2000 --num-srt 8 \
    --target-mode frac --target-frac 0.1
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.linalg import inv
from scipy.integrate import solve_ivp

import dimod


# -----------------------------
# CR3BP planar dynamics + STM
# -----------------------------
def rhs_planar(t: float, s: np.ndarray, mu: float) -> np.ndarray:
    x, y, vx, vy = s
    r1 = math.sqrt((x + mu) ** 2 + y**2)
    r2 = math.sqrt((x - (1.0 - mu)) ** 2 + y**2)
    ax = 2.0 * vy + x - (1.0 - mu) * (x + mu) / (r1**3) - mu * (x - (1.0 - mu)) / (r2**3)
    ay = -2.0 * vx + y - (1.0 - mu) * y / (r1**3) - mu * y / (r2**3)
    return np.array([vx, vy, ax, ay], dtype=float)


def jac_planar(s: np.ndarray, mu: float) -> np.ndarray:
    x, y, vx, vy = s
    r1sq = (x + mu) ** 2 + y**2
    r2sq = (x - (1.0 - mu)) ** 2 + y**2
    r1 = math.sqrt(r1sq)
    r2 = math.sqrt(r2sq)

    d1 = 1.0 - mu
    d2 = mu

    r1_3 = r1sq * r1
    r2_3 = r2sq * r2
    r1_5 = r1_3 * r1sq
    r2_5 = r2_3 * r2sq

    Uxx = 1.0 - d1 * (1.0 / r1_3 - 3.0 * (x + mu) ** 2 / r1_5) - d2 * (
        1.0 / r2_3 - 3.0 * (x - (1.0 - mu)) ** 2 / r2_5
    )
    Uyy = 1.0 - d1 * (1.0 / r1_3 - 3.0 * y**2 / r1_5) - d2 * (1.0 / r2_3 - 3.0 * y**2 / r2_5)
    Uxy = 3.0 * y * (d1 * (x + mu) / r1_5 + d2 * (x - (1.0 - mu)) / r2_5)

    J = np.zeros((4, 4), dtype=float)
    J[0, 2] = 1.0
    J[1, 3] = 1.0
    J[2, 0] = Uxx
    J[2, 1] = Uxy
    J[2, 3] = 2.0
    J[3, 0] = Uxy
    J[3, 1] = Uyy
    J[3, 2] = -2.0
    return J


def rhs_state_and_stm(t: float, y: np.ndarray, mu: float) -> np.ndarray:
    s = y[:4]
    Phi = y[4:].reshape(4, 4)
    ds = rhs_planar(t, s, mu)
    dPhi = jac_planar(s, mu) @ Phi
    out = np.zeros_like(y)
    out[:4] = ds
    out[4:] = dPhi.reshape(-1)
    return out


# -----------------------------
# Units and errors (only for printout)
# -----------------------------
@dataclass(frozen=True)
class Units:
    lunit_km: float
    tunit_s: float

    @property
    def vunit_m_s(self) -> float:
        return (self.lunit_km / self.tunit_s) * 1000.0


def split_errors(err: np.ndarray, units: Units) -> Tuple[float, float]:
    pos_km = float(np.hypot(err[0], err[1]) * units.lunit_km)
    vel_m_s = float(np.hypot(err[2], err[3]) * units.vunit_m_s)
    return pos_km, vel_m_s


# -----------------------------
# Load JPL periodic orbit row
# -----------------------------
def load_jpl_orbit_row(json_path: str, row_index: int) -> Tuple[float, Units, Dict[str, float]]:
    with open(json_path, "r") as f:
        obj = json.load(f)
    fields = obj["fields"]
    idx = {name: i for i, name in enumerate(fields)}
    row = obj["data"][row_index]
    orbit = {k: float(row[idx[k]]) for k in fields}
    mu = float(obj["system"]["mass_ratio"])
    units = Units(lunit_km=float(obj["system"]["lunit"]), tunit_s=float(obj["system"]["tunit"]))
    return mu, units, orbit


# -----------------------------
# Linearized targeting model
# -----------------------------
@dataclass
class LinearizedModel:
    A: np.ndarray          # (4, 2K)
    b: np.ndarray          # (4,)
    W: np.ndarray          # (4,4)
    lam: float
    T: float


def build_linearized_model(
    mu: float,
    s_ref0: np.ndarray,
    T: float,
    delta0: np.ndarray,
    K: int,
    W_mode: str,
    lam: float,
    rtol: float,
    atol: float,
) -> LinearizedModel:
    if K < 1:
        raise ValueError("K must be >= 1")

    tks = (np.arange(1, K + 1, dtype=float) * T) / (K + 1)
    times = np.unique(np.concatenate(([0.0], tks, [T])))

    Phi0 = np.eye(4)
    y0_full = np.concatenate([s_ref0, Phi0.reshape(-1)])

    sol = solve_ivp(
        fun=lambda t, y: rhs_state_and_stm(t, y, mu),
        t_span=(0.0, T),
        y0=y0_full,
        t_eval=times,
        rtol=rtol,
        atol=atol,
        method="DOP853",
    )
    if not sol.success:
        raise RuntimeError(f"STM integration failed: {sol.message}")

    Phi_at = {float(t): sol.y[4:, i].reshape(4, 4).copy() for i, t in enumerate(sol.t)}
    Phi_T0 = Phi_at[float(T)]

    Bmat = np.array([[0, 0], [0, 0], [1, 0], [0, 1]], dtype=float)
    b = -(Phi_T0 @ delta0)

    blocks = []
    for tk in tks:
        Phi_tk0 = Phi_at[float(tk)]
        Phi_Ttk = Phi_T0 @ inv(Phi_tk0)
        blocks.append(Phi_Ttk @ Bmat)

    A = np.hstack(blocks)

    if W_mode == "identity":
        W = np.eye(4)
    elif W_mode == "period_scaled":
        W = np.diag([1.0, 1.0, T * T, T * T])
    elif W_mode == "vel_down_0p1":
        W = np.diag([1.0, 1.0, 0.01, 0.01])
    else:
        raise ValueError("W_mode must be: identity | period_scaled | vel_down_0p1")

    return LinearizedModel(A=A, b=b, W=W, lam=lam, T=T)


def objective_continuous(u: np.ndarray, model: LinearizedModel) -> float:
    err = model.A @ u - model.b
    return float(err.T @ model.W @ err + model.lam * (u @ u))


# -----------------------------
# QUBO encoding (same as QPU code)
# -----------------------------
@dataclass
class QuboEncoding:
    names: List[str]
    u0: np.ndarray     # (m,)
    C: np.ndarray      # (m, nb)
    step: float
    K: int
    bits: int


def build_qubo(model: LinearizedModel, K: int, bits: int, vmax: float) -> Tuple[dimod.BinaryQuadraticModel, QuboEncoding]:
    A, b, W, lam = model.A, model.b, model.W, model.lam
    m = 2 * K
    H = A.T @ W @ A + lam * np.eye(m)
    g = -2.0 * (A.T @ W @ b)

    step = (2.0 * vmax) / (2**bits - 1)
    u0 = -vmax * np.ones(m, dtype=float)

    nb = m * bits
    C = np.zeros((m, nb), dtype=float)
    names: List[str] = []
    col = 0
    for i in range(m):
        for j in range(bits):
            C[i, col] = step * (2**j)
            names.append(f"u{i}_b{j}")
            col += 1

    Qsym = C.T @ H @ C
    lin = (2.0 * (u0 @ H @ C)) + (g @ C)

    qubo: Dict[Tuple[str, str], float] = {}
    for p in range(nb):
        qubo[(names[p], names[p])] = float(Qsym[p, p] + lin[p])
    for p in range(nb):
        for q in range(p + 1, nb):
            val = float(2.0 * Qsym[p, q])
            if val != 0.0:
                qubo[(names[p], names[q])] = val

    bqm = dimod.BinaryQuadraticModel.from_qubo(qubo)
    return bqm, QuboEncoding(names=names, u0=u0, C=C, step=step, K=K, bits=bits)


def decode_u_from_bits(Z: np.ndarray, enc: QuboEncoding) -> np.ndarray:
    """
    Z shape: (B, nb) bits 0/1.
    returns U shape: (B, m)
    """
    # U = u0 + C z
    return enc.u0[None, :] + Z @ enc.C.T


def objective_batch(U: np.ndarray, model: LinearizedModel) -> np.ndarray:
    """
    U shape: (B, m)
    returns objective per sample: shape (B,)
    """
    # err = A u - b  -> (B,4)
    ERR = U @ model.A.T - model.b[None, :]
    # (Au-b)^T W (Au-b) efficiently:
    WERR = ERR @ model.W.T
    term = np.einsum("bi,bi->b", ERR, WERR)
    reg = model.lam * np.einsum("bi,bi->b", U, U)
    return term + reg


# -----------------------------
# BQM -> Ising matrix for your kernel
# -----------------------------
def bqm_to_kernel_int_matrices(
    bqm_binary: dimod.BinaryQuadraticModel,
    scale: Optional[float],
    negate_for_kernel: bool,
) -> Tuple[np.ndarray, np.ndarray, float, List[str]]:
    """
    Convert the logical BINARY BQM to an Ising (SPIN) model, then export:
      J_int: int32 NxN (symmetric, diag=0)
      h_int: int32 N
    Variable order matches `var_order` list.
    """
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

    if negate_for_kernel:
        h = -h
        J = -J

    max_abs = float(max(np.max(np.abs(h)), np.max(np.abs(J))))
    if max_abs == 0.0:
        max_abs = 1.0

    # Choose scale automatically if not provided: fit in int32 with headroom.
    if scale is None:
        # headroom: keep max coefficient around 1e7 to reduce overflow in local field accumulation
        # (local fields can accumulate O(N*J)).
        target = 1.0e6
        scale = target / max_abs

    h_int = np.rint(h * scale).astype(np.int32)
    J_int = np.rint(J * scale).astype(np.int32)

    # Ensure diag is zero (kernel never uses it, but keep clean)
    np.fill_diagonal(J_int, 0)
    return J_int, h_int, float(scale), var_order


# -----------------------------
# TTS math
# -----------------------------
def estimate_tts(successes: List[bool], times_s: List[float], confidence: float) -> Dict[str, float]:
    p = float(np.mean(successes)) if successes else 0.0
    t_med = float(np.median(times_s)) if times_s else float("nan")

    # Case 1: 0% Success
    if p <= 0.0:
        return {"p_attempt": 0.0, "n_attempts": float("inf"), "tts_s": float("inf"), "t_per_attempt_s": t_med}
    
    # Case 2: 100% Success (The fix)
    # If we never fail, we technically only need 1 attempt.
    if p >= 1.0:
         return {"p_attempt": 1.0, "n_attempts": 1.0, "tts_s": t_med, "t_per_attempt_s": t_med}

    # Case 3: Partial Success (Standard Formula)
    n = math.ceil(math.log(1.0 - confidence) / math.log(1.0 - p))
    return {"p_attempt": p, "n_attempts": float(n), "tts_s": float(n) * t_med, "t_per_attempt_s": t_med}

# -----------------------------
# ctypes wrapper for your library
# -----------------------------
class IsingKernelLib:
    def __init__(self, lib_path: str):
        self.lib = ctypes.CDLL(lib_path)

        self.lib.init_gpu.argtypes = [ctypes.c_int, ctypes.c_int]
        self.lib.init_gpu.restype = None

        self.lib.upload_J.argtypes = [ctypes.POINTER(ctypes.c_int32)]
        self.lib.upload_J.restype = None

        self.lib.solve_batch.argtypes = [
            ctypes.POINTER(ctypes.c_int32),     # h_host
            ctypes.POINTER(ctypes.c_uint8),     # s_host
            ctypes.c_int,                       # batch_size
            ctypes.c_int,                       # iters
            ctypes.c_uint64,                    # seed
        ]
        self.lib.solve_batch.restype = None

        self.lib.free_gpu.argtypes = []
        self.lib.free_gpu.restype = None

    def init(self, N: int, max_batch_size: int) -> None:
        self.lib.init_gpu(int(N), int(max_batch_size))

    def upload_J(self, J_int: np.ndarray) -> None:
        assert J_int.dtype == np.int32 and J_int.flags["C_CONTIGUOUS"]
        self.lib.upload_J(J_int.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))

    def solve_batch(self, h_batch_int: np.ndarray, batch_size: int, iters: int, seed: int) -> np.ndarray:
        """
        Returns s_out as uint8 array shape (batch_size, N) with bits 0/1.
        """
        assert h_batch_int.dtype == np.int32 and h_batch_int.flags["C_CONTIGUOUS"]
        N = h_batch_int.size // batch_size
        out = np.empty((batch_size, N), dtype=np.uint8, order="C")

        self.lib.solve_batch(
            h_batch_int.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            int(batch_size),
            int(iters),
            ctypes.c_uint64(seed),
        )
        return out

    def close(self) -> None:
        self.lib.free_gpu()


# -----------------------------
# Main: run attempts and compute TTS
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

    ap.add_argument("--lib", required=True, help="Path to libising_int.so")
    ap.add_argument("--iters", type=int, default=5000, help="Kernel iterations per solve")
    ap.add_argument("--scale", type=float, default=None, help="Quantization scale (float). If omitted, auto-scale.")
    ap.add_argument("--no-negate", action="store_true", help="Do NOT negate h,J before giving to kernel (debug).")

    ap.add_argument("--num-reads", type=int, default=2000, help="Batch size per kernel call")
    ap.add_argument("--num-srt", type=int, default=8, help="Kernel calls per attempt (like SRT/gauges); time sums.")
    ap.add_argument("--repeats", type=int, default=50, help="Number of attempts")
    ap.add_argument("--confidence", type=float, default=0.99)

    ap.add_argument("--target-mode", default="frac", choices=["frac", "absolute"])
    ap.add_argument("--target-frac", type=float, default=0.1)
    ap.add_argument("--target-abs", type=float, default=1e-6)

    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    mu, units, orbit = load_jpl_orbit_row(args.json, args.row)
    s_ref0 = np.array([orbit["x"], orbit["y"], orbit["vx"], orbit["vy"]], dtype=float)
    T = float(orbit["period"])
    delta0 = np.array([args.dx, args.dy, args.dvx, args.dvy], dtype=float)

    print("\n=== Reference orbit ===")
    print(f"row_index: {args.row}")
    print(f"mu: {mu:.16e}")
    print(f"T: {T:.15f} TU  ~ {T*units.tunit_s/86400.0:.3f} days")

    model = build_linearized_model(
        mu=mu,
        s_ref0=s_ref0,
        T=T,
        delta0=delta0,
        K=args.K,
        W_mode=args.W,
        lam=args.lam,
        rtol=args.rtol,
        atol=args.atol,
    )

    # Build QUBO
    bqm, enc = build_qubo(model, args.K, args.bits, args.vmax)
    nb = len(bqm.variables)

    # Baseline and target
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

    # Convert to kernel ints
    J_int, h_int, used_scale, var_order = bqm_to_kernel_int_matrices(
        bqm_binary=bqm,
        scale=args.scale,
        negate_for_kernel=(not args.no_negate),
    )

    print("\n=== Kernel Ising encoding ===")
    print(f"N (variables): {nb}")
    print(f"used_scale: {used_scale:.6e}")
    print(f"max |J_int|: {int(np.max(np.abs(J_int)))}")
    print(f"max |h_int|: {int(np.max(np.abs(h_int)))}")

    # Load library and init
    lib = IsingKernelLib(args.lib)
    max_batch = args.num_reads  # per call
    lib.init(nb, max_batch)
    lib.upload_J(np.ascontiguousarray(J_int, dtype=np.int32))

    # Prepare h_batch once (same biases for all "reads")
    h_batch = np.tile(h_int[None, :], (args.num_reads, 1)).astype(np.int32, copy=False)
    h_batch_flat = np.ascontiguousarray(h_batch.reshape(-1), dtype=np.int32)

    rng = np.random.default_rng(args.seed)

    print("\n=== Attempts ===")
    print(f"repeats: {args.repeats}")
    print(f"num_reads (batch per call): {args.num_reads}")
    print(f"num_srt (calls per attempt): {args.num_srt}")
    print(f"iters per call: {args.iters}")

    successes: List[bool] = []
    times_s: List[float] = []
    best_objs: List[float] = []

    try:
        for r in range(args.repeats):
            t0 = time.perf_counter()
            best_obj_attempt = float("inf")

            for k in range(max(1, args.num_srt)):
                seed_k = int(rng.integers(0, 2**63 - 1))
                Z = lib.solve_batch(h_batch_flat, batch_size=args.num_reads, iters=args.iters, seed=seed_k)
                # Z shape (num_reads, nb) bits 0/1 in var_order order

                # Ensure order matches encoding variable order
                # Your QUBO encoding used names in bqm; we built kernel var_order from bqm_spin variables
                # dimod preserves variable labels; order should match bqm variables order if we keep it consistent.
                # We need Z columns to match enc.names order. We'll build an index map once.
                # (Do it on first loop only)
                if r == 0 and k == 0:
                    # Map kernel var order -> enc.names
                    if set(var_order) != set(enc.names):
                        raise RuntimeError("Variable-label mismatch between kernel encoding and QUBO encoding.")
                    perm = np.array([var_order.index(name) for name in enc.names], dtype=int)
                Zp = Z[:, perm]

                U = decode_u_from_bits(Zp.astype(np.float64), enc)
                objs = objective_batch(U, model)
                m = float(np.min(objs))
                if m < best_obj_attempt:
                    best_obj_attempt = m

            t1 = time.perf_counter()
            dt = t1 - t0

            ok = best_obj_attempt <= J_target
            successes.append(ok)
            times_s.append(dt)
            best_objs.append(best_obj_attempt)

            if not args.quiet:
                print(f"  [{r+1:03d}/{args.repeats}] best J={best_obj_attempt:.3e} | success={ok} | time_s={dt:.6f}")

    finally:
        lib.close()

    res = estimate_tts(successes, times_s, args.confidence)

    print("\n=== GPU Ising Kernel Performance Summary ===")
    print(f"p_attempt: {res['p_attempt']:.4f}")
    print(f"median time per attempt (s): {res['t_per_attempt_s']:.6f}")
    print(f"TTS@{args.confidence:.2f} (s): {res['tts_s']:.6f}")
    print(f"best objective (min over attempts): {float(np.min(best_objs)):.12e}")
    print(f"median objective (over attempts):  {float(np.median(best_objs)):.12e}")
    print(f"J_baseline: {J_baseline:.12e}")
    print(f"J_target:   {J_target:.12e}")


if __name__ == "__main__":
    main()

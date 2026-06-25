"""
Bit-faithful JAX port of the CUDA kernel's RSA annealing dynamics.

Replicates the kernel's EXACT fixed-point pipeline and RNG (splitmix32, the
Q8.8 beta schedule, the piecewise-linear sigmoid LUT in Q32, integer fields and
energy) so that JAX trajectories match the kernel (and the rsa_trace.cpp tracer)
bit-for-bit. x64 is REQUIRED for the uint64 splitmix32 arithmetic.

Bit-faithful port of the RSA path in ../gh200_k2048_rsa_rwa.cu (functions
splitmix32, p_plus_from_field_q32, sigmoid_p_q32_from_zq, run_attempt_cpu RSA branch).
"""
import jax
jax.config.update("jax_enable_x64", True)
assert jax.config.jax_enable_x64, "rsa_bitfaithful requires JAX x64 (uint64 splitmix32); do not set JAX_ENABLE_X64=0"

import argparse, time, math
import numpy as np
import jax.numpy as jnp
from jax import lax
import ref_instance as R

# ---------------------------------------------------------------- constants
P_MAX    = np.uint32(0xFFFFFFFF)
Z_CLAMPQ = 2048                     # 8<<8
_U64     = lambda v: jnp.uint64(v)

SIG_P_ANCHOR = jnp.array(
    [2147483648,2673442470,3139872686,3511455636,3782994643,3969158893,
     4091274721,4169072223,4217717111,4247778736,4266221719,4277486187,
     4284347459,4288519766,4291054360,4292593129,4293526977], dtype=jnp.uint32)
SIG_P_SLOPE = jnp.array(
    [4109053,3643986,2902992,2121398,1454408,954030,607793,380038,
     234856,144086,88004,53604,32596,19802,12022,7296], dtype=jnp.uint32)

C_ADD = jnp.uint64(0x9E3779B97F4A7C15)
C_M1  = jnp.uint64(0xBF58476D1CE4E5B9)
C_M2  = jnp.uint64(0x94D049BB133111EB)
C_COIN= jnp.uint64(0xD1B54A32D192ED03)  # domain-separation salt: coin-flip draw uses the same mix but a different stream than site selection


def splitmix32(x):
    """x: uint64 -> uint32."""
    x = x + C_ADD
    z = x
    z = (z ^ (z >> jnp.uint64(30))) * C_M1
    z = (z ^ (z >> jnp.uint64(27))) * C_M2
    return (z >> jnp.uint64(32)).astype(jnp.uint32)


def to_q8_8(beta):
    """round(clamp(beta,0,255)*256), round-half-away-from-zero -> uint16."""
    b = min(max(float(beta), 0.0), 255.0)
    v = b * 256.0
    r = math.floor(v + 0.5) if v >= 0 else math.ceil(v - 0.5)
    return np.uint16(int(r))


def beta_q_schedule(stages, beta_start, beta_end):
    out = np.empty(stages, dtype=np.uint16)
    for k in range(stages):
        alpha = (k / (stages - 1)) if stages > 1 else 1.0
        beta_f = (1.0 - alpha) * beta_start + alpha * beta_end
        out[k] = to_q8_8(beta_f)
    return jnp.array(out)  # uint16 (stages,)


def sigmoid_p_q32_from_zq(zq):
    """zq: int32 (already >=0). returns uint32."""
    idx = (zq // 128)                       # 0..15 (zq<2048 ensured by caller path)
    rez = zq - idx * 128                    # 0..127
    # SIG_P_ANCHOR has 17 entries; entry [16] is the zq>=Z_CLAMPQ boundary value returned by the clamp branch, never reached via interpolation (idx clipped to 0..15)
    idx_c = jnp.clip(idx, 0, 15)
    anchor = SIG_P_ANCHOR[idx_c]
    slope  = SIG_P_SLOPE[idx_c]
    lin = anchor + slope * rez.astype(jnp.uint32)
    # boundary cases
    lin = jnp.where(zq <= 0, jnp.uint32(2147483648), lin)
    lin = jnp.where(zq >= Z_CLAMPQ, jnp.uint32(4293526977), lin)
    return lin


def p_plus_from_field_q32(Li, beta_q):
    """Li: int32 field. beta_q: uint16 (scalar). returns uint32."""
    absL = jnp.abs(Li).astype(jnp.uint32)
    zq = (beta_q.astype(jnp.uint32) << jnp.uint32(1)) * absL   # uint32
    zq = jnp.minimum(zq, jnp.uint32(Z_CLAMPQ))
    p = sigmoid_p_q32_from_zq(zq.astype(jnp.int32))
    p = jnp.where(Li < 0, P_MAX - p, p)
    p = jnp.where(Li == 0, jnp.uint32(2147483648), p)
    return p


def _run_one(J, spin0, spin_seed, replica, N, stages, iters_per_stage,
             beta_q, target, total_steps):
    """Single replica. Returns (final_E, best_E, hit_step, found, Etraj)."""
    Ji = J.astype(jnp.int32)                       # (N,N) +-1, 0 diag
    s = spin0.astype(jnp.int32)                    # (N,) +-1
    fields0 = (Ji @ s).astype(jnp.int32)           # int32 fields
    E0 = (-(jnp.sum(s * fields0)) // 2).astype(jnp.int64)

    seed64 = _U64(spin_seed)
    r64 = _U64(replica)

    def step(carry, g):
        s, fields, E, best, hit, found = carry
        gstep = g + jnp.int64(1)                    # 1-based global_step
        stage = (g // iters_per_stage).astype(jnp.int64)
        t = (g % iters_per_stage).astype(jnp.int64)

        mix = (seed64 ^ (r64 << _U64(52)) ^ (stage.astype(jnp.uint64) << _U64(32))
               ^ t.astype(jnp.uint64))
        r_idx = splitmix32(mix)                                  # uint32
        j = ((r_idx.astype(jnp.uint64) * _U64(N)) >> _U64(32)).astype(jnp.int32)

        bq = beta_q[stage.astype(jnp.int32)]
        Lj = fields[j]
        p_plus = p_plus_from_field_q32(Lj, bq)
        old1 = (s[j] == 1)
        r_coin = splitmix32(mix ^ C_COIN ^ j.astype(jnp.uint64))
        new1 = (r_coin < p_plus)
        flip = (new1 != old1)

        s_old = jnp.where(old1, jnp.int32(1), jnp.int32(-1))
        dE = jnp.where(flip, (jnp.int64(2) * s_old.astype(jnp.int64)
                              * Lj.astype(jnp.int64)), jnp.int64(0))
        E = E + dE
        # field update: fields[i] -= 2*J[i][j]*s_old for i!=j; diagonal J=0 so
        # the i==j term is 0 automatically.
        df = jnp.where(flip, (jnp.int32(2) * Ji[:, j] * s_old), jnp.int32(0))
        fields = fields - df
        s = s.at[j].set(jnp.where(flip, -s[j], s[j]))

        best = jnp.minimum(best, E)
        hitnow = (~found) & (E <= target)
        hit = jnp.where(hitnow, gstep, hit)
        found = found | (E <= target)
        return (s, fields, E, best, hit, found), E

    found0 = E0 <= target
    hit0 = jnp.where(found0, jnp.int64(0), jnp.int64(total_steps))
    init = (s, fields0, E0, E0, hit0, found0)
    (s, fields, E, best, hit, found), Etraj = lax.scan(
        step, init, jnp.arange(total_steps, dtype=jnp.int64))
    return E, best, hit, found, Etraj


def trajectory(graph_seed, spin_seed, replica, N, stages, iters_per_stage,
               beta_start, beta_end):
    """Return per-step energy trajectory (total_steps,) int for one replica.

    Used by the test for bit-exact comparison vs rsa_trace.cpp."""
    J = jnp.array(R.make_graph(graph_seed, N))
    spin0 = jnp.array(R.make_init_spins(spin_seed, replica, N))
    beta_q = beta_q_schedule(stages, beta_start, beta_end)
    total = stages * iters_per_stage
    f = jax.jit(_run_one, static_argnums=(4, 5, 6, 9))
    _, _, _, _, Etraj = f(J, spin0, spin_seed, replica, N, stages,
                          iters_per_stage, beta_q, jnp.int64(-(1 << 62)), total)
    return np.array(Etraj).astype(np.int64)


def tts99_steps(total_steps, p):
    if p <= 0:
        return float("inf")
    if p >= 1 - 1e-15:
        return float(total_steps)
    return total_steps * math.log(1 - 0.99) / math.log(1 - p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=32)
    ap.add_argument("--stages", type=int, default=64)
    ap.add_argument("--iters-per-stage", type=int, default=16384)
    ap.add_argument("--beta-start", type=float, default=0.25)
    ap.add_argument("--beta-end", type=float, default=6.0)
    ap.add_argument("--target-energy", type=int, default=-67040)
    ap.add_argument("--graph-seed", type=int, default=1)
    ap.add_argument("--spin-seed", type=int, default=2)
    ap.add_argument("--N", type=int, default=2048)
    ap.add_argument("--traj-out", type=str, default="")
    a = ap.parse_args()

    total = a.stages * a.iters_per_stage
    J = jnp.array(R.make_graph(a.graph_seed, a.N))
    inits = jnp.stack([jnp.array(R.make_init_spins(a.spin_seed, r, a.N))
                       for r in range(a.repeats)])
    beta_q = beta_q_schedule(a.stages, a.beta_start, a.beta_end)
    target = jnp.int64(a.target_energy)
    replicas = jnp.arange(a.repeats, dtype=jnp.int64)

    run = jax.jit(jax.vmap(
        _run_one,
        in_axes=(None, 0, None, 0, None, None, None, None, None, None)),
        static_argnums=(4, 5, 6, 9))

    t0 = time.time()
    E, best, hit, found, Etraj = run(J, inits, a.spin_seed, replicas, a.N,
                                     a.stages, a.iters_per_stage, beta_q,
                                     target, total)
    E.block_until_ready()
    wall_ms = (time.time() - t0) * 1e3

    found = np.array(found); succ = int(found.sum()); p = succ / a.repeats
    hit = np.array(hit); best = np.array(best)
    avg_hit = float(np.where(found, hit, total).mean())
    print(f"backend=bitfaithful mode=rsa repeats={a.repeats} stages={a.stages} "
          f"iters_per_stage={a.iters_per_stage} total_steps={total} "
          f"target_energy={a.target_energy}")
    print(f"successes={succ} success_ratio={p:.6f} avg_hit_steps={avg_hit:.2f} "
          f"TTS99_steps={tts99_steps(total, p):.1f} batch_wall_ms={wall_ms:.3f}")
    print(f"best_energy_overall={int(best.min())}")
    if a.traj_out:
        np.save(a.traj_out, np.array(Etraj))
        print(f"saved trajectory -> {a.traj_out}")


if __name__ == "__main__":
    main()

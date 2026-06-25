import argparse, time, math
from functools import partial
import numpy as np, jax, jax.numpy as jnp
import ref_instance as R
from thrml_update import gibbs_update_site

def beta_per_step(stages, iters_per_stage, b0, b1):
    if stages > 1:
        betas = [(1 - k/(stages-1))*b0 + (k/(stages-1))*b1 for k in range(stages)]
    else:
        betas = [b1]
    return jnp.array(np.repeat(betas, iters_per_stage).astype(np.float32))  # (T,)

def make_chain(J_np, target, total_steps, record_every=1):
    N = J_np.shape[0]
    assert total_steps % record_every == 0, "record_every must divide total_steps"
    n_records = total_steps // record_every
    Jf = jnp.array(J_np.astype(np.float32))
    def run_one(init_s_pm1, betas, key):
        s_bool = init_s_pm1 > 0
        s = init_s_pm1.astype(jnp.float32)
        L = Jf @ s                                   # local fields (no beta)
        E = -0.5 * jnp.dot(s, L)
        def inner_step(carry, inp):
            s_bool, s, L, E, best, hit, found, t = carry
            beta_t, k = inp
            kj, ks = jax.random.split(k)
            j = jax.random.randint(kj, (), 0, N)
            new_b = gibbs_update_site(s_bool, beta_t * Jf[j], ks)
            s_new_j = jnp.where(new_b, 1.0, -1.0)
            delta = s_new_j - s[j]                    # in {-2,0,2}
            E = E - delta * L[j]
            L = L + Jf[j] * delta
            s = s.at[j].set(s_new_j)
            s_bool = s_bool.at[j].set(new_b)
            best = jnp.minimum(best, E)
            hitnow = (~found) & (E <= target)
            hit = jnp.where(hitnow, t, hit)
            found = found | (E <= target)
            return (s_bool, s, L, E, best, hit, found, t + 1), None
        def outer_step(carry, chunk):
            carry, _ = jax.lax.scan(inner_step, carry, chunk)
            return carry, carry[3]                    # emit E after the chunk
        keys = jax.random.split(key, total_steps)
        betas_ch = betas.reshape(n_records, record_every)
        keys_ch = keys.reshape(n_records, record_every, 2)
        hit0 = jnp.where(E <= target, jnp.int32(0), jnp.int32(total_steps))
        init = (s_bool, s, L, E, E, hit0, E <= target, jnp.int32(0))  # 5th elem best=E initially
        (s_bool, s, L, E, best, hit, found, t), Etraj = jax.lax.scan(
            outer_step, init, (betas_ch, keys_ch))
        return E, best, hit, found, Etraj            # Etraj shape (n_records,)
    return jax.jit(jax.vmap(run_one, in_axes=(0, None, 0)))

def tts99_steps(total_steps, p):
    if p <= 0: return float("inf")
    if p >= 1 - 1e-15: return float(total_steps)
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
    J = R.make_graph(a.graph_seed, a.N)
    inits = np.stack([R.make_init_spins(a.spin_seed, r, a.N) for r in range(a.repeats)])
    betas = beta_per_step(a.stages, a.iters_per_stage, a.beta_start, a.beta_end)
    n_pts_target = 2048
    record_every = max(1, total // n_pts_target)
    while total % record_every != 0:
        record_every += 1
    chain = make_chain(J, a.target_energy, total, record_every)
    keys = jax.random.split(jax.random.PRNGKey(a.spin_seed), a.repeats)
    t0 = time.time()
    E, best, hit, found, Etraj = chain(jnp.array(inits), betas, keys)
    E.block_until_ready(); wall_ms = (time.time() - t0) * 1e3
    found = np.array(found); succ = int(found.sum()); p = succ / a.repeats
    hit = np.array(hit); best = np.array(best)
    avg_hit = float(np.where(found, hit, total).mean())
    print(f"backend=thrml mode=rsa repeats={a.repeats} stages={a.stages} "
          f"iters_per_stage={a.iters_per_stage} total_steps={total} target_energy={a.target_energy}")
    print(f"successes={succ} success_ratio={p:.6f} avg_hit_steps={avg_hit:.2f} "
          f"TTS99_steps={tts99_steps(total, p):.1f} batch_wall_ms={wall_ms:.3f}")
    print(f"best_energy_overall={int(best.min())}")
    if a.traj_out:
        np.save(a.traj_out, np.array(Etraj))  # (repeats, total_steps)
        print(f"saved trajectory -> {a.traj_out}")

if __name__ == "__main__":
    main()

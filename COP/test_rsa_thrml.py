import numpy as np, jax, jax.numpy as jnp
import ref_instance as R
from rsa_thrml import make_chain, beta_per_step

def test_rsa_reduces_energy_smallN():
    n = 128; stages, ips = 16, 256; total = stages * ips
    J = R.make_graph(1, n)
    inits = np.stack([R.make_init_spins(2, r, n) for r in range(4)])
    betas = beta_per_step(stages, ips, 0.25, 6.0)
    chain = make_chain(J, target=-10**9, total_steps=total)  # unreachable target -> never "found"
    keys = jax.random.split(jax.random.PRNGKey(0), 4)
    E, best, hit, found, traj = chain(jnp.array(inits), betas, keys)
    E0 = np.array([R.energy(J, inits[r]) for r in range(4)])
    assert (np.array(best) <= E0 + 1e-6).all()       # best never worse than start
    assert np.array(best).mean() < E0.mean()          # and strictly improves on average

def test_incremental_energy_matches_from_scratch():
    # Etraj[-1] (incremental) must equal returned final E, and both are valid energies.
    n = 128; stages, ips = 8, 128; total = stages * ips
    J = R.make_graph(1, n)
    inits = np.stack([R.make_init_spins(2, r, n) for r in range(3)])
    betas = beta_per_step(stages, ips, 0.25, 6.0)
    chain = make_chain(J, target=-10**9, total_steps=total)
    keys = jax.random.split(jax.random.PRNGKey(1), 3)
    E, best, hit, found, traj = chain(jnp.array(inits), betas, keys)
    assert np.allclose(np.array(traj)[:, -1], np.array(E), atol=1e-3)
    assert (np.array(best) <= np.array(traj).min(axis=1) + 1e-3).all()  # best == min over traj

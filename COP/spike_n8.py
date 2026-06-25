"""
spike_n8.py -- Validated minimal "RSA on thrml" at N=8.

RSA = Random-Sequential single-site heat-bath annealing on a COMPLETE-graph Ising
problem, built on Extropic's thrml (0.1.3) native model + Gibbs conditional machinery.

Per RSA step:
  1. pick site j uniformly in [0,N)
  2. local field L_j = sum_{i!=j} J_ij s_i      (s in {+1,-1})
  3. p_plus = sigmoid(2*beta*L_j); new s_j = +1 if u<p_plus else -1
  4. energy E(s) = -sum_{i<j} J_ij s_i s_j

KEY API FACTS (from reading the thrml source):
  * thrml.pgm.SpinNode: a spin variable in {-1,+1}, stored as BOOL, True=+1, False=-1.
  * IsingEBM(nodes, edges, biases, weights, beta) has energy
        E(s) = -beta*( sum_i b_i s_i + sum_(i,j) J_ij s_i s_j ).
    NOTE the leading -beta: ebm.energy(...) returns beta*(bare J energy), so divide by
    beta to recover -sum_{i<j} J_ij s_i s_j.
  * SpinGibbsConditional (a BernoulliConditional) samples P(s=+1)=sigmoid(2*gamma),
    where gamma = sum over DiscreteEBMInteractions of  spin_prod * W * active.
    For a single-spin-tail interaction with weights = beta*J[j,:] and state = all spins,
    gamma_j = sum_i (beta*J[j,i]) s_i = beta*L_j  -> EXACTLY the heat-bath field.

THE STATIC-INDEX QUESTION (answer: (A) data-driven works, fully jitted):
  sample_single_block(key, state_free, clamp_state, program, block, sampler_state)
  takes `block` as a STATIC python int -- it indexes python lists
  (program.per_block_interaction_global_inds[block], program.samplers[block], ...).
  Passing a TRACED index raises TracerIntegerConversionError. It also rejects a native
  batch dim (its _split_states validation requires state shape exactly [n,k]).
  => For random-site-under-jit we DO NOT call sample_single_block in the inner loop.
     Instead we call thrml's SpinGibbsConditional.sample directly, feeding it a
     DiscreteEBMInteraction whose weights are the (traced) row beta*J[j,:] gathered
     from the dense coupling matrix. thrml still computes gamma; the site index is
     fully traced; the whole RSA scan jits. Batch over replicas with jax.vmap
     (the conditional itself runs un-batched per replica).
"""
import time
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp

from thrml.pgm import SpinNode
from thrml.block_management import Block
from thrml.models.ising import IsingEBM, IsingSamplingProgram
from thrml.models.discrete_ebm import SpinGibbsConditional, DiscreteEBMInteraction

N = 8
BETA = 4.0
N_STEPS = 400


# ---------- problem: complete-graph +-1 Ising ----------
def make_J(n):
    J = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            v = 1.0 if ((i * 131 + j * 17) % 2) else -1.0
            J[i, j] = v
            J[j, i] = v
    return J


J = make_J(N)
assert np.allclose(J, J.T) and np.allclose(np.diag(J), 0)
Jb = jnp.array(BETA * J)  # beta-folded coupling matrix (diagonal 0)


# ---------- thrml IsingEBM (used for the energy-convention check) ----------
nodes = [SpinNode() for _ in range(N)]
edges, weights = [], []
for i in range(N):
    for j in range(i + 1, N):
        edges.append((nodes[i], nodes[j]))
        weights.append(J[i, j])
ebm = IsingEBM(
    nodes,
    edges,
    jnp.zeros(N, dtype=jnp.float32),          # biases
    jnp.array(weights, dtype=jnp.float32),    # weights J_ij
    jnp.array(BETA, dtype=jnp.float32),       # beta
)
# one single-node free block per site (this is how thrml would do per-site Gibbs)
free_blocks = [Block([nodes[k]]) for k in range(N)]
program = IsingSamplingProgram(ebm, free_blocks, [])  # IsingSamplingProgram (FactorSamplingProgram)


# ---------- spin <-> energy helpers ----------
def bool_to_pm(s_bool):
    """thrml stores spins as bool, True=+1 / False=-1.  Bool -> {+1,-1}."""
    return 2.0 * np.asarray(s_bool).astype(np.float32) - 1.0


def E_numpy(s_bool):
    s = bool_to_pm(s_bool)
    return -0.5 * float(s @ J @ s)            # = -sum_{i<j} J_ij s_i s_j


# ======================================================================
# VALIDATION 1: energy convention (numpy  vs  thrml IsingEBM.energy)
# ======================================================================
rng = np.random.default_rng(0)
s_bool0 = rng.random(N) < 0.5
state_free = [jnp.array([bool(s_bool0[k])], dtype=jnp.bool_) for k in range(N)]  # per-block (1,) bool
E_np = E_numpy(s_bool0)
E_thrml = float(ebm.energy(state_free, free_blocks))   # returns beta * bare energy
print("[1] ENERGY CONVENTION")
print("    spins(+-1):", bool_to_pm(s_bool0).astype(int).tolist())
print("    E_numpy(bare)            =", E_np)
print("    ebm.energy (=beta*bare)  =", E_thrml, " ; /beta =", E_thrml / BETA)
assert abs(E_thrml / BETA - E_np) < 1e-4, "energy convention mismatch"
print("    -> MATCH (ebm.energy == beta * (-sum_{i<j} J_ij s_i s_j))")


# ======================================================================
# RSA inner step: thrml SpinGibbsConditional with a TRACED site index
# ======================================================================
_samp = SpinGibbsConditional()


def gibbs_update_site(spins_bool, j, key):
    """One heat-bath update of site j (traced) using thrml's conditional.
    spins_bool: (N,) bool global spin vector.  Returns new scalar bool for site j."""
    w_row = Jb[j]                                       # (N,)  beta*J[j,:]   (traced gather)
    interaction = DiscreteEBMInteraction(n_spin=1, weights=w_row[None, :])  # 1 head, k=N tail terms
    active = jnp.ones((1, N), dtype=bool)               # all terms active (self-term has J=0)
    states = [[spins_bool[None, :]]]                    # one interaction, one spin-tail group, shape (1,N)
    out_sd = jax.ShapeDtypeStruct((1,), jnp.bool_)
    new_val, _ = _samp.sample(key, [interaction], [active], states, None, out_sd)
    return new_val[0]                                   # thrml drew bernoulli(sigmoid(2*gamma))


# gamma sanity: thrml gamma == beta*L_j
g_check, _ = _samp.compute_parameters(
    jax.random.PRNGKey(0),
    [DiscreteEBMInteraction(1, Jb[3][None, :])],
    [jnp.ones((1, N), bool)],
    [[jnp.array(s_bool0)[None, :]]],
    None,
    jax.ShapeDtypeStruct((1,), jnp.bool_),
)
L3 = float(np.sum([J[3, i] * bool_to_pm(s_bool0)[i] for i in range(N)]))
print("[gamma] thrml gamma_3 =", float(g_check[0]), " expected beta*L_3 =", BETA * L3)
assert abs(float(g_check[0]) - BETA * L3) < 1e-4


def rsa_step(spins, key):
    kj, ks = jax.random.split(key)
    j = jax.random.randint(kj, (), 0, N)               # random site, TRACED
    new = gibbs_update_site(spins, j, ks)
    return spins.at[j].set(new), None


@partial(jax.jit, static_argnums=(2,))
def rsa_run(key, init_spins, n_steps):
    keys = jax.random.split(key, n_steps)
    final, _ = jax.lax.scan(rsa_step, init_spins, keys)
    return final


# ======================================================================
# VALIDATION 2: high-beta correctness -> energy decreases
# ======================================================================
init = jax.random.bernoulli(jax.random.PRNGKey(7), 0.5, (N,))
final = rsa_run(jax.random.PRNGKey(123), init, N_STEPS)
print("[2] CORRECTNESS (beta=%.1f, %d steps, single chain)" % (BETA, N_STEPS))
print("    E_init =", E_numpy(init), "  E_final =", E_numpy(final))
assert E_numpy(final) <= E_numpy(init) + 1e-6


# ======================================================================
# VALIDATION 3: batch dimension (replicas) via vmap
# ======================================================================
R = 32
def run_one(key):
    init_r = jax.random.bernoulli(jax.random.fold_in(key, 0), 0.5, (N,))
    fin_r = rsa_run(jax.random.fold_in(key, 1), init_r, N_STEPS)
    return fin_r, init_r

keys = jax.random.split(jax.random.PRNGKey(55), R)
fin_b, init_b = jax.vmap(run_one)(keys)
Ei = np.array([E_numpy(init_b[r]) for r in range(R)])
Ef = np.array([E_numpy(fin_b[r]) for r in range(R)])
print("[3] BATCH (%d replicas via vmap)" % R)
print("    mean E_init =%.3f   mean E_final =%.3f" % (Ei.mean(), Ef.mean()))
print("    all replicas energy did not increase:", bool(np.all(Ef <= Ei + 1e-6)))


# ======================================================================
# VALIDATION 4: per-step wall time
# ======================================================================
f = rsa_run(jax.random.PRNGKey(1), init, N_STEPS); jax.block_until_ready(f)
t0 = time.time()
for _ in range(10):
    f = rsa_run(jax.random.PRNGKey(2), init, N_STEPS); jax.block_until_ready(f)
dt1 = (time.time() - t0) / 10

run_v = jax.jit(lambda k: jax.vmap(run_one)(jax.random.split(k, R)))
fb = run_v(jax.random.PRNGKey(3)); jax.block_until_ready(fb)
t0 = time.time()
for _ in range(10):
    fb = run_v(jax.random.PRNGKey(4)); jax.block_until_ready(fb)
dtv = (time.time() - t0) / 10

print("[4] TIMING (CPU)")
print("    1 chain  N=8 : %.2f us/step  (%.2f ms / %d steps)" % (dt1/N_STEPS*1e6, dt1*1e3, N_STEPS))
print("    %d replicas  : %.2f us/step  (%.2f ms / %d steps)" % (R, dtv/N_STEPS*1e6, dtv*1e3, N_STEPS))
print("ALL CHECKS PASSED")

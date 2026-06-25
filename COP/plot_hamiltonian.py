import numpy as np, jax, jax.numpy as jnp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import rsa_bitfaithful as B   # sets x64
import ref_instance as R

reps, stages, ips = 32, 64, 16384
b0, b1, N = 0.25, 6.0, 2048
total = stages*ips
J = jnp.array(R.make_graph(1, N))
inits = jnp.stack([jnp.array(R.make_init_spins(2, r, N)) for r in range(reps)])
beta_q = B.beta_q_schedule(stages, b0, b1)
reps_idx = jnp.arange(reps, dtype=jnp.int64)
run = jax.jit(jax.vmap(B._run_one,
        in_axes=(None,0,None,0,None,None,None,None,None,None)),
        static_argnums=(4,5,6,9))
E,best,hit,found,Etraj = run(J, inits, 2, reps_idx, N, stages, ips, beta_q,
                             jnp.int64(-67040), total)
Etraj = np.array(Etraj)                      # (reps, total) int
found = np.array(found)
stride = total//2000
xs = np.arange(0, total, stride) + stride    # MC step numbers
Y = Etraj[:, ::stride]

plt.figure(figsize=(8,5))
for r in range(reps):
    plt.plot(xs, Y[r], color="0.6", lw=0.5, alpha=0.45)
plt.plot(xs, Y.mean(0), color="C0", lw=2.0)          # ensemble mean
plt.xlabel("MC steps"); plt.ylabel("Hamiltonian")
plt.xlim(0, total)
plt.tight_layout()
plt.savefig("hamiltonian.png", dpi=150)
print("wrote hamiltonian.png  final mean H =", float(Y.mean(0)[-1]),
      " best H =", int(Etraj.min()), " successes =", int(found.sum()))

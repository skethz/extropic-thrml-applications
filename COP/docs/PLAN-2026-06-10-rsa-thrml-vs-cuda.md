# RSA on thrml vs CUDA kernel (K2048) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Show that single-site RSA annealing reaches the K2048 Max-Cut target (E <= -67040) with statistically equivalent solution quality whether the heat-bath update is run by Extropic's thrml `SpinGibbsConditional` or by the GH200 CUDA kernel, on the identical instance/schedule/step-budget.

**Architecture:** All work lives in `/scratch/hongse/ising/k2000/thrml_rsa/`. The thrml side keeps a dense (N,N) +/-1 coupling matrix and, each step, calls thrml's `SpinGibbsConditional.sample()` on a traced random site (weights = beta*J[j,:]); energy/field bookkeeping is plain incremental linear algebra. Replicas are vmapped; the step loop is a jitted `lax.scan`. The CUDA side is the existing `gh200_k2048_rsa_rwa` binary. A comparison harness runs both on the same (graph_seed=1, spin_seed=2) instance and reports step-axis metrics.

**Tech Stack:** Python 3.12, jax 0.10.1 (CPU), thrml 0.1.3 (venv `/scratch/hongse/venvs/thrml`); the prebuilt CUDA binary `../gh200_k2048_rsa_rwa`; numpy, matplotlib.

**Reference facts (from the kernel `../gh200_k2048_rsa_rwa.cu`):**
- `splitmix32(x)`: x += 0x9E3779B97F4A7C15; z=x; z=(z^(z>>30))*0xBF58476D1CE4E5B9; z=(z^(z>>27))*0x94D049BB133111EB; return (z>>32)&0xFFFFFFFF  (all uint64 mod 2^64).
- Graph: for i<j, edge_key = seed ^ (i<<21) ^ j; J_ij = +1 if (splitmix32(edge_key)&1) else -1; symmetric; zero diagonal.
- Init spins (replica r): for w in 0..31: lo=splitmix32(seed ^ (r<<32) ^ w); hi=splitmix32(seed ^ (r<<40) ^ w ^ 0xA5A5A5A5); word=(hi<<32)|lo. Spin bit i = (word_{i>>6} >> (i&63))&1 -> +1 if 1 else -1.
- Energy: E(s) = -0.5 * s^T J s = -sum_{i<j} J_ij s_i s_j. target_energy = -67040.
- Validated thrml inner-loop lives in `spike_n8.py` (already in this dir) — lift its exact imports.

**Headline matched config:** mode=rsa, N=2048, stages=64, iters_per_stage=16384 (total_steps=1,048,576), beta 0.25->6.0, repeats=32, graph_seed=1, spin_seed=2, target=-67040. CUDA ref at this config ~2/32 success, best ~ -67604.

**Note on commits:** this remote dir is NOT a git repo. Do Task 0 to enable real commits, else treat each "Commit" step as a save-and-verify checkpoint.

---

### Task 0: (optional) enable version control

**Files:** Create: `/scratch/hongse/ising/k2000/thrml_rsa/.gitignore`

- [ ] **Step 1: init repo (scoped to this subdir only)**

```bash
cd /scratch/hongse/ising/k2000/thrml_rsa
git init -q
printf '__pycache__/\n*.pyc\n*.npy\n*.png\nresults_*/\n' > .gitignore
git add .gitignore SPEC-2026-06-10-rsa-thrml-vs-cuda.md PLAN-2026-06-10-rsa-thrml-vs-cuda.md spike_n8.py
git commit -q -m "chore: spec, plan, validated thrml spike" && echo COMMITTED
```
Expected: `COMMITTED`.

---

### Task 1: Reference instance + energy, with an airtight C++ cross-check

**Files:**
- Create: `thrml_rsa/ref_instance.py`
- Create: `thrml_rsa/init_energy_check.cpp`
- Test: `thrml_rsa/test_ref_instance.py`

- [ ] **Step 1: Write `ref_instance.py`** (numpy reproduction; activate venv first: `source /scratch/hongse/venvs/thrml/bin/activate`)

```python
import numpy as np
N = 2048
M64 = (1 << 64) - 1
def splitmix32(x):
    x = (x + 0x9E3779B97F4A7C15) & M64
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
    return (z >> 32) & 0xFFFFFFFF

def make_graph(seed=1, n=N):
    J = np.full((n, n), -1, dtype=np.int8)
    np.fill_diagonal(J, 0)
    for i in range(n):
        for j in range(i + 1, n):
            key = (seed ^ ((i << 21) & M64) ^ j) & M64
            if splitmix32(key) & 1:
                J[i, j] = 1; J[j, i] = 1
    return J

def make_init_spins(seed=2, replica=0, n=N):
    nw = n // 64
    s = np.empty(n, dtype=np.int8)
    for w in range(nw):
        lo = splitmix32((seed ^ ((replica << 32) & M64) ^ w) & M64)
        hi = splitmix32((seed ^ ((replica << 40) & M64) ^ w ^ 0xA5A5A5A5) & M64)
        word = ((hi << 32) | lo) & M64
        for b in range(64):
            s[w * 64 + b] = 1 if (word >> b) & 1 else -1
    return s

def energy(J, s):
    s = s.astype(np.int64)
    return -0.5 * float(s @ (J.astype(np.int64) @ s))

if __name__ == "__main__":
    J = make_graph(1)
    for r in range(4):
        s = make_init_spins(2, r)
        print(f"replica {r}: E0 = {energy(J, s):.1f}")
```

- [ ] **Step 2: Run it**

Run: `source /scratch/hongse/venvs/thrml/bin/activate && cd /scratch/hongse/ising/k2000/thrml_rsa && python3 ref_instance.py`
Expected: 4 integer-valued E0 lines (graph build ~seconds). Record the replica-0 value.

- [ ] **Step 3: Write `init_energy_check.cpp`** (copies the kernel's exact functions)

```cpp
#include <cstdint>
#include <cstdio>
#include <vector>
static constexpr int N = 2048;
static inline uint32_t splitmix32(uint64_t x){
    x += 0x9E3779B97F4A7C15ull; uint64_t z=x;
    z=(z^(z>>30))*0xBF58476D1CE4E5B9ull; z=(z^(z>>27))*0x94D049BB133111EBull;
    return (uint32_t)(z>>32);
}
int main(){
    static int8_t J[N][N];
    for(int i=0;i<N;i++) for(int j=0;j<N;j++) J[i][j]= (i==j)?0:-1;
    for(int i=0;i<N;i++) for(int j=i+1;j<N;j++){
        uint64_t key = 1ull ^ ((uint64_t)i<<21) ^ (uint64_t)j;
        if(splitmix32(key)&1u){ J[i][j]=1; J[j][i]=1; }
    }
    for(int r=0;r<4;r++){
        std::vector<int> s(N);
        for(int w=0;w<N/64;w++){
            uint64_t lo=splitmix32(2ull ^ ((uint64_t)r<<32) ^ (uint64_t)w);
            uint64_t hi=splitmix32(2ull ^ ((uint64_t)r<<40) ^ (uint64_t)w ^ 0xA5A5A5A5ull);
            uint64_t word=(hi<<32)|lo;
            for(int b=0;b<64;b++) s[w*64+b]= ((word>>b)&1ull)?1:-1;
        }
        long acc=0; for(int i=0;i<N;i++){ long Li=0; for(int j=0;j<N;j++) Li+=(long)J[i][j]*s[j]; acc+=(long)s[i]*Li; }
        printf("replica %d: E0 = %.1f\n", r, -0.5*(double)acc);
    }
}
```

- [ ] **Step 4: Compile + run the C++ check and diff against Python (the hard G0 gate)**

Run:
```bash
cd /scratch/hongse/ising/k2000/thrml_rsa
g++ -O2 init_energy_check.cpp -o init_energy_check && ./init_energy_check > cpp_e0.txt
python3 ref_instance.py > py_e0.txt
diff cpp_e0.txt py_e0.txt && echo "G0 PASS: instance reproduction matches kernel"
```
Expected: `G0 PASS: ...` (zero diff). If it differs, STOP — the Python reproduction of splitmix32/graph/init-spins is wrong; fix before continuing.

- [ ] **Step 5: Write `test_ref_instance.py`** (locks energy + symmetry invariants)

```python
import numpy as np, ref_instance as R
def test_graph_symmetric_pm1():
    J = R.make_graph(1, 64)
    assert (J == J.T).all()
    assert set(np.unique(J)).issubset({-1, 0, 1})
    assert (np.diag(J) == 0).all()
def test_energy_matches_pairwise_sum():
    J = R.make_graph(1, 64); s = R.make_init_spins(2, 0, 64)
    pair = -sum(J[i, j] * s[i] * s[j] for i in range(64) for j in range(i + 1, 64))
    assert R.energy(J, s) == pair
```

- [ ] **Step 6: Run tests, then checkpoint**

Run: `python3 -m pytest test_ref_instance.py -q`  (pip install pytest into the venv if missing)
Expected: `2 passed`.
```bash
git add ref_instance.py init_energy_check.cpp test_ref_instance.py 2>/dev/null
git commit -q -m "feat: K2048 instance reproduction + G0 cross-check" 2>/dev/null && echo COMMITTED || echo CHECKPOINT
```

---

### Task 2: thrml-native single-site heat-bath update (lift + wrap the spike)

**Files:**
- Create: `thrml_rsa/thrml_update.py`
- Test: `thrml_rsa/test_thrml_update.py`

- [ ] **Step 1: Inspect the validated spike to copy its exact imports**

Run: `cd /scratch/hongse/ising/k2000/thrml_rsa && grep -nE "import|DiscreteEBMInteraction|SpinGibbsConditional" spike_n8.py`
Expected: shows the working imports (e.g. `from thrml.models.discrete_ebm import SpinGibbsConditional, DiscreteEBMInteraction`). Use EXACTLY these in the next step.

- [ ] **Step 2: Write `thrml_update.py`** (single thrml-native heat-bath site update; copy import lines verbatim from spike_n8.py)

```python
from functools import partial
import jax, jax.numpy as jnp
# >>> paste the exact import(s) for SpinGibbsConditional and DiscreteEBMInteraction
#     as found in spike_n8.py Step-1 output <<<
from thrml.models.discrete_ebm import SpinGibbsConditional, DiscreteEBMInteraction

_SAMP = SpinGibbsConditional()
_OUT_SD = jax.ShapeDtypeStruct((1,), jnp.bool_)

def gibbs_update_site(spins_bool, w_row, key):
    """thrml heat-bath for one site: returns the new bool spin value.
    spins_bool: (N,) bool (True=+1). w_row: (N,) float = beta*J[j,:]. j is encoded in w_row."""
    interaction = DiscreteEBMInteraction(n_spin=1, weights=w_row[None, :])
    active = jnp.ones((1, w_row.shape[0]), dtype=bool)
    states = [[spins_bool[None, :]]]
    new_val, _ = _SAMP.sample(key, [interaction], [active], states, None, _OUT_SD)
    return new_val[0]
```
(If the spike passed `n_spin`/weights differently, mirror the spike exactly — the spike is the source of truth.)

- [ ] **Step 3: Write `test_thrml_update.py`** (gamma==beta*L and high-beta determinism)

```python
import jax, jax.numpy as jnp, numpy as np
import ref_instance as R
from thrml_update import gibbs_update_site

def test_gamma_equals_beta_L_via_determinism():
    n = 64; beta = 8.0
    J = R.make_graph(1, n).astype(np.float32)
    s = R.make_init_spins(2, 0, n)
    sb = jnp.array(s > 0)
    Js = jnp.array(J) @ jnp.array(s.astype(np.float32))   # L = J s
    for j in [0, 7, 31, 63]:
        w_row = beta * jnp.array(J[j])
        # at large beta, heat-bath is ~deterministic: new spin = sign(L_j)
        key = jax.random.PRNGKey(j)
        new_b = bool(gibbs_update_site(sb, w_row, key))
        assert new_b == (float(Js[j]) > 0)
```

- [ ] **Step 4: Run the test**

Run: `python3 -m pytest test_thrml_update.py -q`
Expected: `1 passed` (thrml's conditional drives the spin to sign(L_j) at beta=8). If it fails, recheck weight scaling (must be beta*J, not J) and the bool=+1 convention against spike_n8.py.

- [ ] **Step 5: Checkpoint**

```bash
git add thrml_update.py test_thrml_update.py 2>/dev/null
git commit -q -m "feat: thrml-native single-site heat-bath update" 2>/dev/null && echo COMMITTED || echo CHECKPOINT
```

---

### Task 3: full RSA driver (anneal + scan + vmap + incremental energy + metrics + CLI)

**Files:**
- Create: `thrml_rsa/rsa_thrml.py`
- Test: `thrml_rsa/test_rsa_thrml.py`

- [ ] **Step 1: Write `rsa_thrml.py`**

```python
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

def make_chain(J_np, target, total_steps):
    N = J_np.shape[0]
    Jf = jnp.array(J_np.astype(np.float32))
    def run_one(init_s_pm1, betas, key):
        s_bool = init_s_pm1 > 0
        s = init_s_pm1.astype(jnp.float32)
        L = Jf @ s                                   # local fields (no beta)
        E = -0.5 * jnp.dot(s, L)
        def step(carry, inp):
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
            return (s_bool, s, L, E, best, hit, found, t + 1), E
        keys = jax.random.split(key, total_steps)
        init = (s_bool, s, L, E, E, jnp.int32(total_steps), E <= target, jnp.int32(0))
        (s_bool, s, L, E, best, hit, found, t), Etraj = jax.lax.scan(step, init, (betas, keys))
        return E, best, hit, found, Etraj
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
    chain = make_chain(J, a.target_energy, total)
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
```

- [ ] **Step 2: Write `test_rsa_thrml.py`** (small-N: RSA must drive energy below the random start)

```python
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
```

- [ ] **Step 3: Run the test**

Run: `cd /scratch/hongse/ising/k2000/thrml_rsa && python3 -m pytest test_rsa_thrml.py -q`
Expected: `1 passed` (a few seconds; small N, 4096 steps).

- [ ] **Step 4: Checkpoint**

```bash
git add rsa_thrml.py test_rsa_thrml.py 2>/dev/null
git commit -q -m "feat: thrml RSA driver with anneal/scan/vmap + metrics CLI" 2>/dev/null && echo COMMITTED || echo CHECKPOINT
```

---

### Task 4: G1 smoke at N=2048 (confirm it runs + measure per-step cost)

**Files:** none new (uses `rsa_thrml.py`).

- [ ] **Step 1: Tiny-budget N=2048 run, timed**

Run:
```bash
cd /scratch/hongse/ising/k2000/thrml_rsa
time python3 rsa_thrml.py --repeats 4 --stages 4 --iters-per-stage 1024 --N 2048
```
Expected: prints a `backend=thrml ... total_steps=4096 ...` block and a `best_energy_overall=` line; wall time is dominated by JAX compile. Record `batch_wall_ms` (excludes compile) to extrapolate.

- [ ] **Step 2: Per-step extrapolation sanity gate**

Compute approx per-step cost = `batch_wall_ms / 4096`. The headline run is 1,048,576 steps. Estimate `est_minutes = per_step_ms * 1.048576e6 / 6e4`. If `est_minutes` > ~60, STOP and report: the CPU budget is heavier than expected — options are fewer replicas, a smaller budget, or revisiting GPU jaxlib. Otherwise proceed.

---

### Task 5: comparison harness (run both backends, table + trajectory plot)

**Files:** Create: `thrml_rsa/compare_rsa.py`

- [ ] **Step 1: Write `compare_rsa.py`**

```python
import argparse, subprocess, re, math, os
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

CUDA_BIN = "/scratch/hongse/ising/k2000/gh200_k2048_rsa_rwa"

def tts99_steps(total, p):
    if p <= 0: return float("inf")
    if p >= 1 - 1e-15: return float(total)
    return total * math.log(0.01) / math.log(1 - p)

def run_cuda(repeats, stages, ips, b0, b1, target):
    cmd = [CUDA_BIN, "--mode", "rsa", "--backend", "gpu", "--repeats", str(repeats),
           "--stages", str(stages), "--iters-per-stage", str(ips),
           "--beta-start", str(b0), "--beta-end", str(b1), "--target-energy", str(target)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="1")
    out = subprocess.run(cmd, capture_output=True, text=True, env=env).stdout
    succ = int(re.search(r"successes=(\d+)", out).group(1))
    p = float(re.search(r"success_ratio=([\d.]+)", out).group(1))
    avg_hit = float(re.search(r"avg_hit_steps=([\d.]+)", out).group(1))
    best = int(re.search(r"best_energy_overall=(-?\d+)", out).group(1))
    total = stages * ips
    return dict(backend="cuda", successes=succ, p=p, avg_hit=avg_hit, best=best,
                tts99=tts99_steps(total, p), total=total)

def run_thrml(repeats, stages, ips, b0, b1, target, traj_path):
    cmd = ["python3", "/scratch/hongse/ising/k2000/thrml_rsa/rsa_thrml.py",
           "--repeats", str(repeats), "--stages", str(stages), "--iters-per-stage", str(ips),
           "--beta-start", str(b0), "--beta-end", str(b1), "--target-energy", str(target),
           "--traj-out", traj_path]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    succ = int(re.search(r"successes=(\d+)", out).group(1))
    p = float(re.search(r"success_ratio=([\d.]+)", out).group(1))
    avg_hit = float(re.search(r"avg_hit_steps=([\d.]+)", out).group(1))
    best = int(re.search(r"best_energy_overall=(-?\d+)", out).group(1))
    total = stages * ips
    return dict(backend="thrml", successes=succ, p=p, avg_hit=avg_hit, best=best,
                tts99=tts99_steps(total, p), total=total)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=32)
    ap.add_argument("--stages", type=int, default=64)
    ap.add_argument("--iters-per-stage", type=int, default=16384)
    ap.add_argument("--beta-start", type=float, default=0.25)
    ap.add_argument("--beta-end", type=float, default=6.0)
    ap.add_argument("--target-energy", type=int, default=-67040)
    ap.add_argument("--outdir", type=str, default="results_headline")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    traj = os.path.join(a.outdir, "thrml_traj.npy")
    c = run_cuda(a.repeats, a.stages, a.iters_per_stage, a.beta_start, a.beta_end, a.target_energy)
    t = run_thrml(a.repeats, a.stages, a.iters_per_stage, a.beta_start, a.beta_end, a.target_energy, traj)
    hdr = f"{'metric':<20}{'thrml':>16}{'cuda':>16}"
    rows = [("successes", t["successes"], c["successes"]),
            ("success_ratio", f"{t['p']:.4f}", f"{c['p']:.4f}"),
            ("avg_hit_steps", f"{t['avg_hit']:.0f}", f"{c['avg_hit']:.0f}"),
            ("best_energy", t["best"], c["best"]),
            ("TTS99_steps", f"{t['tts99']:.0f}", f"{c['tts99']:.0f}")]
    lines = [f"matched config: N=2048 stages={a.stages} iters_per_stage={a.iters_per_stage} "
             f"total_steps={t['total']} beta={a.beta_start}->{a.beta_end} repeats={a.repeats} "
             f"target={a.target_energy}", "", hdr, "-"*52]
    lines += [f"{m:<20}{str(tv):>16}{str(cv):>16}" for m, tv, cv in rows]
    table = "\n".join(lines)
    print(table)
    with open(os.path.join(a.outdir, "table.txt"), "w") as f: f.write(table + "\n")
    # trajectory plot (thrml has per-step E; mark cuda best + target)
    Et = np.load(traj)                                  # (repeats, total)
    stride = max(1, Et.shape[1] // 2000)
    xs = np.arange(0, Et.shape[1], stride)
    plt.figure(figsize=(8, 5))
    plt.plot(xs, Et[:, ::stride].mean(0), color="C0", label="thrml mean E")
    plt.fill_between(xs, Et[:, ::stride].min(0), Et[:, ::stride].max(0), color="C0", alpha=0.2)
    plt.axhline(c["best"], color="C1", ls="--", label=f"cuda best E={c['best']}")
    plt.axhline(a.target_energy, color="k", ls=":", label=f"target={a.target_energy}")
    plt.xlabel("RSA step"); plt.ylabel("energy"); plt.legend(); plt.title("RSA: thrml vs CUDA (K2048)")
    plt.tight_layout(); plt.savefig(os.path.join(a.outdir, "trajectory.png"), dpi=120)
    print(f"\nwrote {a.outdir}/table.txt and {a.outdir}/trajectory.png")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke the harness at a tiny matched budget**

Run:
```bash
cd /scratch/hongse/ising/k2000/thrml_rsa
python3 compare_rsa.py --repeats 8 --stages 4 --iters-per-stage 1024 --outdir results_smoke
```
Expected: a printed two-column table (thrml vs cuda) + `results_smoke/table.txt` and `trajectory.png`. Both backends likely show 0 successes at this tiny budget — that's fine; we're checking the harness runs end-to-end and parses both outputs.

- [ ] **Step 3: Checkpoint**

```bash
git add compare_rsa.py 2>/dev/null
git commit -q -m "feat: thrml-vs-cuda RSA comparison harness" 2>/dev/null && echo COMMITTED || echo CHECKPOINT
```

---

### Task 6: headline run + parity verdict README

**Files:** Create: `thrml_rsa/README.md` (+ generated `results_headline/`)

- [ ] **Step 1: Run the headline matched config** (long — minutes to tens of minutes on CPU; consider running detached)

Run:
```bash
cd /scratch/hongse/ising/k2000/thrml_rsa
python3 compare_rsa.py --repeats 32 --stages 64 --iters-per-stage 16384 \
  --beta-start 0.25 --beta-end 6.0 --target-energy -67040 --outdir results_headline | tee results_headline/run.log
```
Expected: the side-by-side table; CUDA ~2/32 success / best ~ -67604; thrml in the same ballpark (success a small integer / similar best energy). `results_headline/{table.txt,trajectory.png,thrml_traj.npy}` written.

- [ ] **Step 2: Write `README.md`** with the actual numbers from `results_headline/table.txt`

```markdown
# RSA on thrml vs CUDA kernel — K2048 Max-Cut

Same RSA algorithm (single-site heat-bath annealing), same instance (graph_seed=1,
spin_seed=2), same matched step budget; thrml runs the heat-bath via
`SpinGibbsConditional`, CUDA runs the hand-written kernel.

## Headline result (paste from results_headline/table.txt)
<TABLE>

## Verdict
Parity / not-parity statement on success_ratio + best_energy + trajectory overlap.
thrml path used: dense (N,N) beta*J matrix + SpinGibbsConditional (no edge-list build).
Caveat: thrml ran on CPU jax; comparison is on the hardware-agnostic STEP axis
(wall-clock is not comparable to the GH200 kernel).

## Reproduce
    source /scratch/hongse/venvs/thrml/bin/activate
    python3 compare_rsa.py --repeats 32 --stages 64 --iters-per-stage 16384 \
      --beta-start 0.25 --beta-end 6.0 --target-energy -67040 --outdir results_headline
```
Replace `<TABLE>` with the real table and write the one-line verdict from the data.

- [ ] **Step 3: Final checkpoint**

```bash
git add README.md results_headline/table.txt 2>/dev/null
git commit -q -m "docs: RSA thrml-vs-cuda headline result + verdict" 2>/dev/null && echo COMMITTED || echo CHECKPOINT
```

---

## Done criteria
- G0 passes (C++/Python instance match) — Task 1 Step 4.
- thrml update is genuinely thrml's `SpinGibbsConditional` and matches sign(L_j) at high beta — Task 2.
- thrml RSA reduces energy at small N — Task 3.
- N=2048 runs and per-step cost is acceptable — Task 4.
- Harness emits the matched-budget table + trajectory — Task 5.
- Headline table + README verdict produced — Task 6.

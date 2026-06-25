# Spec: RSA on thrml vs RSA on the CUDA kernel (K2048 Max-Cut)

Date: 2026-06-10
Author: hongse (+ Claude)
Status: approved (32 replicas, 1,048,576-step headline budget)

## 1. Goal

Demonstrate that the **RSA** (Random-Sequential-Annealing, single-site heat-bath)
algorithm produces **statistically equivalent solution quality** when its update
is executed by Extropic's `thrml` thermodynamic / TSU-style block-Gibbs sampler
versus the hand-written GH200 CUDA kernel (`gh200_k2048_rsa_rwa.cu`), on the
*identical* K2048 problem instance, schedule, and step budget.

Parity is the win (mirrors the earlier thrml-mask result). This is NOT about RSA
being a good solver — single-site RSA is intrinsically slow on a dense spin glass.

## 2. Scope

In scope:
- RSA only (mode=rsa). RWA is explicitly excluded.
- N = 2048 only (the real K2048 instance). No N=64/256 staging.
- thrml-native float dynamics (real thrml IsingEBM + SpinGibbsConditional heat
  bath, continuous jax.nn.sigmoid). NOT a bit-faithful Q0.32-LUT port.
- thrml on CPU jax (the existing /scratch/hongse/venvs/thrml venv). Accept runtime.
- Comparison on the hardware-agnostic STEP axis.

Out of scope: RWA, GPU jaxlib for thrml, bit-level numeric matching, wall-clock
as a primary metric.

## 3. Problem definition (must match the kernel exactly)

- Complete graph K2048, random +/-1 couplings J_ij from
  `generate_k2048_complete_pm1_graph(graph_seed=1)`: for i<j,
  Jij_pos = (splitmix32(seed ^ (i<<21) ^ j) & 1); J symmetric, zero diagonal,
  values in {+1,-1} (bit 1 => +1, bit 0 => -1).
- Energy convention: E(s) = -(1/2) sum_i s_i L_i = - sum_{i<j} J_ij s_i s_j,
  with local field L_i = sum_{j!=i} J_ij s_j. Spins s in {+1,-1}.
- Initial spins per replica from `make_initial_spins(spin_seed=2, replica)`
  (splitmix32 bit fill). Both backends start from identical states.
- Success: energy <= target_energy = -67040  (equivalently cut >= 33000).

## 4. RSA dynamics (the algorithm both backends run)

Per step (one step == one single-site attempt == one unit on the matched axis):
1. Pick site j uniformly at random in [0, N).
2. p_plus = sigmoid(2 * beta * L_j)   (heat-bath probability that s_j = +1).
3. Draw coin u ~ U(0,1); new s_j = +1 if u < p_plus else -1.
4. If s_j changed: update energy by 2*s_old*L_j, flip the bit, update all L_i.

Schedule: linear beta anneal beta_start -> beta_end over `stages` stages, each of
`iters_per_stage` steps. total_steps = stages * iters_per_stage.

## 5. thrml-native RSA architecture (`rsa_thrml.py`)

VALIDATED via an N=8 spike (see `spike_n8.py`). The heat-bath update is performed by
thrml's real Gibbs conditional `SpinGibbsConditional`; energy/field bookkeeping is
plain linear algebra (measurement only).

- Represent the couplings as a DENSE matrix `Jb = beta * J`  (shape (N,N), int->float,
  zero diagonal). NO edge-list IsingEBM / IsingSamplingProgram / singleton blocks --
  those are static-index + un-batched and do not scale to dense N=2048.
- Per step (traced random site j, jit/lax.scan friendly):
    w_row = Jb[j]                                  # (N,) = beta*J[j,:]
    interaction = DiscreteEBMInteraction(n_spin=1, weights=w_row[None,:])
    new_sj_bool = SpinGibbsConditional().sample(key,[interaction],[ones(bool)],
                                                [[spins_bool[None,:]]],None,out_sd)
  thrml computes gamma = sum_i (beta*J[j,i]) s_i = beta*L_j  (verified EXACT vs beta*L_j)
  and draws bernoulli(sigmoid(2*gamma)) -- exactly RSA's heat bath.
- Spins stored as thrml convention: bool, True=+1 / False=-1. Read out via 2*b-1.
- Replicas: vmap the per-replica chain over 32 keys/initial-states.
- beta anneal: linear beta_start->beta_end across `stages`; rebuild Jb (or scale a base
  J by the current beta) per stage. total_steps = stages*iters_per_stage; one single-site
  update = one step.
- Energy tracking (measurement, NOT part of the sampler): maintain local fields
  L = J @ s and scalar E incrementally on each accepted flip (delta_E = 2*s_old*L_j;
  L_i -= 2*J_ij*s_old) -- O(N)/step, matching the kernel's bookkeeping. Record best_energy
  and first hit-step (E <= target).

DENSE-GRAPH RISK: RESOLVED. The validated path never builds the 2.1M-edge model; it only
holds the dense (N,N) beta*J matrix and does an O(N) gather+dot per step. The optional
IsingEBM is built only at small N for an energy-convention cross-check.


## 6. Operating point (matched budget)

Headline config (smallest budget with non-zero RSA success):
- stages = 64, iters_per_stage = 16384  -> total_steps = 1,048,576
- beta_start = 0.25, beta_end = 6.0
- repeats = 32, graph_seed = 1, spin_seed = 2, target_energy = -67040
- CUDA reference at this config: ~2/32 success, best_energy ~ -67604.

Optional second point if runtime allows: total_steps = 4,194,304 (stages=128,
iters_per_stage=32768), ~4/32 success.

Note: the kernel's `iters_per_stage` is uint16 (max 65535); larger budgets come
from increasing `stages`.

## 7. Metrics & comparison

Common axis = steps. Both backends report, per matched config:
- success_ratio, successes/repeats
- avg_hit_steps (steps to first reach target; = total_steps if never)
- best_energy_overall
- TTS99 in steps: hit_steps * ln(1-0.99)/ln(1-success_ratio)

Deliverable comparison: side-by-side table (thrml vs CUDA) at the headline config,
plus an energy-vs-step trajectory overlay (mean +/- spread across replicas).
Wall-clock reported only as a caveat (CPU thrml vs GH200 kernel is apples-to-oranges).

Parity verdict: success_ratio and best_energy distributions statistically
indistinguishable between backends at matched budget (small-sample: report counts +
a simple two-proportion check; best_energy via mean/spread overlap).

## 8. Validation gates

- G0 (instance + energy): Python/JAX reproduction of splitmix32, the graph, and the
  initial spins MUST match the CUDA kernel. Airtight cross-check: a 12-line
  `init_energy_check.cpp` that copies the kernel's splitmix32 /
  generate_k2048_complete_pm1_graph / make_initial_spins / initial-energy code and
  prints E0 for (graph_seed=1, spin_seed=2, replica); compare to the Python E0. Hard
  gate before any sampling.
- G1 (smoke): N=2048, tiny budget (e.g. total_steps=4096), confirm thrml builds,
  runs, and per-step cost extrapolates to a tolerable full-run time.
- G2 (parity): full headline-config run, compare metrics.

## 9. Deliverables (in /scratch/hongse/ising/k2000/thrml_rsa/)

1. `rsa_thrml.py`  -- thrml-native RSA sampler + metrics; CLI mirroring kernel flags
   (--repeats --stages --iters-per-stage --beta-start --beta-end --target-energy
   --graph-seed --spin-seed).
2. `compare_rsa.py` -- runs both backends on identical instances, emits the
   side-by-side table + trajectory plot.
3. `README.md` -- parity verdict, which thrml path (edge-list vs dense) was used,
   and the caveats.

## 10. Risks

- Dense-graph cost in thrml -- RESOLVED by the validated dense-matrix +
  SpinGibbsConditional path (no edge-list build). G1 smoke still measures per-step cost.
- Long CPU runtime for ~1M+ steps -- accepted; modest replicas (32), 1-2 budgets.
- Low success ratios (~6%) make success-ratio parity noisy at 32 replicas --
  supplement with best_energy distribution + trajectory overlap, not success alone.

# CR3BP Trajectory-Correction QUBO on Extropic `thrml` (CPU) — Design

**Date:** 2026-06-10
**Author:** hongse (+ Claude)
**Status:** Approved (brainstorming) → implementation

## Goal

Replace the bespoke GPU integer Ising kernel (`libising_int.so` /
`ising_int.cu`) in the CR3BP planar-L1-Lyapunov trajectory-correction-maneuver
(TCM) benchmark with Extropic's `thrml` thermodynamic sampler running on **CPU**.
Success criterion = **drop-in parity**: reuse the entire CR3BP / QUBO / TTS
harness and produce a directly comparable summary (`p_attempt`, best/median J,
TTS) against the GPU run. Matches the "thermodynamic parity is the win"
narrative of the earlier thrml-mask work.

## Background

- `cr3bp_ising.py` builds a linearized targeting model (STM via DOP853),
  encodes the continuous TCM least-squares objective as a binary QUBO
  (`build_qubo`), converts it to a SPIN Ising model, and minimizes it with a
  GPU kernel through `IsingKernelLib`.
- The GPU kernel (`ising_int_kernel`) is a **batched zero-temperature greedy
  single-spin-flip** minimizer over int32-quantized fields. It minimizes
  `E_kernel(s) = -(Σ hᵢ sᵢ + Σ Jᵢⱼ sᵢ sⱼ)`. To make it minimize the *standard*
  dimod Ising objective `E(s)=Σ hs+Σ Jss`, the harness negates coefficients
  (`h_kernel=-h_dimod`, `J_kernel=-J_dimod`) in
  `bqm_to_kernel_int_matrices(negate_for_kernel=True)`.
- For default `--K 4 --bits 6` the QUBO has **N = 2·K·bits = 48** spins and is
  **fully connected** (`Qsym = CᵀHC` is dense).

## Key mapping fact (verified against `thrml/models/ising.py`)

`thrml.models.IsingEBM` energy is

    E(s) = -β ( Σ bᵢ sᵢ + Σ Jᵢⱼ sᵢ sⱼ )   with  P(s) ∝ exp(-E(s))

so its **high-probability states minimize** `-(Σ bs + Σ Jss)` — the *same*
objective the GPU kernel minimizes. Therefore the existing negated mapping
carries over unchanged:

    b_thrml = -h_dimod = h_kernel ,   J_thrml = -J_dimod = J_kernel

We consume the **float** Ising coefficients directly (no int32 quantization, no
`--scale`) plus a temperature β.

## Architecture

New standalone script `cr3bp_thrml.py` that **imports** the CR3BP dynamics,
linearized model, QUBO encoding, decode/objective, and TTS math from
`cr3bp_ising.py` (module-level functions; `main()` guarded by `__main__`, so
import has no side effects). The GPU baseline file is **not modified** and stays
runnable side-by-side. Only the solver class changes:

    IsingKernelLib  →  ThrmlIsingSolver

with the same external contract so the driver loop is structurally identical.

### `ThrmlIsingSolver` contract

- `__init__(N, biases, weights_edges, edges, beta_schedule, sweeps_per_stage)`
  — builds the `SpinNode` list and singleton blocks once.
- `solve_batch(batch_size, seed) -> uint8 (batch_size, N)` — runs one annealed
  batch and returns spins in **node order** (0/1), matching the kernel's output
  contract. The driver then applies the existing `perm` to reorder into
  `enc.names`, `decode_u_from_bits`, `objective_batch`, and takes the min.

### Graph & blocks

Fully-connected ⇒ no two nodes conditionally independent ⇒ **singleton blocks**:
`free_blocks = [Block([nodes[i]]) for i in range(N)]`. One sweep = N sequential
single-spin Gibbs updates (`SpinGibbsConditional`). N=48 ⇒ trivial.

### β-annealing (T=0-greedy → finite-T-sampler bridge)

Energy folds β multiplicatively, so we keep model β=1 and **scale the
coefficients** by βₜ per stage (identical to annealing β):

    for βₜ in geometric(β_lo → β_hi, n_beta):
        ebm     = IsingEBM(nodes, edges, βₜ·b, βₜ·J, beta=1.0)
        program = IsingSamplingProgram(ebm, free_blocks, [])
        repeat sweeps_per_stage × sample_blocks(...)  # carry state_free forward
    keep min objective over post-anneal samples

`sample_blocks(key, state_free, [], program, sampler_state)` is the single-sweep
primitive (returns `(state_free, sampler_state)`); we carry `state_free` across
stages. Auto-derived β defaults from coupling magnitude (β_hi·typical_field≈30
"frozen", β_lo such that β_lo·typical_field≈0.5), geometric ramp, all
CLI-overridable.

### Batching & vmap

`hinton_init(key, ebm, free_blocks, batch_shape=(num_reads,))` seeds `num_reads`
parallel chains; `jax.vmap` the anneal over the (chain, key) batch.

- `--num-reads`  = parallel chains (batch per solve)
- `--num-srt`    = independent anneal restarts per attempt (fresh init+keys);
                   min objective taken over restarts
- `--repeats`    = attempts (for TTS statistics)

`estimate_tts` is reused **unchanged** ⇒ summary table directly comparable to the
GPU run.

## CLI (deltas vs `cr3bp_ising.py`)

- **Removed:** `--lib`, `--scale`, `--iters` (kernel-specific).
- **Added:** `--n-beta 16`, `--sweeps-per-stage 20`, `--beta-lo`, `--beta-hi`
  (auto if omitted).
- **Unchanged:** `--json --row --K --bits --vmax --lam --W --dx/dy/dvx/dvy
  --rtol --atol --num-reads 2000 --num-srt 8 --repeats 50 --confidence
  --target-mode --target-frac --target-abs --seed --quiet`.

## Runtime

`/scratch/hongse/venvs/thrml` venv (thrml 0.1.3, jax 0.10.1). Force CPU:
`JAX_PLATFORMS=cpu`.

## Validation

1. **Sign sanity** — a single fully-frozen anneal reaches J ≪ J_baseline on row
   1593 (confirms the negation/energy convention is right).
2. **Parity** — thrml vs GPU kernel on row 1593, matched batch/restarts; compare
   best/median J and `p_attempt`. Parity = win.
3. **Determinism** — same `--seed` ⇒ same result.

## Non-goals

- Beating the GPU kernel on wall-clock (CPU Gibbs is far slower; parity of
  *solution quality* is the point).
- Modifying `cr3bp_ising.py` or the CUDA kernel.
- Retraining / learning the Ising weights (the QUBO is fixed by the CR3BP
  instance).

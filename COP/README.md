# RSA on the thrml/JAX ecosystem vs the CUDA kernel — K2048 Max-Cut

> [!IMPORTANT]
> This repository was created for **Extropic's Hackathon**, where each team
> (1–4 members) had 4 hours to showcase an application of thermodynamic
> hardware using Extropic's THRML emulator. Given the short 4-hour window,
> the results and analysis are necessarily preliminary. **This project won first place**.

> **Note:** This repository contains only the thrml/JAX side of the comparison.
> The hand-written GH200 CUDA kernel (`gh200_k2048_rsa_rwa.cu`) and compiled
> binaries are not included; CUDA numbers in the tables below were produced on
> the original GH200 machine. Install dependencies with
> `pip install -r requirements.txt`.

Same RSA algorithm (single-site heat-bath annealing), same instance (graph_seed=1,
spin_seed=2; identical initial spins — verified bit-exact by the G0 C++/Python gate),
same matched step budget, compared on the hardware-agnostic STEP axis (one single-site
update = one step). Two thrml/JAX-side variants were built:

- `rsa_thrml.py`     — thrml-NATIVE FLOAT: uses thrml's `SpinGibbsConditional`
                       (dense beta*J row -> gamma = beta*L_j, continuous jax sigmoid,
                       jax PRNG). The "runs on a thermodynamic sampler" variant.
- `rsa_bitfaithful.py` — BIT-FAITHFUL: replicates the kernel's exact fixed-point
                       pipeline (Q8.8 beta, Q0.32 sigmoid LUT + Z_CLAMPQ clamp,
                       splitmix32 site/coin RNG, integer field/energy updates) in JAX
                       (x64). thrml's float sampler cannot be bit-faithful, so it is
                       intentionally not used here.

CUDA reference: `../gh200_k2048_rsa_rwa` (GH200). thrml/JAX ran on CPU jax.

## Headline matched config
N=2048, stages=64, iters_per_stage=16384, total_steps=1,048,576, beta 0.25->6.0,
repeats=32, target_energy=-67040.

![Energy trajectory: thrml vs CUDA on the matched step axis](results_headline/trajectory.png)

## Result 1 — thrml-native float: NOT clean parity (modestly worse)

| metric        | thrml-float | cuda    |
|---------------|-------------|---------|
| successes     | 0           | 2       |
| success_ratio | 0.0000      | 0.0625  |
| best_energy   | -66360      | -67604  |
| avg_hit_steps | 1048576     | 1022859 |

thrml-native float RSA is the same algorithm but came out modestly worse: 0/32 vs 2/32
successes, best-of-32 energy ~1244 (1.8%) higher, falling just short of the target.
Causes: (a) small-sample noise (P(0 successes in 32) ~ 0.14 at the kernel's ~6% rate),
and (b) the deliberately-unmatched arithmetic — the kernel clamps the heat-bath argument
2*beta*L at 8 (Z_CLAMPQ), keeping a ~3.4e-4 acceptance FLOOR (a little late-stage escape)
that the unclamped float sigmoid lacks; plus a different PRNG stream.

## Result 2 — bit-faithful: EXACT equivalence (bit-for-bit)

| metric        | bitfaithful JAX | cuda gpu   | cuda cpu   |
|---------------|-----------------|------------|------------|
| successes     | 2               | 2          | 2          |
| success_ratio | 0.062500        | 0.062500   | 0.062500   |
| avg_hit_steps | 1022858.69      | 1022858.69 | 1022858.69 |
| best_energy   | -67604          | -67604     | -67604     |

avg_hit_steps matches to the DECIMAL across all 32 replicas => every per-replica hit-step
is identical => the RNG streams and integer dynamics are bit-identical, not merely close.
The kernel's own GPU and CPU backends also agree exactly, and JAX matches both.

Validation: `rsa_bitfaithful.py`'s per-step integer energy trajectory matches an
independent C++ tracer (`rsa_trace.cpp`, the kernel's RSA loop in scalar C++) EXACTLY over
thousands of steps across multiple configs/replicas (test_bitfaithful.py).

Wall time (incidental, not the comparison axis): bit-faithful JAX 12.2 s vs thrml-float
110.8 s vs CUDA 1.5 s (GH200). The float version's cost was thrml's per-step
SpinGibbsConditional dispatch, not the arithmetic.

## Verdict
- The K2048 RSA algorithm, instance, energy convention, and per-step dynamics are
  reproduced EXACTLY in JAX (bit-faithful) — strict equivalence with the CUDA kernel.
- thrml's NATIVE float sampler yields a qualitatively-faithful RSA that is statistically
  close but not identical, and at this rare-event operating point lands slightly worse.
  The gap is fully explained by the float-vs-fixed-point choice (continuous unclamped
  sigmoid + different PRNG) — confirmed because the bit-faithful fixed-point port closes
  it to zero.

## Files
- ref_instance.py / init_energy_check.cpp / test_ref_instance.py  — instance + G0 gate
- thrml_update.py / test_thrml_update.py                          — thrml heat-bath primitive
- rsa_thrml.py / test_rsa_thrml.py                                — thrml-native float RSA driver
- rsa_bitfaithful.py / rsa_trace.cpp / test_bitfaithful.py        — bit-faithful RSA + tracer
- compare_rsa.py                                                  — thrml-float vs CUDA harness
- results_headline/                                              — float-vs-cuda table + trajectory plot

## Reproduce
    source /scratch/hongse/venvs/thrml/bin/activate
    # bit-faithful equivalence:
    python3 rsa_bitfaithful.py --repeats 32 --stages 64 --iters-per-stage 16384 \
      --beta-start 0.25 --beta-end 6.0 --target-energy -67040
    CUDA_VISIBLE_DEVICES=1 ../gh200_k2048_rsa_rwa --mode rsa --backend gpu --repeats 32 \
      --stages 64 --iters-per-stage 16384 --beta-start 0.25 --beta-end 6.0 --target-energy -67040
    # thrml-native float vs cuda:
    python3 compare_rsa.py --repeats 32 --stages 64 --iters-per-stage 16384 --outdir results_headline

## Citation

```bibtex
@misc{hong2026thrml,
  author       = {Seungki Hong},
  title        = {Applications of Thermodynamic Computing with Extropic THRML},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/skethz/extropic-thrml-applications}},
}
```

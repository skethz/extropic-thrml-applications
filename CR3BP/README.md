# Spacecraft Trajectory Correction Maneuver (TCM) on Extropic `thrml` — CR3BP L1 Lyapunov

> [!IMPORTANT]
> This repository was created for **Extropic's Hackathon**, where each team
> (1–4 members) had 4 hours to showcase an application of thermodynamic
> hardware using Extropic's THRML emulator. Given the short 4-hour window,
> the results and analysis are necessarily preliminary. **This project won first place**.

Trajectory-correction maneuver optimization for a planar Lyapunov orbit about
the Earth–Moon L1 Lagrange point (circular restricted three-body problem),
encoded as a QUBO/Ising problem and solved with
[Extropic's `thrml`](https://github.com/extropic-ai/thrml) thermodynamic
sampler on CPU.

> **Note:** Only the thrml solver path is published here. The original GPU
> baseline used a hand-written CUDA kernel (`ising_int.cu` /
> `libising_int.so`), which is not included. `cr3bp_ising.py` is kept because
> it provides the CR3BP dynamics, STM propagation, QUBO encoding, decoding,
> and TTS math that `cr3bp_thrml.py` imports — its own CUDA `main()` cannot
> run without the excluded library.

## Problem

- **Reference orbit:** planar L1 Lyapunov orbit from NASA JPL's Poincaré
  Catalog of Periodic Orbits (row 1593 of `em_L1_lyapunov.json`;
  μ = 1.21506e-2, period ≈ 24.8 days).
- **Dynamics:** CR3BP equations of motion propagated with DOP853; the state
  transition matrix (STM) is propagated in parallel to linearize the
  sensitivity of the terminal state to impulses.
- **Correction:** K = 16 impulsive Δv maneuvers (2 components each), each
  component encoded with 8 bits → a fully-connected Ising problem over
  **N = 256 spins**. The quadratic targeting objective J(u) trades terminal
  position/velocity error against total maneuver effort.

## thrml solver (`cr3bp_thrml.py`)

Faithful drop-in for the GPU kernel's architecture — `num_reads` independent
chains, `iters` random single-site updates per chain — with the
zero-temperature greedy flip replaced by thrml's `SpinGibbsConditional`
heat-bath update, β annealed hot→cold so the cold tail matches greedy descent.
Implemented as `jax.lax.scan` over steps and `jax.vmap` over chains; per-node
biases enter through a clamped ghost spin. The mapping was verified against
`thrml/models/ising.py` (see `docs/2026-06-10-cr3bp-thrml-design.md`).

## Results (N = 256, 10 attempts, 2000 chains × 8 calls × 20k updates)

From `results/thrml_tts256.log` and `results/traj.log`:

| metric                         | value          |
|--------------------------------|----------------|
| p_attempt (success rate)       | 1.00 (10/10)   |
| TTS@0.99 (CPU)                 | 182.4 s        |
| best objective J               | 5.15e-07       |
| baseline J (no correction)     | 1.01e-05       |
| target J                       | 1.01e-06       |

Physically: a 0.38 km initial perturbation grows to a **573 km** terminal
error on the unstable L1 orbit; the thrml-found maneuvers (16 impulses,
26.3 m/s total Δv) cut it to **~51 km**.

![Trajectory correction: reference vs uncorrected vs thrml-corrected](results/traj_correction.png)

## Files

- `cr3bp_thrml.py`     — thrml TCM solver + TTS benchmark driver
- `cr3bp_ising.py`     — CR3BP dynamics, STM, QUBO encoding/decoding (imported)
- `traj_plot.py`       — reference vs perturbed vs thrml-corrected trajectory plot
- `generate_ppt.py`    — builds the comparison slide deck from the logs
- `em_L1_lyapunov.json`— JPL periodic-orbit catalog data (L1 Lyapunov family)
- `docs/`              — approved design doc for the thrml port
- `results/`           — run logs and the trajectory-correction figure

## Run

```
pip install -r requirements.txt
JAX_PLATFORMS=cpu python3 -u cr3bp_thrml.py --json em_L1_lyapunov.json --row 1593 \
  --K 16 --bits 8 --iters 20000 \
  --repeats 10 --num-reads 2000 --num-srt 8 \
  --target-mode frac --target-frac 0.1
python3 traj_plot.py   # writes traj_correction.png
```

## Data attribution

The reference orbit data in `em_L1_lyapunov.json` is drawn from the Poincaré
Catalog of Periodic Orbits hosted by NASA's Jet Propulsion Laboratory (JPL),
managed by the California Institute of Technology (Caltech).

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

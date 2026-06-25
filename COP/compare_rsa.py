import argparse, subprocess, re, math, os
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

CUDA_BIN = "/scratch/hongse/ising/k2000/gh200_k2048_rsa_rwa"
THRML = "/scratch/hongse/ising/k2000/thrml_rsa/rsa_thrml.py"

def tts99_steps(total, p):
    if p <= 0: return float("inf")
    if p >= 1 - 1e-15: return float(total)
    return total * math.log(0.01) / math.log(1 - p)

def _parse_common(out):
    def _get(pattern, cast, name):
        m = re.search(pattern, out)
        if m is None:
            raise RuntimeError(f"_parse_common: '{name}' not found in backend output:\n{out[:400]}")
        return cast(m.group(1))
    succ    = _get(r"successes=(\d+)",             int,   "successes")
    p       = _get(r"success_ratio=([\d.]+)",      float, "success_ratio")
    avg_hit = _get(r"avg_hit_steps=([\d.]+)",      float, "avg_hit_steps")
    best    = _get(r"best_energy_overall=(-?\d+)", int,   "best_energy_overall")
    return succ, p, avg_hit, best

def run_cuda(repeats, stages, ips, b0, b1, target):
    cmd = [CUDA_BIN, "--mode", "rsa", "--backend", "gpu", "--repeats", str(repeats),
           "--stages", str(stages), "--iters-per-stage", str(ips),
           "--beta-start", str(b0), "--beta-end", str(b1), "--target-energy", str(target)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="1")
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"CUDA run failed:\n{r.stdout}\n{r.stderr}")
    succ, p, avg_hit, best = _parse_common(r.stdout)
    total = stages * ips
    return dict(backend="cuda", successes=succ, p=p, avg_hit=avg_hit, best=best,
                tts99=tts99_steps(total, p), total=total, raw=r.stdout.strip())

def run_thrml(repeats, stages, ips, b0, b1, target, traj_path):
    cmd = ["python3", THRML, "--repeats", str(repeats), "--stages", str(stages),
           "--iters-per-stage", str(ips), "--beta-start", str(b0), "--beta-end", str(b1),
           "--target-energy", str(target), "--traj-out", traj_path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"thrml run failed:\n{r.stdout}\n{r.stderr}")
    succ, p, avg_hit, best = _parse_common(r.stdout)
    total = stages * ips
    return dict(backend="thrml", successes=succ, p=p, avg_hit=avg_hit, best=best,
                tts99=tts99_steps(total, p), total=total, raw=r.stdout.strip())

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
    print("running CUDA backend ..."); c = run_cuda(a.repeats, a.stages, a.iters_per_stage, a.beta_start, a.beta_end, a.target_energy)
    print("running thrml backend (this is the long one) ..."); t = run_thrml(a.repeats, a.stages, a.iters_per_stage, a.beta_start, a.beta_end, a.target_energy, traj)
    total = t["total"]
    hdr = f"{'metric':<20}{'thrml':>16}{'cuda':>16}"
    rows = [("successes", t["successes"], c["successes"]),
            ("success_ratio", f"{t['p']:.4f}", f"{c['p']:.4f}"),
            ("avg_hit_steps", f"{t['avg_hit']:.0f}", f"{c['avg_hit']:.0f}"),
            ("best_energy", t["best"], c["best"]),
            ("TTS99_steps", f"{t['tts99']:.0f}", f"{c['tts99']:.0f}")]
    lines = [f"matched config: N=2048 stages={a.stages} iters_per_stage={a.iters_per_stage} "
             f"total_steps={total} beta={a.beta_start}->{a.beta_end} repeats={a.repeats} "
             f"target={a.target_energy}", "", hdr, "-"*52]
    lines += [f"{m:<20}{str(tv):>16}{str(cv):>16}" for m, tv, cv in rows]
    lines += ["", "[raw thrml] " + t["raw"].replace(chr(10), " | "),
                  "[raw cuda ] " + c["raw"].replace(chr(10), " | ")]
    table = "\n".join(lines)
    print("\n" + table)
    with open(os.path.join(a.outdir, "table.txt"), "w") as f: f.write(table + "\n")
    # trajectory plot: thrml traj is DOWNSAMPLED -> (repeats, n_records); map record idx -> step
    Et = np.load(traj)                                  # (repeats, n_records)
    n_rec = Et.shape[1]
    xs = np.linspace(total / n_rec, total, n_rec)        # step number for each record
    plt.figure(figsize=(8, 5))
    plt.plot(xs, Et.mean(0), color="C0", label="thrml mean E")
    plt.fill_between(xs, Et.min(0), Et.max(0), color="C0", alpha=0.2, label="thrml min/max")
    plt.axhline(c["best"], color="C1", ls="--", label=f"cuda best E={c['best']}")
    plt.axhline(a.target_energy, color="k", ls=":", label=f"target={a.target_energy}")
    plt.xlabel("RSA step"); plt.ylabel("energy"); plt.legend(); plt.title("RSA: thrml vs CUDA (K2048)")
    plt.tight_layout(); plt.savefig(os.path.join(a.outdir, "trajectory.png"), dpi=120)
    print(f"\nwrote {a.outdir}/table.txt and {a.outdir}/trajectory.png")

if __name__ == "__main__":
    main()

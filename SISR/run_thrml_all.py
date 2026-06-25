"""Urban100 x2/x4 super-resolution: GPU Ising kernel baseline vs thrml CPU solver.

Same sparse-coding SR pipeline as run_int_all.py. The ONLY thing that differs
between the two methods is how the per-patch Ising problem is solved:

  - method 'gpu'  : original CUDA kernel solve_batch (iters=1 greedy single flip)
  - method 'thrml': Extropic thrml CPU block-Gibbs to a near-ground state

Reconstruction is identical for both: the existing reconstruct_gpu kernel,
fed via upload_s for the thrml path (libising_int_thrml.so).

Outputs:  results_<method>/Urban100/X<scale>/img_*.png  (+ summary.txt)
Run in the thrml venv with CUDA visible (GPU used for solve_batch + reconstruct).
"""
import os, sys, time, ctypes, glob, argparse
import numpy as np
import cv2
from numpy.ctypeslib import ndpointer

ISING = "/scratch/hongse/ising"
os.chdir(ISING)
sys.path.insert(0, os.path.join(ISING, "thrml_sr"))

BENCH_ROOT = "datasets/benchmark"
LIB_PATH = "./libising_int_thrml.so"
DATASET = "Urban100"  # overridable via --dataset
GPU_ITERS = 1          # CUDA kernel spin-flip proposals (overridable via --gpu_iters)
N_SPINS = 64
MU = 0.001
LAMBDA = 0.01
OVERLAP = 2
THRML_BETA = 2000.0
THRML_WARMUP = 30


class IsingKernel:
    """Thin ctypes wrapper around libising_int_thrml.so (adds upload_s)."""
    def __init__(self, lib_path, N, max_batch):
        self.lib = ctypes.CDLL(lib_path); self.N = N
        self.lib.init_gpu.argtypes = [ctypes.c_int, ctypes.c_int]
        self.lib.upload_J.argtypes = [ndpointer(ctypes.c_int32, flags="C_CONTIGUOUS")]
        self.lib.solve_batch.argtypes = [
            ndpointer(ctypes.c_int32, flags="C_CONTIGUOUS"),
            ndpointer(ctypes.c_uint8, flags="C_CONTIGUOUS"),
            ctypes.c_int, ctypes.c_int, ctypes.c_uint64]
        self.lib.upload_s.argtypes = [
            ndpointer(ctypes.c_uint8, flags="C_CONTIGUOUS"), ctypes.c_int]
        self.lib.reconstruct_gpu.argtypes = [
            ndpointer(ctypes.c_float, flags="F_CONTIGUOUS"),
            ndpointer(ctypes.c_float, flags="F_CONTIGUOUS"),
            ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
            ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
            ndpointer(ctypes.c_int,   flags="C_CONTIGUOUS"),
            ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
            ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self.lib.free_gpu.argtypes = []
        self.lib.init_gpu(N, max_batch)

    def set_problem_matrix(self, Q):
        SF = 10000.0
        self.Qd = np.diag(Q).astype(np.float32)
        Qoff = Q.copy(); np.fill_diagonal(Qoff, 0)
        self.Qrs = np.sum(Qoff, axis=1).astype(np.float32)
        J = -0.25 * (Q + Q.T) / 2.0; np.fill_diagonal(J, 0)
        self.J_float = J.astype(np.float32)
        self.lib.upload_J((J * SF).astype(np.int32)); self.SF = SF

    def biases_int(self, B):
        h = -1.0 * (B / 2.0 + self.Qd / 2.0 + self.Qrs / 2.0)
        return (h * self.SF).astype(np.int32)

    def biases_float(self, B):
        return (-1.0 * (B / 2.0 + self.Qd / 2.0 + self.Qrs / 2.0)).astype(np.float32)

    def solve_gpu(self, h_int, iters=1):
        b = h_int.shape[0]; s = np.zeros((b, self.N), dtype=np.uint8)
        self.lib.solve_batch(np.ascontiguousarray(h_int), s, b, iters, 1234)
        return s

    def upload_s(self, s):
        s = np.ascontiguousarray(s, dtype=np.uint8)
        self.lib.upload_s(s, s.shape[0])

    def reconstruct(self, D_l, D_h, Y, means, coords, h_hr, w_hr, scale, p_h):
        b = Y.shape[0]; M = D_l.shape[0]; P_h2 = D_h.shape[0]
        img = np.zeros((h_hr, w_hr), dtype=np.float32)
        w = np.zeros((h_hr, w_hr), dtype=np.float32)
        self.lib.reconstruct_gpu(
            np.asfortranarray(D_l), np.asfortranarray(D_h),
            np.ascontiguousarray(Y), np.ascontiguousarray(means),
            np.ascontiguousarray(coords).astype(np.int32),
            img, w, M, P_h2, p_h, w_hr, h_hr, scale, b)
        return img, w

    def close(self):
        self.lib.free_gpu()


def patch_p_l(scale):
    return 5 if scale == 2 else 3


def extract_patches(img_y, scale, p_l, step):
    h_lr, w_lr = img_y.shape
    feat_up = cv2.resize(img_y, (w_lr * 2, h_lr * 2), interpolation=cv2.INTER_CUBIC)
    grads = [cv2.filter2D(feat_up, -1, f) for f in [
        np.array([[-1, 0, 1]]), np.array([[-1], [0], [1]]),
        np.array([[1, 0, -2, 0, 1]]), np.array([[1], [0], [-2], [0], [1]])]]
    p_f = p_l * 2
    coords, features, means = [], [], []
    for r in range(0, h_lr - p_l + 1, step):
        for c in range(0, w_lr - p_l + 1, step):
            r2, c2 = r * 2, c * 2
            features.append(np.concatenate(
                [g[r2:r2 + p_f, c2:c2 + p_f].ravel() for g in grads]))
            coords.append([r, c])
            means.append(np.mean(img_y[r:r + p_l, c:c + p_l]))
    return (np.array(features, dtype=np.float32),
            np.array(coords), np.array(means, dtype=np.float32))


def run(scale, method, thrml_solver=None, limit=None):
    p_l = patch_p_l(scale); p_h = p_l * scale
    step = max(1, p_l - OVERLAP)
    lr_dir = os.path.join(BENCH_ROOT, DATASET, "LR_bicubic", f"X{scale}")
    tag = method if method != "gpu" else f"gpu_it{GPU_ITERS}"
    if method == "thrml":
        tag = f"thrml_b{int(thrml_solver.beta)}"
    out_dir = os.path.join(f"results_{tag}", DATASET, f"X{scale}")
    os.makedirs(out_dir, exist_ok=True)

    D_h = np.load(f"dictionaries_x{scale}_{N_SPINS//2}_bicubic/D_h.npy")
    D_l = np.load(f"dictionaries_x{scale}_{N_SPINS//2}_bicubic/D_l.npy")
    D_l_aug = np.hstack([D_l, -D_l]).astype(np.float32)
    D_h_aug = np.hstack([D_h, -D_h]).astype(np.float32)
    Q = (MU ** 2) * (D_l_aug.T @ D_l_aug)
    J_float = -0.25 * (Q + Q.T) / 2.0; np.fill_diagonal(J_float, 0)
    if method == "thrml":
        thrml_solver.set_problem_matrix(J_float)  # compiles once per scale

    img_files = sorted(glob.glob(os.path.join(lr_dir, "*.png")))
    if limit:
        img_files = img_files[:limit]
    runtimes_solve, runtimes_total = [], []
    print(f"\n=== {DATASET} X{scale} | method={method} | "
          f"p_l={p_l} step={step} imgs={len(img_files)} ===", flush=True)

    for img_path in img_files:
        name = os.path.basename(img_path)
        img_in = cv2.imread(img_path)
        if img_in is None:
            continue
        img_yuv = cv2.cvtColor(img_in, cv2.COLOR_BGR2YCrCb)
        img_y = img_yuv[:, :, 0].astype(np.float32) / 255.0
        h_lr, w_lr = img_y.shape
        h_hr, w_hr = h_lr * scale, w_lr * scale

        Y, coords, means = extract_patches(img_y, scale, p_l, step)
        if len(coords) == 0:
            continue
        B = (-2 * MU) * (Y @ D_l_aug) + (LAMBDA * MU)

        t0 = time.time()
        ker = IsingKernel(LIB_PATH, N_SPINS, len(coords))
        ker.set_problem_matrix(Q)
        if method == "gpu":
            h_int = ker.biases_int(B)
            ts = time.time(); ker.solve_gpu(h_int, iters=GPU_ITERS); t_solve = time.time() - ts
        else:
            h_float = ker.biases_float(B)
            ts = time.time(); s = thrml_solver.solve_batch(h_float); t_solve = time.time() - ts
            ker.upload_s(s)
        img_out, w_out = ker.reconstruct(
            D_l_aug, D_h_aug, Y, means, coords, h_hr, w_hr, scale, p_h)
        ker.close()
        t_total = time.time() - t0
        runtimes_solve.append(t_solve); runtimes_total.append(t_total)

        mask = w_out > 0
        img_out[mask] /= w_out[mask]
        img_bicubic = cv2.resize(img_y, (w_hr, h_hr), interpolation=cv2.INTER_CUBIC)
        img_out[~mask] = img_bicubic[~mask]
        final_y = np.clip(img_out * 255, 0, 255).astype(np.uint8)
        img_cr = cv2.resize(img_yuv[:, :, 1], (w_hr, h_hr), interpolation=cv2.INTER_CUBIC)
        img_cb = cv2.resize(img_yuv[:, :, 2], (w_hr, h_hr), interpolation=cv2.INTER_CUBIC)
        final_bgr = cv2.cvtColor(cv2.merge([final_y, img_cr, img_cb]), cv2.COLOR_YCrCb2BGR)
        cv2.imwrite(os.path.join(out_dir, name), final_bgr)
        print(f"  {name} patches={len(coords)} solve={t_solve:.3f}s total={t_total:.3f}s",
              flush=True)

    if runtimes_total:
        with open(os.path.join(out_dir, "summary.txt"), "w") as f:
            f.write(f"method={method} dataset={DATASET} scale={scale}\n")
            f.write(f"images={len(runtimes_total)}\n")
            f.write(f"avg_solve_s={np.mean(runtimes_solve):.4f}\n")
            f.write(f"avg_total_s={np.mean(runtimes_total):.4f}\n")
        print(f"  [summary] avg_solve={np.mean(runtimes_solve):.3f}s "
              f"avg_total={np.mean(runtimes_total):.3f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", type=int, nargs="+", default=[2, 4])
    ap.add_argument("--methods", nargs="+", default=["gpu", "thrml"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dataset", default="Urban100")
    ap.add_argument("--gpu_iters", type=int, default=1)
    ap.add_argument("--thrml_beta", type=float, default=THRML_BETA)
    ap.add_argument("--thrml_warmup", type=int, default=THRML_WARMUP)
    args = ap.parse_args()
    global DATASET, GPU_ITERS
    DATASET = args.dataset
    GPU_ITERS = args.gpu_iters

    thrml_solver = None
    if "thrml" in args.methods:
        from thrml_solver import ThrmlIsingSolver
        thrml_solver = ThrmlIsingSolver(N_SPINS, beta=args.thrml_beta,
                                        n_warmup=args.thrml_warmup, init="down")

    for scale in args.scales:
        for method in args.methods:
            run(scale, method, thrml_solver, limit=args.limit)


if __name__ == "__main__":
    main()

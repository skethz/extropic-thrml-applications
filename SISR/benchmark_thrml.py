"""De-risk + validate the thrml Ising solver against the GPU kernel on ONE real
image (img_024 x4), and benchmark CPU throughput at several batch sizes.

Run in the thrml venv (CPU). Compares the Ising objective (energy) reached by
the GPU greedy kernel vs thrml block-Gibbs, and times the thrml solve.
"""
import os, time, ctypes
import numpy as np
import cv2
from numpy.ctypeslib import ndpointer

ISING = "/scratch/hongse/ising"
os.chdir(ISING)
from thrml_solver import ThrmlIsingSolver  # noqa: E402

SCALE = 4
N = 64
MU = 0.001
LAMBDA = 0.01
IMG = "datasets/benchmark/Urban100/LR_bicubic/X4/img_024.png"


def extract(img_path, scale, p_l, step):
    img_in = cv2.imread(img_path)
    img_y = cv2.cvtColor(img_in, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32) / 255.0
    h_lr, w_lr = img_y.shape
    feat_up = cv2.resize(img_y, (w_lr * 2, h_lr * 2), interpolation=cv2.INTER_CUBIC)
    grads = [cv2.filter2D(feat_up, -1, f) for f in [
        np.array([[-1, 0, 1]]), np.array([[-1], [0], [1]]),
        np.array([[1, 0, -2, 0, 1]]), np.array([[1], [0], [-2], [0], [1]])]]
    p_f = p_l * 2
    features = []
    for r in range(0, h_lr - p_l + 1, step):
        for c in range(0, w_lr - p_l + 1, step):
            r2, c2 = r * 2, c * 2
            features.append(np.concatenate(
                [g[r2:r2 + p_f, c2:c2 + p_f].ravel() for g in grads]))
    return np.array(features, dtype=np.float32)


# ---- GPU kernel reference ----
class GPU:
    def __init__(self, lib, N, B):
        self.lib = ctypes.CDLL(lib); self.N = N
        self.lib.init_gpu.argtypes = [ctypes.c_int, ctypes.c_int]
        self.lib.upload_J.argtypes = [ndpointer(ctypes.c_int32, flags="C_CONTIGUOUS")]
        self.lib.solve_batch.argtypes = [
            ndpointer(ctypes.c_int32, flags="C_CONTIGUOUS"),
            ndpointer(ctypes.c_uint8, flags="C_CONTIGUOUS"),
            ctypes.c_int, ctypes.c_int, ctypes.c_uint64]
        self.lib.free_gpu.argtypes = []
        self.lib.init_gpu(N, B)

    def set_J(self, Q):
        SF = 10000.0
        self.Qd = np.diag(Q).astype(np.float32)
        Qoff = Q.copy(); np.fill_diagonal(Qoff, 0)
        self.Qrs = np.sum(Qoff, axis=1).astype(np.float32)
        J = -0.25 * (Q + Q.T) / 2.0; np.fill_diagonal(J, 0)
        self.lib.upload_J((J * SF).astype(np.int32)); self.SF = SF

    def biases(self, B):
        h = -1.0 * (B / 2.0 + self.Qd / 2.0 + self.Qrs / 2.0)
        return (h * self.SF).astype(np.int32), h

    def solve(self, h_int, iters=1):
        b = h_int.shape[0]; s = np.zeros((b, self.N), dtype=np.uint8)
        self.lib.solve_batch(np.ascontiguousarray(h_int), s, b, iters, 1234)
        return s


def main():
    p_l = 3; step = 1
    Y = extract(IMG, SCALE, p_l, step)
    print(f"patches={Y.shape[0]} feat_dim={Y.shape[1]}")

    D_l = np.load(f"dictionaries_x{SCALE}_{N//2}_bicubic/D_l.npy")
    D_l_aug = np.hstack([D_l, -D_l]).astype(np.float32)
    Q = (MU ** 2) * (D_l_aug.T @ D_l_aug)
    B = (-2 * MU) * (Y @ D_l_aug) + (LAMBDA * MU)

    gpu = GPU("./libising_int.so", N, Y.shape[0])
    gpu.set_J(Q)
    h_int, h_float = gpu.biases(B)
    J_float = -0.25 * (Q + Q.T) / 2.0; np.fill_diagonal(J_float, 0)

    # GPU reference (iters=1, as in run_int_all.py)
    s_gpu = gpu.solve(h_int, iters=1)
    gpu_active = s_gpu.sum(axis=1)
    print(f"\n[GPU iters=1] mean active spins/patch = {gpu_active.mean():.3f}, "
          f"patches with >=1 atom = {(gpu_active>0).mean()*100:.1f}%")

    # thrml energy helper uses J_float, h_float
    solver_probe = ThrmlIsingSolver(N, beta=1.0)
    solver_probe.set_problem_matrix(J_float)
    e_gpu = solver_probe.energy(h_float, s_gpu).mean()
    e_down = solver_probe.energy(h_float, np.zeros_like(s_gpu)).mean()
    print(f"[energy] all-down = {e_down:.6e}   GPU = {e_gpu:.6e}")

    # ---- thrml configs ----
    for beta, warm, init in [(2000.0, 1, "down"), (2000.0, 30, "down"),
                             (2000.0, 60, "down"), (5000.0, 60, "hinton")]:
        s = ThrmlIsingSolver(N, beta=beta, n_warmup=warm, init=init)
        s.set_problem_matrix(J_float)
        t0 = time.time(); _ = s.solve_batch(h_float[:64]); t_compile = time.time() - t0
        t0 = time.time(); out = s.solve_batch(h_float); t_run = time.time() - t0
        e = s.energy(h_float, out).mean()
        act = out.sum(axis=1)
        agree = (out == s_gpu).mean()
        print(f"[thrml beta={beta:>6} warm={warm:>3} init={init:>6}] "
              f"E={e:.6e}  active={act.mean():.3f}  agree_gpu={agree*100:.1f}%  "
              f"compile={t_compile:.1f}s  run({Y.shape[0]}p)={t_run:.2f}s")


if __name__ == "__main__":
    main()

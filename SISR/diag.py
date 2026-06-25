"""Why does converged-CUDA beat high-beta thrml? Compare spins/energy directly."""
import os, ctypes
import numpy as np, cv2
from numpy.ctypeslib import ndpointer
os.chdir("/scratch/hongse/ising")
import sys; sys.path.insert(0, "thrml_sr")
from thrml_solver import ThrmlIsingSolver

SCALE, N, MU, LAM = 2, 64, 0.001, 0.01
IMG = "datasets/benchmark/Set5/LR_bicubic/X2/img_001.png"
p_l, step = 5, 3

img_y = cv2.cvtColor(cv2.imread(IMG), cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)/255.0
h_lr, w_lr = img_y.shape
feat_up = cv2.resize(img_y, (w_lr*2, h_lr*2), interpolation=cv2.INTER_CUBIC)
grads = [cv2.filter2D(feat_up, -1, f) for f in [
    np.array([[-1,0,1]]), np.array([[-1],[0],[1]]),
    np.array([[1,0,-2,0,1]]), np.array([[1],[0],[-2],[0],[1]])]]
p_f = p_l*2
feats = []
for r in range(0, h_lr-p_l+1, step):
    for c in range(0, w_lr-p_l+1, step):
        r2, c2 = r*2, c*2
        feats.append(np.concatenate([g[r2:r2+p_f, c2:c2+p_f].ravel() for g in grads]))
Y = np.array(feats, dtype=np.float32)

D_l = np.load(f"dictionaries_x{SCALE}_{N//2}_bicubic/D_l.npy")
D_l_aug = np.hstack([D_l, -D_l]).astype(np.float32)
Q = (MU**2)*(D_l_aug.T @ D_l_aug)
Qd = np.diag(Q).astype(np.float32); Qoff = Q.copy(); np.fill_diagonal(Qoff, 0)
Qrs = Qoff.sum(1).astype(np.float32)
Jf = -0.25*(Q+Q.T)/2.0; np.fill_diagonal(Jf, 0)
B = (-2*MU)*(Y @ D_l_aug) + (LAM*MU)
h_float = (-1.0*(B/2.0 + Qd/2.0 + Qrs/2.0)).astype(np.float32)

print(f"patches={Y.shape[0]}  |J| mean={np.abs(Jf).mean():.3e}  |h| mean={np.abs(h_float).mean():.3e}")
print(f"coupling/field ratio (max |J·1| / |h|) ~ {np.abs(Jf).sum(1).mean()/np.abs(h_float).mean():.4f}")

# CUDA kernel
lib = ctypes.CDLL("./libising_int_thrml.so")
lib.init_gpu.argtypes=[ctypes.c_int,ctypes.c_int]
lib.upload_J.argtypes=[ndpointer(ctypes.c_int32,flags="C_CONTIGUOUS")]
lib.solve_batch.argtypes=[ndpointer(ctypes.c_int32,flags="C_CONTIGUOUS"),
    ndpointer(ctypes.c_uint8,flags="C_CONTIGUOUS"),ctypes.c_int,ctypes.c_int,ctypes.c_uint64]
lib.init_gpu(N, Y.shape[0])
SF=10000.0
lib.upload_J((Jf*SF).astype(np.int32))
h_int=(h_float*SF).astype(np.int32)
def cuda(iters):
    s=np.zeros((Y.shape[0],N),dtype=np.uint8)
    lib.solve_batch(np.ascontiguousarray(h_int),s,Y.shape[0],iters,1234); return s

solver_e = ThrmlIsingSolver(N, beta=1.0); solver_e.set_problem_matrix(Jf)
def E(s): return solver_e.energy(h_float, s).mean()

s_sign = (h_float > 0).astype(np.uint8)            # exact sign(h) (coupling ignored)
s_c1 = cuda(1); s_c1024 = cuda(1024)
th = ThrmlIsingSolver(N, beta=200000.0, n_warmup=60, init="down"); th.set_problem_matrix(Jf)
s_th = th.solve_batch(h_float)

def firstatom(s):  # index of first active spin per patch, -1 if none
    idx = np.argmax(s>0, axis=1); has = (s>0).any(1)
    return np.where(has, idx, -1)

for name, s in [("sign(h)", s_sign), ("CUDA it=1", s_c1),
                ("CUDA it=1024", s_c1024), ("thrml b2e5", s_th)]:
    fa = firstatom(s)
    print(f"{name:14s} E={E(s):.6e}  active/patch={s.sum(1).mean():6.2f}  "
          f"patches_with_atom={(fa>=0).mean()*100:5.1f}%  "
          f"agree_vs_sign(h)={(s==s_sign).mean()*100:5.1f}%  "
          f"firstatom_eq_sign={np.mean(fa==firstatom(s_sign))*100:5.1f}%")

"""
Reproduces the CUDA kernel's exact K2048 instance.
The kernel uses graph_seed=1 and spin_seed=2.
Spins are ±1.
Energy convention: E = -0.5 * s^T J s = -sum_{i<j} J_ij s_i s_j
"""
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
    assert n % 64 == 0, "n must be a multiple of 64"
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
    s64 = s.astype(np.int64)
    return -0.5 * float(s64 @ (J.astype(np.int64) @ s64))

if __name__ == "__main__":
    J = make_graph(1)
    for r in range(4):
        s = make_init_spins(2, r)
        print(f"replica {r}: E0 = {energy(J, s):.1f}")

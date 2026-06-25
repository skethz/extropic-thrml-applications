"""thrml CPU Ising solver for sparse-coding super-resolution.

Drop-in replacement for the GPU `solve_batch` kernel in libising_int.so.

The SR pipeline poses, per image patch, a QUBO / Ising problem over N spins
(dictionary-atom selection) that share ONE dense coupling matrix J (the Gram
matrix of the augmented dictionary) but differ in their per-patch bias h.

We express this with Extropic's thrml as an `IsingEBM` on a COMPLETE graph of N
SpinNodes (energy E(s) = -beta * (sum_i b_i s_i + sum_{i<j} J_ij s_i s_j)) and
sample it with block-Gibbs (each node its own singleton block, since a complete
graph has no smaller chromatic partition). A large `beta` (low temperature)
turns the sampler into an optimizer, matching the GPU's zero-temperature greedy
descent. The per-patch bias is handled by `jax.vmap`-ing the whole solve over
the batch of bias vectors.
"""
import numpy as np
import jax
import jax.numpy as jnp
from thrml import SpinNode, Block, SamplingSchedule, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init


class ThrmlIsingSolver:
    def __init__(self, N, beta=2000.0, n_warmup=30, init="down"):
        """
        N        : number of spins (dictionary atoms, augmented)
        beta     : inverse temperature. Large -> acts as a greedy optimizer.
        n_warmup : number of block-Gibbs sweeps before reading the state.
        init     : 'down' -> all spins start at -1 (matches GPU init);
                   'hinton' -> Bernoulli(sigmoid(beta*h)) marginal init.
        """
        self.N = N
        self.beta = float(beta)
        self.n_warmup = int(n_warmup)
        self.init = init

        # Complete-graph topology: N SpinNodes, all i<j edges.
        self.nodes = [SpinNode() for _ in range(N)]
        iu = np.triu_indices(N, k=1)
        self.edge_i = iu[0]
        self.edge_j = iu[1]
        self.edges = [(self.nodes[int(a)], self.nodes[int(b)])
                      for a, b in zip(self.edge_i, self.edge_j)]
        # Each node is its own block (complete graph => no 2-coloring).
        self.free_blocks = [Block([n]) for n in self.nodes]
        self.out_block = Block(self.nodes)

        self._J = None          # full symmetric (N,N) float32
        self._w_edges = None    # (n_edges,) edge weights = J_ij for i<j
        self._solve = None      # compiled vmapped solver

    def set_problem_matrix(self, J_full):
        """J_full: dense symmetric (N,N) Ising coupling (the GPU J_float)."""
        J_full = np.asarray(J_full, dtype=np.float32)
        self._J = J_full
        self._w_edges = jnp.asarray(J_full[self.edge_i, self.edge_j])
        self._build()

    def _build(self):
        nodes, edges = self.nodes, self.edges
        free_blocks, out_block = self.free_blocks, self.out_block
        beta = jnp.asarray(self.beta, dtype=jnp.float32)
        weights = self._w_edges
        n_warmup = self.n_warmup
        init_mode = self.init
        N = self.N

        def solve_one(key, bias_vec):
            model = IsingEBM(nodes, edges, bias_vec, weights, beta)
            program = IsingSamplingProgram(model, free_blocks, [])
            k_init, k_samp = jax.random.split(key)
            if init_mode == "hinton":
                init_state = hinton_init(k_init, model, free_blocks, ())
            else:  # all spins down (-1) == boolean False
                init_state = [jnp.zeros((1,), dtype=jnp.bool_) for _ in free_blocks]
            sched = SamplingSchedule(n_warmup=n_warmup, n_samples=1,
                                     steps_per_sample=1)
            out = sample_states(k_samp, program, sched, init_state, [], [out_block])
            return out[0][0]  # (N,) bool

        self._solve = jax.jit(jax.vmap(solve_one))

    def solve_batch(self, h_batch, seed=1234):
        """h_batch: (B, N) float bias. Returns (B, N) uint8 in {0,1}."""
        B = h_batch.shape[0]
        keys = jax.random.split(jax.random.key(seed), B)
        spins = self._solve(keys, jnp.asarray(h_batch, dtype=jnp.float32))
        return np.asarray(spins).astype(np.uint8)

    def energy(self, h_batch, s01):
        """Mean Ising energy E = -(sum b_i s_i + sum_{i<j} J_ij s_i s_j),
        s in {-1,+1}. Lower is better. (beta excluded; just the raw objective.)"""
        s = 2.0 * np.asarray(s01, dtype=np.float64) - 1.0
        h = np.asarray(h_batch, dtype=np.float64)
        lin = np.sum(h * s, axis=1)
        quad = np.einsum("bi,ij,bj->b", s, self._J.astype(np.float64), s) * 0.5
        return -(lin + quad)

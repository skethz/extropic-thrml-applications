"""thrml-native single-site heat-bath update.

Lifts the EXACT validated API incantation from spike_n8.py:
  * SpinGibbsConditional / DiscreteEBMInteraction from thrml.models.discrete_ebm
  * DiscreteEBMInteraction(n_spin=1, weights=w_row[None, :])
  * _SAMP.sample(key, [interaction], [active], states, None, out_sd) -> (new_val, _)

thrml's SpinGibbsConditional computes gamma = sum_i (beta*J[j,i]) s_i = beta*L_j
and draws bernoulli(sigmoid(2*gamma)), so the heat-bath is computed by thrml, not
hand-rolled. Pure & jit-friendly: no python-side branching on traced values, so the
next task can wrap it in lax.scan with a traced site index and vmap over replicas.
"""
from functools import partial
import jax, jax.numpy as jnp

from thrml.models.discrete_ebm import SpinGibbsConditional, DiscreteEBMInteraction

_SAMP = SpinGibbsConditional()
_OUT_SD = jax.ShapeDtypeStruct((1,), jnp.bool_)


def gibbs_update_site(spins_bool, w_row, key):
    """thrml heat-bath for one site: returns the new bool spin value (scalar bool array).
    spins_bool: (N,) bool (True=+1, False=-1). w_row: (N,) float = beta*J[j,:] for the chosen site j.
    Uses thrml's SpinGibbsConditional so the conditional/heat-bath is computed by thrml, not hand-rolled.
    Caller must ensure w_row[j] == 0 (the diagonal of J must be zeroed); the active mask is all-ones and does not zero the self-coupling."""
    interaction = DiscreteEBMInteraction(n_spin=1, weights=w_row[None, :])
    active = jnp.ones((1, w_row.shape[0]), dtype=bool)
    states = [[spins_bool[None, :]]]
    new_val, _ = _SAMP.sample(key, [interaction], [active], states, None, _OUT_SD)
    return new_val[0]

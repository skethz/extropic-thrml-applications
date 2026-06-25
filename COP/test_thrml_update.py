import jax, jax.numpy as jnp, numpy as np
import ref_instance as R
from thrml_update import gibbs_update_site


def test_gamma_equals_beta_L_via_determinism():
    n = 64; beta = 8.0
    J = R.make_graph(1, n).astype(np.float32)
    s = R.make_init_spins(2, 0, n)
    sb = jnp.array(s > 0)
    Js = jnp.array(J) @ jnp.array(s.astype(np.float32))   # L = J s
    for j in [0, 7, 31, 63]:
        assert float(Js[j]) != 0.0   # sign test ill-defined if L_j == 0
        w_row = beta * jnp.array(J[j])
        key = jax.random.PRNGKey(j)
        new_b = bool(gibbs_update_site(sb, w_row, key))
        assert new_b == (float(Js[j]) > 0)   # at beta=8, heat-bath -> sign(L_j)


def test_gamma_value_exact():
    # thrml's internal conditional parameter gamma must equal beta*L_j EXACTLY
    # (not just sign(L_j)); a dropped beta factor would still pass the sign test.
    import jax as _jax
    from thrml.models.discrete_ebm import DiscreteEBMInteraction
    from thrml_update import _SAMP, _OUT_SD

    n = 64; beta = 8.0
    J = R.make_graph(1, n).astype(np.float32)
    s = R.make_init_spins(2, 0, n)
    sb = jnp.array(s > 0)
    Js = jnp.array(J) @ jnp.array(s.astype(np.float32))   # L = J s
    for j in [0, 7, 31, 63]:
        w_row = beta * jnp.array(J[j])
        interaction = DiscreteEBMInteraction(n_spin=1, weights=w_row[None, :])
        active = jnp.ones((1, n), dtype=bool)
        states = [[sb[None, :]]]
        # mirror spike_n8.py: g_check, _ = _samp.compute_parameters(key,[interaction],[active],states,None,out_sd)
        g_check, _ = _SAMP.compute_parameters(
            _jax.random.PRNGKey(0), [interaction], [active], states, None, _OUT_SD
        )
        assert abs(float(g_check[0]) - beta * float(Js[j])) < 1e-3

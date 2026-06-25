import numpy as np, ref_instance as R
def test_graph_symmetric_pm1():
    J = R.make_graph(1, 64)
    assert (J == J.T).all()
    assert set(np.unique(J)).issubset({-1, 0, 1})
    assert (np.diag(J) == 0).all()
def test_energy_matches_pairwise_sum():
    J = R.make_graph(1, 64); s = R.make_init_spins(2, 0, 64)
    pair = -sum(J[i, j] * s[i] * s[j] for i in range(64) for j in range(i + 1, 64))
    assert R.energy(J, s) == pair
def test_init_energy_matches_known_n2048():
    J = R.make_graph(1)              # n=2048
    expected = {0: 1486.0, 1: -1090.0, 2: -1974.0, 3: 638.0}
    for r, e in expected.items():
        assert R.energy(J, R.make_init_spins(2, r)) == e

"""
optimal_weight is where a subtle bug once silenced the whole derivative:
with a fixed leaf radius the geometric estimator is identically zero, and a
naive minimum-variance formula returns w=0, discarding the only working
estimator. These tests pin the degenerate branches.
"""
import numpy as np

from canopi.derivative import optimal_weight


def test_degenerate_edge_uses_fictive_only():
    rng = np.random.default_rng(0)
    D_fic = rng.normal(size=(500, 3))
    D_edg = np.zeros((500, 3))          # fixed leaf radius => Var(D_edg) = 0
    w = optimal_weight(D_fic, D_edg)
    assert np.allclose(w, 1.0), w


def test_degenerate_fictive_uses_edge_only():
    rng = np.random.default_rng(0)
    D_fic = np.zeros((500, 3))
    D_edg = rng.normal(size=(500, 3))
    w = optimal_weight(D_fic, D_edg)
    assert np.allclose(w, 0.0), w


def test_weight_is_bounded():
    rng = np.random.default_rng(1)
    D_fic = rng.normal(0.0, 1.0, size=(2000, 4))
    D_edg = rng.normal(0.0, 3.0, size=(2000, 4))
    w = optimal_weight(D_fic, D_edg)
    assert w.shape == (4,)
    assert np.all(w >= 0.0) and np.all(w <= 1.0)


def test_weight_minimises_variance():
    """The returned w must beat both pure estimators on a mixed problem."""
    rng = np.random.default_rng(2)
    n = 20000
    truth = 1.0
    D_fic = truth + rng.normal(0.0, 1.0, size=(n, 1))
    D_edg = truth + rng.normal(0.0, 2.0, size=(n, 1))
    w = optimal_weight(D_fic, D_edg)[0]

    def var_at(a):
        return np.var(a * D_fic[:, 0] + (1.0 - a) * D_edg[:, 0])

    assert var_at(w) <= var_at(0.0)
    assert var_at(w) <= var_at(1.0)
    # the lower-variance estimator should dominate the blend
    assert w > 0.5

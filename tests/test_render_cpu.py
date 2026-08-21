"""
End-to-end CPU render. Slow (Numba JIT warmup dominates), so it is marked
`slow` — deselect with `pytest -m "not slow"`.
"""
import numpy as np
import pytest

from canopi import SceneConfig, render


@pytest.mark.slow
def test_tiny_cpu_render_produces_finite_brf_and_derivative():
    cfg = SceneConfig(
        LAI_eff=0.5, P_LAI=0.5, rho=0.06, tau=0.02, rsoil=0.03,
        ALA="UNI", sun_zenith_deg=41.0,
        scene_xy=3.0, scene_z=2.0,
        n_photons=1000, obs_step_deg=30.0, max_depth=8,
        compute_derivative=True, seed=1,
    )
    res = render(cfg)

    brf = np.asarray(res.brf)
    assert brf.size > 0
    assert np.all(np.isfinite(brf))
    assert np.all(brf >= 0.0) and np.all(brf <= 1.0)

    for name in ("deriv_fic", "deriv_edg", "deriv_opt", "opt_weights"):
        arr = np.asarray(getattr(res, name))
        assert np.all(np.isfinite(arr)), name

    assert np.asarray(res.deriv_opt).shape == brf.shape
    w = np.asarray(res.opt_weights)
    assert np.all(w >= 0.0) and np.all(w <= 1.0)

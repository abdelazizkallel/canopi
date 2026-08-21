"""Scene construction invariants (small scenes — these run in seconds)."""
import numpy as np

from canopi.config import SceneConfig
from canopi.scene import build_scene

SMALL = dict(scene_xy=3.0, scene_z=2.0, P_LAI=0.5)


def _scene(**kw):
    cfg = SceneConfig(**{**SMALL, **kw})
    return cfg, build_scene(cfg, np.random.default_rng(0))


def test_geometric_variability_scales_with_config():
    """std_LAI = leaf_radius_std_mul * LAI_r — the geometric-variability knob."""
    for mul in (0.1, 0.3, 0.5):
        cfg, s = _scene(LAI_eff=0.5, leaf_radius_std_mul=mul)
        assert np.isclose(s.std_LAI / s.LAI_r, mul, rtol=1e-9)


def test_more_lai_means_more_leaves():
    _, lo = _scene(LAI_eff=0.3)
    _, hi = _scene(LAI_eff=0.6)
    assert hi.n_leaf > lo.n_leaf


def test_p_lai_carried_onto_scene():
    cfg, s = _scene(LAI_eff=0.5)
    assert np.isclose(s.P_LAI, cfg.P_LAI)

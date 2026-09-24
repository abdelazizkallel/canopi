"""
CANOPI — CANopy prObabilistic rendering and Inversion.

Bidirectional Monte Carlo radiative transfer in plant canopies that returns
reflectance together with its analytic derivative with respect to leaf area
index and leaf reflectance, from a single pass.

Usage
-----
    from canopi import SceneConfig, render

    cfg = SceneConfig(n_photons=200_000, sun_zenith_deg=41, ALA='UNI',
                      compute_derivative=True)
    result = render(cfg)          # CPU
                                  # canopi.gpu.render_gpu(cfg) for CUDA

See docs/model.md for the model, docs/api.md for the full API, and
docs/figures.md to reproduce the paper figures.
"""
from .config import SceneConfig
from .renderer import render, render_backward_image, render_backward_image_deriv, RenderResult

__version__ = "1.0.1"

__all__ = ['SceneConfig', 'render', 'render_backward_image',
           'render_backward_image_deriv', 'RenderResult', '__version__']

"""
quickstart.py — smallest useful CANOPI run.

Renders a homogeneous (turbid) canopy over the viewing hemisphere and returns
the BRF together with the three derivative estimators, from a single pass.

    python examples/quickstart.py                 # GPU if available, else CPU
    python examples/quickstart.py --cpu           # force the pure-Python tracer
    python examples/quickstart.py --n 1000000     # more photons

Needs no scene data download.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from canopi import SceneConfig, render
from canopi.gpu import GPU_AVAILABLE, render_gpu

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=200_000, help="photons (default 200000)")
ap.add_argument("--lai", type=float, default=2.0, help="effective LAI (default 2.0)")
ap.add_argument("--cpu", action="store_true", help="force the CPU tracer")
args = ap.parse_args()

# RED-band leaf and soil optics, in the range of the RAMI-5 JBS scene
cfg = SceneConfig(
    LAI_eff=args.lai, P_LAI=0.5,
    rho=0.0632, tau=0.0213, rsoil=0.0274,
    ALA="UNI",
    sun_zenith_deg=41.0,
    n_photons=args.n,
    compute_derivative=True,
    seed=42,
)

use_gpu = GPU_AVAILABLE and not args.cpu
if not use_gpu:
    print("Running on CPU — this is slow; a few thousand photons is a sensible "
          "ceiling here. Use the GPU path for anything real.")

t0 = time.time()
res = render_gpu(cfg) if use_gpu else render(cfg)
dt = time.time() - t0

brf = np.asarray(res.brf)
zen = np.asarray(res.obs_zenith_deg)
dop = np.asarray(res.deriv_opt)

print(f"\n{'device':10s}: {'GPU' if use_gpu else 'CPU'}   "
      f"photons={cfg.n_photons:,}   LAI_eff={cfg.LAI_eff}   ({dt:.1f}s)")
print(f"{'BRF':10s}: mean={brf.mean():.6f}  min={brf.min():.6f}  max={brf.max():.6f}")
print(f"{'dBRF/dLAI':10s}: mean={dop.mean():+.6f}  (optimal estimator)")

print(f"\n  {'view zenith':>12s}  {'BRF':>10s}  {'dBRF/dLAI':>12s}")
for i in range(0, len(zen), max(1, len(zen) // 8)):
    print(f"  {zen[i]:12.1f}  {brf.ravel()[i]:10.6f}  {dop.ravel()[i]:+12.6f}")

print("\nres also carries: deriv_fic (structural), deriv_edg (geometric), "
      "opt_weights (the minimum-variance blend w).")

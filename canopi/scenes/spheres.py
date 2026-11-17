"""
Heterogeneous stand of spherical crowns.

The scene used for the heterogeneous figures in the CANOPI paper (and in
`Heterogeneous_Sampling.m` / `Heterogeneous_Sampling_LAI.m`): discrete leaf
disks scattered uniformly inside spherical crowns, the crowns scattered over a
100 x 100 m box.

    from canopi.scenes.spheres import load_sphere_scene
    scene, centres = load_sphere_scene(P_LAI=0.5, seed=42)

Defaults reproduce the published scene exactly: 15 crowns of radius 10 m, LAI 5
per crown (leaf area / crown cross-section), leaf radius 0.05 m, spherical leaf
angles. That gives a stand-level LAI_max of 4.71 and LAI_eff = P_LAI * LAI_max.

Structure follows the same convention as the rest of the package: the canopy
occupies z in [-H, 0], x and y are periodic, and the scene is over-populated by
1/P_LAI so that each photon sees LAI_eff = P_LAI * LAI_max on average.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from canopi.scene import Scene, _assign_leaves_fast

# ── published scene ─────────────────────────────────────────────────────────
LX, LY, H = 100.0, 100.0, 40.0
R_SPH = 10.0
LAI_PER_SPHERE = 5.0
LEAF_RADIUS = 0.05
VOXEL_SIZE = 1.0

# Illumination of the published heterogeneous figures
ILLUMINATION = dict(sun_zenith_deg=50.0, sun_azimuth_deg=0.0)

# Optics of the published heterogeneous figures (conservative scattering)
BAND = dict(rho=0.5, tau=0.5, rsoil=0.1)

# The 15 crown centres, as in the Matlab scene. Stored in the Matlab frame
# (origin at the box centre, z up); shifted into scene coordinates below.
_CENTRES_RAW = np.array([
    [-24.8302,   11.6110,   15.6210],
    [-38.3380,   -9.06342,  17.6094],
    [-31.3562,   32.5297,   14.3051],
    [ 11.9126,    8.32062,  12.1220],
    [ 39.4855,   37.1247,   16.3647],
    [ -8.30999,  -4.39422,  14.4241],
    [ 12.2769,  -25.1402,   12.4492],
    [ 32.4756,  -26.9023,   16.3684],
    [ -7.46991, -32.2138,   12.6678],
    [-27.4794,  -32.0266,   15.9146],
    [ 33.5709,   -6.31039,  14.5332],
    [ -5.24326,  39.8405,   16.8247],
    [ 18.1699,   35.8223,   11.5683],
    [ -5.26569,  18.9618,   17.2535],
    [ 37.8226,   14.6769,   11.5936],
], dtype=np.float64)

SPHERE_CENTRES = _CENTRES_RAW + np.array([50.0, 50.0, -30.0])


def _sample_centres(n: int, r_sph: float, lx: float, ly: float, h: float,
                    rng: np.random.Generator) -> np.ndarray:
    """Scatter n crown centres, rejecting overlaps where it is possible to."""
    z_lo, z_hi = -(h - r_sph), -r_sph
    centres: list = []
    for _ in range(n * 400):
        if len(centres) == n:
            break
        c = np.array([rng.uniform(0.0, lx), rng.uniform(0.0, ly),
                      rng.uniform(z_lo, z_hi) if z_hi > z_lo else -h / 2.0])
        if all(np.linalg.norm(c - o) > 2.0 * r_sph for o in centres):
            centres.append(c)
    while len(centres) < n:                       # too crowded to stay disjoint
        centres.append(np.array([rng.uniform(0.0, lx), rng.uniform(0.0, ly),
                                 rng.uniform(z_lo, z_hi) if z_hi > z_lo else -h / 2.0]))
    return np.asarray(centres)


def _leaves_in_sphere(n: int, r: float, rng: np.random.Generator) -> np.ndarray:
    """n points uniform inside a sphere of radius r, by rejection."""
    out, got = [], 0
    while got < n:
        pts = rng.uniform(-r, r, size=(max(2 * n, 10_000), 3))
        keep = pts[(pts ** 2).sum(1) <= r * r]
        out.append(keep)
        got += len(keep)
    return np.vstack(out)[:n]


def load_sphere_scene(
    n_spheres: int = 15,
    r_sph: float = R_SPH,
    lai_per_sphere: float = LAI_PER_SPHERE,
    leaf_radius: float = LEAF_RADIUS,
    scene_xy: float = LX,
    height: float = H,
    voxel_size: float = VOXEL_SIZE,
    P_LAI: float = 0.5,
    leaf_radius_std_mul: float = 0.0,
    rho: float = BAND["rho"],
    tau: float = BAND["tau"],
    rsoil: float = BAND["rsoil"],
    max_leaves: Optional[int] = None,
    seed: int = 42,
    verbose: bool = False,
) -> Tuple[Scene, np.ndarray]:
    """
    Build a heterogeneous stand of spherical crowns.

    Parameters
    ----------
    n_spheres, r_sph : crown count and radius [m]
    lai_per_sphere   : leaf area / crown cross-section (pi r_sph^2)
    leaf_radius      : leaf disk radius [m]
    scene_xy, height : box footprint and canopy depth [m]; z in [-height, 0]
    P_LAI            : Bernoulli activation probability. The scene is
                       over-populated by 1/P_LAI so LAI_eff = P_LAI * LAI_max.
    max_leaves       : cap on the number of *real* leaves before over-population
                       (thins the stand, and so lowers LAI_max)
    leaf_radius_std_mul : leaf-size spread; 0 keeps the radius fixed, which makes
                       q_k a step function and the geometric estimator degenerate

    Returns
    -------
    scene   : canopi.scene.Scene
    centres : (n_spheres, 3) crown centres in scene coordinates
    """
    rng = np.random.default_rng(seed)

    if (n_spheres == len(SPHERE_CENTRES) and abs(r_sph - R_SPH) < 1e-9
            and abs(scene_xy - LX) < 1e-9 and abs(height - H) < 1e-9):
        centres = SPHERE_CENTRES.copy()           # the published arrangement
    else:
        centres = _sample_centres(n_spheres, r_sph, scene_xy, scene_xy, height, rng)

    # leaf area per crown = lai_per_sphere * pi r_sph^2  ->  n = lai * (R/r)^2
    n_per_sphere = int(round(lai_per_sphere * (r_sph / leaf_radius) ** 2))
    if max_leaves:
        n_per_sphere = min(n_per_sphere, max(1, int(max_leaves) // max(1, n_spheres)))

    pos, nrm = [], []
    for cx, cy, cz in centres:
        pts = _leaves_in_sphere(n_per_sphere, r_sph, rng)
        pts[:, 0] = (pts[:, 0] + cx) % scene_xy
        pts[:, 1] = (pts[:, 1] + cy) % scene_xy
        pts[:, 2] = np.clip(pts[:, 2] + cz, -height, 0.0)
        pos.append(pts)

        # spherical leaf angle distribution, normals kept in the upper hemisphere
        cos_z = rng.uniform(-1.0, 1.0, n_per_sphere)
        phi = rng.uniform(0.0, 2 * np.pi, n_per_sphere)
        sin_z = np.sqrt(np.maximum(0.0, 1.0 - cos_z ** 2))
        nz = np.column_stack([sin_z * np.cos(phi), sin_z * np.sin(phi), cos_z])
        nz[nz[:, 2] < 0] *= -1
        nrm.append(nz)

    leaf_pos = np.vstack(pos)
    leaf_nrm = np.vstack(nrm)
    n_real = len(leaf_pos)

    # ── over-populate by 1/P_LAI: the extra leaves are the fictive ones ──────
    n_extra = int(round(n_real / P_LAI)) - n_real
    if n_extra > 0:
        idx = rng.choice(n_real, n_extra, replace=True)
        jitter = rng.normal(0.0, leaf_radius * 0.1, (n_extra, 3))
        jitter[:, 2] = 0.0
        extra = leaf_pos[idx] + jitter
        extra[:, 0] %= scene_xy
        extra[:, 1] %= scene_xy
        extra[:, 2] = np.clip(extra[:, 2], -height, 0.0)
        leaf_pos = np.vstack([leaf_pos, extra])
        leaf_nrm = np.vstack([leaf_nrm, leaf_nrm[idx]])

    n_leaf = len(leaf_pos)
    LAI_r = math.pi * leaf_radius ** 2
    surf = scene_xy * scene_xy
    LAI_Max = n_leaf * LAI_r / surf

    std_LAI = float(leaf_radius_std_mul) * LAI_r
    radii = np.full(n_leaf, leaf_radius, dtype=np.float64)
    r_max = radii if std_LAI <= 0 else np.full(
        n_leaf, math.sqrt(max(LAI_r + 2.5 * std_LAI, 0.0) / math.pi))

    pos_max = np.array([scene_xy, scene_xy, -height], dtype=np.float64)
    max_voxel = np.maximum((np.abs(pos_max) / voxel_size).astype(int), 1)
    voxel_count, voxel_leaves = _assign_leaves_fast(leaf_pos, voxel_size, max_voxel)

    nz = np.clip(leaf_nrm[:, 2], -1.0, 1.0)
    scene = Scene(
        leaf_pos=leaf_pos, leaf_normal=leaf_nrm,
        leaf_zenith=np.arccos(nz),
        leaf_azimuth=np.arctan2(leaf_nrm[:, 1], leaf_nrm[:, 0]),
        leaf_radius=radii, leaf_radius_max=r_max,
        pos_max=pos_max, voxel_size=float(voxel_size),
        max_voxel=max_voxel, voxel_count=voxel_count, voxel_leaves=voxel_leaves,
        rho=float(rho), tau=float(tau), rsoil=float(rsoil),
        LAI_r=float(LAI_r), std_LAI=float(std_LAI),
        LAI_Max=float(LAI_Max), P_LAI=float(P_LAI),
        n_leaf_norm=float(n_leaf / surf),
    )
    if verbose:
        print(f"[spheres] {n_spheres} crowns r={r_sph} m  "
              f"{n_leaf:,} leaves ({n_real:,} real)  "
              f"LAI_max={LAI_Max:.4f}  LAI_eff={LAI_Max * P_LAI:.4f}")
    return scene, centres

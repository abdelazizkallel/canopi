"""
Canopy RT Simulator — Configuration

RAMI-style scene
----------------
Leaves are disk-shaped, randomly placed in 3-D (uniform x,y,z).
LAI density is controlled via two parameters:

    LAI_eff  : effective (observed/gap-corrected) LAI
    P_LAI    : Bernoulli survival probability  ∈ (0, 1]

The total number of leaf disks placed in the scene is:

    N_leaves = LAI_Max * scene_xy^2 / (pi * leaf_radius^2)

where:

    LAI_Max = LAI_eff / P_LAI          (total leaf density including fictive)

Each photon, every leaf is independently kept (real, leaf_effect=1)
or made fictive (transparent, leaf_effect=0) with probability P_LAI.
The expected visible LAI seen by any photon is then:

    E[LAI_visible] = P_LAI * LAI_Max = LAI_eff   ✓

Per-voxel LAI statistics
------------------------
When voxel_lai_stats=True, build_scene() computes for every voxel:

    voxel_lai_mean[v]  = n_leaves_in_voxel[v] * LAI_r / voxel_volume
    voxel_lai_std[v]   = sqrt(n_leaves_in_voxel[v]) * LAI_r / voxel_volume

These can be used for heterogeneous canopy analyses.
"""
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class SceneConfig:
    # ── Scene dimensions ──────────────────────────────────────────────────────
    scene_xy: float = 4.0      # scene width & depth [m] (periodic in x, y)
    scene_z:  float = 1.0      # canopy height [m]  (z ∈ [-scene_z, 0])
    voxel_size: float = 0.25   # voxel cell size [m]

    # ── Leaf parameters ───────────────────────────────────────────────────────
    leaf_radius: float = 0.02            # mean leaf disk radius [m]
    leaf_radius_std_mul: float = 0.300   # std_LAI = std_mul * LAI_r  (0.001 ≈ deterministic)
    rho:   float = 0.5                   # leaf reflectance (Lambertian)
    tau:   float = 0.5                   # leaf transmittance
    rsoil: float = 1.0                   # soil reflectance (Lambertian)

    # ── LAI / canopy structure ────────────────────────────────────────────────
    # LAI_eff  : effective LAI (what the scene looks like on average)
    # P_LAI    : Bernoulli survival prob.  LAI_Max = LAI_eff / P_LAI
    #
    # Density interpretation:
    #   compute_derivative=True  → N_leaves based on LAI_Max (includes fictive)
    #   compute_derivative=False → P_LAI forced to 1, N_leaves based on LAI_eff
    LAI_eff: float = 0.5   # effective (gap-corrected) LAI
    P_LAI:   float = 0.5   # Bernoulli survival probability
    ALA: str = 'UNI'       # angular leaf distribution: UNI, PLA, ERE, SPH

    # ── Per-voxel LAI statistics ──────────────────────────────────────────────
    # When True, build_scene() populates scene.voxel_lai_mean and
    # scene.voxel_lai_std arrays (shape: n_voxels) after leaf placement.
    voxel_lai_stats: bool = False

    # ── Illumination ──────────────────────────────────────────────────────────
    sun_zenith_deg:  float = 0.0    # 0 = overhead sun
    sun_azimuth_deg: float = 180.0  # [°]

    # ── Observation directions (principal plane + cross-plane) ────────────────
    obs_zenith_max_deg: float = 60.0  # max obs zenith [°]
    obs_step_deg:       float = 5.0   # step [°]

    # ── Rendering ─────────────────────────────────────────────────────────────
    n_photons: int = 2000
    # Photon start x/y are sampled in:
    # [photon_start_xy_min_frac * scene_xy, photon_start_xy_max_frac * scene_xy]
    photon_start_xy_min_frac: float = 0.0
    photon_start_xy_max_frac: float = 1.0
    max_depth:  int   = 50
    rr_depth:   int   = 999        # Russian Roulette start depth (999 = disabled)
    rr_min:     float = 0.5
    image_size: Tuple[int, int] = (64, 64)

    # ── Probabilistic derivative ──────────────────────────────────────────────
    compute_derivative: bool = True   # compute dBRF/dLAI per photon

    # ── Thresholds ────────────────────────────────────────────────────────────
    energy_threshold: float = 1e-5
    edge_threshold:   float = 2.5    # std multiples for near-miss detection

    # ── Misc ──────────────────────────────────────────────────────────────────
    seed: int = 42

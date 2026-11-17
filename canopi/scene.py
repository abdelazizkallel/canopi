"""
Canopy RT — Scene: leaf placement + voxel acceleration structure.

RAMI-style canopy
-----------------
Leaves are disk-shaped and placed at uniformly random 3-D positions
inside the scene box (x ∈ [0,Lx], y ∈ [0,Ly], z ∈ [-H, 0]).

The fundamental density relationship is:

    LAI_Max  = LAI_eff / P_LAI        (total leaf density)
    N_leaves = LAI_Max * Lx*Ly / LAI_r

where LAI_r = π * r² is the area of one leaf disk.

In the forward path each leaf is kept real with probability P_LAI
(Bernoulli thinning).  The expected effective LAI seen per photon is:

    E[LAI_seen] = P_LAI * LAI_Max = LAI_eff   ✓

When compute_derivative=False, P_LAI is forced to 1 and N_leaves is
computed directly from LAI_eff (no fictive leaves needed).

Per-voxel LAI statistics
------------------------
When cfg.voxel_lai_stats=True, build_scene() counts how many leaf
disks fall into each voxel and computes:

    voxel_lai_mean[v] = n_v * LAI_r / V_voxel      [m² / m³]
    voxel_lai_std[v]  = sqrt(n_v) * LAI_r / V_voxel

These arrays are stored in scene.voxel_lai_mean and scene.voxel_lai_std.
They describe the spatial heterogeneity of leaf density across the canopy.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from .config import SceneConfig
from .lad import sample_leaf_orientations
from .geometry import pos3d_index


@dataclass
class Scene:
    # ── Leaf data ─────────────────────────────────────────────────────────────
    leaf_pos:     np.ndarray   # (N, 3)   disk centre positions
    leaf_normal:  np.ndarray   # (N, 3)   unit normals (disk orientation)
    leaf_zenith:  np.ndarray   # (N,)     zenith  [rad]
    leaf_azimuth: np.ndarray   # (N,)     azimuth [rad]
    leaf_radius:  np.ndarray   # (N,)     disk radii (resampled per photon)
    leaf_radius_max: np.ndarray  # (N,)   r_max for voxel coverage (fixed)

    # ── Scene box ─────────────────────────────────────────────────────────────
    pos_max:    np.ndarray     # [Lx, Ly, -H]
    voxel_size: float

    # ── Voxel grid ────────────────────────────────────────────────────────────
    max_voxel:    np.ndarray   # (3,) int  — number of voxels per axis
    voxel_count:  np.ndarray   # (nv,)     — leaf disk centres per voxel
    voxel_leaves: np.ndarray   # (nv, cap) — leaf indices

    # ── Optical properties ────────────────────────────────────────────────────
    rho:   float
    tau:   float
    rsoil: float

    # ── Derived scalars ───────────────────────────────────────────────────────
    LAI_r:       float   # π * r_mean²  — area of one mean disk
    std_LAI:     float   # std of per-leaf LAI distribution
    LAI_Max:     float   # total leaf density = LAI_eff / P_LAI
    P_LAI:       float   # Bernoulli survival probability
    n_leaf_norm: float   # n_leaf * LAI_r / surf  / LAI_r  = n_leaf / surf * LAI_r... kept for derivative
    periodic_xy: int = 1 # 1: periodic x/y boundaries, 0: finite walls

    # ── Per-voxel LAI statistics (optional, set by build_scene) ──────────────
    voxel_lai_mean: Optional[np.ndarray] = None   # (nv,) mean LAI density [m²/m³]
    voxel_lai_std:  Optional[np.ndarray] = None   # (nv,) std LAI density

    @property
    def n_leaf(self) -> int:
        return len(self.leaf_pos)

    def resample_radii(self, rng: np.random.Generator) -> None:
        """
        Re-draw leaf disk radii each photon (probabilistic rendering mode).

        Leaf LAI_i ~ N(LAI_r, std_LAI²) clipped to [0, ∞).
        radius_i = sqrt(LAI_i / π).
        """
        LAI_vals = np.maximum(
            self.LAI_r + self.std_LAI * rng.standard_normal(self.n_leaf),
            0.0,
        )
        self.leaf_radius[:] = np.sqrt(LAI_vals / np.pi)

    def __setstate__(self, state):
        # Scenes pickled before 2026-11-16 may have a voxel grid that stops above the soil.
        self.__dict__.update(state)
        ensure_soil_in_grid(self)


# ─────────────────────────────────────────────────────────────────────────────
#  Voxel grid size
# ─────────────────────────────────────────────────────────────────────────────

def voxel_grid_dims(pos_max, voxel_size) -> np.ndarray:
    """
    Number of voxels per axis.

    x, y: floor(L / voxel_size) (unchanged; the kernels use n * voxel_size as the periodic period).
    z   : CEIL(H / voxel_size), so that the grid reaches the soil at z = -H.

    Why z must reach the soil: the DDA clamps the ray start to the grid,
    _Pz = max(Pz, -mvz*vs).  With floor, a scene of H = 30.513 m gets 30 layers and every
    ray leaving the soil restarts 0.513 m higher at the SAME (x, y), off the sunlit path it
    should retrace.  This broke the soil hot spot (JBS O08: hot-spot BRF 14 % low against
    DART; with ceil, CANOPI 0.03961 = DART 0.03961).  Found 2026-11-16, DART comparison.
    """
    pos_max = np.asarray(pos_max, dtype=np.float64)
    q = np.abs(pos_max) / voxel_size
    n = np.maximum(q.astype(int), 1)
    # x, y: when L is (within 1 %) a whole number of voxels, take that number, so that the
    # periodic period n * voxel_size equals the scene size (see periodic_voxel_size).
    for k in (0, 1):
        if abs(q[k] - round(q[k])) < 0.01:
            n[k] = max(int(round(q[k])), 1)
    n[2] = max(int(np.ceil(abs(pos_max[2]) / voxel_size - 1e-9)), 1)
    return n


def periodic_voxel_size(lx: float, ly: float, target: float = 1.0, tol: float = 2e-3) -> float:
    """
    Voxel size close to `target` such that lx and ly are both (within tol m) whole numbers of
    voxels, so the kernels' periodic period (n * voxel_size) equals the scene size.

    The kernels wrap with Lx = mvx * vs. With vs = 1 m the JBS stand (105.5115 x 106.1535 m)
    is wrapped with period 105 x 106 m and strips of 0.51 m and 0.15 m carry leaves twice.
    For JBS this returns 0.643364 m (164 x 165 voxels = 105.5117 x 106.1551 m), which is also
    the DART cell size that makes DART's scene (rounded UP to whole cells) match.
    """
    best = None
    for nx in range(max(1, int(lx / (2 * target))), int(lx / (0.5 * target)) + 1):
        vs = lx / nx
        ny = round(ly / vs)
        err = max(abs(nx * vs - lx), abs(ny * vs - ly))
        # nudge vs up so that ceil(L / vs) == n for both axes (DART rounds up)
        vs_up = max(lx / nx, ly / ny) * (1 + 1e-7)
        err = max(abs(nx * vs_up - lx), abs(ny * vs_up - ly))
        if err < tol and (best is None or abs(vs_up - target) < abs(best - target)):
            best = vs_up
    if best is None:
        raise ValueError(f"no voxel size within {tol} m for {lx} x {ly}")
    return best


def _assign_leaves_bbox(leaf_pos, leaf_radius, voxel_size, max_voxel) -> tuple:
    """
    Vectorised assignment of each leaf to EVERY voxel its disk can overlap (the voxels touched by
    the cube centre +- r; at most 8). The primary-voxel assignment (_assign_leaves_fast) misses
    the part of a disk that sticks into a neighbouring voxel: a ray through that voxel only does
    not see the leaf. Periodic wrap in x, y; clipped in z.
    """
    mv = np.asarray(max_voxel, dtype=np.int64)
    N = len(leaf_pos)
    ap = np.abs(leaf_pos)
    r = np.asarray(leaf_radius, dtype=np.float64)[:, None]
    lo = np.floor((ap - r) / voxel_size).astype(np.int64)
    hi = np.floor((ap + r) / voxel_size).astype(np.int64)
    ids, cells = [], []
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                c = np.stack([np.where(dx, hi[:, 0], lo[:, 0]), np.where(dy, hi[:, 1], lo[:, 1]),
                              np.where(dz, hi[:, 2], lo[:, 2])], 1)
                c[:, 0] %= mv[0]; c[:, 1] %= mv[1]
                c[:, 2] = np.clip(c[:, 2], 0, mv[2] - 1)
                ids.append(np.arange(N)); cells.append(c[:, 0] + c[:, 1] * mv[0] + c[:, 2] * mv[0] * mv[1])
    ids = np.concatenate(ids); cells = np.concatenate(cells)
    key = np.unique(cells * N + ids)                       # drop duplicates (leaf, voxel)
    cells, ids = key // N, key % N
    n_vox = int(np.prod(mv))
    voxel_count = np.bincount(cells, minlength=n_vox).astype(np.int32)
    cap = max(int(voxel_count.max()) + 4, 8)
    cum = np.concatenate([[0], np.cumsum(voxel_count)])
    local = np.arange(len(cells)) - cum[cells]
    voxel_leaves = np.zeros((n_vox, cap), dtype=np.int32)
    voxel_leaves[cells, local] = ids
    return voxel_count, voxel_leaves


def ensure_soil_in_grid(scene, verbose: bool = True) -> bool:
    """
    Extend a scene whose voxel grid stops above the soil (built before the voxel_grid_dims fix)
    and re-assign its leaves. Edits scene.max_voxel IN PLACE (callers may hold a reference).
    Returns True if the scene was changed.
    """
    mv = getattr(scene, "max_voxel", None)
    if mv is None or getattr(scene, "voxel_count", None) is None:
        return False
    vs = float(scene.voxel_size)
    nz = int(voxel_grid_dims(scene.pos_max, vs)[2])
    if int(mv[2]) >= nz:
        return False
    old = int(mv[2])
    new_mv = np.array([int(mv[0]), int(mv[1]), nz], dtype=np.int64)
    rim = int(np.sum(scene.voxel_count)) > len(scene.leaf_pos)   # rim-assigned scene
    if rim:
        n_vox = int(np.prod(new_mv))
        vc = np.zeros(n_vox, dtype=np.int32)
        vl = np.zeros((n_vox, scene.voxel_leaves.shape[1]), dtype=np.int32)
        _assign_all_leaves(scene.leaf_pos, scene.leaf_radius, scene.leaf_normal, vs, new_mv,
                           vc, vl, leaf_radius_max=getattr(scene, "leaf_radius_max", None))
    else:
        vc, vl = _assign_leaves_fast(scene.leaf_pos, vs, new_mv)
    try:
        mv[2] = nz                       # in place
    except (TypeError, ValueError):
        scene.max_voxel = new_mv
    scene.voxel_count, scene.voxel_leaves = vc, vl
    if getattr(scene, "voxel_lai_mean", None) is not None:
        scene.voxel_lai_mean = scene.voxel_lai_std = None     # per-voxel stats no longer match
    if verbose:
        print(f"[scene] voxel grid extended to the soil: {old} -> {nz} z-layers "
              f"(H = {abs(float(scene.pos_max[2])):.3f} m), leaves re-assigned")
    return True





def scene_voxel_grid(pos_max, voxel_size=None, exact_geometry=True, default=1.0):
    """
    Voxel size and grid of a stand. voxel_size=None: with exact_geometry, the size closest to
    `default` whose grid period equals the scene (periodic_voxel_size); else `default`.
    Returns (voxel_size, max_voxel).
    """
    if voxel_size is None:
        voxel_size = default
        if exact_geometry:
            try:
                voxel_size = periodic_voxel_size(abs(float(pos_max[0])), abs(float(pos_max[1])), target=default)
            except ValueError:
                pass
    return float(voxel_size), voxel_grid_dims(pos_max, voxel_size)


def assign_leaves(leaf_pos, leaf_radius_max, voxel_size, max_voxel, exact_geometry=True):
    """Voxel lists: every voxel a disk overlaps (exact_geometry) or the centre voxel only (v1.0)."""
    if exact_geometry:
        return _assign_leaves_bbox(leaf_pos, leaf_radius_max, voxel_size, max_voxel)
    return _assign_leaves_fast(leaf_pos, voxel_size, max_voxel)

# ─────────────────────────────────────────────────────────────────────────────
#  Main scene builder
# ─────────────────────────────────────────────────────────────────────────────

def build_scene(cfg: SceneConfig, rng: np.random.Generator) -> Scene:
    """
    Build a RAMI-style canopy scene.

    Leaf disks are placed at uniformly random 3-D positions.
    The total leaf count is determined by:

        N_leaves = LAI_Max * scene_area / LAI_r

    where LAI_Max = LAI_eff / P_LAI when computing derivatives,
    and LAI_Max = LAI_eff when not (P_LAI forced to 1).
    """
    r       = cfg.leaf_radius
    pos_max = np.array([cfg.scene_xy, cfg.scene_xy, -cfg.scene_z], dtype=float)
    surf    = cfg.scene_xy ** 2

    # ── Leaf area (disk) parameters ───────────────────────────────────────────
    # LAI_r = area of one mean disk = π r²
    # std_LAI = standard deviation of the per-leaf LAI distribution
    #           (quasi-deterministic when std_mul ≈ 0.001)
    LAI_r   = np.pi * r ** 2
    std_LAI = cfg.leaf_radius_std_mul * LAI_r

    # ── Density: LAI_Max = LAI_eff / P_LAI ───────────────────────────────────
    # When not computing derivatives, all leaves are real → P_LAI = 1
    # → N_leaves corresponds exactly to LAI_eff, no over-population.
    effective_P = cfg.P_LAI if cfg.compute_derivative else 1.0
    LAI_Max     = cfg.LAI_eff / effective_P        # total (real + fictive) density
    N_leaves    = max(1, round(LAI_Max * surf / LAI_r))

    # ── Leaf disk radii ───────────────────────────────────────────────────────
    # Each leaf draws its own LAI_i ~ N(LAI_r, std_LAI²).
    # radius_i = sqrt(LAI_i / π).  Clipped to zero; very tight distribution
    # by default (std_mul=0.001 → near-deterministic radius).
    leaf_LAI_i  = np.maximum(
        LAI_r + std_LAI * rng.standard_normal(N_leaves),
        0.0,
    )
    leaf_radius = np.sqrt(leaf_LAI_i / np.pi)

    # ── Leaf positions: uniform random in 3-D ─────────────────────────────────
    # x, y: periodic → full [0, Lx] × [0, Ly].
    #   The disk can straddle the boundary; cyclic BC in DDA handles this.
    # z: closed [−H, 0].  Inset by r_max so the disk never clips through
    #   the sky (z=0) or soil (z=−H) faces.
    r_max_scalar = np.sqrt(max(LAI_r + 6.0 * std_LAI, 0.0) / np.pi)

    # x, y: no inset needed — periodic BC already handles wrap-around
    leaf_x = rng.uniform(0.0, pos_max[0], N_leaves)
    leaf_y = rng.uniform(0.0, pos_max[1], N_leaves)

    # z: inset by r_max so disk stays within canopy slab
    z_lo = pos_max[2] + r_max_scalar   # near soil,   e.g. -H + r_max  (negative)
    z_hi = -r_max_scalar               # near sky top, e.g. -r_max      (negative)
    if z_lo >= z_hi:                   # fallback for very thin canopies
        z_lo, z_hi = pos_max[2], 0.0
    leaf_z = rng.uniform(z_lo, z_hi, N_leaves)

    leaf_pos = np.column_stack([leaf_x, leaf_y, leaf_z])

    # ── Leaf orientations (disk normals) ──────────────────────────────────────
    # Drawn from the angular leaf distribution (ALA) set in cfg.
    # UNI = uniform over the sphere = spherical LAD, matching RAMI default.
    zenith, azimuth, normals = sample_leaf_orientations(N_leaves, cfg.ALA, rng)

    # ── Voxel acceleration grid ───────────────────────────────────────────────
    voxel_size = cfg.voxel_size
    max_voxel  = voxel_grid_dims(pos_max, voxel_size)   # z reaches the soil
    n_vox = int(max_voxel[0] * max_voxel[1] * max_voxel[2])

    # capacity: average leaves per voxel × safety factor
    avg_per_vox = N_leaves / n_vox
    cap = max(8, round(avg_per_vox * 8) + 4)

    voxel_count  = np.zeros(n_vox, dtype=np.int32)
    voxel_leaves = np.zeros((n_vox, cap), dtype=np.int32)

    # r_max per leaf: upper bound on radius across ALL photon resamplings.
    # 6-sigma coverage of LAI_i ~ N(LAI_r, std_LAI²).
    r_max = np.maximum(leaf_radius, r_max_scalar)

    _assign_all_leaves(
        leaf_pos, leaf_radius, normals,
        voxel_size, max_voxel,
        voxel_count, voxel_leaves,
        leaf_radius_max=r_max,
    )

    # ── Effective LAI realised in this scene ──────────────────────────────────
    LAI_actual  = N_leaves * LAI_r / surf   # ≈ LAI_Max by construction
    n_leaf_norm = LAI_actual / LAI_r        # = N_leaves / surf (used by derivative)

    # ── Per-voxel LAI statistics (optional) ──────────────────────────────────
    voxel_lai_mean = None
    voxel_lai_std  = None
    if cfg.voxel_lai_stats:
        voxel_lai_mean, voxel_lai_std = _compute_voxel_lai_stats(
            voxel_count, voxel_size, LAI_r,
        )

    return Scene(
        leaf_pos        = leaf_pos,
        leaf_normal     = normals,
        leaf_zenith     = zenith,
        leaf_azimuth    = azimuth,
        leaf_radius     = leaf_radius,
        leaf_radius_max = r_max,
        pos_max         = pos_max,
        voxel_size      = voxel_size,
        max_voxel       = max_voxel,
        voxel_count     = voxel_count,
        voxel_leaves    = voxel_leaves,
        rho             = cfg.rho,
        tau             = cfg.tau,
        rsoil           = cfg.rsoil,
        LAI_r           = LAI_r,
        std_LAI         = std_LAI,
        LAI_Max         = LAI_Max,
        P_LAI           = effective_P,
        n_leaf_norm     = n_leaf_norm,
        periodic_xy     = 1,
        voxel_lai_mean  = voxel_lai_mean,
        voxel_lai_std   = voxel_lai_std,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Per-voxel LAI statistics
# ─────────────────────────────────────────────────────────────────────────────

def _compute_voxel_lai_stats(
    voxel_count: np.ndarray,
    voxel_size:  float,
    LAI_r:       float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-voxel leaf area density statistics.

    For voxel v containing n_v leaf disk centres:

        LAI_density_mean[v] = n_v * LAI_r / V_voxel      [m² / m³]
        LAI_density_std[v]  = sqrt(n_v) * LAI_r / V_voxel

    The std follows from a Poisson count model: Var(n_v) = E[n_v] = n_v.

    Returns
    -------
    lai_mean : (n_vox,) float64
    lai_std  : (n_vox,) float64
    """
    V_voxel  = voxel_size ** 3                        # [m³]
    n_v      = voxel_count.astype(np.float64)
    lai_mean = n_v * LAI_r / V_voxel
    lai_std  = np.sqrt(n_v) * LAI_r / V_voxel        # Poisson std
    return lai_mean, lai_std


def lai_stats_summary(scene: Scene) -> dict:
    """
    Return a summary of voxel-level LAI statistics.

    Only meaningful when cfg.voxel_lai_stats=True was used in build_scene().

    Returns
    -------
    dict with keys: mean, std, min, max, cv (coefficient of variation),
                    fraction_empty  (voxels with zero leaves)
    """
    if scene.voxel_lai_mean is None:
        raise ValueError("voxel_lai_stats not computed. "
                         "Set cfg.voxel_lai_stats=True and rebuild the scene.")
    m = scene.voxel_lai_mean
    return dict(
        mean           = float(np.mean(m)),
        std            = float(np.std(m)),
        min            = float(np.min(m)),
        max            = float(np.max(m)),
        cv             = float(np.std(m) / np.mean(m)) if np.mean(m) > 0 else 0.0,
        fraction_empty = float(np.mean(m == 0.0)),
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Fast vectorised voxel assignment (primary voxel only)
# ─────────────────────────────────────────────────────────────────────────────

def _assign_leaves_fast(
    leaf_pos:    np.ndarray,   # (N, 3)
    voxel_size:  float,
    max_voxel:   np.ndarray,   # (3,) int
) -> tuple:
    """
    Vectorised O(N) voxel assignment — primary voxel only.

    Valid when leaf_radius << voxel_size (here 0.05 m << 1.0 m).
    Returns (voxel_count, voxel_leaves) with the same dtype contract
    as _assign_all_leaves.
    """
    mvx, mvy, mvz = int(max_voxel[0]), int(max_voxel[1]), int(max_voxel[2])
    n_vox = mvx * mvy * mvz
    N     = len(leaf_pos)

    # 3D voxel index for each leaf (use abs because z is negative)
    vidx = np.clip(
        (np.abs(leaf_pos) / voxel_size).astype(np.int64),
        0, np.array([mvx - 1, mvy - 1, mvz - 1], dtype=np.int64),
    )                                               # (N, 3)
    prim = (vidx[:, 0]
            + vidx[:, 1] * mvx
            + vidx[:, 2] * mvx * mvy).astype(np.int64)   # (N,)

    # Sort leaves by primary voxel for contiguous writes
    order       = np.argsort(prim, kind='stable')
    prim_sorted = prim[order]

    # Count leaves per voxel
    voxel_count = np.bincount(prim_sorted, minlength=n_vox).astype(np.int32)
    cap         = max(int(voxel_count.max()) + 4, 8)

    # Build voxel_leaves using cumulative offsets — fully vectorised
    cum   = np.concatenate([[0], np.cumsum(voxel_count)])  # (n_vox+1,)
    # Local rank of each leaf within its voxel
    local = np.arange(N, dtype=np.int64) - cum[prim_sorted]

    voxel_leaves = np.zeros((n_vox, cap), dtype=np.int32)
    voxel_leaves[prim_sorted, local] = order

    return voxel_count, voxel_leaves


# ─────────────────────────────────────────────────────────────────────────────
#  Per-leaf rim voxel assignment (slow Python loop — kept for small scenes)
# ─────────────────────────────────────────────────────────────────────────────

def _add_to_voxel(idx: int, leaf_idx: int,
                   voxel_count: np.ndarray, voxel_leaves: np.ndarray) -> None:
    if idx < 0 or idx >= len(voxel_count):
        return
    n = voxel_count[idx]
    if n < voxel_leaves.shape[1]:
        voxel_leaves[idx, n] = leaf_idx
        voxel_count[idx] = n + 1


def _assign_all_leaves(
    leaf_pos,
    leaf_radius,
    leaf_normal,
    voxel_size,
    max_voxel,
    voxel_count,
    voxel_leaves,
    leaf_radius_max=None,
):
    """
    Assign each leaf disk to every voxel its rim can reach.

    The primary voxel (centre position) is always assigned.
    Then the disk rim is sampled at 32 angles; any new voxel hit is added.
    Uses leaf_radius_max (= r_max, 6-sigma upper bound) so that all
    possible resampled radii are covered in the fixed voxel assignment.
    """
    n_leaf = len(leaf_pos)
    n_ech  = 32
    angles = np.linspace(0, 2 * np.pi, n_ech, endpoint=False)
    cos_a  = np.cos(angles)
    sin_a  = np.sin(angles)
    vc     = np.array([1.0, 0.0, 0.0])

    if leaf_radius_max is None:
        leaf_radius_max = leaf_radius

    for i in range(n_leaf):
        lpos = leaf_pos[i]
        r    = leaf_radius_max[i]
        nrm  = leaf_normal[i]

        # Primary voxel of leaf centre (use abs because z is negative)
        abs_pos = np.abs(lpos)
        pv      = np.clip((abs_pos / voxel_size).astype(int), 0, max_voxel - 1)
        primary = pos3d_index(pv, max_voxel)
        _add_to_voxel(primary, i, voxel_count, voxel_leaves)

        # Check whether disk rim crosses into neighbouring voxels
        frac = abs_pos / voxel_size - np.floor(abs_pos / voxel_size)
        if (np.any(frac < r / voxel_size) or
                np.any((1.0 - frac) < r / voxel_size)):

            # Local orthonormal frame of the disk
            xlf = np.cross(nrm, vc)
            if np.linalg.norm(xlf) < 1e-8:
                xlf = np.cross(nrm, np.array([0.0, 1.0, 0.0]))
            xlf = xlf / np.linalg.norm(xlf) * r
            ylf = np.cross(nrm, xlf)

            seen = {primary}
            for k in range(n_ech):
                p_rim = lpos + cos_a[k] * xlf + sin_a[k] * ylf
                av    = np.clip(
                    (np.abs(p_rim) / voxel_size).astype(int),
                    0, max_voxel - 1,
                )
                idx = pos3d_index(av, max_voxel)
                if idx not in seen:
                    seen.add(idx)
                    _add_to_voxel(idx, i, voxel_count, voxel_leaves)

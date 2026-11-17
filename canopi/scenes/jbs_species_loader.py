"""
canopi.scenes.jbs_species_loader.py
================================
Extends canopi.scenes.jbs_loader with per-species leaf tracking.

Re-exports everything from canopi.scenes.jbs_loader and adds:
  - SPECIES_LIST, SPECIES_TO_ID, N_SPECIES  constants
  - load_jbs_scene_species()  — same as load_jbs_scene() but adds
    ``_return_species=True`` behaviour: returns
    (scene, wood, leaf_species_id, species_stats)

The original load_jbs_scene() is also re-exported unchanged.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

# Re-export everything from the base loader
from canopi.scenes.jbs_loader import (
    # constants
    LX, LY, H, LEAF_RADIUS, LAI_EFF, VOXEL_SIZE,
    SPECIES_LEAF_AREA_M2,
    # tables
    SpeciesParams, JBS_SPECIES, JBS_GROUPS, GROUP_TO_SPECIES, SPECIES_WEIGHT,
    # spectral tables
    BANDS, ILLUMINATION, BAND_WAVELENGTH_NM,
    # data containers
    TreeInstance, WoodMesh,
    # file parsers
    parse_scene_def, parse_leaf_positions, parse_obj_mesh,
    parse_treelocations, _instances_from_treelocations,
    # coord helpers
    _rot_z, _apply_tree_transform,
    # synthetic generators
    _synthetic_crown_leaves, _synthetic_tree_wood, _synthetic_instances,
    # public API
    load_jbs_scene, get_band_params, jbs_obs_config,
)

# ---------------------------------------------------------------------------
# New constants — deterministic species order
# ---------------------------------------------------------------------------

SPECIES_LIST  = list(JBS_SPECIES.keys())   # ['BEPE','TICO','ALGL','POTR','PIAB','FREX','ACPL']
SPECIES_TO_ID = {s: i for i, s in enumerate(SPECIES_LIST)}
N_SPECIES     = len(SPECIES_LIST)          # 7

# 18 size-class groups (matches _N_SP = 18 in canopi.gpu)
GROUP_LIST  = list(JBS_GROUPS.keys())      # ['ACPL','BEPE1',...,'POTR2']
GROUP_TO_ID = {g: i for i, g in enumerate(GROUP_LIST)}
N_GROUPS    = len(GROUP_LIST)              # 18


# ---------------------------------------------------------------------------
# load_jbs_scene_species()  — species-aware loader
# ---------------------------------------------------------------------------

def load_jbs_scene_species(
    scene_dir: Optional[str],
    band:  str = "PUR",
    month: str = "Jul",
    P_LAI: float = 0.5,
    leaf_radius_std_mul: float = 0.5,
    voxel_size: Optional[float] = None,
    exact_geometry: bool = True,
    seed:  int = 42,
    max_leaves: Optional[int] = None,
    verbose: bool = True,
    return_pools: bool = False,
) -> Tuple:
    """
    Load (or synthesise) the JBS stand with per-leaf species tracking.

    Identical to load_jbs_scene() but returns a 4-tuple:
        (scene, wood, leaf_species_id, species_stats)

    Parameters
    ----------
    (same as load_jbs_scene)
    return_pools : bool
        If True, also extract per-species leaf pools (pre-over-population)
        and return a 5-tuple: (scene, wood, leaf_species_id, species_stats, leaf_pools)

    Returns
    -------
    scene           : canopi.scene.Scene
    wood            : WoodMesh
    leaf_species_id : (N_leaves,) int32  — species index 0..N_SPECIES-1 per leaf
    species_stats   : dict with keys:
        leaf_species_id  — same array
        LAI_Max_sp       — (N_SPECIES,) float64  per-species LAI_Max
        n_leaf_norm_sp   — (N_SPECIES,) float64  per-species n_leaf / surf
        species_list     — SPECIES_LIST
    leaf_pools      : dict (only when return_pools=True)
        {s: {'pos': ndarray(N_s,3), 'nrm': ndarray(N_s,3)}} pre-over-pop leaves
    """
    from canopi.scene import Scene, _assign_leaves_fast, scene_voxel_grid, assign_leaves

    rng = np.random.default_rng(seed)
    from canopi.scenes.jbs_loader import get_band_params as _get_band_params, ILLUMINATION as _ILLUM
    bp    = _get_band_params(band)
    illum = _ILLUM[month]

    use_files = (scene_dir is not None and
                 Path(scene_dir, "scene.def").exists())

    if verbose:
        src = f"RAMI files in '{scene_dir}'" if use_files else "synthetic model"
        print(f"[JBS] Loading HET09_JBS_SUM from {src}")
        print(f"[JBS] Band={band} ({bp['wavelength_nm']:.0f} nm)  "
              f"Month={month}  SZA={illum['sun_zenith_deg']}°  P_LAI={P_LAI}")

    # ── 1. Tree instances ───────────────────────────────────────────────────
    if use_files:
        instances = parse_scene_def(str(Path(scene_dir, "scene.def")))
        if verbose:
            print(f"[JBS]   {len(instances)} tree instances from scene.def")
    else:
        instances = _synthetic_instances(rng)
        if verbose:
            print(f"[JBS]   {len(instances)} synthetic tree instances")

    # ── 2. Collect leaf positions, normals, species ids, wood triangles ────
    all_pos: List[np.ndarray] = []
    all_nrm: List[np.ndarray] = []
    all_sid: List[np.ndarray] = []   # <-- new: per-leaf species id
    all_vtx: List[np.ndarray] = []
    all_fcs: List[np.ndarray] = []
    all_fsp: List[np.ndarray] = []
    vtx_offset = 0

    for inst in instances:
        sp = JBS_SPECIES.get(inst.species)
        if sp is None:
            continue

        # ── Foliage ──────────────────────────────────────────────────────
        if use_files:
            lp_path = Path(scene_dir, inst.species, f"{inst.stem_id}.lp")
            if lp_path.exists():
                loc_pos, loc_nrm = parse_leaf_positions(str(lp_path))
            else:
                loc_pos, loc_nrm = _synthetic_crown_leaves(inst, sp, rng)
        else:
            loc_pos, loc_nrm = _synthetic_crown_leaves(inst, sp, rng)

        if len(loc_pos) == 0:
            continue

        w_pos, w_nrm = _apply_tree_transform(loc_pos, loc_nrm, inst)

        # Wrap x/y; shift z to the canopi convention (soil at -H)
        w_pos[:, 0] %= LX
        w_pos[:, 1] %= LY
        w_pos[:, 2] = np.clip(w_pos[:, 2], 0.0, H)
        w_pos[:, 2] = w_pos[:, 2] - H  # height above soil -> z = h - H (soil at -H, sky at 0); was -h, which turned the stand upside down

        all_pos.append(w_pos)
        all_nrm.append(w_nrm)
        # species id for every leaf of this tree
        all_sid.append(
            np.full(len(w_pos), SPECIES_TO_ID[inst.species], dtype=np.int32)
        )

        # ── Wood ─────────────────────────────────────────────────────────
        if use_files:
            obj_path = Path(scene_dir, inst.species, f"{inst.stem_id}.obj")
            if obj_path.exists():
                loc_vtx, loc_fcs = parse_obj_mesh(str(obj_path))
            else:
                loc_vtx, loc_fcs = _synthetic_tree_wood(inst, sp, rng)
        else:
            loc_vtx, loc_fcs = _synthetic_tree_wood(inst, sp, rng)

        if len(loc_vtx) > 0:
            w_vtx, _ = _apply_tree_transform(
                loc_vtx, np.zeros_like(loc_vtx), inst
            )
            w_vtx[:, 0] %= LX
            w_vtx[:, 1] %= LY
            w_vtx[:, 2] = np.clip(w_vtx[:, 2], 0.0, H)
            w_vtx[:, 2] = w_vtx[:, 2] - H
            all_vtx.append(w_vtx)
            all_fcs.append(loc_fcs + vtx_offset)
            all_fsp.append(np.full(len(loc_fcs), inst.species, dtype=object))
            vtx_offset += len(w_vtx)

    # ── 3. Concatenate ──────────────────────────────────────────────────────
    if not all_pos:
        raise RuntimeError("No leaves generated — check species table / files.")

    leaf_pos = np.vstack(all_pos)
    leaf_nrm = np.vstack(all_nrm)
    leaf_sid = np.concatenate(all_sid) if all_sid else np.array([], dtype=np.int32)

    if max_leaves is not None and len(leaf_pos) > max_leaves:
        idx = rng.choice(len(leaf_pos), max_leaves, replace=False)
        leaf_pos = leaf_pos[idx]
        leaf_nrm = leaf_nrm[idx]
        leaf_sid = leaf_sid[idx]
        if verbose:
            print(f"[JBS]   Leaves capped to {max_leaves:,}")

    n_leaf_rami = len(leaf_pos)

    # ── Over-populate by 1/P_LAI ──────────────────────────────────────────
    n_leaf_target = max(n_leaf_rami, round(n_leaf_rami / P_LAI))
    n_extra = n_leaf_target - n_leaf_rami
    if n_extra > 0:
        idx_extra = rng.choice(n_leaf_rami, n_extra, replace=True)
        jitter    = rng.normal(0.0, LEAF_RADIUS * 0.1, (n_extra, 3))
        jitter[:, 2] = 0.0
        extra_pos = leaf_pos[idx_extra] + jitter
        extra_pos[:, 0] %= LX
        extra_pos[:, 1] %= LY
        extra_pos[:, 2] = np.clip(extra_pos[:, 2], -H, 0.0)
        extra_nrm = leaf_nrm[idx_extra]
        extra_sid = leaf_sid[idx_extra]
        leaf_pos = np.vstack([leaf_pos, extra_pos])
        leaf_nrm = np.vstack([leaf_nrm, extra_nrm])
        leaf_sid = np.concatenate([leaf_sid, extra_sid])
        if verbose:
            print(f"[JBS]   Over-populated: {n_leaf_rami:,} → {len(leaf_pos):,} "
                  f"leaves  (× 1/P_LAI = {1/P_LAI:.1f})")

    # Extract per-species pools from the pre-over-population leaves
    if return_pools:
        leaf_pools = {}
        sid_pre = leaf_sid[:n_leaf_rami]
        for s in range(N_SPECIES):
            mask = sid_pre == s
            leaf_pools[s] = {
                'pos': leaf_pos[:n_leaf_rami][mask].copy(),
                'nrm': leaf_nrm[:n_leaf_rami][mask].copy(),
            }

    n_leaf = len(leaf_pos)
    if verbose:
        print(f"[JBS]   Total leaves: {n_leaf:,}")

    # ── 4. Leaf radii ──────────────────────────────────────────────────────
    std_mul = leaf_radius_std_mul
    LAI_r   = math.pi * LEAF_RADIUS ** 2
    std_LAI = std_mul * LAI_r
    if verbose:
        print(f"[JBS]   leaf_radius_std_mul={std_mul}  →  "
              f"std_LAI={std_LAI:.6g}  (LAI_r={LAI_r:.6g})")

    leaf_radius     = np.full(n_leaf, LEAF_RADIUS, dtype=np.float64)
    r_max_scalar    = math.sqrt(max(LAI_r + 6.0 * std_LAI, 0.0) / math.pi)
    leaf_radius_max = np.maximum(leaf_radius, r_max_scalar)

    surf         = LX * LY
    LAI_Max      = n_leaf * LAI_r / surf
    n_leaf_norm  = n_leaf / surf

    if verbose:
        actual_LAI_eff = LAI_Max * P_LAI
        from canopi.scenes.jbs_loader import LAI_EFF as _LAI_EFF
        print(f"[JBS]   LAI_Max: {LAI_Max:.3f}  "
              f"E[LAI_seen]={actual_LAI_eff:.3f}  (target: {_LAI_EFF})")

    # ── 5. Voxel grid ──────────────────────────────────────────────────────
    pos_max   = np.array([LX, LY, -H], dtype=np.float64)
    voxel_size, max_voxel = scene_voxel_grid(pos_max, voxel_size, exact_geometry, VOXEL_SIZE)
    n_vox     = int(max_voxel[0] * max_voxel[1] * max_voxel[2])

    if verbose:
        print(f"[JBS]   Voxel grid: {max_voxel[0]}×{max_voxel[1]}×{max_voxel[2]}"
              f"  ({n_vox:,} voxels)")

    voxel_count, voxel_leaves = assign_leaves(leaf_pos, leaf_radius_max, voxel_size, max_voxel, exact_geometry)

    if verbose:
        occupied = int((voxel_count > 0).sum())
        print(f"[JBS]   Occupied voxels: {occupied:,} / {n_vox:,}  "
              f"({100*occupied/n_vox:.1f}%)")

    # ── 6. Build Scene ──────────────────────────────────────────────────────
    nz           = np.clip(leaf_nrm[:, 2], -1.0, 1.0)
    leaf_zenith  = np.arccos(nz)
    leaf_azimuth = np.arctan2(leaf_nrm[:, 1], leaf_nrm[:, 0])

    scene = Scene(
        leaf_pos=leaf_pos,
        leaf_normal=leaf_nrm,
        leaf_zenith=leaf_zenith,
        leaf_azimuth=leaf_azimuth,
        leaf_radius=leaf_radius.copy(),
        leaf_radius_max=leaf_radius_max,
        pos_max=pos_max,
        voxel_size=float(voxel_size),
        max_voxel=max_voxel,
        voxel_count=voxel_count,
        voxel_leaves=voxel_leaves,
        rho=float(bp["rho"]),
        tau=float(bp["tau"]),
        rsoil=float(bp["rsoil"]),
        LAI_r=float(LAI_r),
        std_LAI=float(std_LAI),
        LAI_Max=float(LAI_Max),
        P_LAI=float(P_LAI),
        n_leaf_norm=float(n_leaf_norm),
    )

    # ── 7. WoodMesh ─────────────────────────────────────────────────────────
    if all_vtx:
        wood = WoodMesh(
            vertices=np.vstack(all_vtx),
            faces=np.vstack(all_fcs).reshape(-1, 3).astype(np.int32),
            face_species=np.concatenate(all_fsp),
        )
    else:
        wood = WoodMesh(
            vertices=np.zeros((0, 3)),
            faces=np.zeros((0, 3), dtype=np.int32),
            face_species=np.array([], dtype=object),
        )

    if verbose:
        print(f"[JBS]   Wood mesh: {wood.n_vertices:,} vertices, "
              f"{wood.n_faces:,} triangles")
        print(f"[JBS]   rho={scene.rho:.5f}  tau={scene.tau:.5f}  "
              f"rsoil={scene.rsoil:.5f}")
        print(f"[JBS]   Scene ready.")

    # ── 8. Per-species statistics ────────────────────────────────────────────
    leaf_sid = leaf_sid.astype(np.int32)
    LAI_Max_sp     = np.zeros(N_SPECIES, dtype=np.float64)
    n_leaf_norm_sp = np.zeros(N_SPECIES, dtype=np.float64)
    for s in range(N_SPECIES):
        cnt = int((leaf_sid == s).sum())
        LAI_Max_sp[s]     = cnt * LAI_r / surf
        n_leaf_norm_sp[s] = cnt / surf

    species_stats = dict(
        leaf_species_id=leaf_sid,
        LAI_Max_sp=LAI_Max_sp,
        n_leaf_norm_sp=n_leaf_norm_sp,
        species_list=SPECIES_LIST,
    )

    if return_pools:
        return scene, wood, leaf_sid, species_stats, leaf_pools
    return scene, wood, leaf_sid, species_stats


# ---------------------------------------------------------------------------
# build_scene_for_lai()  — rebuild Scene with arbitrary per-species LAI
# ---------------------------------------------------------------------------

def build_scene_for_lai(
    leaf_pools: dict,       # {sp_idx: {'pos': ndarray(N,3), 'nrm': ndarray(N,3)}}
    LAI_eff_sp: np.ndarray, # (N_SPECIES,) target effective LAI per species
    P_LAI: float,
    band_params: dict,      # keys: rho, tau, rsoil
    voxel_size: Optional[float] = None,
    exact_geometry: bool = True,
    seed: int = 0,
    verbose: bool = False,
    nested_species: set = None,  # species indices using sorted (nested) selection
) -> tuple:
    """
    Build a Scene with per-species leaf counts matching LAI_eff_sp.

    For species s: n_leaf_s = round(LAI_eff_sp[s] / P_LAI * surf / LAI_r)
    Leaves are sampled (with replacement if needed) from leaf_pools[s].
    The full leaf set is then over-populated by 1/P_LAI for derivative estimation.

    Returns: (scene, leaf_sid, species_stats)
      scene        — canopi.scene.Scene ready for path tracer
      leaf_sid     — (N_leaves,) int32 species index per leaf
      species_stats — same dict structure as load_jbs_scene_species
    """
    from canopi.scene import Scene, _assign_leaves_fast, scene_voxel_grid, assign_leaves
    import math

    rng  = np.random.default_rng(seed)
    surf = LX * LY
    LAI_r = math.pi * LEAF_RADIUS ** 2

    all_pos, all_nrm, all_sid = [], [], []

    for s in range(N_SPECIES):
        pool_pos = leaf_pools[s]['pos']
        pool_nrm = leaf_pools[s]['nrm']
        n_pool   = len(pool_pos)

        # target leaf count for this species
        n_target = max(1, round(LAI_eff_sp[s] / P_LAI * surf / LAI_r))

        if nested_species is not None and s in nested_species:
            # Deterministic: take first n_target from the (pre-sorted) pool.
            # Leaf sets are nested — larger n_target is a superset of smaller.
            n_take = min(n_target, n_pool)
            idx    = np.arange(n_take)
            if n_target > n_pool:
                idx = np.concatenate([idx, rng.choice(n_pool, n_target - n_pool, replace=True)])
        elif n_target <= n_pool:
            idx = rng.choice(n_pool, n_target, replace=False)
        else:
            idx = rng.choice(n_pool, n_target, replace=True)

        all_pos.append(pool_pos[idx])
        all_nrm.append(pool_nrm[idx])
        all_sid.append(np.full(n_target, s, dtype=np.int32))

    leaf_pos = np.vstack(all_pos)
    leaf_nrm = np.vstack(all_nrm)
    leaf_sid = np.concatenate(all_sid)
    n_leaf_base = len(leaf_pos)

    # Over-populate by 1/P_LAI
    n_target_total = max(n_leaf_base, round(n_leaf_base / P_LAI))
    n_extra = n_target_total - n_leaf_base
    if n_extra > 0:
        idx_ex = rng.choice(n_leaf_base, n_extra, replace=True)
        jitter = rng.normal(0.0, LEAF_RADIUS * 0.1, (n_extra, 3))
        jitter[:, 2] = 0.0
        ex_pos = leaf_pos[idx_ex] + jitter
        ex_pos[:, 0] %= LX; ex_pos[:, 1] %= LY
        ex_pos[:, 2] = np.clip(ex_pos[:, 2], -H, 0.0)
        leaf_pos = np.vstack([leaf_pos, ex_pos])
        leaf_nrm = np.vstack([leaf_nrm, leaf_nrm[idx_ex]])
        leaf_sid = np.concatenate([leaf_sid, leaf_sid[idx_ex]])

    n_leaf = len(leaf_pos)
    leaf_radius = np.full(n_leaf, LEAF_RADIUS, dtype=np.float64)
    r_max_scalar = math.sqrt(LAI_r / math.pi)
    leaf_radius_max = np.maximum(leaf_radius, r_max_scalar)

    LAI_Max     = n_leaf * LAI_r / surf
    n_leaf_norm = n_leaf / surf

    pos_max   = np.array([LX, LY, -H], dtype=np.float64)
    voxel_size, max_voxel = scene_voxel_grid(pos_max, voxel_size, exact_geometry, VOXEL_SIZE)
    voxel_count, voxel_leaves = assign_leaves(leaf_pos, leaf_radius_max, voxel_size, max_voxel, exact_geometry)

    nz = np.clip(leaf_nrm[:, 2], -1.0, 1.0)
    scene = Scene(
        leaf_pos=leaf_pos, leaf_normal=leaf_nrm,
        leaf_zenith=np.arccos(nz),
        leaf_azimuth=np.arctan2(leaf_nrm[:, 1], leaf_nrm[:, 0]),
        leaf_radius=leaf_radius.copy(),
        leaf_radius_max=leaf_radius_max,
        pos_max=pos_max,
        voxel_size=float(voxel_size),
        max_voxel=max_voxel,
        voxel_count=voxel_count,
        voxel_leaves=voxel_leaves,
        rho=float(band_params['rho']),
        tau=float(band_params['tau']),
        rsoil=float(band_params['rsoil']),
        LAI_r=float(LAI_r),
        std_LAI=0.0,
        LAI_Max=float(LAI_Max),
        P_LAI=float(P_LAI),
        n_leaf_norm=float(n_leaf_norm),
    )

    leaf_sid = leaf_sid.astype(np.int32)
    LAI_Max_sp     = np.zeros(N_SPECIES, dtype=np.float64)
    n_leaf_norm_sp = np.zeros(N_SPECIES, dtype=np.float64)
    for s in range(N_SPECIES):
        cnt = int((leaf_sid == s).sum())
        LAI_Max_sp[s]     = cnt * LAI_r / surf
        n_leaf_norm_sp[s] = cnt / surf

    species_stats = dict(
        leaf_species_id=leaf_sid,
        LAI_Max_sp=LAI_Max_sp,
        n_leaf_norm_sp=n_leaf_norm_sp,
        species_list=SPECIES_LIST,
    )
    return scene, leaf_sid, species_stats


# ---------------------------------------------------------------------------
# load_jbs_scene_groups() — 18 size-class groups, matches _N_SP=18 in GPU
# ---------------------------------------------------------------------------

def load_jbs_scene_groups(
    scene_dir: Optional[str],
    band:      str   = "O08",
    month:     str   = "Jul",
    P_LAI:     float = 0.5,
    voxel_size: Optional[float] = None,
    exact_geometry: bool = True,
    seed:      int   = 42,
    max_leaves: Optional[int] = None,
    verbose:   bool  = True,
) -> Tuple:
    """
    Load JBS stand with per-leaf GROUP tracking (18 size classes).

    Returns
    -------
    scene        : canopi.scene.Scene
    wood         : WoodMesh
    leaf_group_id: (N_leaves,) int32  group index 0..17 per leaf
    group_stats  : dict with keys:
        LAI_Max_sp     — (18,) float64  per-group LAI_Max (incl. fictive)
        n_leaf_norm_sp — (18,) float64  per-group n_leaf / surf
        rho_sp_arr     — (18,) float64  per-group leaf rho
        species_list   — GROUP_LIST
    """
    from canopi.scene import Scene, _assign_leaves_fast, scene_voxel_grid, assign_leaves

    rng   = np.random.default_rng(seed)
    bp    = get_band_params(band)
    illum = ILLUMINATION[month]
    surf  = LX * LY

    # ── 1. Tree instances ───────────────────────────────────────────────────
    use_treeloc = (scene_dir is not None and
                   Path(scene_dir, "ACPL_treelocations.dat").exists())
    if use_treeloc:
        instances = _instances_from_treelocations(scene_dir, rng)
        if verbose:
            print(f"[JBS-groups] {len(instances)} instances from treelocations")
    else:
        instances = _synthetic_instances(rng)
        if verbose:
            print(f"[JBS-groups] {len(instances)} synthetic instances")

    # ── 2. Collect leaves with per-leaf group ID ────────────────────────────
    all_pos, all_nrm, all_gid, all_rad = [], [], [], []
    all_vtx, all_fcs, all_fsp = [], [], []
    vtx_offset = 0

    for inst in instances:
        grp = JBS_GROUPS.get(inst.group) or JBS_GROUPS.get(inst.species)
        if grp is None:
            continue
        gid = GROUP_TO_ID.get(inst.group, GROUP_TO_ID.get(inst.species, 0))
        sp_code   = GROUP_TO_SPECIES.get(inst.group, inst.species)
        leaf_area = SPECIES_LEAF_AREA_M2.get(sp_code, math.pi * LEAF_RADIUS**2)
        r_sp      = math.sqrt(leaf_area / math.pi)

        loc_pos, loc_nrm = _synthetic_crown_leaves(inst, grp, rng)
        if len(loc_pos) == 0:
            continue

        w_pos, w_nrm = _apply_tree_transform(loc_pos, loc_nrm, inst)
        w_pos[:, 0] %= LX
        w_pos[:, 1] %= LY
        w_pos[:, 2]  = np.clip(w_pos[:, 2], 0.0, H)
        w_pos[:, 2]  = w_pos[:, 2] - H  # height above soil -> z = h - H (soil at -H, sky at 0); was -h, which turned the stand upside down

        all_pos.append(w_pos)
        all_nrm.append(w_nrm)
        all_gid.append(np.full(len(w_pos), gid,  dtype=np.int32))
        all_rad.append(np.full(len(w_pos), r_sp, dtype=np.float64))

        loc_vtx, loc_fcs = _synthetic_tree_wood(inst, grp, rng)
        if len(loc_vtx) > 0:
            w_vtx, _ = _apply_tree_transform(loc_vtx, np.zeros_like(loc_vtx), inst)
            w_vtx[:, 0] %= LX; w_vtx[:, 1] %= LY
            w_vtx[:, 2] = np.clip(w_vtx[:, 2], 0.0, H); w_vtx[:, 2] = w_vtx[:, 2] - H
            all_vtx.append(w_vtx)
            all_fcs.append(loc_fcs + vtx_offset)
            all_fsp.append(np.full(len(loc_fcs), inst.species, dtype=object))
            vtx_offset += len(w_vtx)

    if not all_pos:
        raise RuntimeError("No leaves generated.")

    leaf_pos = np.vstack(all_pos)
    leaf_nrm = np.vstack(all_nrm)
    leaf_gid = np.concatenate(all_gid)
    leaf_rad = np.concatenate(all_rad)

    if max_leaves is not None and len(leaf_pos) > max_leaves:
        idx = rng.choice(len(leaf_pos), max_leaves, replace=False)
        leaf_pos = leaf_pos[idx]; leaf_nrm = leaf_nrm[idx]
        leaf_gid = leaf_gid[idx]; leaf_rad = leaf_rad[idx]

    n_leaf_rami = len(leaf_pos)

    # ── Over-populate by 1/P_LAI ──────────────────────────────────────────
    n_extra = max(0, round(n_leaf_rami / P_LAI) - n_leaf_rami)
    if n_extra > 0:
        idx_ex = rng.choice(n_leaf_rami, n_extra, replace=True)
        jitter = rng.normal(0.0, float(leaf_rad.mean()) * 0.1, (n_extra, 3))
        jitter[:, 2] = 0.0
        ex_pos = leaf_pos[idx_ex] + jitter
        ex_pos[:, 0] %= LX; ex_pos[:, 1] %= LY
        ex_pos[:, 2] = np.clip(ex_pos[:, 2], -H, 0.0)
        leaf_pos = np.vstack([leaf_pos, ex_pos])
        leaf_nrm = np.vstack([leaf_nrm, leaf_nrm[idx_ex]])
        leaf_gid = np.concatenate([leaf_gid, leaf_gid[idx_ex]])
        leaf_rad = np.concatenate([leaf_rad, leaf_rad[idx_ex]])

    n_leaf  = len(leaf_pos)
    LAI_r   = float(np.mean(np.pi * leaf_rad**2))
    std_LAI = 0.0
    LAI_Max = float(np.sum(np.pi * leaf_rad**2)) / surf
    n_leaf_norm = n_leaf / surf

    if verbose:
        print(f"[JBS-groups] {n_leaf:,} leaves  LAI_Max={LAI_Max:.3f}  "
              f"E[LAI]={LAI_Max*P_LAI:.3f}")

    # ── Voxel grid ─────────────────────────────────────────────────────────
    pos_max   = np.array([LX, LY, -H], dtype=np.float64)
    voxel_size, max_voxel = scene_voxel_grid(pos_max, voxel_size, exact_geometry, VOXEL_SIZE)
    voxel_count, voxel_leaves = assign_leaves(leaf_pos, leaf_rad, voxel_size, max_voxel, exact_geometry)

    # ── Scene ───────────────────────────────────────────────────────────────
    nz = np.clip(leaf_nrm[:, 2], -1.0, 1.0)
    scene = Scene(
        leaf_pos=leaf_pos, leaf_normal=leaf_nrm,
        leaf_zenith=np.arccos(nz),
        leaf_azimuth=np.arctan2(leaf_nrm[:, 1], leaf_nrm[:, 0]),
        leaf_radius=leaf_rad.copy(), leaf_radius_max=leaf_rad.copy(),
        pos_max=pos_max, voxel_size=float(voxel_size),
        max_voxel=max_voxel, voxel_count=voxel_count, voxel_leaves=voxel_leaves,
        rho=float(bp["rho"]), tau=float(bp["tau"]), rsoil=float(bp["rsoil"]),
        LAI_r=float(LAI_r), std_LAI=std_LAI,
        LAI_Max=float(LAI_Max), P_LAI=float(P_LAI),
        n_leaf_norm=float(n_leaf_norm),
    )

    # ── Wood ────────────────────────────────────────────────────────────────
    if all_vtx:
        wood = WoodMesh(np.vstack(all_vtx),
                        np.vstack(all_fcs).reshape(-1, 3).astype(np.int32),
                        np.concatenate(all_fsp))
    else:
        wood = WoodMesh(np.zeros((0,3)), np.zeros((0,3), dtype=np.int32),
                        np.array([], dtype=object))

    # ── Per-group stats ─────────────────────────────────────────────────────
    leaf_gid       = leaf_gid.astype(np.int32)
    LAI_Max_sp     = np.zeros(N_GROUPS, dtype=np.float64)
    n_leaf_norm_sp = np.zeros(N_GROUPS, dtype=np.float64)
    rho_sp_arr     = np.zeros(N_GROUPS, dtype=np.float64)

    for g, grp_name in enumerate(GROUP_LIST):
        mask = leaf_gid == g
        cnt  = int(mask.sum())
        r_g  = float(leaf_rad[mask].mean()) if cnt > 0 else LEAF_RADIUS
        LAI_r_g = math.pi * r_g**2
        LAI_Max_sp[g]     = cnt * LAI_r_g / surf
        n_leaf_norm_sp[g] = cnt / surf
        sp_code = GROUP_TO_SPECIES[grp_name]
        # Per-group rho from band-specific spectral table
        from canopi.scenes.jbs_loader import _REFL, _BAND_IDX
        bi = _BAND_IDX.get(band, 3)
        rho_sp_arr[g] = _REFL[sp_code][bi]

    group_stats = dict(
        LAI_Max_sp=LAI_Max_sp,
        n_leaf_norm_sp=n_leaf_norm_sp,
        rho_sp_arr=rho_sp_arr,
        species_list=GROUP_LIST,
    )

    if verbose:
        print(f"[JBS-groups] Per-group LAI_Max:")
        for g, grp_name in enumerate(GROUP_LIST):
            print(f"  {grp_name:<6}: LAI_Max={LAI_Max_sp[g]:.4f}")

    return scene, wood, leaf_gid, group_stats

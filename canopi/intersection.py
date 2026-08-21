"""
Canopy RT — Ray-disk intersection with voxel DDA, JIT-compiled with Numba.

Numba constraints:
  - No Python sets/dicts/lists → fixed-size np arrays + counters
  - All inner loops decorated @nb.njit
  - Public wrappers handle scene unpacking
"""
from __future__ import annotations
import numpy as np
import numba as nb
from .scene import Scene


# ─────────────────────────────────────────────────────────────────────────────
#  Numba-compiled core
# ─────────────────────────────────────────────────────────────────────────────

@nb.njit(cache=True)
def _pos3d_idx(vi, vj, vk, mx, my):
    """Flat voxel index from (i,j,k) 0-based triple."""
    return vi + vj * mx + vk * mx * my


@nb.njit(cache=True)
def _find_contact_nb(
    # ray
    Px, Py, Pz,
    vx, vy, vz,
    origin_leaf,          # int, leaf to skip
    # leaves
    leaf_px, leaf_py, leaf_pz,     # (N,) positions
    leaf_nx, leaf_ny, leaf_nz,     # (N,) normals
    leaf_radius,                    # (N,) radii
    leaf_effect,                    # (N,) int8: 1=real, 0=fictive
    # voxel grid
    voxel_count,           # (nv,) int32
    voxel_leaves,          # (nv, cap) int32
    vs,                    # float: voxel size
    mvx, mvy, mvz,         # int: max voxels per axis
    # derivative params
    mean_r, std_r,
    edge_threshold,
    collect_edge,          # bool
    max_steps,             # int
    max_out,               # int  — max fictive/edge results
):
    """
    DDA voxel traversal + ray-disk intersection.

    Returns (contact, leaf_idx, Hx, Hy, Hz, n_v_contact,
             fic_ids, fic_nvs, n_fic,
             edg_ids, edg_nvs, n_edg)
    where fic_ids/edg_ids are fixed arrays of length max_out.
    """
    EPS = 1e-8

    # clamp origin
    _Px = min(max(Px, EPS), vs * mvx - EPS)
    _Py = min(max(Py, EPS), vs * mvy - EPS)
    _Pz = min(max(Pz, -vs * mvz + EPS), -EPS)

    # initial voxel (abs-indexed)
    cur_i = min(int(abs(_Px) / vs), mvx - 1)
    cur_j = min(int(abs(_Py) / vs), mvy - 1)
    cur_k = min(int(abs(_Pz) / vs), mvz - 1)

    cur_px, cur_py, cur_pz = _Px, _Py, _Pz
    accumulated_t = 0.0   # total DDA distance from ray origin

    # accumulators
    contact     = False
    inter_leaf  = -1
    Hx = Hy = Hz = 0.0
    n_v_contact = 0.0
    min_dis     = 1e18   # GLOBAL distance from origin (= accumulated_t + OH)

    # tested array (bool per leaf) – allocated once, kept as int8
    n_leaf = leaf_px.shape[0]
    tested = np.zeros(n_leaf, dtype=nb.int8)

    fic_ids  = np.full(max_out, -1, dtype=nb.int32)
    fic_nvs  = np.zeros(max_out, dtype=nb.float64)
    fic_dists= np.zeros(max_out, dtype=nb.float64)
    n_fic    = 0

    edg_ids  = np.full(max_out, -1, dtype=nb.int32)
    edg_nvs  = np.zeros(max_out, dtype=nb.float64)
    edg_dists= np.zeros(max_out, dtype=nb.float64)
    n_edg    = 0

    edge_r_thresh = mean_r   # mean_r = r_max_scalar (max resampled radius)

    for _step in range(max_steps):
        vox_idx  = _pos3d_idx(cur_i, cur_j, cur_k, mvx, mvy)
        n_in_vox = voxel_count[vox_idx]

        for k in range(n_in_vox):
            i = voxel_leaves[vox_idx, k]
            if i == origin_leaf or tested[i]:
                continue
            tested[i] = 1

            # leaf plane intersection — use cur_px (wrapped) as reference
            ptx = leaf_px[i] - cur_px
            pty = leaf_py[i] - cur_py
            ptz = leaf_pz[i] - cur_pz
            # Shortest-path correction for x/y periodicity
            Lx_ = mvx * vs; Ly_ = mvy * vs
            # if ptx >  Lx_ * 0.5: ptx -= Lx_
            # if ptx < -Lx_ * 0.5: ptx += Lx_
            # if pty >  Ly_ * 0.5: pty -= Ly_
            # if pty < -Ly_ * 0.5: pty += Ly_
            denom = vx * leaf_nx[i] + vy * leaf_ny[i] + vz * leaf_nz[i]
            if abs(denom) < 1e-12:
                continue
            OH = (ptx * leaf_nx[i] + pty * leaf_ny[i] + ptz * leaf_nz[i]) / denom
            # OH is local (from cur_px); global distance from ray origin:
            oh_global = accumulated_t + OH
            if OH < 0.0 or oh_global >= min_dis:
                continue

            MHx = -ptx + OH * vx
            MHy = -pty + OH * vy
            MHz = -ptz + OH * vz
            n_v = (MHx*MHx + MHy*MHy + MHz*MHz) ** 0.5
            r_i = leaf_radius[i]

            if n_v < r_i:
                if leaf_effect[i]:
                    inter_leaf  = i
                    min_dis     = oh_global   # store GLOBAL distance
                    n_v_contact = n_v
                    contact     = True
                    Hx = cur_px + OH * vx
                    Hy = cur_py + OH * vy
                    Hz = cur_pz + OH * vz
                else:
                    if n_fic < max_out:
                        fic_ids[n_fic]   = i
                        fic_nvs[n_fic]   = n_v
                        fic_dists[n_fic] = oh_global   # store GLOBAL distance
                        n_fic += 1
            else:
                if collect_edge and n_v < edge_r_thresh:
                    if n_edg < max_out:
                        edg_ids[n_edg]   = i
                        edg_nvs[n_edg]   = n_v
                        edg_dists[n_edg] = oh_global   # store GLOBAL distance
                        n_edg += 1

        # ── DDA step ──────────────────────────────────────────────────────
        t_min = 1e18
        ni, nj, nk = cur_i, cur_j, cur_k  # candidate next voxel

        # x-axis
        if abs(vx) > 1e-12:
            eff_x = 1 if vx > 0 else -1
            face_x = (cur_i + (1 if eff_x > 0 else 0)) * vs
            tx = abs(face_x - abs(cur_px)) / abs(vx)
            if tx <= EPS:
                ni2 = cur_i + eff_x
                face_x = (ni2 + (1 if eff_x > 0 else 0)) * vs
                tx = abs(face_x - abs(cur_px)) / abs(vx)
            if tx > EPS and tx < t_min:
                t_min = tx
                ni = cur_i + eff_x; nj = cur_j; nk = cur_k

        # y-axis
        if abs(vy) > 1e-12:
            eff_y = 1 if vy > 0 else -1
            face_y = (cur_j + (1 if eff_y > 0 else 0)) * vs
            ty = abs(face_y - abs(cur_py)) / abs(vy)
            if ty <= EPS:
                nj2 = cur_j + eff_y
                face_y = (nj2 + (1 if eff_y > 0 else 0)) * vs
                ty = abs(face_y - abs(cur_py)) / abs(vy)
            if ty > EPS and ty < t_min:
                t_min = ty
                ni = cur_i; nj = cur_j + eff_y; nk = cur_k

        # z-axis (abs(z) increases as z decreases → eff = -sign(vz))
        if abs(vz) > 1e-12:
            eff_z = -1 if vz > 0 else 1
            face_z = (cur_k + (1 if eff_z > 0 else 0)) * vs
            tz = abs(face_z - abs(cur_pz)) / abs(vz)
            if tz <= EPS:
                nk2 = cur_k + eff_z
                face_z = (nk2 + (1 if eff_z > 0 else 0)) * vs
                tz = abs(face_z - abs(cur_pz)) / abs(vz)
            if tz > EPS and tz < t_min:
                t_min = tz
                ni = cur_i; nj = cur_j; nk = cur_k + eff_z

        # z out-of-bounds: ray exits top or bottom of canopy → stop
        if nk < 0 or nk >= mvz:
            break

        # x/y out-of-bounds: periodic BC → wrap around
        ni = ni % mvx
        nj = nj % mvy

        # stop if next voxel is beyond the hit (compare global distances)
        if contact and accumulated_t + t_min > min_dis:
            break

        cur_i, cur_j, cur_k = ni, nj, nk
        cur_px += t_min * vx + (1e-8 if vx > 0 else -1e-8 if vx < 0 else 0.0)
        cur_py += t_min * vy + (1e-8 if vy > 0 else -1e-8 if vy < 0 else 0.0)
        cur_pz += t_min * vz + (1e-8 if vz > 0 else -1e-8 if vz < 0 else 0.0)
        accumulated_t += t_min
        # wrap physical position for correct voxel index on next step
        cur_px = cur_px % (mvx * vs)
        cur_py = cur_py % (mvy * vs)

    # trim to before contact
    if contact:
        n_fic_out = 0
        for q in range(n_fic):
            if fic_dists[q] < min_dis:
                fic_ids[n_fic_out]  = fic_ids[q]
                fic_nvs[n_fic_out]  = fic_nvs[q]
                n_fic_out += 1
        n_fic = n_fic_out

        n_edg_out = 0
        for q in range(n_edg):
            if edg_dists[q] < min_dis:
                edg_ids[n_edg_out] = edg_ids[q]
                edg_nvs[n_edg_out] = edg_nvs[q]
                n_edg_out += 1
        n_edg = n_edg_out

    return (contact, inter_leaf, Hx, Hy, Hz, n_v_contact,
            fic_ids, fic_nvs, n_fic,
            edg_ids, edg_nvs, n_edg)


# ─────────────────────────────────────────────────────────────────────────────
#  Python wrappers (unpack Scene → numba-compatible arrays)
# ─────────────────────────────────────────────────────────────────────────────

def _scene_arrays(scene: Scene):
    """Pre-extract scene arrays once per call site."""
    mv = scene.max_voxel
    return (
        scene.leaf_pos[:, 0].copy(), scene.leaf_pos[:, 1].copy(), scene.leaf_pos[:, 2].copy(),
        scene.leaf_normal[:, 0].copy(), scene.leaf_normal[:, 1].copy(), scene.leaf_normal[:, 2].copy(),
        scene.leaf_radius,
        scene.voxel_count,
        scene.voxel_leaves,
        scene.voxel_size,
        int(mv[0]), int(mv[1]), int(mv[2]),
        float(np.max(scene.leaf_radius_max)),    # r_max_scalar: threshold for set B
        0.0,   # std_r not used: edge_r_thresh = mean_r + 0*std_r = r_max_scalar
        int(np.sum(mv)) * 4 + 20,   # max_steps
    )


def find_contact(
    scene:          Scene,
    P:              np.ndarray,
    vec_P:          np.ndarray,
    origin_leaf:    int,
    leaf_effect:    np.ndarray,
    edge_threshold: float = 2.5,
    collect_edge:   bool  = True,
    max_out:        int   = 200,
):
    """
    Find first ray-disk intersection along (P, vec_P).

    Returns
    -------
    contact        : bool
    leaf_idx       : int
    H              : (3,) hit point or None
    n_v_contact    : float
    fictive_leaves : list[(leaf_id, n_v)]
    edge_leaves    : list[(leaf_id, n_v)]
    """
    (lpx, lpy, lpz, lnx, lny, lnz,
     lrad, vcnt, vlvs, vs, mx, my, mz,
     mean_r, std_r, max_steps) = _scene_arrays(scene)

    le = leaf_effect.astype(np.int8)

    (contact, leaf_idx, Hx, Hy, Hz, n_v,
     fic_ids, fic_nvs, n_fic,
     edg_ids, edg_nvs, n_edg) = _find_contact_nb(
        float(P[0]), float(P[1]), float(P[2]),
        float(vec_P[0]), float(vec_P[1]), float(vec_P[2]),
        int(origin_leaf),
        lpx, lpy, lpz, lnx, lny, lnz, lrad, le,
        vcnt, vlvs, vs, mx, my, mz,
        mean_r, std_r,
        float(edge_threshold), bool(collect_edge),
        max_steps, max_out,
    )

    if contact:
        # Wrap x,y into [0, Lx] x [0, Ly] — cur_px is not wrapped inside
        # the Numba DDA so Hx/Hy may be slightly outside the scene bounds
        # when the contact leaf is near a periodic boundary.
        Lx = float(mx) * float(vs)
        Ly = float(my) * float(vs)
        Hx = Hx % Lx
        Hy = Hy % Ly
    H              = np.array([Hx, Hy, Hz]) if contact else None
    fictive_leaves = [(int(fic_ids[q]), float(fic_nvs[q])) for q in range(n_fic)]
    edge_leaves    = [(int(edg_ids[q]), float(edg_nvs[q])) for q in range(n_edg)]

    return contact, int(leaf_idx), H, float(n_v), fictive_leaves, edge_leaves


def check_visibility(
    scene:          Scene,
    P:              np.ndarray,
    vec_obs:        np.ndarray,
    origin_leaf:    int,
    leaf_effect:    np.ndarray,
    edge_threshold: float = 2.5,
):
    """True if P has unobstructed line-of-sight in direction vec_obs."""
    contact, _, _, _, fic, edg = find_contact(
        scene, P, vec_obs, origin_leaf, leaf_effect,
        edge_threshold, collect_edge=True,
    )
    return (not contact), fic, edg


def warmup(scene: Scene):
    """Call once to trigger Numba JIT compilation."""
    rng = np.random.default_rng(0)
    le  = np.ones(scene.n_leaf, dtype=np.int8)
    P   = np.array([scene.pos_max[0]/2, scene.pos_max[1]/2, 0.0])
    v   = np.array([0.0, 0.0, -1.0])
    find_contact(scene, P, v, -1, le)

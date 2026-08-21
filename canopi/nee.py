"""
Canopy RT — Vectorised NEE: all obs directions in one Numba kernel.
"""
from __future__ import annotations
import numpy as np
import numba as nb
from .scene import Scene

PI = 3.141592653589793


@nb.njit(cache=True)
def _nee_all_dirs_nb(
    Px, Py, Pz,
    obs_vx, obs_vy, obs_vz,        # (n_obs,) each
    origin_leaf,
    leaf_px, leaf_py, leaf_pz,
    leaf_nx, leaf_ny, leaf_nz,
    leaf_radius, leaf_effect,
    hit_nr_x, hit_nr_y, hit_nr_z,  # normal of hit surface
    voxel_count, voxel_leaves,
    vs, mvx, mvy, mvz,
    max_steps,
    rho, tau, rsoil,
    is_soil,
    in_vx, in_vy, in_vz,           # incoming ray direction
    max_out,
    mean_r, std_r, edge_threshold,
    collect_deriv,
):
    n_obs = obs_vx.shape[0]
    n_leaf = leaf_px.shape[0]

    vis_flags   = np.zeros(n_obs, dtype=nb.boolean)
    nee_weights = np.zeros(n_obs, dtype=nb.float64)
    fic_ids_all = np.full((n_obs, max_out), -1, dtype=nb.int32)
    fic_nvs_all = np.zeros((n_obs, max_out), dtype=nb.float64)
    n_fic_all   = np.zeros(n_obs, dtype=nb.int32)
    edg_ids_all = np.full((n_obs, max_out), -1, dtype=nb.int32)
    edg_nvs_all = np.zeros((n_obs, max_out), dtype=nb.float64)
    n_edg_all   = np.zeros(n_obs, dtype=nb.int32)

    EPS           = 1e-8
    edge_r_thresh = mean_r   # mean_r = r_max_scalar (max resampled radius)

    for i_obs in range(n_obs):
        vx = obs_vx[i_obs]
        vy = obs_vy[i_obs]
        vz = obs_vz[i_obs]
        if vz <= 0.0:
            continue

        # clamp origin
        _Px = min(max(Px, EPS), vs * mvx - EPS)
        _Py = min(max(Py, EPS), vs * mvy - EPS)
        _Pz = min(max(Pz, -vs * mvz + EPS), -EPS)

        cur_i = min(int(abs(_Px) / vs), mvx - 1)
        cur_j = min(int(abs(_Py) / vs), mvy - 1)
        cur_k = min(int(abs(_Pz) / vs), mvz - 1)
        cur_px = _Px; cur_py = _Py; cur_pz = _Pz

        tested  = np.zeros(n_leaf, dtype=nb.int8)
        n_fic   = 0
        n_edg   = 0
        visible = True

        for _step in range(max_steps):
            vox_idx  = cur_i + cur_j * mvx + cur_k * mvx * mvy
            n_in_vox = voxel_count[vox_idx]

            for k in range(n_in_vox):
                i = voxel_leaves[vox_idx, k]
                if i == origin_leaf or tested[i]:
                    continue
                tested[i] = 1

                # Use cur_px/py/pz (wrapped position) as reference, not Px/Py/Pz.
                # After cyclic wrap, leaves in the new voxel are correctly
                # positioned relative to the current ray position.
                ptx = leaf_px[i] - cur_px
                pty = leaf_py[i] - cur_py
                ptz = leaf_pz[i] - cur_pz
                # Shortest-path correction for x/y periodicity
                Lx_ = mvx * vs; Ly_ = mvy * vs
                if ptx >  Lx_ * 0.5: ptx -= Lx_
                if ptx < -Lx_ * 0.5: ptx += Lx_
                if pty >  Ly_ * 0.5: pty -= Ly_
                if pty < -Ly_ * 0.5: pty += Ly_
                denom = vx*leaf_nx[i] + vy*leaf_ny[i] + vz*leaf_nz[i]
                if abs(denom) < 1e-12:
                    continue
                OH = (ptx*leaf_nx[i] + pty*leaf_ny[i] + ptz*leaf_nz[i]) / denom
                if OH < 0.0:
                    continue

                MHx = -ptx + OH * vx
                MHy = -pty + OH * vy
                MHz = -ptz + OH * vz
                n_v = (MHx*MHx + MHy*MHy + MHz*MHz) ** 0.5

                if n_v < leaf_radius[i]:
                    if leaf_effect[i]:
                        visible = False
                        break
                    elif collect_deriv and n_fic < max_out:
                        fic_ids_all[i_obs, n_fic] = i
                        fic_nvs_all[i_obs, n_fic] = n_v
                        n_fic += 1
                elif collect_deriv and n_v < edge_r_thresh and n_edg < max_out:
                    edg_ids_all[i_obs, n_edg] = i
                    edg_nvs_all[i_obs, n_edg] = n_v
                    n_edg += 1

            if not visible:
                break

            # DDA step
            t_min = 1e18
            ni = cur_i; nj = cur_j; nk = cur_k

            if abs(vx) > 1e-12:
                eff = 1 if vx > 0 else -1
                face = (cur_i + (1 if eff > 0 else 0)) * vs
                tx = abs(face - abs(cur_px)) / abs(vx)
                if tx <= EPS:
                    ni2 = cur_i + eff
                    face = (ni2 + (1 if eff > 0 else 0)) * vs
                    tx = abs(face - abs(cur_px)) / abs(vx)
                if tx > EPS and tx < t_min:
                    t_min = tx; ni = cur_i + eff; nj = cur_j; nk = cur_k

            if abs(vy) > 1e-12:
                eff = 1 if vy > 0 else -1
                face = (cur_j + (1 if eff > 0 else 0)) * vs
                ty = abs(face - abs(cur_py)) / abs(vy)
                if ty <= EPS:
                    nj2 = cur_j + eff
                    face = (nj2 + (1 if eff > 0 else 0)) * vs
                    ty = abs(face - abs(cur_py)) / abs(vy)
                if ty > EPS and ty < t_min:
                    t_min = ty; ni = cur_i; nj = cur_j + eff; nk = cur_k

            if abs(vz) > 1e-12:
                eff = -1 if vz > 0 else 1
                face = (cur_k + (1 if eff > 0 else 0)) * vs
                tz = abs(face - abs(cur_pz)) / abs(vz)
                if tz <= EPS:
                    nk2 = cur_k + eff
                    face = (nk2 + (1 if eff > 0 else 0)) * vs
                    tz = abs(face - abs(cur_pz)) / abs(vz)
                if tz > EPS and tz < t_min:
                    t_min = tz; ni = cur_i; nj = cur_j; nk = cur_k + eff

            # z out-of-bounds → ray exits top or bottom of canopy → visible
            if nk < 0 or nk >= mvz:
                break
            # x/y out-of-bounds → periodic BC: wrap around
            ni = ni % mvx
            nj = nj % mvy

            cur_i = ni; cur_j = nj; cur_k = nk
            cur_px += t_min * vx + (1e-8 if vx > 0 else -1e-8 if vx < 0 else 0.0)
            cur_py += t_min * vy + (1e-8 if vy > 0 else -1e-8 if vy < 0 else 0.0)
            cur_pz += t_min * vz + (1e-8 if vz > 0 else -1e-8 if vz < 0 else 0.0)
            # wrap physical position too
            cur_px = cur_px % (mvx * vs)
            cur_py = cur_py % (mvy * vs)

        vis_flags[i_obs]  = visible
        n_fic_all[i_obs]  = n_fic
        n_edg_all[i_obs]  = n_edg

        if visible:
            # Normalisation géométrique (MATLAB: Norm_fac):
            #   Norm_fac = |dot(leaf_norm, obs_dir)| / dot([0,0,1], obs_dir)
            #            = |cos(angle_feuille_obs)| / cos(θ_obs)
            # Convertit la BRDF locale feuille → BRF référentiel terrestre.
            # Pour le sol (horizontal): leaf_norm=[0,0,1] → Norm_fac = 1
            cos_obs_earth = vz   # dot([0,0,1], obs_dir) = vz = cos(θ_obs)
            if is_soil:
                # Sol horizontal: leaf_norm=[0,0,1], Norm_fac=cos_obs_earth/cos_obs_earth=1
                nee_weights[i_obs] = rsoil / PI
            else:
                cos_in      = -(in_vx*hit_nr_x + in_vy*hit_nr_y + in_vz*hit_nr_z)
                cos_obs_leaf = vx*hit_nr_x + vy*hit_nr_y + vz*hit_nr_z
                # Norm_fac = |dot(leaf_norm, obs_dir)| / cos(θ_obs)
                norm_fac = abs(cos_obs_leaf) / max(cos_obs_earth, 1e-6)
                if cos_in * cos_obs_leaf < 0.0:
                    nee_weights[i_obs] = tau / PI * norm_fac
                else:
                    nee_weights[i_obs] = rho / PI * norm_fac

    return (vis_flags, nee_weights,
            fic_ids_all, fic_nvs_all, n_fic_all,
            edg_ids_all, edg_nvs_all, n_edg_all)


class NEEEngine:
    def __init__(self, scene: Scene, edge_threshold: float = 2.5,
                 max_out: int = 100, collect_deriv: bool = True):
        mv = scene.max_voxel
        self.lpx  = scene.leaf_pos[:, 0].astype(np.float64)
        self.lpy  = scene.leaf_pos[:, 1].astype(np.float64)
        self.lpz  = scene.leaf_pos[:, 2].astype(np.float64)
        self.lnx  = scene.leaf_normal[:, 0].astype(np.float64)
        self.lny  = scene.leaf_normal[:, 1].astype(np.float64)
        self.lnz  = scene.leaf_normal[:, 2].astype(np.float64)
        self.lrad = scene.leaf_radius.astype(np.float64)
        self.vcnt = scene.voxel_count
        self.vlvs = scene.voxel_leaves
        self.vs   = float(scene.voxel_size)
        self.mvx  = int(mv[0]); self.mvy = int(mv[1]); self.mvz = int(mv[2])
        self.max_steps   = int(np.sum(mv)) * 4 + 20
        # r_max_scalar: maximum possible resampled radius = threshold for set B
        self.mean_r      = float(np.max(scene.leaf_radius_max))
        self.std_r       = 0.0   # not used: edge_r_thresh = mean_r + 0*std_r
        self.edge_thresh = float(edge_threshold)
        self.max_out     = int(max_out)
        self.collect_deriv = bool(collect_deriv)
        self.rho   = float(scene.rho)
        self.tau   = float(scene.tau)
        self.rsoil = float(scene.rsoil)

    def batch_nee(self, P, obs_dirs, origin_leaf, leaf_effect,
                  leaf_normal, vec_in, is_soil, leaf_radius=None):
        """
        leaf_radius : current per-photon leaf radii (scene.leaf_radius).
                      Must be passed explicitly so NEE stays in sync with
                      the forward path when radii are resampled per photon.
        """
        le   = leaf_effect.astype(np.int8)
        lrad = (leaf_radius.astype(np.float64)
                if leaf_radius is not None else self.lrad)
        return _nee_all_dirs_nb(
            float(P[0]), float(P[1]), float(P[2]),
            obs_dirs[:, 0].astype(np.float64),
            obs_dirs[:, 1].astype(np.float64),
            obs_dirs[:, 2].astype(np.float64),
            int(origin_leaf),
            self.lpx, self.lpy, self.lpz,
            self.lnx, self.lny, self.lnz,
            lrad, le,
            float(leaf_normal[0]), float(leaf_normal[1]), float(leaf_normal[2]),
            self.vcnt, self.vlvs,
            self.vs, self.mvx, self.mvy, self.mvz,
            self.max_steps,
            self.rho, self.tau, self.rsoil,
            bool(is_soil),
            float(vec_in[0]), float(vec_in[1]), float(vec_in[2]),
            self.max_out, self.mean_r, self.std_r,
            self.edge_thresh, self.collect_deriv,
        )

    def warmup(self, obs_dirs, leaf_effect):
        P   = np.array([0.5, 0.5, -0.1])
        nrm = np.array([0., 0., 1.])
        vin = np.array([0., 0., -1.])
        self.batch_nee(P, obs_dirs, -1, leaf_effect, nrm, vin, True)

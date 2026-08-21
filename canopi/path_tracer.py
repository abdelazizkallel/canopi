"""
Canopy RT — Forward path tracer (vectorised NEE via NEEEngine).

One NEEEngine.batch_nee() call replaces the n_obs Python-level
check_visibility loop → all observation directions in one Numba kernel.
"""
from __future__ import annotations
import numpy as np
from .scene import Scene
from .geometry import propagation_limit_face, cyclic_point
from .intersection import find_contact
from .scattering import sample_scatter_direction, sample_soil_direction
from .nee import NEEEngine


def _make_contact_record(n_obs: int) -> dict:
    return {
        'leaf_id':    -1,
        'n_v':         0.0,
        'fic_ids':     [],
        'fic_nvs':     [],
        'edg_ids':     [],
        'edg_nvs':     [],
        'obs_fic_ids': [[] for _ in range(n_obs)],
        'obs_fic_nvs': [[] for _ in range(n_obs)],
        'obs_edg_ids': [[] for _ in range(n_obs)],
        'obs_edg_nvs': [[] for _ in range(n_obs)],
    }


def trace_photon(
    scene:         Scene,
    nee:           NEEEngine,
    P_start:       np.ndarray,
    vec_sun:       np.ndarray,
    obs_dirs:      np.ndarray,      # (n_obs, 3)
    D_Om_s:        float,
    D_Om_o:        float,
    leaf_effect:   np.ndarray,
    max_depth:     int,
    energy_thresh: float,
    edge_thresh:   float,
    rng:           np.random.Generator,
    compute_deriv: bool = True,
    rr_depth:      int  = 3,
    rr_min:        float = 0.1,
):
    n_obs      = len(obs_dirs)
    max_c      = max_depth + 5
    ref_tran   = np.zeros((max_c, n_obs))
    contacts   = []

    P          = P_start.copy()
    vec_P      = vec_sun.copy()
    energy = 1.0
    origin     = -1
    n_contacts = 0
    depth      = 0

    seg_fic_ids: list = []
    seg_fic_nvs: list = []

    while energy > energy_thresh and depth < max_depth:

        # ── Russian Roulette ─────────────────────────────────────────────
        # q = rr_min fixe, indépendant de l'énergie (même si energy=1)
        # E[contribution] = q×(energy/q) + (1-q)×0 = energy  → non-biaisé ✓
        if depth >= rr_depth:
            if rng.random() > rr_min:
                break
            energy /= rr_min   # boost 1/q — keeps estimator unbiased

        # ── intersection ─────────────────────────────────────────────────
        contact, leaf_idx, H, n_v, fic_leaves, edge_leaves = find_contact(
            scene, P, vec_P, origin, leaf_effect, edge_thresh,
        )

        fic_ids = [x[0] for x in fic_leaves]
        fic_nvs = [x[1] for x in fic_leaves]
        edg_ids = [x[0] for x in edge_leaves]
        edg_nvs = [x[1] for x in edge_leaves]

        seg_fic_ids += fic_ids
        seg_fic_nvs += fic_nvs

        if contact:
            if n_contacts >= max_c:
                break

            leaf_nr = scene.leaf_normal[leaf_idx]

            # ── VECTORISED NEE: all obs dirs in one Numba call ────────────
            vis, weights, fic_all, fic_nvs_all, n_fic, \
                edg_all, edg_nvs_all, n_edg = nee.batch_nee(
                    H, obs_dirs, leaf_idx, leaf_effect,
                    leaf_nr, vec_P, is_soil=False,
                    leaf_radius=scene.leaf_radius,
                )
            ref_tran[n_contacts] = energy * weights * vis

            # build contact record (for derivative)
            rec = _make_contact_record(n_obs)
            rec['leaf_id'] = leaf_idx
            rec['n_v']     = n_v
            rec['fic_ids'] = list(seg_fic_ids)
            rec['fic_nvs'] = list(seg_fic_nvs)
            rec['edg_ids'] = list(edg_ids)
            rec['edg_nvs'] = list(edg_nvs)

            if compute_deriv:
                for i_obs in range(n_obs):
                    nf = int(n_fic[i_obs])
                    ne = int(n_edg[i_obs])
                    rec['obs_fic_ids'][i_obs] = [int(fic_all[i_obs, q]) for q in range(nf)]
                    rec['obs_fic_nvs'][i_obs] = [float(fic_nvs_all[i_obs, q]) for q in range(nf)]
                    rec['obs_edg_ids'][i_obs] = [int(edg_all[i_obs, q]) for q in range(ne)]
                    rec['obs_edg_nvs'][i_obs] = [float(edg_nvs_all[i_obs, q]) for q in range(ne)]

            contacts.append(rec)
            n_contacts += 1

            # scatter
            vec_out, is_refl, albedo = sample_scatter_direction(
                rng, leaf_nr, vec_P, scene.rho, scene.tau,
            )
            # energy *= albedo TOTAL (rho+tau) — représente l'énergie physiquement
            # conservée. Le choix stochastique refl/trans est de l'importance sampling.
            energy *= (scene.rho + scene.tau)

            P      = H
            vec_P  = vec_out
            origin = leaf_idx
            depth += 1
            # Keep accumulating seg_fic_ids across contacts (MATLAB accumulates v_fic_all_iter)
            # seg_fic_ids = []   ← do NOT reset here

        else:
            p_bound, face = propagation_limit_face(P, vec_P, scene.pos_max)

            if face == 1 or face == -1:
                break

            elif face == 6:
                # ── soil hit ─────────────────────────────────────────────
                if n_contacts >= max_c:
                    break

                soil_normal = np.array([0., 0., 1.])

                vis, weights, fic_all, fic_nvs_all, n_fic, \
                    edg_all, edg_nvs_all, n_edg = nee.batch_nee(
                        p_bound, obs_dirs, -1, leaf_effect,
                        soil_normal, vec_P, is_soil=True,
                        leaf_radius=scene.leaf_radius,
                    )
                ref_tran[n_contacts] = energy * weights * vis

                rec = _make_contact_record(n_obs)
                rec['leaf_id'] = -1
                rec['n_v']     = 0.0
                rec['fic_ids'] = list(seg_fic_ids)
                rec['fic_nvs'] = list(seg_fic_nvs)
                rec['edg_ids'] = list(edg_ids)
                rec['edg_nvs'] = list(edg_nvs)

                if compute_deriv:
                    for i_obs in range(n_obs):
                        nf = int(n_fic[i_obs])
                        ne = int(n_edg[i_obs])
                        rec['obs_fic_ids'][i_obs] = [int(fic_all[i_obs, q]) for q in range(nf)]
                        rec['obs_fic_nvs'][i_obs] = [float(fic_nvs_all[i_obs, q]) for q in range(nf)]
                        rec['obs_edg_ids'][i_obs] = [int(edg_all[i_obs, q]) for q in range(ne)]
                        rec['obs_edg_nvs'][i_obs] = [float(edg_nvs_all[i_obs, q]) for q in range(ne)]

                contacts.append(rec)
                n_contacts += 1

                vec_out    = sample_soil_direction(rng)
                energy *= scene.rsoil

                P      = p_bound + np.array([0., 0., 1e-7])
                vec_P  = vec_out
                origin = -1
                depth += 1
                # Keep accumulating seg_fic_ids across contacts
                # seg_fic_ids = []   ← do NOT reset here

            else:
                P      = cyclic_point(p_bound, face, scene.pos_max)
                depth += 1

    return ref_tran, contacts, n_contacts


# ─────────────────────────────────────────────────────────────────────────────
#  Backward pixel tracer  (sensor → scene, NEE toward sun)
# ─────────────────────────────────────────────────────────────────────────────

def trace_backward_pixel(
    scene:          Scene,
    nee:            NEEEngine,
    P_start:        np.ndarray,   # pixel entry at z=0
    ray_dir:        np.ndarray,   # -obs_dir (into scene, vz < 0)
    vec_sun_up:     np.ndarray,   # -vec_sun (toward sun, vz > 0)
    leaf_effect:    np.ndarray,
    max_depth:      int,
    energy_thresh:  float,
    edge_thresh:    float,
    rng:            np.random.Generator,
    rr_depth:       int   = 999,
    rr_min:         float = 0.5,
) -> float:
    """
    Trace one backward ray from a sensor pixel through the scene.

    The ray enters the scene at P_start (z=0) in direction ray_dir = -obs_dir.
    At each scattering event, NEE toward the sun (vec_sun_up) accumulates
    radiance.  Returns total luminance for this pixel (multiply by π for BRF).
    """
    sun_dirs = vec_sun_up.reshape(1, 3)   # batch_nee expects (n_obs, 3)

    P      = P_start.copy()
    vec_P  = ray_dir.copy()
    energy = 1.0
    lum    = 0.0
    origin = -1
    depth  = 0

    while energy > energy_thresh and depth < max_depth:

        if depth >= rr_depth:
            if rng.random() > rr_min:
                break
            energy /= rr_min

        contact, leaf_idx, H, n_v, fic_leaves, edge_leaves = find_contact(
            scene, P, vec_P, origin, leaf_effect, edge_thresh,
        )

        if contact:
            leaf_nr = scene.leaf_normal[leaf_idx]

            vis, weights, *_ = nee.batch_nee(
                H, sun_dirs, leaf_idx, leaf_effect,
                leaf_nr, vec_P, is_soil=False,
                leaf_radius=scene.leaf_radius,
            )
            lum += energy * float(weights[0]) * float(vis[0])

            vec_out, _, _ = sample_scatter_direction(
                rng, leaf_nr, vec_P, scene.rho, scene.tau,
            )
            energy *= (scene.rho + scene.tau)
            P      = H
            vec_P  = vec_out
            origin = leaf_idx
            depth += 1

        else:
            p_bound, face = propagation_limit_face(P, vec_P, scene.pos_max)

            if face == 1 or face == -1:
                break   # ray escaped to sky — no more contribution

            elif face == 6:   # soil hit
                soil_normal = np.array([0., 0., 1.])
                vis, weights, *_ = nee.batch_nee(
                    p_bound, sun_dirs, -1, leaf_effect,
                    soil_normal, vec_P, is_soil=True,
                    leaf_radius=scene.leaf_radius,
                )
                lum += energy * float(weights[0]) * float(vis[0])

                vec_out    = sample_soil_direction(rng)
                energy    *= scene.rsoil
                P          = p_bound + np.array([0., 0., 1e-7])
                vec_P      = vec_out
                origin     = -1
                depth     += 1

            else:
                P     = cyclic_point(p_bound, face, scene.pos_max)
                depth += 1

    return lum


# ─────────────────────────────────────────────────────────────────────────────
#  Backward pixel tracer WITH derivative recording
# ─────────────────────────────────────────────────────────────────────────────

def trace_backward_pixel_with_deriv(
    scene:          Scene,
    nee:            NEEEngine,
    P_start:        np.ndarray,
    ray_dir:        np.ndarray,
    vec_sun_up:     np.ndarray,
    leaf_effect:    np.ndarray,
    max_depth:      int,
    energy_thresh:  float,
    edge_thresh:    float,
    rng:            np.random.Generator,
    rr_depth:       int   = 999,
    rr_min:         float = 0.5,
):
    """
    Backward ray tracer that records full contact data for derivative computation.

    Mirrors trace_photon() but:
      - direction is ray_dir = -obs_dir (into scene)
      - NEE is toward a single direction: the sun (vec_sun_up)
      - n_obs = 1

    Returns
    -------
    ref_tran   : (max_c, 1)  — NEE radiance per contact
    contacts   : list of contact records (format identical to trace_photon)
    n_contacts : int
    """
    n_obs  = 1              # single NEE direction: the sun
    max_c  = max_depth + 5
    sun_dirs = vec_sun_up.reshape(1, 3)

    ref_tran   = np.zeros((max_c, n_obs))
    contacts   = []

    P          = P_start.copy()
    vec_P      = ray_dir.copy()
    energy     = 1.0
    origin     = -1
    n_contacts = 0
    depth      = 0

    seg_fic_ids: list = []
    seg_fic_nvs: list = []

    while energy > energy_thresh and depth < max_depth:

        if depth >= rr_depth:
            if rng.random() > rr_min:
                break
            energy /= rr_min

        contact, leaf_idx, H, n_v, fic_leaves, edge_leaves = find_contact(
            scene, P, vec_P, origin, leaf_effect, edge_thresh,
        )

        fic_ids = [x[0] for x in fic_leaves]
        fic_nvs = [x[1] for x in fic_leaves]
        edg_ids = [x[0] for x in edge_leaves]
        edg_nvs = [x[1] for x in edge_leaves]

        seg_fic_ids += fic_ids
        seg_fic_nvs += fic_nvs

        if contact:
            if n_contacts >= max_c:
                break

            leaf_nr = scene.leaf_normal[leaf_idx]

            vis, weights, fic_all, fic_nvs_all, n_fic, \
                edg_all, edg_nvs_all, n_edg = nee.batch_nee(
                    H, sun_dirs, leaf_idx, leaf_effect,
                    leaf_nr, vec_P, is_soil=False,
                    leaf_radius=scene.leaf_radius,
                )
            ref_tran[n_contacts, 0] = energy * float(weights[0]) * float(vis[0])

            rec = _make_contact_record(n_obs)
            rec['leaf_id'] = leaf_idx
            rec['n_v']     = n_v
            rec['fic_ids'] = list(seg_fic_ids)
            rec['fic_nvs'] = list(seg_fic_nvs)
            rec['edg_ids'] = list(edg_ids)
            rec['edg_nvs'] = list(edg_nvs)
            nf = int(n_fic[0]);  ne = int(n_edg[0])
            rec['obs_fic_ids'][0] = [int(fic_all[0, q]) for q in range(nf)]
            rec['obs_fic_nvs'][0] = [float(fic_nvs_all[0, q]) for q in range(nf)]
            rec['obs_edg_ids'][0] = [int(edg_all[0, q]) for q in range(ne)]
            rec['obs_edg_nvs'][0] = [float(edg_nvs_all[0, q]) for q in range(ne)]

            contacts.append(rec)
            n_contacts += 1

            vec_out, _, _ = sample_scatter_direction(
                rng, leaf_nr, vec_P, scene.rho, scene.tau,
            )
            energy *= (scene.rho + scene.tau)
            P      = H
            vec_P  = vec_out
            origin = leaf_idx
            depth += 1

        else:
            p_bound, face = propagation_limit_face(P, vec_P, scene.pos_max)

            if face == 1 or face == -1:
                break

            elif face == 6:
                if n_contacts >= max_c:
                    break

                soil_normal = np.array([0., 0., 1.])
                vis, weights, fic_all, fic_nvs_all, n_fic, \
                    edg_all, edg_nvs_all, n_edg = nee.batch_nee(
                        p_bound, sun_dirs, -1, leaf_effect,
                        soil_normal, vec_P, is_soil=True,
                        leaf_radius=scene.leaf_radius,
                    )
                ref_tran[n_contacts, 0] = energy * float(weights[0]) * float(vis[0])

                rec = _make_contact_record(n_obs)
                rec['leaf_id'] = -1
                rec['n_v']     = 0.0
                rec['fic_ids'] = list(seg_fic_ids)
                rec['fic_nvs'] = list(seg_fic_nvs)
                rec['edg_ids'] = list(edg_ids)
                rec['edg_nvs'] = list(edg_nvs)
                nf = int(n_fic[0]);  ne = int(n_edg[0])
                rec['obs_fic_ids'][0] = [int(fic_all[0, q]) for q in range(nf)]
                rec['obs_fic_nvs'][0] = [float(fic_nvs_all[0, q]) for q in range(nf)]
                rec['obs_edg_ids'][0] = [int(edg_all[0, q]) for q in range(ne)]
                rec['obs_edg_nvs'][0] = [float(edg_nvs_all[0, q]) for q in range(ne)]

                contacts.append(rec)
                n_contacts += 1

                vec_out    = sample_soil_direction(rng)
                energy    *= scene.rsoil
                P          = p_bound + np.array([0., 0., 1e-7])
                vec_P      = vec_out
                origin     = -1
                depth     += 1

            else:
                P     = cyclic_point(p_bound, face, scene.pos_max)
                depth += 1

    return ref_tran, contacts, n_contacts

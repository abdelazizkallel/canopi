"""
Canopy RT — dBRF/dLAI derivative estimators.

Two estimators per photon per interaction per observation direction:
  1. fic  — fictive-leaf (discrete thinning)
  2. edg  — edge (continuous area variation)

Reference: VRT_SolVeg_Drivative_LAI2_Eq2_iter_Unified.m
"""
from __future__ import annotations
import numpy as np
from scipy.stats import norm as scipy_norm


# ─────────────────────────────────────────────────────────────────────────────
#  q_j function: probability that a leaf at distance n_v hits the ray
# ─────────────────────────────────────────────────────────────────────────────

def qj_with_derivative(
    n_v_list: list[float],   # distances (ray → leaf centre) for several leaves
    LAI_r:    float,         # mean leaf area pi*r^2
    std_LAI:  float,         # std of leaf LAI distribution
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each leaf at distance n_v from the ray, compute:
        q_j   = P(leaf_radius > n_v) = P(leaf_LAI > pi * n_v^2)
        dq_j  = d(q_j) / d(LAI_r)

    Leaf LAI ~ N(LAI_r, std_LAI^2), so:
        q_j   = 1 - Φ( (π n_v² - LAI_r) / std_LAI )
        dq_j  = φ( (π n_v² - LAI_r) / std_LAI ) / std_LAI

    Returns q (array), dq (array), both shape (len(n_v_list),)
    """
    if len(n_v_list) == 0:
        return np.array([]), np.array([])

    n_v_arr   = np.asarray(n_v_list, float)
    leaf_LAI_thresh = np.pi * n_v_arr ** 2

    if std_LAI < 1e-12:
        # deterministic radius — step function
        q  = (leaf_LAI_thresh <= LAI_r).astype(float)
        dq = np.zeros_like(q)
        return q, dq

    z   = (leaf_LAI_thresh - LAI_r) / std_LAI
    q   = 1.0 - scipy_norm.cdf(z)          # survival function
    dq  = scipy_norm.pdf(z) / std_LAI      # d/d(LAI_r) = phi(z)/std_LAI
    return q, dq


# ─────────────────────────────────────────────────────────────────────────────
#  Merge helpers (deduplicate + keep minimum/maximum distance)
# ─────────────────────────────────────────────────────────────────────────────

def merge_unique_min(
    ids_a, nv_a,
    ids_b, nv_b,
) -> tuple[list, list]:
    """Merge two (id, nv) lists, keeping min(nv) for duplicates."""
    d = {}
    for i, n in zip(ids_a, nv_a):
        if i not in d or n < d[i]:
            d[i] = n
    for i, n in zip(ids_b, nv_b):
        if i not in d or n < d[i]:
            d[i] = n
    if not d:
        return [], []
    ids = list(d.keys())
    nvs = [d[k] for k in ids]
    return ids, nvs


def merge_unique_max(ids_a, nv_a, ids_b, nv_b):
    """Merge two (id, nv) lists, keeping max(nv) for duplicates."""
    d = {}
    for i, n in zip(ids_a, nv_a):
        if i not in d or n > d[i]:
            d[i] = n
    for i, n in zip(ids_b, nv_b):
        if i not in d or n > d[i]:
            d[i] = n
    if not d:
        return [], []
    ids = list(d.keys())
    nvs = [d[k] for k in ids]
    return ids, nvs


def split_by_membership(ids_a, nv_a, ids_b):
    """
    Split ids_a into:
      - those whose id is in ids_b  (with_b)
      - those whose id is NOT in ids_b (without_b)
    """
    b_set = set(ids_b)
    with_b_ids, with_b_nv     = [], []
    without_b_ids, without_b_nv = [], []
    for i, n in zip(ids_a, nv_a):
        if i in b_set:
            with_b_ids.append(i); with_b_nv.append(n)
        else:
            without_b_ids.append(i); without_b_nv.append(n)
    return with_b_ids, with_b_nv, without_b_ids, without_b_nv


# ─────────────────────────────────────────────────────────────────────────────
#  Per-photon derivative computation (called after path is complete)
# ─────────────────────────────────────────────────────────────────────────────

def compute_photon_derivative(
    photon_contacts:       list,    # list of ContactRecord
    ref_contributions:     np.ndarray,  # (max_contacts, n_obs) radiance per interaction
    n_contacts:            int,
    LAI_r:                 float,
    std_LAI:               float,
    LAI_Max:               float,
    P_LAI:                 float,
    n_leaf_norm:           float,
    n_obs:                 int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute fictive and edge derivative estimates for one photon.

    Returns
    -------
    deriv_fic : (n_contacts, n_obs)
    deriv_edg : (n_contacts, n_obs)
    """
    EPS = 1e-12

    deriv_fic = np.zeros((n_contacts, n_obs))
    deriv_edg = np.zeros((n_contacts, n_obs))

    # ── Accumulators over the forward path ────────────────────────────────
    # H: real leaf contacts (leaves only, soil excluded)
    v_contact_ids  = []
    nv_contact     = []

    # B_fwd: fictive/edge leaves from forward path segments
    v_fic_all      = []
    nv_fic_all     = []
    v_edg_all      = []
    nv_edg_all     = []

    for i_iter in range(n_contacts):
        rec = photon_contacts[i_iter]

        # ── Update H: real leaf contacts ──────────────────────────────────
        v = rec['leaf_id']
        d = rec['n_v']
        if v >= 0:   # leaf only (not soil)
            found = False
            for idx_c, vc in enumerate(v_contact_ids):
                if vc == v:
                    nv_contact[idx_c] = max(nv_contact[idx_c], d)
                    found = True; break
            if not found:
                v_contact_ids.append(v)
                nv_contact.append(d)

        n_contact = len(v_contact_ids)

        # ── Update B_fwd: fictive/edge leaves from forward segments ───────
        v_fic_all, nv_fic_all = merge_unique_min(
            rec['fic_ids'], rec['fic_nvs'], v_fic_all, nv_fic_all)
        v_edg_all, nv_edg_all = merge_unique_min(
            rec['edg_ids'], rec['edg_nvs'], v_edg_all, nv_edg_all)

        (v_edg_cont, nv_edg_cont,
         v_edg_no_cont, nv_edg_no_cont) = split_by_membership(
            v_edg_all, nv_edg_all, v_contact_ids)

        for i_obs in range(n_obs):
            ref = ref_contributions[i_iter, i_obs]
            if ref == 0.0:
                continue   # NEE ray blocked — this path does not reach sensor

            # ── Path: Sun → ... → current_contact → sensor ────────────────
            # Each successful NEE ray defines an INDEPENDENT path.
            # H and B are built for THIS path only:
            #   H = real leaf hits on forward segments (v_contact_ids)
            #   B = fictives on forward segments (v_fic_all)
            #     + fictives on current NEE segment (rec['obs_fic_ids'][i_obs])
            # No accumulation from other NEE rays (past or future).

            rec_obs_fic_ids = rec['obs_fic_ids'][i_obs]
            rec_obs_fic_nvs = rec['obs_fic_nvs'][i_obs]
            rec_obs_edg_ids = rec['obs_edg_ids'][i_obs]
            rec_obs_edg_nvs = rec['obs_edg_nvs'][i_obs]

            # B\H = (B_fwd + B_nee_current) excluding H
            v_fic_path, nv_fic_path = merge_unique_min(
                v_fic_all, nv_fic_all,
                rec_obs_fic_ids, rec_obs_fic_nvs)

            (v_edg_obs_cont, nv_edg_obs_cont,
             v_edg_obs_no_cont, nv_edg_obs_no_cont) = split_by_membership(
                rec_obs_edg_ids, rec_obs_edg_nvs, v_contact_ids)

            v_nc_all, nv_nc_all = merge_unique_min(
                v_fic_path, nv_fic_path, v_edg_no_cont, nv_edg_no_cont)
            v_nc_all, nv_nc_all = merge_unique_min(
                v_nc_all, nv_nc_all, v_edg_obs_no_cont, nv_edg_obs_no_cont)

            # d_l^B = min over all segments (Eq. 28), as for B\H
            v_ec_all, nv_ec_all = merge_unique_min(
                v_edg_cont, nv_edg_cont, v_edg_obs_cont, nv_edg_obs_cont)

            # ── Build the three disjoint sets ─────────────────────────────
            # H∩B : contacts that also appear in edge set (d_l^H and d_l^B known)
            # H\B : contacts NOT in edge set (only d_j^H)
            # B\H : edge/fictive leaves NOT in contacts (only d_k^B) = v_nc_all

            # For H∩B: need both d_l^H (from contacts) and d_l^B (from v_ec_all)
            # Build a dict: leaf_id → d_j^H for all contacts
            contact_dist = {v_contact_ids[ci]: nv_contact[ci]
                            for ci in range(len(v_contact_ids))}

            # H∩B: set(v_ec_all) ∩ set(v_contact_ids)
            # H\B: set(v_contact_ids) \ set(v_ec_all)
            ec_set = set(v_ec_all)

            nv_H_only  = []   # d_j^H for j ∈ H\B
            nv_HB_hit  = []   # d_l^H for l ∈ H∩B
            nv_HB_blk  = []   # d_l^B for l ∈ H∩B

            for ci, vid in enumerate(v_contact_ids):
                if vid in ec_set:
                    # l ∈ H∩B: look up d_l^B from v_ec_all
                    idx_e = v_ec_all.index(vid)
                    nv_HB_hit.append(nv_contact[ci])
                    nv_HB_blk.append(nv_ec_all[idx_e])
                else:
                    # j ∈ H\B
                    nv_H_only.append(nv_contact[ci])

            # ── q values for each set ──────────────────────────────────────
            # B\H
            q_nc,  dq_nc  = qj_with_derivative(nv_nc_all,  LAI_r, std_LAI)
            # H\B
            q_ho,  dq_ho  = qj_with_derivative(nv_H_only,  LAI_r, std_LAI)
            # H∩B — hit side (d_l^H)
            q_hh,  dq_hh  = qj_with_derivative(nv_HB_hit,  LAI_r, std_LAI)
            # H∩B — blocking side (d_l^B)
            q_hb,  dq_hb  = qj_with_derivative(nv_HB_blk,  LAI_r, std_LAI)

            # ── fictive estimator ─────────────────────────────────────────
            # ref/LAI_Max * (|H|/P - Σ_{k∈B\H} q_k/(1-P*q_k))
            sum_q_nc = np.sum(q_nc / np.maximum(1.0 - q_nc * P_LAI, EPS))
            deriv_fic[i_iter, i_obs] = (
                ref / LAI_Max * (n_contact / P_LAI - sum_q_nc)
            )

            # ── edge estimator ────────────────────────────────────────────
            # H_term  = Σ_{j∈H\B}  dq(d_j^H) / q(d_j^H)
            H_term  = np.sum(dq_ho / np.maximum(q_ho, EPS))

            # B_term  = Σ_{k∈B\H}  P * dq(d_k^B) / (1 - P*q(d_k^B))
            B_term  = np.sum(P_LAI * dq_nc / np.maximum(1.0 - q_nc * P_LAI, EPS))

            # H_B_term = Σ_{l∈H∩B} (dq(d_l^H) - dq(d_l^B)) / (q(d_l^H) - q(d_l^B))
            # Each term pairs the HIT distance d_l^H with the BLOCKING distance d_l^B
            # for the SAME leaf l.
            HB_denom = q_hh - q_hb   # q(d_l^H) - q(d_l^B)  [should be ≥ 0]
            HB_num   = dq_hh - dq_hb
            H_B_term = np.sum(HB_num / np.where(np.abs(HB_denom) > EPS,
                                                  HB_denom, EPS))

            deriv_edg[i_iter, i_obs] = (
                ref / n_leaf_norm / P_LAI * (H_term - B_term + H_B_term)
            )

    return deriv_fic, deriv_edg


# ─────────────────────────────────────────────────────────────────────────────
#  Optimal linear combination
# ─────────────────────────────────────────────────────────────────────────────

def optimal_weight(
    D_fic: np.ndarray,   # (n_photons, n_obs)
    D_edg: np.ndarray,   # (n_photons, n_obs)
) -> np.ndarray:
    """
    Compute per-observation optimal weight w such that:
        D_opt[:,k] = w[k]*D_fic[:,k] + (1-w[k])*D_edg[:,k]
    minimises variance.
    """
    n_obs = D_fic.shape[1]
    w = np.zeros(n_obs)
    _EPS_VAR = 1e-30
    for k in range(n_obs):
        X = D_fic[:, k]
        Y = D_edg[:, k]
        vX   = np.var(X)
        vY   = np.var(Y)
        cov  = np.cov(X, Y)[0, 1] if len(X) > 1 else 0.0
        # Degenerate cases: one estimator has zero variance (e.g. std_LAI=0
        # makes the edge estimator identically zero).
        if vY < _EPS_VAR:
            w[k] = 1.0   # edge is degenerate → use fictive only
        elif vX < _EPS_VAR:
            w[k] = 0.0   # fictive is degenerate → use edge only
        else:
            denom = vX + vY - 2 * cov
            if abs(denom) > 1e-12:
                w[k] = np.clip((vY - cov) / denom, 0.0, 1.0)
            else:
                w[k] = 0.5
    return w

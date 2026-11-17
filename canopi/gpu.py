"""
canopi.gpu.py
==================
GPU-accelerated canopy RT — BRF and dBRF/dLAI — using Numba CUDA.

Architecture
------------
Two kernels share the same device-function library:

  _trace_kernel        — BRF only (fast, minimal registers)
  _trace_deriv_kernel  — BRF + dBRF/dLAI inline in one forward pass

Derivative algorithm (mirrors derivative.py) runs entirely in thread-local
fixed-size arrays — no global-memory round-trips for intermediate sets.

Thread-local budget per photon (_MAX_C=15, _MAX_FIC=24):
  ≈ 5 KB  → fits comfortably in L1-cached local memory.

Public API
----------
  # BRF only (original)
  compute_brf_gpu(scene, obs_dirs, vec_sun, N, ...)  → lum (N, n_obs)
  brf_gpu(scene, obs_dirs, vec_sun, N, ...)          → brf (n_obs,)

  # BRF + derivative
  compute_deriv_gpu(scene, obs_dirs, vec_sun, N, ...) → lum, deriv_fic, deriv_edg
  deriv_gpu(scene, obs_dirs, vec_sun, N, ...)         → brf, deriv_fic, deriv_edg, deriv_opt

  # Drop-in render() replacement
  render_gpu(cfg, ...)                                → RenderResult

  # Throughput table
  benchmark(scene, obs_dirs, vec_sun, cfg, N_list)

Requirements
------------
    pip install numba cupy-cuda12x   # adjust cuda version to match driver

Usage example
-------------
    from canopi.gpu import render_gpu, GPU_AVAILABLE
    from canopi import SceneConfig, render

    cfg = SceneConfig(LAI_eff=1.0, P_LAI=0.5, rho=0.4, tau=0.1, rsoil=0.1,
                      n_photons=50_000, compute_derivative=True)

    result = render_gpu(cfg) if GPU_AVAILABLE else render(cfg)
"""
import gc
import os
import numpy as np
import math

# ─────────────────────────────────────────────────────────────────────────────
# GPU detection
# ─────────────────────────────────────────────────────────────────────────────
try:
    from numba import cuda
    import numba as nb
    from numba.cuda.random import (create_xoroshiro128p_states,
                                   xoroshiro128p_uniform_float64)
    GPU_AVAILABLE = cuda.is_available()
except ImportError:
    GPU_AVAILABLE = False

if GPU_AVAILABLE:
    print('[canopi.gpu] GPU detected — CUDA acceleration enabled')
else:
    print('[canopi.gpu] No GPU — CPU fallback active')


# ─────────────────────────────────────────────────────────────────────────────
# Compile-time constants for thread-local derivative arrays
# Increase _MAX_C / _MAX_FIC if you see truncated derivatives at very high LAI.
# ─────────────────────────────────────────────────────────────────────────────
_MAX_C   = 60    # max real-leaf contacts tracked per photon  (H set)
_MAX_FIC = 200   # max fictive / edge leaves per set
# Sun-visibility (shadow) rays historically block on the SCALAR scene.P_LAI, which
# makes the forward model species-blind: a per-species LAI change is felt by the
# camera path but not by the shadow ray, so a per-species Jacobian column does not
# describe the model's actual response. Set CANOPI_SPECIES_SHADOW=1 (or pass
# species_shadow=True) to make the shadow ray read P_LAI_sp[species] as well.
_SPECIES_SHADOW = os.environ.get("CANOPI_SPECIES_SHADOW", "0") not in ("0", "", "false", "False")

_N_SP    = 18    # number of JBS size-class groups (ACPL, BEPE1-4, ALGL1-4, FREX, TICO1-5, PIAB, POTR1-2)


# ─────────────────────────────────────────────────────────────────────────────
# CUDA device functions  (compiled only when GPU present)
# ─────────────────────────────────────────────────────────────────────────────
if GPU_AVAILABLE:

    # ── Gaussian helpers (replace scipy on-device) ────────────────────────────

    @cuda.jit(device=True)
    def _gaussian_sf(z):
        """Survival function 1 - Φ(z) = 0.5 * erfc(z / √2)."""
        return 0.5 * math.erfc(z * 0.7071067811865476)   # 1/√2

    @cuda.jit(device=True)
    def _gaussian_pdf(z):
        """Standard normal PDF φ(z) = exp(-z²/2) / √(2π)."""
        return math.exp(-0.5 * z * z) * 0.3989422804014327  # 1/√(2π)

    @cuda.jit(device=True)
    def _q_dq(n_v, LAI_r, std_LAI):
        """
        q(n_v)       = P(leaf_radius > n_v) = 1 - Φ((π·n_v² - LAI_r) / std_LAI)
        dq/d(LAI_r)  = φ(z) / std_LAI

        Mirrors qj_with_derivative() in derivative.py — no scipy, no allocations.
        Returns (q, dq).
        """
        PI  = 3.141592653589793
        llt = PI * n_v * n_v          # π·n_v²  (leaf LAI threshold)

        if std_LAI < 1e-12:           # deterministic radius — step function
            if llt <= LAI_r:
                return 1.0, 0.0
            return 0.0, 0.0

        z  = (llt - LAI_r) / std_LAI
        q  = _gaussian_sf(z)
        dq = _gaussian_pdf(z) / std_LAI
        return q, dq

    @cuda.jit(device=True)
    def _q_dq_ln(n_v, LAI_r, sigma_ln):
        """
        Log-normal survival function and derivative.
        leaf_LAI ~ LogNormal(mu_ln, sigma_ln²),  mu_ln = log(LAI_r) - sigma_ln²/2
        q   = P(leaf_LAI > llt)    where llt = π·n_v²
        dq  = ∂q/∂LAI_r = φ(z) / (LAI_r · sigma_ln)
        """
        PI = 3.141592653589793
        llt = PI * n_v * n_v
        if llt < 1e-200:
            return 1.0, 0.0
        if sigma_ln < 1e-12:
            if llt <= LAI_r:
                return 1.0, 0.0
            return 0.0, 0.0
        mu_ln = math.log(LAI_r) - 0.5 * sigma_ln * sigma_ln
        z  = (math.log(llt) - mu_ln) / sigma_ln
        q  = _gaussian_sf(z)
        dq = _gaussian_pdf(z) / (LAI_r * sigma_ln)
        return q, dq

    @cuda.jit(device=True)
    def _leaf_radius_draw_ln(photon_id, leaf_id, global_seed, LAI_r, sigma_ln):
        """
        Per-photon per-leaf radius draw: leaf_LAI ~ LogNormal(mu_ln, sigma_ln²).
        mu_ln = log(LAI_r) - sigma_ln²/2.  Returns sqrt(leaf_LAI / π).
        Uses the same hash mixing as _leaf_radius_draw for reproducibility.
        """
        PI = 3.141592653589793
        if sigma_ln < 1e-12:
            return math.sqrt(LAI_r / PI)
        mu_ln = math.log(LAI_r) - 0.5 * sigma_ln * sigma_ln
        seed_r = (global_seed ^ 0xBEEF1234) & 0xFFFFFFFF
        h1 = (photon_id * 2654435761 + leaf_id * 1234567891 + seed_r) & 0xFFFFFFFF
        h1 ^= h1 >> 16
        h1  = (h1 * 0x45d9f3b) & 0xFFFFFFFF
        h1 ^= h1 >> 16
        h2 = (photon_id * 1013904223 + leaf_id * 2246822519 + (seed_r ^ 0xABCD)) & 0xFFFFFFFF
        h2 ^= h2 >> 13
        h2  = (h2 * 0xc2b2ae35) & 0xFFFFFFFF
        h2 ^= h2 >> 16
        u1 = max((h1 & 0x7FFFFFFF) / 2147483648.0, 1e-10)
        u2 = (h2 & 0x7FFFFFFF) / 2147483648.0
        z  = math.sqrt(-2.0 * math.log(u1)) * math.cos(6.283185307179586 * u2)
        LAI_i = math.exp(mu_ln + sigma_ln * z)
        return math.sqrt(LAI_i / PI)

    @cuda.jit(device=True)
    def _q_dq_any(n_v, LAI_r, std_LAI, use_lognormal, sigma_ln):
        """Dispatch to Gaussian or log-normal (q, dq) based on use_lognormal flag."""
        if use_lognormal:
            return _q_dq_ln(n_v, LAI_r, sigma_ln)
        return _q_dq(n_v, LAI_r, std_LAI)

    # ── Linear search (no Python dicts/sets in CUDA) ──────────────────────────

    @cuda.jit(device=True)
    def _find_id(arr, n, val):
        """Return index of val in arr[0:n], or -1 if absent."""
        for i in range(n):
            if arr[i] == val:
                return i
        return -1

    @cuda.jit(device=True)
    def _leaf_real(photon_id, leaf_id, global_seed, P_LAI):
        """
        Hash-based Bernoulli draw: returns 1 (real) or 0 (fictive).

        Replaces the (N × n_leaf) int8 leaf_effect_all array.
        Uses a two-round integer hash of (photon_id, leaf_id, seed)
        to produce a uniform value in [0, 1) and compares with P_LAI.

        Properties:
          • Deterministic: same (photon, leaf, seed) → same result.
          • Independent across leaves and photons (no spatial correlation).
          • Zero memory: O(1) per call, no pre-allocation.
        """
        # Philox-style mixing (unsigned 32-bit arithmetic via masking)
        h = (photon_id * 1664525 + leaf_id * 1013904223 + global_seed) & 0xFFFFFFFF
        h ^= h >> 16
        h  = (h * 0x45d9f3b) & 0xFFFFFFFF
        h ^= h >> 16
        h ^= h >> 13
        h  = (h * 0xc2b2ae35) & 0xFFFFFFFF
        h ^= h >> 16
        u = (h & 0x7FFFFFFF) / 2147483648.0   # uniform in [0, 1)
        return 1 if u < P_LAI else 0

    @cuda.jit(device=True)
    def _leaf_radius_draw(photon_id, leaf_id, global_seed, LAI_r, std_LAI):
        """
        Per-photon per-leaf radius draw via hash-based Box-Muller transform.

        Leaf LAI_i ~ N(LAI_r, std_LAI²), clipped to [0, ∞).
        Returns radius_i = sqrt(max(LAI_i, 0) / π).

        Uses a DIFFERENT hash mixing from _leaf_real to ensure independence
        between the Bernoulli (real/fictive) and the radius draws.

        This matches the MATLAB per-photon resampling:
            leaf_LAI = max(LAI_r*(1 + std_LAI/LAI_r*randn(n_leaf,1)), 0);
            leaf_radius = sqrt(leaf_LAI/pi);
        making the derivative formula's q_j = P(r > nv) self-consistent.
        """
        PI = 3.141592653589793
        if std_LAI < 1e-12:
            return math.sqrt(LAI_r / PI)

        # Two independent hashes — different multipliers + XOR'd seed
        seed_r = (global_seed ^ 0xBEEF1234) & 0xFFFFFFFF
        h1 = (photon_id * 2654435761 + leaf_id * 1234567891 + seed_r) & 0xFFFFFFFF
        h1 ^= h1 >> 16
        h1  = (h1 * 0x45d9f3b) & 0xFFFFFFFF
        h1 ^= h1 >> 16
        h2 = (photon_id * 1013904223 + leaf_id * 2246822519 + (seed_r ^ 0xABCD)) & 0xFFFFFFFF
        h2 ^= h2 >> 13
        h2  = (h2 * 0xc2b2ae35) & 0xFFFFFFFF
        h2 ^= h2 >> 16

        u1 = max((h1 & 0x7FFFFFFF) / 2147483648.0, 1e-10)  # in (0, 1]
        u2 = (h2 & 0x7FFFFFFF) / 2147483648.0               # in [0, 1)

        # Box-Muller: z ~ N(0, 1)
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(6.283185307179586 * u2)

        LAI_i = LAI_r + std_LAI * z
        if LAI_i < 0.0:
            LAI_i = 0.0
        return math.sqrt(LAI_i / PI)

    # ─────────────────────────────────────────────────────────────────────────
    # _dda_contact — original BRF-only DDA (no fictive collection)
    # ─────────────────────────────────────────────────────────────────────────

    @cuda.jit(device=True)
    def _dda_contact(Px, Py, Pz, vx, vy, vz, origin,
                     leaf_px, leaf_py, leaf_pz,
                     leaf_nx, leaf_ny, leaf_nz, leaf_radius,
                     photon_id, global_seed, P_LAI,
                     voxel_count, voxel_leaves, vs, mvx, mvy, mvz,
                     out_idx, out_Hx, out_Hy, out_Hz, out_nv,
                     periodic_xy):
        """
        DDA ray traversal — returns True if a real leaf was hit.

        periodic_xy : 1 = periodic x/y BC (homogeneous scenes);
                      0 = absorbing x/y walls (finite heterogeneous scenes,
                          matches MATLAB verif_cyclic=false for RAMI HET01).
        """
        contact = False
        min_dis = 1e18       # GLOBAL distance from ray origin
        accumulated_t = 0.0  # total DDA distance from ray origin
        Lx = mvx * vs;  Ly = mvy * vs
        if periodic_xy:
            _Px = Px % Lx
            if _Px < 0.0: _Px += Lx
            _Py = Py % Ly
            if _Py < 0.0: _Py += Ly
        else:
            _Px = min(max(Px, 0.0), Lx - 1e-8)
            _Py = min(max(Py, 0.0), Ly - 1e-8)
        _Pz    = min(max(Pz, -mvz * vs + 1e-8), -1e-8)
        cur_i  = min(int(_Px / vs), mvx - 1)
        cur_j  = min(int(_Py / vs), mvy - 1)
        cur_k  = min(int(math.fabs(_Pz) / vs), mvz - 1)
        cur_px = _Px;  cur_py = _Py;  cur_pz = _Pz
        max_steps = (mvx + mvy + mvz) * 4 + 20

        for _s in range(max_steps):
            vox = cur_i + cur_j * mvx + cur_k * mvx * mvy
            for k in range(voxel_count[vox]):
                i = voxel_leaves[vox, k]
                if i == origin:
                    continue
                ptx = leaf_px[i] - cur_px
                pty = leaf_py[i] - cur_py
                ptz = leaf_pz[i] - cur_pz
                if periodic_xy:
                    if ptx >  Lx * 0.5: ptx -= Lx
                    if ptx < -Lx * 0.5: ptx += Lx
                    if pty >  Ly * 0.5: pty -= Ly
                    if pty < -Ly * 0.5: pty += Ly
                den = vx*leaf_nx[i] + vy*leaf_ny[i] + vz*leaf_nz[i]
                if math.fabs(den) < 1e-12:
                    continue
                OH = (ptx*leaf_nx[i] + pty*leaf_ny[i] + ptz*leaf_nz[i]) / den
                oh_global = accumulated_t + OH
                if OH < 0.0 or oh_global >= min_dis:
                    continue
                MHx = -ptx + OH*vx;  MHy = -pty + OH*vy;  MHz = -ptz + OH*vz
                nv  = math.sqrt(MHx*MHx + MHy*MHy + MHz*MHz)
                if nv < leaf_radius[i] and _leaf_real(photon_id, i, global_seed, P_LAI):
                    min_dis    = oh_global
                    out_idx[0] = i;  out_nv[0] = nv
                    out_Hx[0]  = cur_px + OH * vx
                    out_Hy[0]  = cur_py + OH * vy
                    out_Hz[0]  = cur_pz + OH * vz
                    contact    = True

            t_min = 1e18;  step_axis = 0
            ni = cur_i;  nj = cur_j;  nk = cur_k
            if math.fabs(vx) > 1e-12:
                ex = 1 if vx > 0 else -1
                tx = math.fabs((cur_i + (1 if ex > 0 else 0))*vs - cur_px) / math.fabs(vx)
                if tx < 1e-8: tx = math.fabs((cur_i + (1 if ex > 0 else 0) + ex)*vs - cur_px) / math.fabs(vx)
                if tx > 1e-8 and tx < t_min: t_min = tx; step_axis = 1; ni = cur_i + ex; nj = cur_j; nk = cur_k
            if math.fabs(vy) > 1e-12:
                ey = 1 if vy > 0 else -1
                ty = math.fabs((cur_j + (1 if ey > 0 else 0))*vs - cur_py) / math.fabs(vy)
                if ty < 1e-8: ty = math.fabs((cur_j + (1 if ey > 0 else 0) + ey)*vs - cur_py) / math.fabs(vy)
                if ty > 1e-8 and ty < t_min: t_min = ty; step_axis = 2; ni = cur_i; nj = cur_j + ey; nk = cur_k
            if math.fabs(vz) > 1e-12:
                ez = -1 if vz > 0 else 1
                tz = math.fabs((cur_k + (1 if ez > 0 else 0))*vs - math.fabs(cur_pz)) / math.fabs(vz)
                if tz < 1e-8: tz = math.fabs((cur_k + (1 if ez > 0 else 0) + ez)*vs - math.fabs(cur_pz)) / math.fabs(vz)
                if tz > 1e-8 and tz < t_min: t_min = tz; step_axis = 3; ni = cur_i; nj = cur_j; nk = cur_k + ez
            if nk < 0 or nk >= mvz:
                break
            if periodic_xy:
                if ni < 0: ni += mvx
                elif ni >= mvx: ni -= mvx
                if nj < 0: nj += mvy
                elif nj >= mvy: nj -= mvy
            else:
                if ni < 0 or ni >= mvx or nj < 0 or nj >= mvy:
                    break   # photon exits finite scene in x/y
            if contact and accumulated_t + t_min > min_dis:
                break
            cur_px += t_min*vx;  cur_py += t_min*vy;  cur_pz += t_min*vz
            if vx > 1e-12: cur_px += 1e-8
            elif vx < -1e-12: cur_px -= 1e-8
            if vy > 1e-12: cur_py += 1e-8
            elif vy < -1e-12: cur_py -= 1e-8
            if vz > 1e-12: cur_pz += 1e-8
            elif vz < -1e-12: cur_pz -= 1e-8
            if periodic_xy:
                cur_px = cur_px % Lx
                if cur_px < 0.0: cur_px += Lx
                cur_py = cur_py % Ly
                if cur_py < 0.0: cur_py += Ly
            cur_i = ni;  cur_j = nj;  cur_k = nk
            accumulated_t += t_min

        return contact

    # ─────────────────────────────────────────────────────────────────────────
    # _dda_fwd_deriv

    @cuda.jit(device=True)
    def _dda_fwd_deriv(
        Px, Py, Pz, vx, vy, vz, origin,
        leaf_px, leaf_py, leaf_pz,
        leaf_nx, leaf_ny, leaf_nz,
        leaf_radius, photon_id, global_seed, P_LAI,
        voxel_count, voxel_leaves, vs, mvx, mvy, mvz, periodic_xy,
        # ── output: first real hit ────────────────────────────────────────
        out_idx, out_Hx, out_Hy, out_Hz, out_nv,
        # ── output: derivative sets (written into caller thread-local arrs)
        fic_ids, fic_nvs, fic_ohs, n_fic,      # fictive leaves (effect=0, nv < r)
        edg_ids, edg_nvs, edg_ohs, n_edg,      # edge    leaves (r ≤ nv < edge_r_thresh)
        edge_r_thresh,
        LAI_r, std_LAI, use_lognormal, sigma_ln, # per-photon radius distribution params
    ):
        """
        DDA forward traversal that simultaneously:
          • finds the first real-leaf contact  (contact=True on return)
          • collects fictive leaves (leaf_effect=0, nv < r_photon) → fic_*
          • collects edge    leaves (nv in [r_photon, edge_r_thresh)) → edg_*

        Each leaf radius is drawn per-photon via _leaf_radius_draw() so that the
        derivative formula's q_j = P(r > nv) is self-consistent with the model.

        Duplicates (same leaf ID in multiple voxels) are resolved with min(nv).
        n_fic and n_edg are 1-element int32 arrays used as mutable counters.
        Both counters are reset to 0 on entry.
        """
        contact = False
        min_dis = 1e18       # GLOBAL distance from ray origin
        accumulated_t = 0.0  # total DDA distance from ray origin
        # Wrap or clamp x/y depending on boundary condition; clamp z into volume
        Lx = mvx * vs;  Ly = mvy * vs
        if periodic_xy:
            _Px = Px % Lx
            if _Px < 0.0: _Px += Lx
            _Py = Py % Ly
            if _Py < 0.0: _Py += Ly
        else:
            _Px = min(max(Px, 0.0), Lx - 1e-8)
            _Py = min(max(Py, 0.0), Ly - 1e-8)
        _Pz    = min(max(Pz, -mvz * vs + 1e-8), -1e-8)
        cur_i  = min(int(_Px / vs), mvx - 1)
        cur_j  = min(int(_Py / vs), mvy - 1)
        cur_k  = min(int(math.fabs(_Pz) / vs), mvz - 1)
        cur_px = _Px
        cur_py = _Py
        cur_pz = _Pz
        max_steps = (mvx + mvy + mvz) * 4 + 20

        n_fic[0] = 0
        n_edg[0] = 0

        for _s in range(max_steps):
            vox = cur_i + cur_j * mvx + cur_k * mvx * mvy

            for k in range(voxel_count[vox]):
                i = voxel_leaves[vox, k]
                if i == origin:
                    continue

                ptx = leaf_px[i] - cur_px
                pty = leaf_py[i] - cur_py
                ptz = leaf_pz[i] - cur_pz
                if periodic_xy:
                    if ptx >  Lx * 0.5: ptx -= Lx
                    if ptx < -Lx * 0.5: ptx += Lx
                    if pty >  Ly * 0.5: pty -= Ly
                    if pty < -Ly * 0.5: pty += Ly

                den = vx*leaf_nx[i] + vy*leaf_ny[i] + vz*leaf_nz[i]
                if math.fabs(den) < 1e-12:
                    continue
                OH = (ptx*leaf_nx[i] + pty*leaf_ny[i] + ptz*leaf_nz[i]) / den
                oh_global = accumulated_t + OH
                if OH < 0.0 or oh_global >= min_dis:
                    continue

                MHx = -ptx + OH*vx
                MHy = -pty + OH*vy
                MHz = -ptz + OH*vz
                nv  = math.sqrt(MHx*MHx + MHy*MHy + MHz*MHz)

                if use_lognormal:
                    r_i = _leaf_radius_draw_ln(photon_id, i, global_seed, LAI_r, sigma_ln)
                else:
                    r_i = _leaf_radius_draw(photon_id, i, global_seed, LAI_r, std_LAI)

                if nv < r_i:
                    if _leaf_real(photon_id, i, global_seed, P_LAI):   # ── REAL ──
                        min_dis    = oh_global
                        out_idx[0] = i;  out_nv[0] = nv
                        out_Hx[0]  = cur_px + OH * vx
                        out_Hy[0]  = cur_py + OH * vy
                        out_Hz[0]  = cur_pz + OH * vz
                        contact    = True
                    else:                                              # ── FICTIVE ──
                        if _find_id(fic_ids, n_fic[0], i) < 0 and n_fic[0] < _MAX_FIC:
                            fic_ids[n_fic[0]] = i
                            fic_nvs[n_fic[0]] = nv
                            fic_ohs[n_fic[0]] = oh_global
                            n_fic[0] += 1
                elif nv < edge_r_thresh:                    # ── EDGE ──
                    if _find_id(edg_ids, n_edg[0], i) < 0 and n_edg[0] < _MAX_FIC:
                        edg_ids[n_edg[0]] = i
                        edg_nvs[n_edg[0]] = nv
                        edg_ohs[n_edg[0]] = oh_global
                        n_edg[0] += 1

            # advance DDA
            t_min = 1e18;  step_axis = 0
            ni = cur_i;  nj = cur_j;  nk = cur_k
            if math.fabs(vx) > 1e-12:
                ex = 1 if vx > 0 else -1
                tx = math.fabs((cur_i + (1 if ex > 0 else 0))*vs - cur_px) / math.fabs(vx)
                if tx < 1e-8: tx = math.fabs((cur_i + (1 if ex > 0 else 0) + ex)*vs - cur_px) / math.fabs(vx)
                if tx > 1e-8 and tx < t_min: t_min = tx; step_axis = 1; ni = cur_i + ex; nj = cur_j; nk = cur_k
            if math.fabs(vy) > 1e-12:
                ey = 1 if vy > 0 else -1
                ty = math.fabs((cur_j + (1 if ey > 0 else 0))*vs - cur_py) / math.fabs(vy)
                if ty < 1e-8: ty = math.fabs((cur_j + (1 if ey > 0 else 0) + ey)*vs - cur_py) / math.fabs(vy)
                if ty > 1e-8 and ty < t_min: t_min = ty; step_axis = 2; ni = cur_i; nj = cur_j + ey; nk = cur_k
            if math.fabs(vz) > 1e-12:
                ez = -1 if vz > 0 else 1
                tz = math.fabs((cur_k + (1 if ez > 0 else 0))*vs - math.fabs(cur_pz)) / math.fabs(vz)
                if tz < 1e-8: tz = math.fabs((cur_k + (1 if ez > 0 else 0) + ez)*vs - math.fabs(cur_pz)) / math.fabs(vz)
                if tz > 1e-8 and tz < t_min: t_min = tz; step_axis = 3; ni = cur_i; nj = cur_j; nk = cur_k + ez
            if nk < 0 or nk >= mvz:
                break
            if periodic_xy:
                if ni < 0: ni += mvx
                elif ni >= mvx: ni -= mvx
                if nj < 0: nj += mvy
                elif nj >= mvy: nj -= mvy
            else:
                if ni < 0 or ni >= mvx or nj < 0 or nj >= mvy:
                    break
            if contact and accumulated_t + t_min > min_dis:
                # Post-filter: discard fic/edg entries beyond the real hit
                n_out = 0
                for k in range(n_edg[0]):
                    if edg_ohs[k] < min_dis:
                        edg_ids[n_out] = edg_ids[k]
                        edg_nvs[n_out] = edg_nvs[k]
                        edg_ohs[n_out] = edg_ohs[k]
                        n_out += 1
                n_edg[0] = n_out

                n_out = 0
                for k in range(n_fic[0]):
                    if fic_ohs[k] < min_dis:
                        fic_ids[n_out] = fic_ids[k]
                        fic_nvs[n_out] = fic_nvs[k]
                        fic_ohs[n_out] = fic_ohs[k]
                        n_out += 1
                n_fic[0] = n_out
                break
            cur_px += t_min*vx;  cur_py += t_min*vy;  cur_pz += t_min*vz
            if vx > 1e-12: cur_px += 1e-8
            elif vx < -1e-12: cur_px -= 1e-8
            if vy > 1e-12: cur_py += 1e-8
            elif vy < -1e-12: cur_py -= 1e-8
            if vz > 1e-12: cur_pz += 1e-8
            elif vz < -1e-12: cur_pz -= 1e-8
            if periodic_xy:
                cur_px = cur_px % Lx
                if cur_px < 0.0: cur_px += Lx
                cur_py = cur_py % Ly
                if cur_py < 0.0: cur_py += Ly
            cur_i = ni;  cur_j = nj;  cur_k = nk
            accumulated_t += t_min


        # Trim the fictive and edge sets to what lies before the hit, on EVERY
        # exit -- as the CPU tracer does (canopi/intersection.py).  The in-loop
        # trim runs only when the next step passes the hit, but a hit in the top
        # (bottom) voxel layer of an upward (downward) ray leaves the grid on that
        # step and breaks out first.  Leaves are visited in list order, so a
        # fictive leaf beyond the hit may already be recorded; left in the set it
        # is charged -1/(1-P) at every later vertex and dBRF/dLAI reads low.
        if contact:
            n_out = 0
            for k in range(n_edg[0]):
                if edg_ohs[k] < min_dis:
                    edg_ids[n_out] = edg_ids[k]
                    edg_nvs[n_out] = edg_nvs[k]
                    edg_ohs[n_out] = edg_ohs[k]
                    n_out += 1
            n_edg[0] = n_out
            n_out = 0
            for k in range(n_fic[0]):
                if fic_ohs[k] < min_dis:
                    fic_ids[n_out] = fic_ids[k]
                    fic_nvs[n_out] = fic_nvs[k]
                    fic_ohs[n_out] = fic_ohs[k]
                    n_out += 1
            n_fic[0] = n_out
        return contact

    # ─────────────────────────────────────────────────────────────────────────
    # _dda_fwd_deriv_sp — per-species P_LAI variant of _dda_fwd_deriv.
    # Identical except leaf activation uses P_LAI_sp_arr[leaf_species_arr[i]].

    @cuda.jit(device=True)
    def _dda_fwd_deriv_sp(
        Px, Py, Pz, vx, vy, vz, origin,
        leaf_px, leaf_py, leaf_pz,
        leaf_nx, leaf_ny, leaf_nz,
        leaf_radius, photon_id, global_seed,
        voxel_count, voxel_leaves, vs, mvx, mvy, mvz, periodic_xy,
        out_idx, out_Hx, out_Hy, out_Hz, out_nv,
        fic_ids, fic_nvs, fic_ohs, n_fic,
        edg_ids, edg_nvs, edg_ohs, n_edg,
        edge_r_thresh,
        LAI_r, std_LAI, use_lognormal, sigma_ln,
        leaf_species_arr, P_LAI_sp_arr,
    ):
        contact = False;  min_dis = 1e18;  accumulated_t = 0.0
        Lx = mvx * vs;  Ly = mvy * vs
        if periodic_xy:
            _Px = Px % Lx
            if _Px < 0.0: _Px += Lx
            _Py = Py % Ly
            if _Py < 0.0: _Py += Ly
        else:
            _Px = min(max(Px, 0.0), Lx - 1e-8)
            _Py = min(max(Py, 0.0), Ly - 1e-8)
        _Pz    = min(max(Pz, -mvz * vs + 1e-8), -1e-8)
        cur_i  = min(int(_Px / vs), mvx - 1)
        cur_j  = min(int(_Py / vs), mvy - 1)
        cur_k  = min(int(math.fabs(_Pz) / vs), mvz - 1)
        cur_px = _Px;  cur_py = _Py;  cur_pz = _Pz
        max_steps = (mvx + mvy + mvz) * 4 + 20
        n_fic[0] = 0;  n_edg[0] = 0

        for _s in range(max_steps):
            vox = cur_i + cur_j * mvx + cur_k * mvx * mvy
            for k in range(voxel_count[vox]):
                i = voxel_leaves[vox, k]
                if i == origin:
                    continue
                ptx = leaf_px[i] - cur_px;  pty = leaf_py[i] - cur_py;  ptz = leaf_pz[i] - cur_pz
                if periodic_xy:
                    if ptx >  Lx * 0.5: ptx -= Lx
                    if ptx < -Lx * 0.5: ptx += Lx
                    if pty >  Ly * 0.5: pty -= Ly
                    if pty < -Ly * 0.5: pty += Ly
                den = vx*leaf_nx[i] + vy*leaf_ny[i] + vz*leaf_nz[i]
                if math.fabs(den) < 1e-12:
                    continue
                OH = (ptx*leaf_nx[i] + pty*leaf_ny[i] + ptz*leaf_nz[i]) / den
                oh_global = accumulated_t + OH
                if OH < 0.0 or oh_global >= min_dis:
                    continue
                MHx = -ptx + OH*vx;  MHy = -pty + OH*vy;  MHz = -ptz + OH*vz
                nv  = math.sqrt(MHx*MHx + MHy*MHy + MHz*MHz)
                if use_lognormal:
                    r_i = _leaf_radius_draw_ln(photon_id, i, global_seed, LAI_r, sigma_ln)
                else:
                    r_i = _leaf_radius_draw(photon_id, i, global_seed, LAI_r, std_LAI)
                if nv < r_i:
                    p_sp_i = P_LAI_sp_arr[leaf_species_arr[i]]   # per-species P_LAI
                    if _leaf_real(photon_id, i, global_seed, p_sp_i):
                        min_dis = oh_global
                        out_idx[0] = i;  out_nv[0] = nv
                        out_Hx[0] = cur_px + OH*vx;  out_Hy[0] = cur_py + OH*vy
                        out_Hz[0] = cur_pz + OH*vz;  contact = True
                    else:
                        if _find_id(fic_ids, n_fic[0], i) < 0 and n_fic[0] < _MAX_FIC:
                            fic_ids[n_fic[0]] = i;  fic_nvs[n_fic[0]] = nv
                            fic_ohs[n_fic[0]] = oh_global;  n_fic[0] += 1
                elif nv < edge_r_thresh:
                    if _find_id(edg_ids, n_edg[0], i) < 0 and n_edg[0] < _MAX_FIC:
                        edg_ids[n_edg[0]] = i;  edg_nvs[n_edg[0]] = nv
                        edg_ohs[n_edg[0]] = oh_global;  n_edg[0] += 1

            t_min = 1e18;  ni = cur_i;  nj = cur_j;  nk = cur_k
            if math.fabs(vx) > 1e-12:
                ex = 1 if vx > 0 else -1
                tx = math.fabs((cur_i+(1 if ex>0 else 0))*vs - cur_px) / math.fabs(vx)
                if tx < 1e-8: tx = math.fabs((cur_i+(1 if ex>0 else 0)+ex)*vs - cur_px) / math.fabs(vx)
                if tx > 1e-8 and tx < t_min: t_min = tx; ni = cur_i + ex; nj = cur_j; nk = cur_k
            if math.fabs(vy) > 1e-12:
                ey = 1 if vy > 0 else -1
                ty = math.fabs((cur_j+(1 if ey>0 else 0))*vs - cur_py) / math.fabs(vy)
                if ty < 1e-8: ty = math.fabs((cur_j+(1 if ey>0 else 0)+ey)*vs - cur_py) / math.fabs(vy)
                if ty > 1e-8 and ty < t_min: t_min = ty; ni = cur_i; nj = cur_j + ey; nk = cur_k
            if math.fabs(vz) > 1e-12:
                ez = -1 if vz > 0 else 1
                tz = math.fabs((cur_k+(1 if ez>0 else 0))*vs - math.fabs(cur_pz)) / math.fabs(vz)
                if tz < 1e-8: tz = math.fabs((cur_k+(1 if ez>0 else 0)+ez)*vs - math.fabs(cur_pz)) / math.fabs(vz)
                if tz > 1e-8 and tz < t_min: t_min = tz; ni = cur_i; nj = cur_j; nk = cur_k + ez
            if nk < 0 or nk >= mvz:
                break
            if periodic_xy:
                if ni < 0: ni += mvx
                elif ni >= mvx: ni -= mvx
                if nj < 0: nj += mvy
                elif nj >= mvy: nj -= mvy
            else:
                if ni < 0 or ni >= mvx or nj < 0 or nj >= mvy:
                    break
            if contact and accumulated_t + t_min > min_dis:
                n_out = 0
                for k in range(n_edg[0]):
                    if edg_ohs[k] < min_dis:
                        edg_ids[n_out]=edg_ids[k]; edg_nvs[n_out]=edg_nvs[k]
                        edg_ohs[n_out]=edg_ohs[k]; n_out += 1
                n_edg[0] = n_out;  n_out = 0
                for k in range(n_fic[0]):
                    if fic_ohs[k] < min_dis:
                        fic_ids[n_out]=fic_ids[k]; fic_nvs[n_out]=fic_nvs[k]
                        fic_ohs[n_out]=fic_ohs[k]; n_out += 1
                n_fic[0] = n_out;  break
            cur_px += t_min*vx;  cur_py += t_min*vy;  cur_pz += t_min*vz
            if vx > 1e-12: cur_px += 1e-8
            elif vx < -1e-12: cur_px -= 1e-8
            if vy > 1e-12: cur_py += 1e-8
            elif vy < -1e-12: cur_py -= 1e-8
            if periodic_xy:
                cur_px = cur_px % Lx
                if cur_px < 0.0: cur_px += Lx
                cur_py = cur_py % Ly
                if cur_py < 0.0: cur_py += Ly
            cur_i = ni;  cur_j = nj;  cur_k = nk
            accumulated_t += t_min


        # Trim the fictive and edge sets to what lies before the hit, on EVERY
        # exit -- as the CPU tracer does (canopi/intersection.py).  The in-loop
        # trim runs only when the next step passes the hit, but a hit in the top
        # (bottom) voxel layer of an upward (downward) ray leaves the grid on that
        # step and breaks out first.  Leaves are visited in list order, so a
        # fictive leaf beyond the hit may already be recorded; left in the set it
        # is charged -1/(1-P) at every later vertex and dBRF/dLAI reads low.
        if contact:
            n_out = 0
            for k in range(n_edg[0]):
                if edg_ohs[k] < min_dis:
                    edg_ids[n_out] = edg_ids[k]
                    edg_nvs[n_out] = edg_nvs[k]
                    edg_ohs[n_out] = edg_ohs[k]
                    n_out += 1
            n_edg[0] = n_out
            n_out = 0
            for k in range(n_fic[0]):
                if fic_ohs[k] < min_dis:
                    fic_ids[n_out] = fic_ids[k]
                    fic_nvs[n_out] = fic_nvs[k]
                    fic_ohs[n_out] = fic_ohs[k]
                    n_out += 1
            n_fic[0] = n_out
        return contact

    # ─────────────────────────────────────────────────────────────────────────
    # _nee_deriv — NEE visibility check with fictive/edge collection
    # ─────────────────────────────────────────────────────────────────────────

    @cuda.jit(device=True)
    def _nee_deriv(
        Hx, Hy, Hz, ox, oy, oz, origin,
        leaf_px, leaf_py, leaf_pz,
        leaf_nx, leaf_ny, leaf_nz,
        leaf_radius, photon_id, global_seed, P_LAI,
        voxel_count, voxel_leaves, vs, mvx, mvy, mvz, periodic_xy,
        nee_fic_ids, nee_fic_nvs, n_nfic,
        nee_edg_ids, nee_edg_nvs, n_nedg,
        edge_r_thresh,
        LAI_r, std_LAI, use_lognormal, sigma_ln, # per-photon radius distribution params
    ):
        """
        NEE ray from (Hx,Hy,Hz) toward obs direction (ox,oy,oz).
        Returns True if unblocked (visible).
        Also populates nee_fic_* / nee_edg_* for derivative bookkeeping.
        Both counters are reset to 0 on entry.
        Uses per-photon radius via _leaf_radius_draw() for consistency with
        the derivative formula's q_j = P(r > nv).
        """
        visible = True
        Lx = mvx * vs;  Ly = mvy * vs
        EPS = 1e-8

        if periodic_xy:
            _Px = Hx % Lx
            if _Px < 0.0: _Px += Lx
            _Py = Hy % Ly
            if _Py < 0.0: _Py += Ly
        else:
            _Px = min(max(Hx, 0.0), Lx - EPS)
            _Py = min(max(Hy, 0.0), Ly - EPS)
        _Pz = min(max(Hz, -vs * mvz + EPS), -EPS)

        cur_i  = min(int(_Px / vs), mvx - 1)
        cur_j  = min(int(_Py / vs), mvy - 1)
        cur_k  = min(int(math.fabs(_Pz) / vs), mvz - 1)
        cur_px = _Px;  cur_py = _Py;  cur_pz = _Pz
        max_steps = (mvx + mvy + mvz) * 4 + 20

        n_nfic[0] = 0;  n_nedg[0] = 0

        for _s in range(max_steps):
            vox = cur_i + cur_j * mvx + cur_k * mvx * mvy
            for k in range(voxel_count[vox]):
                i = voxel_leaves[vox, k]
                if i == origin:
                    continue
                ptx = leaf_px[i] - cur_px
                pty = leaf_py[i] - cur_py
                ptz = leaf_pz[i] - cur_pz
                # if periodic_xy:
                #     if ptx >  Lx * 0.5: ptx -= Lx
                #     if ptx < -Lx * 0.5: ptx += Lx
                #     if pty >  Ly * 0.5: pty -= Ly
                #     if pty < -Ly * 0.5: pty += Ly
                den = ox*leaf_nx[i] + oy*leaf_ny[i] + oz*leaf_nz[i]
                if math.fabs(den) < 1e-12:
                    continue
                OH = (ptx*leaf_nx[i] + pty*leaf_ny[i] + ptz*leaf_nz[i]) / den
                if OH < 0.0:
                    continue
                MHx2 = -ptx + OH*ox;  MHy2 = -pty + OH*oy;  MHz2 = -ptz + OH*oz
                nv = math.sqrt(MHx2*MHx2 + MHy2*MHy2 + MHz2*MHz2)

                if use_lognormal:
                    r_i = _leaf_radius_draw_ln(photon_id, i, global_seed, LAI_r, sigma_ln)
                else:
                    r_i = _leaf_radius_draw(photon_id, i, global_seed, LAI_r, std_LAI)

                if nv < r_i:
                    if _leaf_real(photon_id, i, global_seed, P_LAI):   # blocked
                        visible = False
                        break
                    else:                                              # fictive
                        if _find_id(nee_fic_ids, n_nfic[0], i) < 0 and n_nfic[0] < _MAX_FIC:
                            nee_fic_ids[n_nfic[0]] = i
                            nee_fic_nvs[n_nfic[0]] = nv
                            n_nfic[0] += 1
                elif nv < edge_r_thresh:                    # edge
                    if _find_id(nee_edg_ids, n_nedg[0], i) < 0 and n_nedg[0] < _MAX_FIC:
                        nee_edg_ids[n_nedg[0]] = i
                        nee_edg_nvs[n_nedg[0]] = nv
                        n_nedg[0] += 1

            if not visible:
                break

            t_min = 1e18;  step_axis = 0
            ni = cur_i;  nj = cur_j;  nk = cur_k
            if math.fabs(ox) > 1e-12:
                ex = 1 if ox > 0 else -1
                tx = math.fabs((cur_i + (1 if ex > 0 else 0))*vs - cur_px) / math.fabs(ox)
                if tx < 1e-8: tx = math.fabs((cur_i + (1 if ex > 0 else 0) + ex)*vs - cur_px) / math.fabs(ox)
                if tx > 1e-8 and tx < t_min: t_min = tx; step_axis = 1; ni = cur_i + ex; nj = cur_j; nk = cur_k
            if math.fabs(oy) > 1e-12:
                ey = 1 if oy > 0 else -1
                ty = math.fabs((cur_j + (1 if ey > 0 else 0))*vs - cur_py) / math.fabs(oy)
                if ty < 1e-8: ty = math.fabs((cur_j + (1 if ey > 0 else 0) + ey)*vs - cur_py) / math.fabs(oy)
                if ty > 1e-8 and ty < t_min: t_min = ty; step_axis = 2; ni = cur_i; nj = cur_j + ey; nk = cur_k
            if math.fabs(oz) > 1e-12:
                ez = -1 if oz > 0 else 1
                tz = math.fabs((cur_k + (1 if ez > 0 else 0))*vs - math.fabs(cur_pz)) / math.fabs(oz)
                if tz < 1e-8: tz = math.fabs((cur_k + (1 if ez > 0 else 0) + ez)*vs - math.fabs(cur_pz)) / math.fabs(oz)
                if tz > 1e-8 and tz < t_min: t_min = tz; step_axis = 3; ni = cur_i; nj = cur_j; nk = cur_k + ez
            if nk < 0 or nk >= mvz:
                break
            if periodic_xy:
                if ni < 0: ni += mvx
                elif ni >= mvx: ni -= mvx
                if nj < 0: nj += mvy
                elif nj >= mvy: nj -= mvy
            else:
                if ni < 0 or ni >= mvx or nj < 0 or nj >= mvy:
                    break
            cur_px += t_min*ox;  cur_py += t_min*oy;  cur_pz += t_min*oz
            if ox > 1e-12: cur_px += 1e-8
            elif ox < -1e-12: cur_px -= 1e-8
            if oy > 1e-12: cur_py += 1e-8
            elif oy < -1e-12: cur_py -= 1e-8
            if oz > 1e-12: cur_pz += 1e-8
            elif oz < -1e-12: cur_pz -= 1e-8
            if periodic_xy:
                cur_px = cur_px % Lx
                if cur_px < 0.0: cur_px += Lx
                cur_py = cur_py % Ly
                if cur_py < 0.0: cur_py += Ly
            cur_i = ni;  cur_j = nj;  cur_k = nk

        return visible

    # ─────────────────────────────────────────────────────────────────────────
    # _nee_deriv_sp — as _nee_deriv, but the blocking test can be species-aware
    # ─────────────────────────────────────────────────────────────────────────

    @cuda.jit(device=True)
    def _nee_deriv_sp(
        Hx, Hy, Hz, ox, oy, oz, origin,
        leaf_px, leaf_py, leaf_pz,
        leaf_nx, leaf_ny, leaf_nz,
        leaf_radius, photon_id, global_seed, P_LAI,
        leaf_species_arr, P_LAI_sp_arr, species_shadow,
        voxel_count, voxel_leaves, vs, mvx, mvy, mvz, periodic_xy,
        nee_fic_ids, nee_fic_nvs, n_nfic,
        nee_edg_ids, nee_edg_nvs, n_nedg,
        edge_r_thresh,
        LAI_r, std_LAI, use_lognormal, sigma_ln, # per-photon radius distribution params
    ):
        """
        NEE ray from (Hx,Hy,Hz) toward obs direction (ox,oy,oz).
        Returns True if unblocked (visible).
        Also populates nee_fic_* / nee_edg_* for derivative bookkeeping.
        Both counters are reset to 0 on entry.
        Uses per-photon radius via _leaf_radius_draw() for consistency with
        the derivative formula's q_j = P(r > nv).
        
        Species-aware variant of _nee_deriv: when species_shadow is non-zero the
        blocking test uses P_LAI_sp_arr[species] instead of the scalar P_LAI, so a
        per-species LAI change is felt by the shadow ray as well as the camera path.
        With species_shadow = 0 it is identical to _nee_deriv.
        """
        visible = True
        Lx = mvx * vs;  Ly = mvy * vs
        EPS = 1e-8

        if periodic_xy:
            _Px = Hx % Lx
            if _Px < 0.0: _Px += Lx
            _Py = Hy % Ly
            if _Py < 0.0: _Py += Ly
        else:
            _Px = min(max(Hx, 0.0), Lx - EPS)
            _Py = min(max(Hy, 0.0), Ly - EPS)
        _Pz = min(max(Hz, -vs * mvz + EPS), -EPS)

        cur_i  = min(int(_Px / vs), mvx - 1)
        cur_j  = min(int(_Py / vs), mvy - 1)
        cur_k  = min(int(math.fabs(_Pz) / vs), mvz - 1)
        cur_px = _Px;  cur_py = _Py;  cur_pz = _Pz
        max_steps = (mvx + mvy + mvz) * 4 + 20

        n_nfic[0] = 0;  n_nedg[0] = 0

        for _s in range(max_steps):
            vox = cur_i + cur_j * mvx + cur_k * mvx * mvy
            for k in range(voxel_count[vox]):
                i = voxel_leaves[vox, k]
                if i == origin:
                    continue
                ptx = leaf_px[i] - cur_px
                pty = leaf_py[i] - cur_py
                ptz = leaf_pz[i] - cur_pz
                # if periodic_xy:
                #     if ptx >  Lx * 0.5: ptx -= Lx
                #     if ptx < -Lx * 0.5: ptx += Lx
                #     if pty >  Ly * 0.5: pty -= Ly
                #     if pty < -Ly * 0.5: pty += Ly
                den = ox*leaf_nx[i] + oy*leaf_ny[i] + oz*leaf_nz[i]
                if math.fabs(den) < 1e-12:
                    continue
                OH = (ptx*leaf_nx[i] + pty*leaf_ny[i] + ptz*leaf_nz[i]) / den
                if OH < 0.0:
                    continue
                MHx2 = -ptx + OH*ox;  MHy2 = -pty + OH*oy;  MHz2 = -ptz + OH*oz
                nv = math.sqrt(MHx2*MHx2 + MHy2*MHy2 + MHz2*MHz2)

                if use_lognormal:
                    r_i = _leaf_radius_draw_ln(photon_id, i, global_seed, LAI_r, sigma_ln)
                else:
                    r_i = _leaf_radius_draw(photon_id, i, global_seed, LAI_r, std_LAI)

                if nv < r_i:
                    p_blk = P_LAI
                    if species_shadow:
                        p_blk = P_LAI_sp_arr[leaf_species_arr[i]]
                    if _leaf_real(photon_id, i, global_seed, p_blk):   # blocked
                        visible = False
                        break
                    else:                                              # fictive
                        if _find_id(nee_fic_ids, n_nfic[0], i) < 0 and n_nfic[0] < _MAX_FIC:
                            nee_fic_ids[n_nfic[0]] = i
                            nee_fic_nvs[n_nfic[0]] = nv
                            n_nfic[0] += 1
                elif nv < edge_r_thresh:                    # edge
                    if _find_id(nee_edg_ids, n_nedg[0], i) < 0 and n_nedg[0] < _MAX_FIC:
                        nee_edg_ids[n_nedg[0]] = i
                        nee_edg_nvs[n_nedg[0]] = nv
                        n_nedg[0] += 1

            if not visible:
                break

            t_min = 1e18;  step_axis = 0
            ni = cur_i;  nj = cur_j;  nk = cur_k
            if math.fabs(ox) > 1e-12:
                ex = 1 if ox > 0 else -1
                tx = math.fabs((cur_i + (1 if ex > 0 else 0))*vs - cur_px) / math.fabs(ox)
                if tx < 1e-8: tx = math.fabs((cur_i + (1 if ex > 0 else 0) + ex)*vs - cur_px) / math.fabs(ox)
                if tx > 1e-8 and tx < t_min: t_min = tx; step_axis = 1; ni = cur_i + ex; nj = cur_j; nk = cur_k
            if math.fabs(oy) > 1e-12:
                ey = 1 if oy > 0 else -1
                ty = math.fabs((cur_j + (1 if ey > 0 else 0))*vs - cur_py) / math.fabs(oy)
                if ty < 1e-8: ty = math.fabs((cur_j + (1 if ey > 0 else 0) + ey)*vs - cur_py) / math.fabs(oy)
                if ty > 1e-8 and ty < t_min: t_min = ty; step_axis = 2; ni = cur_i; nj = cur_j + ey; nk = cur_k
            if math.fabs(oz) > 1e-12:
                ez = -1 if oz > 0 else 1
                tz = math.fabs((cur_k + (1 if ez > 0 else 0))*vs - math.fabs(cur_pz)) / math.fabs(oz)
                if tz < 1e-8: tz = math.fabs((cur_k + (1 if ez > 0 else 0) + ez)*vs - math.fabs(cur_pz)) / math.fabs(oz)
                if tz > 1e-8 and tz < t_min: t_min = tz; step_axis = 3; ni = cur_i; nj = cur_j; nk = cur_k + ez
            if nk < 0 or nk >= mvz:
                break
            if periodic_xy:
                if ni < 0: ni += mvx
                elif ni >= mvx: ni -= mvx
                if nj < 0: nj += mvy
                elif nj >= mvy: nj -= mvy
            else:
                if ni < 0 or ni >= mvx or nj < 0 or nj >= mvy:
                    break
            cur_px += t_min*ox;  cur_py += t_min*oy;  cur_pz += t_min*oz
            if ox > 1e-12: cur_px += 1e-8
            elif ox < -1e-12: cur_px -= 1e-8
            if oy > 1e-12: cur_py += 1e-8
            elif oy < -1e-12: cur_py -= 1e-8
            if oz > 1e-12: cur_pz += 1e-8
            elif oz < -1e-12: cur_pz -= 1e-8
            if periodic_xy:
                cur_px = cur_px % Lx
                if cur_px < 0.0: cur_px += Lx
                cur_py = cur_py % Ly
                if cur_py < 0.0: cur_py += Ly
            cur_i = ni;  cur_j = nj;  cur_k = nk

        return visible

    # ─────────────────────────────────────────────────────────────────────────
    # _trace_kernel — BRF only  (original, unchanged)
    # ─────────────────────────────────────────────────────────────────────────

    @cuda.jit
    def _trace_kernel(
        leaf_px, leaf_py, leaf_pz,
        leaf_nx, leaf_ny, leaf_nz, leaf_radius,
        voxel_count, voxel_leaves, vs, mvx, mvy, mvz,
        obs_vx, obs_vy, obs_vz,
        sun_vx, sun_vy, sun_vz,
        rho, tau, rsoil, max_depth, z_bottom,
        P_LAI, global_seed,
        periodic_xy,
        start_x, start_y,
        rng_states,
        lum_out,
    ):
        """One thread = one photon.  BRF only."""
        n     = cuda.grid(1)
        N     = lum_out.shape[0]
        n_obs = obs_vx.shape[0]
        if n >= N:
            return

        out_idx = cuda.local.array(1, dtype=nb.int32)
        out_Hx  = cuda.local.array(1, dtype=nb.float64)
        out_Hy  = cuda.local.array(1, dtype=nb.float64)
        out_Hz  = cuda.local.array(1, dtype=nb.float64)
        out_nv  = cuda.local.array(1, dtype=nb.float64)

        Px = start_x[n];  Py = start_y[n];  Pz = 0.0
        vx = sun_vx;      vy = sun_vy;      vz = sun_vz
        origin = -1;  energy = 1.0;  depth = 0

        while energy > 1e-5 and depth < max_depth:
            out_idx[0] = -1
            hit = _dda_contact(
                Px, Py, Pz, vx, vy, vz, origin,
                leaf_px, leaf_py, leaf_pz,
                leaf_nx, leaf_ny, leaf_nz,
                leaf_radius, n, global_seed, P_LAI,
                voxel_count, voxel_leaves, vs, mvx, mvy, mvz,
                out_idx, out_Hx, out_Hy, out_Hz, out_nv,
                periodic_xy,
            )

            if hit:
                li  = out_idx[0]
                Hx  = out_Hx[0];  Hy = out_Hy[0];  Hz = out_Hz[0]
                lnx = leaf_nx[li];  lny = leaf_ny[li];  lnz = leaf_nz[li]
                cos_in = -(vx*lnx + vy*lny + vz*lnz)

                for io in range(n_obs):
                    ox = obs_vx[io];  oy = obs_vy[io];  oz = obs_vz[io]
                    if oz <= 0.0:
                        continue
                    cos_out   = ox*lnx + oy*lny + oz*lnz
                    cos_earth = oz if oz > 1e-6 else 1e-6
                    f = rho/3.14159265 if cos_in*cos_out > 0 else tau/3.14159265
                    norm_fac  = math.fabs(cos_out) / cos_earth

                    vis_idx = cuda.local.array(1, dtype=nb.int32)
                    vis_Hx  = cuda.local.array(1, dtype=nb.float64)
                    vis_Hy  = cuda.local.array(1, dtype=nb.float64)
                    vis_Hz  = cuda.local.array(1, dtype=nb.float64)
                    vis_nv  = cuda.local.array(1, dtype=nb.float64)
                    vis_idx[0] = -1
                    blocked = _dda_contact(
                        Hx, Hy, Hz, ox, oy, oz, li,
                        leaf_px, leaf_py, leaf_pz,
                        leaf_nx, leaf_ny, leaf_nz,
                        leaf_radius, n, global_seed, P_LAI,
                        voxel_count, voxel_leaves, vs, mvx, mvy, mvz,
                        vis_idx, vis_Hx, vis_Hy, vis_Hz, vis_nv,
                        periodic_xy,
                    )
                    if not blocked:
                        cuda.atomic.add(lum_out, (n, io), energy * f * norm_fac)

                u1 = xoroshiro128p_uniform_float64(rng_states, n)
                u2 = xoroshiro128p_uniform_float64(rng_states, n)
                u3 = xoroshiro128p_uniform_float64(rng_states, n)
                sr  = math.sqrt(u1);  phi = 6.28318530 * u2
                sx  = sr * math.cos(phi);  sy = sr * math.sin(phi)
                sz  = math.sqrt(max(0.0, 1.0 - u1))
                cif = vx*lnx + vy*lny + vz*lnz
                if u3 < rho / (rho + tau):
                    nox = lnx if cif < 0 else -lnx
                    noy = lny if cif < 0 else -lny
                    noz = lnz if cif < 0 else -lnz
                else:
                    nox = -lnx if cif < 0 else lnx
                    noy = -lny if cif < 0 else lny
                    noz = -lnz if cif < 0 else lnz
                if math.fabs(nox) < 0.9: tx=1.; ty=0.; tz=0.
                else:                    tx=0.; ty=1.; tz=0.
                d2=tx*nox+ty*noy+tz*noz; tx-=d2*nox; ty-=d2*noy; tz-=d2*noz
                tl=math.sqrt(tx*tx+ty*ty+tz*tz); tx/=tl; ty/=tl; tz/=tl
                bx=noy*tz-noz*ty; by=noz*tx-nox*tz; bz=nox*ty-noy*tx
                vx=sx*tx+sy*bx+sz*nox; vy=sx*ty+sy*by+sz*noy; vz=sx*tz+sy*bz+sz*noz
                energy *= (rho + tau)
                Px=Hx; Py=Hy; Pz=Hz; origin=li; depth+=1

            else:
                tz_top = (0.0      - Pz) / vz if vz >  1e-12 else 1e18
                tz_bot = (z_bottom - Pz) / vz if vz < -1e-12 else 1e18
                if tz_top < tz_bot:
                    break
                soil_Px = Px + tz_bot*vx;  soil_Py = Py + tz_bot*vy
                for io in range(n_obs):
                    ox=obs_vx[io]; oy=obs_vy[io]; oz=obs_vz[io]
                    if oz <= 0.0: continue
                    vis_idx2=cuda.local.array(1,dtype=nb.int32)
                    vis_Hx2 =cuda.local.array(1,dtype=nb.float64)
                    vis_Hy2 =cuda.local.array(1,dtype=nb.float64)
                    vis_Hz2 =cuda.local.array(1,dtype=nb.float64)
                    vis_nv2 =cuda.local.array(1,dtype=nb.float64)
                    vis_idx2[0]=-1
                    blocked2=_dda_contact(
                        soil_Px,soil_Py,z_bottom+1e-7,ox,oy,oz,-1,
                        leaf_px,leaf_py,leaf_pz,leaf_nx,leaf_ny,leaf_nz,
                        leaf_radius, n, global_seed, P_LAI,
                        voxel_count,voxel_leaves,vs,mvx,mvy,mvz,
                        vis_idx2,vis_Hx2,vis_Hy2,vis_Hz2,vis_nv2,
                        periodic_xy,
                    )
                    if not blocked2:
                        cuda.atomic.add(lum_out,(n,io),energy*rsoil/3.14159265)
                u1b=xoroshiro128p_uniform_float64(rng_states,n)
                u2b=xoroshiro128p_uniform_float64(rng_states,n)
                sr2=math.sqrt(u1b); phi2=6.28318530*u2b
                vx=sr2*math.cos(phi2); vy=sr2*math.sin(phi2)
                vz=math.sqrt(max(0.0,1.0-u1b))
                energy*=rsoil
                Px=soil_Px; Py=soil_Py; Pz=z_bottom+1e-7
                origin=-1; depth+=1

    # ─────────────────────────────────────────────────────────────────────────
    # _trace_deriv_kernel — BRF + dBRF/dLAI in one pass
    # ─────────────────────────────────────────────────────────────────────────

    @cuda.jit
    def _trace_deriv_kernel(
        leaf_px, leaf_py, leaf_pz,
        leaf_nx, leaf_ny, leaf_nz, leaf_radius,
        voxel_count, voxel_leaves, vs, mvx, mvy, mvz,
        obs_vx, obs_vy, obs_vz,
        sun_vx, sun_vy, sun_vz,
        rho, tau, rsoil, max_depth, z_bottom,
        global_seed,
        periodic_xy,
        start_x, start_y,
        rng_states,
        lum_out,        # (N, n_obs)
        deriv_fic_out,  # (N, n_obs)
        deriv_edg_out,  # (N, n_obs)
        sum_q_out,      # (N, n_obs)  sum_q_nc / LAI_Max per photon
        nH_out,         # (N, n_obs)  n_H / (P * LAI_Max) per photon
        brkt_out,       # (N, n_obs)  (n_H/P - sum_q_nc) / LAI_Max per photon (debug)
        nH_path_out,    # (N,)        n_H[0]    at path end — unique active leaves
        nF_path_out,    # (N,)        n_ffic[0] at path end — unique fwd fictive leaves
        K_out,          # (N,)        total leaf scatterings per photon (K for dBRF/dρ)
        drho_out,       # (N, n_obs)  rho score per photon: Σ_contrib k_eff·ref (÷ρ later)
        LAI_r, std_LAI, LAI_Max, P_LAI, n_leaf_norm,
        edge_r_thresh, use_lognormal, sigma_ln,
    ):
        """
        One CUDA thread = one photon.
        Computes BRF and dBRF/dLAI (fictive + edge) in one forward pass.

        All intermediate derivative sets (H, fwd_fic, fwd_edg, nee_fic,
        nee_edg, b_fic, bnH_edg, HB_nv_B) live in thread-local fixed arrays.

        Derivative algorithm mirrors compute_photon_derivative() in derivative.py:
          H        — real leaf contacts on the forward path (max n_v per leaf)
          fwd_fic  — fictive leaves accumulated across all forward segments (min n_v)
          fwd_edg  — edge    leaves accumulated across all forward segments (min n_v)
          Per (bounce x obs_dir) when NEE is unblocked:
            B\\H fic  = fwd_fic ∪ nee_fic  (min n_v, all fic ∉ H by definition)
            B\\H edg  = (fwd_edg ∪ nee_edg) whose id ∉ H_ids  (min n_v)
            H∩B      = H contacts whose id ∈ (fwd_edg ∪ nee_edg), track max d_B
        """
        n     = cuda.grid(1)
        N     = lum_out.shape[0]
        n_obs = obs_vx.shape[0]
        if n >= N:
            return

        EPS = 1e-12

        # ── Thread-local derivative state (photon lifetime) ───────────────────
        H_ids = cuda.local.array(_MAX_C,   dtype=nb.int32)
        H_nvs = cuda.local.array(_MAX_C,   dtype=nb.float64)
        n_H   = cuda.local.array(1,        dtype=nb.int32)
        n_H[0] = 0

        fwd_fic_ids = cuda.local.array(_MAX_FIC, dtype=nb.int32)
        fwd_fic_nvs = cuda.local.array(_MAX_FIC, dtype=nb.float64)
        n_ffic      = cuda.local.array(1,        dtype=nb.int32)
        n_ffic[0]   = 0

        fwd_edg_ids = cuda.local.array(_MAX_FIC, dtype=nb.int32)
        fwd_edg_nvs = cuda.local.array(_MAX_FIC, dtype=nb.float64)
        n_fedg      = cuda.local.array(1,        dtype=nb.int32)
        n_fedg[0]   = 0

        # ── Per-segment scratch (reset by _dda_fwd_deriv on entry) ────────────
        seg_fic_ids = cuda.local.array(_MAX_FIC, dtype=nb.int32)
        seg_fic_nvs = cuda.local.array(_MAX_FIC, dtype=nb.float64)
        seg_fic_ohs = cuda.local.array(_MAX_FIC, dtype=nb.float64)
        n_sfic      = cuda.local.array(1,        dtype=nb.int32)

        seg_edg_ids = cuda.local.array(_MAX_FIC, dtype=nb.int32)
        seg_edg_ohs = cuda.local.array(_MAX_FIC, dtype=nb.float64)
        seg_edg_nvs = cuda.local.array(_MAX_FIC, dtype=nb.float64)
        n_sedg      = cuda.local.array(1,        dtype=nb.int32)

        # ── Per-obs-dir NEE scratch (reset by _nee_deriv on entry) ───────────
        nee_fic_ids = cuda.local.array(_MAX_FIC, dtype=nb.int32)
        nee_fic_nvs = cuda.local.array(_MAX_FIC, dtype=nb.float64)
        n_nfic      = cuda.local.array(1,        dtype=nb.int32)

        nee_edg_ids = cuda.local.array(_MAX_FIC, dtype=nb.int32)
        nee_edg_nvs = cuda.local.array(_MAX_FIC, dtype=nb.float64)
        n_nedg      = cuda.local.array(1,        dtype=nb.int32)

        # ── Per-obs-dir merged set scratch ────────────────────────────────────
        b_fic_ids   = cuda.local.array(_MAX_FIC, dtype=nb.int32)
        b_fic_nvs   = cuda.local.array(_MAX_FIC, dtype=nb.float64)
        bnH_edg_ids = cuda.local.array(_MAX_FIC, dtype=nb.int32)
        bnH_edg_nvs = cuda.local.array(_MAX_FIC, dtype=nb.float64)
        HB_nv_B     = cuda.local.array(_MAX_C,   dtype=nb.float64)
        HB_in_B     = cuda.local.array(_MAX_C,   dtype=nb.int32)

        # ── DDA hit scratch ───────────────────────────────────────────────────
        out_idx = cuda.local.array(1, dtype=nb.int32)
        out_Hx  = cuda.local.array(1, dtype=nb.float64)
        out_Hy  = cuda.local.array(1, dtype=nb.float64)
        out_Hz  = cuda.local.array(1, dtype=nb.float64)
        out_nv  = cuda.local.array(1, dtype=nb.float64)

        # ── Path state ────────────────────────────────────────────────────────
        Px = start_x[n];  Py = start_y[n];  Pz = 0.0
        vx = sun_vx;      vy = sun_vy;      vz = sun_vz
        origin = -1;  energy = 1.0;  depth = 0;  K_local = 0

        # ═════════════════════════════════════════════════════════════════════
        while energy > 1e-5 and depth < max_depth:

            # ── 1. Forward DDA ─────────────────────────────────────────────
            out_idx[0] = -1
            hit = _dda_fwd_deriv(
                Px, Py, Pz, vx, vy, vz, origin,
                leaf_px, leaf_py, leaf_pz,
                leaf_nx, leaf_ny, leaf_nz,
                leaf_radius, n, global_seed, P_LAI,
                voxel_count, voxel_leaves, vs, mvx, mvy, mvz, periodic_xy,
                out_idx, out_Hx, out_Hy, out_Hz, out_nv,
                seg_fic_ids, seg_fic_nvs, seg_fic_ohs, n_sfic,
                seg_edg_ids, seg_edg_nvs, seg_edg_ohs, n_sedg,
                edge_r_thresh,
                LAI_r, std_LAI, use_lognormal, sigma_ln,
            )

            # ── 2. Merge segment sets into running fwd accumulators ────────
            for k in range(n_sfic[0]):
                vid = seg_fic_ids[k];  nv = seg_fic_nvs[k]
                fi  = _find_id(fwd_fic_ids, n_ffic[0], vid)
                if fi >= 0:
                    if nv < fwd_fic_nvs[fi]: fwd_fic_nvs[fi] = nv
                elif n_ffic[0] < _MAX_FIC:
                    fwd_fic_ids[n_ffic[0]] = vid
                    fwd_fic_nvs[n_ffic[0]] = nv
                    n_ffic[0] += 1

            for k in range(n_sedg[0]):
                vid = seg_edg_ids[k];  nv = seg_edg_nvs[k]
                ei  = _find_id(fwd_edg_ids, n_fedg[0], vid)
                if ei >= 0:
                    if nv < fwd_edg_nvs[ei]: fwd_edg_nvs[ei] = nv
                elif n_fedg[0] < _MAX_FIC:
                    fwd_edg_ids[n_fedg[0]] = vid
                    fwd_edg_nvs[n_fedg[0]] = nv
                    n_fedg[0] += 1

            # =================================================================
            if hit:
            # =================================================================

                li  = out_idx[0]
                Hx  = out_Hx[0];  Hy = out_Hy[0];  Hz = out_Hz[0]
                lnx = leaf_nx[li];  lny = leaf_ny[li];  lnz = leaf_nz[li]
                cos_in = -(vx*lnx + vy*lny + vz*lnz)

                # ── 3. Update H ────────────────────────────────────────────
                h_exist = _find_id(H_ids, n_H[0], li)
                if h_exist >= 0:
                    if out_nv[0] > H_nvs[h_exist]: H_nvs[h_exist] = out_nv[0]
                elif n_H[0] < _MAX_C:
                    H_ids[n_H[0]] = li
                    H_nvs[n_H[0]] = out_nv[0]
                    n_H[0] += 1
                n_H_now = n_H[0]

                # ── 4. NEE per obs direction ───────────────────────────────
                for io in range(n_obs):
                    ox = obs_vx[io];  oy = obs_vy[io];  oz = obs_vz[io]
                    if oz <= 0.0:
                        continue
                    cos_out   = ox*lnx + oy*lny + oz*lnz
                    cos_earth = oz if oz > 1e-6 else 1e-6
                    f = rho/3.14159265 if cos_in*cos_out > 0 else tau/3.14159265
                    norm_fac  = math.fabs(cos_out) / cos_earth

                    visible = _nee_deriv(
                        Hx, Hy, Hz, ox, oy, oz, li,
                        leaf_px, leaf_py, leaf_pz,
                        leaf_nx, leaf_ny, leaf_nz,
                        leaf_radius, n, global_seed, P_LAI,
                        voxel_count, voxel_leaves, vs, mvx, mvy, mvz, periodic_xy,
                        nee_fic_ids, nee_fic_nvs, n_nfic,
                        nee_edg_ids, nee_edg_nvs, n_nedg,
                        edge_r_thresh,
                        LAI_r, std_LAI, use_lognormal, sigma_ln,
                    )
                    if not visible:
                        continue

                    ref = energy * f * norm_fac
                    cuda.atomic.add(lum_out, (n, io), ref)

                    # rho score (per contribution): k_eff = (prior leaf scatters) + 1
                    # (+1 for d(f)/dρ at THIS leaf; f = ρ/π or τ/π → d log f/dρ = 1/ρ
                    #  under the linked ρ=τ convention).  Divided by ρ at the call site.
                    cuda.atomic.add(drho_out, (n, io),
                                    (nb.float64(K_local) + 1.0) * ref)

                    # ── 5. Build per-obs derivative sets ──────────────────

                    # B\H fic = fwd_fic ∪ nee_fic  (min nv)
                    n_bfic = 0
                    for k in range(n_ffic[0]):
                        b_fic_ids[n_bfic] = fwd_fic_ids[k]
                        b_fic_nvs[n_bfic] = fwd_fic_nvs[k]
                        n_bfic += 1
                    for k in range(n_nfic[0]):
                        vid = nee_fic_ids[k];  nv = nee_fic_nvs[k]
                        fi  = _find_id(b_fic_ids, n_bfic, vid)
                        if fi >= 0:
                            if nv < b_fic_nvs[fi]: b_fic_nvs[fi] = nv
                        elif n_bfic < _MAX_FIC:
                            b_fic_ids[n_bfic] = vid
                            b_fic_nvs[n_bfic] = nv
                            n_bfic += 1

                    # H∩B and B\H edge split from fwd_edg ∪ nee_edg
                    for ci in range(n_H_now):
                        HB_nv_B[ci] = -1.0;  HB_in_B[ci] = 0
                    n_bnH = 0

                    for k in range(n_fedg[0]):
                        vid = fwd_edg_ids[k];  nv_B = fwd_edg_nvs[k]
                        h_ci = _find_id(H_ids, n_H_now, vid)
                        if h_ci >= 0:
                            if HB_in_B[h_ci] == 0 or nv_B < HB_nv_B[h_ci]:  # d_l^B = min (Eq. 28)
                                HB_nv_B[h_ci] = nv_B;  HB_in_B[h_ci] = 1
                        else:
                            bi = _find_id(bnH_edg_ids, n_bnH, vid)
                            if bi >= 0:
                                if nv_B < bnH_edg_nvs[bi]: bnH_edg_nvs[bi] = nv_B
                            elif n_bnH < _MAX_FIC:
                                bnH_edg_ids[n_bnH] = vid
                                bnH_edg_nvs[n_bnH] = nv_B
                                n_bnH += 1

                    for k in range(n_nedg[0]):
                        vid = nee_edg_ids[k];  nv_B = nee_edg_nvs[k]
                        h_ci = _find_id(H_ids, n_H_now, vid)
                        if h_ci >= 0:
                            if HB_in_B[h_ci] == 0 or nv_B < HB_nv_B[h_ci]:  # d_l^B = min (Eq. 28)
                                HB_nv_B[h_ci] = nv_B;  HB_in_B[h_ci] = 1
                        else:
                            bi = _find_id(bnH_edg_ids, n_bnH, vid)
                            if bi >= 0:
                                if nv_B < bnH_edg_nvs[bi]: bnH_edg_nvs[bi] = nv_B
                            elif n_bnH < _MAX_FIC:
                                bnH_edg_ids[n_bnH] = vid
                                bnH_edg_nvs[n_bnH] = nv_B
                                n_bnH += 1

                    # ── 6. Fictive estimator ──────────────────────────────
                    # d_fic = ref/LAI_Max * (|H|/P - Σ_{k∈B\H} q_k/(1-P·q_k))
                    sum_q_nc = 0.0
                    for k in range(n_bfic):
                        q_k, _ = _q_dq_any(b_fic_nvs[k], LAI_r, std_LAI, use_lognormal, sigma_ln)
                        sum_q_nc += q_k / max(1.0 - P_LAI * q_k, EPS)
                    for k in range(n_bnH):
                        q_k, _ = _q_dq_any(bnH_edg_nvs[k], LAI_r, std_LAI, use_lognormal, sigma_ln)
                        sum_q_nc += q_k / max(1.0 - P_LAI * q_k, EPS)

                    d_fic = ref / LAI_Max * (n_H_now / P_LAI - sum_q_nc)
                    cuda.atomic.add(deriv_fic_out, (n, io), d_fic)
                    cuda.atomic.add(sum_q_out,     (n, io), sum_q_nc / LAI_Max)
                    cuda.atomic.add(nH_out,        (n, io), n_H_now / (P_LAI * LAI_Max))
                    cuda.atomic.add(brkt_out,        (n, io), n_H_now / (P_LAI * LAI_Max) - sum_q_nc / LAI_Max)

                    # ── 7. Edge estimator ─────────────────────────────────
                    # B_term = Σ_{k∈B\H} P·dq/(1-P·q)
                    sum_B = 0.0
                    for k in range(n_bfic):
                        q_k, dq_k = _q_dq_any(b_fic_nvs[k], LAI_r, std_LAI, use_lognormal, sigma_ln)
                        sum_B += P_LAI * dq_k / max(1.0 - P_LAI * q_k, EPS)
                    for k in range(n_bnH):
                        q_k, dq_k = _q_dq_any(bnH_edg_nvs[k], LAI_r, std_LAI, use_lognormal, sigma_ln)
                        sum_B += P_LAI * dq_k / max(1.0 - P_LAI * q_k, EPS)

                    # H_term and HB_term split on HB_in_B flag
                    sum_H = 0.0;  sum_HB = 0.0
                    for ci in range(n_H_now):
                        q_H, dq_H = _q_dq_any(H_nvs[ci], LAI_r, std_LAI, use_lognormal, sigma_ln)
                        if HB_in_B[ci]:
                            q_B, dq_B = _q_dq_any(HB_nv_B[ci], LAI_r, std_LAI, use_lognormal, sigma_ln)
                            denom = q_H - q_B
                            if math.fabs(denom) > EPS:
                                sum_HB += (dq_H - dq_B) / denom
                        else:
                            sum_H += dq_H / max(q_H, EPS)

                    d_edg = ref / n_leaf_norm / P_LAI * (sum_H - sum_B + sum_HB)
                    cuda.atomic.add(deriv_edg_out, (n, io), d_edg)

                # ── 8. Scatter ────────────────────────────────────────────
                u1 = xoroshiro128p_uniform_float64(rng_states, n)
                u2 = xoroshiro128p_uniform_float64(rng_states, n)
                u3 = xoroshiro128p_uniform_float64(rng_states, n)
                sr  = math.sqrt(u1);  phi = 6.28318530 * u2
                sx  = sr * math.cos(phi);  sy = sr * math.sin(phi)
                sz  = math.sqrt(max(0.0, 1.0 - u1))
                cif = vx*lnx + vy*lny + vz*lnz
                if u3 < rho / (rho + tau):
                    nox = lnx if cif < 0 else -lnx
                    noy = lny if cif < 0 else -lny
                    noz = lnz if cif < 0 else -lnz
                else:
                    nox = -lnx if cif < 0 else lnx
                    noy = -lny if cif < 0 else lny
                    noz = -lnz if cif < 0 else lnz
                if math.fabs(nox) < 0.9: tx=1.; ty=0.; tz=0.
                else:                    tx=0.; ty=1.; tz=0.
                d2=tx*nox+ty*noy+tz*noz; tx-=d2*nox; ty-=d2*noy; tz-=d2*noz
                tl=math.sqrt(tx*tx+ty*ty+tz*tz); tx/=tl; ty/=tl; tz/=tl
                bx=noy*tz-noz*ty; by=noz*tx-nox*tz; bz=nox*ty-noy*tx
                vx=sx*tx+sy*bx+sz*nox; vy=sx*ty+sy*by+sz*noy; vz=sx*tz+sy*bz+sz*noz
                energy *= (rho + tau)
                K_local += 1
                Px=Hx; Py=Hy; Pz=Hz; origin=li; depth+=1

            # =================================================================
            else:    # sky or soil
            # =================================================================

                tz_top = (0.0      - Pz) / vz if vz >  1e-12 else 1e18
                tz_bot = (z_bottom - Pz) / vz if vz < -1e-12 else 1e18
                if tz_top < tz_bot:
                    break

                soil_Px = Px + tz_bot*vx
                soil_Py = Py + tz_bot*vy
                soil_Pz = z_bottom + 1e-7
                n_H_soil = n_H[0]       # soil does NOT add to H

                for io in range(n_obs):
                    ox = obs_vx[io];  oy = obs_vy[io];  oz = obs_vz[io]
                    if oz <= 0.0:
                        continue

                    visible = _nee_deriv(
                        soil_Px, soil_Py, soil_Pz, ox, oy, oz, -1,
                        leaf_px, leaf_py, leaf_pz,
                        leaf_nx, leaf_ny, leaf_nz,
                        leaf_radius, n, global_seed, P_LAI,
                        voxel_count, voxel_leaves, vs, mvx, mvy, mvz, periodic_xy,
                        nee_fic_ids, nee_fic_nvs, n_nfic,
                        nee_edg_ids, nee_edg_nvs, n_nedg,
                        edge_r_thresh,
                        LAI_r, std_LAI, use_lognormal, sigma_ln,
                    )
                    if not visible:
                        continue

                    ref = energy * rsoil / 3.14159265
                    cuda.atomic.add(lum_out, (n, io), ref)

                    # rho score (per contribution): only prior leaf scatters count —
                    # the soil albedo rsoil does NOT depend on ρ (no +1).
                    cuda.atomic.add(drho_out, (n, io), nb.float64(K_local) * ref)

                    # B\H fic for soil
                    n_bfic = 0
                    for k in range(n_ffic[0]):
                        b_fic_ids[n_bfic] = fwd_fic_ids[k]
                        b_fic_nvs[n_bfic] = fwd_fic_nvs[k]
                        n_bfic += 1
                    for k in range(n_nfic[0]):
                        vid = nee_fic_ids[k];  nv = nee_fic_nvs[k]
                        fi  = _find_id(b_fic_ids, n_bfic, vid)
                        if fi >= 0:
                            if nv < b_fic_nvs[fi]: b_fic_nvs[fi] = nv
                        elif n_bfic < _MAX_FIC:
                            b_fic_ids[n_bfic] = vid
                            b_fic_nvs[n_bfic] = nv
                            n_bfic += 1

                    # H∩B and B\H edg for soil
                    for ci in range(n_H_soil):
                        HB_nv_B[ci] = -1.0;  HB_in_B[ci] = 0
                    n_bnH = 0

                    for k in range(n_fedg[0]):
                        vid = fwd_edg_ids[k];  nv_B = fwd_edg_nvs[k]
                        h_ci = _find_id(H_ids, n_H_soil, vid)
                        if h_ci >= 0:
                            if HB_in_B[h_ci] == 0 or nv_B < HB_nv_B[h_ci]:  # d_l^B = min (Eq. 28)
                                HB_nv_B[h_ci] = nv_B;  HB_in_B[h_ci] = 1
                        else:
                            bi = _find_id(bnH_edg_ids, n_bnH, vid)
                            if bi >= 0:
                                if nv_B < bnH_edg_nvs[bi]: bnH_edg_nvs[bi] = nv_B
                            elif n_bnH < _MAX_FIC:
                                bnH_edg_ids[n_bnH] = vid
                                bnH_edg_nvs[n_bnH] = nv_B
                                n_bnH += 1

                    for k in range(n_nedg[0]):
                        vid = nee_edg_ids[k];  nv_B = nee_edg_nvs[k]
                        h_ci = _find_id(H_ids, n_H_soil, vid)
                        if h_ci >= 0:
                            if HB_in_B[h_ci] == 0 or nv_B < HB_nv_B[h_ci]:  # d_l^B = min (Eq. 28)
                                HB_nv_B[h_ci] = nv_B;  HB_in_B[h_ci] = 1
                        else:
                            bi = _find_id(bnH_edg_ids, n_bnH, vid)
                            if bi >= 0:
                                if nv_B < bnH_edg_nvs[bi]: bnH_edg_nvs[bi] = nv_B
                            elif n_bnH < _MAX_FIC:
                                bnH_edg_ids[n_bnH] = vid
                                bnH_edg_nvs[n_bnH] = nv_B
                                n_bnH += 1

                    # Fictive estimator — soil
                    sum_q_nc = 0.0
                    for k in range(n_bfic):
                        q_k, _ = _q_dq_any(b_fic_nvs[k], LAI_r, std_LAI, use_lognormal, sigma_ln)
                        sum_q_nc += q_k / max(1.0 - P_LAI * q_k, EPS)
                    for k in range(n_bnH):
                        q_k, _ = _q_dq_any(bnH_edg_nvs[k], LAI_r, std_LAI, use_lognormal, sigma_ln)
                        sum_q_nc += q_k / max(1.0 - P_LAI * q_k, EPS)

                    d_fic = ref / LAI_Max * (n_H_soil / P_LAI - sum_q_nc)
                    cuda.atomic.add(deriv_fic_out, (n, io), d_fic)
                    cuda.atomic.add(sum_q_out,     (n, io), sum_q_nc / LAI_Max)
                    cuda.atomic.add(nH_out,        (n, io), n_H_soil / (P_LAI * LAI_Max))
                    cuda.atomic.add(brkt_out,        (n, io), n_H_soil / (P_LAI * LAI_Max) - sum_q_nc / LAI_Max)

                    # Edge estimator — soil
                    sum_B = 0.0
                    for k in range(n_bfic):
                        q_k, dq_k = _q_dq_any(b_fic_nvs[k], LAI_r, std_LAI, use_lognormal, sigma_ln)
                        sum_B += P_LAI * dq_k / max(1.0 - P_LAI * q_k, EPS)
                    for k in range(n_bnH):
                        q_k, dq_k = _q_dq_any(bnH_edg_nvs[k], LAI_r, std_LAI, use_lognormal, sigma_ln)
                        sum_B += P_LAI * dq_k / max(1.0 - P_LAI * q_k, EPS)

                    sum_H = 0.0;  sum_HB = 0.0
                    for ci in range(n_H_soil):
                        q_H, dq_H = _q_dq_any(H_nvs[ci], LAI_r, std_LAI, use_lognormal, sigma_ln)
                        if HB_in_B[ci]:
                            q_B, dq_B = _q_dq_any(HB_nv_B[ci], LAI_r, std_LAI, use_lognormal, sigma_ln)
                            denom = q_H - q_B
                            if math.fabs(denom) > EPS:
                                sum_HB += (dq_H - dq_B) / denom
                        else:
                            sum_H += dq_H / max(q_H, EPS)

                    d_edg = ref / n_leaf_norm / P_LAI * (sum_H - sum_B + sum_HB)
                    cuda.atomic.add(deriv_edg_out, (n, io), d_edg)

                # Lambertian soil scatter
                u1b = xoroshiro128p_uniform_float64(rng_states, n)
                u2b = xoroshiro128p_uniform_float64(rng_states, n)
                sr2  = math.sqrt(u1b);  phi2 = 6.28318530 * u2b
                vx   = sr2 * math.cos(phi2)
                vy   = sr2 * math.sin(phi2)
                vz   = math.sqrt(max(0.0, 1.0 - u1b))
                energy *= rsoil
                Px = soil_Px;  Py = soil_Py;  Pz = soil_Pz
                origin = -1;  depth += 1

        # per-path terminal values — written once after the path ends
        nH_path_out[n] = nb.float64(n_H[0])
        nF_path_out[n] = nb.float64(n_ffic[0])
        K_out[n]       = nb.float64(K_local)

    # ─────────────────────────────────────────────────────────────────────────
    # _trace_bwd_kernel — backward BRF only (single observer direction)
    # Traces N photons in the reversed observer direction, NEE toward sun.
    # By Helmholtz reciprocity: BRF_bwd == BRF_fwd within MC noise.
    # ─────────────────────────────────────────────────────────────────────────

    @cuda.jit
    def _trace_bwd_kernel(
        leaf_px, leaf_py, leaf_pz,
        leaf_nx, leaf_ny, leaf_nz, leaf_radius,
        voxel_count, voxel_leaves, vs, mvx, mvy, mvz,
        ox, oy, oz,
        sun_vx, sun_vy, sun_vz,
        rho, tau, rsoil, max_depth, z_bottom,
        P_LAI, global_seed,
        periodic_xy,
        start_x, start_y,
        rng_states,
        lum_col,
    ):
        """One thread = one backward photon for a single observer direction."""
        n = cuda.grid(1)
        N = lum_col.shape[0]
        if n >= N:
            return

        out_idx = cuda.local.array(1, dtype=nb.int32)
        out_Hx  = cuda.local.array(1, dtype=nb.float64)
        out_Hy  = cuda.local.array(1, dtype=nb.float64)
        out_Hz  = cuda.local.array(1, dtype=nb.float64)
        out_nv  = cuda.local.array(1, dtype=nb.float64)

        Px = start_x[n];  Py = start_y[n];  Pz = 0.0
        vx = -ox;  vy = -oy;  vz = -oz          # reversed observer → into canopy
        origin = -1;  energy = 1.0;  depth = 0

        nex = -sun_vx;  ney = -sun_vy;  nez = -sun_vz   # toward sun source
        cos_earth_sun = nez if nez > 1e-6 else 1e-6

        while energy > 1e-5 and depth < max_depth:
            out_idx[0] = -1
            hit = _dda_contact(
                Px, Py, Pz, vx, vy, vz, origin,
                leaf_px, leaf_py, leaf_pz,
                leaf_nx, leaf_ny, leaf_nz,
                leaf_radius, n, global_seed, P_LAI,
                voxel_count, voxel_leaves, vs, mvx, mvy, mvz,
                out_idx, out_Hx, out_Hy, out_Hz, out_nv,
                periodic_xy,
            )

            if hit:
                li   = out_idx[0]
                Hx   = out_Hx[0];  Hy = out_Hy[0];  Hz = out_Hz[0]
                lnxl = leaf_nx[li];  lnyl = leaf_ny[li];  lnzl = leaf_nz[li]
                cos_in  = -(vx*lnxl + vy*lnyl + vz*lnzl)
                cos_out = nex*lnxl + ney*lnyl + nez*lnzl
                f = rho / 3.14159265 if cos_in * cos_out > 0.0 else tau / 3.14159265
                norm_fac = math.fabs(cos_out) / cos_earth_sun

                vis_idx = cuda.local.array(1, dtype=nb.int32)
                vis_Hx  = cuda.local.array(1, dtype=nb.float64)
                vis_Hy  = cuda.local.array(1, dtype=nb.float64)
                vis_Hz  = cuda.local.array(1, dtype=nb.float64)
                vis_nv  = cuda.local.array(1, dtype=nb.float64)
                vis_idx[0] = -1
                blocked = _dda_contact(
                    Hx, Hy, Hz, nex, ney, nez, li,
                    leaf_px, leaf_py, leaf_pz,
                    leaf_nx, leaf_ny, leaf_nz,
                    leaf_radius, n, global_seed, P_LAI,
                    voxel_count, voxel_leaves, vs, mvx, mvy, mvz,
                    vis_idx, vis_Hx, vis_Hy, vis_Hz, vis_nv,
                    periodic_xy,
                )
                if not blocked:
                    cuda.atomic.add(lum_col, n, energy * f * norm_fac)

                u1 = xoroshiro128p_uniform_float64(rng_states, n)
                u2 = xoroshiro128p_uniform_float64(rng_states, n)
                u3 = xoroshiro128p_uniform_float64(rng_states, n)
                sr  = math.sqrt(u1);  phi = 6.28318530 * u2
                sx  = sr * math.cos(phi);  sy = sr * math.sin(phi)
                sz  = math.sqrt(max(0.0, 1.0 - u1))
                cif = vx*lnxl + vy*lnyl + vz*lnzl
                if u3 < rho / (rho + tau):
                    nox = lnxl if cif < 0 else -lnxl
                    noy = lnyl if cif < 0 else -lnyl
                    noz = lnzl if cif < 0 else -lnzl
                else:
                    nox = -lnxl if cif < 0 else lnxl
                    noy = -lnyl if cif < 0 else lnyl
                    noz = -lnzl if cif < 0 else lnzl
                if math.fabs(nox) < 0.9: tx=1.; ty=0.; tz=0.
                else:                    tx=0.; ty=1.; tz=0.
                d2=tx*nox+ty*noy+tz*noz; tx-=d2*nox; ty-=d2*noy; tz-=d2*noz
                tl=math.sqrt(tx*tx+ty*ty+tz*tz); tx/=tl; ty/=tl; tz/=tl
                bx=noy*tz-noz*ty; by=noz*tx-nox*tz; bz=nox*ty-noy*tx
                vx=sx*tx+sy*bx+sz*nox; vy=sx*ty+sy*by+sz*noy; vz=sx*tz+sy*bz+sz*noz
                energy *= (rho + tau)
                Px=Hx; Py=Hy; Pz=Hz; origin=li; depth+=1

            else:
                tz_top = (0.0      - Pz) / vz if vz >  1e-12 else 1e18
                tz_bot = (z_bottom - Pz) / vz if vz < -1e-12 else 1e18
                if tz_top < tz_bot:
                    break
                soil_Px = Px + tz_bot*vx;  soil_Py = Py + tz_bot*vy
                vis_idx2 = cuda.local.array(1, dtype=nb.int32)
                vis_Hx2  = cuda.local.array(1, dtype=nb.float64)
                vis_Hy2  = cuda.local.array(1, dtype=nb.float64)
                vis_Hz2  = cuda.local.array(1, dtype=nb.float64)
                vis_nv2  = cuda.local.array(1, dtype=nb.float64)
                vis_idx2[0] = -1
                blocked2 = _dda_contact(
                    soil_Px, soil_Py, z_bottom+1e-7, nex, ney, nez, -1,
                    leaf_px, leaf_py, leaf_pz, leaf_nx, leaf_ny, leaf_nz,
                    leaf_radius, n, global_seed, P_LAI,
                    voxel_count, voxel_leaves, vs, mvx, mvy, mvz,
                    vis_idx2, vis_Hx2, vis_Hy2, vis_Hz2, vis_nv2,
                    periodic_xy,
                )
                if not blocked2:
                    cuda.atomic.add(lum_col, n, energy * rsoil / 3.14159265)
                u1b = xoroshiro128p_uniform_float64(rng_states, n)
                u2b = xoroshiro128p_uniform_float64(rng_states, n)
                sr2  = math.sqrt(u1b);  phi2 = 6.28318530 * u2b
                vx   = sr2 * math.cos(phi2)
                vy   = sr2 * math.sin(phi2)
                vz   = math.sqrt(max(0.0, 1.0 - u1b))
                energy *= rsoil
                Px = soil_Px;  Py = soil_Py;  Pz = z_bottom + 1e-7
                origin = -1;  depth += 1

    # ─────────────────────────────────────────────────────────────────────────
    # _trace_bwd_deriv_kernel — backward BRF + dBRF/dLAI (single obs dir)
    # Derivative sets H / fwd_fic / fwd_edg track the observer subpath.
    # NEE toward sun plays the role NEE toward observer plays in forward.
    # ─────────────────────────────────────────────────────────────────────────

    @cuda.jit
    def _trace_bwd_deriv_kernel(
        leaf_px, leaf_py, leaf_pz,
        leaf_nx, leaf_ny, leaf_nz, leaf_radius,
        voxel_count, voxel_leaves, vs, mvx, mvy, mvz,
        ox, oy, oz,
        sun_vx, sun_vy, sun_vz,
        rho, tau, rsoil, max_depth, z_bottom,
        global_seed,
        periodic_xy,
        start_x, start_y,
        rng_states,
        lum_col,        # (N,)
        deriv_fic_col,  # (N,)
        deriv_edg_col,  # (N,)
        LAI_r, std_LAI, LAI_Max, P_LAI, n_leaf_norm,
        edge_r_thresh, use_lognormal, sigma_ln,
    ):
        """Backward tracer for one observer direction: BRF + dBRF/dLAI."""
        n = cuda.grid(1)
        N = lum_col.shape[0]
        if n >= N:
            return

        EPS = 1e-12

        H_ids = cuda.local.array(_MAX_C,   dtype=nb.int32)
        H_nvs = cuda.local.array(_MAX_C,   dtype=nb.float64)
        n_H   = cuda.local.array(1,        dtype=nb.int32)
        n_H[0] = 0

        fwd_fic_ids = cuda.local.array(_MAX_FIC, dtype=nb.int32)
        fwd_fic_nvs = cuda.local.array(_MAX_FIC, dtype=nb.float64)
        n_ffic      = cuda.local.array(1,        dtype=nb.int32)
        n_ffic[0]   = 0

        fwd_edg_ids = cuda.local.array(_MAX_FIC, dtype=nb.int32)
        fwd_edg_nvs = cuda.local.array(_MAX_FIC, dtype=nb.float64)
        n_fedg      = cuda.local.array(1,        dtype=nb.int32)
        n_fedg[0]   = 0

        seg_fic_ids = cuda.local.array(_MAX_FIC, dtype=nb.int32)
        seg_fic_nvs = cuda.local.array(_MAX_FIC, dtype=nb.float64)
        seg_fic_ohs = cuda.local.array(_MAX_FIC, dtype=nb.float64)
        n_sfic      = cuda.local.array(1,        dtype=nb.int32)

        seg_edg_ids = cuda.local.array(_MAX_FIC, dtype=nb.int32)
        seg_edg_nvs = cuda.local.array(_MAX_FIC, dtype=nb.float64)
        seg_edg_ohs = cuda.local.array(_MAX_FIC, dtype=nb.float64)
        n_sedg      = cuda.local.array(1,        dtype=nb.int32)

        nee_fic_ids = cuda.local.array(_MAX_FIC, dtype=nb.int32)
        nee_fic_nvs = cuda.local.array(_MAX_FIC, dtype=nb.float64)
        n_nfic      = cuda.local.array(1,        dtype=nb.int32)

        nee_edg_ids = cuda.local.array(_MAX_FIC, dtype=nb.int32)
        nee_edg_nvs = cuda.local.array(_MAX_FIC, dtype=nb.float64)
        n_nedg      = cuda.local.array(1,        dtype=nb.int32)

        b_fic_ids   = cuda.local.array(_MAX_FIC, dtype=nb.int32)
        b_fic_nvs   = cuda.local.array(_MAX_FIC, dtype=nb.float64)
        bnH_edg_ids = cuda.local.array(_MAX_FIC, dtype=nb.int32)
        bnH_edg_nvs = cuda.local.array(_MAX_FIC, dtype=nb.float64)
        HB_nv_B     = cuda.local.array(_MAX_C,   dtype=nb.float64)
        HB_in_B     = cuda.local.array(_MAX_C,   dtype=nb.int32)

        out_idx = cuda.local.array(1, dtype=nb.int32)
        out_Hx  = cuda.local.array(1, dtype=nb.float64)
        out_Hy  = cuda.local.array(1, dtype=nb.float64)
        out_Hz  = cuda.local.array(1, dtype=nb.float64)
        out_nv  = cuda.local.array(1, dtype=nb.float64)

        Px = start_x[n];  Py = start_y[n];  Pz = 0.0
        vx = -ox;  vy = -oy;  vz = -oz
        origin = -1;  energy = 1.0;  depth = 0

        nex = -sun_vx;  ney = -sun_vy;  nez = -sun_vz
        cos_earth_sun = nez if nez > 1e-6 else 1e-6

        while energy > 1e-5 and depth < max_depth:

            # ── 1. Backward DDA along observer subpath ────────────────────
            out_idx[0] = -1
            hit = _dda_fwd_deriv(
                Px, Py, Pz, vx, vy, vz, origin,
                leaf_px, leaf_py, leaf_pz,
                leaf_nx, leaf_ny, leaf_nz,
                leaf_radius, n, global_seed, P_LAI,
                voxel_count, voxel_leaves, vs, mvx, mvy, mvz, periodic_xy,
                out_idx, out_Hx, out_Hy, out_Hz, out_nv,
                seg_fic_ids, seg_fic_nvs, seg_fic_ohs, n_sfic,
                seg_edg_ids, seg_edg_nvs, seg_edg_ohs, n_sedg,
                edge_r_thresh,
                LAI_r, std_LAI, use_lognormal, sigma_ln,
            )

            # ── 2. Merge segment sets into running fwd accumulators ───────
            for k in range(n_sfic[0]):
                vid = seg_fic_ids[k];  nv = seg_fic_nvs[k]
                fi  = _find_id(fwd_fic_ids, n_ffic[0], vid)
                if fi >= 0:
                    if nv < fwd_fic_nvs[fi]: fwd_fic_nvs[fi] = nv
                elif n_ffic[0] < _MAX_FIC:
                    fwd_fic_ids[n_ffic[0]] = vid
                    fwd_fic_nvs[n_ffic[0]] = nv
                    n_ffic[0] += 1

            for k in range(n_sedg[0]):
                vid = seg_edg_ids[k];  nv = seg_edg_nvs[k]
                ei  = _find_id(fwd_edg_ids, n_fedg[0], vid)
                if ei >= 0:
                    if nv < fwd_edg_nvs[ei]: fwd_edg_nvs[ei] = nv
                elif n_fedg[0] < _MAX_FIC:
                    fwd_edg_ids[n_fedg[0]] = vid
                    fwd_edg_nvs[n_fedg[0]] = nv
                    n_fedg[0] += 1

            if hit:

                li   = out_idx[0]
                Hx   = out_Hx[0];  Hy = out_Hy[0];  Hz = out_Hz[0]
                lnxl = leaf_nx[li];  lnyl = leaf_ny[li];  lnzl = leaf_nz[li]
                cos_in = -(vx*lnxl + vy*lnyl + vz*lnzl)

                # ── 3. Update H ───────────────────────────────────────────
                h_exist = _find_id(H_ids, n_H[0], li)
                if h_exist >= 0:
                    if out_nv[0] > H_nvs[h_exist]: H_nvs[h_exist] = out_nv[0]
                elif n_H[0] < _MAX_C:
                    H_ids[n_H[0]] = li
                    H_nvs[n_H[0]] = out_nv[0]
                    n_H[0] += 1
                n_H_now = n_H[0]

                # ── 4. Single NEE toward sun ──────────────────────────────
                cos_out  = nex*lnxl + ney*lnyl + nez*lnzl
                f        = rho / 3.14159265 if cos_in * cos_out > 0.0 else tau / 3.14159265
                norm_fac = math.fabs(cos_out) / cos_earth_sun

                visible = _nee_deriv(
                    Hx, Hy, Hz, nex, ney, nez, li,
                    leaf_px, leaf_py, leaf_pz,
                    leaf_nx, leaf_ny, leaf_nz,
                    leaf_radius, n, global_seed, P_LAI,
                    voxel_count, voxel_leaves, vs, mvx, mvy, mvz, periodic_xy,
                    nee_fic_ids, nee_fic_nvs, n_nfic,
                    nee_edg_ids, nee_edg_nvs, n_nedg,
                    edge_r_thresh,
                    LAI_r, std_LAI, use_lognormal, sigma_ln,
                )

                if visible:
                    ref = energy * f * norm_fac
                    cuda.atomic.add(lum_col, n, ref)

                    # ── 5. Build B\H fic ──────────────────────────────────
                    n_bfic = 0
                    for k in range(n_ffic[0]):
                        b_fic_ids[n_bfic] = fwd_fic_ids[k]
                        b_fic_nvs[n_bfic] = fwd_fic_nvs[k]
                        n_bfic += 1
                    for k in range(n_nfic[0]):
                        vid = nee_fic_ids[k];  nv = nee_fic_nvs[k]
                        fi  = _find_id(b_fic_ids, n_bfic, vid)
                        if fi >= 0:
                            if nv < b_fic_nvs[fi]: b_fic_nvs[fi] = nv
                        elif n_bfic < _MAX_FIC:
                            b_fic_ids[n_bfic] = vid
                            b_fic_nvs[n_bfic] = nv
                            n_bfic += 1

                    # ── H∩B and B\H edge split ────────────────────────────
                    for ci in range(n_H_now):
                        HB_nv_B[ci] = -1.0;  HB_in_B[ci] = 0
                    n_bnH = 0

                    for k in range(n_fedg[0]):
                        vid = fwd_edg_ids[k];  nv_B = fwd_edg_nvs[k]
                        h_ci = _find_id(H_ids, n_H_now, vid)
                        if h_ci >= 0:
                            if HB_in_B[h_ci] == 0 or nv_B < HB_nv_B[h_ci]:  # d_l^B = min (Eq. 28)
                                HB_nv_B[h_ci] = nv_B;  HB_in_B[h_ci] = 1
                        else:
                            bi = _find_id(bnH_edg_ids, n_bnH, vid)
                            if bi >= 0:
                                if nv_B < bnH_edg_nvs[bi]: bnH_edg_nvs[bi] = nv_B
                            elif n_bnH < _MAX_FIC:
                                bnH_edg_ids[n_bnH] = vid
                                bnH_edg_nvs[n_bnH] = nv_B
                                n_bnH += 1

                    for k in range(n_nedg[0]):
                        vid = nee_edg_ids[k];  nv_B = nee_edg_nvs[k]
                        h_ci = _find_id(H_ids, n_H_now, vid)
                        if h_ci >= 0:
                            if HB_in_B[h_ci] == 0 or nv_B < HB_nv_B[h_ci]:  # d_l^B = min (Eq. 28)
                                HB_nv_B[h_ci] = nv_B;  HB_in_B[h_ci] = 1
                        else:
                            bi = _find_id(bnH_edg_ids, n_bnH, vid)
                            if bi >= 0:
                                if nv_B < bnH_edg_nvs[bi]: bnH_edg_nvs[bi] = nv_B
                            elif n_bnH < _MAX_FIC:
                                bnH_edg_ids[n_bnH] = vid
                                bnH_edg_nvs[n_bnH] = nv_B
                                n_bnH += 1

                    # ── 6. Fictive estimator ──────────────────────────────
                    sum_q_nc = 0.0
                    for k in range(n_bfic):
                        q_k, _ = _q_dq_any(b_fic_nvs[k], LAI_r, std_LAI, use_lognormal, sigma_ln)
                        sum_q_nc += q_k / max(1.0 - P_LAI * q_k, EPS)
                    for k in range(n_bnH):
                        q_k, _ = _q_dq_any(bnH_edg_nvs[k], LAI_r, std_LAI, use_lognormal, sigma_ln)
                        sum_q_nc += q_k / max(1.0 - P_LAI * q_k, EPS)

                    d_fic = ref / LAI_Max * (n_H_now / P_LAI - sum_q_nc)
                    cuda.atomic.add(deriv_fic_col, n, d_fic)

                    # ── 7. Edge estimator ─────────────────────────────────
                    sum_B = 0.0
                    for k in range(n_bfic):
                        q_k, dq_k = _q_dq_any(b_fic_nvs[k], LAI_r, std_LAI, use_lognormal, sigma_ln)
                        sum_B += P_LAI * dq_k / max(1.0 - P_LAI * q_k, EPS)
                    for k in range(n_bnH):
                        q_k, dq_k = _q_dq_any(bnH_edg_nvs[k], LAI_r, std_LAI, use_lognormal, sigma_ln)
                        sum_B += P_LAI * dq_k / max(1.0 - P_LAI * q_k, EPS)

                    sum_H = 0.0;  sum_HB = 0.0
                    for ci in range(n_H_now):
                        q_H, dq_H = _q_dq_any(H_nvs[ci], LAI_r, std_LAI, use_lognormal, sigma_ln)
                        if HB_in_B[ci]:
                            q_B, dq_B = _q_dq_any(HB_nv_B[ci], LAI_r, std_LAI, use_lognormal, sigma_ln)
                            denom = q_H - q_B
                            if math.fabs(denom) > EPS:
                                sum_HB += (dq_H - dq_B) / denom
                        else:
                            sum_H += dq_H / max(q_H, EPS)

                    d_edg = ref / n_leaf_norm / P_LAI * (sum_H - sum_B + sum_HB)
                    cuda.atomic.add(deriv_edg_col, n, d_edg)

                # ── 8. Scatter ────────────────────────────────────────────
                u1 = xoroshiro128p_uniform_float64(rng_states, n)
                u2 = xoroshiro128p_uniform_float64(rng_states, n)
                u3 = xoroshiro128p_uniform_float64(rng_states, n)
                sr  = math.sqrt(u1);  phi = 6.28318530 * u2
                sx  = sr * math.cos(phi);  sy = sr * math.sin(phi)
                sz  = math.sqrt(max(0.0, 1.0 - u1))
                cif = vx*lnxl + vy*lnyl + vz*lnzl
                if u3 < rho / (rho + tau):
                    nox = lnxl if cif < 0 else -lnxl
                    noy = lnyl if cif < 0 else -lnyl
                    noz = lnzl if cif < 0 else -lnzl
                else:
                    nox = -lnxl if cif < 0 else lnxl
                    noy = -lnyl if cif < 0 else lnyl
                    noz = -lnzl if cif < 0 else lnzl
                if math.fabs(nox) < 0.9: tx=1.; ty=0.; tz=0.
                else:                    tx=0.; ty=1.; tz=0.
                d2=tx*nox+ty*noy+tz*noz; tx-=d2*nox; ty-=d2*noy; tz-=d2*noz
                tl=math.sqrt(tx*tx+ty*ty+tz*tz); tx/=tl; ty/=tl; tz/=tl
                bx=noy*tz-noz*ty; by=noz*tx-nox*tz; bz=nox*ty-noy*tx
                vx=sx*tx+sy*bx+sz*nox; vy=sx*ty+sy*by+sz*noy; vz=sx*tz+sy*bz+sz*noz
                energy *= (rho + tau)
                Px=Hx; Py=Hy; Pz=Hz; origin=li; depth+=1

            else:   # no leaf hit — check sky / soil

                tz_top = (0.0      - Pz) / vz if vz >  1e-12 else 1e18
                tz_bot = (z_bottom - Pz) / vz if vz < -1e-12 else 1e18
                if tz_top < tz_bot:
                    break

                soil_Px = Px + tz_bot*vx
                soil_Py = Py + tz_bot*vy
                soil_Pz = z_bottom + 1e-7
                n_H_soil = n_H[0]

                visible = _nee_deriv(
                    soil_Px, soil_Py, soil_Pz, nex, ney, nez, -1,
                    leaf_px, leaf_py, leaf_pz,
                    leaf_nx, leaf_ny, leaf_nz,
                    leaf_radius, n, global_seed, P_LAI,
                    voxel_count, voxel_leaves, vs, mvx, mvy, mvz, periodic_xy,
                    nee_fic_ids, nee_fic_nvs, n_nfic,
                    nee_edg_ids, nee_edg_nvs, n_nedg,
                    edge_r_thresh,
                    LAI_r, std_LAI, use_lognormal, sigma_ln,
                )

                if visible:
                    ref = energy * rsoil / 3.14159265
                    cuda.atomic.add(lum_col, n, ref)

                    n_bfic = 0
                    for k in range(n_ffic[0]):
                        b_fic_ids[n_bfic] = fwd_fic_ids[k]
                        b_fic_nvs[n_bfic] = fwd_fic_nvs[k]
                        n_bfic += 1
                    for k in range(n_nfic[0]):
                        vid = nee_fic_ids[k];  nv = nee_fic_nvs[k]
                        fi  = _find_id(b_fic_ids, n_bfic, vid)
                        if fi >= 0:
                            if nv < b_fic_nvs[fi]: b_fic_nvs[fi] = nv
                        elif n_bfic < _MAX_FIC:
                            b_fic_ids[n_bfic] = vid
                            b_fic_nvs[n_bfic] = nv
                            n_bfic += 1

                    for ci in range(n_H_soil):
                        HB_nv_B[ci] = -1.0;  HB_in_B[ci] = 0
                    n_bnH = 0

                    for k in range(n_fedg[0]):
                        vid = fwd_edg_ids[k];  nv_B = fwd_edg_nvs[k]
                        h_ci = _find_id(H_ids, n_H_soil, vid)
                        if h_ci >= 0:
                            if HB_in_B[h_ci] == 0 or nv_B < HB_nv_B[h_ci]:  # d_l^B = min (Eq. 28)
                                HB_nv_B[h_ci] = nv_B;  HB_in_B[h_ci] = 1
                        else:
                            bi = _find_id(bnH_edg_ids, n_bnH, vid)
                            if bi >= 0:
                                if nv_B < bnH_edg_nvs[bi]: bnH_edg_nvs[bi] = nv_B
                            elif n_bnH < _MAX_FIC:
                                bnH_edg_ids[n_bnH] = vid
                                bnH_edg_nvs[n_bnH] = nv_B
                                n_bnH += 1

                    for k in range(n_nedg[0]):
                        vid = nee_edg_ids[k];  nv_B = nee_edg_nvs[k]
                        h_ci = _find_id(H_ids, n_H_soil, vid)
                        if h_ci >= 0:
                            if HB_in_B[h_ci] == 0 or nv_B < HB_nv_B[h_ci]:  # d_l^B = min (Eq. 28)
                                HB_nv_B[h_ci] = nv_B;  HB_in_B[h_ci] = 1
                        else:
                            bi = _find_id(bnH_edg_ids, n_bnH, vid)
                            if bi >= 0:
                                if nv_B < bnH_edg_nvs[bi]: bnH_edg_nvs[bi] = nv_B
                            elif n_bnH < _MAX_FIC:
                                bnH_edg_ids[n_bnH] = vid
                                bnH_edg_nvs[n_bnH] = nv_B
                                n_bnH += 1

                    sum_q_nc = 0.0
                    for k in range(n_bfic):
                        q_k, _ = _q_dq_any(b_fic_nvs[k], LAI_r, std_LAI, use_lognormal, sigma_ln)
                        sum_q_nc += q_k / max(1.0 - P_LAI * q_k, EPS)
                    for k in range(n_bnH):
                        q_k, _ = _q_dq_any(bnH_edg_nvs[k], LAI_r, std_LAI, use_lognormal, sigma_ln)
                        sum_q_nc += q_k / max(1.0 - P_LAI * q_k, EPS)

                    d_fic = ref / LAI_Max * (n_H_soil / P_LAI - sum_q_nc)
                    cuda.atomic.add(deriv_fic_col, n, d_fic)

                    sum_B = 0.0
                    for k in range(n_bfic):
                        q_k, dq_k = _q_dq_any(b_fic_nvs[k], LAI_r, std_LAI, use_lognormal, sigma_ln)
                        sum_B += P_LAI * dq_k / max(1.0 - P_LAI * q_k, EPS)
                    for k in range(n_bnH):
                        q_k, dq_k = _q_dq_any(bnH_edg_nvs[k], LAI_r, std_LAI, use_lognormal, sigma_ln)
                        sum_B += P_LAI * dq_k / max(1.0 - P_LAI * q_k, EPS)

                    sum_H = 0.0;  sum_HB = 0.0
                    for ci in range(n_H_soil):
                        q_H, dq_H = _q_dq_any(H_nvs[ci], LAI_r, std_LAI, use_lognormal, sigma_ln)
                        if HB_in_B[ci]:
                            q_B, dq_B = _q_dq_any(HB_nv_B[ci], LAI_r, std_LAI, use_lognormal, sigma_ln)
                            denom = q_H - q_B
                            if math.fabs(denom) > EPS:
                                sum_HB += (dq_H - dq_B) / denom
                        else:
                            sum_H += dq_H / max(q_H, EPS)

                    d_edg = ref / n_leaf_norm / P_LAI * (sum_H - sum_B + sum_HB)
                    cuda.atomic.add(deriv_edg_col, n, d_edg)

                u1b = xoroshiro128p_uniform_float64(rng_states, n)
                u2b = xoroshiro128p_uniform_float64(rng_states, n)
                sr2  = math.sqrt(u1b);  phi2 = 6.28318530 * u2b
                vx   = sr2 * math.cos(phi2)
                vy   = sr2 * math.sin(phi2)
                vz   = math.sqrt(max(0.0, 1.0 - u1b))
                energy *= rsoil
                Px = soil_Px;  Py = soil_Py;  Pz = soil_Pz
                origin = -1;  depth += 1


    # _trace_bwd_deriv_species_kernel — backward BRF + per-species dBRF/dLAI
    # Same as _trace_bwd_deriv_kernel but steps 6 & 7 produce _N_SP outputs.
    # deriv_fic_col / deriv_edg_col have shape (N * _N_SP,):
    #   element  n * _N_SP + s  is photon n's contribution to species s.
    # ─────────────────────────────────────────────────────────────────────────

    @cuda.jit
    def _trace_bwd_deriv_species_kernel(
        leaf_px, leaf_py, leaf_pz,
        leaf_nx, leaf_ny, leaf_nz, leaf_radius,
        voxel_count, voxel_leaves, vs, mvx, mvy, mvz,
        ox, oy, oz,
        sun_vx, sun_vy, sun_vz,
        rho, tau, rsoil, max_depth, z_bottom,
        global_seed,
        periodic_xy,
        start_x, start_y,
        rng_states,
        lum_col,            # (N,)
        deriv_fic_col,      # (N * _N_SP,)
        deriv_edg_col,      # (N * _N_SP,)
        deriv_rho_col,      # (N * _N_SP,)  per-species rho score-function
        LAI_r, std_LAI, LAI_Max, P_LAI, n_leaf_norm,
        edge_r_thresh, use_lognormal, sigma_ln,
        leaf_species,       # (n_leaves,)  int32  0.._N_SP-1
        LAI_Max_sp,         # (_N_SP,)     float64  per-species
        n_leaf_norm_sp,     # (_N_SP,)     float64  per-species
        P_LAI_sp,           # (_N_SP,)     float64  per-species activation prob
        rho_sp_arr,         # (_N_SP,)     float64  per-species leaf reflectance rho_s
        tau_sp_arr,         # (_N_SP,)     float64  per-species leaf transmittance tau_s
        species_shadow,     # int  1: the shadow ray uses P_LAI_sp too
    ):
        """Backward tracer: BRF + per-species dBRF/dLAI + dBRF/drho.
        P_LAI_sp[s] controls the activation probability for species s leaves.
        Pass np.full(_N_SP, P_LAI) for the standard uniform-P case.
        rho_sp_arr[s], tau_sp_arr[s]: leaf reflectance / transmittance of species s. The rho score
        (K_sp) assumes tau_s / rho_s is held fixed when rho_s varies: every leaf factor (rho_s or
        tau_s, at a bounce or at the sun connection) is then proportional to rho_s, and the
        reflect/transmit choice rho_s/(rho_s+tau_s) does not depend on rho_s.
        species_shadow = 1 makes the sun-visibility test species-aware as well; 0 keeps
        the historical behaviour, where the shadow ray blocks on the scalar P_LAI."""
        n = cuda.grid(1)
        N = lum_col.shape[0]
        if n >= N:
            return

        EPS = 1e-12

        H_ids = cuda.local.array(_MAX_C,   dtype=nb.int32)
        H_nvs = cuda.local.array(_MAX_C,   dtype=nb.float64)
        n_H   = cuda.local.array(1,        dtype=nb.int32)
        n_H[0] = 0

        fwd_fic_ids = cuda.local.array(_MAX_FIC, dtype=nb.int32)
        fwd_fic_nvs = cuda.local.array(_MAX_FIC, dtype=nb.float64)
        n_ffic      = cuda.local.array(1,        dtype=nb.int32)
        n_ffic[0]   = 0

        fwd_edg_ids = cuda.local.array(_MAX_FIC, dtype=nb.int32)
        fwd_edg_nvs = cuda.local.array(_MAX_FIC, dtype=nb.float64)
        n_fedg      = cuda.local.array(1,        dtype=nb.int32)
        n_fedg[0]   = 0

        seg_fic_ids = cuda.local.array(_MAX_FIC, dtype=nb.int32)
        seg_fic_nvs = cuda.local.array(_MAX_FIC, dtype=nb.float64)
        seg_fic_ohs = cuda.local.array(_MAX_FIC, dtype=nb.float64)
        n_sfic      = cuda.local.array(1,        dtype=nb.int32)

        seg_edg_ids = cuda.local.array(_MAX_FIC, dtype=nb.int32)
        seg_edg_nvs = cuda.local.array(_MAX_FIC, dtype=nb.float64)
        seg_edg_ohs = cuda.local.array(_MAX_FIC, dtype=nb.float64)
        n_sedg      = cuda.local.array(1,        dtype=nb.int32)

        nee_fic_ids = cuda.local.array(_MAX_FIC, dtype=nb.int32)
        nee_fic_nvs = cuda.local.array(_MAX_FIC, dtype=nb.float64)
        n_nfic      = cuda.local.array(1,        dtype=nb.int32)

        nee_edg_ids = cuda.local.array(_MAX_FIC, dtype=nb.int32)
        nee_edg_nvs = cuda.local.array(_MAX_FIC, dtype=nb.float64)
        n_nedg      = cuda.local.array(1,        dtype=nb.int32)

        b_fic_ids   = cuda.local.array(_MAX_FIC, dtype=nb.int32)
        b_fic_nvs   = cuda.local.array(_MAX_FIC, dtype=nb.float64)
        bnH_edg_ids = cuda.local.array(_MAX_FIC, dtype=nb.int32)
        bnH_edg_nvs = cuda.local.array(_MAX_FIC, dtype=nb.float64)
        HB_nv_B     = cuda.local.array(_MAX_C,   dtype=nb.float64)
        HB_in_B     = cuda.local.array(_MAX_C,   dtype=nb.int32)

        out_idx = cuda.local.array(1, dtype=nb.int32)
        out_Hx  = cuda.local.array(1, dtype=nb.float64)
        out_Hy  = cuda.local.array(1, dtype=nb.float64)
        out_Hz  = cuda.local.array(1, dtype=nb.float64)
        out_nv  = cuda.local.array(1, dtype=nb.float64)

        # per-species accumulators (reused each scoring event)
        n_H_sp    = cuda.local.array(_N_SP, dtype=nb.int32)
        sum_q_sp  = cuda.local.array(_N_SP, dtype=nb.float64)
        sum_B_sp  = cuda.local.array(_N_SP, dtype=nb.float64)
        sum_H_sp  = cuda.local.array(_N_SP, dtype=nb.float64)
        sum_HB_sp = cuda.local.array(_N_SP, dtype=nb.float64)

        # K_sp[s] = total leaf-scatter count from species s along path so far.
        # Incremented at every energy *= (rho+tau) for a species-s leaf.
        # Used for the rho score-function estimator: dBRF/drho_s = E[K_s*lum]/rho.
        K_sp = cuda.local.array(_N_SP, dtype=nb.int32)
        for s in range(_N_SP):
            K_sp[s] = 0

        Px = start_x[n];  Py = start_y[n];  Pz = 0.0
        vx = -ox;  vy = -oy;  vz = -oz
        origin = -1;  energy = 1.0;  depth = 0

        nex = -sun_vx;  ney = -sun_vy;  nez = -sun_vz
        cos_earth_sun = nez if nez > 1e-6 else 1e-6

        while energy > 1e-5 and depth < max_depth:

            # ── 1. Backward DDA ───────────────────────────────────────────
            out_idx[0] = -1
            hit = _dda_fwd_deriv_sp(
                Px, Py, Pz, vx, vy, vz, origin,
                leaf_px, leaf_py, leaf_pz,
                leaf_nx, leaf_ny, leaf_nz,
                leaf_radius, n, global_seed,
                voxel_count, voxel_leaves, vs, mvx, mvy, mvz, periodic_xy,
                out_idx, out_Hx, out_Hy, out_Hz, out_nv,
                seg_fic_ids, seg_fic_nvs, seg_fic_ohs, n_sfic,
                seg_edg_ids, seg_edg_nvs, seg_edg_ohs, n_sedg,
                edge_r_thresh,
                LAI_r, std_LAI, use_lognormal, sigma_ln,
                leaf_species, P_LAI_sp,
            )

            # ── 2. Merge segment sets into fwd accumulators ───────────────
            for k in range(n_sfic[0]):
                vid = seg_fic_ids[k];  nv = seg_fic_nvs[k]
                fi  = _find_id(fwd_fic_ids, n_ffic[0], vid)
                if fi >= 0:
                    if nv < fwd_fic_nvs[fi]: fwd_fic_nvs[fi] = nv
                elif n_ffic[0] < _MAX_FIC:
                    fwd_fic_ids[n_ffic[0]] = vid
                    fwd_fic_nvs[n_ffic[0]] = nv
                    n_ffic[0] += 1

            for k in range(n_sedg[0]):
                vid = seg_edg_ids[k];  nv = seg_edg_nvs[k]
                ei  = _find_id(fwd_edg_ids, n_fedg[0], vid)
                if ei >= 0:
                    if nv < fwd_edg_nvs[ei]: fwd_edg_nvs[ei] = nv
                elif n_fedg[0] < _MAX_FIC:
                    fwd_edg_ids[n_fedg[0]] = vid
                    fwd_edg_nvs[n_fedg[0]] = nv
                    n_fedg[0] += 1

            if hit:

                li   = out_idx[0]
                sp_li = leaf_species[li]     # species of THIS leaf (was set only when sunlit)
                Hx   = out_Hx[0];  Hy = out_Hy[0];  Hz = out_Hz[0]
                lnxl = leaf_nx[li];  lnyl = leaf_ny[li];  lnzl = leaf_nz[li]
                cos_in = -(vx*lnxl + vy*lnyl + vz*lnzl)

                # ── 3. Update H ───────────────────────────────────────────
                h_exist = _find_id(H_ids, n_H[0], li)
                if h_exist >= 0:
                    if out_nv[0] > H_nvs[h_exist]: H_nvs[h_exist] = out_nv[0]
                elif n_H[0] < _MAX_C:
                    H_ids[n_H[0]] = li
                    H_nvs[n_H[0]] = out_nv[0]
                    n_H[0] += 1
                n_H_now = n_H[0]

                # ── 4. NEE toward sun ─────────────────────────────────────
                # viewer side = sign(cos_in), sun side = sign(cos_out):
                # same side -> reflection (rho_s), opposite sides -> transmission (tau_s)
                cos_out  = nex*lnxl + ney*lnyl + nez*lnzl
                if cos_in * cos_out > 0.0:
                    f = rho_sp_arr[sp_li] / 3.14159265
                else:
                    f = tau_sp_arr[sp_li] / 3.14159265
                norm_fac = math.fabs(cos_out) / cos_earth_sun

                visible = _nee_deriv_sp(
                    Hx, Hy, Hz, nex, ney, nez, li,
                    leaf_px, leaf_py, leaf_pz,
                    leaf_nx, leaf_ny, leaf_nz,
                    leaf_radius, n, global_seed, P_LAI,
                    leaf_species, P_LAI_sp, species_shadow,
                    voxel_count, voxel_leaves, vs, mvx, mvy, mvz, periodic_xy,
                    nee_fic_ids, nee_fic_nvs, n_nfic,
                    nee_edg_ids, nee_edg_nvs, n_nedg,
                    edge_r_thresh,
                    LAI_r, std_LAI, use_lognormal, sigma_ln,
                )

                if visible:
                    ref = energy * f * norm_fac
                    cuda.atomic.add(lum_col, n, ref)

                    # ── 5. Build B\H fic ──────────────────────────────────
                    n_bfic = 0
                    for k in range(n_ffic[0]):
                        b_fic_ids[n_bfic] = fwd_fic_ids[k]
                        b_fic_nvs[n_bfic] = fwd_fic_nvs[k]
                        n_bfic += 1
                    for k in range(n_nfic[0]):
                        vid = nee_fic_ids[k];  nv = nee_fic_nvs[k]
                        fi  = _find_id(b_fic_ids, n_bfic, vid)
                        if fi >= 0:
                            if nv < b_fic_nvs[fi]: b_fic_nvs[fi] = nv
                        elif n_bfic < _MAX_FIC:
                            b_fic_ids[n_bfic] = vid
                            b_fic_nvs[n_bfic] = nv
                            n_bfic += 1

                    # H∩B and B\H edge split
                    for ci in range(n_H_now):
                        HB_nv_B[ci] = -1.0;  HB_in_B[ci] = 0
                    n_bnH = 0

                    for k in range(n_fedg[0]):
                        vid = fwd_edg_ids[k];  nv_B = fwd_edg_nvs[k]
                        h_ci = _find_id(H_ids, n_H_now, vid)
                        if h_ci >= 0:
                            if HB_in_B[h_ci] == 0 or nv_B < HB_nv_B[h_ci]:  # d_l^B = min (Eq. 28)
                                HB_nv_B[h_ci] = nv_B;  HB_in_B[h_ci] = 1
                        else:
                            bi = _find_id(bnH_edg_ids, n_bnH, vid)
                            if bi >= 0:
                                if nv_B < bnH_edg_nvs[bi]: bnH_edg_nvs[bi] = nv_B
                            elif n_bnH < _MAX_FIC:
                                bnH_edg_ids[n_bnH] = vid
                                bnH_edg_nvs[n_bnH] = nv_B
                                n_bnH += 1

                    for k in range(n_nedg[0]):
                        vid = nee_edg_ids[k];  nv_B = nee_edg_nvs[k]
                        h_ci = _find_id(H_ids, n_H_now, vid)
                        if h_ci >= 0:
                            if HB_in_B[h_ci] == 0 or nv_B < HB_nv_B[h_ci]:  # d_l^B = min (Eq. 28)
                                HB_nv_B[h_ci] = nv_B;  HB_in_B[h_ci] = 1
                        else:
                            bi = _find_id(bnH_edg_ids, n_bnH, vid)
                            if bi >= 0:
                                if nv_B < bnH_edg_nvs[bi]: bnH_edg_nvs[bi] = nv_B
                            elif n_bnH < _MAX_FIC:
                                bnH_edg_ids[n_bnH] = vid
                                bnH_edg_nvs[n_bnH] = nv_B
                                n_bnH += 1

                    # ── 6 & 7. Per-species fictive + edge estimators ──────
                    for s in range(_N_SP):
                        n_H_sp[s]    = 0
                        sum_q_sp[s]  = 0.0
                        sum_B_sp[s]  = 0.0
                        sum_H_sp[s]  = 0.0
                        sum_HB_sp[s] = 0.0

                    for ci in range(n_H_now):
                        sp = leaf_species[H_ids[ci]]
                        n_H_sp[sp] += 1

                    for k in range(n_bfic):
                        sp = leaf_species[b_fic_ids[k]]
                        q_k, dq_k = _q_dq_any(b_fic_nvs[k], LAI_r, std_LAI, use_lognormal, sigma_ln)
                        denom = max(1.0 - P_LAI_sp[sp] * q_k, EPS)
                        sum_q_sp[sp] += q_k / denom
                        sum_B_sp[sp] += P_LAI_sp[sp] * dq_k / denom

                    for k in range(n_bnH):
                        sp = leaf_species[bnH_edg_ids[k]]
                        q_k, dq_k = _q_dq_any(bnH_edg_nvs[k], LAI_r, std_LAI, use_lognormal, sigma_ln)
                        denom = max(1.0 - P_LAI_sp[sp] * q_k, EPS)
                        sum_q_sp[sp] += q_k / denom
                        sum_B_sp[sp] += P_LAI_sp[sp] * dq_k / denom

                    for ci in range(n_H_now):
                        sp = leaf_species[H_ids[ci]]
                        q_H, dq_H = _q_dq_any(H_nvs[ci], LAI_r, std_LAI, use_lognormal, sigma_ln)
                        if HB_in_B[ci]:
                            q_B, dq_B = _q_dq_any(HB_nv_B[ci], LAI_r, std_LAI, use_lognormal, sigma_ln)
                            denom = q_H - q_B
                            if math.fabs(denom) > EPS:
                                sum_HB_sp[sp] += (dq_H - dq_B) / denom
                        else:
                            sum_H_sp[sp] += dq_H / max(q_H, EPS)

                    # rho score-function: dBRF/drho_s = E[(K_s + I[s=s(li)]) * ref] / rho_s
                    # K_sp[s] counts scatters from species s BEFORE this bounce.
                    # The +1 for s==sp_li accounts for d(f)/drho_s at the current leaf
                    # (f = rho_sp_arr[sp_li]/pi; +1 because rho_s enters f directly).
                    for s in range(_N_SP):
                        k_eff = nb.float64(K_sp[s])
                        if s == sp_li:
                            k_eff += 1.0
                        cuda.atomic.add(deriv_rho_col, n * _N_SP + s, k_eff * ref)

                    for s in range(_N_SP):
                        if LAI_Max_sp[s] > 1e-12:
                            d_fic_s = ref / LAI_Max_sp[s] * (n_H_sp[s] / P_LAI_sp[s] - sum_q_sp[s])
                            cuda.atomic.add(deriv_fic_col, n * _N_SP + s, d_fic_s)
                        if n_leaf_norm_sp[s] > 1e-30:
                            d_edg_s = ref / n_leaf_norm_sp[s] / P_LAI_sp[s] * (
                                sum_H_sp[s] - sum_B_sp[s] + sum_HB_sp[s])
                            cuda.atomic.add(deriv_edg_col, n * _N_SP + s, d_edg_s)

                # ── 8. Scatter ────────────────────────────────────────────
                u1 = xoroshiro128p_uniform_float64(rng_states, n)
                u2 = xoroshiro128p_uniform_float64(rng_states, n)
                u3 = xoroshiro128p_uniform_float64(rng_states, n)
                sr  = math.sqrt(u1);  phi = 6.28318530 * u2
                sx  = sr * math.cos(phi);  sy = sr * math.sin(phi)
                sz  = math.sqrt(max(0.0, 1.0 - u1))
                cif = vx*lnxl + vy*lnyl + vz*lnzl
                if u3 < rho_sp_arr[sp_li] / (rho_sp_arr[sp_li] + tau_sp_arr[sp_li]):
                    nox = lnxl if cif < 0 else -lnxl
                    noy = lnyl if cif < 0 else -lnyl
                    noz = lnzl if cif < 0 else -lnzl
                else:
                    nox = -lnxl if cif < 0 else lnxl
                    noy = -lnyl if cif < 0 else lnyl
                    noz = -lnzl if cif < 0 else lnzl
                if math.fabs(nox) < 0.9: tx=1.; ty=0.; tz=0.
                else:                    tx=0.; ty=1.; tz=0.
                d2=tx*nox+ty*noy+tz*noz; tx-=d2*nox; ty-=d2*noy; tz-=d2*noz
                tl=math.sqrt(tx*tx+ty*ty+tz*tz); tx/=tl; ty/=tl; tz/=tl
                bx=noy*tz-noz*ty; by=noz*tx-nox*tz; bz=nox*ty-noy*tx
                vx=sx*tx+sy*bx+sz*nox; vy=sx*ty+sy*by+sz*noy; vz=sx*tz+sy*bz+sz*noz
                energy *= rho_sp_arr[sp_li] + tau_sp_arr[sp_li]
                K_sp[leaf_species[li]] += 1
                Px=Hx; Py=Hy; Pz=Hz; origin=li; depth+=1

            else:   # no leaf hit — soil

                tz_top = (0.0      - Pz) / vz if vz >  1e-12 else 1e18
                tz_bot = (z_bottom - Pz) / vz if vz < -1e-12 else 1e18
                if tz_top < tz_bot:
                    break

                soil_Px = Px + tz_bot*vx
                soil_Py = Py + tz_bot*vy
                soil_Pz = z_bottom + 1e-7
                n_H_soil = n_H[0]

                visible = _nee_deriv_sp(
                    soil_Px, soil_Py, soil_Pz, nex, ney, nez, -1,
                    leaf_px, leaf_py, leaf_pz,
                    leaf_nx, leaf_ny, leaf_nz,
                    leaf_radius, n, global_seed, P_LAI,
                    leaf_species, P_LAI_sp, species_shadow,
                    voxel_count, voxel_leaves, vs, mvx, mvy, mvz, periodic_xy,
                    nee_fic_ids, nee_fic_nvs, n_nfic,
                    nee_edg_ids, nee_edg_nvs, n_nedg,
                    edge_r_thresh,
                    LAI_r, std_LAI, use_lognormal, sigma_ln,
                )

                if visible:
                    ref = energy * rsoil / 3.14159265
                    cuda.atomic.add(lum_col, n, ref)

                    n_bfic = 0
                    for k in range(n_ffic[0]):
                        b_fic_ids[n_bfic] = fwd_fic_ids[k]
                        b_fic_nvs[n_bfic] = fwd_fic_nvs[k]
                        n_bfic += 1
                    for k in range(n_nfic[0]):
                        vid = nee_fic_ids[k];  nv = nee_fic_nvs[k]
                        fi  = _find_id(b_fic_ids, n_bfic, vid)
                        if fi >= 0:
                            if nv < b_fic_nvs[fi]: b_fic_nvs[fi] = nv
                        elif n_bfic < _MAX_FIC:
                            b_fic_ids[n_bfic] = vid
                            b_fic_nvs[n_bfic] = nv
                            n_bfic += 1

                    for ci in range(n_H_soil):
                        HB_nv_B[ci] = -1.0;  HB_in_B[ci] = 0
                    n_bnH = 0

                    for k in range(n_fedg[0]):
                        vid = fwd_edg_ids[k];  nv_B = fwd_edg_nvs[k]
                        h_ci = _find_id(H_ids, n_H_soil, vid)
                        if h_ci >= 0:
                            if HB_in_B[h_ci] == 0 or nv_B < HB_nv_B[h_ci]:  # d_l^B = min (Eq. 28)
                                HB_nv_B[h_ci] = nv_B;  HB_in_B[h_ci] = 1
                        else:
                            bi = _find_id(bnH_edg_ids, n_bnH, vid)
                            if bi >= 0:
                                if nv_B < bnH_edg_nvs[bi]: bnH_edg_nvs[bi] = nv_B
                            elif n_bnH < _MAX_FIC:
                                bnH_edg_ids[n_bnH] = vid
                                bnH_edg_nvs[n_bnH] = nv_B
                                n_bnH += 1

                    for k in range(n_nedg[0]):
                        vid = nee_edg_ids[k];  nv_B = nee_edg_nvs[k]
                        h_ci = _find_id(H_ids, n_H_soil, vid)
                        if h_ci >= 0:
                            if HB_in_B[h_ci] == 0 or nv_B < HB_nv_B[h_ci]:  # d_l^B = min (Eq. 28)
                                HB_nv_B[h_ci] = nv_B;  HB_in_B[h_ci] = 1
                        else:
                            bi = _find_id(bnH_edg_ids, n_bnH, vid)
                            if bi >= 0:
                                if nv_B < bnH_edg_nvs[bi]: bnH_edg_nvs[bi] = nv_B
                            elif n_bnH < _MAX_FIC:
                                bnH_edg_ids[n_bnH] = vid
                                bnH_edg_nvs[n_bnH] = nv_B
                                n_bnH += 1

                    # ── Per-species fictive + edge (soil bounce) ──────────
                    for s in range(_N_SP):
                        n_H_sp[s]    = 0
                        sum_q_sp[s]  = 0.0
                        sum_B_sp[s]  = 0.0
                        sum_H_sp[s]  = 0.0
                        sum_HB_sp[s] = 0.0

                    for ci in range(n_H_soil):
                        sp = leaf_species[H_ids[ci]]
                        n_H_sp[sp] += 1

                    for k in range(n_bfic):
                        sp = leaf_species[b_fic_ids[k]]
                        q_k, dq_k = _q_dq_any(b_fic_nvs[k], LAI_r, std_LAI, use_lognormal, sigma_ln)
                        denom = max(1.0 - P_LAI_sp[sp] * q_k, EPS)
                        sum_q_sp[sp] += q_k / denom
                        sum_B_sp[sp] += P_LAI_sp[sp] * dq_k / denom

                    for k in range(n_bnH):
                        sp = leaf_species[bnH_edg_ids[k]]
                        q_k, dq_k = _q_dq_any(bnH_edg_nvs[k], LAI_r, std_LAI, use_lognormal, sigma_ln)
                        denom = max(1.0 - P_LAI_sp[sp] * q_k, EPS)
                        sum_q_sp[sp] += q_k / denom
                        sum_B_sp[sp] += P_LAI_sp[sp] * dq_k / denom

                    for ci in range(n_H_soil):
                        sp = leaf_species[H_ids[ci]]
                        q_H, dq_H = _q_dq_any(H_nvs[ci], LAI_r, std_LAI, use_lognormal, sigma_ln)
                        if HB_in_B[ci]:
                            q_B, dq_B = _q_dq_any(HB_nv_B[ci], LAI_r, std_LAI, use_lognormal, sigma_ln)
                            denom = q_H - q_B
                            if math.fabs(denom) > EPS:
                                sum_HB_sp[sp] += (dq_H - dq_B) / denom
                        else:
                            sum_H_sp[sp] += dq_H / max(q_H, EPS)

                    for s in range(_N_SP):
                        if LAI_Max_sp[s] > 1e-12:
                            d_fic_s = ref / LAI_Max_sp[s] * (n_H_sp[s] / P_LAI_sp[s] - sum_q_sp[s])
                            cuda.atomic.add(deriv_fic_col, n * _N_SP + s, d_fic_s)
                        if n_leaf_norm_sp[s] > 1e-30:
                            d_edg_s = ref / n_leaf_norm_sp[s] / P_LAI_sp[s] * (
                                sum_H_sp[s] - sum_B_sp[s] + sum_HB_sp[s])
                            cuda.atomic.add(deriv_edg_col, n * _N_SP + s, d_edg_s)
                        # Soil bounce: f = rsoil/pi, no direct rho contribution.
                        # Only K_sp[s] accumulated leaf scatters matter.
                        cuda.atomic.add(deriv_rho_col, n * _N_SP + s,
                                        nb.float64(K_sp[s]) * ref)

                u1b = xoroshiro128p_uniform_float64(rng_states, n)
                u2b = xoroshiro128p_uniform_float64(rng_states, n)
                sr2  = math.sqrt(u1b);  phi2 = 6.28318530 * u2b
                vx   = sr2 * math.cos(phi2)
                vy   = sr2 * math.sin(phi2)
                vz   = math.sqrt(max(0.0, 1.0 - u1b))
                energy *= rsoil
                Px = soil_Px;  Py = soil_Py;  Pz = soil_Pz
                origin = -1;  depth += 1


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers: upload scene / obs geometry to device once per call
# ─────────────────────────────────────────────────────────────────────────────

def _upload_scene(scene):
    """Transfer read-only scene arrays to GPU.  Returns dict of device arrays."""
    # Safety net: a grid that stops above the soil breaks rays leaving the soil (hot spot).
    # Extends scene.max_voxel IN PLACE, so callers holding `mv = scene.max_voxel` see it.
    from canopi.scene import ensure_soil_in_grid
    ensure_soil_in_grid(scene)
    return dict(
        lpx = cuda.to_device(scene.leaf_pos[:,0].astype(np.float64)),
        lpy = cuda.to_device(scene.leaf_pos[:,1].astype(np.float64)),
        lpz = cuda.to_device(scene.leaf_pos[:,2].astype(np.float64)),
        lnx = cuda.to_device(scene.leaf_normal[:,0].astype(np.float64)),
        lny = cuda.to_device(scene.leaf_normal[:,1].astype(np.float64)),
        lnz = cuda.to_device(scene.leaf_normal[:,2].astype(np.float64)),
        lr  = cuda.to_device(scene.leaf_radius.astype(np.float64)),
        vc  = cuda.to_device(scene.voxel_count.astype(np.int32)),
        vl  = cuda.to_device(scene.voxel_leaves.astype(np.int32)),
    )


def _upload_obs(obs_dirs):
    """Transfer observation direction arrays to GPU."""
    return dict(
        ovx = cuda.to_device(obs_dirs[:,0].astype(np.float64)),
        ovy = cuda.to_device(obs_dirs[:,1].astype(np.float64)),
        ovz = cuda.to_device(obs_dirs[:,2].astype(np.float64)),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public API — BRF only  (original interface, unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def compute_brf_gpu(scene, obs_dirs, vec_sun, N,
                    max_depth=99,
                    threads_per_block=128,
                    seed=42,
                    start_xy_min_frac=0.0,
                    start_xy_max_frac=1.0):
    """
    GPU BRF path tracer — returns raw luminance array (N, n_obs).

        BRF = π * lum.sum(axis=0) / N

    Parameters
    ----------
    scene             : Scene        from build_scene()
    obs_dirs          : (n_obs, 3)   upward sensor directions (z > 0)
    vec_sun           : (3,)         sun direction (z < 0)
    N                 : int          number of photons
    max_depth         : int          max scattering bounces (from cfg.max_depth)
    threads_per_block : int          CUDA launch parameter (power of 2)
    seed              : int

    Returns
    -------
    lum : (N, n_obs) float64   raw per-photon luminance

    Memory
    ------
    Leaf Bernoulli draws (P_LAI) are generated ON-DEVICE via a hash function.
    The old (N × n_leaf) int8 array is gone — memory is now O(n_leaf) only.
    """
    if not GPU_AVAILABLE:
        raise RuntimeError('No CUDA GPU — use the CPU renderer.')

    rng_np  = np.random.default_rng(seed)
    n_obs   = len(obs_dirs)
    mv      = scene.max_voxel
    vs      = float(scene.voxel_size)
    periodic_xy = int(getattr(scene, "periodic_xy", 1))

    start_xy_min_frac = float(np.clip(start_xy_min_frac, 0.0, 1.0))
    start_xy_max_frac = float(np.clip(start_xy_max_frac, 0.0, 1.0))
    if start_xy_max_frac < start_xy_min_frac:
        start_xy_min_frac, start_xy_max_frac = start_xy_max_frac, start_xy_min_frac
    x_min = start_xy_min_frac * float(scene.pos_max[0])
    x_max = start_xy_max_frac * float(scene.pos_max[0])
    y_min = start_xy_min_frac * float(scene.pos_max[1])
    y_max = start_xy_max_frac * float(scene.pos_max[1])
    start_x = rng_np.uniform(x_min, x_max, N).astype(np.float64)
    start_y = rng_np.uniform(y_min, y_max, N).astype(np.float64)

    g = _upload_scene(scene)
    o = _upload_obs(obs_dirs)
    d_sx  = cuda.to_device(start_x)
    d_sy  = cuda.to_device(start_y)
    d_lum = cuda.to_device(np.zeros((N, n_obs), dtype=np.float64))
    rng_states = create_xoroshiro128p_states(N, seed=seed)

    blocks = (N + threads_per_block - 1) // threads_per_block
    _trace_kernel[blocks, threads_per_block](
        g['lpx'], g['lpy'], g['lpz'],
        g['lnx'], g['lny'], g['lnz'], g['lr'],
        g['vc'],  g['vl'],
        vs, int(mv[0]), int(mv[1]), int(mv[2]),
        o['ovx'], o['ovy'], o['ovz'],
        float(vec_sun[0]), float(vec_sun[1]), float(vec_sun[2]),
        float(scene.rho), float(scene.tau), float(scene.rsoil),
        int(max_depth), float(scene.pos_max[2]),
        float(scene.P_LAI), int(seed),
        periodic_xy,
        d_sx, d_sy, rng_states, d_lum,
    )
    cuda.synchronize()
    return d_lum.copy_to_host()


def brf_gpu(scene, obs_dirs, vec_sun, N, max_depth=99, seed=42):
    """
    Convenience wrapper: returns BRF (n_obs,).
    Leaf Bernoulli draws use the GPU hash (no CPU pre-allocation needed).
    """
    lum = compute_brf_gpu(scene, obs_dirs, vec_sun, N,
                          max_depth=max_depth, seed=seed)
    return np.pi * lum.sum(axis=0) / N


# ─────────────────────────────────────────────────────────────────────────────
# Internal helper — allocates buffers, launches _trace_deriv_kernel, returns
# all four output arrays (lum, dfic, dedg, sum_q) as host numpy arrays.
# ─────────────────────────────────────────────────────────────────────────────

def _launch_deriv_kernel(scene, obs_dirs, vec_sun, N,
                         max_depth=99, threads_per_block=128, seed=42,
                         start_xy_min_frac=0.0, start_xy_max_frac=1.0,
                         leaf_dist='gaussian'):
    if not GPU_AVAILABLE:
        raise RuntimeError('No CUDA GPU — use renderer.render() instead.')

    rng_np  = np.random.default_rng(seed)
    n_obs   = len(obs_dirs)
    mv      = scene.max_voxel
    vs      = float(scene.voxel_size)
    er      = float(np.max(scene.leaf_radius_max))
    periodic_xy = int(getattr(scene, "periodic_xy", 1))

    start_xy_min_frac = float(np.clip(start_xy_min_frac, 0.0, 1.0))
    start_xy_max_frac = float(np.clip(start_xy_max_frac, 0.0, 1.0))
    if start_xy_max_frac < start_xy_min_frac:
        start_xy_min_frac, start_xy_max_frac = start_xy_max_frac, start_xy_min_frac
    x_min = start_xy_min_frac * float(scene.pos_max[0])
    x_max = start_xy_max_frac * float(scene.pos_max[0])
    y_min = start_xy_min_frac * float(scene.pos_max[1])
    y_max = start_xy_max_frac * float(scene.pos_max[1])
    start_x = rng_np.uniform(x_min, x_max, N).astype(np.float64)
    start_y = rng_np.uniform(y_min, y_max, N).astype(np.float64)

    use_lognormal = int(leaf_dist == 'lognormal')
    if use_lognormal:
        cv2      = (scene.std_LAI / max(scene.LAI_r, 1e-12)) ** 2
        sigma_ln = float(np.sqrt(np.log(1.0 + cv2)))
    else:
        sigma_ln = 0.0

    g = _upload_scene(scene)
    o = _upload_obs(obs_dirs)
    d_sx    = cuda.to_device(start_x)
    d_sy    = cuda.to_device(start_y)
    d_lum      = cuda.to_device(np.zeros((N, n_obs), dtype=np.float64))
    d_dfic     = cuda.to_device(np.zeros((N, n_obs), dtype=np.float64))
    d_dedg     = cuda.to_device(np.zeros((N, n_obs), dtype=np.float64))
    d_sumq     = cuda.to_device(np.zeros((N, n_obs), dtype=np.float64))
    d_nH       = cuda.to_device(np.zeros((N, n_obs), dtype=np.float64))
    d_brkt     = cuda.to_device(np.zeros((N, n_obs), dtype=np.float64))
    d_nH_path  = cuda.to_device(np.zeros(N,           dtype=np.float64))
    d_nF_path  = cuda.to_device(np.zeros(N,           dtype=np.float64))
    d_K_ph     = cuda.to_device(np.zeros(N,           dtype=np.float64))
    d_drho     = cuda.to_device(np.zeros((N, n_obs), dtype=np.float64))
    rng_states = create_xoroshiro128p_states(N, seed=seed)

    blocks = (N + threads_per_block - 1) // threads_per_block
    _trace_deriv_kernel[blocks, threads_per_block](
        g['lpx'], g['lpy'], g['lpz'],
        g['lnx'], g['lny'], g['lnz'], g['lr'],
        g['vc'],  g['vl'],
        vs, int(mv[0]), int(mv[1]), int(mv[2]),
        o['ovx'], o['ovy'], o['ovz'],
        float(vec_sun[0]), float(vec_sun[1]), float(vec_sun[2]),
        float(scene.rho), float(scene.tau), float(scene.rsoil),
        int(max_depth), float(scene.pos_max[2]),
        int(seed), periodic_xy, d_sx, d_sy, rng_states,
        d_lum, d_dfic, d_dedg, d_sumq, d_nH, d_brkt, d_nH_path, d_nF_path,
        d_K_ph, d_drho,
        float(scene.LAI_r),       float(scene.std_LAI),
        float(scene.LAI_Max),     float(scene.P_LAI),
        float(scene.n_leaf_norm), er,
        use_lognormal, sigma_ln,
    )
    cuda.synchronize()
    return (d_lum.copy_to_host(), d_dfic.copy_to_host(),
            d_dedg.copy_to_host(), d_sumq.copy_to_host(),
            d_nH.copy_to_host(),  d_brkt.copy_to_host(),
            d_nH_path.copy_to_host(), d_nF_path.copy_to_host(),
            d_K_ph.copy_to_host(), d_drho.copy_to_host())


# ─────────────────────────────────────────────────────────────────────────────
# Public API — BRF + derivative
# ─────────────────────────────────────────────────────────────────────────────

def compute_deriv_gpu(scene, obs_dirs, vec_sun, N,
                      max_depth=99,
                      threads_per_block=128,
                      seed=42,
                      start_xy_min_frac=0.0,
                      start_xy_max_frac=1.0,
                      leaf_dist='gaussian',
                      return_drho=False):
    """
    GPU BRF + dBRF/dLAI path tracer.

    Parameters
    ----------
    scene             : Scene   must have derivative params:
                                LAI_r, std_LAI, LAI_Max, P_LAI,
                                n_leaf_norm, leaf_radius_max
    obs_dirs          : (n_obs, 3)
    vec_sun           : (3,)
    N                 : int
    max_depth         : int   max scattering bounces (from cfg.max_depth)
    threads_per_block : int
    seed              : int
    leaf_dist         : str   'gaussian' (default) or 'lognormal'

    Returns
    -------
    lum       : (N, n_obs) float64   raw luminance — BRF = π·lum.sum(0)/N
    deriv_fic : (N, n_obs) float64   fictive estimator per photon
    deriv_edg : (N, n_obs) float64   edge    estimator per photon
    K_ph      : (N,)       float64   leaf scattering count per photon
    drho      : (N, n_obs) float64   (only if return_drho) rho score per photon:
                                     Σ_contrib k_eff·ref; dBRF/dρ = π·drho.sum(0)/(N·ρ)
    """
    lum, dfic, dedg, _, _, _, _, _, K_ph, drho = _launch_deriv_kernel(
        scene, obs_dirs, vec_sun, N,
        max_depth=max_depth,
        threads_per_block=threads_per_block,
        seed=seed,
        start_xy_min_frac=start_xy_min_frac,
        start_xy_max_frac=start_xy_max_frac,
        leaf_dist=leaf_dist,
    )
    if return_drho:
        return lum, dfic, dedg, K_ph, drho
    return lum, dfic, dedg, K_ph


def compute_deriv_sumq_gpu(scene, obs_dirs, vec_sun, N,
                           max_depth=99,
                           threads_per_block=128,
                           seed=42,
                           start_xy_min_frac=0.0,
                           start_xy_max_frac=1.0,
                           leaf_dist='gaussian'):
    """
    Like compute_deriv_gpu but also returns the two counting terms (without ref).

    With near-deterministic radii (std_LAI≈0):
      sum_q  ≈ n_fictive / (1−P) / LAI_Max
      nH     ≈ n_real    / P     / LAI_Max
      nH - sum_q = (n_H/P − sum_q_nc) / LAI_Max  = d_fic / ref  (bracket term)

    Returns
    -------
    lum       : (N, n_obs) float64
    deriv_fic : (N, n_obs) float64
    deriv_edg : (N, n_obs) float64
    sum_q     : (N, n_obs) float64   sum_q_nc / LAI_Max  per photon (per bounce sum)
    nH        : (N, n_obs) float64   n_H / (P · LAI_Max) per photon (per bounce sum)
    brkt_out  : (N, n_obs) float64   (n_H/P - sum_q_nc)/LAI_Max per photon — direct bracket (= NH_all - SQ_all)
    nH_path   : (N,)       float64   unique active leaves at path end  (integer-valued)
    nF_path   : (N,)       float64   unique fwd fictive leaves at path end (integer-valued)
    """
    # [:9] drops the appended per-contribution drho array → preserves the
    # historical 9-tuple contract that existing callers unpack.
    return _launch_deriv_kernel(
        scene, obs_dirs, vec_sun, N,
        max_depth=max_depth,
        threads_per_block=threads_per_block,
        seed=seed,
        start_xy_min_frac=start_xy_min_frac,
        start_xy_max_frac=start_xy_max_frac,
        leaf_dist=leaf_dist,
    )[:9]


def compute_path_endpoint_gpu(scene, obs_dirs, vec_sun, N,
                              max_depth=99,
                              threads_per_block=128,
                              seed=42,
                              start_xy_min_frac=0.0,
                              start_xy_max_frac=1.0,
                              leaf_dist='gaussian'):
    """
    Returns per-path terminal leaf counts recorded at the END of each photon path.

    These are the UNIQUE counts accumulated over the ENTIRE path trajectory,
    NOT sums of per-bounce contributions:
      nH_path[n]  = n_H[0]    — total unique active (real) leaves hit
      nF_path[n]  = n_ffic[0] — total unique fictive leaves on the FORWARD path

    With std_mul=0: nF_path is exactly integer.
    NegBin check:  nF_path | nH_path=h  should be ~ NegBin(h, P)
                   if the path model follows the simplified sequential model.

    Returns
    -------
    nH_path : (N,) float64   unique active leaves at path end
    nF_path : (N,) float64   unique fwd fictive leaves at path end
    """
    _, _, _, _, _, _, nH_path, nF_path, _ = _launch_deriv_kernel(
        scene, obs_dirs, vec_sun, N,
        max_depth=max_depth,
        threads_per_block=threads_per_block,
        seed=seed,
        start_xy_min_frac=start_xy_min_frac,
        start_xy_max_frac=start_xy_max_frac,
        leaf_dist=leaf_dist,
    )
    return nH_path, nF_path


def deriv_gpu(scene, obs_dirs, vec_sun, N, max_depth=99, seed=42, leaf_dist='gaussian'):
    """
    Convenience wrapper: returns (brf, deriv_fic, deriv_edg, deriv_opt).
    Mirrors the four derivative fields of RenderResult.
    Generates leaf_effect from scene.P_LAI internally.
    """
    from canopi.derivative import optimal_weight

    lum, D_fic, D_edg, _ = compute_deriv_gpu(
        scene, obs_dirs, vec_sun, N, max_depth=max_depth, seed=seed,
        leaf_dist=leaf_dist)

    brf       = np.pi  * lum.sum(axis=0) / N
    deriv_fic = D_fic.mean(axis=0)
    deriv_edg = D_edg.mean(axis=0)
    w         = optimal_weight(D_fic, D_edg)
    deriv_opt = w * deriv_fic + (1.0 - w) * deriv_edg
    return brf, deriv_fic, deriv_edg, deriv_opt


# ─────────────────────────────────────────────────────────────────────────────
# Public API — backward BRF only
# ─────────────────────────────────────────────────────────────────────────────

def compute_brf_bwd_gpu(scene, obs_dirs, vec_sun, N,
                        max_depth=99,
                        threads_per_block=128,
                        seed=42,
                        start_xy_min_frac=0.0,
                        start_xy_max_frac=1.0):
    """
    Backward GPU BRF tracer.  Returns raw luminance (N, n_obs).

    Launches N photons per observer direction tracing the reversed observer
    ray into the canopy, with NEE toward the sun at each bounce.
    By Helmholtz reciprocity the result matches compute_brf_gpu() within MC noise.

    BRF = π * lum.sum(axis=0) / N
    """
    if not GPU_AVAILABLE:
        raise RuntimeError('No CUDA GPU — use the CPU renderer.')

    rng_np  = np.random.default_rng(seed)
    n_obs   = len(obs_dirs)
    mv      = scene.max_voxel
    vs      = float(scene.voxel_size)
    periodic_xy = int(getattr(scene, "periodic_xy", 1))

    start_xy_min_frac = float(np.clip(start_xy_min_frac, 0.0, 1.0))
    start_xy_max_frac = float(np.clip(start_xy_max_frac, 0.0, 1.0))
    if start_xy_max_frac < start_xy_min_frac:
        start_xy_min_frac, start_xy_max_frac = start_xy_max_frac, start_xy_min_frac
    x_min = start_xy_min_frac * float(scene.pos_max[0])
    x_max = start_xy_max_frac * float(scene.pos_max[0])
    y_min = start_xy_min_frac * float(scene.pos_max[1])
    y_max = start_xy_max_frac * float(scene.pos_max[1])

    g      = _upload_scene(scene)
    blocks = (N + threads_per_block - 1) // threads_per_block
    lum_host = np.zeros((N, n_obs), dtype=np.float64)

    for io in range(n_obs):
        start_x = rng_np.uniform(x_min, x_max, N).astype(np.float64)
        start_y = rng_np.uniform(y_min, y_max, N).astype(np.float64)
        d_sx    = cuda.to_device(start_x)
        d_sy    = cuda.to_device(start_y)
        d_col   = cuda.to_device(np.zeros(N, dtype=np.float64))
        rng_states = create_xoroshiro128p_states(N, seed=seed + io)

        _trace_bwd_kernel[blocks, threads_per_block](
            g['lpx'], g['lpy'], g['lpz'],
            g['lnx'], g['lny'], g['lnz'], g['lr'],
            g['vc'],  g['vl'],
            vs, int(mv[0]), int(mv[1]), int(mv[2]),
            float(obs_dirs[io, 0]), float(obs_dirs[io, 1]), float(obs_dirs[io, 2]),
            float(vec_sun[0]), float(vec_sun[1]), float(vec_sun[2]),
            float(scene.rho), float(scene.tau), float(scene.rsoil),
            int(max_depth), float(scene.pos_max[2]),
            float(scene.P_LAI), int(seed),
            periodic_xy,
            d_sx, d_sy, rng_states, d_col,
        )
        cuda.synchronize()
        lum_host[:, io] = d_col.copy_to_host()

    return lum_host


def brf_bwd_gpu(scene, obs_dirs, vec_sun, N, max_depth=99, seed=42):
    """Convenience wrapper: returns BRF (n_obs,) from backward tracer."""
    lum = compute_brf_bwd_gpu(scene, obs_dirs, vec_sun, N,
                               max_depth=max_depth, seed=seed)
    return np.pi * lum.sum(axis=0) / N


# ─────────────────────────────────────────────────────────────────────────────
# Public API — backward BRF + derivative
# ─────────────────────────────────────────────────────────────────────────────

def compute_deriv_bwd_gpu(scene, obs_dirs, vec_sun, N,
                           max_depth=99,
                           threads_per_block=128,
                           seed=42,
                           start_xy_min_frac=0.0,
                           start_xy_max_frac=1.0,
                           leaf_dist='gaussian'):
    """
    Backward GPU BRF + dBRF/dLAI tracer.

    Parameters
    ----------
    leaf_dist : str  'gaussian' (default) or 'lognormal'

    Returns
    -------
    lum       : (N, n_obs) float64
    deriv_fic : (N, n_obs) float64
    deriv_edg : (N, n_obs) float64
    """
    if not GPU_AVAILABLE:
        raise RuntimeError('No CUDA GPU — use renderer.render() instead.')

    rng_np  = np.random.default_rng(seed)
    n_obs   = len(obs_dirs)
    mv      = scene.max_voxel
    vs      = float(scene.voxel_size)
    er      = float(np.max(scene.leaf_radius_max))
    periodic_xy = int(getattr(scene, "periodic_xy", 1))

    start_xy_min_frac = float(np.clip(start_xy_min_frac, 0.0, 1.0))
    start_xy_max_frac = float(np.clip(start_xy_max_frac, 0.0, 1.0))
    if start_xy_max_frac < start_xy_min_frac:
        start_xy_min_frac, start_xy_max_frac = start_xy_max_frac, start_xy_min_frac
    x_min = start_xy_min_frac * float(scene.pos_max[0])
    x_max = start_xy_max_frac * float(scene.pos_max[0])
    y_min = start_xy_min_frac * float(scene.pos_max[1])
    y_max = start_xy_max_frac * float(scene.pos_max[1])

    use_lognormal = int(leaf_dist == 'lognormal')
    if use_lognormal:
        cv2      = (scene.std_LAI / max(scene.LAI_r, 1e-12)) ** 2
        sigma_ln = float(np.sqrt(np.log(1.0 + cv2)))
    else:
        sigma_ln = 0.0

    g      = _upload_scene(scene)
    blocks = (N + threads_per_block - 1) // threads_per_block
    lum_host  = np.zeros((N, n_obs), dtype=np.float64)
    dfic_host = np.zeros((N, n_obs), dtype=np.float64)
    dedg_host = np.zeros((N, n_obs), dtype=np.float64)

    for io in range(n_obs):
        start_x = rng_np.uniform(x_min, x_max, N).astype(np.float64)
        start_y = rng_np.uniform(y_min, y_max, N).astype(np.float64)
        d_sx    = cuda.to_device(start_x)
        d_sy    = cuda.to_device(start_y)
        d_col   = cuda.to_device(np.zeros(N, dtype=np.float64))
        d_dfic  = cuda.to_device(np.zeros(N, dtype=np.float64))
        d_dedg  = cuda.to_device(np.zeros(N, dtype=np.float64))
        rng_states = create_xoroshiro128p_states(N, seed=seed + io)

        _trace_bwd_deriv_kernel[blocks, threads_per_block](
            g['lpx'], g['lpy'], g['lpz'],
            g['lnx'], g['lny'], g['lnz'], g['lr'],
            g['vc'],  g['vl'],
            vs, int(mv[0]), int(mv[1]), int(mv[2]),
            float(obs_dirs[io, 0]), float(obs_dirs[io, 1]), float(obs_dirs[io, 2]),
            float(vec_sun[0]), float(vec_sun[1]), float(vec_sun[2]),
            float(scene.rho), float(scene.tau), float(scene.rsoil),
            int(max_depth), float(scene.pos_max[2]),
            int(seed), periodic_xy, d_sx, d_sy, rng_states,
            d_col, d_dfic, d_dedg,
            float(scene.LAI_r),       float(scene.std_LAI),
            float(scene.LAI_Max),     float(scene.P_LAI),
            float(scene.n_leaf_norm), er,
            use_lognormal, sigma_ln,
        )
        cuda.synchronize()
        lum_host[:, io]  = d_col.copy_to_host()
        dfic_host[:, io] = d_dfic.copy_to_host()
        dedg_host[:, io] = d_dedg.copy_to_host()

    return lum_host, dfic_host, dedg_host


def compute_deriv_bwd_species_gpu(scene, obs_dirs, vec_sun, N,
                                   leaf_species_id, species_stats,
                                   max_depth=99, threads_per_block=128,
                                   seed=42, P_LAI_sp=None, rho_sp_arr=None,
                                   tau_sp_arr=None, species_shadow=None):
    """
    Backward BRF + per-species dBRF/dLAI for multiple observation directions.
    P_LAI_sp, rho_sp_arr, tau_sp_arr, species_shadow: as render_image_bwd_deriv_species_gpu.

    Returns
    -------
    lum  : (N, n_obs)         float64  raw per-photon luminance
    dfic : (N, n_obs, N_SP)   float64  per-photon per-species fictive deriv
    dedg : (N, n_obs, N_SP)   float64  per-photon per-species edge deriv
    """
    if not GPU_AVAILABLE:
        raise RuntimeError('No CUDA GPU available.')

    rng_np  = np.random.default_rng(seed)
    n_obs   = len(obs_dirs)
    mv      = scene.max_voxel
    vs      = float(scene.voxel_size)
    er      = float(np.max(scene.leaf_radius_max))
    periodic_xy = int(getattr(scene, 'periodic_xy', 1))
    N_SP    = _N_SP

    use_lognormal = 0
    sigma_ln      = 0.0

    g = _upload_scene(scene)
    blocks = (N + threads_per_block - 1) // threads_per_block

    d_lsp    = cuda.to_device(leaf_species_id.astype(np.int32))
    d_lai_sp = cuda.to_device(species_stats['LAI_Max_sp'].astype(np.float64))
    d_nln_sp = cuda.to_device(species_stats['n_leaf_norm_sp'].astype(np.float64))
    if P_LAI_sp is None:
        P_LAI_sp = np.full(_N_SP, float(scene.P_LAI), dtype=np.float64)
    if rho_sp_arr is None:
        rho_sp_arr = np.full(_N_SP, float(scene.rho), dtype=np.float64)
    rho_sp_arr = np.asarray(rho_sp_arr, dtype=np.float64)
    if tau_sp_arr is None:
        ratio = float(scene.tau) / float(scene.rho) if float(scene.rho) > 0 else 1.0
        tau_sp_arr = rho_sp_arr * ratio
    d_p_lai_sp = cuda.to_device(np.asarray(P_LAI_sp, dtype=np.float64))
    d_rho_sp   = cuda.to_device(rho_sp_arr)
    d_tau_sp   = cuda.to_device(np.asarray(tau_sp_arr, dtype=np.float64))
    shadow     = int(bool(_SPECIES_SHADOW if species_shadow is None else species_shadow))

    lum_host  = np.zeros((N, n_obs),       dtype=np.float64)
    dfic_host = np.zeros((N, n_obs, N_SP), dtype=np.float64)
    dedg_host = np.zeros((N, n_obs, N_SP), dtype=np.float64)

    for io in range(n_obs):
        start_x = rng_np.uniform(0, float(scene.pos_max[0]), N).astype(np.float64)
        start_y = rng_np.uniform(0, float(scene.pos_max[1]), N).astype(np.float64)
        d_sx   = cuda.to_device(start_x)
        d_sy   = cuda.to_device(start_y)
        d_col  = cuda.to_device(np.zeros(N,        dtype=np.float64))
        d_dfic = cuda.to_device(np.zeros(N * N_SP, dtype=np.float64))
        d_dedg = cuda.to_device(np.zeros(N * N_SP, dtype=np.float64))
        d_drho = cuda.to_device(np.zeros(N * N_SP, dtype=np.float64))
        rng_states = create_xoroshiro128p_states(N, seed=seed + io)

        _trace_bwd_deriv_species_kernel[blocks, threads_per_block](
            g['lpx'], g['lpy'], g['lpz'],
            g['lnx'], g['lny'], g['lnz'], g['lr'],
            g['vc'],  g['vl'],
            vs, int(mv[0]), int(mv[1]), int(mv[2]),
            float(obs_dirs[io, 0]), float(obs_dirs[io, 1]), float(obs_dirs[io, 2]),
            float(vec_sun[0]), float(vec_sun[1]), float(vec_sun[2]),
            float(scene.rho), float(scene.tau), float(scene.rsoil),
            int(max_depth), float(scene.pos_max[2]),
            int(seed), periodic_xy,
            d_sx, d_sy, rng_states,
            d_col, d_dfic, d_dedg, d_drho,
            float(scene.LAI_r),       float(scene.std_LAI),
            float(scene.LAI_Max),     float(scene.P_LAI),
            float(scene.n_leaf_norm), er,
            use_lognormal, sigma_ln,
            d_lsp, d_lai_sp, d_nln_sp,
            d_p_lai_sp, d_rho_sp, d_tau_sp, shadow,
        )
        cuda.synchronize()
        lum_host[:, io]    = d_col.copy_to_host()
        tmp = d_dfic.copy_to_host().reshape(N, N_SP)
        dfic_host[:, io, :] = tmp
        tmp = d_dedg.copy_to_host().reshape(N, N_SP)
        dedg_host[:, io, :] = tmp
        del d_sx, d_sy, d_col, d_dfic, d_dedg, d_drho, rng_states

    del g, d_lsp, d_lai_sp, d_nln_sp, d_p_lai_sp, d_rho_sp, d_tau_sp
    gc.collect()
    cuda.current_context().deallocations.clear()
    return lum_host, dfic_host, dedg_host


def deriv_bwd_gpu(scene, obs_dirs, vec_sun, N, max_depth=99, seed=42, leaf_dist='gaussian'):
    """Convenience wrapper: returns (brf, deriv_fic, deriv_edg, deriv_opt)."""
    from canopi.derivative import optimal_weight

    lum, D_fic, D_edg = compute_deriv_bwd_gpu(
        scene, obs_dirs, vec_sun, N, max_depth=max_depth, seed=seed,
        leaf_dist=leaf_dist)

    brf       = np.pi  * lum.sum(axis=0) / N
    deriv_fic = D_fic.mean(axis=0)
    deriv_edg = D_edg.mean(axis=0)
    w         = optimal_weight(D_fic, D_edg)
    deriv_opt = w * deriv_fic + (1.0 - w) * deriv_edg
    return brf, deriv_fic, deriv_edg, deriv_opt


# ─────────────────────────────────────────────────────────────────────────────
# render_gpu() — full drop-in replacement for renderer.render()
# ─────────────────────────────────────────────────────────────────────────────

def render_gpu(cfg, threads_per_block=128, scene=None):
    """
    GPU-accelerated render.  Returns a RenderResult with the same structure
    as renderer.render(cfg).

    Parameters
    ----------
    cfg : SceneConfig
    threads_per_block : int
    scene : canopi.scene.Scene or None
        If provided, use this pre-built scene (e.g. from load_jbs_scene)
        instead of building a random homogeneous canopy from cfg.
        The scene's rho/tau/rsoil are overridden by cfg values so that
        the same geometry can be reused across bands.

    Usage
    -----
        from canopi.gpu import render_gpu, GPU_AVAILABLE
        from canopi import render

        result = render_gpu(cfg) if GPU_AVAILABLE else render(cfg)

    Differences from CPU render()
    ------------------------------
    • Leaf radii are resampled ONCE (in build_scene) rather than per-photon.
      This slightly increases derivative variance; compensate by increasing N —
      the GPU makes larger N cheap.
    • The `image` field is a zero-filled placeholder.  Per-photon hit positions
      are not tracked on the GPU (the global write per bounce would serialise
      threads); run a separate CPU imaging pass if needed.
    """
    if not GPU_AVAILABLE:
        raise RuntimeError('No CUDA GPU — use renderer.render(cfg) instead.')

    from canopi.renderer   import RenderResult, _build_obs_dirs
    from canopi.scene      import build_scene
    from canopi.geometry   import polar_vect
    from canopi.derivative import optimal_weight

    rng = np.random.default_rng(cfg.seed)
    if scene is None:
        scene = build_scene(cfg, rng)
    else:
        # Override optical params from cfg so geometry is reusable across bands
        scene.rho   = float(cfg.rho)
        scene.tau   = float(cfg.tau)
        scene.rsoil = float(cfg.rsoil)

    obs_deg, obs_dirs, n_pp = _build_obs_dirs(cfg)
    n_obs = len(obs_dirs)

    deg     = np.pi / 180
    tha_s   = (180.0 - cfg.sun_zenith_deg) * deg
    phi_s   = cfg.sun_azimuth_deg * deg
    vec_sun = polar_vect(tha_s, phi_s)

    N = cfg.n_photons

    start_xy_min_frac = float(getattr(cfg, "photon_start_xy_min_frac", 0.0))
    start_xy_max_frac = float(getattr(cfg, "photon_start_xy_max_frac", 1.0))

    if cfg.compute_derivative:
        print(
            f'  [render_gpu] N={N:,}  derivative=ON  '
            f'_MAX_C={_MAX_C}  _MAX_FIC={_MAX_FIC}  '
            f'threads/block={threads_per_block}'
        )
        lum, D_fic, D_edg, K_ph, D_rho = compute_deriv_gpu(
            scene, obs_dirs, vec_sun, N,
            max_depth=cfg.max_depth,
            threads_per_block=threads_per_block,
            seed=cfg.seed,
            start_xy_min_frac=start_xy_min_frac,
            start_xy_max_frac=start_xy_max_frac,
            return_drho=True,
        )
        brf       = np.pi * lum.sum(axis=0) / N
        deriv_fic = D_fic.mean(axis=0)
        deriv_edg = D_edg.mean(axis=0)
        w         = optimal_weight(D_fic, D_edg)
        deriv_opt = w * deriv_fic + (1.0 - w) * deriv_edg
        # dBRF/dρ: per-contribution score Σ k_eff·ref / ρ (k_eff weights each
        # local-estimation by the scatter order at which it was emitted, +1 for
        # the connecting leaf BRDF).  Replaces the buggy K_ph·Σ(lum) form.
        if scene.rho > 0:
            deriv_rho = D_rho.sum(axis=0) / (N * scene.rho)
        else:
            deriv_rho = np.zeros(n_obs)
    else:
        print(
            f'  [render_gpu] N={N:,}  derivative=OFF  '
            f'threads/block={threads_per_block}'
        )
        lum_raw   = compute_brf_gpu(
            scene, obs_dirs, vec_sun, N,
            max_depth=cfg.max_depth,
            threads_per_block=threads_per_block,
            seed=cfg.seed,
            start_xy_min_frac=start_xy_min_frac,
            start_xy_max_frac=start_xy_max_frac,
        )
        brf       = np.pi * lum_raw.sum(axis=0) / N
        deriv_fic = np.zeros(n_obs)
        deriv_edg = np.zeros(n_obs)
        deriv_opt = np.zeros(n_obs)
        deriv_rho = np.zeros(n_obs)
        w         = np.full(n_obs, 0.5)

    img_H, img_W = cfg.image_size
    return RenderResult(
        obs_zenith_deg = obs_deg,
        obs_dirs       = obs_dirs,
        n_pp           = n_pp,
        brf            = brf,
        image          = np.zeros((img_H, img_W)),
        deriv_fic      = deriv_fic,
        deriv_edg      = deriv_edg,
        deriv_opt      = deriv_opt,
        deriv_rho      = deriv_rho,
        n_photons      = N,
        opt_weights    = w,
    )


# ─────────────────────────────────────────────────────────────────────────────
# render_image_bwd_gpu() — spatially-resolved backward image (GPU)
# ─────────────────────────────────────────────────────────────────────────────

def render_image_bwd_gpu(scene, obs_dir, vec_sun,
                          img_H=64, img_W=64, N_spp=4,
                          max_depth=99, threads_per_block=128, seed=42):
    """
    Render a spatially-resolved image using backward ray tracing (GPU).

    For each pixel (i, j) in img_H × img_W, N_spp backward rays are traced
    from the sensor direction (-obs_dir) through the scene, with NEE toward
    the sun at each bounce.  By Helmholtz reciprocity the pixel values match
    the forward BRF at the corresponding spatial location.

    Parameters
    ----------
    scene            : Scene  — built by build_scene()
    obs_dir          : (3,)   — unit vector FROM scene TOWARD sensor (z > 0)
    vec_sun          : (3,)   — sun direction FROM sun TOWARD scene (z < 0)
    img_H, img_W     : int    — image height × width in pixels
    N_spp            : int    — MC samples per pixel (anti-aliasing)
    max_depth        : int
    threads_per_block: int
    seed             : int

    Returns
    -------
    image : (img_H, img_W) float64  — π × mean luminance per pixel (BRF units)
    """
    if not GPU_AVAILABLE:
        raise RuntimeError('No CUDA GPU — use render_backward_image() (CPU) instead.')

    Lx = float(scene.pos_max[0])
    Ly = float(scene.pos_max[1])
    mv = scene.max_voxel
    vs = float(scene.voxel_size)
    periodic_xy = int(getattr(scene, 'periodic_xy', 1))
    N_total = img_H * img_W * N_spp

    rng_np = np.random.default_rng(seed)

    # Build pixel → scene-position mapping
    # Photon n = i*img_W*N_spp + j*N_spp + s
    idx   = np.arange(N_total)
    i_idx = idx // (img_W * N_spp)        # pixel row
    j_idx = (idx // N_spp) % img_W        # pixel col

    # Jittered positions within each pixel (stratified sampling)
    jitter_x = rng_np.random(N_total)
    jitter_y = rng_np.random(N_total)
    start_x  = ((j_idx + jitter_x) / img_W * Lx).astype(np.float64)
    start_y  = ((i_idx + jitter_y) / img_H * Ly).astype(np.float64)

    g      = _upload_scene(scene)
    blocks = (N_total + threads_per_block - 1) // threads_per_block

    d_sx       = cuda.to_device(start_x)
    d_sy       = cuda.to_device(start_y)
    d_col      = cuda.to_device(np.zeros(N_total, dtype=np.float64))
    rng_states = create_xoroshiro128p_states(N_total, seed=seed)

    ox, oy, oz = float(obs_dir[0]), float(obs_dir[1]), float(obs_dir[2])

    _trace_bwd_kernel[blocks, threads_per_block](
        g['lpx'], g['lpy'], g['lpz'],
        g['lnx'], g['lny'], g['lnz'], g['lr'],
        g['vc'],  g['vl'],
        vs, int(mv[0]), int(mv[1]), int(mv[2]),
        ox, oy, oz,
        float(vec_sun[0]), float(vec_sun[1]), float(vec_sun[2]),
        float(scene.rho), float(scene.tau), float(scene.rsoil),
        int(max_depth), float(scene.pos_max[2]),
        float(scene.P_LAI), int(seed),
        periodic_xy,
        d_sx, d_sy, rng_states, d_col,
    )
    cuda.synchronize()

    lum_flat = d_col.copy_to_host()               # (N_total,)
    lum_3d   = lum_flat.reshape(img_H, img_W, N_spp)
    image    = np.pi * lum_3d.mean(axis=2)        # (img_H, img_W)
    return image


# ─────────────────────────────────────────────────────────────────────────────
# render_image_bwd_deriv_gpu() — BRF + dBRF/dLAI image (GPU)
# ─────────────────────────────────────────────────────────────────────────────

def render_image_bwd_deriv_gpu(scene, obs_dir, vec_sun,
                                img_H=64, img_W=64, N_spp=4,
                                max_depth=99, threads_per_block=128, seed=42):
    """
    Render spatially-resolved BRF + dBRF/dLAI images using backward ray tracing (GPU).

    Calls _trace_bwd_deriv_kernel with a pixel-grid of start positions, so
    every photon maps to a specific pixel.  After the kernel, the flat per-photon
    arrays are reshaped to (img_H, img_W, N_spp) and averaged.  A global
    optimal weight (minimising variance across all N_total samples) combines
    fictive and edge derivative estimates.

    Parameters
    ----------
    scene, obs_dir, vec_sun : same as render_image_bwd_gpu
    img_H, img_W, N_spp     : image size and samples per pixel
    max_depth, seed          : ray depth and random seed

    Returns
    -------
    brf_image   : (img_H, img_W) float64
    dfic_image  : (img_H, img_W) float64  — fictive estimator
    dedg_image  : (img_H, img_W) float64  — edge estimator
    dopt_image  : (img_H, img_W) float64  — optimal weighted
    """
    if not GPU_AVAILABLE:
        raise RuntimeError('No CUDA GPU — use render_backward_image_deriv() (CPU).')

    from canopi.derivative import optimal_weight

    Lx = float(scene.pos_max[0])
    Ly = float(scene.pos_max[1])
    mv = scene.max_voxel
    vs = float(scene.voxel_size)
    periodic_xy = int(getattr(scene, 'periodic_xy', 1))
    N_total = img_H * img_W * N_spp
    er      = float(np.max(scene.leaf_radius_max))

    rng_np = np.random.default_rng(seed)

    idx   = np.arange(N_total)
    i_idx = idx // (img_W * N_spp)
    j_idx = (idx // N_spp) % img_W

    jitter_x = rng_np.random(N_total)
    jitter_y = rng_np.random(N_total)
    start_x  = ((j_idx + jitter_x) / img_W * Lx).astype(np.float64)
    start_y  = ((i_idx + jitter_y) / img_H * Ly).astype(np.float64)

    g      = _upload_scene(scene)
    blocks = (N_total + threads_per_block - 1) // threads_per_block

    d_sx    = cuda.to_device(start_x)
    d_sy    = cuda.to_device(start_y)
    d_col   = cuda.to_device(np.zeros(N_total, dtype=np.float64))
    d_dfic  = cuda.to_device(np.zeros(N_total, dtype=np.float64))
    d_dedg  = cuda.to_device(np.zeros(N_total, dtype=np.float64))
    rng_states = create_xoroshiro128p_states(N_total, seed=seed)

    ox, oy, oz = float(obs_dir[0]), float(obs_dir[1]), float(obs_dir[2])

    _trace_bwd_deriv_kernel[blocks, threads_per_block](
        g['lpx'], g['lpy'], g['lpz'],
        g['lnx'], g['lny'], g['lnz'], g['lr'],
        g['vc'],  g['vl'],
        vs, int(mv[0]), int(mv[1]), int(mv[2]),
        ox, oy, oz,
        float(vec_sun[0]), float(vec_sun[1]), float(vec_sun[2]),
        float(scene.rho), float(scene.tau), float(scene.rsoil),
        int(max_depth), float(scene.pos_max[2]),
        int(seed), periodic_xy,
        d_sx, d_sy, rng_states,
        d_col, d_dfic, d_dedg,
        float(scene.LAI_r),       float(scene.std_LAI),
        float(scene.LAI_Max),     float(scene.P_LAI),
        float(scene.n_leaf_norm), er,
        np.int32(0), np.float64(0.0),   # use_lognormal=0, sigma_ln=0 (normal/step)
    )
    cuda.synchronize()

    lum_flat  = d_col.copy_to_host()
    dfic_flat = d_dfic.copy_to_host()
    dedg_flat = d_dedg.copy_to_host()

    # Reshape to (img_H, img_W, N_spp) and average over samples
    brf_image  = np.pi * lum_flat.reshape (img_H, img_W, N_spp).mean(axis=2)
    dfic_image =         dfic_flat.reshape(img_H, img_W, N_spp).mean(axis=2)
    dedg_image =         dedg_flat.reshape(img_H, img_W, N_spp).mean(axis=2)

    # Global optimal weight from all N_total samples (single direction → n_obs=1)
    w = optimal_weight(dfic_flat.reshape(-1, 1),
                       dedg_flat.reshape(-1, 1))   # (1,)
    dopt_image = w[0] * dfic_image + (1.0 - w[0]) * dedg_image

    return brf_image, dfic_image, dedg_image, dopt_image


# ─────────────────────────────────────────────────────────────────────────────
# render_image_bwd_deriv_species_gpu() — per-species dBRF/dLAI images
# ─────────────────────────────────────────────────────────────────────────────

def upload_scene_gpu(scene, leaf_species_id=None, species_stats=None):
    """Pre-upload scene + optional species arrays to GPU once."""
    g = _upload_scene(scene)
    if leaf_species_id is not None:
        g['d_lsp'] = cuda.to_device(leaf_species_id.astype(np.int32))
    if species_stats is not None:
        g['d_lai_sp'] = cuda.to_device(species_stats['LAI_Max_sp'].astype(np.float64))
        g['d_nln_sp'] = cuda.to_device(species_stats['n_leaf_norm_sp'].astype(np.float64))
    return g


def create_rng_states(N_total, seed=42):
    """Pre-create RNG states once; pass as _rng_states to avoid re-init per render."""
    return create_xoroshiro128p_states(N_total, seed=seed)


def render_image_bwd_deriv_species_gpu(scene, obs_dir, vec_sun,
                                        leaf_species_id, species_stats,
                                        img_H=64, img_W=64, N_spp=4,
                                        max_depth=99, threads_per_block=128, seed=42,
                                        P_LAI_sp=None, rho_sp_arr=None, tau_sp_arr=None,
                                        species_shadow=None,
                                        _scene_gpu=None, _rng_states=None):
    """
    Render BRF + per-species dBRF/dLAI_s images (backward ray tracing, GPU).

    Parameters
    ----------
    scene            : Scene
    obs_dir          : (3,)  observer direction (unit vector, z > 0)
    vec_sun          : (3,)  sun direction (z < 0)
    leaf_species_id  : (N_leaves,) int32  species index 0..6 per leaf
    species_stats    : dict with keys LAI_Max_sp, n_leaf_norm_sp, species_list
    img_H, img_W     : image size
    N_spp            : samples per pixel
    max_depth, seed  : ray depth and RNG seed

    Returns
    -------
    brf_image        : (img_H, img_W)          BRF
    dfic_images      : (N_SP, img_H, img_W)    fictive estimator per species
    dedg_images      : (N_SP, img_H, img_W)    edge estimator per species
    dopt_images      : (N_SP, img_H, img_W)    optimal weighted per species
    """
    if not GPU_AVAILABLE:
        raise RuntimeError('No CUDA GPU available.')

    from canopi.derivative import optimal_weight

    N_SP = _N_SP
    Lx   = float(scene.pos_max[0])
    Ly   = float(scene.pos_max[1])
    mv   = scene.max_voxel
    vs   = float(scene.voxel_size)
    periodic_xy = int(getattr(scene, 'periodic_xy', 1))
    N_total = img_H * img_W * N_spp
    er      = float(np.max(scene.leaf_radius_max))

    use_lognormal = 0
    sigma_ln      = 0.0

    rng_np = np.random.default_rng(seed)
    idx    = np.arange(N_total)
    i_idx  = idx // (img_W * N_spp)
    j_idx  = (idx // N_spp) % img_W
    jitter_x = rng_np.random(N_total)
    jitter_y = rng_np.random(N_total)
    start_x  = ((j_idx + jitter_x) / img_W * Lx).astype(np.float64)
    start_y  = ((i_idx + jitter_y) / img_H * Ly).astype(np.float64)

    g      = _scene_gpu if _scene_gpu is not None else _upload_scene(scene)
    blocks = (N_total + threads_per_block - 1) // threads_per_block

    d_sx        = cuda.to_device(start_x)
    d_sy        = cuda.to_device(start_y)
    d_col       = cuda.to_device(np.zeros(N_total, dtype=np.float64))
    d_dfic      = cuda.to_device(np.zeros(N_total * N_SP, dtype=np.float64))
    d_dedg      = cuda.to_device(np.zeros(N_total * N_SP, dtype=np.float64))
    d_drho      = cuda.to_device(np.zeros(N_total * N_SP, dtype=np.float64))
    d_lsp        = _scene_gpu['d_lsp']       if (_scene_gpu and 'd_lsp'    in _scene_gpu) \
                   else cuda.to_device(leaf_species_id.astype(np.int32))
    d_lai_sp     = _scene_gpu['d_lai_sp']    if (_scene_gpu and 'd_lai_sp' in _scene_gpu) \
                   else cuda.to_device(species_stats['LAI_Max_sp'].astype(np.float64))
    d_nln_sp     = _scene_gpu['d_nln_sp']    if (_scene_gpu and 'd_nln_sp' in _scene_gpu) \
                   else cuda.to_device(species_stats['n_leaf_norm_sp'].astype(np.float64))
    if P_LAI_sp is None:
        P_LAI_sp = np.full(_N_SP, float(scene.P_LAI), dtype=np.float64)
    d_p_lai_sp   = cuda.to_device(np.asarray(P_LAI_sp, dtype=np.float64))
    # Leaf optics per species.  Defaults: scene.rho / scene.tau for every species; if only
    # rho_sp_arr is given, tau_s = rho_s * scene.tau / scene.rho (fixed ratio; = rho_s when the
    # caller set scene.tau = scene.rho, the former symmetric behaviour).
    if rho_sp_arr is None:
        rho_sp_arr = np.full(_N_SP, float(scene.rho), dtype=np.float64)
    rho_sp_arr = np.asarray(rho_sp_arr, dtype=np.float64)
    if tau_sp_arr is None:
        ratio = float(scene.tau) / float(scene.rho) if float(scene.rho) > 0 else 1.0
        tau_sp_arr = rho_sp_arr * ratio
    d_rho_sp_arr = cuda.to_device(rho_sp_arr)
    d_tau_sp_arr = cuda.to_device(np.asarray(tau_sp_arr, dtype=np.float64))
    rng_states  = _rng_states if _rng_states is not None else \
                  create_xoroshiro128p_states(N_total, seed=seed)

    ox, oy, oz = float(obs_dir[0]), float(obs_dir[1]), float(obs_dir[2])

    _trace_bwd_deriv_species_kernel[blocks, threads_per_block](
        g['lpx'], g['lpy'], g['lpz'],
        g['lnx'], g['lny'], g['lnz'], g['lr'],
        g['vc'],  g['vl'],
        vs, int(mv[0]), int(mv[1]), int(mv[2]),
        ox, oy, oz,
        float(vec_sun[0]), float(vec_sun[1]), float(vec_sun[2]),
        float(scene.rho), float(scene.tau), float(scene.rsoil),
        int(max_depth), float(scene.pos_max[2]),
        int(seed), periodic_xy,
        d_sx, d_sy, rng_states,
        d_col, d_dfic, d_dedg, d_drho,
        float(scene.LAI_r),       float(scene.std_LAI),
        float(scene.LAI_Max),     float(scene.P_LAI),
        float(scene.n_leaf_norm), er,
        use_lognormal, sigma_ln,
        d_lsp, d_lai_sp, d_nln_sp,
        d_p_lai_sp,
        d_rho_sp_arr,
        d_tau_sp_arr,
        int(bool(_SPECIES_SHADOW if species_shadow is None else species_shadow)),
    )
    cuda.synchronize()

    lum_flat  = d_col.copy_to_host()
    dfic_flat = d_dfic.copy_to_host().reshape(N_total, N_SP)
    dedg_flat = d_dedg.copy_to_host().reshape(N_total, N_SP)
    drho_flat = d_drho.copy_to_host().reshape(N_total, N_SP)

    del d_sx, d_sy, d_col, d_dfic, d_dedg, d_drho, d_lsp, d_lai_sp, d_nln_sp, d_p_lai_sp, d_rho_sp_arr, d_tau_sp_arr, rng_states, g
    gc.collect()
    cuda.current_context().deallocations.clear()

    brf_image = np.pi * lum_flat.reshape(img_H, img_W, N_spp).mean(axis=2)

    dfic_images = np.zeros((N_SP, img_H, img_W))
    dedg_images = np.zeros((N_SP, img_H, img_W))
    dopt_images = np.zeros((N_SP, img_H, img_W))
    drho_images = np.zeros((N_SP, img_H, img_W))

    for s in range(N_SP):
        dfic_s = dfic_flat[:, s].reshape(img_H, img_W, N_spp).mean(axis=2)
        dedg_s = dedg_flat[:, s].reshape(img_H, img_W, N_spp).mean(axis=2)
        w_s    = optimal_weight(dfic_flat[:, s:s+1], dedg_flat[:, s:s+1])
        dopt_s = w_s[0] * dfic_s + (1.0 - w_s[0]) * dedg_s
        dfic_images[s] = dfic_s
        dedg_images[s] = dedg_s
        dopt_images[s] = dopt_s
        # drho_images[s] = pi * E[K_s * lum] / rho  (divide by rho at call site)
        drho_images[s] = np.pi * drho_flat[:, s].reshape(img_H, img_W, N_spp).mean(axis=2)

    return brf_image, dfic_images, dedg_images, dopt_images, drho_images


# ─────────────────────────────────────────────────────────────────────────────
# benchmark() — CPU vs GPU throughput table (BRF and derivative)
# ─────────────────────────────────────────────────────────────────────────────

def benchmark(scene, obs_dirs, vec_sun, cfg, N_list=None):
    """
    Print a side-by-side throughput table:
      CPU      — trace_photon() + compute_photon_derivative() (Numba JIT, 1 core)
      GPU BRF  — compute_brf_gpu()
      GPU+deriv — compute_deriv_gpu()

    Usage
    -----
        from canopi.gpu import benchmark
        benchmark(scene, obs_dirs, vec_sun, cfg, N_list=[1000, 10000, 100000])
    """
    import time
    from canopi.intersection import warmup
    from canopi.path_tracer  import trace_photon
    from canopi.nee          import NEEEngine
    from canopi.derivative   import compute_photon_derivative

    if N_list is None:
        N_list = [500, 1_000, 5_000, 10_000, 50_000]

    # ── CPU warmup + timing ───────────────────────────────────────────────
    warmup(scene)
    nee = NEEEngine(scene, collect_deriv=True)
    nee.warmup(obs_dirs, np.ones(scene.n_leaf, dtype=np.int8))

    rng = np.random.default_rng(0)
    le  = np.ones(scene.n_leaf, dtype=np.int8)
    P0  = np.array([scene.pos_max[0] / 2, scene.pos_max[1] / 2, 0.0])

    n_rep = 10
    t0 = time.perf_counter()
    for _ in range(n_rep):
        ref, contacts, nc = trace_photon(
            scene, nee, P0, vec_sun, obs_dirs,
            6.77e-5, 1e-12, le, cfg.max_depth,
            1e-5, cfg.edge_threshold, rng,
            compute_deriv=True, rr_depth=999,
        )
        if nc > 0:
            compute_photon_derivative(
                contacts, ref, nc,
                scene.LAI_r, scene.std_LAI, scene.LAI_Max,
                scene.P_LAI, scene.n_leaf_norm, len(obs_dirs),
            )
    dt_cpu_ms = (time.perf_counter() - t0) / n_rep * 1e3
    print(f'CPU (JIT + derivative): {dt_cpu_ms:.2f} ms / photon\n')

    hdr = (f'{"N":>8}  {"CPU (s)":>9}  '
           f'{"GPU BRF (s)":>12}  {"Speedup":>8}  '
           f'{"GPU+deriv (s)":>14}  {"Speedup":>8}')
    print(hdr)
    print('─' * len(hdr))

    for N in N_list:
        t_cpu = dt_cpu_ms * N / 1e3

        if not GPU_AVAILABLE:
            print(f'{N:>8}  {t_cpu:>9.2f}  {"N/A":>12}  {"—":>8}  {"N/A":>14}  {"—":>8}')
            continue

        # Warmup (first launch compiles PTX)
        brf_gpu(scene, obs_dirs, vec_sun, 64, seed=0)
        deriv_gpu(scene, obs_dirs, vec_sun, 64, seed=0)

        t0 = time.perf_counter()
        compute_brf_gpu(scene, obs_dirs, vec_sun, N, seed=0)
        t_brf = time.perf_counter() - t0

        t0 = time.perf_counter()
        compute_deriv_gpu(scene, obs_dirs, vec_sun, N, seed=0)
        t_drv = time.perf_counter() - t0

        print(
            f'{N:>8}  {t_cpu:>9.2f}  '
            f'{t_brf:>12.3f}  {t_cpu/t_brf:>7.1f}x  '
            f'{t_drv:>14.3f}  {t_cpu/t_drv:>7.1f}x'
        )

"""
Canopy RT — Main renderer.

Runs the photon loop, accumulates:
  - BRF per observation direction
  - 2-D image (top-view projection)
  - dBRF/dLAI per direction (fictive + edge + optimal-weighted)
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from .config import SceneConfig
from .scene import Scene, build_scene
from .geometry import polar_vect
from .path_tracer import (trace_photon, trace_backward_pixel,
                          trace_backward_pixel_with_deriv)
from .nee import NEEEngine
from .derivative import (
    compute_photon_derivative,
    optimal_weight,
)
from .intersection import warmup as jit_warmup


# ─────────────────────────────────────────────────────────────────────────────
#  Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RenderResult:
    # observation geometry
    obs_zenith_deg: np.ndarray     # (2*n,) signed zenith [°] — PP then CP
    obs_dirs:       np.ndarray     # (2*n, 3)
    n_pp:           int            # number of principal-plane directions

    # BRF
    brf:            np.ndarray     # (n_obs,)  π * <radiance>

    # 2-D image (top-down, looking upward from sensor)
    image:          np.ndarray     # (H, W) mean radiance

    # Derivatives dBRF/dLAI
    deriv_fic:      np.ndarray     # (n_obs,) fictive estimator
    deriv_edg:      np.ndarray     # (n_obs,) edge estimator
    deriv_opt:      np.ndarray     # (n_obs,) optimal weighted combination

    # Derivative dBRF/drho  (assumes rho=tau, score-function estimator: K*w^(K-1))
    deriv_rho:      np.ndarray     # (n_obs,)  mean_photons(K * lum_ph) / rho

    # convergence
    n_photons:      int
    opt_weights:    np.ndarray     # (n_obs,) optimal weights


# ─────────────────────────────────────────────────────────────────────────────
#  Build observation directions
# ─────────────────────────────────────────────────────────────────────────────

def _build_obs_dirs(cfg: SceneConfig):
    """
    Build observation direction vectors for principal plane AND cross-plane.

    Convention (matches MATLAB):
      Principal plane: phi=180 (backward, theta<0) or phi=0 (forward, theta>0)
      Cross-plane:     phi=90 (left) or phi=270 (right)

    Returns
    -------
    obs_zenith_deg : (2*n,)   signed zenith [deg] — first n=PP, last n=CP
    obs_dirs       : (2*n, 3) unit vectors pointing toward sensor (z > 0)
    n_pp           : int      number of principal-plane directions
    """
    deg  = np.pi / 180
    step = cfg.obs_step_deg * deg
    mx   = cfg.obs_zenith_max_deg * deg

    thetas = np.arange(-mx, mx + step / 2, step)
    abs_th = np.abs(thetas)

    # principal plane
    phi_pp  = np.where(thetas < 0, np.pi, 0.0)
    dirs_pp = polar_vect(abs_th, phi_pp)
    if dirs_pp.ndim == 1:
        dirs_pp = dirs_pp[np.newaxis, :]

    # cross-plane (perpendicular to solar plane, phi=90/270)
    phi_cp  = np.where(thetas < 0, 3*np.pi/2, np.pi/2)
    dirs_cp = polar_vect(abs_th, phi_cp)
    if dirs_cp.ndim == 1:
        dirs_cp = dirs_cp[np.newaxis, :]

    obs_dirs       = np.vstack([dirs_pp, dirs_cp])
    obs_zenith_deg = np.concatenate([thetas/deg, thetas/deg])
    n_pp           = len(thetas)

    return obs_zenith_deg, obs_dirs, n_pp


# ─────────────────────────────────────────────────────────────────────────────
#  Solid angles
# ─────────────────────────────────────────────────────────────────────────────

def _solar_solid_angle(sun_ang_deg: float = 0.532) -> float:
    a = sun_ang_deg / 2 * np.pi / 180
    return 2 * np.pi * (1 - np.cos(a))


def _sensor_solid_angle(radius_m: float = 0.5, altitude_m: float = 700_000) -> float:
    half_angle = np.arctan(radius_m / altitude_m)
    return 2 * np.pi * (1 - np.cos(half_angle))


# ─────────────────────────────────────────────────────────────────────────────
#  Main render loop
# ─────────────────────────────────────────────────────────────────────────────

def render(cfg: SceneConfig) -> RenderResult:
    rng = np.random.default_rng(cfg.seed)

    # ── build scene ──
    scene = build_scene(cfg, rng)

    # ── observation directions ──
    obs_deg, obs_dirs, n_pp = _build_obs_dirs(cfg)
    n_obs = len(obs_dirs)

    # ── solid angles ──
    D_Om_s = _solar_solid_angle()
    D_Om_o = _sensor_solid_angle()

    # ── JIT warmup (first call compiles Numba kernels, ~6s) ──────────────
    print("  [Numba JIT warmup...]", end="\r", flush=True)
    jit_warmup(scene)
    nee = NEEEngine(scene, edge_threshold=cfg.edge_threshold,
                    max_out=100, collect_deriv=cfg.compute_derivative)
    _le_tmp = np.ones(scene.n_leaf, dtype=np.int8)
    nee.warmup(obs_dirs, _le_tmp)
    print("  [JIT ready]         ", flush=True)

    # ── sun direction ──
    deg     = np.pi / 180
    tha_s   = (180 - cfg.sun_zenith_deg) * deg
    # phi_sun is the solar azimuth (direction FROM which sun shines).
    # The photon travels in the OPPOSITE horizontal direction, so add 180°.
    # This matches MATLAB convention: phi_s = 180° for sun_azimuth_deg = 0°.
    phi_s   = (cfg.sun_azimuth_deg + 180.0) * deg
    vec_sun = polar_vect(tha_s, phi_s)

    # ── accumulators ──
    img_H, img_W     = cfg.image_size
    image_accum      = np.zeros((img_H, img_W))
    image_count      = np.zeros((img_H, img_W), dtype=int)
    luminance        = np.zeros(n_obs)   # sum over photons

    # derivative accumulators  (n_photons, max_contacts, n_obs)
    max_c = cfg.max_depth + 5
    All_Deriv_fic = np.zeros((cfg.n_photons, max_c, n_obs))
    All_Deriv_edg = np.zeros((cfg.n_photons, max_c, n_obs))
    D_rho_ph      = np.zeros((cfg.n_photons, n_obs))    # dBRF/drho per photon

    # ── photon loop ──
    Lx, Ly = scene.pos_max[0], scene.pos_max[1]
    x0_frac = float(np.clip(getattr(cfg, "photon_start_xy_min_frac", 0.0), 0.0, 1.0))
    x1_frac = float(np.clip(getattr(cfg, "photon_start_xy_max_frac", 1.0), 0.0, 1.0))
    if x1_frac < x0_frac:
        x0_frac, x1_frac = x1_frac, x0_frac
    x0_min, x0_max = x0_frac * Lx, x1_frac * Lx
    y0_min, y0_max = x0_frac * Ly, x1_frac * Ly

    for n_pt in range(cfg.n_photons):
        # ── per-photon leaf randomisation (probabilistic rendering) ──
        if cfg.compute_derivative:
            # Probabilistic rendering: fictive leaves for derivative computation
            leaf_effect = (rng.random(size=scene.n_leaf) < scene.P_LAI).astype(np.int8)
        else:
            # BRF only: all leaves real. Scene was built with effective P_LAI,
            # so leaf_effect=all 1 uses the correct LAI_eff directly.
            leaf_effect = np.ones(scene.n_leaf, dtype=np.int8)
        if cfg.compute_derivative:
            scene.resample_radii(rng)   # re-draw leaf radii

        # ── random starting position on top face ──
        x0 = rng.uniform(x0_min, x0_max)
        y0 = rng.uniform(y0_min, y0_max)
        P  = np.array([x0, y0, 0.0])

        # ── trace ──
        ref_tran, contacts, n_contacts = trace_photon(
            scene      = scene,
            nee        = nee,
            P_start    = P,
            vec_sun    = vec_sun,
            obs_dirs   = obs_dirs,
            D_Om_s     = D_Om_s,
            D_Om_o     = D_Om_o,
            leaf_effect= leaf_effect,
            max_depth  = cfg.max_depth,
            energy_thresh = cfg.energy_threshold,
            edge_thresh= cfg.edge_threshold,
            rng        = rng,
            compute_deriv = cfg.compute_derivative,
            rr_depth      = cfg.rr_depth,
            rr_min        = cfg.rr_min,
        )

        # ── accumulate BRF ──
        for i_obs in range(n_obs):
            luminance[i_obs] += np.sum(ref_tran[:n_contacts, i_obs])

        # ── rho derivative: K * lum_ph / rho  (score-function estimator) ──
        if n_contacts > 0 and scene.rho > 0:
            K_ph = sum(1 for j in range(n_contacts) if contacts[j]['leaf_id'] >= 0)
            if K_ph > 0:
                lum_ph = np.sum(ref_tran[:n_contacts], axis=0)  # (n_obs,)
                D_rho_ph[n_pt] = K_ph * lum_ph / scene.rho

        # ── accumulate image (top-down view) ──
        # primary hit position
        if n_contacts > 0 and contacts[0]['leaf_id'] >= 0:
            leaf_idx = contacts[0]['leaf_id']
            lp = scene.leaf_pos[leaf_idx]
            ix = int(np.clip(lp[0] / Lx * img_W, 0, img_W - 1))
            iy = int(np.clip(lp[1] / Ly * img_H, 0, img_H - 1))
            image_accum[iy, ix] += np.sum(ref_tran[:n_contacts, :])  # total flux ce photon
            image_count[iy, ix] += 1

        # ── accumulate derivatives ──
        if cfg.compute_derivative and n_contacts > 0:
            d_fic, d_edg = compute_photon_derivative(
                photon_contacts   = contacts,
                ref_contributions = ref_tran,
                n_contacts        = n_contacts,
                LAI_r             = scene.LAI_r,
                std_LAI           = scene.std_LAI,
                LAI_Max           = scene.LAI_Max,
                P_LAI             = scene.P_LAI,
                n_leaf_norm       = scene.n_leaf_norm,
                n_obs             = n_obs,
            )
            nc = min(n_contacts, max_c)
            All_Deriv_fic[n_pt, :nc, :] = d_fic[:nc]
            All_Deriv_edg[n_pt, :nc, :] = d_edg[:nc]

    # ── normalise ──
    N = cfg.n_photons
    # NEE weights already contain 1/π from Lambertian BRDF (e.g. rsoil/π * cos_obs)
    # so: luminance/N = <rsoil/π * cos> for pure soil
    # and: π * luminance/N → BRF = rsoil * cos ≈ rsoil at nadir  ✓
    # We store raw (no π) and apply π at plot time, matching MATLAB convention
    brf = np.pi * luminance / N  # BRF = π × <L_NEE>  (Lambertian: L=ρ/π, BRF=π×L=ρ)

    # image
    with np.errstate(invalid='ignore'):
        image = np.where(image_count > 0, image_accum / image_count, 0.0)

    # derivatives: sum over contacts then average over photons
    D_fic_ph = np.sum(All_Deriv_fic, axis=1)   # (n_photons, n_obs)
    D_edg_ph = np.sum(All_Deriv_edg, axis=1)

    deriv_fic = np.mean(D_fic_ph, axis=0)  # (n_obs,)
    deriv_edg = np.mean(D_edg_ph, axis=0)

    w = optimal_weight(D_fic_ph, D_edg_ph)
    deriv_opt = w * deriv_fic + (1 - w) * deriv_edg
    deriv_rho = np.mean(D_rho_ph, axis=0)   # (n_obs,) — multiply by π at use site

    return RenderResult(
        obs_zenith_deg = obs_deg,
        obs_dirs       = obs_dirs,
        n_pp           = n_pp,
        brf            = brf,
        image          = image,
        deriv_fic      = deriv_fic,
        deriv_edg      = deriv_edg,
        deriv_opt      = deriv_opt,
        deriv_rho      = deriv_rho,
        n_photons      = N,
        opt_weights    = w,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Backward image renderer  (CPU, sensor-to-scene)
# ─────────────────────────────────────────────────────────────────────────────

def render_backward_image(
    cfg:             SceneConfig,
    obs_zenith_deg:  float = 0.0,
    obs_azimuth_deg: float = 0.0,
    img_H:           int   = 64,
    img_W:           int   = 64,
    N_spp:           int   = 1,
) -> np.ndarray:
    """
    Render a spatially-resolved image using backward (sensor→scene) ray tracing.

    For every pixel (i, j) in img_H × img_W, N_spp backward rays are fired
    from the sensor direction into the canopy.  At each scattering event,
    Next-Event Estimation (NEE) toward the sun accumulates radiance.

    Parameters
    ----------
    cfg             : SceneConfig  — scene + sun geometry (sun_zenith_deg, etc.)
    obs_zenith_deg  : float        — sensor zenith angle [°]  (0 = nadir)
    obs_azimuth_deg : float        — sensor azimuth angle [°]
    img_H, img_W    : int          — image height and width in pixels
    N_spp           : int          — Monte Carlo samples per pixel (1 = deterministic)

    Returns
    -------
    image : (img_H, img_W) float64  — π × mean luminance  (BRF-equivalent units)
    """
    rng   = np.random.default_rng(cfg.seed)
    scene = build_scene(cfg, rng)

    # ── observation direction (from scene toward sensor, upward) ──────────────
    deg     = np.pi / 180
    obs_dir = polar_vect(obs_zenith_deg * deg, obs_azimuth_deg * deg)
    ray_dir = -obs_dir                        # backward ray: into scene

    # ── sun direction ─────────────────────────────────────────────────────────
    tha_s      = (180 - cfg.sun_zenith_deg) * deg
    phi_s      = (cfg.sun_azimuth_deg + 180.0) * deg
    vec_sun    = polar_vect(tha_s, phi_s)     # sun→scene (downward)
    vec_sun_up = -vec_sun                     # scene→sun (upward, for NEE)

    # ── JIT warmup ───────────────────────────────────────────────────────────
    print("  [Numba JIT warmup...]", end="\r", flush=True)
    jit_warmup(scene)
    nee = NEEEngine(scene, edge_threshold=cfg.edge_threshold,
                    max_out=100, collect_deriv=False)
    nee.warmup(vec_sun_up.reshape(1, 3),
               np.ones(scene.n_leaf, dtype=np.int8))
    print("  [JIT ready]         ", flush=True)

    # ── pixel loop ───────────────────────────────────────────────────────────
    Lx, Ly        = scene.pos_max[0], scene.pos_max[1]
    image_accum   = np.zeros((img_H, img_W))
    total_pixels  = img_H * img_W * N_spp

    k = 0
    for spp in range(N_spp):
        for pi in range(img_H):
            for pj in range(img_W):

                # per-photon leaf realization
                if cfg.compute_derivative:
                    leaf_effect = (rng.random(size=scene.n_leaf)
                                   < scene.P_LAI).astype(np.int8)
                    scene.resample_radii(rng)
                else:
                    leaf_effect = np.ones(scene.n_leaf, dtype=np.int8)

                # pixel position with jitter
                jx = rng.random() if N_spp > 1 else 0.5
                jy = rng.random() if N_spp > 1 else 0.5
                x0 = (pj + jx) / img_W * Lx
                y0 = (pi + jy) / img_H * Ly
                P_start = np.array([x0, y0, 0.0])

                lum = trace_backward_pixel(
                    scene       = scene,
                    nee         = nee,
                    P_start     = P_start,
                    ray_dir     = ray_dir,
                    vec_sun_up  = vec_sun_up,
                    leaf_effect = leaf_effect,
                    max_depth   = cfg.max_depth,
                    energy_thresh = cfg.energy_threshold,
                    edge_thresh = cfg.edge_threshold,
                    rng         = rng,
                    rr_depth    = cfg.rr_depth,
                    rr_min      = cfg.rr_min,
                )
                image_accum[pi, pj] += lum

                k += 1
                if k % max(1, total_pixels // 20) == 0:
                    pct = 100 * k / total_pixels
                    print(f"  [{pct:5.1f}%]", end="\r", flush=True)

    print("  [done]              ", flush=True)
    return np.pi * image_accum / N_spp


# ─────────────────────────────────────────────────────────────────────────────
#  Backward image renderer WITH dBRF/dLAI  (CPU)
# ─────────────────────────────────────────────────────────────────────────────

def render_backward_image_deriv(
    cfg:             SceneConfig,
    obs_zenith_deg:  float = 0.0,
    obs_azimuth_deg: float = 0.0,
    img_H:           int   = 64,
    img_W:           int   = 64,
    N_spp:           int   = 1,
):
    """
    Backward image renderer that also computes dBRF/dLAI images.

    Each pixel fires N_spp backward rays from the sensor.  At every scattering
    event NEE toward the sun gives the radiance contribution; the same contact
    records are used to compute per-photon fictive and edge derivative estimates.
    The optimal-weight image uses a global variance-minimising weight estimated
    from all N_total = H×W×N_spp samples.

    Returns
    -------
    brf_image   : (img_H, img_W) — π × mean luminance
    dfic_image  : (img_H, img_W) — dBRF/dLAI fictive estimator
    dedg_image  : (img_H, img_W) — dBRF/dLAI edge estimator
    dopt_image  : (img_H, img_W) — dBRF/dLAI optimal weighted
    """
    rng   = np.random.default_rng(cfg.seed)
    scene = build_scene(cfg, rng)

    deg        = np.pi / 180
    obs_dir    = polar_vect(obs_zenith_deg * deg, obs_azimuth_deg * deg)
    ray_dir    = -obs_dir
    tha_s      = (180 - cfg.sun_zenith_deg) * deg
    phi_s      = (cfg.sun_azimuth_deg + 180.0) * deg
    vec_sun    = polar_vect(tha_s, phi_s)
    vec_sun_up = -vec_sun

    print("  [Numba JIT warmup...]", end="\r", flush=True)
    jit_warmup(scene)
    nee = NEEEngine(scene, edge_threshold=cfg.edge_threshold,
                    max_out=100, collect_deriv=True)
    nee.warmup(vec_sun_up.reshape(1, 3),
               np.ones(scene.n_leaf, dtype=np.int8))
    print("  [JIT ready]         ", flush=True)

    Lx, Ly      = scene.pos_max[0], scene.pos_max[1]
    n_obs       = 1          # single NEE direction: sun
    max_c       = cfg.max_depth + 5
    total       = img_H * img_W * N_spp

    lum_all   = np.zeros((total,))
    dfic_all  = np.zeros((total,))
    dedg_all  = np.zeros((total,))

    k = 0
    for spp in range(N_spp):
        for pi in range(img_H):
            for pj in range(img_W):
                leaf_effect = (rng.random(size=scene.n_leaf)
                               < scene.P_LAI).astype(np.int8)
                scene.resample_radii(rng)

                jx = rng.random() if N_spp > 1 else 0.5
                jy = rng.random() if N_spp > 1 else 0.5
                x0 = (pj + jx) / img_W * Lx
                y0 = (pi + jy) / img_H * Ly
                P_start = np.array([x0, y0, 0.0])

                ref_tran, contacts, n_contacts = trace_backward_pixel_with_deriv(
                    scene, nee, P_start, ray_dir, vec_sun_up,
                    leaf_effect, cfg.max_depth, cfg.energy_threshold,
                    cfg.edge_threshold, rng, cfg.rr_depth, cfg.rr_min,
                )

                # linear index: spp*H*W + pi*W + pj
                idx = spp * img_H * img_W + pi * img_W + pj
                lum_all[idx] = np.sum(ref_tran[:n_contacts, 0])

                if n_contacts > 0:
                    d_fic, d_edg = compute_photon_derivative(
                        photon_contacts   = contacts,
                        ref_contributions = ref_tran,
                        n_contacts        = n_contacts,
                        LAI_r             = scene.LAI_r,
                        std_LAI           = scene.std_LAI,
                        LAI_Max           = scene.LAI_Max,
                        P_LAI             = scene.P_LAI,
                        n_leaf_norm       = scene.n_leaf_norm,
                        n_obs             = n_obs,
                    )
                    dfic_all[idx] = np.sum(d_fic[:n_contacts, 0])
                    dedg_all[idx] = np.sum(d_edg[:n_contacts, 0])

                k += 1
                if k % max(1, total // 20) == 0:
                    print(f"  [{100*k/total:5.1f}%]", end="\r", flush=True)

    print("  [done]              ", flush=True)

    # Reshape to (N_spp, img_H, img_W) then average over samples
    lum_3d  = lum_all.reshape (N_spp, img_H, img_W)
    dfic_3d = dfic_all.reshape(N_spp, img_H, img_W)
    dedg_3d = dedg_all.reshape(N_spp, img_H, img_W)

    brf_image  = np.pi * lum_3d.mean(axis=0)
    dfic_image = dfic_3d.mean(axis=0)
    dedg_image = dedg_3d.mean(axis=0)

    # Global optimal weight (estimated from all N_total samples × 1 direction)
    w = optimal_weight(
        dfic_all.reshape(-1, 1),   # (N_total, 1)
        dedg_all.reshape(-1, 1),
    )
    dopt_image = w[0] * dfic_image + (1.0 - w[0]) * dedg_image

    return brf_image, dfic_image, dedg_image, dopt_image

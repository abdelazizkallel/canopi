"""
lai_all_deriv_inversion_jbs_image.py
======================================
Per-species LAI inversion for ALL 7 JBS species using the full Jacobian.

Algorithm: joint Gauss-Newton + exponential line search
  At each iteration:
    1. Render ONCE → BRF image + 7 derivative images dopt[s] simultaneously
    2. Build J (H*W × N_SP):  J[:,s] = π·LAI_Max_s × dopt[s].ravel() = d(BRF_px)/d(P_s)
    3. r = BRF - I_obs   (H*W residual)
    4. LM step in LAI_eff space:
         Δ(LAI_eff) = -(J^T J + λ diag(J^T J))^{-1} J^T r    (N_SP vector)
         ΔP[s]      = Δ(LAI_eff[s]) / LAI_Max_sp[s]           (convert to P space)
    5. Exponential line search (k = k_max → 0): try amp = 2^k × ΔP
       Accept the LARGEST amp that reduces ||BRF(P+ΔP) - I_obs||².  Costs are
       compared under common random numbers (same seed), so the difference
       reflects the step and not the sampling noise.
    6. Update all P_sp and repeat.

One render per iteration (+ k_max+1 cost renders during line search).
Far fewer renders than coordinate-descent GSS which needed
N_species × n_gss renders per cycle.

With same seed + same N_spp: f(P_sp = P_true) = 0 exactly (deterministic zero).
Reusing the seed is not needed for correctness — it is what lets the line search
converge at a handful of rays per pixel instead of averaging the noise away.
The concave BRF(LAI) makes the GN step undershoot → amplification > 1 is typical.
"""
import sys, os, time, argparse, gc
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from canopi.scenes.jbs_species_loader import (
    load_jbs_scene_species, get_band_params, ILLUMINATION, SPECIES_LIST, N_SPECIES,
)
from canopi.geometry import polar_vect
from canopi.paths import scene_dir as _resolve_scene_dir
from numba import cuda
from canopi.gpu import GPU_AVAILABLE, render_image_bwd_deriv_species_gpu

# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--band',       default='PUR')
parser.add_argument('--month',      default='Jul')
parser.add_argument('--max-leaves', type=int,   default=None)
parser.add_argument('--img-size',   type=int,   default=64)
parser.add_argument('--n-spp',      type=int,   default=8,
    help='Samples/pixel — same for obs and forward, so costs stay comparable at a low ray count')
parser.add_argument('--max-iter',   type=int,   default=20)
parser.add_argument('--p-lai',      type=float, default=0.5,
    help='True P_LAI for all species')
parser.add_argument('--init-p-lai', type=float, default=0.25,
    help='Initial P_LAI guess for all species (default = half of true)')
parser.add_argument('--init-fracs', type=float, nargs='+', default=None,
    help='Per-species init fractions of true LAI, length 7 '
         '(e.g. --init-fracs 0.2 0.25 0.6 0.4 0.5 0.33 0.3)')
parser.add_argument('--lm-lambda',  type=float, default=0.05,
    help='Levenberg-Marquardt damping (fraction of diag(J^T J))')
parser.add_argument('--k-max',      type=int,   default=9,
    help='Max exponent for line search: try 2^k_max ... 2^0 * ΔP')
parser.add_argument('--seed',       type=int,   default=42)
args = parser.parse_args()

if not GPU_AVAILABLE:
    raise RuntimeError('GPU required.')

P_LAI_TRUE = args.p_lai
if args.init_fracs is not None:
    assert len(args.init_fracs) == N_SPECIES, \
        f'Need {N_SPECIES} fracs, got {len(args.init_fracs)}'
    _INIT_FRACS = np.array(args.init_fracs, dtype=float)
else:
    _INIT_FRACS = np.full(N_SPECIES, args.init_p_lai / P_LAI_TRUE)

H = W = args.img_size
deg = np.pi / 180

illum     = ILLUMINATION[args.month]
SZA       = illum['sun_zenith_deg'];  SAA = illum['sun_azimuth_deg']
vec_sun   = polar_vect((180.0 - SZA)*deg, (SAA + 180.0)*deg)
obs_nadir = np.array([0.0, 0.0, 1.0])

print(f'\n{"="*68}')
print(f'  JBS All-Species LAI Inversion  [joint Jacobian Gauss-Newton]')
print(f'{"="*68}')
print(f'  Band={args.band}  Image={H}x{W}  N_spp={args.n_spp}  '
      f'max_iter={args.max_iter}  λ={args.lm_lambda}  k_max={args.k_max}')
print(f'  P_LAI_true={P_LAI_TRUE}  P_LAI_init={args.init_p_lai}')

# ─────────────────────────────────────────────────────────────────────────────
# [1] Load scene (fixed throughout)
# ─────────────────────────────────────────────────────────────────────────────
print('\n[1] Loading true JBS scene ...')
t0 = time.time()
scene, wood, leaf_sid, sp_stats = load_jbs_scene_species(
    scene_dir=_resolve_scene_dir(), band=args.band, month=args.month,
    P_LAI=P_LAI_TRUE, leaf_radius_std_mul=0.0,
    max_leaves=args.max_leaves, seed=args.seed, verbose=True,
)
scene.std_LAI = 0.0
bp = get_band_params(args.band)
scene.rsoil = float(bp['rsoil'])
print(f'  Loaded in {time.time()-t0:.1f}s  n_leaf={scene.n_leaf:,}')

LAI_Max_sp   = sp_stats['LAI_Max_sp']       # (N_SP,) fixed
LAI_eff_true = LAI_Max_sp * P_LAI_TRUE      # (N_SP,) true effective LAI

print('\n  True / init parameters:')
print(f'  {"Sp":<6}  {"LAI_Max":>9}  {"LAI_true":>10}  {"P_true":>8}  '
      f'{"P_init":>8}  {"LAI_init":>10}')
for s, sp in enumerate(SPECIES_LIST):
    p_init_s = float(np.clip(_INIT_FRACS[s] * P_LAI_TRUE, 1e-4, 1 - 1e-4))
    print(f'  {sp:<6}  {LAI_Max_sp[s]:9.5f}  {LAI_eff_true[s]:10.5f}  '
          f'{P_LAI_TRUE:8.4f}  {p_init_s:8.4f}  {LAI_Max_sp[s]*p_init_s:10.5f}')

# ─────────────────────────────────────────────────────────────────────────────
# [2] Render true observations  (same seed + N_spp as all forward renders)
# ─────────────────────────────────────────────────────────────────────────────
print(f'\n[2] Rendering observations ({H}x{W}, N_spp={args.n_spp}) ...')
t0 = time.time()
I_obs, _, _, _, _ = render_image_bwd_deriv_species_gpu(
    scene, obs_nadir, vec_sun, leaf_sid, sp_stats,
    img_H=H, img_W=W, N_spp=args.n_spp,
    max_depth=50, seed=args.seed,
    P_LAI_sp=np.full(N_SPECIES, P_LAI_TRUE, dtype=np.float64),
)
print(f'  Done in {time.time()-t0:.1f}s   BRF [{I_obs.min():.4f}, {I_obs.max():.4f}]')
gc.collect(); cuda.current_context().deallocations.clear()

# ─────────────────────────────────────────────────────────────────────────────
# Render helpers
# ─────────────────────────────────────────────────────────────────────────────
def render_forward(p_sp_arr):
    """Render BRF + all-species dopt images. Returns (brf_image, dopt_images)."""
    gc.collect(); cuda.current_context().deallocations.clear()
    Ik, _, _, dopt, _ = render_image_bwd_deriv_species_gpu(
        scene, obs_nadir, vec_sun, leaf_sid, sp_stats,
        img_H=H, img_W=W, N_spp=args.n_spp,
        max_depth=50, seed=args.seed,
        P_LAI_sp=p_sp_arr.astype(np.float64),
    )
    return Ik, dopt


def render_cost(p_sp_arr):
    """Cost-only evaluation (derivative output discarded)."""
    Ik, _ = render_forward(p_sp_arr)
    r = Ik.ravel() - I_obs.ravel()
    return 0.5 * float(np.dot(r, r)), float(np.sqrt(np.mean(r**2))), Ik

# ─────────────────────────────────────────────────────────────────────────────
# [3] Joint Gauss-Newton + exponential line search
# ─────────────────────────────────────────────────────────────────────────────
p_sp = np.clip(_INIT_FRACS * P_LAI_TRUE, 1e-4, 1.0 - 1e-4).astype(np.float64)

print(f'\n[3] Joint Gauss-Newton  (λ={args.lm_lambda}  k_max={args.k_max}  '
      f'max_iter={args.max_iter})')
print(f'  {"it":>3}  {"RMSE":>10}  {"max_err%":>9}  {"||ΔP||∞":>9}  '
      f'{"amp":>7}  t[s]')
print(f'  {"-"*55}')

p_hist    = [p_sp.copy()]
err_hist  = []
rmse_hist = []
cost_hist = []

for it in range(args.max_iter):
    t_it = time.time()

    # ── Step A: one render → BRF + full Jacobian ─────────────────────────────
    Ik, dopt_images = render_forward(p_sp)
    r    = (Ik - I_obs).ravel()              # (H*W,)
    cost = 0.5 * float(np.dot(r, r))
    rmse = float(np.sqrt(np.mean(r**2)))
    rmse_hist.append(rmse)
    cost_hist.append(cost)

    # ── Step B: build J (H*W × N_SP) w.r.t. P  ──────────────────────────────
    # dBRF/dP_s = dBRF/dLAI_eff_s × dLAI_eff_s/dP_s = π·dopt[s] · LAI_Max_s
    J   = np.stack([np.pi * LAI_Max_sp[s] * dopt_images[s].ravel()
                    for s in range(N_SPECIES)], axis=1)   # d(BRF_pixel)/d(P_s)
    JTJ = J.T @ J          # (N_SP × N_SP)
    JTr = J.T @ r          # (N_SP,)

    # ── Step C: LM-regularised step directly in P space ──────────────────────
    lm = args.lm_lambda * np.diag(JTJ)
    try:
        dP = -np.linalg.solve(JTJ + np.diag(lm), JTr)   # (N_SP,)  step in P space
    except np.linalg.LinAlgError:
        print('  Singular J^T J — stopping'); break

    err_arr = (np.abs(p_sp * LAI_Max_sp - LAI_eff_true)
               / np.maximum(LAI_eff_true, 1e-6) * 100)
    err_hist.append(err_arr)
    max_err = float(err_arr.max())

    # ── Step D: ASCENDING line search  (k = 0 → k_max) ─────────────────────
    # Start at Newton step (amp=1), double while cost keeps decreasing.
    # Stop at first cost increase → no overshoot past minimum.
    # Same scalar amp for all species → Jacobian direction is preserved.
    # Cap amp_max so no species clips the boundary.
    with np.errstate(divide='ignore', invalid='ignore'):
        max_up  = np.where(dP > 0, (1.0 - 1e-4 - p_sp) / dP, np.inf)
        max_dn  = np.where(dP < 0, (1e-4      - p_sp) / dP, np.inf)
    amp_max = float(np.min(np.minimum(max_up, max_dn)))

    best_cost = cost
    best_p    = p_sp.copy()
    best_amp  = 0.0

    for k in range(0, args.k_max + 1):
        amp    = min(2.0 ** k, amp_max)
        p_cand = p_sp + amp * dP
        if np.max(np.abs(p_cand - p_sp)) < 1e-7:
            break
        c_cand, _, _ = render_cost(p_cand)
        if c_cand < best_cost:
            best_cost = c_cand
            best_p    = p_cand
            best_amp  = amp
        else:
            break   # cost started increasing → overshot, keep previous best

    elapsed = time.time() - t_it
    print(f'  {it:>3}  {rmse:>10.4e}  {max_err:>9.2f}%  '
          f'{np.max(np.abs(dP)):>9.5f}  x{best_amp:>5.1f}  {elapsed:.1f}s')

    # per-species detail
    for s, sp in enumerate(SPECIES_LIST):
        sign = '+' if dP[s] >= 0 else '-'
        print(f'       {sp:<6}  P:{p_sp[s]:.4f}->{best_p[s]:.4f}  '
              f'LAI:{LAI_Max_sp[s]*p_sp[s]:.4f}->{LAI_Max_sp[s]*best_p[s]:.4f}  '
              f'err:{err_arr[s]:+.1f}%  dP:{sign}{abs(dP[s]):.5f}')

    p_sp = best_p
    p_hist.append(p_sp.copy())

    if best_amp == 0:
        print('  No improving step — converged.'); break
    if np.max(np.abs(dP)) < 1e-5:
        print(f'  Step tiny (||ΔP||∞={np.max(np.abs(dP)):.2e}) — converged.'); break

# ─────────────────────────────────────────────────────────────────────────────
# Final summary
# ─────────────────────────────────────────────────────────────────────────────
print('\n  Final retrieved parameters:')
print(f'  {"Sp":<6}  {"P_init":>8}  {"P_final":>8}  {"P_true":>8}  '
      f'{"LAI_init":>10}  {"LAI_final":>10}  {"LAI_true":>10}  {"err%":>8}')
print(f'  {"-"*82}')
for s, sp in enumerate(SPECIES_LIST):
    p_init_s  = float(np.clip(_INIT_FRACS[s] * P_LAI_TRUE, 1e-4, 1 - 1e-4))
    lai_init  = LAI_Max_sp[s] * p_init_s
    lai_final = LAI_Max_sp[s] * p_sp[s]
    err_pct   = 100.0 * (lai_final - LAI_eff_true[s]) / max(LAI_eff_true[s], 1e-6)
    print(f'  {sp:<6}  {p_init_s:8.4f}  {p_sp[s]:8.5f}  {P_LAI_TRUE:8.4f}  '
          f'{lai_init:10.5f}  {lai_final:10.5f}  {LAI_eff_true[s]:10.5f}  '
          f'{err_pct:+8.2f}%')

# ─────────────────────────────────────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────────────────────────────────────
SP_COLOR = {
    'BEPE': '#2ca02c', 'TICO': '#1f77b4', 'ALGL': '#ff7f0e',
    'POTR': '#9467bd', 'PIAB': '#8c564b', 'FREX': '#e377c2', 'ACPL': '#7f7f7f',
}
colors   = [SP_COLOR[sp] for sp in SPECIES_LIST]
p_hist   = np.array(p_hist)          # (n_iters+1, N_SP)
err_hist = np.array(err_hist) if err_hist else np.zeros((0, N_SPECIES))
n_done   = len(rmse_hist)
x_pts    = np.arange(len(p_hist))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(
    f'All-Species LAI Inversion [{args.band}]  joint Jacobian GN  '
    f'N_spp={args.n_spp}  λ={args.lm_lambda}  k_max={args.k_max}',
    fontsize=10)

# (a) LAI convergence per species
ax = axes[0]
for s, sp in enumerate(SPECIES_LIST):
    lai_vals = p_hist[:, s] * LAI_Max_sp[s]
    ax.plot(x_pts, lai_vals, '-o', color=colors[s], lw=2, ms=6, label=sp)
    ax.axhline(LAI_eff_true[s], ls='--', lw=1, color=colors[s], alpha=0.5)
ax.set_xlabel('Iteration'); ax.set_ylabel('LAI_eff')
ax.set_title('(a) Per-species LAI  (dashed = true)')
ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
ax.set_ylim(bottom=0)

# (b) Error % per species (log scale)
ax = axes[1]
if len(err_hist) > 0:
    for s, sp in enumerate(SPECIES_LIST):
        ax.semilogy(np.arange(n_done), np.maximum(err_hist[:, s], 0.001),
                    '-o', color=colors[s], lw=2, ms=6, label=sp)
ax.axhline(1.0, ls=':', lw=1, color='gray', alpha=0.7, label='1%')
ax.set_xlabel('Iteration'); ax.set_ylabel('|LAI error| %')
ax.set_title('(b) Per-species LAI error %')
ax.legend(fontsize=7, ncol=2); ax.grid(True, which='both', alpha=0.3)

# (c) Initial BRF | Observed BRF | Final residual
_, _, Ik_init  = render_cost(p_hist[0])
_, _, Ik_final = render_cost(p_sp)
Ik_init  = Ik_init.reshape(H, W)
Ik_final = Ik_final.reshape(H, W)

vmin_brf = min(I_obs.min(), Ik_init.min())
vmax_brf = max(I_obs.max(), Ik_init.max())

gs_c = fig.add_gridspec(1, 3, left=0.67, right=0.98, wspace=0.05)
ax1 = fig.add_subplot(gs_c[0, 0])
ax2 = fig.add_subplot(gs_c[0, 1])
ax3 = fig.add_subplot(gs_c[0, 2])

ax1.imshow(Ik_init, origin='lower', cmap='YlGn',
           vmin=vmin_brf, vmax=vmax_brf, interpolation='nearest')
ax1.set_title('Initial BRF', fontsize=8); ax1.set_xticks([]); ax1.set_yticks([])

ax2.imshow(I_obs, origin='lower', cmap='YlGn',
           vmin=vmin_brf, vmax=vmax_brf, interpolation='nearest')
ax2.set_title('Observed BRF', fontsize=8); ax2.set_xticks([]); ax2.set_yticks([])

res = Ik_final - I_obs; ar = max(abs(res).max(), 1e-6)
im3 = ax3.imshow(res, origin='lower', cmap='RdBu_r', vmin=-ar, vmax=ar, interpolation='nearest')
ax3.set_title('Final residual', fontsize=8); ax3.set_xticks([]); ax3.set_yticks([])
fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04, format='%.3f')
axes[2].set_visible(False)

plt.tight_layout()
out = 'lai_all_deriv_inversion_jbs_image.png'
plt.savefig(out, dpi=140, bbox_inches='tight')
print(f'\nFigure -> {out}')
print('Done.')

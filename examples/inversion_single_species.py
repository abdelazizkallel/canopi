"""
lai_BEPE_inversion_jbs_image.py
================================
Single-species LAI inversion for BEPE using per-species P_LAI.

Key insight: keep the FULL TRUE SCENE fixed. Vary BEPE effective LAI by
changing P_LAI_BEPE (the per-leaf activation probability for BEPE leaves).
Because _leaf_real uses a deterministic hash(photon_id, leaf_id, seed),
  - {real at P=0.25} is a SUBSET of {real at P=0.50}
  - The leaf sets are perfectly NESTED as LAI varies
  - No spatial pattern mismatch → clean residual signal

Parameter:  LAI_eff_BEPE = LAI_Max_sp_BEPE × P_LAI_BEPE
Jacobian:   j[px] = pi/P_LAI_BEPE × dfic_BEPE(px)   (scalar column)
LM update:  delta = -j^T r / ((1+lm_lambda) * j^T j)

Output: lai_BEPE_inversion_jbs_image.png
"""
import sys, os, time, argparse, gc, math
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from canopi.scenes.jbs_species_loader import (
    load_jbs_scene_species, build_scene_for_lai,
    get_band_params, ILLUMINATION, SPECIES_LIST, N_SPECIES,
)
from canopi.geometry import polar_vect
from canopi.paths import scene_dir as _resolve_scene_dir
from numba import cuda
from canopi.gpu import GPU_AVAILABLE, render_image_bwd_deriv_species_gpu

parser = argparse.ArgumentParser()
parser.add_argument('--band',       default='PUR')
parser.add_argument('--month',      default='Jul')
parser.add_argument('--max-leaves', type=int,   default=None)
parser.add_argument('--img-size',   type=int,   default=64)
parser.add_argument('--n-spp-obs',  type=int,   default=32)
parser.add_argument('--n-spp',      type=int,   default=8)
parser.add_argument('--max-iter',   type=int,   default=15)
parser.add_argument('--p-lai',      type=float, default=0.5)
parser.add_argument('--lm-lambda',  type=float, default=0.1)
parser.add_argument('--seed',       type=int,   default=42)
args = parser.parse_args()

if not GPU_AVAILABLE:
    raise RuntimeError('GPU required.')

P_LAI_TRUE = args.p_lai       # true activation probability for all species
H = W = args.img_size
deg      = np.pi / 180
BEPE_IDX = SPECIES_LIST.index('BEPE')

illum     = ILLUMINATION[args.month]
SZA       = illum['sun_zenith_deg'];  SAA = illum['sun_azimuth_deg']
vec_sun   = polar_vect((180.0 - SZA)*deg, (SAA + 180.0)*deg)
obs_nadir = np.array([0.0, 0.0, 1.0])

print(f'\n{"="*62}')
print(f'  JBS BEPE-only LAI Inversion  [per-species P_LAI]')
print(f'{"="*62}')
print(f'  Band={args.band}  Image={H}x{W}  N_spp={args.n_spp}  N_spp_obs={args.n_spp_obs}')

# [1] Load FULL JBS scene (fixed for all iterations)
print('\n[1] Loading true JBS scene ...')
t0 = time.time()
scene_true, wood, leaf_sid_true, sp_stats_true = load_jbs_scene_species(
    scene_dir=_resolve_scene_dir(), band=args.band, month=args.month,
    P_LAI=P_LAI_TRUE, leaf_radius_std_mul=0.0,
    max_leaves=args.max_leaves, seed=args.seed, verbose=True,
)
scene_true.std_LAI = 0.0
bp = get_band_params(args.band)
scene_true.rsoil = float(bp['rsoil'])
print(f'  Loaded in {time.time()-t0:.1f}s  n_leaf={scene_true.n_leaf:,}')

LAI_Max_sp_true = sp_stats_true['LAI_Max_sp']   # (7,)  fixed from true scene
LAI_eff_true    = LAI_Max_sp_true * P_LAI_TRUE   # (7,)  true effective LAI

print(f'\n  BEPE: LAI_Max={LAI_Max_sp_true[BEPE_IDX]:.4f}  '
      f'LAI_eff_true={LAI_eff_true[BEPE_IDX]:.4f}')

# [2] Render true observations at P_LAI_BEPE = P_LAI_TRUE (full LAI)
p_lai_sp_true = np.full(N_SPECIES, P_LAI_TRUE, dtype=np.float64)
print(f'\n[2] Rendering true observations ({H}x{W}, N_spp={args.n_spp_obs}) ...')
t0 = time.time()
I_obs, _, _, _, _ = render_image_bwd_deriv_species_gpu(
    scene_true, obs_nadir, vec_sun,
    leaf_sid_true, sp_stats_true,
    img_H=H, img_W=W, N_spp=args.n_spp,   # same N_spp → same photon IDs → nested hash
    max_depth=50, seed=args.seed,
    P_LAI_sp=p_lai_sp_true,
)
print(f'  Done in {time.time()-t0:.1f}s   BRF [{I_obs.min():.4f}, {I_obs.max():.4f}]')

gc.collect(); cuda.current_context().deallocations.clear()

# [3] Inversion: vary P_LAI_BEPE, all other species fixed at P_LAI_TRUE
P_LAI_bepe_init = 0.5 * P_LAI_TRUE    # start at half -> LAI_eff = half true
LAI_bepe_true   = LAI_eff_true[BEPE_IDX]
LAI_Max_bepe    = LAI_Max_sp_true[BEPE_IDX]

print(f'\n  BEPE init: P_LAI_BEPE={P_LAI_bepe_init:.4f}  '
      f'LAI_eff_init={LAI_Max_bepe*P_LAI_bepe_init:.4f}  ')
print(f'  BEPE true: P_LAI_BEPE={P_LAI_TRUE:.4f}  '
      f'LAI_eff_true={LAI_bepe_true:.4f}')

# ── Golden Section Search on P_LAI_BEPE ─────────────────────────────────────
# With same seed + same N_spp: f(P_bepe=P_true) = 0 exactly (deterministic).
# The shared seed is a variance reduction, not a requirement: it keeps the cost
# comparable between evaluations at a low ray count.
# The cost function is 1D and unimodal. GSS finds minimum in O(log(1/eps)) evals.
# At each step we evaluate f at two interior points and shrink the bracket.

def _cost(p_val):
    """Evaluate ||BRF(p_val) - BRF_obs||² using the fixed scene and same seed."""
    gc.collect(); cuda.current_context().deallocations.clear()
    p_sp = np.full(N_SPECIES, P_LAI_TRUE, dtype=np.float64)
    p_sp[BEPE_IDX] = p_val
    Ik, _, _, _, _ = render_image_bwd_deriv_species_gpu(
        scene_true, obs_nadir, vec_sun,
        leaf_sid_true, sp_stats_true,
        img_H=H, img_W=W, N_spp=args.n_spp,
        max_depth=50, seed=args.seed,
        P_LAI_sp=p_sp,
    )
    r = Ik.ravel() - I_obs.ravel()
    return 0.5 * float(np.dot(r, r)), float(np.sqrt(np.mean(r**2))), Ik

GOLD = (math.sqrt(5) - 1) / 2   # ≈ 0.618

lo, hi = 1e-4, 1.0 - 1e-4       # search bracket for P_LAI_BEPE
print(f'\n[3] Golden Section Search  bracket=[{lo:.4f}, {hi:.4f}]')
print(f'  {"it":>3}  {"lo":>8}  {"hi":>8}  {"P_BEPE":>8}  {"LAI_BEPE":>10}  {"error%":>8}  {"RMSE":>10}  t[s]')
print(f'  {"-"*75}')

lai_hist  = [LAI_Max_bepe * P_LAI_bepe_init]
rmse_hist = []
cost_hist = []
last_Ik   = None

# Evaluate at two initial interior points
p1 = hi - GOLD * (hi - lo)
p2 = lo + GOLD * (hi - lo)
t0 = time.time()
f1, rmse1, _ = _cost(p1)
f2, rmse2, Ik2 = _cost(p2)
print(f'  init  lo={lo:.4f}  hi={hi:.4f}  f(p1={p1:.4f})={f1:.3e}  f(p2={p2:.4f})={f2:.3e}  {time.time()-t0:.1f}s')

for it in range(args.max_iter):
    t_it = time.time()
    if f1 > f2:
        lo = p1
        p1, f1 = p2, f2
        p2 = lo + GOLD * (hi - lo)
        f2, rmse2, last_Ik = _cost(p2)
        p_best, rmse_best = p2, rmse2
    else:
        hi = p2
        p2, f2 = p1, f1
        p1 = hi - GOLD * (hi - lo)
        f1, rmse1, last_Ik = _cost(p1)
        p_best, rmse_best = p1, rmse1

    p_mid    = 0.5 * (lo + hi)
    lai_mid  = LAI_Max_bepe * p_mid
    err_pct  = 100.0 * (lai_mid - LAI_bepe_true) / LAI_bepe_true
    cost_mid = min(f1, f2)
    cost_hist.append(cost_mid); rmse_hist.append(rmse_best)
    lai_hist.append(lai_mid)
    elapsed  = time.time() - t_it
    print(f'  {it:>3}  {lo:>8.5f}  {hi:>8.5f}  {p_best:>8.5f}  {lai_mid:>10.5f}  {err_pct:>+8.2f}%  {rmse_best:>10.4e}  {elapsed:.1f}s')
    if hi - lo < 1e-4:
        print('  Bracket converged.'); break

n_done = len(cost_hist)
p_final   = 0.5 * (lo + hi)
lai_final = LAI_Max_bepe * p_final
err_final = 100.0 * (lai_final - LAI_bepe_true) / LAI_bepe_true
print(f'\n  Final  LAI_BEPE={lai_final:.5f}  (true={LAI_bepe_true:.5f}  error={err_final:+.2f}%)')

# Figure
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle(f'BEPE LAI Inversion [{args.band}] per-species P_LAI  '
             f'true={LAI_bepe_true:.4f}  init={LAI_Max_bepe*P_LAI_bepe_init:.4f}  '
             f'final={lai_final:.4f} ({err_final:+.1f}%)  N_spp={args.n_spp}',
             fontsize=10)
iters = np.arange(len(lai_hist))
ax = axes[0]
ax.plot(iters, lai_hist, 'b-o', lw=2, ms=6)
ax.axhline(LAI_bepe_true, color='red', ls='--', lw=2, label=f'true={LAI_bepe_true:.4f}')
ax.axhline(LAI_Max_bepe*P_LAI_bepe_init, color='orange', ls=':', lw=1.5)
ax.fill_between(iters, LAI_bepe_true*0.95, LAI_bepe_true*1.05, color='red', alpha=0.1)
ax.set_xlabel('Iteration'); ax.set_ylabel('LAI_BEPE')
ax.set_title('(a) BEPE LAI convergence'); ax.legend(fontsize=7); ax.grid(alpha=0.3)
ax = axes[1]
ax.semilogy(range(n_done), rmse_hist, 'b-o', lw=2, ms=5)
ax.set_xlabel('Iteration'); ax.set_ylabel('RMSE')
ax.set_title('(b) Image RMSE'); ax.grid(True, which='both', alpha=0.3)
plt.tight_layout()
out = 'lai_BEPE_inversion_jbs_image.png'
plt.savefig(out, dpi=140, bbox_inches='tight')
print(f'\nFigure -> {out}')
print('Done.')

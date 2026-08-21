"""
jbs_fd_check.py
===============
Finite-difference check of the per-species dBRF/dLAI derivative at
SZA=41°, nadir observer (JBS scene, band O08).

Key insight
-----------
Loading at different P_LAI values gives n_leaf ∝ 1/P_LAI, keeping
LAI_eff = P_LAI × LAI_Max CONSTANT → useless as FD.

Correct FD: load scene ONCE (fixed leaf geometry), then vary P_LAI_sp
(the acceptance probability) between renders via the optional P_LAI_sp
argument of render_image_bwd_deriv_species_gpu. This genuinely changes
LAI_eff while keeping all leaf positions identical.

Global check
------------
Perturb all species together (scalar P_LAI → all P_LAI_sp[s] = P±δ):

  FD   = [ BRF(P+δ) − BRF(P−δ) ] / (2δ)   ← dBRF / dP_LAI

Analytical counterpart (from reference render at P):
  AN   = Σ_s  π · dopt[s].mean() · LAI_Max_sp[s]

Because dopt[s] = (1/π) · d(BRF)/d(LAI_eff_s)  and  LAI_eff_s = P · LAI_Max_sp[s].

Usage:
    python jbs_fd_check.py [--delta 0.02] [--res 64] [--n-spp 256] [--seed 42]
"""
import sys, os, argparse, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from canopi.scenes.jbs_species_loader import (
    load_jbs_scene_species, get_band_params, ILLUMINATION,
    SPECIES_LIST, N_SPECIES,
)
from canopi.geometry import polar_vect
from canopi.paths import scene_dir as _resolve_scene_dir
from canopi.gpu import render_image_bwd_deriv_species_gpu

# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--delta',  type=float, default=0.02,
    help='P_LAI perturbation (default 0.02)')
parser.add_argument('--p-lai',  type=float, default=0.5)
parser.add_argument('--res',    type=int,   default=64)
parser.add_argument('--n-spp',  type=int,   default=256)
parser.add_argument('--seed',   type=int,   default=42)
args = parser.parse_args()

BAND  = 'O08'
MONTH = 'Jul'
DEG   = np.pi / 180.0

illum   = ILLUMINATION[MONTH]
SZA     = illum['sun_zenith_deg']
SAA     = illum['sun_azimuth_deg']
vec_sun = polar_vect((180.0 - SZA) * DEG, (SAA + 180.0) * DEG)
obs_nad = polar_vect(0.0, 0.0)

print(f'Band={BAND}  Month={MONTH}  SZA={SZA}°  obs=nadir')
print(f'P_LAI={args.p_lai}  δ={args.delta}  res={args.res}  N_spp={args.n_spp}')
print('FD approach: FIXED leaf geometry, vary P_LAI_sp (acceptance probability)')
print('=' * 65)

# ─────────────────────────────────────────────────────────────────────────────
# Load scene ONCE at reference P_LAI
# ─────────────────────────────────────────────────────────────────────────────
print(f'\nLoading scene at P_LAI={args.p_lai} (single load)...')
t0 = time.time()
scene, wood, leaf_spp_id, spp_stats = load_jbs_scene_species(
    scene_dir=_resolve_scene_dir(), band=BAND, month=MONTH,
    P_LAI=args.p_lai, seed=args.seed, verbose=False,
)
scene.std_LAI = 0.0
LAI_Max_sp = spp_stats['LAI_Max_sp']
print(f'  {scene.n_leaf:,} leaves  LAI_Max={scene.LAI_Max:.4f}  ({time.time()-t0:.1f}s)')

# ─────────────────────────────────────────────────────────────────────────────
def render_at_p(p_val, label):
    """Render BRF with fixed leaf geometry but P_LAI_sp overridden to p_val."""
    print(f'\n[{label}]  P_LAI_sp={p_val:.4f}', flush=True)
    p_orig = scene.P_LAI
    scene.P_LAI = p_val   # keep scalar consistent with per-species array
    P_sp = np.full(N_SPECIES, p_val, dtype=np.float64)
    t0 = time.time()
    brf, dfic, dedg, dopt, drho = render_image_bwd_deriv_species_gpu(
        scene, obs_nad, vec_sun,
        leaf_spp_id, spp_stats,
        img_H=args.res, img_W=args.res, N_spp=args.n_spp,
        max_depth=50, seed=args.seed,
        P_LAI_sp=P_sp,
    )
    scene.P_LAI = p_orig
    print(f'  BRF_mean={brf.mean():.6f}  ({time.time()-t0:.1f}s)', flush=True)
    return brf.mean(), dopt

# ─────────────────────────────────────────────────────────────────────────────
# Three renders (same leaf positions, different P_LAI_sp)
# ─────────────────────────────────────────────────────────────────────────────
brf0, dopt0 = render_at_p(args.p_lai,           'REFERENCE (analytical derivative)')
brf_p, _    = render_at_p(args.p_lai + args.delta, 'P + δ  (forward BRF only)')
brf_m, _    = render_at_p(args.p_lai - args.delta, 'P − δ  (forward BRF only)')

# ─────────────────────────────────────────────────────────────────────────────
# Global finite difference  dBRF/dP_LAI
# ─────────────────────────────────────────────────────────────────────────────
fd_global = (brf_p - brf_m) / (2.0 * args.delta)

# Analytical: dBRF/dP = Σ_s π·dopt[s]·LAI_Max_s
an_global = float(sum(np.pi * dopt0[s].mean() * LAI_Max_sp[s]
                      for s in range(N_SPECIES)))

print('\n' + '=' * 65)
print('RESULT — dBRF / dP_LAI  (global, all species together):')
print(f'  Finite difference (central):  {fd_global:+.6f}')
print(f'  Analytical (Σ π·dopt·LAI_M):  {an_global:+.6f}')
ratio = fd_global / an_global if abs(an_global) > 1e-12 else float('nan')
print(f'  Ratio  FD / AN:               {ratio:.4f}')
print(f'  Sign agreement:               {"YES ✓" if fd_global * an_global > 0 else "NO ✗"}')

print('\nPer-species analytical contribution to dBRF/dP_LAI:')
print(f'  {"Species":6s}  {"LAI_Max":>8s}  {"dopt_mean":>12s}  {"π·dopt·LAI_Max":>16s}')
total = 0.0
for s, sp in enumerate(SPECIES_LIST):
    contrib = np.pi * dopt0[s].mean() * LAI_Max_sp[s]
    total  += contrib
    print(f'  {sp:6s}  {LAI_Max_sp[s]:8.4f}  {dopt0[s].mean():+12.6f}  {contrib:+16.6f}')
print(f'  {"TOTAL":6s}  {"":8s}  {"":12s}  {total:+16.6f}')

print(f'\nNote: BRF_ref={brf0:.6f}  BRF(P+δ)={brf_p:.6f}  BRF(P-δ)={brf_m:.6f}')
print(f'  ΔBRF = BRF(P+δ)−BRF(P-δ) = {brf_p - brf_m:+.6f}')

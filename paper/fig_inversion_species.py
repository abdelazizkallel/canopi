"""
lai_rho_species_deriv_inversion.py
====================================
Joint per-SPECIES LAI + leaf reflectance (rho) inversion for 7 JBS species,
using the 18-group renderer (each group renders with the shared species P/rho).

Bands: O08 (RED, 665 nm) + O17 (NIR, 865 nm).

Parameter vector (21):
  theta = [P_1..P_7,  rho_O08_1..rho_O08_7,  rho_O17_1..rho_O17_7]
  P_s       leaf activation probability (shared across all groups of species s)
  rho_O08_s per-species RED leaf reflectance
  rho_O17_s per-species NIR leaf reflectance

Relation to groups:  all groups g that belong to species s share the same
  P_g = P_s  and  rho_g = rho_s.

Tau/rho constraint:  tau_s / rho_s = kappa_s  (known, from RAMI spectral
tables).  rho_eff_g = rho_g × (1+kappa_g)/2  is passed to the GPU kernel.

Jacobian collapsing:
  Group-level Jacobians (H*W, 18) are folded to species level (H*W, 7) by
  summing over all groups within each species:
      J_P_species[:,s]   = Σ_{g∈s} π·LAI_Max_g·dopt_g           (H·W column sum)
      J_rho_species[:,s] = Σ_{g∈s} drho_g / rho_g               (score sums)
  This is exact: ∂BRF/∂θ_s = Σ_{g∈s} ∂BRF/∂θ_g  (shared parameter chain rule).
  Implemented as  J_species = J_groups @ G2S_mat.T  where G2S_mat[s,g] = 1_{g∈s}.

Algorithm (single fully-joint Jacobian GN + ascending line search):
  Render both bands once per iteration → assemble ONE (2HW × 21) Jacobian:
          P_s               rho_O08_s      rho_O17_s
    O08 [ J_P_O08_spec     J_rho_O08_sp   0            ]
    O17 [ J_P_O17_spec     0              J_rho_O17_sp ]
  Per-band scale weighting (1/sigma_b) → single LM step → ascending line search.

Usage:
    python lai_rho_species_deriv_inversion.py
    python lai_rho_species_deriv_inversion.py --img-size 64 --n-spp 8 --max-iter 20
    python lai_rho_species_deriv_inversion.py --true-rho
    python lai_rho_species_deriv_inversion.py --freeze-rho 3
"""
import sys, os, time, argparse, gc
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from canopi.scenes.jbs_species_loader import (
    load_jbs_scene_groups,
    GROUP_LIST, N_GROUPS, GROUP_TO_SPECIES,
    SPECIES_LIST, SPECIES_TO_ID, N_SPECIES,
)
from canopi.scenes.jbs_loader import (
    get_band_params, ILLUMINATION, _REFL, _TRAN, _BAND_IDX,
)
from canopi.geometry import polar_vect
from canopi.paths import scene_dir as _resolve_scene_dir
from numba import cuda
from canopi.gpu import GPU_AVAILABLE, render_image_bwd_deriv_species_gpu
import pickle

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description='JBS 7-species joint LAI + rho inversion (O08+O17, kappa constraint)')
parser.add_argument('--scene-dir',  default=None)
parser.add_argument('--month',      default='Jul', choices=['Apr','Jul'])
parser.add_argument('--max-leaves', type=int,   default=None)
parser.add_argument('--img-size',   type=int,   default=64)
parser.add_argument('--n-spp',      type=int,   default=8)
parser.add_argument('--max-iter',   type=int,   default=20)
parser.add_argument('--p-lai',      type=float, default=0.5)
parser.add_argument('--init-p-lai', type=float, default=0.5,
    help='Uniform initial P for all species (default 0.5 = P_LAI_TRUE)')
parser.add_argument('--rand-p-init', action='store_true',
    help='Draw each species P_init independently from U[0.2, 0.8] using --seed')
parser.add_argument('--noise-rho',  type=float, default=0.2)
parser.add_argument('--freeze-rho', type=int,   default=0)
parser.add_argument('--true-rho',   action='store_true')
parser.add_argument('--lm-lambda',     type=float, default=0.05,
    help='LM regularisation for the joint column-scaled step (one lambda for '
         'all 21 params — column scaling makes diag(JᵀJ)≈I, so a single value '
         'serves LAI and rho alike)')
parser.add_argument('--lm-lambda-rho', type=float, default=1.0,
    help='DEPRECATED / unused — kept for CLI back-compat. The joint scaled '
         'step uses a single --lm-lambda for all parameters.')
parser.add_argument('--k-max',      type=int,   default=9)
parser.add_argument('--k-back',     type=int,   default=12)
parser.add_argument('--seed',       type=int,   default=42)
parser.add_argument('--plot-only',  action='store_true',
    help='Reload pkl from previous run and replot (no simulation)')
parser.add_argument('--out-prefix', type=str, default='lai_rho_species',
    help='Prefix for all output files (default: lai_rho_species)')
args = parser.parse_args()

if not GPU_AVAILABLE:
    raise RuntimeError('GPU (CUDA) required.')

PFX        = args.out_prefix   # output file prefix, used throughout
P_LAI_TRUE = args.p_lai
H = W = args.img_size
deg = np.pi / 180

BAND_O08 = get_band_params('O08')
BAND_O17 = get_band_params('O17')
RHO_O08_MAX = 0.15
RHO_O17_MAX = 0.60
SEED_O08    = args.seed + 1

illum   = ILLUMINATION[args.month]
SZA     = illum['sun_zenith_deg'];  SAA = illum['sun_azimuth_deg']
vec_sun = polar_vect((180.0 - SZA)*deg, (SAA + 180.0)*deg)
obs_nadir = np.array([0.0, 0.0, 1.0])

print(f'\n{"="*74}')
print(f'  JBS 7-Species Joint LAI + rho Inversion  [O08+O17, kappa constraint]')
print(f'  21 parameters: P×7, rho_O08×7, rho_O17×7')
print(f'{"="*74}')
print(f'  O08 (665nm): rho={BAND_O08["rho"]:.5f}  tau={BAND_O08["tau"]:.5f}  '
      f'rsoil={BAND_O08["rsoil"]:.5f}')
print(f'  O17 (866nm): rho={BAND_O17["rho"]:.5f}  tau={BAND_O17["tau"]:.5f}  '
      f'rsoil={BAND_O17["rsoil"]:.5f}')
print(f'  Image={H}×{W}  N_spp={args.n_spp}  max_iter={args.max_iter}  '
      f'lm_lambda={args.lm_lambda}  k_max={args.k_max}  '
      f'freeze_rho={args.freeze_rho}  true_rho={args.true_rho}')

# ─────────────────────────────────────────────────────────────────────────────
# [1] Load scene + build group→species mappings
# ─────────────────────────────────────────────────────────────────────────────
print('\n[1] Loading JBS scene (18 groups, 7 species) ...')
t0 = time.time()
scene, wood, leaf_gid, grp_stats = load_jbs_scene_groups(
    scene_dir  = args.scene_dir or _resolve_scene_dir(),
    band       = 'O08',
    month      = args.month,
    P_LAI      = P_LAI_TRUE,
    max_leaves = args.max_leaves,
    seed       = args.seed,
    verbose    = True,
)
scene.std_LAI = 0.0
print(f'  Loaded in {time.time()-t0:.1f}s  n_leaf={len(scene.leaf_pos):,}')

LAI_Max_grp = grp_stats['LAI_Max_sp']   # (18,) group-level LAI_Max

# ── Group → species index array and projection matrix ────────────────────────
g_to_s = np.array([SPECIES_TO_ID[GROUP_TO_SPECIES[g]] for g in GROUP_LIST],
                  dtype=np.int32)   # (18,)  g_to_s[g] = species index of group g

G2S_mat = np.zeros((N_SPECIES, N_GROUPS), dtype=np.float64)
for g in range(N_GROUPS):
    G2S_mat[g_to_s[g], g] = 1.0   # (7, 18): G2S_mat[s, g] = 1 iff g ∈ species s

# Per-species LAI_Max = sum of group LAI_Max within species
LAI_Max_sp = G2S_mat @ LAI_Max_grp   # (7,)
LAI_eff_true = LAI_Max_sp * P_LAI_TRUE

# ── Per-species true rho / kappa ─────────────────────────────────────────────
bi_O08 = _BAND_IDX['O08']
bi_O17 = _BAND_IDX['O17']

rho_O08_true = np.array([_REFL[sp][bi_O08] for sp in SPECIES_LIST])  # (7,)
rho_O17_true = np.array([_REFL[sp][bi_O17] for sp in SPECIES_LIST])
kappa_O08    = np.array([_TRAN[sp][bi_O08] / _REFL[sp][bi_O08]
                          for sp in SPECIES_LIST])
kappa_O17    = np.array([_TRAN[sp][bi_O17] / _REFL[sp][bi_O17]
                          for sp in SPECIES_LIST])

# Expand species kappa to groups (needed in render_band)
kappa_O08_grp = kappa_O08[g_to_s]   # (18,)
kappa_O17_grp = kappa_O17[g_to_s]

print('\n  True per-species parameters:')
print(f'  {"Sp":6}  {"LAI_Max":>8}  {"LAI_eff":>8}  '
      f'{"rho_O08":>9}  {"κ_O08":>8}  {"rho_O17":>9}  {"κ_O17":>8}')
for s, sp in enumerate(SPECIES_LIST):
    print(f'  {sp:6}  {LAI_Max_sp[s]:8.4f}  {LAI_eff_true[s]:8.4f}  '
          f'{rho_O08_true[s]:9.6f}  {kappa_O08[s]:8.5f}  '
          f'{rho_O17_true[s]:9.6f}  {kappa_O17[s]:8.5f}')

# ─────────────────────────────────────────────────────────────────────────────
# Render helper
# ─────────────────────────────────────────────────────────────────────────────
def render_band(p_sp_k, rho_sp_k, kappa_sp, kappa_grp, band_params, seed_val,
                fixed_mean_rho=None, fixed_mean_tau=None):
    """
    Render one band at per-SPECIES P and rho.

    Species params are expanded to group level before the GPU call.
    Returns (BRF image, dopt (18,H,W), drho (18,H,W)).

    fixed_mean_rho/tau: if provided, scene.rho/tau are frozen to these values
    instead of being recomputed from rho_sp_k.  Pass the values from the
    gradient render so that the reflect/transmit branching (u3 < rho/(rho+tau))
    uses the same random paths during the ascending search as during the
    gradient computation, making cost comparisons deterministic w.r.t. rho_sp.
    """
    gc.collect(); cuda.current_context().deallocations.clear()

    # Expand species → groups
    p_grp   = p_sp_k[g_to_s]    # (18,)
    rho_grp = rho_sp_k[g_to_s]  # (18,)
    kap_grp = kappa_grp          # (18,) precomputed

    rho_eff = rho_grp * (1.0 + kap_grp) / 2.0

    if fixed_mean_rho is not None:
        scene.rho = float(np.clip(fixed_mean_rho, 1e-5, 0.999))
        scene.tau = float(np.clip(fixed_mean_tau, 1e-5, 0.999))
    else:
        lai_eff  = LAI_Max_grp * p_grp
        tau_grp  = kap_grp * rho_grp
        lai_sum  = max(float(lai_eff.sum()), 1e-12)
        scene.rho = float(np.clip(np.dot(lai_eff, rho_grp) / lai_sum, 1e-5, 0.999))
        scene.tau = float(np.clip(np.dot(lai_eff, tau_grp) / lai_sum, 1e-5, 0.999))
    scene.rsoil = float(band_params['rsoil'])

    Ik, _, _, dopt, drho = render_image_bwd_deriv_species_gpu(
        scene, obs_nadir, vec_sun,
        leaf_gid, grp_stats,
        img_H=H, img_W=W, N_spp=args.n_spp,
        max_depth=50, seed=seed_val,
        P_LAI_sp=p_grp.astype(np.float64),
        rho_sp_arr=rho_eff.astype(np.float64),
    )
    return Ik, dopt, drho


def _mean_rho_tau(p_sp_k, rho_sp_k, kappa_grp):
    """Compute LAI-weighted mean rho and tau for the given species params."""
    p_grp   = p_sp_k[g_to_s]
    rho_grp = rho_sp_k[g_to_s]
    tau_grp = kappa_grp * rho_grp
    lai_eff = LAI_Max_grp * p_grp
    lai_sum = max(float(lai_eff.sum()), 1e-12)
    return float(np.dot(lai_eff, rho_grp) / lai_sum), float(np.dot(lai_eff, tau_grp) / lai_sum)


def _o08_cost(p_sp_k, rho_o08_k, mr=None, mt=None):
    Ir, _, _ = render_band(p_sp_k, rho_o08_k, kappa_O08, kappa_O08_grp, BAND_O08, SEED_O08,
                           fixed_mean_rho=mr, fixed_mean_tau=mt)
    r = Ir.ravel() - I_obs_O08.ravel()
    return 0.5 * float(np.dot(r, r)), float(np.sqrt(np.mean(r**2)))


def _o17_cost(p_sp_k, rho_o17_k, mr=None, mt=None):
    In, _, _ = render_band(p_sp_k, rho_o17_k, kappa_O17, kappa_O17_grp, BAND_O17, args.seed,
                           fixed_mean_rho=mr, fixed_mean_tau=mt)
    r = In.ravel() - I_obs_O17.ravel()
    return 0.5 * float(np.dot(r, r)), float(np.sqrt(np.mean(r**2)))


def _amp_max_P(p_sp_k, dP, max_abs_step=0.25):
    """Bound-preserving amp AND absolute-step cap.
    max_abs_step prevents the column-scaled LM step from moving P by more than
    this value at amp=1, keeping the ascending search in the linearisation regime.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        up = np.where(dP > 0, (1.0 - 1e-4 - p_sp_k) / dP, np.inf)
        dn = np.where(dP < 0, (1e-4 - p_sp_k) / dP, np.inf)
    amp_bound = float(np.min(np.minimum(up, dn)))
    max_step  = float(np.max(np.abs(dP)))
    amp_step  = max_abs_step / max_step if max_step > 1e-12 else np.inf
    return min(amp_bound, amp_step)


def _amp_max_rho(rho_k, drho, rho_max, max_rel_step=0.3):
    """Bound-preserving amp AND relative-step cap (max_rel_step × min(rho)).
    Prevents overshooting the minimum when the rho Jacobian is small.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        up = np.where(drho > 0, (rho_max - rho_k) / drho, np.inf)
        dn = np.where(drho < 0, (1e-4 - rho_k) / drho, np.inf)
    amp_bound = float(np.min(np.minimum(up, dn)))
    max_step  = float(np.max(np.abs(drho)))
    amp_step  = max_rel_step * float(np.min(rho_k)) / max_step if max_step > 1e-12 else np.inf
    return min(amp_bound, amp_step)


def _lm_step_scaled(J, r, lm_lambda):
    """Scale-invariant Levenberg-Marquardt step.

    Each Jacobian column is normalised to unit L2 norm BEFORE the normal
    equations are formed, so every parameter direction contributes a term of
    the SAME ORDER regardless of its physical units / sensitivity magnitude
    (a unit change in P vs a unit change in rho).  This is what allows LAI and
    rho to be inverted in ONE joint step: without it the rho columns (~10^4×
    larger) swamp the LAI columns and the LM step barely moves P.

        J~        = J · diag(s),   s_j = 1 / ||J[:,j]||_2
        (J~ᵀJ~ + λI) δy = -J~ᵀ r        (diag(J~ᵀJ~) ≡ 1 ⇒ λ·diag = λ·I)
        δθ        = s ⊙ δy

    Returns the UNSCALED parameter step δθ.
    """
    col_norm  = np.sqrt(np.sum(J * J, axis=0))
    s         = 1.0 / np.maximum(col_norm, 1e-30)
    Js        = J * s[None, :]
    JTJ       = Js.T @ Js
    JTr       = Js.T @ r
    lm        = lm_lambda * np.diag(JTJ)          # ≈ lm_lambda · I (unit diag)
    try:
        dy = np.linalg.solve(JTJ + np.diag(lm), -JTr)
    except np.linalg.LinAlgError:
        return np.zeros(J.shape[1])
    return s * dy


# ─────────────────────────────────────────────────────────────────────────────
# [2] Render true observations
# ─────────────────────────────────────────────────────────────────────────────
p_true = np.full(N_SPECIES, P_LAI_TRUE, dtype=np.float64)

print(f'\n[2] Rendering observations ({H}×{W}, N_spp={args.n_spp}) ...')
t0 = time.time()
I_obs_O17, _, _ = render_band(p_true, rho_O17_true, kappa_O17, kappa_O17_grp, BAND_O17, args.seed)
I_obs_O08, _, _ = render_band(p_true, rho_O08_true, kappa_O08, kappa_O08_grp, BAND_O08, SEED_O08)
print(f'  Done in {time.time()-t0:.1f}s  '
      f'O08 [{I_obs_O08.min():.4f},{I_obs_O08.max():.4f}]  '
      f'O17 [{I_obs_O17.min():.4f},{I_obs_O17.max():.4f}]')

SIG_O08 = float(np.sqrt(np.mean(I_obs_O08**2)))
SIG_O17 = float(np.sqrt(np.mean(I_obs_O17**2)))
print(f'  Band weights:  sigma_O08={SIG_O08:.5f}  sigma_O17={SIG_O17:.5f}')
gc.collect(); cuda.current_context().deallocations.clear()

# ─────────────────────────────────────────────────────────────────────────────
# [3] Initial parameters
# ─────────────────────────────────────────────────────────────────────────────
if args.rand_p_init:
    p_sp = np.random.default_rng(args.seed + 99).uniform(0.2, 0.8, N_SPECIES)
    p_sp = p_sp.astype(np.float64)
else:
    p_sp = np.full(N_SPECIES, args.init_p_lai, dtype=np.float64)
p_sp = np.clip(p_sp, 1e-4, 1.0 - 1e-4)

rng     = np.random.default_rng(args.seed + 99)
noise_r = np.clip(1.0 + args.noise_rho * rng.standard_normal(N_SPECIES), 0.1, 2.0)
noise_n = np.clip(1.0 + args.noise_rho * rng.standard_normal(N_SPECIES), 0.1, 2.0)
rho_O08_sp = np.clip(rho_O08_true * noise_r, 1e-4, RHO_O08_MAX).astype(np.float64)
rho_O17_sp = np.clip(rho_O17_true * noise_n, 1e-4, RHO_O17_MAX).astype(np.float64)

if args.true_rho:
    rho_O08_sp = rho_O08_true.copy()
    rho_O17_sp = rho_O17_true.copy()
    args.freeze_rho = args.max_iter + 1
    print('  [--true-rho] rho fixed at true values.')

print('\n  Initial parameters:')
print(f'  {"Sp":6}  {"P_init":>8}  {"P_true":>7}  '
      f'{"rR_init":>8}  {"rR_true":>8}  {"rN_init":>8}  {"rN_true":>8}')
for s, sp in enumerate(SPECIES_LIST):
    print(f'  {sp:6}  {p_sp[s]:8.4f}  {P_LAI_TRUE:7.4f}  '
          f'{rho_O08_sp[s]:8.5f}  {rho_O08_true[s]:8.5f}  '
          f'{rho_O17_sp[s]:8.5f}  {rho_O17_true[s]:8.5f}')

# ─────────────────────────────────────────────────────────────────────────────
# [4] Fully-joint Gauss-Newton (ONE Jacobian for LAI + rho) + line search
#   The 21-column Jacobian [P | rho_O08 | rho_O17] mixes parameters whose raw
#   derivatives differ by ~10^4 (rho columns ≫ P columns).  A naive joint solve
#   would therefore move only rho and leave LAI essentially frozen.
#   We fix this with COLUMN SCALING (scale-invariant LM, see _lm_step_scaled):
#   every Jacobian column is normalised to unit norm so all parameter
#   directions are of the same order — i.e. varying LAI is treated as exactly
#   as important as varying rho.  Band rows are pre-weighted by 1/sigma_b so
#   O08 (~0.004 RMS) and O17 (~0.31 RMS) also contribute equally.
#
#   Joint Jacobian layout (2·H*W × 21), rows already 1/sigma_b weighted:
#               P_s              rho_O08_s        rho_O17_s
#     O08 [ J_P_O08/σ08      J_rho_O08/σ08    0              ]
#     O17 [ J_P_O17/σ17      0                J_rho_O17/σ17  ]
# ─────────────────────────────────────────────────────────────────────────────
print(f'\n[4] Joint scaled GN  (lm={args.lm_lambda}  k_max={args.k_max}  '
      f'max_iter={args.max_iter}  freeze_rho={args.freeze_rho})')
print(f'  {"it":>3}  {"RMSE_O08":>10}  {"RMSE_O17":>10}  '
      f'{"maxLAI%":>8}  {"maxRR%":>7}  {"maxRN%":>7}  '
      f'{"amp":>6}  {"cond":>9}  t[s]')
print(f'  {"-"*82}')

p_hist       = [p_sp.copy()]
rho_O08_hist = [rho_O08_sp.copy()]
rho_O17_hist = [rho_O17_sp.copy()]
err_lai_hist, err_rho_r_hist, err_rho_n_hist = [], [], []
rmse_O08_h, rmse_O17_h, cost_hist = [], [], []

for it in range(args.max_iter):
    t_it = time.time()

    # ── Render both bands with dynamic scene.rho/tau ──────────────────────
    # Dynamic scene.rho (not frozen) ensures the derivative estimator is
    # consistent with the ACTUAL cost function gradient: the scan confirms
    # cost is monotone when mr adapts naturally to P.  We accept slight
    # stochasticity in ascending-search comparisons and rely on N_spp ≥ 32
    # for reliable cost estimates.  Cost comparisons use the same seeds so
    # the comparison is consistent per render pair even if not pixel-exact.
    I_O08, dopt_O08, drho_O08 = render_band(
        p_sp, rho_O08_sp, kappa_O08, kappa_O08_grp, BAND_O08, SEED_O08)
    I_O17, dopt_O17, drho_O17 = render_band(
        p_sp, rho_O17_sp, kappa_O17, kappa_O17_grp, BAND_O17, args.seed)
    # Dynamic mr: cost evaluations compute their own scene.rho/tau
    mr08, mt08 = None, None
    mr17, mt17 = None, None

    r_O08    = I_O08.ravel() - I_obs_O08.ravel()
    r_O17    = I_O17.ravel() - I_obs_O17.ravel()
    rmse_O08 = float(np.sqrt(np.mean(r_O08**2)))
    rmse_O17 = float(np.sqrt(np.mean(r_O17**2)))
    cost     = 0.5 * (float(np.dot(r_O08, r_O08)) + float(np.dot(r_O17, r_O17)))
    rmse_O08_h.append(rmse_O08); rmse_O17_h.append(rmse_O17); cost_hist.append(cost)

    # ── Collapse group Jacobians → species level via G2S_mat ──────────────
    J_P_grp_O08   = np.stack([np.pi * LAI_Max_grp[g] * dopt_O08[g].ravel()
                               for g in range(N_GROUPS)], axis=1)   # (H*W, 18)
    J_P_grp_O17   = np.stack([np.pi * LAI_Max_grp[g] * dopt_O17[g].ravel()
                               for g in range(N_GROUPS)], axis=1)
    J_rho_grp_O08 = drho_O08.reshape(N_GROUPS, -1).T / rho_O08_sp[g_to_s][None, :]
    J_rho_grp_O17 = drho_O17.reshape(N_GROUPS, -1).T / rho_O17_sp[g_to_s][None, :]

    J_P_O08   = J_P_grp_O08   @ G2S_mat.T   # (H*W, 7)
    J_P_O17   = J_P_grp_O17   @ G2S_mat.T
    J_rho_O08 = J_rho_grp_O08 @ G2S_mat.T
    J_rho_O17 = J_rho_grp_O17 @ G2S_mat.T

    # ── Assemble ONE joint Jacobian over all 21 params (P, rho08, rho17) ──
    # Rows pre-weighted by 1/sigma_b; the off-band rho blocks are zero because
    # rho_O08 only affects O08 and rho_O17 only affects O17.
    do_rho = (it >= args.freeze_rho)
    Z = np.zeros((H * W, N_SPECIES))
    J_top  = np.hstack([J_P_O08 / SIG_O08, J_rho_O08 / SIG_O08, Z              ])
    J_bot  = np.hstack([J_P_O17 / SIG_O17, Z,                   J_rho_O17 / SIG_O17])
    J_full = np.vstack([J_top, J_bot])                       # (2·H*W, 21)
    r_full = np.concatenate([r_O08 / SIG_O08, r_O17 / SIG_O17])

    # When rho is frozen, solve over the P columns only (both-band rows).
    cols = np.arange(3 * N_SPECIES) if do_rho else np.arange(N_SPECIES)
    Jc   = J_full[:, cols]

    # Report conditioning of the COLUMN-SCALED system (what LM actually solves).
    _cn = np.sqrt(np.sum(Jc * Jc, axis=0))
    _Js = Jc / np.maximum(_cn, 1e-30)[None, :]
    try:
        cond = np.linalg.cond(_Js.T @ _Js)
    except np.linalg.LinAlgError:
        cond = np.inf

    dtheta_c = _lm_step_scaled(Jc, r_full, args.lm_lambda)
    dtheta   = np.zeros(3 * N_SPECIES)
    dtheta[cols] = dtheta_c
    dP   = dtheta[0          : N_SPECIES]
    dR08 = dtheta[N_SPECIES  : 2 * N_SPECIES]
    dR17 = dtheta[2*N_SPECIES: 3 * N_SPECIES]

    # ── Single ascending line search over the full joint step ─────────────
    # scene.rho/tau branching frozen (mr*/mt*) so cost is deterministic in θ.
    def _joint_cost_full(p_k, r08_k, r17_k):
        co, _ = _o08_cost(p_k, r08_k, mr08, mt08)
        cn, _ = _o17_cost(p_k, r17_k, mr17, mt17)
        return co / SIG_O08**2 + cn / SIG_O17**2

    amp_max = min(_amp_max_P(p_sp, dP),
                  _amp_max_rho(rho_O08_sp, dR08, RHO_O08_MAX),
                  _amp_max_rho(rho_O17_sp, dR17, RHO_O17_MAX))

    best_cost = _joint_cost_full(p_sp, rho_O08_sp, rho_O17_sp)
    amp = 0.0
    # Diagnostic (first iteration only)
    if it == 0:
        _cn = np.sqrt(np.sum(Jc*Jc, axis=0))
        _Js = Jc / np.maximum(_cn, 1e-30)[None, :]
        _JTr = _Js.T @ r_full
        print(f'  [diag] r_O08: mean={r_O08.mean():+.5f}  r_O17: mean={r_O17.mean():+.5f}')
        print(f'  [diag] JTr_scaled={_JTr.round(4)}')
        print(f'  [diag] dP={dP.round(4)}  max|dP|={np.max(np.abs(dP)):.5f}  amp_max={amp_max:.4f}')
        c1 = _joint_cost_full(np.clip(p_sp+1.0*dP,1e-4,.9999), rho_O08_sp, rho_O17_sp)
        print(f'  [diag] cost_base={best_cost:.3f}  cost@amp=1:{c1:.3f}  delta={c1-best_cost:+.3f}')

    def _eval(a):
        p_c   = np.clip(p_sp       + a * dP,   1e-4, 1.0 - 1e-4)
        r08_c = np.clip(rho_O08_sp + a * dR08, 1e-4, RHO_O08_MAX)
        r17_c = np.clip(rho_O17_sp + a * dR17, 1e-4, RHO_O17_MAX)
        if (np.max(np.abs(p_c   - p_sp))      < 1e-7 and
                np.max(np.abs(r08_c - rho_O08_sp)) < 1e-7 and
                np.max(np.abs(r17_c - rho_O17_sp)) < 1e-7):
            return None
        return p_c, r08_c, r17_c, _joint_cost_full(p_c, r08_c, r17_c)

    # Ascending phase: try amp = 1, 2, 4 … 2^k_max (capped at amp_max)
    for k in range(0, args.k_max + 1):
        res = _eval(min(2.0 ** k, amp_max))
        if res is None: break
        p_c, r08_c, r17_c, c = res
        if c < best_cost:
            best_cost = c
            p_sp, rho_O08_sp, rho_O17_sp = p_c, r08_c, r17_c
            amp = min(2.0 ** k, amp_max)
        else:
            break

    # Backtracking phase: if ascending found nothing, try amp = 0.5, 0.25, …
    # Needed when all species step in the same direction and collectively overshoot.
    if amp == 0.0:
        for k in range(1, args.k_back + 1):
            res = _eval(2.0 ** (-k))
            if res is None: break
            p_c, r08_c, r17_c, c = res
            if c < best_cost:
                best_cost = c
                p_sp, rho_O08_sp, rho_O17_sp = p_c, r08_c, r17_c
                amp = 2.0 ** (-k)
                break   # accept first improving backtrack step

    # ── Logging ────────────────────────────────────────────────────────────
    err_lai   = np.abs(p_sp * LAI_Max_sp - LAI_eff_true) / np.maximum(LAI_eff_true, 1e-6) * 100
    err_rho_r = np.abs(rho_O08_sp - rho_O08_true) / np.maximum(rho_O08_true, 1e-6) * 100
    err_rho_n = np.abs(rho_O17_sp - rho_O17_true) / np.maximum(rho_O17_true, 1e-6) * 100
    err_lai_hist.append(err_lai); err_rho_r_hist.append(err_rho_r); err_rho_n_hist.append(err_rho_n)

    elapsed = time.time() - t_it
    print(f'  {it:>3}  {rmse_O08:>10.4e}  {rmse_O17:>10.4e}  '
          f'{err_lai.max():>8.2f}%  {err_rho_r.max():>7.2f}%  {err_rho_n.max():>7.2f}%  '
          f'x{amp:>4.1f}  {cond:>9.2e}  {elapsed:.1f}s')

    p_hist.append(p_sp.copy())
    rho_O08_hist.append(rho_O08_sp.copy())
    rho_O17_hist.append(rho_O17_sp.copy())

    if amp == 0:
        print('  No improving joint step — converged.'); break

# ─────────────────────────────────────────────────────────────────────────────
# Final summary
# ─────────────────────────────────────────────────────────────────────────────
n_done         = len(cost_hist)
p_hist_a       = np.array(p_hist)
rho_O08_hist_a = np.array(rho_O08_hist)
rho_O17_hist_a = np.array(rho_O17_hist)
err_lai_a      = np.array(err_lai_hist)   if err_lai_hist   else np.zeros((0, N_SPECIES))
err_rho_r_a    = np.array(err_rho_r_hist) if err_rho_r_hist else np.zeros((0, N_SPECIES))
err_rho_n_a    = np.array(err_rho_n_hist) if err_rho_n_hist else np.zeros((0, N_SPECIES))

# ── Save inversion data for --plot-only reruns ─────────────────────────────
_tag = (f'_randp' if args.rand_p_init else f'_p{args.init_p_lai}') + f'_k{args.k_max}'
PKL_FILE = f'{PFX}{_tag}.pkl'
with open(PKL_FILE, 'wb') as _f:
    pickle.dump(dict(
        n_done=n_done, SPECIES_LIST=SPECIES_LIST,
        LAI_Max_sp=LAI_Max_sp, LAI_eff_true=LAI_eff_true,
        rho_O08_true=rho_O08_true, rho_O17_true=rho_O17_true,
        p_hist_a=p_hist_a, rho_O08_hist_a=rho_O08_hist_a, rho_O17_hist_a=rho_O17_hist_a,
        err_lai_a=err_lai_a, err_rho_r_a=err_rho_r_a, err_rho_n_a=err_rho_n_a,
        rmse_O08_h=rmse_O08_h, rmse_O17_h=rmse_O17_h, cost_hist=cost_hist,
    ), _f)
print(f'Data → {PKL_FILE}')

print('\n  Final retrieved parameters:')
print(f'  {"Sp":6}  {"P_fin":>7}  {"LAI_err%":>9}  '
      f'{"rR_ini":>8}  {"rR_fin":>8}  {"rR_tru":>8}  {"rR_e%":>7}  '
      f'{"rN_ini":>8}  {"rN_fin":>8}  {"rN_tru":>8}  {"rN_e%":>7}')
print(f'  {"-"*104}')
for s, sp in enumerate(SPECIES_LIST):
    lai_e = 100*(p_sp[s]*LAI_Max_sp[s] - LAI_eff_true[s]) / max(LAI_eff_true[s], 1e-6)
    rr_e  = 100*(rho_O08_sp[s] - rho_O08_true[s]) / max(rho_O08_true[s], 1e-6)
    rn_e  = 100*(rho_O17_sp[s] - rho_O17_true[s]) / max(rho_O17_true[s], 1e-6)
    print(f'  {sp:6}  {p_sp[s]:7.5f}  {lai_e:+9.2f}%  '
          f'{rho_O08_hist_a[0,s]:8.5f}  {rho_O08_sp[s]:8.5f}  '
          f'{rho_O08_true[s]:8.5f}  {rr_e:+7.2f}%  '
          f'{rho_O17_hist_a[0,s]:8.5f}  {rho_O17_sp[s]:8.5f}  '
          f'{rho_O17_true[s]:8.5f}  {rn_e:+7.2f}%')

# ─────────────────────────────────────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────────────────────────────────────
SP_COLOR = {
    'BEPE':'#2166AC','TICO':'#4DAF4A','ALGL':'#FF7F00',
    'POTR':'#984EA3','FREX':'#E41A1C','ACPL':'#795548','PIAB':'#009688',
}
colors_s = [SP_COLOR[sp] for sp in SPECIES_LIST]
x_pts    = np.arange(len(p_hist_a))

fig = plt.figure(figsize=(22, 15))
fig.suptitle(
    f'JBS 7-Species Joint LAI + ρ Inversion  [O08+O17, τ/ρ=κ, 21 params]\n'
    f'N_spp={args.n_spp}  lm_λ={args.lm_lambda}  k_max={args.k_max}  '
    f'freeze_rho={args.freeze_rho}  true_rho={args.true_rho}',
    fontsize=11)

gs0 = fig.add_gridspec(2, 4, top=0.88, bottom=0.34, hspace=0.40, wspace=0.32)

ax = fig.add_subplot(gs0[0, 0])
for s, sp in enumerate(SPECIES_LIST):
    ax.plot(x_pts, p_hist_a[:, s] * LAI_Max_sp[s],
            '-o', color=colors_s[s], lw=1.8, ms=4, label=sp)
    ax.axhline(LAI_eff_true[s], ls='--', lw=0.9, color=colors_s[s], alpha=0.6)
ax.set_xlabel('Iteration'); ax.set_ylabel('LAI_eff')
ax.set_title('(a) LAI  (dashed=true)'); ax.legend(fontsize=7); ax.grid(alpha=0.3); ax.set_ylim(bottom=0)

ax = fig.add_subplot(gs0[0, 1])
for s, sp in enumerate(SPECIES_LIST):
    ax.plot(x_pts, rho_O08_hist_a[:, s],
            '-o', color=colors_s[s], lw=1.8, ms=4, label=sp)
    ax.axhline(rho_O08_true[s], ls='--', lw=0.9, color=colors_s[s], alpha=0.6)
ax.set_xlabel('Iteration'); ax.set_ylabel('ρ_O08')
ax.set_title('(b) ρ_O08  (dashed=true)'); ax.legend(fontsize=7); ax.grid(alpha=0.3)

ax = fig.add_subplot(gs0[0, 2])
for s, sp in enumerate(SPECIES_LIST):
    ax.plot(x_pts, rho_O17_hist_a[:, s],
            '-o', color=colors_s[s], lw=1.8, ms=4, label=sp)
    ax.axhline(rho_O17_true[s], ls='--', lw=0.9, color=colors_s[s], alpha=0.6)
ax.set_xlabel('Iteration'); ax.set_ylabel('ρ_O17')
ax.set_title('(c) ρ_O17  (dashed=true)'); ax.legend(fontsize=7); ax.grid(alpha=0.3)

ax = fig.add_subplot(gs0[0, 3])
if len(err_lai_a):
    for s, sp in enumerate(SPECIES_LIST):
        ax.semilogy(range(n_done), np.maximum(err_lai_a[:, s], 0.01),
                    '-o', color=colors_s[s], lw=1.8, ms=4, label=sp)
ax.axhline(1.0, ls=':', lw=1, color='gray')
ax.set_xlabel('Iteration'); ax.set_ylabel('|LAI error| %')
ax.set_title('(d) LAI error %  (log)'); ax.legend(fontsize=7)
ax.grid(True, which='both', alpha=0.3)

ax = fig.add_subplot(gs0[1, 0])
if len(err_rho_r_a):
    for s, sp in enumerate(SPECIES_LIST):
        ax.semilogy(range(n_done), np.maximum(err_rho_r_a[:, s], 0.01),
                    '-o', color=colors_s[s], lw=1.8, ms=4, label=sp)
ax.axhline(1.0, ls=':', lw=1, color='gray')
ax.set_xlabel('Iteration'); ax.set_ylabel('|ρ_O08 error| %')
ax.set_title('(e) ρ_O08 error %  (log)'); ax.legend(fontsize=7)
ax.grid(True, which='both', alpha=0.3)

ax = fig.add_subplot(gs0[1, 1])
if len(err_rho_n_a):
    for s, sp in enumerate(SPECIES_LIST):
        ax.semilogy(range(n_done), np.maximum(err_rho_n_a[:, s], 0.01),
                    '-o', color=colors_s[s], lw=1.8, ms=4, label=sp)
ax.axhline(1.0, ls=':', lw=1, color='gray')
ax.set_xlabel('Iteration'); ax.set_ylabel('|ρ_O17 error| %')
ax.set_title('(f) ρ_O17 error %  (log)'); ax.legend(fontsize=7)
ax.grid(True, which='both', alpha=0.3)

ax = fig.add_subplot(gs0[1, 2])
ax.semilogy(range(n_done), rmse_O08_h, 'r-o', ms=4, lw=1.5, label='O08 (RED)')
ax.semilogy(range(n_done), rmse_O17_h, 'b-s', ms=4, lw=1.5, label='O17 (NIR)')
ax.set_xlabel('Iteration'); ax.set_ylabel('RMSE')
ax.set_title('(g) Per-band RMSE  (log)'); ax.legend(fontsize=8)
ax.grid(True, which='both', alpha=0.3)

ax = fig.add_subplot(gs0[1, 3])
ax.semilogy(range(n_done), cost_hist, 'k-o', lw=1.8, ms=4)
ax.set_xlabel('Iteration'); ax.set_ylabel('½(‖r_O08‖²+‖r_O17‖²)')
ax.set_title('(h) Joint cost  (log)')
ax.grid(True, which='both', alpha=0.3)

print('\n  Rendering init + final BRF images ...')
I_O08_init, _, _ = render_band(p_hist_a[0], rho_O08_hist_a[0], kappa_O08, kappa_O08_grp, BAND_O08, SEED_O08)
I_O17_init, _, _ = render_band(p_hist_a[0], rho_O17_hist_a[0], kappa_O17, kappa_O17_grp, BAND_O17, args.seed)
I_O08_fin,  _, _ = render_band(p_sp,        rho_O08_sp,         kappa_O08, kappa_O08_grp, BAND_O08, SEED_O08)
I_O17_fin,  _, _ = render_band(p_sp,        rho_O17_sp,         kappa_O17, kappa_O17_grp, BAND_O17, args.seed)

gs1 = fig.add_gridspec(2, 4, top=0.30, bottom=0.03, hspace=0.05, wspace=0.05)

def _im(ax, img, title, cmap='YlGn', **kw):
    ax.imshow(img, origin='lower', interpolation='nearest', cmap=cmap, **kw)
    ax.set_title(title, fontsize=8); ax.set_xticks([]); ax.set_yticks([])

vn0 = min(I_obs_O17.min(), I_O17_init.min())
vn1 = max(I_obs_O17.max(), I_O17_init.max())
vo0 = min(I_obs_O08.min(), I_O08_init.min())
vo1 = max(I_obs_O08.max(), I_O08_init.max())

_im(fig.add_subplot(gs1[0, 0]), I_obs_O17,  'O17 obs',   vmin=vn0, vmax=vn1)
_im(fig.add_subplot(gs1[0, 1]), I_O17_init, 'O17 init',  vmin=vn0, vmax=vn1)
_im(fig.add_subplot(gs1[0, 2]), I_O17_fin,  'O17 final', vmin=vn0, vmax=vn1)
res_n = I_O17_fin - I_obs_O17; ar_n = max(float(np.abs(res_n).max()), 1e-8)
ax_rn = fig.add_subplot(gs1[0, 3])
ax_rn.imshow(res_n, origin='lower', cmap='RdBu_r', vmin=-ar_n, vmax=ar_n, interpolation='nearest')
ax_rn.set_title('O17 residual', fontsize=8); ax_rn.set_xticks([]); ax_rn.set_yticks([])

_im(fig.add_subplot(gs1[1, 0]), I_obs_O08,  'O08 obs',   vmin=vo0, vmax=vo1)
_im(fig.add_subplot(gs1[1, 1]), I_O08_init, 'O08 init',  vmin=vo0, vmax=vo1)
_im(fig.add_subplot(gs1[1, 2]), I_O08_fin,  'O08 final', vmin=vo0, vmax=vo1)
res_r = I_O08_fin - I_obs_O08; ar_r = max(float(np.abs(res_r).max()), 1e-8)
ax_rr = fig.add_subplot(gs1[1, 3])
ax_rr.imshow(res_r, origin='lower', cmap='RdBu_r', vmin=-ar_r, vmax=ar_r, interpolation='nearest')
ax_rr.set_title('O08 residual', fontsize=8); ax_rr.set_xticks([]); ax_rr.set_yticks([])

plt.tight_layout()
out_png = f'{PFX}_inversion_jbs.png'
plt.savefig(out_png, dpi=140, bbox_inches='tight')
print(f'\nFigure -> {out_png}')

def _save_eps(fname, draw, figsize=(6, 5)):
    f, a = plt.subplots(figsize=figsize)
    draw(a)
    f.savefig(fname, format='eps', bbox_inches='tight')
    plt.close(f)
    print(f'  EPS -> {fname}')

def _img_eps(fname, img, title=None, cmap='YlGn', vmin=None, vmax=None, figsize=(4.8, 4.2)):
    """Save a single BRF/residual image as EPS with a proper colorbar."""
    f, a = plt.subplots(figsize=figsize)
    im = a.imshow(img, origin='lower', interpolation='nearest',
                  cmap=cmap, vmin=vmin, vmax=vmax)
    if title:
        a.set_title(title, fontsize=11)
    a.set_xticks([]); a.set_yticks([])
    f.colorbar(im, ax=a, fraction=0.046, pad=0.04, format='%.3f')
    f.savefig(fname, format='eps', bbox_inches='tight')
    plt.close(f)
    print(f'  EPS -> {fname}')

def _conv(a, hist, true, ylab, title):
    for s, sp in enumerate(SPECIES_LIST):
        a.plot(x_pts, hist[:, s], '-o', color=colors_s[s], lw=1.8, ms=4, label=sp)
        a.axhline(true[s], ls='--', lw=0.9, color=colors_s[s], alpha=0.6)
    a.set_xlabel('Iteration'); a.set_ylabel(ylab); a.set_title(title)
    a.legend(fontsize=7); a.grid(alpha=0.3)

def _errlog(a, hist, ylab, title):
    if len(hist):
        for s, sp in enumerate(SPECIES_LIST):
            a.semilogy(range(n_done), np.maximum(hist[:, s], 0.01),
                       '-o', color=colors_s[s], lw=1.8, ms=4, label=sp)
    a.axhline(1.0, ls=':', lw=1, color='gray')
    a.set_xlabel('Iteration'); a.set_ylabel(ylab); a.set_title(title)
    a.legend(fontsize=7); a.grid(True, which='both', alpha=0.3)

_save_eps(f'{PFX}_lai_conv.eps',
          lambda a: _conv(a, p_hist_a * LAI_Max_sp[None, :],
                          LAI_eff_true, 'LAI_eff', 'LAI  (dashed = true)'))
_save_eps(f'{PFX}_rho_O08_conv.eps',
          lambda a: _conv(a, rho_O08_hist_a, rho_O08_true, 'ρ_O08', 'ρ_O08  (dashed = true)'))
_save_eps(f'{PFX}_rho_O17_conv.eps',
          lambda a: _conv(a, rho_O17_hist_a, rho_O17_true, 'ρ_O17', 'ρ_O17  (dashed = true)'))
_save_eps(f'{PFX}_lai_error.eps',
          lambda a: _errlog(a, err_lai_a, '|LAI error| %', 'LAI error %  (log)'))
_save_eps(f'{PFX}_rho_O08_error.eps',
          lambda a: _errlog(a, err_rho_r_a, '|ρ_O08 error| %', 'ρ_O08 error %  (log)'))
_save_eps(f'{PFX}_rho_O17_error.eps',
          lambda a: _errlog(a, err_rho_n_a, '|ρ_O17 error| %', 'ρ_O17 error %  (log)'))

def _panel_rmse(a):
    a.semilogy(range(n_done), rmse_O08_h, 'r-o', ms=4, lw=1.5, label='O08 (RED)')
    a.semilogy(range(n_done), rmse_O17_h, 'b-s', ms=4, lw=1.5, label='O17 (NIR)')
    a.set_xlabel('Iteration'); a.set_ylabel('RMSE'); a.set_title('Per-band RMSE  (log)')
    a.legend(fontsize=8); a.grid(True, which='both', alpha=0.3)

def _panel_cost(a):
    a.semilogy(range(n_done), cost_hist, 'k-o', lw=1.8, ms=4)
    a.set_xlabel('Iteration'); a.set_ylabel('½(‖r_O08‖²+‖r_O17‖²)')
    a.set_title('Joint cost  (log)'); a.grid(True, which='both', alpha=0.3)

_save_eps(f'{PFX}_rmse.eps',  _panel_rmse)
_save_eps(f'{PFX}_cost.eps',  _panel_cost)

# ── Individual BRF images with colorbar ───────────────────────────────────────
_img_eps(f'{PFX}_O08_obs.eps',   I_obs_O08,  'O08 observed', vmin=vo0, vmax=vo1)
_img_eps(f'{PFX}_O08_init.eps',  I_O08_init, 'O08 init',     vmin=vo0, vmax=vo1)
_img_eps(f'{PFX}_O08_final.eps', I_O08_fin,  'O08 final',    vmin=vo0, vmax=vo1)
_img_eps(f'{PFX}_O08_resid.eps', res_r, 'O08 residual', cmap='RdBu_r', vmin=-ar_r, vmax=ar_r)
_img_eps(f'{PFX}_O17_obs.eps',   I_obs_O17,  'O17 observed', vmin=vn0, vmax=vn1)
_img_eps(f'{PFX}_O17_init.eps',  I_O17_init, 'O17 init',     vmin=vn0, vmax=vn1)
_img_eps(f'{PFX}_O17_final.eps', I_O17_fin,  'O17 final',    vmin=vn0, vmax=vn1)
_img_eps(f'{PFX}_O17_resid.eps', res_n, 'O17 residual', cmap='RdBu_r', vmin=-ar_n, vmax=ar_n)

# ── Total LAI error vs iteration ───────────────────────────────────────────
# Total canopy LAI = sum_s P_s × LAI_Max_s; error relative to true total.
LAI_hist_total  = p_hist_a @ LAI_Max_sp          # (n_iter+1,) total LAI at each step
LAI_true_total  = float(LAI_eff_true.sum())
LAI_err_total   = np.abs(LAI_hist_total - LAI_true_total) / LAI_true_total * 100

def _panel_total_lai(a):
    a.plot(range(len(LAI_hist_total)), LAI_err_total, 'k-o', lw=2, ms=6)
    a.axhline(0, color='k', lw=0.8, ls='--', alpha=0.4)
    a.set_xlabel('Iteration', fontsize=13)
    a.set_ylabel('Total LAI error (%)', fontsize=13)
    a.set_title(fr'Total LAI error  '
                fr'(true = {LAI_true_total:.4f})', fontsize=11)
    a.grid(True, alpha=0.4)
    # annotate initial and final values
    a.annotate(f'{LAI_err_total[0]:.1f}%',
               xy=(0, LAI_err_total[0]), xytext=(0.5, LAI_err_total[0]),
               fontsize=9, color='gray')
    if len(LAI_err_total) > 1:
        a.annotate(f'{LAI_err_total[-1]:.2f}%',
                   xy=(len(LAI_err_total)-1, LAI_err_total[-1]),
                   xytext=(len(LAI_err_total)-2, LAI_err_total[-1]+0.5),
                   fontsize=9, color='gray')

_save_eps(f'{PFX}_total_lai_error.eps', _panel_total_lai, figsize=(6, 4))

# also a combined figure: total LAI error + per-species LAI error on same axes
def _panel_total_and_species(a):
    iters = range(len(LAI_hist_total))
    for s, sp in enumerate(SPECIES_LIST):
        per_s = np.abs(p_hist_a[:, s] * LAI_Max_sp[s] - LAI_eff_true[s]) \
                / max(LAI_eff_true[s], 1e-6) * 100
        a.plot(iters, per_s, '--', lw=1.2, alpha=0.6,
               color=colors_s[s], label=sp)
    a.plot(iters, LAI_err_total, 'k-o', lw=2.5, ms=7, label='Total')
    a.set_xlabel('Iteration', fontsize=13)
    a.set_ylabel('LAI error (%)', fontsize=13)
    a.set_title('Per-species (dashed) and total (black) LAI error', fontsize=10)
    a.legend(fontsize=7, ncol=2); a.grid(True, alpha=0.4)

_save_eps(f'{PFX}_total_lai_combined.eps', _panel_total_and_species, figsize=(7, 5))

print('Done.')

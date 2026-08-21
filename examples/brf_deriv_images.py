"""
jbs_backward_image_species.py
==============================
Spatially-resolved BRF + per-species dBRF/dLAI_s images of the JBS stand.

Each of the 7 tree species (BEPE, TICO, ALGL, POTR, PIAB, FREX, ACPL) has its
own LAI, and the derivative dBRF/dLAI_s is computed independently for each
species using separate fictive and edge estimators.

Output figures
--------------
  jbs_species_brf_deriv.png        — nadir BRF + per-species dopt maps (1+7 cols)
  jbs_species_fic_edg.png          — per-species dfic vs dedg comparison (nadir)
  jbs_species_views.png            — 3 views × 7 species dopt maps
  jbs_species_brf_contribution.png — nadir total BRF + per-species BRF contribution (1+7 cols)

Usage:
    python jbs_backward_image_species.py
    python jbs_backward_image_species.py --band O08 --res 128 --n-spp 32
"""
import sys, os, time, argparse
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from canopi.scenes.jbs_species_loader import (
    load_jbs_scene_species, get_band_params, ILLUMINATION,
    JBS_SPECIES, SPECIES_LIST, SPECIES_TO_ID, N_SPECIES,
    _synthetic_instances, BANDS, LX, LY,
)
from canopi.geometry import polar_vect
from canopi.paths import scene_dir as _resolve_scene_dir

try:
    from canopi.gpu import (GPU_AVAILABLE,
                                  render_image_bwd_deriv_species_gpu)
except ImportError:
    GPU_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description='JBS per-species backward image: BRF + dBRF/dLAI_s')
parser.add_argument('--scene-dir', default=None)
parser.add_argument('--band',      default='O08', choices=BANDS)
parser.add_argument('--month',     default='Jul', choices=['Apr', 'Jul'],
    help='Illumination month (overridden by --sza/--saa if given)')
parser.add_argument('--sza',       type=float, default=None,
    help='Override sun zenith angle [deg] (e.g. 0 for nadir sun)')
parser.add_argument('--saa',       type=float, default=None,
    help='Override sun azimuth angle [deg] (default 0 when --sza given)')
parser.add_argument('--p-lai',     type=float, default=0.5)
parser.add_argument('--res',       type=int,   default=128,
    help='Image resolution (default 128)')
parser.add_argument('--n-spp',     type=int,   default=32,
    help='MC samples per pixel (default 32)')
parser.add_argument('--max-leaves', type=int,  default=None)
parser.add_argument('--seed',      type=int,   default=42)
parser.add_argument('--std-mul',   type=float, default=0.0,
    help='leaf_radius_std_mul (default 0.0 — deterministic, step-function q_k)')
args = parser.parse_args()

IMG   = args.res
N_SPP = args.n_spp

# ─────────────────────────────────────────────────────────────────────────────
# Load scene with species tracking
# ─────────────────────────────────────────────────────────────────────────────
print('=' * 70)
print('  JBS Backward Image — Per-Species BRF + dBRF/dLAI_s')
print('=' * 70)

t0 = time.time()
scene, wood, leaf_species_id, species_stats = load_jbs_scene_species(
    scene_dir   = args.scene_dir or _resolve_scene_dir(),
    band        = args.band,
    month       = args.month,
    P_LAI       = args.p_lai,
    leaf_radius_std_mul = args.std_mul,
    max_leaves  = args.max_leaves,
    seed        = args.seed,
    verbose     = True,
)
t_load = time.time() - t0

# force std_LAI=0 for clean derivative (step-function q_k)
scene.std_LAI = 0.0

bp    = get_band_params(args.band)
illum = ILLUMINATION[args.month]
SZA   = args.sza if args.sza is not None else illum['sun_zenith_deg']
SAA   = args.saa if args.saa is not None else (0.0 if args.sza is not None else illum['sun_azimuth_deg'])
print(f'Scene loaded in {t_load:.1f}s')
print(f'Leaves: {scene.n_leaf:,}   LAI_Max={scene.LAI_Max:.3f}')

# Per-species LAI summary
print('\nPer-species LAI_Max:')
for s, sp_name in enumerate(SPECIES_LIST):
    print(f'  {sp_name}: LAI_Max={species_stats["LAI_Max_sp"][s]:.4f}  '
          f'n_leaf={int(species_stats["n_leaf_norm_sp"][s] * LX * LY):,}')

backend = 'GPU (CUDA)' if GPU_AVAILABLE else 'CPU (not supported for species mode)'
print(f'\nBackend: {backend}')
print(f'Image: {IMG}x{IMG} px  N_spp={N_SPP}')

if not GPU_AVAILABLE:
    raise RuntimeError('Per-species derivative requires GPU (CUDA). No GPU found.')

# ─────────────────────────────────────────────────────────────────────────────
# Sun and observer directions
# ─────────────────────────────────────────────────────────────────────────────
deg     = np.pi / 180
tha_s   = (180.0 - SZA) * deg
phi_s   = (SAA + 180.0) * deg
vec_sun = polar_vect(tha_s, phi_s)

VIEWS = [
    dict(label=f'Nadir  (theta_v=0)',
         zen_deg=0.0, az_deg=0.0),
    dict(label=f'Hotspot  (theta_v={SZA:.0f}, phi={SAA:.0f})',
         zen_deg=SZA, az_deg=SAA),
    dict(label=f'Forward  (theta_v={SZA:.0f}, phi={(SAA+180)%360:.0f})',
         zen_deg=SZA, az_deg=(SAA + 180.0) % 360.0),
]

# ─────────────────────────────────────────────────────────────────────────────
# Leaf density map
# ─────────────────────────────────────────────────────────────────────────────
lp = scene.leaf_pos
ix = (lp[:, 0] / LX * IMG).astype(int).clip(0, IMG - 1)
iy = (lp[:, 1] / LY * IMG).astype(int).clip(0, IMG - 1)
density_map    = np.bincount(iy * IMG + ix, minlength=IMG*IMG).reshape(IMG, IMG).astype(float)
density_smooth = gaussian_filter(density_map, sigma=2.0)
crown_thresh   = 0.30 * density_smooth.max()
extent         = [0, LX, 0, LY]

# ─────────────────────────────────────────────────────────────────────────────
# Species colours
# ─────────────────────────────────────────────────────────────────────────────
SP_COLOR = {
    'BEPE': '#2ca02c', 'TICO': '#1f77b4', 'ALGL': '#ff7f0e',
    'POTR': '#9467bd', 'PIAB': '#8c564b', 'FREX': '#e377c2', 'ACPL': '#7f7f7f',
}
SP_COLORS = [SP_COLOR[sp] for sp in SPECIES_LIST]

# ─────────────────────────────────────────────────────────────────────────────
# Render all views
# ─────────────────────────────────────────────────────────────────────────────
all_results = []
for view in VIEWS:
    obs_dir = polar_vect(view['zen_deg'] * deg, view['az_deg'] * deg)
    print(f'\nRendering: {view["label"]} ...')
    t0 = time.time()
    brf, dfic, dedg, dopt, drho = render_image_bwd_deriv_species_gpu(
        scene, obs_dir, vec_sun,
        leaf_species_id, species_stats,
        img_H=IMG, img_W=IMG, N_spp=N_SPP,
        max_depth=50, seed=args.seed,
    )
    dt = time.time() - t0
    print(f'  BRF_mean={brf.mean():.4f}  ({dt:.1f}s)')
    for s, sp_name in enumerate(SPECIES_LIST):
        print(f'  {sp_name}: dopt_mean={dopt[s].mean():+.6f}  '
              f'dfic={dfic[s].mean():+.6f}  dedg={dedg[s].mean():+.6f}  '
              f'brf_contrib={drho[s].mean():.6f}')
    all_results.append((brf, dfic, dedg, dopt, drho))

brf_nad, dfic_nad, dedg_nad, dopt_nad, drho_nad = all_results[0]

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Nadir — BRF + per-species dopt
# ─────────────────────────────────────────────────────────────────────────────
n_cols  = N_SPECIES + 1   # BRF + 7 species
fig1, axes1 = plt.subplots(1, n_cols, figsize=(4.5 * n_cols, 5.0))

vmax_brf = brf_nad.max()

# BRF
ax = axes1[0]
im = ax.imshow(brf_nad, origin='lower', extent=extent,
               cmap='YlGn', vmin=0, vmax=vmax_brf,
               aspect='equal', interpolation='nearest')
ax.contour(density_smooth, levels=[crown_thresh], colors=['white'],
           linewidths=[0.7], alpha=0.6, origin='lower', extent=extent)
fig1.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=7)
ax.set_title(f'BRF\nmean={brf_nad.mean():.4f}', fontsize=9)
ax.set_xlabel('x [m]', fontsize=8); ax.set_ylabel('y [m]', fontsize=8)

# Per-species dopt — each panel uses its own colour scale
for s, sp_name in enumerate(SPECIES_LIST):
    ax = axes1[s + 1]
    img_s = dopt_nad[s]
    abs_s = max(float(np.abs(img_s).max()), 1e-8)
    norm  = TwoSlopeNorm(vmin=-abs_s, vcenter=0, vmax=abs_s)
    im = ax.imshow(img_s, origin='lower', extent=extent,
                   cmap='RdBu_r', norm=norm, aspect='equal',
                   interpolation='nearest')
    ax.contour(density_smooth, levels=[crown_thresh], colors=['white'],
               linewidths=[0.5], alpha=0.55, origin='lower', extent=extent)
    fig1.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=7)
    ax.set_title(f'{sp_name}\ndopt_mean={img_s.mean():+.5f}', fontsize=9,
                 color=SP_COLOR[sp_name])
    ax.set_xlabel('x [m]', fontsize=8)
    ax.tick_params(labelsize=7)

fig1.suptitle(
    f'JBS Nadir — BRF + per-species dBRF/dLAI_s  [GPU]\n'
    f'band={args.band}  SZA={SZA:.0f}  {IMG}x{IMG}  N_spp={N_SPP}  P_LAI={args.p_lai}',
    fontsize=10, y=1.01,
)
plt.tight_layout()
out1 = 'jbs_species_brf_deriv.png'
plt.savefig(out1, dpi=130, bbox_inches='tight')
print(f'\nFigure 1 saved -> {out1}')

# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Nadir — per-species BRF contribution maps
# ─────────────────────────────────────────────────────────────────────────────
# drho_nad[s] = π * E[K_s * lum]  = species s contribution to scene radiance
# (proportional to BRF fraction from species s; shared colorbar across all species)
fig4, axes4 = plt.subplots(1, N_SPECIES + 1, figsize=(4.5 * (N_SPECIES + 1), 5.0))

# left panel: total nadir BRF for reference
ax = axes4[0]
im = ax.imshow(brf_nad, origin='lower', extent=extent,
               cmap='YlGn', vmin=0, vmax=vmax_brf,
               aspect='equal', interpolation='nearest')
ax.contour(density_smooth, levels=[crown_thresh], colors=['white'],
           linewidths=[0.7], alpha=0.6, origin='lower', extent=extent)
fig4.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=7)
ax.set_title(f'Total BRF\nmean={brf_nad.mean():.4f}', fontsize=9)
ax.set_xlabel('x [m]', fontsize=8); ax.set_ylabel('y [m]', fontsize=8)

# shared colour scale across all species so magnitudes are comparable
vmax_drho = max(float(drho_nad[s].max()) for s in range(N_SPECIES))
vmax_drho = max(vmax_drho, 1e-8)

for s, sp_name in enumerate(SPECIES_LIST):
    ax    = axes4[s + 1]
    img_s = drho_nad[s]
    im = ax.imshow(img_s, origin='lower', extent=extent,
                   cmap='YlGn', vmin=0, vmax=vmax_drho,
                   aspect='equal', interpolation='nearest')
    ax.contour(density_smooth, levels=[crown_thresh], colors=['white'],
               linewidths=[0.5], alpha=0.55, origin='lower', extent=extent)
    fig4.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=7)
    ax.set_title(f'{sp_name}\nmean={img_s.mean():.5f}', fontsize=9,
                 color=SP_COLOR[sp_name])
    ax.set_xlabel('x [m]', fontsize=8)
    ax.tick_params(labelsize=7)

fig4.suptitle(
    f'JBS Nadir — Per-species BRF contribution  [GPU]\n'
    f'band={args.band}  SZA={SZA:.0f}  {IMG}x{IMG}  N_spp={N_SPP}  P_LAI={args.p_lai}',
    fontsize=10, y=1.01,
)
plt.tight_layout()
out4 = 'jbs_species_brf_contribution.png'
plt.savefig(out4, dpi=130, bbox_inches='tight')
print(f'Figure 4 saved -> {out4}')

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Nadir — dfic vs dedg per species
# ─────────────────────────────────────────────────────────────────────────────
fig2, axes2 = plt.subplots(N_SPECIES, 3, figsize=(14, 3.5 * N_SPECIES))

for s, sp_name in enumerate(SPECIES_LIST):
    # per-row (per-species) scale so dfic/dedg/dopt are comparable within a species
    abs_s = max(float(np.abs(dfic_nad[s]).max()),
                float(np.abs(dedg_nad[s]).max()),
                float(np.abs(dopt_nad[s]).max()), 1e-8)
    for col, (img_s, lbl) in enumerate([
        (dfic_nad[s], 'dfic'), (dedg_nad[s], 'dedg'), (dopt_nad[s], 'dopt')
    ]):
        ax = axes2[s, col]
        norm = TwoSlopeNorm(vmin=-abs_s, vcenter=0, vmax=abs_s)
        im = ax.imshow(img_s, origin='lower', extent=extent,
                       cmap='RdBu_r', norm=norm, aspect='equal',
                       interpolation='nearest')
        ax.contour(density_smooth, levels=[crown_thresh], colors=['white'],
                   linewidths=[0.5], alpha=0.55, origin='lower', extent=extent)
        fig2.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=7)
        ax.set_title(f'{sp_name} — {lbl}\nmean={img_s.mean():+.5f}',
                     fontsize=8, color=SP_COLOR[sp_name])
        ax.set_xlabel('x [m]', fontsize=7)
        if col == 0:
            ax.set_ylabel('y [m]', fontsize=7)
        ax.tick_params(labelsize=6)

fig2.suptitle(
    f'JBS Nadir — Per-species dfic / dedg / dopt  [band={args.band}]',
    fontsize=11, y=1.005,
)
plt.tight_layout()
out2 = 'jbs_species_fic_edg.png'
plt.savefig(out2, dpi=130, bbox_inches='tight')
print(f'Figure 2 saved -> {out2}')

# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: 3 views × 7 species dopt
# ─────────────────────────────────────────────────────────────────────────────
n_views = len(VIEWS)
fig3 = plt.figure(figsize=(4.0 * N_SPECIES, 4.0 * n_views))
gs3  = gridspec.GridSpec(n_views, N_SPECIES, figure=fig3, hspace=0.45, wspace=0.25)

# per-column (per-species) scale across all views — comparable across viewing angles
abs_dopt_sp = [
    max(max(float(np.abs(all_results[v][3][s]).max()) for v in range(n_views)), 1e-8)
    for s in range(N_SPECIES)
]

for row, (view, (brf_v, _, _, dopt_v, _drho_v)) in enumerate(zip(VIEWS, all_results)):
    for col, sp_name in enumerate(SPECIES_LIST):
        ax  = fig3.add_subplot(gs3[row, col])
        img_s = dopt_v[col]
        norm  = TwoSlopeNorm(vmin=-abs_dopt_sp[col], vcenter=0, vmax=abs_dopt_sp[col])
        im = ax.imshow(img_s, origin='lower', extent=extent,
                       cmap='RdBu_r', norm=norm, aspect='equal',
                       interpolation='nearest')
        ax.contour(density_smooth, levels=[crown_thresh], colors=['white'],
                   linewidths=[0.4], alpha=0.5, origin='lower', extent=extent)
        fig3.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=6)
        title = f'{sp_name}\n' if row == 0 else ''
        title += f'{view["label"][:10]}\nmean={img_s.mean():+.5f}'
        ax.set_title(title, fontsize=7, color=SP_COLOR[sp_name])
        ax.tick_params(labelsize=6)
        if col == 0:
            ax.set_ylabel('y [m]', fontsize=7)
        if row == n_views - 1:
            ax.set_xlabel('x [m]', fontsize=7)

fig3.suptitle(
    f'JBS — Per-species dBRF/dLAI_s (optimal)  [band={args.band}  {IMG}x{IMG}  N_spp={N_SPP}]',
    fontsize=11, y=1.005,
)
out3 = 'jbs_species_views.png'
plt.savefig(out3, dpi=120, bbox_inches='tight')
print(f'Figure 3 saved -> {out3}')

# ─────────────────────────────────────────────────────────────────────────────
# Print per-species statistics table
# ─────────────────────────────────────────────────────────────────────────────
print('\n  Per-species spatial statistics (nadir view):')
print(f'  {"Species":<8}  {"LAI_Max":>8}  {"BRF_contrib":>12}  {"dfic_mean":>12}  {"dedg_mean":>12}  {"dopt_mean":>12}  {"dopt_std":>10}')
print('  ' + '-' * 88)
for s, sp_name in enumerate(SPECIES_LIST):
    lai_s = species_stats['LAI_Max_sp'][s]
    print(f'  {sp_name:<8}  {lai_s:8.4f}  '
          f'{drho_nad[s].mean():12.6f}  '
          f'{dfic_nad[s].mean():12.6f}  '
          f'{dedg_nad[s].mean():12.6f}  '
          f'{dopt_nad[s].mean():12.6f}  '
          f'{dopt_nad[s].std():10.6f}')

print('\nDone.')

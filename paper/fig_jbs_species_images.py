"""
jbs_species_images.py
======================
High-resolution BRF + per-species dBRF/dLAI images  —  nadir & hotspot.

Renders 512×512 with 18 groups, aggregates to 7 species, saves:
  • Combined figures (2×4 grid)
  • Individual EPS per species (no title — caption added in article)

Usage:
    python jbs_species_images.py --band O17
    python jbs_species_images.py --band O08 --n-spp 64
"""
import sys, os, time, argparse, shutil
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from canopi.scenes.jbs_species_loader import (
    load_jbs_scene_groups,
    GROUP_LIST, N_GROUPS, GROUP_TO_SPECIES,
    SPECIES_LIST, N_SPECIES,
    BANDS, ILLUMINATION, LX, LY,
)
from canopi.geometry import polar_vect
from canopi.paths import figure_dir, scene_dir

try:
    from canopi.gpu import GPU_AVAILABLE, render_image_bwd_deriv_species_gpu
except ImportError:
    GPU_AVAILABLE = False

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--scene-dir', default=None,
                    help='RAMI-5 treelocations dir (default: auto-detect)')
parser.add_argument('--band',      default='O17', choices=BANDS)
parser.add_argument('--month',     default='Jul', choices=['Apr','Jul'])
parser.add_argument('--p-lai',     type=float, default=0.5)
parser.add_argument('--res',       type=int,   default=512)
parser.add_argument('--n-spp',     type=int,   default=32)
parser.add_argument('--max-leaves',type=int,   default=None)
parser.add_argument('--seed',      type=int,   default=42)
args = parser.parse_args()

IMG  = args.res
BAND = args.band
NSPP = args.n_spp

print('=' * 70)
print(f'  JBS species images  band={BAND}  {IMG}×{IMG}  N_spp={NSPP}')
print('=' * 70)

if not GPU_AVAILABLE:
    raise RuntimeError('GPU required.')

ART = str(figure_dir())

# ── Load scene ────────────────────────────────────────────────────────────────
t0 = time.time()
scene, wood, leaf_gid, group_stats = load_jbs_scene_groups(
    scene_dir=args.scene_dir or scene_dir(), band=BAND, month=args.month,
    P_LAI=args.p_lai, max_leaves=args.max_leaves, seed=args.seed, verbose=True,
)
scene.std_LAI = 0.0
print(f'Loaded in {time.time()-t0:.1f}s')

illum = ILLUMINATION[args.month]
SZA   = illum['sun_zenith_deg']
SAA   = illum['sun_azimuth_deg']
deg   = np.pi / 180
vec_sun = polar_vect((180.0 - SZA) * deg, (SAA + 180.0) * deg)

VIEWS = {
    'nadir':   polar_vect(0.0,            0.0),
    'hotspot': polar_vect(SZA * deg,      SAA * deg),
}

# ── Smoothing & display parameters ───────────────────────────────────────────
SMOOTH  = 4.0   # Gaussian sigma [pixels] to remove 1 m voxel-grid steps
SP_COLOR = {
    'BEPE':'#2166AC','TICO':'#4DAF4A','ALGL':'#FF7F00',
    'POTR':'#984EA3','FREX':'#E41A1C','ACPL':'#795548','PIAB':'#009688',
}
SP_FULL = {
    'BEPE':'B. pendula', 'TICO':'T. cordata',  'ALGL':'A. glutinosa',
    'POTR':'P. tremuloides','PIAB':'P. abies', 'FREX':'F. excelsior',
    'ACPL':'A. platanoides',
}
extent = [0, LX, 0, LY]

# ── Crown density (for internal use only) ────────────────────────────────────
lp = scene.leaf_pos
ix = (lp[:,0] / LX * IMG).astype(int).clip(0, IMG-1)
iy = (lp[:,1] / LY * IMG).astype(int).clip(0, IMG-1)
density = np.bincount(iy*IMG + ix, minlength=IMG*IMG).reshape(IMG,IMG).astype(float)
density = gaussian_filter(density, sigma=3.0)

# ── Helpers ───────────────────────────────────────────────────────────────────
def save_to_art(fname):
    shutil.copy(fname, os.path.join(ART, fname))

def render_or_load(view_name, obs_dir):
    cache = f'jbs_{BAND}_{IMG}_{view_name}_raw.npz'
    if os.path.exists(cache):
        print(f'  Loading cache: {cache}')
        d = np.load(cache)
        return d['brf'], d['dfic'], d['dedg'], d['dopt'], d['drho']
    print(f'  Rendering {view_name} {IMG}×{IMG} N_spp={NSPP} ...')
    t0 = time.time()
    brf, dfic, dedg, dopt, drho = render_image_bwd_deriv_species_gpu(
        scene, obs_dir, vec_sun, leaf_gid, group_stats,
        img_H=IMG, img_W=IMG, N_spp=NSPP, max_depth=50, seed=args.seed,
        rho_sp_arr=group_stats['rho_sp_arr'],
    )
    print(f'  Done in {time.time()-t0:.1f}s  BRF_mean={brf.mean():.4f}')
    np.savez_compressed(cache, brf=brf, dfic=dfic, dedg=dedg, dopt=dopt, drho=drho)
    return brf, dfic, dedg, dopt, drho

def aggregate(dopt_grp, drho_grp):
    dopt_sp = np.zeros((N_SPECIES, IMG, IMG))
    drho_sp = np.zeros((N_SPECIES, IMG, IMG))
    for g, grp in enumerate(GROUP_LIST):
        s = SPECIES_LIST.index(GROUP_TO_SPECIES[grp])
        dopt_sp[s] += dopt_grp[g]
        drho_sp[s] += drho_grp[g]
    dopt_sp = np.stack([gaussian_filter(dopt_sp[s], SMOOTH) for s in range(N_SPECIES)])
    drho_sp = np.stack([gaussian_filter(drho_sp[s], SMOOTH) for s in range(N_SPECIES)])
    return dopt_sp, drho_sp

def save_individual(img_data, cmap, norm_or_vmax, sp, view, kind, xlabel=True):
    """Save one species panel as a standalone EPS (no title)."""
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    if isinstance(norm_or_vmax, tuple):
        vmin, vmax = norm_or_vmax
        im = ax.imshow(img_data, origin='lower', extent=extent,
                       cmap=cmap, vmin=vmin, vmax=vmax,
                       aspect='equal', interpolation='bilinear')
    else:
        im = ax.imshow(img_data, origin='lower', extent=extent,
                       cmap=cmap, norm=norm_or_vmax,
                       aspect='equal', interpolation='bilinear')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=7)
    ax.set_xlabel('$x$ [m]', fontsize=9)
    ax.set_ylabel('$y$ [m]', fontsize=9)
    ax.tick_params(labelsize=8)
    plt.tight_layout(pad=0.3)
    stem = f'jbs_{BAND}_{view}_{kind}_{sp}'
    for ext in ('eps', 'pdf', 'png'):
        fig.savefig(f'{stem}.{ext}', format=ext,
                    dpi=200 if ext == 'png' else 150,
                    bbox_inches='tight')
    save_to_art(f'{stem}.eps')
    plt.close()
    print(f'    {stem}.eps')

def save_combined(brf, dopt_sp, drho_sp, view):
    """Save 2×4 combined figures (BRF contrib + derivative)."""
    vmax_c = max(float(drho_sp[s].max()) for s in range(N_SPECIES))

    # — BRF contribution figure —
    fig, axes = plt.subplots(2, 4, figsize=(14, 7.5))
    axes = axes.ravel()
    ax = axes[0]
    im = ax.imshow(brf, origin='lower', extent=extent, cmap='YlGn',
                   vmin=0, vmax=brf.max(), aspect='equal', interpolation='bilinear')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=7)
    ax.set_xlabel('$x$ [m]', fontsize=8); ax.set_ylabel('$y$ [m]', fontsize=8)
    for s, sp in enumerate(SPECIES_LIST):
        ax = axes[s+1]
        im = ax.imshow(drho_sp[s], origin='lower', extent=extent, cmap='YlGn',
                       vmin=0, vmax=max(vmax_c,1e-8), aspect='equal',
                       interpolation='bilinear')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=7)
        ax.set_title(f'{sp} — {SP_FULL[sp]}\nmean={drho_sp[s].mean():.5f}',
                     fontsize=8, color=SP_COLOR[sp])
        ax.set_xlabel('$x$ [m]', fontsize=7); ax.tick_params(labelsize=7)
    fig.suptitle(f'JBS {view} — per-species BRF contribution  [{BAND}  {IMG}×{IMG}]',
                 fontsize=11, y=1.01)
    plt.tight_layout()
    stem = f'jbs_{BAND}_{view}_brf_contrib_species'
    for ext in ('eps','pdf','png'):
        fig.savefig(f'{stem}.{ext}', format=ext,
                    dpi=200 if ext=='png' else 150, bbox_inches='tight')
    save_to_art(f'{stem}.eps')
    plt.close()
    print(f'  Combined BRF contrib → {stem}.eps')

    # — Derivative figure —
    fig, axes = plt.subplots(2, 4, figsize=(14, 7.5))
    axes = axes.ravel()
    ax = axes[0]
    im = ax.imshow(brf, origin='lower', extent=extent, cmap='YlGn',
                   vmin=0, vmax=brf.max(), aspect='equal', interpolation='bilinear')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=7)
    ax.set_xlabel('$x$ [m]', fontsize=8); ax.set_ylabel('$y$ [m]', fontsize=8)
    for s, sp in enumerate(SPECIES_LIST):
        ax = axes[s+1]
        img = dopt_sp[s]
        abs_v = max(float(np.abs(img).max()), 1e-8)
        norm  = TwoSlopeNorm(vmin=-abs_v, vcenter=0, vmax=abs_v)
        im = ax.imshow(img, origin='lower', extent=extent, cmap='seismic',
                       norm=norm, aspect='equal', interpolation='bilinear')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=7)
        ax.set_title(f'{sp} — {SP_FULL[sp]}\nmean={img.mean():+.5f}',
                     fontsize=8, color=SP_COLOR[sp])
        ax.set_xlabel('$x$ [m]', fontsize=7); ax.tick_params(labelsize=7)
    fig.suptitle(f'JBS {view} — per-species dBRF/dLAI  [{BAND}  {IMG}×{IMG}]',
                 fontsize=11, y=1.01)
    plt.tight_layout()
    stem = f'jbs_{BAND}_{view}_dopt_species'
    for ext in ('eps','pdf','png'):
        fig.savefig(f'{stem}.{ext}', format=ext,
                    dpi=200 if ext=='png' else 150, bbox_inches='tight')
    save_to_art(f'{stem}.eps')
    plt.close()
    print(f'  Combined derivative  → {stem}.eps')

# ── Main loop: nadir then hotspot ─────────────────────────────────────────────
for view_name, obs_dir in VIEWS.items():
    print(f'\n{"─"*60}')
    print(f'  View: {view_name}')
    print(f'{"─"*60}')

    brf, dfic, dedg, dopt, drho = render_or_load(view_name, obs_dir)
    dopt_sp, drho_sp = aggregate(dopt, drho)

    # Print stats
    for s, sp in enumerate(SPECIES_LIST):
        print(f'  {sp}: dopt={dopt_sp[s].mean():+.6f}  contrib={drho_sp[s].mean():.6f}')

    # ── Total BRF standalone ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(4.0, 4.0))
    im = ax.imshow(brf, origin='lower', extent=extent, cmap='YlGn',
                   vmin=0, vmax=brf.max(), aspect='equal', interpolation='bilinear')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label('BRF', fontsize=9)
    ax.set_xlabel('$x$ [m]', fontsize=9); ax.set_ylabel('$y$ [m]', fontsize=9)
    ax.tick_params(labelsize=8)
    plt.tight_layout(pad=0.3)
    stem = f'jbs_{BAND}_{view_name}_brf'
    for ext in ('eps','pdf','png'):
        fig.savefig(f'{stem}.{ext}', format=ext,
                    dpi=200 if ext=='png' else 150, bbox_inches='tight')
    save_to_art(f'{stem}.eps')
    plt.close()
    print(f'  BRF standalone → {stem}.eps')

    # ── Individual EPS per species (no title) ────────────────────────────────
    vmax_c = max(float(drho_sp[s].max()) for s in range(N_SPECIES))
    print('  Individual species EPS:')
    for s, sp in enumerate(SPECIES_LIST):
        # BRF contribution
        save_individual(drho_sp[s], 'YlGn', (0, max(vmax_c,1e-8)),
                        sp, view_name, 'contrib')
        # Derivative (per-species symmetric scale)
        img   = dopt_sp[s]
        abs_v = max(float(np.abs(img).max()), 1e-8)
        norm  = TwoSlopeNorm(vmin=-abs_v, vcenter=0, vmax=abs_v)
        save_individual(img, 'seismic', norm, sp, view_name, 'dopt')

    # ── Combined figure ───────────────────────────────────────────────────────
    save_combined(brf, dopt_sp, drho_sp, view_name)

print('\n\nAll done.')
print(f'EPS files → {ART}')

"""
replot_inversion_figures.py
===========================
Regenerate all EPS figures for the two-band inversion subsection
(LAI-only and joint LAI+rho, P_init=0.35, k_max=0) with:
  - LaTeX text everywhere (text.usetex=True)
  - Larger fonts (14 pt labels, 12 pt ticks, 11-12 pt legend)
  - No subplot titles (captions already in LaTeX subfigure)
  - Total LAI error curve added to lai_error.eps
  - Legend at lower-centre for lai_conv.eps and rho_O08_error.eps
  - O17 BRF images: vmax=1 for obs/init/final
  - O08/O17 init images included (needed for 4-per-row article figures)

Usage:
    python replot_inversion_figures.py
"""
import sys, os, gc, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from numba import cuda

HERE  = os.path.dirname(os.path.abspath(__file__))
ART   = str(figure_dir())

# ── Global LaTeX style ────────────────────────────────────────────────────────
# Figures are scaled down ~0.45× in LaTeX (\includegraphics[width=0.32\linewidth]).
# A 22pt label becomes ~10pt in print — minimum readable size.
# A 16pt legend becomes ~7pt — readable but tight.
matplotlib.rcParams.update({
    'text.usetex'          : True,
    'text.latex.preamble'  : r'\usepackage{amsmath}',
    'font.family'          : 'serif',
    'font.size'            : 22,
    'axes.labelsize'       : 22,
    'xtick.labelsize'      : 18,
    'ytick.labelsize'      : 18,
    'legend.fontsize'      : 14,
    'legend.framealpha'    : 0.85,
    'lines.markersize'     : 5,
    'lines.linewidth'      : 2.0,
})

SP_COLOR = {
    'BEPE':'#2166AC','TICO':'#4DAF4A','ALGL':'#FF7F00',
    'POTR':'#984EA3','FREX':'#E41A1C','ACPL':'#795548','PIAB':'#009688',
}

# ── Load pkls ─────────────────────────────────────────────────────────────────
PKL_TR  = os.path.join(HERE, 'lai_species_truerho_p035_k0_p0.35_k0.pkl')
PKL_JT  = os.path.join(HERE, 'lai_rho_species_p035_k0_p0.35_k0.pkl')

with open(PKL_TR, 'rb') as f:
    dtr = pickle.load(f)
with open(PKL_JT, 'rb') as f:
    djt = pickle.load(f)

SPECIES_LIST   = dtr['SPECIES_LIST']
N_SPECIES      = len(SPECIES_LIST)
colors_s       = [SP_COLOR[sp] for sp in SPECIES_LIST]
LAI_Max_sp     = dtr['LAI_Max_sp']
LAI_eff_true   = dtr['LAI_eff_true']
LAI_true_total = float(LAI_eff_true.sum())

# ── Species name → LaTeX label ────────────────────────────────────────────────
SP_TEX = {
    'BEPE': r'\textit{B.\,pendula}',
    'TICO': r'\textit{T.\,cordata}',
    'ALGL': r'\textit{A.\,glutinosa}',
    'POTR': r'\textit{P.\,tremuloides}',
    'PIAB': r'\textit{P.\,abies}',
    'FREX': r'\textit{F.\,excelsior}',
    'ACPL': r'\textit{A.\,platanoides}',
}

# ── Scene + render setup ──────────────────────────────────────────────────────
from canopi.scenes.jbs_species_loader import (
    load_jbs_scene_groups, GROUP_LIST, N_GROUPS, GROUP_TO_SPECIES,
    SPECIES_TO_ID, ILLUMINATION,
)
from canopi.scenes.jbs_loader import get_band_params, _REFL, _TRAN, _BAND_IDX
from canopi.geometry import polar_vect
from canopi.paths import figure_dir, scene_dir
from canopi.gpu import GPU_AVAILABLE, render_image_bwd_deriv_species_gpu

SEED_BASE = 42
P_LAI_TRUE = 0.5; H = W = 96; N_spp = 8
BAND_O08 = get_band_params('O08'); BAND_O17 = get_band_params('O17')
SEED_O08 = SEED_BASE + 1

illum = ILLUMINATION['Jul']
deg = np.pi / 180
vsun = polar_vect((180 - illum['sun_zenith_deg'])*deg,
                  (illum['sun_azimuth_deg'] + 180)*deg)
obs_nadir = np.array([0., 0., 1.])

print('Loading JBS scene ...')
t0 = time.time()
scene, wood, leaf_gid, grp_stats = load_jbs_scene_groups(
    scene_dir=scene_dir(), band='O08', month='Jul',
    P_LAI=P_LAI_TRUE, seed=SEED_BASE, verbose=False)
scene.std_LAI = 0.0
LAI_Max_grp = grp_stats['LAI_Max_sp']
g_to_s = np.array([SPECIES_TO_ID[GROUP_TO_SPECIES[g]] for g in GROUP_LIST])
G2S = np.zeros((N_SPECIES, N_GROUPS))
for g in range(N_GROUPS): G2S[g_to_s[g], g] = 1.
bi08 = _BAND_IDX['O08']; bi17 = _BAND_IDX['O17']
rho08_true = np.array([_REFL[s][bi08] for s in SPECIES_LIST])
rho17_true = np.array([_REFL[s][bi17] for s in SPECIES_LIST])
kap08 = np.array([_TRAN[s][bi08]/_REFL[s][bi08] for s in SPECIES_LIST])
kap17 = np.array([_TRAN[s][bi17]/_REFL[s][bi17] for s in SPECIES_LIST])
k08g = kap08[g_to_s]; k17g = kap17[g_to_s]
print(f'  Scene loaded in {time.time()-t0:.1f}s')


def rndr(p_sp, rho_sp, kg, band, seed):
    gc.collect(); cuda.current_context().deallocations.clear()
    pg = p_sp[g_to_s]; rg = rho_sp[g_to_s]; re = rg*(1+kg)/2
    lai = LAI_Max_grp*pg; ls = max(float(lai.sum()), 1e-12)
    scene.rho   = float(np.clip(np.dot(lai, rg)/ls, 1e-5, .999))
    scene.tau   = float(np.clip(np.dot(lai, kg*rg)/ls, 1e-5, .999))
    scene.rsoil = float(band['rsoil'])
    I, _, _, _, _ = render_image_bwd_deriv_species_gpu(
        scene, obs_nadir, vsun, leaf_gid, grp_stats,
        img_H=H, img_W=W, N_spp=N_spp, max_depth=50, seed=seed,
        P_LAI_sp=pg.astype(np.float64), rho_sp_arr=re.astype(np.float64))
    return I


# ── Render images ─────────────────────────────────────────────────────────────
p_true = np.full(N_SPECIES, P_LAI_TRUE)

print('Rendering TRUE-RHO images ...')
I_obs_O08 = rndr(p_true, rho08_true, k08g, BAND_O08, SEED_O08)
I_obs_O17 = rndr(p_true, rho17_true, k17g, BAND_O17, SEED_BASE)

p_init_tr   = dtr['p_hist_a'][0]
rho08_init_tr = dtr['rho_O08_hist_a'][0]
rho17_init_tr = dtr['rho_O17_hist_a'][0]
p_fin_tr    = dtr['p_hist_a'][-1]
rho08_fin_tr  = dtr['rho_O08_hist_a'][-1]
rho17_fin_tr  = dtr['rho_O17_hist_a'][-1]

I_init_O08_tr = rndr(p_init_tr, rho08_init_tr, k08g, BAND_O08, SEED_O08)
I_init_O17_tr = rndr(p_init_tr, rho17_init_tr, k17g, BAND_O17, SEED_BASE)
I_fin_O08_tr  = rndr(p_fin_tr,  rho08_fin_tr,  k08g, BAND_O08, SEED_O08)
I_fin_O17_tr  = rndr(p_fin_tr,  rho17_fin_tr,  k17g, BAND_O17, SEED_BASE)

print('Rendering JOINT images ...')
p_init_jt    = djt['p_hist_a'][0]
rho08_init_jt = djt['rho_O08_hist_a'][0]
rho17_init_jt = djt['rho_O17_hist_a'][0]
p_fin_jt     = djt['p_hist_a'][-1]
rho08_fin_jt  = djt['rho_O08_hist_a'][-1]
rho17_fin_jt  = djt['rho_O17_hist_a'][-1]

I_init_O08_jt = rndr(p_init_jt, rho08_init_jt, k08g, BAND_O08, SEED_O08)
I_init_O17_jt = rndr(p_init_jt, rho17_init_jt, k17g, BAND_O17, SEED_BASE)
I_fin_O08_jt  = rndr(p_fin_jt,  rho08_fin_jt,  k08g, BAND_O08, SEED_O08)
I_fin_O17_jt  = rndr(p_fin_jt,  rho17_fin_jt,  k17g, BAND_O17, SEED_BASE)
print('All renders done.')


# ── Save helpers ──────────────────────────────────────────────────────────────
def _save(fig, name):
    eps = os.path.join(HERE, name)
    fig.savefig(eps, format='eps', bbox_inches='tight')
    plt.close(fig)
    import shutil
    shutil.copy(eps, os.path.join(ART, name))
    print(f'  {name}')


def _img_eps(name, img, cmap='YlGn', vmin=None, vmax=None):
    """BRF or residual image: no title, large colorbar font.
    Figures are ~0.35x scaled in LaTeX 4-column → need 20pt ticks for 7pt output."""
    f, a = plt.subplots(figsize=(4.5, 4.0))
    im = a.imshow(img, origin='lower', interpolation='nearest',
                  cmap=cmap, vmin=vmin, vmax=vmax)
    a.set_xticks([]); a.set_yticks([])
    cb = f.colorbar(im, ax=a, fraction=0.046, pad=0.04, format='%.2f')
    cb.ax.tick_params(labelsize=20)
    f.savefig(os.path.join(HERE, name), format='eps', bbox_inches='tight')
    plt.close(f)
    import shutil
    shutil.copy(os.path.join(HERE, name), os.path.join(ART, name))
    print(f'  {name}')


# ─────────────────────────────────────────────────────────────────────────────
# FIRST FIGURE: LAI-only curves
# ─────────────────────────────────────────────────────────────────────────────
print('\n--- First figure: LAI-only curves ---')

p_hist_tr   = dtr['p_hist_a']
err_lai_tr  = dtr['err_lai_a']
rmse_O08_tr = dtr['rmse_O08_h']
rmse_O17_tr = dtr['rmse_O17_h']
cost_tr     = dtr['cost_hist']
n_done_tr   = len(cost_tr)

x_pts_tr = np.arange(len(p_hist_tr))

# (a) LAI convergence — legend lower-centre
fig, ax = plt.subplots(figsize=(5, 4.5))
for s, sp in enumerate(SPECIES_LIST):
    ax.plot(x_pts_tr, p_hist_tr[:, s] * LAI_Max_sp[s],
            '-o', color=colors_s[s], lw=2.0, ms=4, label=SP_TEX[sp])
    ax.axhline(LAI_eff_true[s], ls='--', lw=1.0, color=colors_s[s], alpha=0.6)
ax.set_xlabel(r'Iteration')
ax.set_ylabel(r'$\mathrm{LAI}_{\mathrm{eff},s}$')
ax.set_ylim(bottom=0)
ax.legend(fontsize=12, ncol=2, loc='lower center',
          bbox_to_anchor=(0.5, 0.01), framealpha=0.85)
ax.grid(alpha=0.3)
fig.tight_layout(pad=0.5)
_save(fig, 'lai_species_truerho_p035_k0_lai_conv.eps')

# (b) LAI error — with total LAI curve
LAI_hist_total_tr = p_hist_tr @ LAI_Max_sp
LAI_err_total_tr  = np.abs(LAI_hist_total_tr - LAI_true_total) / LAI_true_total * 100

fig, ax = plt.subplots(figsize=(5, 4.5))
if len(err_lai_tr):
    for s, sp in enumerate(SPECIES_LIST):
        ax.semilogy(range(n_done_tr), np.maximum(err_lai_tr[:, s], 0.01),
                    '-o', color=colors_s[s], lw=2.0, ms=4, label=SP_TEX[sp])
ax.semilogy(range(len(LAI_err_total_tr)-1), np.maximum(LAI_err_total_tr[1:], 0.01),
            'k-o', lw=3.0, ms=5, label=r'\textbf{Total}')
ax.axhline(1.0, ls=':', lw=1.5, color='gray')
ax.set_xlabel(r'Iteration')
ax.set_ylabel(r'$|\mathrm{LAI}_s\;\mathrm{error}|$ (\%)')
ax.legend(fontsize=12, ncol=2, framealpha=0.85)
ax.grid(True, which='both', alpha=0.3)
fig.tight_layout(pad=0.5)
_save(fig, 'lai_species_truerho_p035_k0_lai_error.eps')

# (c) Per-band RMSE
fig, ax = plt.subplots(figsize=(5, 4.5))
ax.semilogy(range(n_done_tr), rmse_O08_tr, 'r-o', ms=5, lw=2.0,
            label=r'O08 ($665\,\mathrm{nm}$)')
ax.semilogy(range(n_done_tr), rmse_O17_tr, 'b-s', ms=5, lw=2.0,
            label=r'O17 ($865\,\mathrm{nm}$)')
ax.set_xlabel(r'Iteration')
ax.set_ylabel(r'RMSE')
ax.legend(fontsize=14, framealpha=0.85)
ax.grid(True, which='both', alpha=0.3)
fig.tight_layout(pad=0.5)
_save(fig, 'lai_species_truerho_p035_k0_rmse.eps')

# ─────────────────────────────────────────────────────────────────────────────
# SECOND FIGURE: LAI-only BRF images (4 per row: obs/init/final/resid)
# ─────────────────────────────────────────────────────────────────────────────
print('\n--- Second figure: LAI-only images ---')

vo0 = min(I_obs_O08.min(), I_init_O08_tr.min())
vo1 = max(I_obs_O08.max(), I_init_O08_tr.max())
vn_max = 1.0  # O17 forced to [0, 1]

res_O08_tr = I_fin_O08_tr - I_obs_O08
res_O17_tr = I_fin_O17_tr - I_obs_O17
ar_r = max(float(np.abs(res_O08_tr).max()), 1e-8)
ar_n = max(float(np.abs(res_O17_tr).max()), 1e-8)

_img_eps('lai_species_truerho_p035_k0_O17_obs.eps',   I_obs_O17,    'YlGn', 0, vn_max)
_img_eps('lai_species_truerho_p035_k0_O17_init.eps',  I_init_O17_tr,'YlGn', 0, vn_max)
_img_eps('lai_species_truerho_p035_k0_O17_final.eps', I_fin_O17_tr, 'YlGn', 0, vn_max)
_img_eps('lai_species_truerho_p035_k0_O17_resid.eps', res_O17_tr,   'RdBu_r', -ar_n, ar_n)
_img_eps('lai_species_truerho_p035_k0_O08_obs.eps',   I_obs_O08,    'YlGn', vo0, vo1)
_img_eps('lai_species_truerho_p035_k0_O08_init.eps',  I_init_O08_tr,'YlGn', vo0, vo1)
_img_eps('lai_species_truerho_p035_k0_O08_final.eps', I_fin_O08_tr, 'YlGn', vo0, vo1)
_img_eps('lai_species_truerho_p035_k0_O08_resid.eps', res_O08_tr,   'RdBu_r', -ar_r, ar_r)

# ─────────────────────────────────────────────────────────────────────────────
# THIRD FIGURE: Joint convergence curves
# ─────────────────────────────────────────────────────────────────────────────
print('\n--- Third figure: Joint convergence curves ---')

p_hist_jt     = djt['p_hist_a']
rho08_hist_jt = djt['rho_O08_hist_a']
rho17_hist_jt = djt['rho_O17_hist_a']
err_lai_jt    = djt['err_lai_a']
err_rho_r_jt  = djt['err_rho_r_a']
err_rho_n_jt  = djt['err_rho_n_a']
rmse_O08_jt   = djt['rmse_O08_h']
rmse_O17_jt   = djt['rmse_O17_h']
cost_jt       = djt['cost_hist']
n_done_jt     = len(cost_jt)
rho08_true_jt = djt['rho_O08_true']
rho17_true_jt = djt['rho_O17_true']
x_pts_jt      = np.arange(len(p_hist_jt))


def _conv(ax, hist, true, ylab):
    for s, sp in enumerate(SPECIES_LIST):
        ax.plot(x_pts_jt, hist[:, s], '-o', color=colors_s[s],
                lw=2.0, ms=4, label=SP_TEX[sp])
        ax.axhline(true[s], ls='--', lw=1.0, color=colors_s[s], alpha=0.6)
    ax.set_xlabel(r'Iteration'); ax.set_ylabel(ylab); ax.grid(alpha=0.3)


def _errlog(ax, hist, ylab, loc='best', bbox=None):
    if len(hist):
        for s, sp in enumerate(SPECIES_LIST):
            ax.semilogy(range(n_done_jt), np.maximum(hist[:, s], 0.01),
                        '-o', color=colors_s[s], lw=2.0, ms=4,
                        label=SP_TEX[sp])
    ax.axhline(1.0, ls=':', lw=1.5, color='gray')
    ax.set_xlabel(r'Iteration'); ax.set_ylabel(ylab)
    kw = dict(fontsize=12, ncol=2, loc=loc, framealpha=0.85)
    if bbox is not None:
        kw['bbox_to_anchor'] = bbox
    ax.legend(**kw)
    ax.grid(True, which='both', alpha=0.3)


# (a) LAI convergence — legend lower-centre
fig, ax = plt.subplots(figsize=(5, 4.5))
_conv(ax, p_hist_jt * LAI_Max_sp[None, :], LAI_eff_true,
      r'$\mathrm{LAI}_{\mathrm{eff},s}$')
ax.set_ylim(bottom=0)
ax.legend(fontsize=12, ncol=2, loc='lower center',
          bbox_to_anchor=(0.5, 0.01), framealpha=0.85)
fig.tight_layout(pad=0.5)
_save(fig, 'lai_rho_species_p035_k0_lai_conv.eps')

# (b) rho_O08 convergence
fig, ax = plt.subplots(figsize=(5, 4.5))
_conv(ax, rho08_hist_jt, rho08_true_jt, r'$\rho_{\mathrm{O08}}$')
ax.legend(fontsize=12, ncol=2, framealpha=0.85)
fig.tight_layout(pad=0.5)
_save(fig, 'lai_rho_species_p035_k0_rho_O08_conv.eps')

# (c) rho_O17 convergence
fig, ax = plt.subplots(figsize=(5, 4.5))
_conv(ax, rho17_hist_jt, rho17_true_jt, r'$\rho_{\mathrm{O17}}$')
ax.legend(fontsize=12, ncol=2, framealpha=0.85)
fig.tight_layout(pad=0.5)
_save(fig, 'lai_rho_species_p035_k0_rho_O17_conv.eps')

# (d) LAI error
fig, ax = plt.subplots(figsize=(5, 4.5))
_errlog(ax, err_lai_jt, r'$|\mathrm{LAI}_s\;\mathrm{error}|$ (\%)')
fig.tight_layout(pad=0.5)
_save(fig, 'lai_rho_species_p035_k0_lai_error.eps')

# (e) rho_O08 error — legend lower-centre
fig, ax = plt.subplots(figsize=(5, 4.5))
_errlog(ax, err_rho_r_jt, r'$|\rho_{\mathrm{O08}}\;\mathrm{error}|$ (\%)',
        loc='lower center', bbox=(0.5, 0.01))
fig.tight_layout(pad=0.5)
_save(fig, 'lai_rho_species_p035_k0_rho_O08_error.eps')

# (f) rho_O17 error
fig, ax = plt.subplots(figsize=(5, 4.5))
_errlog(ax, err_rho_n_jt, r'$|\rho_{\mathrm{O17}}\;\mathrm{error}|$ (\%)')
fig.tight_layout(pad=0.5)
_save(fig, 'lai_rho_species_p035_k0_rho_O17_error.eps')

# (g) Per-band RMSE
fig, ax = plt.subplots(figsize=(5, 4.5))
ax.semilogy(range(n_done_jt), rmse_O08_jt, 'r-o', ms=5, lw=2.0,
            label=r'O08 ($665\,\mathrm{nm}$)')
ax.semilogy(range(n_done_jt), rmse_O17_jt, 'b-s', ms=5, lw=2.0,
            label=r'O17 ($865\,\mathrm{nm}$)')
ax.set_xlabel(r'Iteration'); ax.set_ylabel(r'RMSE')
ax.legend(fontsize=14, framealpha=0.85)
ax.grid(True, which='both', alpha=0.3)
fig.tight_layout(pad=0.5)
_save(fig, 'lai_rho_species_p035_k0_rmse.eps')

# (h) Joint cost
fig, ax = plt.subplots(figsize=(5, 4.5))
ax.semilogy(range(n_done_jt), cost_jt, 'k-o', lw=2.0, ms=5)
ax.set_xlabel(r'Iteration')
ax.set_ylabel(r'$\frac{1}{2}(\|r_{\mathrm{O08}}\|^2+\|r_{\mathrm{O17}}\|^2)$')
ax.grid(True, which='both', alpha=0.3)
fig.tight_layout(pad=0.5)
_save(fig, 'lai_rho_species_p035_k0_cost.eps')

# ─────────────────────────────────────────────────────────────────────────────
# FOURTH FIGURE: Joint BRF images (O17 vmax=1)
# ─────────────────────────────────────────────────────────────────────────────
print('\n--- Fourth figure: Joint images ---')

vo0j = min(I_obs_O08.min(), I_init_O08_jt.min())
vo1j = max(I_obs_O08.max(), I_init_O08_jt.max())

res_O08_jt = I_fin_O08_jt - I_obs_O08
res_O17_jt = I_fin_O17_jt - I_obs_O17
ar_rj = max(float(np.abs(res_O08_jt).max()), 1e-8)
ar_nj = max(float(np.abs(res_O17_jt).max()), 1e-8)

_img_eps('lai_rho_species_p035_k0_O17_obs.eps',   I_obs_O17,     'YlGn', 0, 1.0)
_img_eps('lai_rho_species_p035_k0_O17_init.eps',  I_init_O17_jt, 'YlGn', 0, 1.0)
_img_eps('lai_rho_species_p035_k0_O17_final.eps', I_fin_O17_jt,  'YlGn', 0, 1.0)
_img_eps('lai_rho_species_p035_k0_O17_resid.eps', res_O17_jt,    'RdBu_r', -ar_nj, ar_nj)
_img_eps('lai_rho_species_p035_k0_O08_obs.eps',   I_obs_O08,     'YlGn', vo0j, vo1j)
_img_eps('lai_rho_species_p035_k0_O08_init.eps',  I_init_O08_jt, 'YlGn', vo0j, vo1j)
_img_eps('lai_rho_species_p035_k0_O08_final.eps', I_fin_O08_jt,  'YlGn', vo0j, vo1j)
_img_eps('lai_rho_species_p035_k0_O08_resid.eps', res_O08_jt,    'RdBu_r', -ar_rj, ar_rj)

print('\nAll done.')

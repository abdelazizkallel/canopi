"""
Toy joint inversion of LAI and per-band leaf reflectance ρ via Gauss-Newton.

Configurable band set (RED and/or NIR), CHRIS/PROBA multi-angular geometry.
Within each band ρ = τ (symmetric scattering), w = ρ + τ = 2ρ.

Unknowns: θ = [LAI, ρ_b for each active band].  Fitted to (n_bands × n_angles)
observations.  ρ_b affects only band b; LAI is shared across all bands.

Identifiability:
  * Single band (e.g. NIR): LAI and ρ trade off → flat LAI–ρ valley → degenerate;
    only weakly broken by many view angles.
  * Two bands (RED low-ρ soil/gap-dominated + NIR high-ρ scatter-dominated): the
    shared LAI + the two ρ regimes make the Jacobian well-conditioned.
  Angular sampling density and the number of free ρ both control conditioning.

The full result (observations, BRF fits, convergence histories) is cached to a
pickle after each run; `--replot` reloads it and regenerates the figures WITHOUT
re-running the Monte-Carlo inversion.

Output: lai_inversion_toy.png + per-panel EPS for the article.
"""
import sys, os, time, argparse, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataclasses import replace

from canopi import SceneConfig, render
from canopi.renderer import _build_obs_dirs
from canopi.scene import build_scene

try:
    from canopi.gpu import render_gpu, GPU_AVAILABLE
    _render = render_gpu
    _backend = 'GPU'
except ImportError:
    GPU_AVAILABLE = False
    _render = render
    _backend = 'CPU'

# ─────────────────────────────────────────────────────────────────────────────
# Band registry and true parameters
# ─────────────────────────────────────────────────────────────────────────────
LAI_TRUE = 2.0
BAND_DB = {                                   # rho_true (= tau_true), soil reflectance
    'RED': dict(rho_true=0.06, rsoil=0.05),   # chlorophyll absorption (low ρ)
    'NIR': dict(rho_true=0.40, rsoil=0.20),   # near-infrared plateau (high ρ)
}
BAND_COL = {'RED': 'crimson', 'NIR': 'navy'}

# CHRIS/PROBA 5 along-track fly-by view zenith angles (principal plane).
CHRIS_VZA = np.array([-55.0, -36.0, 0.0, 36.0, 55.0])

LAI_INIT  = 0.8
RHO_INIT  = 0.20    # common initial guess for every band's ρ

MAX_ITER  = 20
TOL       = 1e-3
LM_LAMBDA = 0.05
RHO_MIN, RHO_MAX = 0.01, 0.50   # ρ box; upper 0.5 keeps w = ρ+τ ≤ 1

N_OBS  = 40_000
N_ITER = 20_000

BASE_CFG = SceneConfig(
    n_photons=N_OBS, sun_zenith_deg=20.0, sun_azimuth_deg=180.0,
    ALA='UNI', LAI_eff=LAI_TRUE, P_LAI=0.5,
    rho=BAND_DB['NIR']['rho_true'], tau=BAND_DB['NIR']['rho_true'], rsoil=BAND_DB['NIR']['rsoil'],
    leaf_radius=0.02, leaf_radius_std_mul=0.0,
    scene_xy=4.0, scene_z=1.0, voxel_size=0.25,
    photon_start_xy_min_frac=0.25, photon_start_xy_max_frac=0.75,
    obs_zenith_max_deg=55, obs_step_deg=1.0,
    max_depth=80, rr_depth=999,
    compute_derivative=True,
    image_size=(64, 64), seed=100,
)

# ── CLI ──────────────────────────────────────────────────────────────────────
_ap = argparse.ArgumentParser()
_ap.add_argument('--k-max',     dest='k_max',     type=int,   default=9)
_ap.add_argument('--lm-lambda', dest='lm_lambda', type=float, default=LM_LAMBDA)
_ap.add_argument('--max-iter',  dest='max_iter',  type=int,   default=MAX_ITER)
_ap.add_argument('--fix-rho',   dest='fix_rho',   action='store_true',
    help='fix every ρ at its true value and estimate LAI only')
_ap.add_argument('--nir-only',  dest='nir_only',  action='store_true', help='use NIR band only')
_ap.add_argument('--red-only',  dest='red_only',  action='store_true', help='use RED band only')
_ap.add_argument('--lai-init',  dest='lai_init',  type=float, default=LAI_INIT)
_ap.add_argument('--rho-init',  dest='rho_init',  type=float, default=RHO_INIT,
    help='common initial ρ guess for all bands')
_ap.add_argument('--rho-red-init', dest='rho_red_init', type=float, default=None)
_ap.add_argument('--rho-nir-init', dest='rho_nir_init', type=float, default=None)
_ap.add_argument('--lai-true',  dest='lai_true',  type=float, default=LAI_TRUE)
_ap.add_argument('--nadir-only', dest='nadir_only', action='store_true',
    help='use ONLY the nadir view (θ=0)')
_ap.add_argument('--n-angles',  dest='n_angles',  type=int, default=None,
    help='use N view angles evenly spaced over ±55° (overrides the 5 CHRIS angles)')
_ap.add_argument('--rsoil',     dest='rsoil',     type=float, default=None,
    help='override soil reflectance for ALL active bands (e.g. 0 = black soil)')
_ap.add_argument('--sza',       dest='sza',       type=float, default=None,
    help='sun zenith angle [deg] (default 20)')
_ap.add_argument('--ala',       dest='ala',       type=str,   default=None,
    help='leaf angle distribution: UNI/SPH, PLA, ERE (erectophile), HOR, EXT')
_ap.add_argument('--replot',    dest='replot',    action='store_true',
    help='skip the inversion; reload the cached result and only redraw the figures')
args = _ap.parse_args()

LAI_TRUE  = args.lai_true
K_MAX     = args.k_max
LM_LAMBDA = args.lm_lambda
MAX_ITER  = args.max_iter
FIX_RHO   = args.fix_rho
LAI_INIT  = args.lai_init
REPLOT    = args.replot

if   args.nir_only: BANDS = ['NIR']
elif args.red_only: BANDS = ['RED']
else:               BANDS = ['RED', 'NIR']
NB = len(BANDS)

_rho_init_override = {'RED': args.rho_red_init, 'NIR': args.rho_nir_init}
rho_true = np.array([BAND_DB[b]['rho_true'] for b in BANDS])
rho_init = np.array([_rho_init_override[b] if _rho_init_override[b] is not None
                     else args.rho_init for b in BANDS])
rsoil_of = {b: (args.rsoil if args.rsoil is not None else BAND_DB[b]['rsoil'])
            for b in BANDS}

if args.nadir_only:
    CHRIS_VZA = np.array([0.0])
elif args.n_angles is not None:
    CHRIS_VZA = np.linspace(-55.0, 55.0, args.n_angles)

if args.sza is not None or args.ala is not None:
    BASE_CFG = replace(BASE_CFG,
                       sun_zenith_deg=args.sza if args.sza is not None else BASE_CFG.sun_zenith_deg,
                       ALA=args.ala if args.ala is not None else BASE_CFG.ALA)

SEED  = int(BASE_CFG.seed)   # common random numbers → cost(truth)=0 exactly
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lai_inversion_toy.pkl')

# keys saved to / restored from the pickle (everything the figures need)
_CACHE_KEYS = ['obs_angles', 'n_sel', 'NB', 'BANDS', 'rho_true', 'LAI_TRUE',
               'LAI_INIT', 'RHO_MAX', 'y_obs', 'brf_init', 'brf_final',
               'lai_hist', 'rho_hist', 'err_lai_h', 'err_rho_h', 'cost_hist',
               'lai', 'rhos']


def _build(lai, seed):
    rng = np.random.default_rng(seed)
    return build_scene(replace(BASE_CFG, LAI_eff=float(lai)), rng)


def _run_band(lai, rho, rsoil, n_photons, seed, deriv=True, scene=None):
    cfg = replace(BASE_CFG, LAI_eff=float(lai), rho=float(rho), tau=float(rho),
                  rsoil=float(rsoil), n_photons=n_photons, seed=seed,
                  compute_derivative=deriv)
    if GPU_AVAILABLE and scene is not None:
        return render_gpu(cfg, scene=scene)
    return _render(cfg)


def _forward(lai, rhos, n_photons, seed, deriv=False):
    """Render all active bands from a shared scene; return list of RenderResult."""
    sc = _build(lai, seed=seed)
    return [_run_band(lai, rhos[i], rsoil_of[b], n_photons, seed, deriv=deriv, scene=sc)
            for i, b in enumerate(BANDS)]


def _stack_brf(results):
    return np.concatenate([r.brf[sel_idx] for r in results])


def _cost(lai, rhos):
    res = _forward(lai, rhos, N_OBS, SEED, deriv=False)
    rr  = _stack_brf(res) - y_obs
    return 0.5 * float(np.dot(rr, rr))


# ─────────────────────────────────────────────────────────────────────────────
# Inversion (skipped on --replot, which reloads the cache instead)
# ─────────────────────────────────────────────────────────────────────────────
if REPLOT:
    with open(CACHE, 'rb') as f:
        globals().update(pickle.load(f))
    NB = len(BANDS)
    print(f"[replot] loaded {CACHE}:  bands={'+'.join(BANDS)}, {n_sel} angles, "
          f"LAI={lai:.4f}, " + ", ".join(f"ρ_{b}={rhos[i]:.4f}" for i, b in enumerate(BANDS)))
else:
    # ── observation geometry — view angles (principal plane) ────────────────
    zen, _, n_pp = _build_obs_dirs(BASE_CFG)
    tha = zen[:n_pp]
    sel_idx    = np.array([int(np.abs(tha - v).argmin()) for v in CHRIS_VZA])
    n_sel      = len(sel_idx)
    obs_angles = tha[sel_idx]

    print(f"\n{'='*72}")
    print(f"  Joint LAI + ρ Inversion  bands={'+'.join(BANDS)}  [{_backend}]")
    print(f"{'='*72}")
    print(f"  True:  LAI={LAI_TRUE}   " + "  ".join(f"ρ_{b}={rho_true[i]}" for i, b in enumerate(BANDS)))
    print(f"  Init:  LAI={LAI_INIT}   " + "  ".join(f"ρ_{b}={rho_init[i]}" for i, b in enumerate(BANDS)))
    print(f"  {n_sel} view angles: {np.round(obs_angles,1)} °  "
          f"({NB} band(s) × {n_sel} = {NB*n_sel} data pts, {1+NB} unknowns)   ρ∈[{RHO_MIN},{RHO_MAX}]")

    # ── Step 1: synthetic observations at truth ─────────────────────────────
    print(f"\n[1] Rendering observations ...", flush=True)
    t0 = time.time()
    res_obs = _forward(LAI_TRUE, rho_true, N_OBS, SEED, deriv=False)
    y_obs   = _stack_brf(res_obs)
    for i, b in enumerate(BANDS):
        print(f"    {b}: BRF_obs = {np.array2string(res_obs[i].brf[sel_idx], precision=4)}")
    print(f"    done in {time.time()-t0:.1f}s")

    # ── Step 3: Gauss-Newton + two-phase line search ────────────────────────
    _mode = "LAI-only (ρ fixed at true)" if FIX_RHO else f"joint LAI + {NB}×ρ"
    print(f"\n[3] Gauss-Newton + line search [{_mode}] (k_max={K_MAX}, λ={LM_LAMBDA}) ...")
    _rho_cols = "  ".join(f"ρ_{b:>3}" for b in BANDS)
    print(f"  {'it':>3}  {'LAI':>7}  {_rho_cols}  {'|Δθ|∞':>7}  {'cost':>10}  {'amp':>6}  time")
    print(f"  {'-'*(40 + 9*NB)}")

    lai  = float(LAI_INIT)
    rhos = (rho_true.copy() if FIX_RHO else rho_init.copy()).astype(float)

    lai_hist = [lai];  rho_hist = [rhos.copy()];  cost_hist = []
    err_lai_h = [abs(lai - LAI_TRUE)]
    err_rho_h = [np.abs(rhos - rho_true)]
    brf_init = brf_final = None

    for it in range(MAX_ITER):
        t_it = time.time()
        res = _forward(lai, rhos, N_ITER, SEED, deriv=True)
        y_pred = _stack_brf(res)
        r = y_pred - y_obs
        cost = 0.5 * float(np.dot(r, r)); cost_hist.append(cost)
        if it == 0: brf_init = y_pred.copy()
        brf_final = y_pred.copy()

        # Jacobian: col 0 = ∂/∂LAI (all bands); col i+1 = ∂/∂ρ_i (band-i rows only)
        J_lai = np.pi * np.concatenate([res[i].deriv_opt[sel_idx] for i in range(NB)])
        J_cols = [J_lai]
        for i in range(NB):
            blk = [np.zeros(n_sel)] * NB
            blk[i] = res[i].deriv_rho[sel_idx]
            J_cols.append(np.pi * np.concatenate(blk))
        J = np.column_stack(J_cols)                       # (NB·n_sel, 1+NB)

        if FIX_RHO:
            jl = J[:, 0]
            jtj = float(jl @ jl); jtr = float(jl @ r)
            delta = np.zeros(1 + NB)
            delta[0] = -jtr / (jtj * (1.0 + LM_LAMBDA)) if jtj > 1e-30 else 0.0
        else:
            JTJ = J.T @ J; JTr = J.T @ r
            try:
                delta = -np.linalg.solve(JTJ + np.diag(LM_LAMBDA * np.diag(JTJ)), JTr)
            except np.linalg.LinAlgError:
                print("    Singular — stopping"); break
        d_lai = float(delta[0]); d_rho = delta[1:].astype(float)

        # bounds → max amplification keeping LAI∈[0.05,8], ρ∈[RHO_MIN,RHO_MAX]
        def _abound(x, dx, lo, hi):
            if dx > 0: return (hi - 1e-4 - x) / dx
            if dx < 0: return (lo + 1e-4 - x) / dx
            return np.inf
        amp_max = _abound(lai, d_lai, 0.05, 8.0)
        for i in range(NB):
            amp_max = min(amp_max, _abound(rhos[i], d_rho[i], RHO_MIN, RHO_MAX))
        amp_max = max(amp_max, 0.0)

        def _try(amp):
            return float(lai + amp * d_lai), (rhos + amp * d_rho)

        base = _cost(lai, rhos)
        best_cost, best_lai, best_rhos, best_amp = base, lai, rhos.copy(), 0.0
        for k in range(0, K_MAX + 1):                      # amplify
            amp = min(2.0 ** k, amp_max)
            lc, rc = _try(amp)
            if max(abs(lc - lai), float(np.max(np.abs(rc - rhos)))) < 1e-7: break
            c = _cost(lc, rc)
            if c < best_cost: best_cost, best_lai, best_rhos, best_amp = c, lc, rc, amp
            else: break
        if best_amp == 0.0:                                # backtrack
            for k in range(1, 13):
                amp = min(2.0 ** (-k), amp_max)
                lc, rc = _try(amp)
                if max(abs(lc - lai), float(np.max(np.abs(rc - rhos)))) < 1e-9: break
                c = _cost(lc, rc)
                if c < best_cost:
                    best_cost, best_lai, best_rhos, best_amp = c, lc, rc, amp; break

        dmax = max(abs(d_lai), float(np.max(np.abs(d_rho))) if NB else 0.0)
        elapsed = time.time() - t_it
        _rho_str = "  ".join(f"{v:>5.3f}" for v in rhos)
        print(f"  {it:>3}  {lai:>7.4f}  {_rho_str}  {dmax:>7.4f}  {base:>10.6f}  "
              f"x{best_amp:>5.1f}  {elapsed:.1f}s")

        lai, rhos = best_lai, best_rhos
        lai_hist.append(lai); rho_hist.append(rhos.copy())
        err_lai_h.append(abs(lai - LAI_TRUE)); err_rho_h.append(np.abs(rhos - rho_true))

        if best_amp == 0.0:
            print("    No improving step — converged (stuck)."); break
        if best_amp * dmax < TOL:
            print(f"    Converged  |Δθ|={best_amp*dmax:.2e} < tol={TOL}"); break

    print(f"\n  Final  LAI={lai:.5f} (true {LAI_TRUE})   " +
          "  ".join(f"ρ_{b}={rhos[i]:.5f} (true {rho_true[i]})" for i, b in enumerate(BANDS)))

    with open(CACHE, 'wb') as f:
        pickle.dump({k: globals()[k] for k in _CACHE_KEYS}, f)
    print(f"  [cache] saved → {CACHE}  (use --replot to redraw without rerunning)")

# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Figure  (uses the variables produced or reloaded above)
# ─────────────────────────────────────────────────────────────────────────────
iters   = np.arange(len(lai_hist))
iters_e = np.arange(len(cost_hist))
rho_arr = np.array(rho_hist)            # (n_iter+1, NB)
err_rho = np.array(err_rho_h)           # (n_iter+1, NB)
bslice  = {b: slice(i * n_sel, (i + 1) * n_sel) for i, b in enumerate(BANDS)}
ang     = -obs_angles   # principal plane: backscatter / hot-spot side at NEGATIVE VZA

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f"Joint LAI + ρ — bands={'+'.join(BANDS)}, {n_sel} angles  |  "
             f"true LAI={LAI_TRUE}  [{_backend}]", fontsize=10)

ax = axes[0, 0]
for b in BANDS:
    s = bslice[b]; c = BAND_COL[b]
    ax.plot(ang, y_obs[s], 'o', color='k', ms=7)
    ax.plot(ang, brf_init[s], '--', color=c, lw=1.5, label=f'{b} init')
    ax.plot(ang, brf_final[s], '-', color=c, lw=2, label=f'{b} final')
ax.set_xlabel('View zenith angle [°]'); ax.set_ylabel('BRF (○ = observed)')
ax.set_title('(a) Angular signature'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.plot(iters, lai_hist, 'b-o', lw=2, ms=7, label='LAI')
ax.axhline(LAI_TRUE, color='red', ls='--', lw=2, label=f'LAI_true={LAI_TRUE}')
ax.set_xlabel('Iteration'); ax.set_ylabel('LAI estimate')
ax.set_title('(b) LAI convergence'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
ax.set_ylim(0, max(LAI_TRUE, LAI_INIT) * 1.6)

ax = axes[1, 0]
for i, b in enumerate(BANDS):
    ax.plot(iters, rho_arr[:, i], '-s', color=BAND_COL[b], lw=2, ms=6, label=f'ρ_{b}')
    ax.axhline(rho_true[i], color=BAND_COL[b], ls=':', lw=1.5)
ax.set_xlabel('Iteration'); ax.set_ylabel('ρ (= τ)')
ax.set_title('(c) ρ convergence (dotted = true)'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
ax.set_ylim(0, RHO_MAX * 1.1)

ax = axes[1, 1]
ax.semilogy(iters, err_lai_h, 'b-o', lw=2, ms=6, label='|LAI − true|')
for i, b in enumerate(BANDS):
    ax.semilogy(iters, err_rho[:, i], '-s', color=BAND_COL[b], lw=2, ms=6, label=f'|ρ_{b} − true|')
ax.set_xlabel('Iteration'); ax.set_ylabel('|Error|')
ax.set_title('(d) Parameter errors'); ax.legend(fontsize=7); ax.grid(True, which='both', alpha=0.3)

plt.tight_layout()
plt.savefig('lai_inversion_toy.png', dpi=140, bbox_inches='tight')
print(f"\nFigure → lai_inversion_toy.png")

# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Per-panel EPS (LaTeX labels, no titles)
# ─────────────────────────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    'text.usetex': True, 'font.family': 'serif', 'font.size': 22,
    'axes.labelsize': 24, 'xtick.labelsize': 19, 'ytick.labelsize': 19,
    'legend.fontsize': 16, 'lines.linewidth': 2.4, 'lines.markersize': 9,
})


def _save_eps(fname, draw, figsize=(6.6, 5.6)):
    f, a = plt.subplots(figsize=figsize)
    draw(a); a.grid(True, alpha=0.3)
    f.savefig(fname, format='eps', bbox_inches='tight'); plt.close(f)
    print(f"  eps → {fname}")


def _panel_a(a):
    for b in BANDS:
        s = bslice[b]; c = BAND_COL[b]; pre = '' if NB == 1 else b + ' '
        a.plot(ang, brf_init[s],  '--', color='0.55', lw=2, label=pre + r'initial')
        a.plot(ang, y_obs[s],     'ko', ms=10,             label=pre + r'actual')
        a.plot(ang, brf_final[s], '-',  color=c, lw=2,     label=pre + r'final')
    a.set_xlabel(r'View zenith angle [$^\circ$]'); a.set_ylabel(r'$\mathrm{BRF}$')
    a.legend(loc='upper right')


def _panel_b(a):
    a.plot(iters, lai_hist, 'b-o', label=r'LAI iterates')
    a.axhline(LAI_TRUE, color='red', ls='--', label=r'$\mathrm{LAI}_{\mathrm{true}}=%.1f$' % LAI_TRUE)
    a.set_xlabel(r'Iteration'); a.set_ylabel(r'LAI estimate')
    a.legend(); a.set_ylim(0, max(LAI_TRUE, LAI_INIT) * 1.6)


def _panel_c(a):
    for i, b in enumerate(BANDS):
        a.plot(iters, rho_arr[:, i], '-s', color=BAND_COL[b], label=r'$\rho_{\mathrm{%s}}$' % b)
        a.axhline(rho_true[i], color=BAND_COL[b], ls=':')
    a.set_xlabel(r'Iteration'); a.set_ylabel(r'$\rho\;(=\tau)$')
    a.legend(); a.set_ylim(0, RHO_MAX * 1.1)


def _panel_d(a):
    a.semilogy(iters, err_lai_h, 'b-o', label=r'$|\mathrm{LAI}-\mathrm{true}|$')
    for i, b in enumerate(BANDS):
        a.semilogy(iters, err_rho[:, i], '-s', color=BAND_COL[b],
                   label=r'$|\rho_{\mathrm{%s}}-\mathrm{true}|$' % b)
    a.set_xlabel(r'Iteration'); a.set_ylabel(r'$|\mathrm{Error}|$')
    a.legend()


_save_eps('lai_inversion_toy_angular.eps',  _panel_a)
_save_eps('lai_inversion_toy_lai_conv.eps', _panel_b)
_save_eps('lai_inversion_toy_rho_conv.eps', _panel_c)
_save_eps('lai_inversion_toy_error.eps',    _panel_d)

print("Done ✓")

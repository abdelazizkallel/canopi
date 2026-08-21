"""
convergence_fic_edg.py
======================
Root Mean Square Difference (RMSD) between fictive (PR) and edge (IS)
derivative estimators vs N photons.

Both estimators are unbiased → their difference must → 0 as N → ∞.
RMSD(N) = sqrt( E[ (D_fic(N) - D_edg(N))^2 ] )  ∝  1/√N

Parameters fixed:
  P = 0.5, CV (edge sigma) = 0.5, ALA='UNI', rho=tau=0.5, SZA=60

LAI sweep: {0.5}  (then extend to {0.5, 1, 2, 3})

Outputs (parametric names):
  convergence_fic_edg_<TAG>.png
  convergence_fic_edg_<TAG>.eps
  convergence_fic_edg_<TAG>_data.pkl

Run:
  python convergence_fic_edg.py              # full MC + plot
  python convergence_fic_edg.py --plot-only  # load saved data, re-plot
  python convergence_fic_edg.py --add-LAI 1.0 2.0 3.0  # extend to more LAI
"""
import sys, os, time, argparse, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────────────────────────────────────
P               = 0.5
CV              = 0.5        # leaf radius CV → std_mul for compute_deriv_gpu
RHO, TAU, RSOIL = 0.5, 0.5, 1.0
SZA             = 60
OBS_THETA       = 45.0
N_list          = [100, 200, 500, 1000, 2000, 5000, 10000, 20000]
N_RUNS          = 50

LAI_LIST_INIT   = [0.5, 1.0, 2.0, 3.0]    # run all at once

_ALA_TAG = 'SPH'
TAG      = f'P{P:g}_CV{CV:g}_LAD{_ALA_TAG}_SZA{SZA}'
DATA_FILE = os.path.join(HERE, f'convergence_fic_edg_{TAG}_data.pkl')


# ─────────────────────────────────────────────────────────────────────────────
# Scene builder
# ─────────────────────────────────────────────────────────────────────────────
def _build(LAI_eff):
    from canopi.config   import SceneConfig
    from canopi.scene    import build_scene
    from canopi.geometry import polar_vect
    deg = np.pi / 180
    cfg = SceneConfig(
        LAI_eff=LAI_eff, P_LAI=P,
        leaf_radius_std_mul=CV,
        ALA='UNI', rsoil=RSOIL, rho=RHO, tau=TAU, leaf_radius=0.005,
        max_depth=200, rr_depth=999, compute_derivative=True, seed=0,
    )
    scene    = build_scene(cfg, np.random.default_rng(0))
    obs_dirs = np.array([[np.sin(OBS_THETA * deg), 0., np.cos(OBS_THETA * deg)]])
    vec_sun  = polar_vect((180 - SZA) * deg, 180 * deg)
    return cfg, scene, obs_dirs, vec_sun


# ─────────────────────────────────────────────────────────────────────────────
# MC for one LAI value
# ─────────────────────────────────────────────────────────────────────────────
def _run_one_lai(LAI_eff, leaf_dist='lognormal'):
    from canopi.gpu      import compute_deriv_gpu
    from canopi.derivative import optimal_weight

    cfg, scene, obs_dirs, vec_sun = _build(LAI_eff)
    LAI_Max = scene.LAI_Max
    N_max   = max(N_list)
    print(f'  LAI_eff={LAI_eff}  LAI_Max={LAI_Max:.4f}  n_leaf={scene.n_leaf}'
          f'  dist={leaf_dist}')

    diff_runs = {N: [] for N in N_list}
    fic_runs  = {N: [] for N in N_list}
    edg_runs  = {N: [] for N in N_list}

    for i in range(N_RUNS):
        seed = i * 9999
        _, df, de, _ = compute_deriv_gpu(
            scene, obs_dirs, vec_sun, N_max,
            max_depth=cfg.max_depth, seed=seed, leaf_dist=leaf_dist)
        df = df[:, 0];  de = de[:, 0]

        for N in N_list:
            fv = float(np.pi * df[:N].mean() / LAI_Max)
            ev = float(np.pi * de[:N].mean() / LAI_Max)
            fic_runs[N].append(fv)
            edg_runs[N].append(ev)
            diff_runs[N].append(fv - ev)

        if (i + 1) % 10 == 0:
            print(f'    run {i+1:2d}/{N_RUNS}', flush=True)

    rmsd     = {N: float(np.sqrt(np.mean(np.array(diff_runs[N])**2))) for N in N_list}
    mean_fic = {N: float(np.mean(fic_runs[N]))  for N in N_list}
    mean_edg = {N: float(np.mean(edg_runs[N]))  for N in N_list}
    std_fic  = {N: float(np.std(fic_runs[N]))   for N in N_list}
    std_edg  = {N: float(np.std(edg_runs[N]))   for N in N_list}

    return dict(LAI_eff=LAI_eff, LAI_Max=LAI_Max, N_list=N_list,
                rmsd=rmsd, mean_fic=mean_fic, mean_edg=mean_edg,
                std_fic=std_fic, std_edg=std_edg)


# ─────────────────────────────────────────────────────────────────────────────
# Full MC run
# ─────────────────────────────────────────────────────────────────────────────
def run_mc(lai_list):
    from canopi.gpu import GPU_AVAILABLE
    if not GPU_AVAILABLE:
        print('No GPU detected.'); sys.exit(1)

    print(f'P={P}  CV={CV}  rho=tau={RHO}  N_runs={N_RUNS}')
    print(f'N_list={N_list}')
    t0 = time.time()

    results = {}
    for LAI_eff in lai_list:
        print(f'\n=== LAI_eff={LAI_eff} ===')
        results[LAI_eff] = _run_one_lai(LAI_eff)

    print(f'\nTotal: {time.time() - t0:.0f}s')

    data = dict(P=P, CV=CV, RHO=RHO, TAU=TAU, SZA=SZA,
                N_RUNS=N_RUNS, N_list=N_list,
                LAI_LIST=sorted(results.keys()),
                results=results)
    with open(DATA_FILE, 'wb') as f:
        pickle.dump(data, f)
    print(f'Data → {DATA_FILE}')
    return data


def run_add_lai(extra_lais):
    from canopi.gpu import GPU_AVAILABLE
    if not GPU_AVAILABLE:
        print('No GPU detected.'); sys.exit(1)
    if not os.path.exists(DATA_FILE):
        print(f'Data file not found: {DATA_FILE}'); sys.exit(1)
    with open(DATA_FILE, 'rb') as f:
        data = pickle.load(f)
    print(f'Loaded {DATA_FILE}')

    for LAI_eff in extra_lais:
        print(f'\n=== LAI_eff={LAI_eff} ===')
        data['results'][LAI_eff] = _run_one_lai(LAI_eff)

    data['LAI_LIST'] = sorted(data['results'].keys())
    with open(DATA_FILE, 'wb') as f:
        pickle.dump(data, f)
    print(f'Data saved → {DATA_FILE}')
    return data


def run_add_dist(leaf_dist):
    """Run MC for all existing LAI values with a different distribution, store in results_<dist>."""
    from canopi.gpu import GPU_AVAILABLE
    if not GPU_AVAILABLE:
        print('No GPU detected.'); sys.exit(1)
    if not os.path.exists(DATA_FILE):
        print(f'Data file not found: {DATA_FILE}'); sys.exit(1)
    with open(DATA_FILE, 'rb') as f:
        data = pickle.load(f)
    print(f'Loaded {DATA_FILE}')

    key = f'results_{leaf_dist}'
    if key not in data:
        data[key] = {}

    t0 = time.time()
    for LAI_eff in data['LAI_LIST']:
        print(f'\n=== LAI_eff={LAI_eff}  dist={leaf_dist} ===')
        data[key][LAI_eff] = _run_one_lai(LAI_eff, leaf_dist=leaf_dist)

    print(f'\nTotal: {time.time() - t0:.0f}s')
    with open(DATA_FILE, 'wb') as f:
        pickle.dump(data, f)
    print(f'Data saved → {DATA_FILE}')
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────
def make_figures(data):
    LAI_LIST_ = data['LAI_LIST']
    N_list_   = data['N_list']
    Ns        = np.array(N_list_, dtype=float)

    matplotlib.rcParams.update({
        'text.usetex': False,
        'mathtext.fontset': 'cm',
        'font.family': 'serif',
        'font.size': 11,
        'axes.labelsize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
    })

    colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e']

    has_gauss = 'results_gaussian' in data

    def draw(ax, lognormal_only=False, fs_label=13, fs_legend=8, fs_tick=10):
        show_gauss = has_gauss and not lognormal_only
        for k, LAI_eff in enumerate(LAI_LIST_):
            color = colors[k % len(colors)]
            res  = data['results'][LAI_eff]
            rmsd = np.array([res['rmsd'][N] for N in N_list_])
            lbl  = fr'LAI$={LAI_eff}$' if lognormal_only else fr'LAI$={LAI_eff}$ Log-N'
            ax.loglog(Ns, rmsd, color=color, marker='o', ls='-', lw=2, ms=6,
                      label=lbl)
            if show_gauss:
                res_g  = data['results_gaussian'][LAI_eff]
                rmsd_g = np.array([res_g['rmsd'][N] for N in N_list_])
                ax.loglog(Ns, rmsd_g, color=color, ls='--', lw=1.5,
                          label=fr'LAI$={LAI_eff}$ Gauss')

        # 1/√N reference
        res0  = data['results'][LAI_LIST_[0]]
        rmsd0 = np.array([res0['rmsd'][N] for N in N_list_])
        ref   = rmsd0[-1] * np.sqrt(N_list_[-1] / Ns)
        ax.loglog(Ns, ref, color='#aaaaaa', ls=':', lw=1.5,
                  label=r'$1/\sqrt{N}$', zorder=0)

        ax.set_xlabel(r'$N$ photons', fontsize=fs_label)
        ax.set_ylabel(r'RMSD$(D_{\rm fic},\, D_{\rm edg})$', fontsize=fs_label)
        ax.legend(fontsize=fs_legend, loc='upper right', ncol=2 if show_gauss else 1)
        ax.tick_params(labelsize=fs_tick)
        ax.grid(True, which='both', alpha=0.3)

    lognormal_only = data.get('_lognormal_only', False)

    # ── PNG ───────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle(
        fr'Convergence: RMSD$(D_{{\rm fic}},D_{{\rm edg}})$ vs $N$  —  '
        fr'$P$={data["P"]}, CV={data["CV"]}, spherical LAD, '
        fr'$\rho$=$\tau$={data["RHO"]}, SZA={data["SZA"]}°, {data["N_RUNS"]} runs',
        fontsize=9)
    draw(ax, lognormal_only)
    plt.tight_layout()
    out_png = os.path.join(HERE, f'convergence_fic_edg_{TAG}.png')
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'PNG → {out_png}')

    # ── EPS (no title) ────────────────────────────────────────────────────────
    # Included in the paper at 10 cm on a 12.7 cm canvas, so the text was oversized
    # relative to the body font; reduced here.
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    draw(ax2, lognormal_only, fs_label=10, fs_legend=8, fs_tick=8)
    plt.tight_layout()
    out_eps = os.path.join(HERE, f'convergence_fic_edg_{TAG}.eps')
    fig2.savefig(out_eps, format='eps', bbox_inches='tight')
    plt.close(fig2)
    print(f'EPS → {out_eps}')

    make_bias_figure(data)
    print('All done ✓')


# ─────────────────────────────────────────────────────────────────────────────
# Bias figure: mean(D_fic) vs mean(D_edg) for both distributions
# ─────────────────────────────────────────────────────────────────────────────
def make_bias_figure(data):
    """
    Two-panel figure: LogNormal (left) vs Gaussian (right).
    Each panel: mean(D_fic) solid + mean(D_edg) dashed, per LAI value.
    For a conservative medium true value ≈ 0.  A Gaussian bias (truncation
    mismatch) shows up as D_edg staying away from zero while D_fic → 0.
    """
    LAI_LIST_ = data['LAI_LIST']
    N_list_   = data['N_list']
    Ns        = np.array(N_list_, dtype=float)
    has_gauss = 'results_gaussian' in data
    if not has_gauss:
        print('No Gaussian data — skip bias figure.')
        return

    matplotlib.rcParams.update({
        'text.usetex': False, 'mathtext.fontset': 'cm',
        'font.family': 'serif', 'font.size': 11,
        'axes.labelsize': 12, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
        'legend.fontsize': 8,
    })
    colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    fig.suptitle(
        fr'Bias check: mean $D_{{\rm fic}}$ (solid) vs mean $D_{{\rm edg}}$ (dashed)'
        fr'  —  P={data["P"]}, CV={data["CV"]}, '
        fr'$\rho$=$\tau$={data["RHO"]}, SZA={data["SZA"]}°, {data["N_RUNS"]} runs',
        fontsize=9)

    for col, (dist_key, title) in enumerate([
            ('results',          'Log-Normal IS'),
            ('results_gaussian', 'Gaussian IS')]):
        ax = axes[col]
        for k, LAI_eff in enumerate(LAI_LIST_):
            color  = colors[k % len(colors)]
            res    = data[dist_key][LAI_eff]
            mf = np.array([res['mean_fic'][N] for N in N_list_])
            me = np.array([res['mean_edg'][N] for N in N_list_])
            ax.semilogx(Ns, mf, color=color, ls='-',  lw=2, marker='o', ms=5,
                        label=fr'$D_{{\rm fic}}$ LAI={LAI_eff}')
            ax.semilogx(Ns, me, color=color, ls='--', lw=1.5,
                        label=fr'$D_{{\rm edg}}$ LAI={LAI_eff}')
        ax.axhline(0, color='gray', lw=0.8, ls=':')
        ax.set_xlabel(r'$N$ photons', fontsize=12)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=7, loc='best', ncol=2)
        ax.grid(True, which='both', alpha=0.3)

    axes[0].set_ylabel(r'Mean $dBRF/dLAI$', fontsize=12)
    plt.tight_layout()
    out_png = os.path.join(HERE, f'convergence_bias_{TAG}.png')
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Bias PNG → {out_png}')

    # EPS version
    fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for col, (dist_key, title) in enumerate([
            ('results',          'Log-Normal IS'),
            ('results_gaussian', 'Gaussian IS')]):
        ax = axes2[col]
        for k, LAI_eff in enumerate(LAI_LIST_):
            color = colors[k % len(colors)]
            res   = data[dist_key][LAI_eff]
            mf = np.array([res['mean_fic'][N] for N in N_list_])
            me = np.array([res['mean_edg'][N] for N in N_list_])
            ax.semilogx(Ns, mf, color=color, ls='-',  lw=2, marker='o', ms=5,
                        label=fr'$D_{{\rm fic}}$ LAI={LAI_eff}')
            ax.semilogx(Ns, me, color=color, ls='--', lw=1.5,
                        label=fr'$D_{{\rm edg}}$ LAI={LAI_eff}')
        ax.axhline(0, color='gray', lw=0.8, ls=':')
        ax.set_xlabel(r'$N$ photons', fontsize=11)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7, loc='best', ncol=2)
        ax.grid(True, which='both', alpha=0.3)
    axes2[0].set_ylabel(r'Mean $dBRF/dLAI$', fontsize=11)
    plt.tight_layout()
    out_eps = os.path.join(HERE, f'convergence_bias_{TAG}.eps')
    fig2.savefig(out_eps, format='eps', bbox_inches='tight')
    plt.close(fig2)
    print(f'Bias EPS → {out_eps}')


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot-only', action='store_true')
    parser.add_argument('--lognormal-only', action='store_true',
                        help='Plot only LogNormal curves, no Gaussian, no dist label')
    parser.add_argument('--add-LAI', nargs='+', type=float, metavar='LAI',
                        help='Run MC only for these LAI values and merge')
    parser.add_argument('--add-dist', type=str, metavar='DIST',
                        help='Run MC for all existing LAI with this distribution (gaussian/lognormal)')
    args = parser.parse_args()

    if args.add_dist:
        data = run_add_dist(args.add_dist)
    elif args.add_LAI:
        data = run_add_lai(args.add_LAI)
    elif args.plot_only or args.lognormal_only:
        if not os.path.exists(DATA_FILE):
            print(f'Data file not found: {DATA_FILE}'); sys.exit(1)
        with open(DATA_FILE, 'rb') as f:
            data = pickle.load(f)
        print(f'Loaded {DATA_FILE}')
    else:
        data = run_mc(LAI_LIST_INIT)

    data['_lognormal_only'] = args.lognormal_only
    make_figures(data)


if __name__ == '__main__':
    main()

"""
bias_vs_LAI.py
==============
Test: IS (edge) estimator bias from Gaussian radius truncation at r=0.

Reference : D_edg(CV=0.1, Gaussian) — nearly unbiased (0.2% mass at r<0).
Comparison: D_edg(CV=X, Gaussian) and D_edg(CV=X, LogNormal)
            for X in {0.2, 0.4, 0.6, 0.8, 1.0}.

RMSD(CV, dist, LAI) = sqrt( mean_runs[ (D_edg_X - D_edg_ref)^2 ] )

Expected:
  LogNormal: RMSD ≈ noise floor — no bias regardless of CV
  Gaussian:  RMSD grows with CV and with LAI (truncation bias accumulates)

x-axis: LAI_eff ∈ {0.5, 1.0, 1.5, 2.0, 2.5, 3.0}
y-axis: RMSD
Colors: CV value; solid = Gaussian, dashed = LogNormal

Scene: homogeneous, ALA='UNI', P=0.99, rho=tau=0.1, rsoil=1.0, SZA=60
       Built with CV=1.0 so voxel edge-threshold covers all CV values.

Run:
  python bias_vs_LAI.py              # full MC + plot
  python bias_vs_LAI.py --plot-only  # reload saved data and replot
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
P               = 0.99
RHO, TAU, RSOIL = 0.1, 0.1, 1.0
SZA             = 60
OBS_THETA       = 45.0
N_PHOTONS       = 10000
N_RUNS          = 20
LAI_LIST        = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
CV_REF          = 0.1
CV_LIST         = [0.2, 0.4, 0.6, 0.8, 1.0]
CV_BUILD        = max(CV_LIST)   # build scene with largest CV → safe voxel threshold

_ALA_TAG = 'SPH'
TAG      = f'P{P:g}_LAD{_ALA_TAG}_rho{RHO:g}_SZA{SZA}_N{N_PHOTONS}'
DATA_FILE = os.path.join(HERE, f'bias_vs_LAI_{TAG}_data.pkl')


# ─────────────────────────────────────────────────────────────────────────────
# Scene builder (built with CV_BUILD so edge-threshold covers all CV values)
# ─────────────────────────────────────────────────────────────────────────────
def _build(LAI_eff):
    from canopi.config   import SceneConfig
    from canopi.scene    import build_scene
    from canopi.geometry import polar_vect
    deg = np.pi / 180
    cfg = SceneConfig(
        LAI_eff=LAI_eff, P_LAI=P,
        leaf_radius_std_mul=CV_BUILD,
        ALA='UNI', rsoil=RSOIL, rho=RHO, tau=TAU, leaf_radius=0.005,
        max_depth=200, rr_depth=999, compute_derivative=True, seed=0,
    )
    scene    = build_scene(cfg, np.random.default_rng(0))
    obs_dirs = np.array([[np.sin(OBS_THETA * deg), 0., np.cos(OBS_THETA * deg)]])
    vec_sun  = polar_vect((180 - SZA) * deg, 180 * deg)
    return cfg, scene, obs_dirs, vec_sun


# ─────────────────────────────────────────────────────────────────────────────
# MC
# ─────────────────────────────────────────────────────────────────────────────
def run_mc():
    from canopi.gpu import compute_deriv_gpu, GPU_AVAILABLE
    if not GPU_AVAILABLE:
        print('No GPU detected.'); sys.exit(1)

    print(f'P={P}  rho=tau={RHO}  rsoil={RSOIL}  N={N_PHOTONS}  N_RUNS={N_RUNS}')
    print(f'CV_REF={CV_REF}  CV_LIST={CV_LIST}')
    print(f'LAI_LIST={LAI_LIST}')
    t0 = time.time()

    # results[LAI_eff][(CV, dist)] = array of N_RUNS D_edg values
    results = {}

    all_cvs = [CV_REF] + CV_LIST

    for LAI_eff in LAI_LIST:
        cfg, scene, obs_dirs, vec_sun = _build(LAI_eff)
        LAI_Max    = scene.LAI_Max
        LAI_r_base = float(scene.LAI_r)
        print(f'\nLAI_eff={LAI_eff}  LAI_Max={LAI_Max:.4f}  n_leaf={scene.n_leaf}')

        # Pre-allocate storage
        storage = {(cv, dist): [] for cv in all_cvs for dist in ['gaussian', 'lognormal']}

        for run_i in range(N_RUNS):
            seed = run_i * 9999
            for cv in all_cvs:
                scene.std_LAI = cv * LAI_r_base   # override CV for this call
                for leaf_dist in ['gaussian', 'lognormal']:
                    _, _, de, _ = compute_deriv_gpu(
                        scene, obs_dirs, vec_sun, N_PHOTONS,
                        max_depth=cfg.max_depth, seed=seed, leaf_dist=leaf_dist)
                    val = float(np.pi * de[:, 0].mean() / LAI_Max)
                    storage[(cv, leaf_dist)].append(val)

            if (run_i + 1) % 5 == 0:
                print(f'  run {run_i+1:2d}/{N_RUNS}', flush=True)

        results[LAI_eff] = {k: np.array(v) for k, v in storage.items()}

    print(f'\nTotal: {time.time() - t0:.0f}s')

    data = dict(
        P=P, RHO=RHO, TAU=TAU, RSOIL=RSOIL, SZA=SZA,
        N_PHOTONS=N_PHOTONS, N_RUNS=N_RUNS,
        LAI_LIST=LAI_LIST, CV_REF=CV_REF, CV_LIST=CV_LIST,
        results=results,
    )
    with open(DATA_FILE, 'wb') as f:
        pickle.dump(data, f)
    print(f'Data → {DATA_FILE}')
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────
def make_figures(data):
    LAI_LIST_ = data['LAI_LIST']
    CV_LIST_  = data['CV_LIST']
    CV_REF_   = data['CV_REF']
    results   = data['results']
    LAIs      = np.array(LAI_LIST_, dtype=float)

    matplotlib.rcParams.update({
        'text.usetex': False, 'mathtext.fontset': 'cm',
        'font.family': 'serif', 'font.size': 11,
        'axes.labelsize': 13, 'xtick.labelsize': 10,
        'ytick.labelsize': 10, 'legend.fontsize': 9,
    })

    cv_colors = {
        0.2: '#1f77b4',
        0.4: '#d62728',
        0.6: '#2ca02c',
        0.8: '#ff7f0e',
        1.0: '#9467bd',
    }

    def rmsd(arr_X, arr_ref):
        diff = np.asarray(arr_X) - np.asarray(arr_ref)
        return float(np.sqrt(np.mean(diff ** 2)))

    def draw(ax, normalize, fs_label=13, fs_legend=12, fs_tick=10):
        for cv in CV_LIST_:
            color = cv_colors[cv]
            rg, rl = [], []
            for LAI_eff in LAI_LIST_:
                ref  = results[LAI_eff][(CV_REF_, 'gaussian')]
                dref = abs(float(np.mean(ref)))
                norm = (dref if dref > 1e-12 else 1.0) if normalize else 1.0
                rg.append(rmsd(results[LAI_eff][(cv, 'gaussian')],  ref) / norm)
                rl.append(rmsd(results[LAI_eff][(cv, 'lognormal')], ref) / norm)
            # Log-Normal is the law adopted in the paper -> solid; Gaussian -> dashed
            ax.plot(LAIs, rl, color=color, ls='-',  lw=2,   marker='s', ms=6,
                    label=fr'Log-N  CV={cv}')
            ax.plot(LAIs, rg, color=color, ls='--', lw=1.5, marker='o', ms=5,
                    label=fr'Gauss  CV={cv}')

        ax.set_xlabel(r'$LAI$', fontsize=fs_label)
        if normalize:
            ax.set_ylabel(r'RMSD$(D_{\rm edg},\, D_{\rm ref})\,/\,|D_{\rm ref}|$',
                          fontsize=fs_label)
        else:
            ax.set_ylabel(r'RMSD$(D_{\rm edg},\, D_{\rm ref})$', fontsize=fs_label)
        ax.legend(fontsize=fs_legend, loc='upper right' if not normalize else 'upper left',
                  ncol=2, columnspacing=1.0, handlelength=2.2, labelspacing=0.3,
                  framealpha=0.9)
        ax.tick_params(labelsize=fs_tick)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(LAI_LIST_)
        # Headroom so the enlarged legend does not sit on top of the curves.
        y0, y1 = ax.get_ylim()
        ax.set_ylim(y0, y0 + (y1 - y0) * 1.45)

    suptitle = (
        fr'IS estimator bias: RMSD vs LAI  —  '
        fr'P={data["P"]}, $\rho$=$\tau$={data["RHO"]}, rsoil={data["RSOIL"]}, '
        fr'SZA={data["SZA"]}°,  $N$={data["N_PHOTONS"]},  {data["N_RUNS"]} runs  '
        fr'(ref = Gauss CV={data["CV_REF"]})')

    for normalize in [False, True]:
        suffix = '_rel' if normalize else '_abs'

        # PNG (with title)
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.suptitle(suptitle, fontsize=9)
        draw(ax, normalize)
        plt.tight_layout()
        out_png = os.path.join(HERE, f'bias_vs_LAI_{TAG}{suffix}.png')
        fig.savefig(out_png, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'PNG → {out_png}')

        # EPS (no title). The paper includes this panel at 7.5 cm, i.e. the canvas is
        # scaled down by ~0.6, so the fonts are set larger here to stay readable.
        fig2, ax2 = plt.subplots(figsize=(5.0, 3.5))
        draw(ax2, normalize, fs_label=15, fs_legend=11, fs_tick=12)
        plt.tight_layout()
        out_eps = os.path.join(HERE, f'bias_vs_LAI_{TAG}{suffix}.eps')
        fig2.savefig(out_eps, format='eps', bbox_inches='tight')
        plt.close(fig2)
        print(f'EPS → {out_eps}')

    print('All done ✓')


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot-only', action='store_true')
    args = parser.parse_args()

    if args.plot_only:
        if not os.path.exists(DATA_FILE):
            print(f'Data file not found: {DATA_FILE}'); sys.exit(1)
        with open(DATA_FILE, 'rb') as f:
            data = pickle.load(f)
        print(f'Loaded {DATA_FILE}')
    else:
        data = run_mc()

    make_figures(data)


if __name__ == '__main__':
    main()

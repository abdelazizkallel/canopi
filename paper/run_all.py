"""
run_all.py — drive the paper figure scripts.

    python paper/run_all.py --list                 show the stages and exit
    python paper/run_all.py --only tree_positions  run one stage
    python paper/run_all.py --fast                 run only the cheap stages
    python paper/run_all.py                        run everything (many hours, GPU)

Each stage is a separate subprocess, so one failure does not abort the rest;
a summary with per-stage timings is printed at the end.

Figures are written to $CANOPI_FIG_DIR (default ./figures).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable

# name, cheap?, argv, one-line description
STAGES: list[tuple[str, bool, list[str], str]] = [
    ("tree_positions", True,
     ["fig_tree_positions.py"],
     "JBS stand map coloured by species (CPU, seconds)"),

    ("bias_vs_lai", True,
     ["fig_bias_vs_lai.py"],
     "estimator bias vs LAI (GPU, minutes)"),

    ("convergence", True,
     ["fig_convergence_fic_edg.py"],
     "convergence of the fictive and edge estimators (GPU, minutes)"),

    ("multiang_toy", True,
     ["fig_multiang_toy.py"],
     "multi-angular toy LAI+rho retrieval -> multiang_* (GPU, minutes)"),

    ("species_images", False,
     ["fig_jbs_species_images.py", "--band", "O17", "--res", "512", "--n-spp", "32"],
     "512x512 BRF + per-species derivative images, nadir & hotspot (GPU, ~1 h)"),

    ("inversion_joint", False,
     ["fig_inversion_species.py", "--init-p-lai", "0.35", "--k-max", "0",
      "--out-prefix", "lai_rho_species_p035_k0"],
     "joint LAI+rho retrieval, 21 parameters (GPU, hours)"),

    ("inversion_truerho", False,
     ["fig_inversion_species.py", "--init-p-lai", "0.35", "--k-max", "0", "--true-rho",
      "--out-prefix", "lai_species_truerho_p035_k0"],
     "LAI-only retrieval with rho held at truth (GPU, hours)"),

    ("inversion_replot", False,
     ["fig_inversion_replot.py"],
     "publication EPS from the two inversion .pkl files (needs LaTeX)"),
]

QUICK_OVERRIDE = {
    "species_images": ["fig_jbs_species_images.py", "--band", "O17",
                       "--res", "128", "--n-spp", "8"],
    "inversion_joint": ["fig_inversion_species.py", "--init-p-lai", "0.35", "--k-max", "0",
                        "--img-size", "32", "--n-spp", "4", "--max-iter", "5",
                        "--out-prefix", "lai_rho_species_p035_k0"],
    "inversion_truerho": ["fig_inversion_species.py", "--init-p-lai", "0.35", "--k-max", "0",
                          "--true-rho", "--img-size", "32", "--n-spp", "4", "--max-iter", "5",
                          "--out-prefix", "lai_species_truerho_p035_k0"],
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="show stages and exit")
    ap.add_argument("--only", metavar="NAME", action="append",
                    help="run only this stage (repeatable)")
    ap.add_argument("--fast", action="store_true",
                    help="run only the stages that finish in minutes")
    ap.add_argument("--quick", action="store_true",
                    help="run the heavy stages at reduced resolution / iterations "
                         "(exercises the pipeline; does NOT reproduce the published figures)")
    args = ap.parse_args()

    if args.list:
        for name, cheap, _, desc in STAGES:
            print(f"  {name:20s} {'fast' if cheap else 'HEAVY':6s}  {desc}")
        return 0

    stages = STAGES
    if args.only:
        unknown = set(args.only) - {s[0] for s in STAGES}
        if unknown:
            print(f"unknown stage(s): {', '.join(sorted(unknown))}", file=sys.stderr)
            return 2
        stages = [s for s in stages if s[0] in set(args.only)]
    elif args.fast:
        stages = [s for s in stages if s[1]]

    results = []
    for name, _cheap, argv, desc in stages:
        if args.quick and name in QUICK_OVERRIDE:
            argv = QUICK_OVERRIDE[name]
        print(f"\n{'=' * 72}\n  {name} — {desc}\n{'=' * 72}", flush=True)
        t0 = time.time()
        rc = subprocess.call([PY, str(HERE / argv[0]), *argv[1:]])
        results.append((name, rc, time.time() - t0))
        print(f"  -> {name}: {'ok' if rc == 0 else f'FAILED (rc={rc})'} "
              f"in {results[-1][2]:.1f}s", flush=True)

    print(f"\n{'=' * 72}\n  summary\n{'=' * 72}")
    for name, rc, dt in results:
        print(f"  {name:20s} {'ok' if rc == 0 else f'FAILED rc={rc}':12s} {dt:8.1f}s")
    return 0 if all(rc == 0 for _, rc, _ in results) else 1


if __name__ == "__main__":
    sys.exit(main())

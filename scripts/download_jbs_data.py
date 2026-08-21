"""
download_jbs_treelocations.py
==============================
Download all 18 *_treelocations.dat files for HET09_JBS_SUM from the
RAMI-5 benchmark server and save them to a local directory.

Usage
-----
    python download_jbs_treelocations.py [output_dir]

Default output_dir: ./JBS_treelocations
"""

import sys
import urllib.request
from pathlib import Path

BASE_URL = (
    "https://rami-benchmark.jrc.ec.europa.eu"
    "/_www/RAMI5/images/HET09_JBS_SUM"
)

GROUPS = [
    "ACPL",
    "BEPE1", "BEPE2", "BEPE3", "BEPE4",
    "ALGL1", "ALGL2", "ALGL3", "ALGL4",
    "FREX",
    "TICO1", "TICO2", "TICO3", "TICO4", "TICO5",
    "PIAB",
    "POTR1", "POTR2",
]


def download_all(out_dir: str = "JBS_treelocations") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ok, failed = [], []
    for grp in GROUPS:
        fname = f"{grp}_treelocations.dat"
        url   = f"{BASE_URL}/{fname}"
        dest  = out / fname
        try:
            urllib.request.urlretrieve(url, dest)
            # Count non-empty lines
            lines = [l for l in dest.read_text().splitlines() if l.strip()]
            print(f"  {fname:30s}  {len(lines):3d} trees  → {dest}")
            ok.append(grp)
        except Exception as e:
            print(f"  {fname:30s}  FAILED: {e}")
            failed.append(grp)

    print(f"\n{len(ok)}/18 files downloaded to '{out}'")
    if failed:
        print(f"Failed: {failed}")


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "JBS_treelocations"
    download_all(out_dir)

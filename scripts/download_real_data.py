#!/usr/bin/env python3
"""Download public credit datasets for ProScore real-data tests.

Sources (pick one or more):
  - lending_club : Zenodo LC granting model CSV (~168 MB, 1.3M rows)
  - gmsc         : Give Me Some Credit training set (~7 MB, 150k rows, GitHub mirror)
  - home_credit  : Kaggle Home Credit Default Risk (requires ``kaggle`` CLI + API token)

Usage::

    python scripts/download_real_data.py gmsc
    python scripts/download_real_data.py lending_club
    python scripts/download_real_data.py home_credit   # needs ~/.kaggle/kaggle.json
    python scripts/download_real_data.py --all
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

URLS = {
    "gmsc": (
        DATA / "gmsc_train.csv",
        "https://raw.githubusercontent.com/DrIanGregory/Kaggle-GiveMeSomeCredit/"
        "master/data/GiveMeSomeCredit-training.csv",
    ),
    "lending_club": (
        DATA / "lending_club" / "LC_loans_granting_model_dataset.csv",
        "https://zenodo.org/api/records/11295916/files/"
        "LC_loans_granting_model_dataset.csv/content",
    ),
}


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1000:
        print(f"  skip (exists): {dest}")
        return
    print(f"  downloading → {dest}")
    urllib.request.urlretrieve(url, dest)  # noqa: S310 — trusted public dataset URLs
    print(f"  done ({dest.stat().st_size / 1e6:.1f} MB)")


def download_gmsc() -> Path:
    dest, url = URLS["gmsc"]
    _download(url, dest)
    return dest


def download_lending_club() -> Path:
    dest, url = URLS["lending_club"]
    _download(url, dest)
    return dest


def download_home_credit() -> Path:
    out_dir = DATA / "home_credit"
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "application_train.csv"
    if target.exists() and target.stat().st_size > 1_000_000:
        print(f"  skip (exists): {target}")
        return target
    if shutil.which("kaggle") is None:
        print(
            "  home_credit: 需要 Kaggle CLI。\n"
            "    pip install kaggle\n"
            "    配置 ~/.kaggle/kaggle.json 后执行:\n"
            f"    kaggle competitions download -c home-credit-default-risk -p {out_dir}\n"
            f"    unzip {out_dir}/*.zip -d {out_dir}",
            file=sys.stderr,
        )
        sys.exit(1)
    subprocess.run(
        [
            "kaggle", "competitions", "download",
            "-c", "home-credit-default-risk",
            "-f", "application_train.csv",
            "-p", str(out_dir),
        ],
        check=True,
    )
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "datasets",
        nargs="*",
        choices=["gmsc", "lending_club", "home_credit"],
        help="Dataset keys to download",
    )
    parser.add_argument("--all", action="store_true", help="Download all supported sources")
    args = parser.parse_args()
    keys = list(args.datasets)
    if args.all:
        keys = ["gmsc", "lending_club", "home_credit"]
    if not keys:
        parser.print_help()
        sys.exit(0)

    for key in keys:
        print(f"[{key}]")
        if key == "gmsc":
            download_gmsc()
        elif key == "lending_club":
            download_lending_club()
        else:
            download_home_credit()
    print("\nNext: python scripts/prepare_real_scorecard_data.py")


if __name__ == "__main__":
    main()

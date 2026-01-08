#!/usr/bin/env python3
"""
Compute hypervolume (HV) over multiple runs for algorithm2 layout:

- You pass a parent directory.
- Recursively finds run_X directories.
- For each run_X, loads:
    run_X/archive/pareto_history/generation_Y.csv
  where Y is the largest generation number present in that folder.
- CSV format:
    First row: "0,1" (objective ids) -> ignore
    Remaining rows: objective values (assumed MAXIMIZATION)
- Uses pygmo for:
    - nondominated filtering
    - nadir points
    - hypervolume
- If you provide --ref-max, uses that reference point (given in MAXIMIZATION space).
  Otherwise, builds a common reference point across runs (as before).
- Computes per-run HV, mean, and SEM.

Requires: pygmo, numpy
  pip install pygmo numpy
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pygmo as pg


RUN_DIR_RE = re.compile(r"^run_\d+$")
GEN_FILE_RE = re.compile(r"^generation_(\d+)\.csv$")


@dataclass
class RunResult:
    run_dir: Path
    gen_csv: Path
    generation: int
    points_max: np.ndarray          # (N, M) in maximization space
    nd_points_min: np.ndarray       # (K, M) in minimization space
    nadir_min: np.ndarray           # (M,)
    hv: Optional[float] = None


def sem(x: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    return float(np.std(x, ddof=1) / math.sqrt(x.size))


def parse_ref_max(ref_str: str) -> np.ndarray:
    """
    Parses a reference point passed as:
      --ref-max 0,1
      --ref-max "0, 1"
      --ref-max "0 1"
    Returns 1D float array.
    """
    parts = [p for p in re.split(r"[,\s]+", ref_str.strip()) if p != ""]
    if not parts:
        raise ValueError("Empty --ref-max.")
    return np.asarray([float(p) for p in parts], dtype=float)


def find_run_dirs(parent: Path) -> List[Path]:
    return sorted([p for p in parent.rglob("*") if p.is_dir() and RUN_DIR_RE.match(p.name)])


def find_latest_generation_csv(run_dir: Path) -> Tuple[int, Path]:
    hist_dir = run_dir / "archive" / "pareto_history"
    if not hist_dir.is_dir():
        raise FileNotFoundError(f"Missing directory: {hist_dir}")

    best_y: Optional[int] = None
    best_path: Optional[Path] = None

    for p in hist_dir.iterdir():
        if not p.is_file():
            continue
        m = GEN_FILE_RE.match(p.name)
        if not m:
            continue
        y = int(m.group(1))
        if best_y is None or y > best_y:
            best_y = y
            best_path = p

    if best_y is None or best_path is None:
        raise FileNotFoundError(f"No generation_*.csv files found in: {hist_dir}")

    return best_y, best_path


def read_generation_csv(gen_csv: Path) -> np.ndarray:
    """
    Reads generation_Y.csv:
      - First row is objective ids (e.g. "0,1") -> ignore
      - Remaining rows are floats (objective values)
    Returns (N, M) float array.
    """
    rows: List[List[float]] = []
    with gen_csv.open("r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)  # objective ids
        if header is None:
            raise ValueError(f"Empty file: {gen_csv}")

        for line in reader:
            if not line:
                continue
            vals = [float(x) for x in line if str(x).strip() != ""]
            if vals:
                rows.append(vals)

    if not rows:
        raise ValueError(f"No data rows found in: {gen_csv}")

    arr = np.asarray(rows, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 1:
        raise ValueError(f"Bad shape from {gen_csv}: {arr.shape}")
    return arr


def nondominated_front_min(points_min: np.ndarray) -> np.ndarray:
    ndf, _, _, _ = pg.fast_non_dominated_sorting(points_min.tolist())
    if not ndf or len(ndf[0]) == 0:
        raise ValueError("fast_non_dominated_sorting returned empty first front.")
    return points_min[np.asarray(ndf[0], dtype=int)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", type=str, required=True, help="Parent directory for algorithm2")
    ap.add_argument(
        "--ref-max",
        type=str,
        default=None,
        help="Optional reference point in MAXIMIZATION space, e.g. --ref-max '0,5000'",
    )
    ap.add_argument(
        "--ref-margin-scale",
        type=float,
        default=1e-6,
        help="(Auto-ref only) margin scale: ref = ref + scale*(abs(ref)+1).",
    )
    ap.add_argument(
        "--write-results-csv",
        action="store_true",
        help="If set, writes hv_results.csv under --parent",
    )
    args = ap.parse_args()

    parent = Path(args.parent).expanduser().resolve()
    if not parent.exists():
        print(f"ERROR: parent does not exist: {parent}", file=sys.stderr)
        return 2

    run_dirs = find_run_dirs(parent)
    if not run_dirs:
        print(f"ERROR: No run_X directories found under: {parent}", file=sys.stderr)
        return 2

    results: List[RunResult] = []
    for run_dir in run_dirs:
        try:
            gen, gen_csv = find_latest_generation_csv(run_dir)
            pts_max = read_generation_csv(gen_csv)
            pts_min = -pts_max  # maximize -> minimize for pygmo
            nd_min = nondominated_front_min(pts_min)
            nad = pg.nadir(nd_min.tolist())
            results.append(
                RunResult(
                    run_dir=run_dir,
                    gen_csv=gen_csv,
                    generation=gen,
                    points_max=pts_max,
                    nd_points_min=nd_min,
                    nadir_min=np.asarray(nad, dtype=float),
                )
            )
        except Exception as e:
            print(f"WARNING: Skipping {run_dir} due to error: {e}", file=sys.stderr)

    if not results:
        print("ERROR: All runs failed to parse.", file=sys.stderr)
        return 2

    # Validate consistent dimensionality across runs
    m = results[0].points_max.shape[1]
    if any(r.points_max.shape[1] != m for r in results):
        print("ERROR: Not all runs have the same number of objectives.", file=sys.stderr)
        return 2

    # Choose reference point (in minimization space)
    if args.ref_max is not None:
        ref_max = parse_ref_max(args.ref_max)
        if ref_max.shape[0] != m:
            print(
                f"ERROR: --ref-max has dimension {ref_max.shape[0]} but runs have {m} objectives.",
                file=sys.stderr,
            )
            return 2
        ref_min = -ref_max
        ref_mode = "USER"
    else:
        # Common reference point (min space): componentwise max over per-run nadirs
        all_nadirs = np.vstack([r.nadir_min for r in results])
        common_ref_min = np.max(all_nadirs, axis=0)

        # Make it strictly worse with a tiny margin
        margin = args.ref_margin_scale * (np.abs(common_ref_min) + 1.0)
        ref_min = common_ref_min + margin
        ref_mode = "AUTO"

    # Compute HV per run
    hvs: List[float] = []
    for r in results:
        r.hv = float(pg.hypervolume(r.nd_points_min.tolist()).compute(ref_min.tolist()))
        hvs.append(r.hv)

    hvs_arr = np.asarray(hvs, dtype=float)

    # Print summary
    print(f"Found {len(results)} valid runs under: {parent}")
    print(f"Objectives: {m}")
    print(f"Reference mode: {ref_mode}")
    print()
    print("Reference point (MAXIMIZATION space):", (-ref_min))
    print("Reference point (MINIMIZATION space):", (ref_min))
    print()
    print("Per-run hypervolumes:")
    for r in results:
        print(f"  {r.run_dir.name:>10s} | gen={r.generation:<6d} | HV={r.hv:.6g} | {r.gen_csv}")

    mean_hv = float(np.mean(hvs_arr))
    sem_hv = sem(hvs_arr)
    print()
    print(f"HV mean = {mean_hv:.6g}")
    print(f"HV SEM  = {sem_hv:.6g}")

    if args.write_results_csv:
        out_path = parent / "hv_results.csv"
        with out_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["run_dir", "generation", "generation_csv", "hypervolume"])
            for r in results:
                w.writerow([str(r.run_dir), r.generation, str(r.gen_csv), r.hv])
            w.writerow([])
            w.writerow(["REFERENCE_MODE", ref_mode, "", ""])
            w.writerow(["REFERENCE_POINT_MAX", *(-ref_min).tolist()])
            w.writerow(["REFERENCE_POINT_MIN", *(ref_min).tolist()])
            w.writerow([])
            w.writerow(["MEAN", "", "", mean_hv])
            w.writerow(["SEM", "", "", sem_hv])
        print(f"\nWrote: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

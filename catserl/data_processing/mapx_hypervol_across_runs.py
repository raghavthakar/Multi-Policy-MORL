#!/usr/bin/env python3
"""
Compute hypervolume (HV) over multiple runs.

What it does:
- You pass a parent directory (algorithm1 root).
- Recursively finds run_X directories containing pareto_stats.csv.
- For each CSV, reads the *latest* generation row and parses vector_returns into points.
- Assumes objectives are MAXIMIZED, so it negates them to use pygmo (which assumes minimization).
- If you provide --ref-max, uses that reference point (given in MAXIMIZATION space).
  Otherwise, uses a common reference point derived from per-run nadirs (as before).
- Computes HV per run with that reference, plus mean and SEM.

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


@dataclass
class RunResult:
    run_dir: Path
    csv_path: Path
    generation: int
    points_max: np.ndarray          # shape (N, M) in maximization space
    nd_points_min: np.ndarray       # shape (K, M) in minimization space (negated + nondominated)
    nadir_min: np.ndarray           # shape (M,)
    hv: Optional[float] = None      # filled after ref point is known


def parse_vector_returns(s: str) -> np.ndarray:
    """
    Parses strings like:
      [[ 132.2361145  5470.51416016]
       [ 202.15625    5349.72558594]]
    into an (N, M) float numpy array.
    """
    s = s.strip()
    rows = re.findall(r"\[([^\[\]]+)\]", s)
    if not rows:
        raise ValueError(f"Could not parse vector_returns: {s[:120]}...")

    parsed: List[List[float]] = []
    for r in rows:
        r = r.replace(",", " ")
        nums = [float(x) for x in r.split() if x.strip()]
        if not nums:
            continue
        parsed.append(nums)

    if not parsed:
        raise ValueError("Parsed zero rows from vector_returns.")
    dim = len(parsed[0])
    if any(len(row) != dim for row in parsed):
        raise ValueError(f"Inconsistent objective dimension in vector_returns (expected {dim}).")

    return np.asarray(parsed, dtype=float)


def read_latest_points(csv_path: Path) -> Tuple[int, np.ndarray]:
    """
    Reads pareto_stats.csv and returns (latest_generation, points_array).
    Chooses the row with max(generation), not necessarily the last line.
    """
    best_gen: Optional[int] = None
    best_points: Optional[np.ndarray] = None

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if "generation" not in reader.fieldnames or "vector_returns" not in reader.fieldnames:
            raise ValueError(
                f"{csv_path} must have columns 'generation' and 'vector_returns'. "
                f"Found: {reader.fieldnames}"
            )

        for row in reader:
            if row.get("generation") is None or row.get("vector_returns") is None:
                continue
            gen = int(row["generation"])
            if best_gen is None or gen > best_gen:
                pts = parse_vector_returns(row["vector_returns"])
                best_gen = gen
                best_points = pts

    if best_gen is None or best_points is None:
        raise ValueError(f"No valid rows found in {csv_path}")

    return best_gen, best_points


def nondominated_front_min(points_min: np.ndarray) -> np.ndarray:
    """
    Returns the nondominated front for a minimization problem using pygmo.
    """
    ndf, _, _, _ = pg.fast_non_dominated_sorting(points_min.tolist())
    if not ndf or len(ndf[0]) == 0:
        raise ValueError("fast_non_dominated_sorting returned an empty first front.")
    return points_min[np.asarray(ndf[0], dtype=int)]


def find_runs(parent: Path, csv_name: str) -> List[Path]:
    """
    Recursively find pareto_stats.csv files under run_X directories.
    """
    hits: List[Path] = []
    for csv_path in parent.rglob(csv_name):
        if not csv_path.is_file():
            continue
        if any(RUN_DIR_RE.match(p.name) for p in csv_path.parents):
            hits.append(csv_path)
    return sorted(hits)


def sem(x: np.ndarray) -> float:
    """
    Standard error of the mean, using sample std (ddof=1).
    Returns NaN if fewer than 2 samples.
    """
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", type=str, required=True, help="Parent directory for algorithm1")
    ap.add_argument("--csv-name", type=str, default="pareto_stats.csv", help="CSV filename to look for")
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

    csv_paths = find_runs(parent, args.csv_name)
    if not csv_paths:
        print(f"ERROR: No '{args.csv_name}' found under run_X directories in {parent}", file=sys.stderr)
        return 2

    results: List[RunResult] = []
    for csv_path in csv_paths:
        try:
            gen, pts_max = read_latest_points(csv_path)
            pts_min = -pts_max
            nd_min = nondominated_front_min(pts_min)
            nad = pg.nadir(nd_min.tolist())
            rr = RunResult(
                run_dir=next(p for p in csv_path.parents if RUN_DIR_RE.match(p.name)),
                csv_path=csv_path,
                generation=gen,
                points_max=pts_max,
                nd_points_min=nd_min,
                nadir_min=np.asarray(nad, dtype=float),
            )
            results.append(rr)
        except Exception as e:
            print(f"WARNING: Skipping {csv_path} due to error: {e}", file=sys.stderr)

    if not results:
        print("ERROR: All runs failed to parse.", file=sys.stderr)
        return 2

    m = results[0].points_max.shape[1]
    if any(r.points_max.shape[1] != m for r in results):
        print("ERROR: Not all runs have the same number of objectives.", file=sys.stderr)
        return 2

    # Choose reference point (in minimization space).
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
        margin = args.ref_margin_scale * (np.abs(common_ref_min) + 1.0)
        ref_min = common_ref_min + margin
        ref_mode = "AUTO"

    # Compute hypervolume per run
    hvs: List[float] = []
    for r in results:
        hv = pg.hypervolume(r.nd_points_min.tolist()).compute(ref_min.tolist())
        r.hv = float(hv)
        hvs.append(r.hv)

    hvs_arr = np.asarray(hvs, dtype=float)

    # Reporting
    print(f"Found {len(results)} valid runs under: {parent}")
    print(f"Objectives: {m}")
    print(f"Reference mode: {ref_mode}")
    print()
    print("Reference point (MAXIMIZATION space):", (-ref_min))
    print("Reference point (MINIMIZATION space):", (ref_min))
    print()
    print("Per-run hypervolumes:")
    for r in results:
        print(f"  {r.run_dir.name:>10s} | gen={r.generation:<4d} | HV={r.hv:.6g} | {r.csv_path}")

    mean_hv = float(np.mean(hvs_arr))
    sem_hv = sem(hvs_arr)
    print()
    print(f"HV mean = {mean_hv:.6g}")
    print(f"HV SEM  = {sem_hv:.6g}")

    if args.write_results_csv:
        out_path = parent / "hv_results.csv"
        with out_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["run_dir", "csv_path", "latest_generation", "hypervolume"])
            for r in results:
                w.writerow([str(r.run_dir), str(r.csv_path), r.generation, r.hv])
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

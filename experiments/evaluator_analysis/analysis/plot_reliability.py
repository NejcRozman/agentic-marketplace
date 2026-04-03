#!/usr/bin/env python3
"""
Create standard reliability visualizations from reliability analysis outputs.

Outputs:
1. ICC plot with 95% bootstrap confidence intervals.
2. Forest plot of per-sample mean rating with 95% confidence intervals.
3. ICC summary JSON (and optional CSV) for reporting.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot reliability analysis outputs (ICC + forest plot)")
    parser.add_argument(
        "--raw-runs",
        type=Path,
        default=Path("experiments/evaluator_analysis/data/reliability/reliability_v1/raw_runs.jsonl"),
        help="Path to raw_runs.jsonl",
    )
    parser.add_argument(
        "--sample-summary-csv",
        type=Path,
        default=Path("experiments/evaluator_analysis/data/reliability/reliability_v1/sample_summary.csv"),
        help="Path to sample_summary.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("experiments/evaluator_analysis/data/reliability/reliability_v1"),
        help="Output directory for generated artifacts",
    )
    parser.add_argument(
        "--min-successful-runs-for-icc",
        type=int,
        default=3,
        help="Minimum successful repeats required per sample to include in ICC",
    )
    parser.add_argument(
        "--icc-repeats",
        type=int,
        default=None,
        help="Fixed repeats per sample for ICC matrix (default: min repeats among eligible samples)",
    )
    parser.add_argument(
        "--min-successful-runs-for-forest",
        type=int,
        default=1,
        help="Minimum successful repeats required per sample to include in forest plot",
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=2000,
        help="Bootstrap iterations for ICC confidence intervals",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for bootstrap",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Output figure DPI",
    )
    parser.add_argument(
        "--icc-xmin",
        type=float,
        default=0.5,
        help="ICC plot x-axis minimum",
    )
    parser.add_argument(
        "--icc-xmax",
        type=float,
        default=1.0,
        help="ICC plot x-axis maximum",
    )
    return parser.parse_args()


def _to_float(value: Any) -> float:
    if value in ("", None):
        return 0.0
    return float(value)


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def load_sample_summary_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            std = _to_float(row.get("rating_std", "0"))
            mean = _to_float(row.get("rating_mean", "0"))
            runs_total = int(row.get("runs_total", "0"))
            runs_successful = int(row.get("runs_successful", "0"))

            rows.append(
                {
                    "sample_id": row.get("sample_id", ""),
                    "runs_total": runs_total,
                    "runs_successful": runs_successful,
                    "mean": mean,
                    "std": std,
                }
            )
    return rows


def load_raw_runs(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            rows.append(row)
    return rows


def _icc_1_1(matrix: np.ndarray) -> Dict[str, float]:
    """One-way random-effects ICC(1,1) and ICC(1,k) on a balanced matrix."""
    n, k = matrix.shape
    if n < 2 or k < 2:
        return {"icc_1_1": float("nan"), "icc_1_k": float("nan"), "ms_between": float("nan"), "ms_within": float("nan")}

    row_means = matrix.mean(axis=1)
    grand_mean = matrix.mean()

    ss_between = k * float(((row_means - grand_mean) ** 2).sum())
    ss_within = float(((matrix - row_means[:, None]) ** 2).sum())

    ms_between = ss_between / (n - 1)
    ms_within = ss_within / (n * (k - 1))

    denom = ms_between + (k - 1) * ms_within
    if denom == 0 or ms_between == 0:
        icc_11 = float("nan")
        icc_1k = float("nan")
    else:
        icc_11 = (ms_between - ms_within) / denom
        icc_1k = (ms_between - ms_within) / ms_between

    return {
        "icc_1_1": float(icc_11),
        "icc_1_k": float(icc_1k),
        "ms_between": float(ms_between),
        "ms_within": float(ms_within),
    }


def _bootstrap_icc_ci(
    matrix: np.ndarray,
    iterations: int,
    seed: int,
) -> Tuple[Optional[float], Optional[float], int]:
    """Bootstrap 95% CI for ICC(1,1) by resampling samples (rows)."""
    n = matrix.shape[0]
    rng = np.random.default_rng(seed)
    vals: List[float] = []

    for _ in range(iterations):
        idx = rng.integers(0, n, size=n)
        m = matrix[idx, :]
        icc = _icc_1_1(m)["icc_1_1"]
        if _is_number(icc):
            vals.append(float(icc))

    if not vals:
        return None, None, 0

    lower, upper = np.percentile(np.asarray(vals), [2.5, 97.5])
    return float(lower), float(upper), len(vals)


def _build_metric_groups(raw_runs: List[Dict[str, Any]]) -> Dict[str, Dict[str, List[Tuple[int, float]]]]:
    """
    Return metric->sample_id->[(repeat_idx, value), ...] for successful runs only.
    Metric keys: 'rating' plus any dimension keys found in quality_scores.
    """
    groups: Dict[str, Dict[str, List[Tuple[int, float]]]] = {}

    for row in raw_runs:
        if row.get("error") is not None:
            continue

        sample_id = str(row.get("sample_id", ""))
        repeat_idx = int(row.get("repeat_idx", 0))
        if not sample_id:
            continue

        rating = row.get("rating")
        if _is_number(rating):
            groups.setdefault("rating", {}).setdefault(sample_id, []).append((repeat_idx, float(rating)))

        scores = row.get("quality_scores", {}) or {}
        if isinstance(scores, dict):
            for dim, value in scores.items():
                if _is_number(value):
                    key = f"dim:{dim}"
                    groups.setdefault(key, {}).setdefault(sample_id, []).append((repeat_idx, float(value)))

    for per_sample in groups.values():
        for sample_id, vals in per_sample.items():
            per_sample[sample_id] = sorted(vals, key=lambda x: x[0])

    return groups


def _build_balanced_matrix(
    sample_series: Dict[str, List[Tuple[int, float]]],
    min_successful_runs: int,
    icc_repeats: Optional[int],
) -> Tuple[Optional[np.ndarray], List[str], int]:
    eligible = {sid: vals for sid, vals in sample_series.items() if len(vals) >= min_successful_runs}
    if not eligible:
        return None, [], 0

    if icc_repeats is not None:
        k = icc_repeats
    else:
        k = min(len(vals) for vals in eligible.values())

    if k < 2:
        return None, [], k

    sample_ids = sorted([sid for sid, vals in eligible.items() if len(vals) >= k])
    if len(sample_ids) < 2:
        return None, sample_ids, k

    matrix_rows = []
    for sid in sample_ids:
        vals = [v for _, v in eligible[sid][:k]]
        matrix_rows.append(vals)

    matrix = np.asarray(matrix_rows, dtype=float)
    return matrix, sample_ids, k


def compute_icc_results(
    raw_runs: List[Dict[str, Any]],
    min_successful_runs_for_icc: int,
    icc_repeats: Optional[int],
    bootstrap_iterations: int,
    seed: int,
) -> List[Dict[str, Any]]:
    groups = _build_metric_groups(raw_runs)
    results: List[Dict[str, Any]] = []

    for metric, sample_series in sorted(groups.items(), key=lambda kv: (kv[0] != "rating", kv[0])):
        matrix, sample_ids, k = _build_balanced_matrix(
            sample_series=sample_series,
            min_successful_runs=min_successful_runs_for_icc,
            icc_repeats=icc_repeats,
        )

        if matrix is None:
            results.append(
                {
                    "metric": metric,
                    "n_samples": len(sample_ids),
                    "k_repeats": k,
                    "icc_1_1": None,
                    "icc_1_k": None,
                    "ci95_lower": None,
                    "ci95_upper": None,
                    "bootstrap_successes": 0,
                    "status": "insufficient_data",
                }
            )
            continue

        stats = _icc_1_1(matrix)
        ci_lower, ci_upper, n_boot = _bootstrap_icc_ci(
            matrix=matrix,
            iterations=bootstrap_iterations,
            seed=seed,
        )

        results.append(
            {
                "metric": metric,
                "n_samples": int(matrix.shape[0]),
                "k_repeats": int(matrix.shape[1]),
                "icc_1_1": stats["icc_1_1"],
                "icc_1_k": stats["icc_1_k"],
                "ci95_lower": ci_lower,
                "ci95_upper": ci_upper,
                "bootstrap_successes": n_boot,
                "status": "ok",
            }
        )

    return results


def plot_icc(
    results: List[Dict[str, Any]],
    output_path: Path,
    dpi: int,
    x_min: float,
    x_max: float,
) -> None:
    import matplotlib.pyplot as plt

    ok = [r for r in results if r.get("status") == "ok" and _is_number(r.get("icc_1_1"))]
    if not ok:
        raise RuntimeError("No valid ICC results to plot")

    labels = []
    x = []
    xerr_low = []
    xerr_high = []
    meta = []

    for r in ok:
        m = r["metric"]
        if m == "rating":
            label = "overall_rating"
        elif m.startswith("dim:"):
            label = m.split(":", 1)[1]
        else:
            label = m

        icc = float(r["icc_1_1"])
        lo = r.get("ci95_lower")
        hi = r.get("ci95_upper")

        labels.append(label)
        x.append(icc)
        if _is_number(lo) and _is_number(hi):
            xerr_low.append(max(0.0, icc - float(lo)))
            xerr_high.append(max(0.0, float(hi) - icc))
        else:
            xerr_low.append(0.0)
            xerr_high.append(0.0)

        meta.append(f"n={r['n_samples']}, k={r['k_repeats']}")

    y = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10, 0.8 * len(labels) + 2.0))
    ax.errorbar(
        x,
        y,
        xerr=[xerr_low, xerr_high],
        fmt="o",
        color="#1f77b4",
        ecolor="#1f77b4",
        capsize=4,
        markersize=6,
    )

    ax.axvline(0.5, color="#9ca3af", linestyle="--", linewidth=1)
    ax.axvline(0.75, color="#6b7280", linestyle="--", linewidth=1)
    ax.axvline(0.9, color="#374151", linestyle=":", linewidth=1)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("ICC(1,1)")
    ax.set_title("Reliability ICC with 95% Bootstrap CI")
    ax.set_xlim(x_min, x_max)
    ax.grid(axis="x", alpha=0.25)

    for yi, txt in zip(y, meta):
        ax.text(1.01, yi, txt, transform=ax.get_yaxis_transform(), va="center", fontsize=9)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)


def plot_forest(
    rows: List[Dict[str, Any]],
    min_successful_runs: int,
    output_path: Path,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    usable = [r for r in rows if int(r["runs_successful"]) >= min_successful_runs and _is_number(r["mean"]) ]
    if not usable:
        raise RuntimeError("No sample rows available for forest plot with current filters")

    usable = sorted(usable, key=lambda r: float(r["mean"]))

    labels = [r["sample_id"] for r in usable]
    means = np.asarray([float(r["mean"]) for r in usable], dtype=float)
    stds = np.asarray([float(r["std"]) for r in usable], dtype=float)
    ns = np.asarray([int(r["runs_successful"]) for r in usable], dtype=float)

    # Approximate 95% CI by normal approximation.
    with np.errstate(divide="ignore", invalid="ignore"):
        se = np.where(ns > 1, stds / np.sqrt(ns), np.nan)
    ci = 1.96 * se

    y = np.arange(len(usable))

    fig, ax = plt.subplots(figsize=(10, 0.6 * len(usable) + 2.5))
    ax.errorbar(
        means,
        y,
        xerr=ci,
        fmt="o",
        color="#0f766e",
        ecolor="#0f766e",
        capsize=4,
        markersize=5,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Mean Rating (95% CI)")
    ax.set_title("Forest Plot of Sample-Level Mean Ratings")
    ax.grid(axis="x", alpha=0.25)

    for yi, n in zip(y, ns.astype(int)):
        ax.text(1.01, yi, f"n={n}", transform=ax.get_yaxis_transform(), va="center", fontsize=9)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)


def write_icc_summary(results: List[Dict[str, Any]], output_json: Path, output_csv: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    fieldnames = [
        "metric",
        "status",
        "n_samples",
        "k_repeats",
        "icc_1_1",
        "ci95_lower",
        "ci95_upper",
        "icc_1_k",
        "bootstrap_successes",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k) for k in fieldnames})


def main() -> int:
    args = parse_args()

    if not args.raw_runs.exists():
        raise FileNotFoundError(f"raw_runs.jsonl not found: {args.raw_runs}")
    if not args.sample_summary_csv.exists():
        raise FileNotFoundError(f"sample_summary.csv not found: {args.sample_summary_csv}")

    if args.min_successful_runs_for_icc < 2:
        raise ValueError("--min-successful-runs-for-icc must be >= 2")
    if args.min_successful_runs_for_forest < 1:
        raise ValueError("--min-successful-runs-for-forest must be >= 1")
    if args.icc_repeats is not None and args.icc_repeats < 2:
        raise ValueError("--icc-repeats must be >= 2 when provided")
    if args.bootstrap_iterations < 100:
        raise ValueError("--bootstrap-iterations should be >= 100")
    if args.icc_xmin >= args.icc_xmax:
        raise ValueError("--icc-xmin must be smaller than --icc-xmax")
    if args.icc_xmax > 1.0:
        raise ValueError("--icc-xmax should not exceed 1.0 for ICC")

    raw_runs = load_raw_runs(args.raw_runs)
    summary_rows = load_sample_summary_rows(args.sample_summary_csv)
    if not raw_runs:
        raise RuntimeError("No rows found in raw runs file")
    if not summary_rows:
        raise RuntimeError("No rows found in sample summary file")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    icc_results = compute_icc_results(
        raw_runs=raw_runs,
        min_successful_runs_for_icc=args.min_successful_runs_for_icc,
        icc_repeats=args.icc_repeats,
        bootstrap_iterations=args.bootstrap_iterations,
        seed=args.seed,
    )

    icc_plot_path = args.out_dir / "icc_plot.png"
    forest_plot_path = args.out_dir / "forest_plot.png"
    icc_json_path = args.out_dir / "icc_summary.json"
    icc_csv_path = args.out_dir / "icc_summary.csv"

    write_icc_summary(icc_results, icc_json_path, icc_csv_path)
    plot_icc(
        icc_results,
        icc_plot_path,
        dpi=args.dpi,
        x_min=args.icc_xmin,
        x_max=args.icc_xmax,
    )
    plot_forest(
        rows=summary_rows,
        min_successful_runs=args.min_successful_runs_for_forest,
        output_path=forest_plot_path,
        dpi=args.dpi,
    )

    print(f"Saved ICC plot: {icc_plot_path}")
    print(f"Saved forest plot: {forest_plot_path}")
    print(f"Saved ICC summary JSON: {icc_json_path}")
    print(f"Saved ICC summary CSV: {icc_csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

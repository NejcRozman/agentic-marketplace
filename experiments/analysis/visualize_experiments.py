"""Generate research-oriented charts from aggregated experiment CSV outputs.

Usage:
  python experiments/analysis/visualize_experiments.py
  python experiments/analysis/visualize_experiments.py --input-dir experiments/analysis/output/h1_set --fig-dir experiments/analysis/figures/h1_set
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt


def read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def to_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def save_fig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def label_of(row: Dict[str, Any]) -> str:
    scenario = row.get("scenario", "unknown")
    return f"{row.get('architecture', 'unknown')}-{row.get('reasoning_mode', 'unknown')}-{scenario}"


def plot_market_dynamics(auction_rows: List[Dict[str, Any]], fig_dir: Path) -> None:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in auction_rows:
        grouped[label_of(row)].append(row)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ax_price, ax_ratio = axes

    for label in sorted(grouped.keys()):
        series = sorted(grouped[label], key=lambda r: (r.get("experiment_id", ""), to_int(r.get("auction_id"))))
        x = list(range(1, len(series) + 1))
        y_price = [to_float(r.get("winning_bid_usdc"), to_float(r.get("winning_bid_amount")) / 1e6) for r in series]
        y_ratio = [to_float(r.get("winning_bid_ratio"), 0.0) for r in series]

        ax_price.plot(x, y_price, marker="o", linewidth=1.3, label=label)
        ax_ratio.plot(x, y_ratio, marker="o", linewidth=1.3, label=label)

    ax_price.set_title("Winning Price Over Auctions")
    ax_price.set_xlabel("Auction Index")
    ax_price.set_ylabel("Winning Bid (USDC)")

    ax_ratio.set_title("Price Ratio Over Auctions")
    ax_ratio.set_xlabel("Auction Index")
    ax_ratio.set_ylabel("Winning Bid / Budget")
    ax_ratio.set_ylim(bottom=0)

    handles, labels = ax_price.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8)

    save_fig(fig, fig_dir / "fig_market_dynamics.png")


def plot_profit_fairness(provider_run_rows: List[Dict[str, Any]], run_economics_rows: List[Dict[str, Any]], fig_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ax_dist, ax_frontier = axes

    by_label_balances: Dict[str, List[float]] = defaultdict(list)
    for row in provider_run_rows:
        by_label_balances[label_of(row)].append(to_float(row.get("net_balance"), 0.0))

    labels = sorted(by_label_balances.keys())
    if labels:
        data = [by_label_balances[l] for l in labels]
        ax_dist.boxplot(data, tick_labels=labels, showmeans=True)
        ax_dist.tick_params(axis="x", rotation=20)
    ax_dist.set_title("Long-run Profit Distribution")
    ax_dist.set_ylabel("Net Balance (USD)")

    by_label_runs: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in run_economics_rows:
        by_label_runs[label_of(row)].append(row)

    for label in sorted(by_label_runs.keys()):
        rows = by_label_runs[label]
        xs = [to_float(r.get("profit_per_completed_auction"), 0.0) for r in rows]
        ys = [to_float(r.get("profit_fairness_index"), 0.0) for r in rows]
        ax_frontier.scatter(xs, ys, alpha=0.8, label=label)

        if xs and ys:
            ax_frontier.scatter([fmean(xs)], [fmean(ys)], marker="X", s=120, edgecolors="black")

    ax_frontier.set_title("Efficiency vs Fairness")
    ax_frontier.set_xlabel("Profit Per Completed Auction (USD)")
    ax_frontier.set_ylabel("Profit Fairness Index (1 - Gini)")
    ax_frontier.set_ylim(0, 1.05)
    if by_label_runs:
        ax_frontier.legend(fontsize=8)

    save_fig(fig, fig_dir / "fig_profit_fairness.png")


def plot_cost_structure(provider_summary_rows: List[Dict[str, Any]], fig_dir: Path) -> None:
    run_costs: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(lambda: {"llm": 0.0, "gas": 0.0, "other": 0.0})

    for row in provider_summary_rows:
        label = label_of(row)
        run_id = row.get("experiment_id", "")
        llm = to_float(row.get("llm_costs"), 0.0)
        gas = to_float(row.get("gas_costs"), 0.0)
        total = to_float(row.get("total_costs"), 0.0)
        other = max(0.0, total - llm - gas)
        run_costs[(label, run_id)]["llm"] += llm
        run_costs[(label, run_id)]["gas"] += gas
        run_costs[(label, run_id)]["other"] += other

    by_label: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: {"llm": [], "gas": [], "other": []})
    for (label, _run_id), costs in run_costs.items():
        by_label[label]["llm"].append(costs["llm"])
        by_label[label]["gas"].append(costs["gas"])
        by_label[label]["other"].append(costs["other"])

    labels = sorted(by_label.keys())
    x = list(range(len(labels)))
    llm_vals = [fmean(by_label[l]["llm"]) for l in labels]
    gas_vals = [fmean(by_label[l]["gas"]) for l in labels]
    other_vals = [fmean(by_label[l]["other"]) for l in labels]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, llm_vals, label="LLM cost")
    ax.bar(x, gas_vals, bottom=llm_vals, label="Gas cost")
    stacked_bottom = [llm_vals[i] + gas_vals[i] for i in range(len(labels))]
    ax.bar(x, other_vals, bottom=stacked_bottom, label="Other cost")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_title("System Cost Composition (Mean Per Run)")
    ax.set_ylabel("Cost (USD)")
    ax.legend()

    save_fig(fig, fig_dir / "fig_cost_structure.png")


def plot_reputation_vs_bid(bid_market_rows: List[Dict[str, Any]], fig_dir: Path) -> None:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in bid_market_rows:
        grouped[label_of(row)].append(row)

    if not grouped:
        return

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for label in sorted(grouped.keys()):
        rows = grouped[label]
        x_win = [to_float(r.get("normalized_bid"), 0.0) for r in rows if to_int(r.get("won_auction"), 0) == 1]
        y_win = [to_float(r.get("reputation_before_auction"), 50.0) for r in rows if to_int(r.get("won_auction"), 0) == 1]
        x_lose = [to_float(r.get("normalized_bid"), 0.0) for r in rows if to_int(r.get("won_auction"), 0) == 0]
        y_lose = [to_float(r.get("reputation_before_auction"), 50.0) for r in rows if to_int(r.get("won_auction"), 0) == 0]

        if x_lose:
            ax.scatter(x_lose, y_lose, alpha=0.18, s=12, label=f"{label} lost")
        if x_win:
            ax.scatter(x_win, y_win, alpha=0.9, s=25, marker="x", label=f"{label} won")

    ax.set_title("Reputation vs Normalized Bid")
    ax.set_xlabel("Bid / Budget")
    ax.set_ylabel("Reputation Before Auction")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=7, ncol=2)
    save_fig(fig, fig_dir / "fig_reputation_vs_bid.png")


def plot_architecture_rates(group_rows: List[Dict[str, Any]], fig_dir: Path) -> None:
    labels = []
    completion = []
    completion_err = []
    acceptance = []
    acceptance_err = []

    for row in group_rows:
        label = f"{row.get('architecture', 'unknown')}-{row.get('reasoning_mode', 'unknown')}-{row.get('scenario', 'unknown')}"
        labels.append(label)
        completion.append(to_float(row.get("mean_auction_completion_rate"), 0.0))
        completion_err.append(to_float(row.get("std_auction_completion_rate"), 0.0))
        acceptance.append(to_float(row.get("mean_bid_acceptance_rate"), 0.0))
        acceptance_err.append(to_float(row.get("std_bid_acceptance_rate"), 0.0))

    if not labels:
        return

    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = list(range(len(labels)))
    w = 0.38
    ax.bar([i - w / 2 for i in x], completion, yerr=completion_err, width=w, capsize=3, label="Completion rate")
    ax.bar([i + w / 2 for i in x], acceptance, yerr=acceptance_err, width=w, capsize=3, label="Bid acceptance rate")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title("Completion and Acceptance Rates")
    ax.set_ylabel("Rate")
    ax.legend()
    save_fig(fig, fig_dir / "fig_architecture_rates.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create analysis charts from aggregated CSV outputs")
    parser.add_argument("--input-dir", type=Path, default=Path("experiments/analysis/output"))
    parser.add_argument("--fig-dir", type=Path, default=Path("experiments/analysis/figures"))
    args = parser.parse_args()

    run_rows = read_csv(args.input_dir / "run_summary.csv")
    group_rows = read_csv(args.input_dir / "architecture_summary.csv")
    auction_rows = read_csv(args.input_dir / "auction_outcomes.csv")
    provider_summary_rows = read_csv(args.input_dir / "provider_financials_summary.csv")
    provider_run_rows = read_csv(args.input_dir / "provider_run_summary.csv")
    run_economics_rows = read_csv(args.input_dir / "run_economics.csv")
    bid_market_rows = read_csv(args.input_dir / "bid_market_data.csv")

    if not run_rows:
        raise SystemExit(f"No aggregated data found in {args.input_dir}. Run aggregate_experiments.py first.")

    if auction_rows:
        plot_market_dynamics(auction_rows, args.fig_dir)
    if provider_run_rows and run_economics_rows:
        plot_profit_fairness(provider_run_rows, run_economics_rows, args.fig_dir)
    if provider_summary_rows:
        plot_cost_structure(provider_summary_rows, args.fig_dir)
    if bid_market_rows:
        plot_reputation_vs_bid(bid_market_rows, args.fig_dir)
    if group_rows:
        plot_architecture_rates(group_rows, args.fig_dir)

    print(f"Saved figures to {args.fig_dir}")


if __name__ == "__main__":
    main()

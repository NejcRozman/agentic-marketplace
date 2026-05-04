"""Generate research-oriented charts from aggregated experiment CSV outputs.

Usage:
  python experiments/analysis/visualize_experiments.py
  python experiments/analysis/visualize_experiments.py --input-dir experiments/analysis/output/h1_set --fig-dir experiments/analysis/figures/h1_set
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime
from datetime import timedelta
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


def parse_log_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S,%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def parse_unix_timestamp(value: Any) -> datetime | None:
    try:
        ts = int(float(value))
    except Exception:
        return None
    if ts <= 0:
        return None
    try:
        return datetime.fromtimestamp(ts)
    except Exception:
        return None


def build_system_financial_series(provider_fin_rows: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    ordered = []
    for row in provider_fin_rows:
        ts = parse_log_timestamp(row.get("timestamp"))
        if ts is None:
            continue
        ordered.append((ts, to_int(row.get("provider_id"), -1), row))

    ordered.sort(key=lambda item: (item[0], item[1]))
    if not ordered:
        return {
            "timestamps": [],
            "system_revenue": [],
            "system_llm_costs": [],
            "system_gas_costs": [],
            "system_other_costs": [],
            "system_costs": [],
            "system_profit": [],
            "system_profit_excluding_gas": [],
            "system_profit_excluding_llm": [],
            "provider_balance": {},
        }

    latest_by_provider: Dict[int, Dict[str, float]] = {}
    provider_balance_points: Dict[int, List[Tuple[datetime, float]]] = defaultdict(list)

    ts_points: List[datetime] = []
    revenue_points: List[float] = []
    llm_cost_points: List[float] = []
    gas_cost_points: List[float] = []
    other_cost_points: List[float] = []
    cost_points: List[float] = []
    profit_points: List[float] = []
    profit_ex_gas_points: List[float] = []
    profit_ex_llm_points: List[float] = []

    i = 0
    n = len(ordered)
    while i < n:
        ts = ordered[i][0]
        while i < n and ordered[i][0] == ts:
            _ts, provider_id, row = ordered[i]
            if provider_id >= 0:
                latest_by_provider[provider_id] = {
                    "revenue": to_float(row.get("revenue"), 0.0),
                    "llm_costs": to_float(row.get("llm_costs"), 0.0),
                    "gas_costs": to_float(row.get("gas_costs"), 0.0),
                    "total_costs": to_float(row.get("total_costs"), 0.0),
                    "net_balance": to_float(row.get("net_balance"), 0.0),
                }
                provider_balance_points[provider_id].append((ts, latest_by_provider[provider_id]["net_balance"]))
            i += 1

        system_revenue = sum(v["revenue"] for v in latest_by_provider.values())
        system_llm_costs = sum(v["llm_costs"] for v in latest_by_provider.values())
        system_gas_costs = sum(v["gas_costs"] for v in latest_by_provider.values())
        system_costs = sum(v["total_costs"] for v in latest_by_provider.values())
        system_other_costs = max(0.0, system_costs - system_llm_costs - system_gas_costs)
        system_profit = sum(v["net_balance"] for v in latest_by_provider.values())
        system_profit_excluding_gas = system_revenue - (system_costs - system_gas_costs)
        system_profit_excluding_llm = system_revenue - (system_costs - system_llm_costs)

        ts_points.append(ts)
        revenue_points.append(system_revenue)
        llm_cost_points.append(system_llm_costs)
        gas_cost_points.append(system_gas_costs)
        other_cost_points.append(system_other_costs)
        cost_points.append(system_costs)
        profit_points.append(system_profit)
        profit_ex_gas_points.append(system_profit_excluding_gas)
        profit_ex_llm_points.append(system_profit_excluding_llm)

    provider_balance_series = {
        provider_id: {
            "timestamps": [p[0] for p in points],
            "balances": [p[1] for p in points],
        }
        for provider_id, points in provider_balance_points.items()
    }

    return {
        "timestamps": ts_points,
        "system_revenue": revenue_points,
        "system_llm_costs": llm_cost_points,
        "system_gas_costs": gas_cost_points,
        "system_other_costs": other_cost_points,
        "system_costs": cost_points,
        "system_profit": profit_points,
        "system_profit_excluding_gas": profit_ex_gas_points,
        "system_profit_excluding_llm": profit_ex_llm_points,
        "provider_balance": provider_balance_series,
    }


def cumulative_max(values: List[float]) -> List[float]:
    out: List[float] = []
    running = float("-inf")
    for v in values:
        running = max(running, v)
        out.append(running)
    return out


def to_elapsed_minutes(timestamps: List[datetime]) -> List[float]:
    """Convert absolute datetimes to elapsed minutes from first point (t=0)."""
    if not timestamps:
        return []
    t0 = timestamps[0]
    return [max(0.0, (ts - t0).total_seconds() / 60.0) for ts in timestamps]


def run_duration_minutes(run_rows: List[Dict[str, Any]]) -> float:
    if not run_rows:
        return 0.0
    return max(0.0, to_float(run_rows[0].get("duration_seconds"), 0.0) / 60.0)


def infer_time_origin(run_rows: List[Dict[str, Any]], timestamps: List[datetime]) -> datetime | None:
    """Infer experiment start from duration and latest known timestamp."""
    if not timestamps:
        return None
    duration_seconds = to_float(run_rows[0].get("duration_seconds"), 0.0) if run_rows else 0.0
    if duration_seconds > 0:
        end_ts = max(timestamps)
        return end_ts - timedelta(seconds=duration_seconds)
    return min(timestamps)


def to_elapsed_minutes_from_origin(timestamps: List[datetime], origin: datetime | None) -> List[float]:
    if not timestamps:
        return []
    if origin is None:
        return to_elapsed_minutes(timestamps)
    return [max(0.0, (ts - origin).total_seconds() / 60.0) for ts in timestamps]


def pad_series_to_bounds(
    x: List[float],
    y: List[float],
    initial_value: float,
    final_x: float,
) -> Tuple[List[float], List[float]]:
    """Ensure series includes initial value at t=0 and extends flat to final_x."""
    if not x or not y:
        return [0.0, max(0.0, final_x)], [initial_value, initial_value]

    px = list(x)
    py = list(y)

    if px[0] > 0.0:
        px.insert(0, 0.0)
        py.insert(0, initial_value)
    elif px[0] == 0.0 and py[0] != initial_value:
        px.insert(0, 0.0)
        py.insert(0, initial_value)

    if final_x > px[-1]:
        px.append(final_x)
        py.append(py[-1])

    return px, py


def cumulative_completed_by_timestamp(auction_rows: List[Dict[str, Any]]) -> List[datetime]:
    completed_ts: List[datetime] = []
    for row in auction_rows:
        if str(row.get("status", "")).lower() != "completed":
            continue
        ts = parse_unix_timestamp(row.get("timestamp_ended"))
        if ts is not None:
            completed_ts.append(ts)
    completed_ts.sort()
    return completed_ts


def completed_auction_revenue_events(auction_rows: List[Dict[str, Any]]) -> List[Tuple[datetime, float]]:
    """Return revenue events at completed-auction timestamps using winning bids."""
    events: List[Tuple[datetime, float]] = []
    for row in auction_rows:
        if str(row.get("status", "")).lower() != "completed":
            continue
        ts = parse_unix_timestamp(row.get("timestamp_ended"))
        if ts is None:
            continue
        revenue_usdc = to_float(row.get("winning_bid_usdc"), 0.0)
        if revenue_usdc <= 0:
            revenue_usdc = to_float(row.get("winning_bid_amount"), 0.0) / 1e6
        events.append((ts, max(0.0, revenue_usdc)))
    events.sort(key=lambda item: item[0])
    return events


def revenue_step_series_from_events(events: List[Tuple[datetime, float]]) -> Tuple[List[float], List[float], List[float]]:
    """Build elapsed-time step series and revenue/completed-average from events."""
    if not events:
        return [], [], []

    t0 = events[0][0]
    x_minutes: List[float] = []
    cumulative_revenue: List[float] = []
    revenue_per_completed: List[float] = []
    running = 0.0

    for idx, (ts, rev) in enumerate(events, start=1):
        running += rev
        x_minutes.append(max(0.0, (ts - t0).total_seconds() / 60.0))
        cumulative_revenue.append(running)
        revenue_per_completed.append(running / idx)

    return x_minutes, cumulative_revenue, revenue_per_completed


def completed_counts_by_series_timestamps(
    series_timestamps: List[datetime],
    auction_rows: List[Dict[str, Any]],
    fallback_completed: int,
) -> List[int]:
    completed_ts = cumulative_completed_by_timestamp(auction_rows)
    completed_counts: List[int] = []
    cursor = 0
    for ts in series_timestamps:
        while cursor < len(completed_ts) and completed_ts[cursor] <= ts:
            cursor += 1
        completed = cursor if cursor > 0 else 0
        if completed == 0 and fallback_completed > 0:
            # If timestamps are unavailable/misaligned, keep denominator stable.
            completed = fallback_completed
        completed_counts.append(completed)
    return completed_counts


def plot_profit_over_time(
    provider_fin_rows: List[Dict[str, Any]],
    auction_rows: List[Dict[str, Any]],
    run_rows: List[Dict[str, Any]],
    fig_dir: Path,
) -> None:
    series = build_system_financial_series(provider_fin_rows)
    ts_points = series["timestamps"]
    if not ts_points:
        return

    fallback_completed = 0
    if run_rows:
        fallback_completed = to_int(run_rows[0].get("completed_auctions"), 0)

    profit_per_completed: List[float] = []
    completed_counts = completed_counts_by_series_timestamps(ts_points, auction_rows, fallback_completed)

    for idx, total_profit in enumerate(series["system_profit"]):
        denom = completed_counts[idx]
        profit_per_completed.append(total_profit / denom if denom > 0 else 0.0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0))
    ax_totals, ax_per_completed = axes

    duration_minutes = run_duration_minutes(run_rows)
    origin = infer_time_origin(run_rows, ts_points)
    x_minutes = to_elapsed_minutes_from_origin(ts_points, origin)
    final_x = max(duration_minutes, x_minutes[-1] if x_minutes else 0.0)

    revenue_series = cumulative_max(series["system_revenue"])
    costs_series = cumulative_max(series["system_costs"])
    x_rev, y_rev = pad_series_to_bounds(x_minutes, revenue_series, 0.0, final_x)
    x_cost, y_cost = pad_series_to_bounds(x_minutes, costs_series, 0.0, final_x)
    x_profit, y_profit = pad_series_to_bounds(x_minutes, series["system_profit"], 0.0, final_x)
    x_ppc, y_ppc = pad_series_to_bounds(x_minutes, profit_per_completed, 0.0, final_x)

    if x_rev:
        ax_totals.step(x_rev, y_rev, where="post", label="Revenue", linewidth=1.5)
    ax_totals.step(x_cost, y_cost, where="post", label="Total costs", linewidth=1.5)
    ax_totals.plot(x_profit, y_profit, label="Profit (net balance)", linewidth=1.6)
    ax_totals.set_title("System Financial Trajectory")
    ax_totals.set_xlabel("Elapsed Time (min)")
    ax_totals.set_ylabel("USD")
    ax_totals.legend(fontsize=8)

    ax_per_completed.plot(x_ppc, y_ppc, color="#2a9d8f", linewidth=1.6)
    ax_per_completed.set_title("Profit Per Completed Auction Over Time")
    ax_per_completed.set_xlabel("Elapsed Time (min)")
    ax_per_completed.set_ylabel("USD")

    save_fig(fig, fig_dir / "fig_profit_over_time.png")


def plot_revenue_over_time(
    provider_fin_rows: List[Dict[str, Any]],
    auction_rows: List[Dict[str, Any]],
    run_rows: List[Dict[str, Any]],
    fig_dir: Path,
) -> None:
    series = build_system_financial_series(provider_fin_rows)
    ts_points = series["timestamps"]
    if not ts_points and not auction_rows:
        return

    events = completed_auction_revenue_events(auction_rows)
    event_ts = [ts for ts, _ in events]
    origin_candidates = list(ts_points) + event_ts
    origin = infer_time_origin(run_rows, origin_candidates)

    event_x_raw, event_revenue, event_revenue_per_completed = revenue_step_series_from_events(events)
    if event_ts and origin is not None:
        event_x = to_elapsed_minutes_from_origin(event_ts, origin)
    else:
        event_x = event_x_raw

    # Fallback to runtime financial series when completed-auction timestamps are unavailable.
    x_minutes = event_x if event_x else to_elapsed_minutes_from_origin(ts_points, origin)
    duration_minutes = run_duration_minutes(run_rows)
    final_x = max(duration_minutes, x_minutes[-1] if x_minutes else 0.0)
    if event_revenue:
        revenue_series = event_revenue
        revenue_per_completed_steps = event_revenue_per_completed
    else:
        revenue_series = cumulative_max(series["system_revenue"])
        fallback_completed = to_int(run_rows[0].get("completed_auctions"), 0) if run_rows else 0
        completed_counts = completed_counts_by_series_timestamps(ts_points, auction_rows, fallback_completed)
        revenue_per_completed_steps: List[float] = []
        for idx, total_revenue in enumerate(revenue_series):
            denom = completed_counts[idx] if idx < len(completed_counts) else 0
            revenue_per_completed_steps.append(total_revenue / denom if denom > 0 else 0.0)

    x_rev, y_rev = pad_series_to_bounds(x_minutes, revenue_series, 0.0, final_x)
    x_rpc, y_rpc = pad_series_to_bounds(x_minutes, revenue_per_completed_steps, 0.0, final_x)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0))
    ax_total, ax_norm = axes

    if x_rev:
        ax_total.step(x_rev, y_rev, where="post", linewidth=1.6, color="#1d3557")
    else:
        ax_total.plot([], [])
    ax_total.set_title("Cumulative Revenue Over Time")
    ax_total.set_xlabel("Elapsed Time (min)")
    ax_total.set_ylabel("USD")

    if x_rpc:
        ax_norm.step(x_rpc, y_rpc, where="post", linewidth=1.6, color="#2a9d8f")
    else:
        ax_norm.plot([], [])
    ax_norm.set_title("Revenue Per Completed Auction Over Time")
    ax_norm.set_xlabel("Elapsed Time (min)")
    ax_norm.set_ylabel("USD")

    save_fig(fig, fig_dir / "fig_revenue_over_time.png")


def plot_decomposed_normalized_metrics(run_economics_rows: List[Dict[str, Any]], fig_dir: Path) -> None:
    if not run_economics_rows:
        return

    by_label: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in run_economics_rows:
        by_label[label_of(row)].append(row)

    labels = sorted(by_label.keys())
    if not labels:
        return

    run_counts = [len(by_label[l]) for l in labels]
    single_run_mode = all(c == 1 for c in run_counts)

    completed_metrics = [
        ("revenue_per_completed_auction", "Revenue/completed"),
        ("cost_per_completed_auction", "Total cost/completed"),
        ("profit_per_completed_auction", "Profit/completed"),
        ("profit_excluding_gas_per_completed_auction", "Profit ex gas/completed"),
        ("profit_excluding_llm_per_completed_auction", "Profit ex LLM/completed"),
    ]
    bid_metrics = [
        ("llm_cost_per_bid_attempted", "LLM / attempted bid"),
        ("gas_cost_per_on_chain_bid", "Gas / on-chain bid"),
        ("cost_per_on_chain_bid", "Total cost / on-chain bid"),
    ]

    completed_vals: Dict[str, List[float]] = {}
    bid_vals: Dict[str, List[float]] = {}
    for key, _ in completed_metrics:
        if single_run_mode:
            completed_vals[key] = [to_float(by_label[l][0].get(key), 0.0) for l in labels]
        else:
            completed_vals[key] = [fmean([to_float(r.get(key), 0.0) for r in by_label[l]]) for l in labels]
    for key, _ in bid_metrics:
        if single_run_mode:
            bid_vals[key] = [to_float(by_label[l][0].get(key), 0.0) for l in labels]
        else:
            bid_vals[key] = [fmean([to_float(r.get(key), 0.0) for r in by_label[l]]) for l in labels]

    x = list(range(len(labels)))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    ax_completed, ax_bid_norm = axes

    w_completed = 0.14
    offsets_completed = [-2 * w_completed, -1 * w_completed, 0.0, 1 * w_completed, 2 * w_completed]
    for idx, (key, label_text) in enumerate(completed_metrics):
        ax_completed.bar(
            [i + offsets_completed[idx] for i in x],
            completed_vals[key],
            width=w_completed,
            label=label_text,
        )
    ax_completed.set_xticks(x)
    ax_completed.set_xticklabels(labels, rotation=20, ha="right")
    if single_run_mode:
        ax_completed.set_title("Decomposed Economics Per Completed Auction (Single Run)")
    else:
        ax_completed.set_title("Decomposed Economics Per Completed Auction (Mean Across Runs)")
    ax_completed.set_ylabel("USD")
    ax_completed.legend(fontsize=7, ncol=2)

    w_bid = 0.24
    offsets_bid = [-1.0 * w_bid, 0.0, 1.0 * w_bid]
    for idx, (key, label_text) in enumerate(bid_metrics):
        ax_bid_norm.bar(
            [i + offsets_bid[idx] for i in x],
            bid_vals[key],
            width=w_bid,
            label=label_text,
        )
    ax_bid_norm.set_xticks(x)
    ax_bid_norm.set_xticklabels(labels, rotation=20, ha="right")
    if single_run_mode:
        ax_bid_norm.set_title("Bid-Normalized Cost Metrics (Single Run)")
    else:
        ax_bid_norm.set_title("Bid-Normalized Cost Metrics (Mean Across Runs)")
    ax_bid_norm.set_ylabel("USD")
    ax_bid_norm.legend(fontsize=7)

    save_fig(fig, fig_dir / "fig_decomposed_normalized_metrics.png")


def plot_balance_trajectories(provider_fin_rows: List[Dict[str, Any]], run_rows: List[Dict[str, Any]], fig_dir: Path) -> None:
    series = build_system_financial_series(provider_fin_rows)
    provider_series = series["provider_balance"]
    if not provider_series:
        return

    all_ts: List[datetime] = []
    for row in provider_series.values():
        all_ts.extend(row.get("timestamps", []))
    origin = infer_time_origin(run_rows, all_ts)
    duration_minutes = run_duration_minutes(run_rows)

    fig, ax = plt.subplots(figsize=(10, 5.0))
    for provider_id in sorted(provider_series.keys()):
        row = provider_series[provider_id]
        x_minutes = to_elapsed_minutes_from_origin(row["timestamps"], origin)
        final_x = max(duration_minutes, x_minutes[-1] if x_minutes else 0.0)
        x_pad, y_pad = pad_series_to_bounds(x_minutes, row["balances"], 0.0, final_x)
        ax.plot(x_pad, y_pad, linewidth=1.4, label=f"Provider {provider_id}")

    ax.set_title("Provider Balance Trajectories")
    ax.set_xlabel("Elapsed Time (min)")
    ax.set_ylabel("Net balance (USD)")
    ax.legend(fontsize=8, ncol=2)
    save_fig(fig, fig_dir / "fig_balance_trajectories.png")


def plot_reputation_trajectories(reputation_rows: List[Dict[str, Any]], fig_dir: Path) -> None:
    by_provider: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    for row in reputation_rows:
        provider_id = to_int(row.get("provider_id"), -1)
        after_auction = to_int(row.get("after_auction"), -1)
        score = to_float(row.get("reputation_score"), -1.0)
        if provider_id < 0 or after_auction < 0 or score < 0:
            continue
        by_provider[provider_id].append((after_auction, score))

    if not by_provider:
        return

    fig, ax = plt.subplots(figsize=(10, 5.0))
    for provider_id in sorted(by_provider.keys()):
        points = sorted(by_provider[provider_id], key=lambda x: x[0])
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.plot(xs, ys, marker="o", linewidth=1.4, label=f"Provider {provider_id}")

    ax.set_title("Reputation Trajectories")
    ax.set_xlabel("After Auction")
    ax.set_ylabel("Reputation score")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8, ncol=2)
    save_fig(fig, fig_dir / "fig_reputation_trajectories.png")


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
        x_win: List[float] = []
        y_win: List[float] = []
        x_lose: List[float] = []
        y_lose: List[float] = []

        for r in rows:
            rep_raw = r.get("reputation_before_auction")
            if rep_raw in (None, "", "null"):
                continue
            rep = to_float(rep_raw, -1.0)
            if rep < 0:
                continue
            bid_norm = to_float(r.get("normalized_bid"), 0.0)
            if to_int(r.get("won_auction"), 0) == 1:
                x_win.append(bid_norm)
                y_win.append(rep)
            else:
                x_lose.append(bid_norm)
                y_lose.append(rep)

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
    parser.add_argument(
        "--profile",
        choices=["focused", "full"],
        default="focused",
        help="focused: key homogeneous plots; full: include legacy comparison plots",
    )
    args = parser.parse_args()

    run_rows = read_csv(args.input_dir / "run_summary.csv")
    group_rows = read_csv(args.input_dir / "architecture_summary.csv")
    auction_rows = read_csv(args.input_dir / "auction_outcomes.csv")
    provider_summary_rows = read_csv(args.input_dir / "provider_financials_summary.csv")
    provider_fin_evolution_rows = read_csv(args.input_dir / "provider_financials_evolution.csv")
    provider_run_rows = read_csv(args.input_dir / "provider_run_summary.csv")
    run_economics_rows = read_csv(args.input_dir / "run_economics.csv")
    bid_market_rows = read_csv(args.input_dir / "bid_market_data.csv")
    reputation_rows = read_csv(args.input_dir / "reputation_evolution.csv")

    if not run_rows:
        raise SystemExit(f"No aggregated data found in {args.input_dir}. Run aggregate_experiments.py first.")

    if provider_fin_evolution_rows:
        plot_profit_over_time(provider_fin_evolution_rows, auction_rows, run_rows, args.fig_dir)
        plot_revenue_over_time(provider_fin_evolution_rows, auction_rows, run_rows, args.fig_dir)
        plot_balance_trajectories(provider_fin_evolution_rows, run_rows, args.fig_dir)
    if reputation_rows:
        plot_reputation_trajectories(reputation_rows, args.fig_dir)
    if bid_market_rows:
        plot_reputation_vs_bid(bid_market_rows, args.fig_dir)
    if run_economics_rows:
        plot_decomposed_normalized_metrics(run_economics_rows, args.fig_dir)

    if args.profile == "full":
        if auction_rows:
            plot_market_dynamics(auction_rows, args.fig_dir)
        if provider_run_rows and run_economics_rows:
            plot_profit_fairness(provider_run_rows, run_economics_rows, args.fig_dir)
        if provider_summary_rows:
            plot_cost_structure(provider_summary_rows, args.fig_dir)
        if group_rows:
            plot_architecture_rates(group_rows, args.fig_dir)

    print(f"Saved figures to {args.fig_dir}")


if __name__ == "__main__":
    main()

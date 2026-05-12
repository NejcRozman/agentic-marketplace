"""Aggregate experiment metrics JSON files into analysis-ready CSV tables.

Usage:
    python experiments/analysis/aggregate_experiments.py
    python experiments/analysis/aggregate_experiments.py --input-dir experiments/data/metrics/h1_homogeneous_set
    python experiments/analysis/aggregate_experiments.py --manifest experiments/analysis/run_manifest.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import fmean, stdev
from typing import Any, Dict, Iterable, List, Optional, Tuple


def normalize_reasoning_label(reasoning_mode: Any) -> str:
    raw = str(reasoning_mode or "").strip().lower()
    if raw == "llm_react":
        return "react"
    if raw == "llm_strategic":
        return "strategic"
    return raw or "unknown"


def provider_display_architecture(profile: Dict[str, Any]) -> str:
    architecture = str(profile.get("architecture") or "").strip().lower()
    reasoning = normalize_reasoning_label(profile.get("reasoning_mode"))
    if architecture.isdigit():
        prefix = "h" if reasoning == "heuristic" else "a"
        return f"{prefix}{architecture}"
    return architecture or "unknown"


def provider_subtype_label(profile: Dict[str, Any]) -> str:
    group_name = str(profile.get("group_name") or "").strip()
    if group_name and group_name != "default":
        return group_name
    architecture = provider_display_architecture(profile)
    reasoning = normalize_reasoning_label(profile.get("reasoning_mode"))
    if reasoning in ("unknown", ""):
        return architecture
    return f"{architecture}_{reasoning}"


def provider_profile_map(provider_profiles: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for profile in provider_profiles:
        try:
            provider_id = int(profile.get("provider_id"))
        except Exception:
            continue
        out[provider_id] = profile
    return out


def provider_mix_string(provider_profiles: List[Dict[str, Any]]) -> str:
    counts: Dict[str, int] = defaultdict(int)
    for profile in provider_profiles:
        counts[provider_subtype_label(profile)] += 1
    return "|".join(f"{label}:{counts[label]}" for label in sorted(counts.keys()))


def enrich_with_provider_profile(row: Dict[str, Any], profile: Optional[Dict[str, Any]], prefix: str = "provider") -> Dict[str, Any]:
    if not profile:
        row[f"{prefix}_group_name"] = None
        row[f"{prefix}_architecture"] = None
        row[f"{prefix}_reasoning_mode"] = None
        row[f"{prefix}_heuristic_strategy"] = None
        row[f"{prefix}_heuristic_min_margin"] = None
        row[f"{prefix}_heuristic_max_margin"] = None
        row[f"{prefix}_subtype_label"] = "unknown"
        return row

    row[f"{prefix}_group_name"] = profile.get("group_name")
    row[f"{prefix}_architecture"] = profile.get("architecture")
    row[f"{prefix}_reasoning_mode"] = normalize_reasoning_label(profile.get("reasoning_mode"))
    row[f"{prefix}_heuristic_strategy"] = profile.get("heuristic_strategy")
    row[f"{prefix}_heuristic_min_margin"] = profile.get("heuristic_min_margin")
    row[f"{prefix}_heuristic_max_margin"] = profile.get("heuristic_max_margin")
    row[f"{prefix}_subtype_label"] = provider_subtype_label(profile)
    return row


def infer_labels(experiment_id: str, provider_profiles: Optional[List[Dict[str, Any]]] = None) -> Tuple[str, str, bool, str, Optional[int]]:
    """Infer architecture/reasoning/scenario labels from experiment_id and provider metadata."""
    exp = experiment_id.lower()
    arch_match = re.search(r"([ah])(\d+)", exp)
    architecture = f"{arch_match.group(1)}{arch_match.group(2)}" if arch_match else "unknown"

    if "heuristic" in exp:
        reasoning = "heuristic"
    elif "react" in exp:
        reasoning = "react"
    elif "strategic" in exp:
        reasoning = "strategic"
    else:
        reasoning = "unknown"

    if "heterogeneous" in exp or "hetero" in exp:
        scenario = "heterogeneous"
    elif "homogeneous" in exp:
        scenario = "homogeneous"
    elif "smoke" in exp:
        scenario = "smoke"
    else:
        scenario = "unknown"

    seed_match = re.search(r"(?:^|[_-])seed[_-]?(\d+)(?:$|[_-])", exp)
    seed = int(seed_match.group(1)) if seed_match else None

    normalized_profiles = [p for p in (provider_profiles or []) if isinstance(p, dict)]
    if normalized_profiles:
        architectures = sorted(
            {
                provider_display_architecture(profile)
                for profile in normalized_profiles
                if provider_display_architecture(profile) != "unknown"
            }
        )
        reasoning_modes = sorted(
            {
                normalize_reasoning_label(profile.get("reasoning_mode"))
                for profile in normalized_profiles
                if normalize_reasoning_label(profile.get("reasoning_mode")) != "unknown"
            }
        )
        if architectures:
            architecture = architectures[0] if len(architectures) == 1 else "mixed"
        if reasoning_modes:
            reasoning = reasoning_modes[0] if len(reasoning_modes) == 1 else "mixed"

    return architecture, reasoning, reasoning == "heuristic", scenario, seed


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def _dedupe_paths(paths: Iterable[Path]) -> List[Path]:
    seen = set()
    out: List[Path] = []
    for p in paths:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        out.append(rp)
    return sorted(out)


def collect_files(
    metrics_root: Path,
    input_dirs: List[Path],
    input_glob: str,
    manifest: Optional[Path],
    include_regex: Optional[str],
) -> List[Path]:
    files: List[Path] = []

    search_roots = input_dirs if input_dirs else [metrics_root]
    for root in search_roots:
        if root.is_file() and root.name == "experiment_metrics.json":
            files.append(root)
            continue
        if root.is_dir():
            # Support input dirs where metrics file is at the directory root.
            direct_metrics = root / "experiment_metrics.json"
            if direct_metrics.exists():
                files.append(direct_metrics)
            files.extend(root.glob(input_glob))

    if manifest:
        if not manifest.exists():
            raise SystemExit(f"Manifest not found: {manifest}")
        for raw in manifest.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            candidate = Path(line)
            if not candidate.is_absolute():
                candidate = (manifest.parent / candidate).resolve()

            if candidate.is_file() and candidate.name == "experiment_metrics.json":
                files.append(candidate)
            elif candidate.is_dir():
                files.extend(candidate.glob(input_glob))
            else:
                # Treat non-path entries as experiment IDs under metrics_root
                exp_file = metrics_root / line / "experiment_metrics.json"
                if exp_file.exists():
                    files.append(exp_file)

    files = _dedupe_paths(files)

    if include_regex:
        pattern = re.compile(include_regex)
        files = [f for f in files if pattern.search(f.parent.name)]

    return files


def compute_stats(values: List[float]) -> Dict[str, float]:
    """Compute robust descriptive stats for a metric list."""
    if not values:
        return {
            "n": 0,
            "mean": 0.0,
            "std": 0.0,
            "sem": 0.0,
            "ci95_low": 0.0,
            "ci95_high": 0.0,
            "min": 0.0,
            "max": 0.0,
        }

    n = len(values)
    mean = fmean(values)
    std = stdev(values) if n > 1 else 0.0
    sem = safe_div(std, math.sqrt(n)) if n > 1 else 0.0
    ci95 = 1.96 * sem
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci95_low": mean - ci95,
        "ci95_high": mean + ci95,
        "min": min(values),
        "max": max(values),
    }


def summarize_groups(
    run_rows: List[Dict[str, Any]],
    group_fields: List[str],
    metric_fields: List[str],
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        key = tuple(row.get(f) for f in group_fields)
        grouped[key].append(row)

    summary_rows: List[Dict[str, Any]] = []
    for key, rows in sorted(grouped.items(), key=lambda kv: kv[0]):
        out: Dict[str, Any] = {field: key[i] for i, field in enumerate(group_fields)}
        out["n_runs"] = len(rows)
        for metric in metric_fields:
            vals = [safe_float(r.get(metric)) for r in rows]
            stats = compute_stats(vals)
            out[f"mean_{metric}"] = stats["mean"]
            out[f"std_{metric}"] = stats["std"]
            out[f"sem_{metric}"] = stats["sem"]
            out[f"ci95_low_{metric}"] = stats["ci95_low"]
            out[f"ci95_high_{metric}"] = stats["ci95_high"]
            out[f"min_{metric}"] = stats["min"]
            out[f"max_{metric}"] = stats["max"]
        summary_rows.append(out)

    return summary_rows


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def gini(values: List[float]) -> float:
    """Compute Gini coefficient (0=fair, 1=unequal)."""
    if not values:
        return 0.0
    arr = sorted(max(0.0, v) for v in values)
    n = len(arr)
    total = sum(arr)
    if total <= 0:
        return 0.0
    weighted_sum = 0.0
    for i, v in enumerate(arr, start=1):
        weighted_sum += i * v
    return (2.0 * weighted_sum) / (n * total) - (n + 1) / n


def reputation_snapshot_map(rep_evolution: List[Dict[str, Any]]) -> Dict[int, Dict[int, float]]:
    """Map after_auction -> {provider_id -> score}."""
    out: Dict[int, Dict[int, float]] = {}
    for item in rep_evolution:
        if not isinstance(item, dict):
            continue
        if "reputations" not in item or not isinstance(item["reputations"], dict):
            continue
        after = item.get("after_auction")
        if isinstance(after, str) and not after.isdigit():
            continue
        after_id = safe_int(after, default=-1)
        if after_id < 0:
            continue
        out[after_id] = {
            safe_int(pid): safe_float(rep.get("score", 50.0))
            for pid, rep in item["reputations"].items()
        }
    return out


def reputation_before_auction(rep_map: Dict[int, Dict[int, float]], auction_id: int, provider_id: int) -> Optional[float]:
    """Find provider reputation from the latest snapshot at or before auction_id - 1."""
    if auction_id <= 0:
        return None
    keys = [k for k in rep_map.keys() if k <= auction_id - 1]
    if not keys:
        return None
    nearest = max(keys)
    return rep_map.get(nearest, {}).get(provider_id)


def parse_reputation_rows(base: Dict[str, Any], rep_evolution: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seq = 0

    for item in rep_evolution:
        if isinstance(item, dict) and "reputations" in item and isinstance(item["reputations"], dict):
            after_auction = item.get("after_auction")
            ts = item.get("timestamp")
            source = item.get("source", "snapshot")
            for provider_id, rep in item["reputations"].items():
                seq += 1
                rows.append(
                    {
                        **base,
                        "sequence": seq,
                        "timestamp": ts,
                        "after_auction": after_auction,
                        "provider_id": int(provider_id),
                        "reputation_score": rep.get("score"),
                        "feedback_count": rep.get("feedback_count"),
                        "source": source,
                    }
                )
        elif isinstance(item, dict) and "provider_id" in item:
            seq += 1
            rows.append(
                {
                    **base,
                    "sequence": seq,
                    "timestamp": item.get("timestamp"),
                    "after_auction": item.get("after_auction"),
                    "provider_id": item.get("provider_id"),
                    "reputation_score": item.get("reputation_score"),
                    "feedback_count": item.get("feedback_count"),
                    "source": item.get("source", "legacy"),
                }
            )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate experiment metrics into CSV tables")
    parser.add_argument("--metrics-root", type=Path, default=Path("experiments/data/metrics"))
    parser.add_argument(
        "--input-dir",
        type=Path,
        nargs="*",
        default=[],
        help="Optional directories/files to analyze instead of scanning metrics-root",
    )
    parser.add_argument(
        "--input-glob",
        type=str,
        default="*/experiment_metrics.json",
        help="Glob used within input directories (default: */experiment_metrics.json)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional manifest file with paths or experiment IDs (one per line)",
    )
    parser.add_argument(
        "--include-regex",
        type=str,
        default=None,
        help="Optional regex filter applied to experiment directory names",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/analysis/output"))
    parser.add_argument(
        "--output-subdir",
        type=str,
        default="",
        help="Optional subdirectory under output-dir (e.g., homogeneous_h1_set)",
    )
    parser.add_argument(
        "--group-by",
        type=str,
        default="architecture,reasoning_mode,scenario",
        help="Comma-separated run-level grouping keys for statistics",
    )
    args = parser.parse_args()

    files = collect_files(
        metrics_root=args.metrics_root,
        input_dirs=args.input_dir,
        input_glob=args.input_glob,
        manifest=args.manifest,
        include_regex=args.include_regex,
    )
    if not files:
        raise SystemExit("No metrics files found for the selected inputs")

    run_rows: List[Dict[str, Any]] = []
    auction_rows: List[Dict[str, Any]] = []
    provider_summary_rows: List[Dict[str, Any]] = []
    provider_fin_evolution_rows: List[Dict[str, Any]] = []
    reputation_rows: List[Dict[str, Any]] = []
    bid_market_rows: List[Dict[str, Any]] = []
    provider_run_rows: List[Dict[str, Any]] = []
    provider_type_run_rows: List[Dict[str, Any]] = []
    run_economics_rows: List[Dict[str, Any]] = []

    for metrics_file in files:
        data = json.loads(metrics_file.read_text(encoding="utf-8"))

        experiment_id = data.get("experiment_id", metrics_file.parent.name)
        provider_profiles = data.get("agents", {}).get("provider_profiles", []) or []
        provider_meta_map = provider_profile_map(provider_profiles)
        architecture, reasoning_mode, is_heuristic, scenario, seed = infer_labels(experiment_id, provider_profiles)
        summary = data.get("summary", {})
        subtype_labels = sorted({provider_subtype_label(profile) for profile in provider_profiles})

        base = {
            "experiment_id": experiment_id,
            "architecture": architecture,
            "reasoning_mode": reasoning_mode,
            "is_heuristic": int(is_heuristic),
            "scenario": scenario,
            "seed": seed,
            "provider_mix": provider_mix_string(provider_profiles),
            "provider_subtype_count": len(subtype_labels),
            "metrics_file": str(metrics_file),
        }

        run_rows.append(
            {
                **base,
                "success": int(bool(data.get("success", False))),
                "duration_seconds": data.get("duration_seconds", 0),
                "total_auctions": summary.get("total_auctions", 0),
                "completed_auctions": summary.get("completed_auctions", 0),
                "failed_auctions": summary.get("failed_auctions", 0),
                "total_bids_attempted": summary.get("total_bids_attempted", 0),
                "total_bids_on_chain": summary.get("total_bids_on_chain", 0),
                "auction_completion_rate": safe_div(summary.get("completed_auctions", 0), summary.get("total_auctions", 0)),
                "bid_acceptance_rate": safe_div(summary.get("total_bids_on_chain", 0), summary.get("total_bids_attempted", 0)),
            }
        )

        auctions = data.get("auctions", [])
        rep_map = reputation_snapshot_map(data.get("reputation_evolution", []))

        provider_bid_stats: Dict[int, Dict[str, float]] = defaultdict(
            lambda: {
                "eligible": 0,
                "participated": 0,
                "attempted": 0,
                "on_chain": 0,
                "wins": 0,
                "attempted_bid_sum": 0.0,
                "on_chain_bid_sum": 0.0,
            }
        )

        for auction in auctions:
            budget = safe_float(auction.get("budget", 0.0))
            auction_id = safe_int(auction.get("auction_id"), default=0)
            winner_id = safe_int(auction.get("winner_id"), default=0)
            on_chain_providers = {safe_int(b.get("agent_id"), default=-1) for b in auction.get("bids_on_chain", [])}
            attempted_providers_in_auction: Set[int] = set()
            eligible_provider_ids = [safe_int(pid, default=-1) for pid in auction.get("eligible_agent_ids", []) or []]
            for eligible_provider_id in eligible_provider_ids:
                if eligible_provider_id < 0:
                    continue
                provider_bid_stats[eligible_provider_id]["eligible"] += 1

            winner_profile = provider_meta_map.get(winner_id)

            auction_row = {
                **base,
                "auction_id": auction_id,
                "timestamp_created": auction.get("timestamp_created"),
                "timestamp_ended": auction.get("timestamp_ended"),
                "status": auction.get("status"),
                "winner_id": winner_id if winner_id > 0 else None,
                "winning_bid_amount": auction.get("winning_bid_amount"),
                "winning_bid_usdc": safe_float(auction.get("winning_bid_amount", 0.0)) / 1e6,
                "budget": budget,
                "budget_usdc": budget / 1e6,
                "winning_bid_ratio": safe_div(safe_float(auction.get("winning_bid_amount", 0.0)), budget),
                "bid_count_on_chain": len(auction.get("bids_on_chain", [])),
                "bid_count_attempted": len(auction.get("bids_attempted", [])),
                "bid_failure_count": len(auction.get("bid_failures", [])),
                "eligible_provider_count": len([pid for pid in eligible_provider_ids if pid >= 0]),
                "service_executed": int(bool(auction.get("service_executed", False))),
                "quality_rating": auction.get("quality_rating"),
                "feedback_submitted": int(bool(auction.get("feedback_submitted", False))),
                "execution_duration_seconds": auction.get("execution_duration_seconds", 0),
                "evaluation_duration_seconds": auction.get("evaluation_duration_seconds", 0),
            }
            auction_rows.append(enrich_with_provider_profile(auction_row, winner_profile, prefix="winner"))

            # Per-attempt bid market dataset (for bid price vs reputation analysis)
            for bid in auction.get("bids_attempted", []):
                provider_id = safe_int(bid.get("provider_id"), default=-1)
                amount = safe_float(bid.get("amount", 0.0))
                if provider_id < 0:
                    continue
                if provider_id not in attempted_providers_in_auction:
                    provider_bid_stats[provider_id]["participated"] += 1
                    attempted_providers_in_auction.add(provider_id)
                provider_bid_stats[provider_id]["attempted"] += 1
                provider_bid_stats[provider_id]["attempted_bid_sum"] += amount
                rep_before = reputation_before_auction(rep_map, auction_id, provider_id)
                bid_row = {
                    **base,
                    "auction_id": auction_id,
                    "provider_id": provider_id,
                    "bid_amount": amount,
                    "bid_amount_usdc": amount / 1e6,
                    "budget": budget,
                    "budget_usdc": budget / 1e6,
                    "normalized_bid": safe_div(amount, budget),
                    "reputation_before_auction": rep_before,
                    "won_auction": int(provider_id == winner_id and winner_id > 0),
                    "has_on_chain_bid": int(provider_id in on_chain_providers),
                }
                bid_market_rows.append(enrich_with_provider_profile(bid_row, provider_meta_map.get(provider_id)))

            for on_chain_bid in auction.get("bids_on_chain", []):
                provider_id = safe_int(on_chain_bid.get("agent_id"), default=-1)
                amount = safe_float(on_chain_bid.get("amount", 0.0))
                if provider_id < 0:
                    continue
                provider_bid_stats[provider_id]["on_chain"] += 1
                provider_bid_stats[provider_id]["on_chain_bid_sum"] += amount

            if winner_id > 0:
                provider_bid_stats[winner_id]["wins"] += 1

        fin = data.get("provider_financials", {})
        by_provider = fin.get("by_provider", {})

        net_balances: List[float] = []
        total_revenue = 0.0
        total_costs = 0.0
        total_llm_costs = 0.0
        total_gas_costs = 0.0
        current_provider_run_rows: List[Dict[str, Any]] = []

        for provider_id, vals in by_provider.items():
            pid = int(provider_id)
            provider_profile = vals.get("profile") or provider_meta_map.get(pid)
            revenue = safe_float(vals.get("revenue", 0.0))
            llm_costs = safe_float(vals.get("llm_costs", 0.0))
            gas_costs = safe_float(vals.get("gas_costs", 0.0))
            total_cost = safe_float(vals.get("total_costs", 0.0))
            net_balance = safe_float(vals.get("net_balance", 0.0))
            provider_summary_row = {
                **base,
                "provider_id": pid,
                "revenue": revenue,
                "llm_costs": llm_costs,
                "gas_costs": gas_costs,
                "total_costs": total_cost,
                "net_balance": net_balance,
                "snapshots": vals.get("snapshots", 0),
            }
            provider_summary_rows.append(enrich_with_provider_profile(provider_summary_row, provider_profile))

            stats = provider_bid_stats.get(pid, {})
            eligible = safe_float(stats.get("eligible", 0.0))
            participated = safe_float(stats.get("participated", 0.0))
            attempted = safe_float(stats.get("attempted", 0.0))
            on_chain = safe_float(stats.get("on_chain", 0.0))
            wins = safe_float(stats.get("wins", 0.0))
            attempted_bid_sum = safe_float(stats.get("attempted_bid_sum", 0.0))
            on_chain_bid_sum = safe_float(stats.get("on_chain_bid_sum", 0.0))

            provider_run_row = {
                **base,
                "provider_id": pid,
                "eligible_auctions": int(eligible),
                "participated_auctions": int(participated),
                "wins": int(wins),
                "bids_attempted": int(attempted),
                "bids_on_chain": int(on_chain),
                "bid_participation_rate": safe_div(participated, eligible),
                "bid_acceptance_rate": safe_div(on_chain, attempted),
                "win_rate_per_eligible": safe_div(wins, eligible),
                "mean_attempted_bid_usdc": safe_div(attempted_bid_sum / 1e6, attempted),
                "mean_on_chain_bid_usdc": safe_div(on_chain_bid_sum / 1e6, on_chain),
                "revenue": revenue,
                "llm_costs": llm_costs,
                "gas_costs": gas_costs,
                "total_costs": total_cost,
                "net_balance": net_balance,
            }
            enriched_provider_run_row = enrich_with_provider_profile(provider_run_row, provider_profile)
            provider_run_rows.append(enriched_provider_run_row)
            current_provider_run_rows.append(enriched_provider_run_row)

            net_balances.append(net_balance)
            total_revenue += revenue
            total_costs += total_cost
            total_llm_costs += llm_costs
            total_gas_costs += gas_costs

        for row in fin.get("evolution", []):
            provider_id = safe_int(row.get("provider_id"), default=-1)
            fin_row = {
                **base,
                "timestamp": row.get("timestamp"),
                "provider_id": provider_id,
                "revenue": row.get("revenue", 0.0),
                "llm_costs": row.get("llm_costs", 0.0),
                "gas_costs": row.get("gas_costs", 0.0),
                "total_costs": row.get("total_costs", 0.0),
                "net_balance": row.get("net_balance", 0.0),
            }
            provider_fin_evolution_rows.append(enrich_with_provider_profile(fin_row, row.get("profile") or provider_meta_map.get(provider_id)))

        for rep_row in parse_reputation_rows(base, data.get("reputation_evolution", [])):
            provider_id = safe_int(rep_row.get("provider_id"), default=-1)
            reputation_rows.append(enrich_with_provider_profile(rep_row, provider_meta_map.get(provider_id)))

        completed_auctions = safe_float(summary.get("completed_auctions", 0))
        total_bids_attempted = safe_float(summary.get("total_bids_attempted", 0))
        total_bids_on_chain = safe_float(summary.get("total_bids_on_chain", 0))
        total_other_costs = max(0.0, total_costs - total_llm_costs - total_gas_costs)

        total_profit = total_revenue - total_costs
        total_profit_excluding_gas = total_revenue - (total_costs - total_gas_costs)
        total_profit_excluding_llm = total_revenue - (total_costs - total_llm_costs)
        total_profit_excluding_llm_gas = total_revenue - total_other_costs

        run_economics_rows.append(
            {
                **base,
                "provider_count": len(by_provider),
                "system_total_revenue": total_revenue,
                "system_total_llm_costs": total_llm_costs,
                "system_total_gas_costs": total_gas_costs,
                "system_total_other_costs": total_other_costs,
                "system_total_costs": total_costs,
                "system_total_profit": total_profit,
                "system_total_profit_excluding_gas": total_profit_excluding_gas,
                "system_total_profit_excluding_llm": total_profit_excluding_llm,
                "system_total_profit_excluding_llm_gas": total_profit_excluding_llm_gas,
                "profit_per_completed_auction": safe_div(total_profit, completed_auctions),
                "revenue_per_completed_auction": safe_div(total_revenue, completed_auctions),
                "cost_per_completed_auction": safe_div(total_costs, completed_auctions),
                "llm_cost_per_completed_auction": safe_div(total_llm_costs, completed_auctions),
                "gas_cost_per_completed_auction": safe_div(total_gas_costs, completed_auctions),
                "other_cost_per_completed_auction": safe_div(total_other_costs, completed_auctions),
                "profit_excluding_gas_per_completed_auction": safe_div(total_profit_excluding_gas, completed_auctions),
                "profit_excluding_llm_per_completed_auction": safe_div(total_profit_excluding_llm, completed_auctions),
                "profit_excluding_llm_gas_per_completed_auction": safe_div(total_profit_excluding_llm_gas, completed_auctions),
                "cost_per_bid_attempted": safe_div(total_costs, total_bids_attempted),
                "cost_per_on_chain_bid": safe_div(total_costs, total_bids_on_chain),
                "gas_cost_per_on_chain_bid": safe_div(total_gas_costs, total_bids_on_chain),
                "llm_cost_per_bid_attempted": safe_div(total_llm_costs, total_bids_attempted),
                "efficiency_profit_to_cost": safe_div(total_profit, total_costs),
                "profit_gini": gini(net_balances),
                "profit_fairness_index": 1.0 - gini(net_balances),
                "top1_profit_share": safe_div(max(net_balances) if net_balances else 0.0, sum(max(0.0, v) for v in net_balances)),
            }
        )

        subtype_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in current_provider_run_rows:
            subtype_groups[row.get("provider_subtype_label", "unknown")].append(row)

        total_provider_count = len(current_provider_run_rows)
        total_wins = sum(safe_float(row.get("wins", 0.0)) for row in current_provider_run_rows)
        total_revenue_for_shares = sum(safe_float(row.get("revenue", 0.0)) for row in current_provider_run_rows)

        for subtype_label, rows in sorted(subtype_groups.items(), key=lambda item: item[0]):
            provider_count = len(rows)
            wins_total = sum(safe_float(row.get("wins", 0.0)) for row in rows)
            revenue_total = sum(safe_float(row.get("revenue", 0.0)) for row in rows)
            llm_costs_total = sum(safe_float(row.get("llm_costs", 0.0)) for row in rows)
            gas_costs_total = sum(safe_float(row.get("gas_costs", 0.0)) for row in rows)
            total_costs_total = sum(safe_float(row.get("total_costs", 0.0)) for row in rows)
            net_balance_total = sum(safe_float(row.get("net_balance", 0.0)) for row in rows)
            eligible_total = sum(safe_float(row.get("eligible_auctions", 0.0)) for row in rows)
            participated_total = sum(safe_float(row.get("participated_auctions", 0.0)) for row in rows)
            bids_attempted_total = sum(safe_float(row.get("bids_attempted", 0.0)) for row in rows)
            bids_on_chain_total = sum(safe_float(row.get("bids_on_chain", 0.0)) for row in rows)
            provider_type_run_rows.append(
                {
                    **base,
                    "provider_subtype_label": subtype_label,
                    "provider_group_name": rows[0].get("provider_group_name"),
                    "provider_architecture": rows[0].get("provider_architecture"),
                    "provider_reasoning_mode": rows[0].get("provider_reasoning_mode"),
                    "provider_heuristic_strategy": rows[0].get("provider_heuristic_strategy"),
                    "provider_count": provider_count,
                    "population_share": safe_div(provider_count, total_provider_count),
                    "eligible_auctions_total": int(eligible_total),
                    "participated_auctions_total": int(participated_total),
                    "bids_attempted_total": int(bids_attempted_total),
                    "bids_on_chain_total": int(bids_on_chain_total),
                    "wins_total": int(wins_total),
                    "win_share": safe_div(wins_total, total_wins),
                    "revenue_total": revenue_total,
                    "revenue_share": safe_div(revenue_total, total_revenue_for_shares),
                    "llm_costs_total": llm_costs_total,
                    "gas_costs_total": gas_costs_total,
                    "total_costs_total": total_costs_total,
                    "net_balance_total": net_balance_total,
                    "mean_net_balance": fmean([safe_float(row.get("net_balance", 0.0)) for row in rows]),
                    "mean_revenue": fmean([safe_float(row.get("revenue", 0.0)) for row in rows]),
                    "mean_bid_participation_rate": fmean([safe_float(row.get("bid_participation_rate", 0.0)) for row in rows]),
                    "mean_bid_acceptance_rate": fmean([safe_float(row.get("bid_acceptance_rate", 0.0)) for row in rows]),
                    "mean_win_rate_per_eligible": fmean([safe_float(row.get("win_rate_per_eligible", 0.0)) for row in rows]),
                    "mean_attempted_bid_usdc": fmean([safe_float(row.get("mean_attempted_bid_usdc", 0.0)) for row in rows]),
                    "mean_on_chain_bid_usdc": fmean([safe_float(row.get("mean_on_chain_bid_usdc", 0.0)) for row in rows]),
                }
            )

    group_fields = [f.strip() for f in args.group_by.split(",") if f.strip()]
    metric_fields = [
        "success",
        "duration_seconds",
        "total_auctions",
        "completed_auctions",
        "failed_auctions",
        "total_bids_attempted",
        "total_bids_on_chain",
        "auction_completion_rate",
        "bid_acceptance_rate",
    ]
    group_summary = summarize_groups(run_rows, group_fields, metric_fields)

    grouped_runs: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        grouped_runs[(row["architecture"], row["reasoning_mode"], row["scenario"])].append(row)

    architecture_summary: List[Dict[str, Any]] = []
    for (architecture, reasoning_mode, scenario), rows in sorted(grouped_runs.items(), key=lambda kv: kv[0]):
        success_vals = [safe_float(r.get("success")) for r in rows]
        completion_vals = [safe_float(r.get("auction_completion_rate")) for r in rows]
        acceptance_vals = [safe_float(r.get("bid_acceptance_rate")) for r in rows]
        duration_vals = [safe_float(r.get("duration_seconds")) for r in rows]

        success_stats = compute_stats(success_vals)
        completion_stats = compute_stats(completion_vals)
        acceptance_stats = compute_stats(acceptance_vals)
        duration_stats = compute_stats(duration_vals)

        architecture_summary.append(
            {
                "architecture": architecture,
                "reasoning_mode": reasoning_mode,
                "scenario": scenario,
                "n_runs": len(rows),
                "mean_success": success_stats["mean"],
                "std_success": success_stats["std"],
                "mean_auction_completion_rate": completion_stats["mean"],
                "std_auction_completion_rate": completion_stats["std"],
                "mean_bid_acceptance_rate": acceptance_stats["mean"],
                "std_bid_acceptance_rate": acceptance_stats["std"],
                "mean_duration_seconds": duration_stats["mean"],
                "std_duration_seconds": duration_stats["std"],
            }
        )

    # Write outputs
    out = args.output_dir / args.output_subdir if args.output_subdir else args.output_dir
    write_csv(
        out / "run_summary.csv",
        run_rows,
        [
            "experiment_id",
            "architecture",
            "reasoning_mode",
            "is_heuristic",
            "scenario",
            "seed",
            "provider_mix",
            "provider_subtype_count",
            "metrics_file",
            "success",
            "duration_seconds",
            "total_auctions",
            "completed_auctions",
            "failed_auctions",
            "total_bids_attempted",
            "total_bids_on_chain",
            "auction_completion_rate",
            "bid_acceptance_rate",
        ],
    )

    write_csv(
        out / "auction_outcomes.csv",
        auction_rows,
        [
            "experiment_id",
            "architecture",
            "reasoning_mode",
            "is_heuristic",
            "scenario",
            "seed",
            "metrics_file",
            "auction_id",
            "timestamp_created",
            "timestamp_ended",
            "status",
            "winner_id",
            "winning_bid_amount",
            "winning_bid_usdc",
            "budget",
            "budget_usdc",
            "winning_bid_ratio",
            "bid_count_on_chain",
            "bid_count_attempted",
            "bid_failure_count",
            "eligible_provider_count",
            "winner_group_name",
            "winner_architecture",
            "winner_reasoning_mode",
            "winner_heuristic_strategy",
            "winner_subtype_label",
            "service_executed",
            "quality_rating",
            "feedback_submitted",
            "execution_duration_seconds",
            "evaluation_duration_seconds",
        ],
    )

    write_csv(
        out / "provider_financials_summary.csv",
        provider_summary_rows,
        [
            "experiment_id",
            "architecture",
            "reasoning_mode",
            "is_heuristic",
            "scenario",
            "seed",
            "provider_mix",
            "provider_subtype_count",
            "metrics_file",
            "provider_id",
            "provider_group_name",
            "provider_architecture",
            "provider_reasoning_mode",
            "provider_heuristic_strategy",
            "provider_heuristic_min_margin",
            "provider_heuristic_max_margin",
            "provider_subtype_label",
            "revenue",
            "llm_costs",
            "gas_costs",
            "total_costs",
            "net_balance",
            "snapshots",
        ],
    )

    write_csv(
        out / "provider_run_summary.csv",
        provider_run_rows,
        [
            "experiment_id",
            "architecture",
            "reasoning_mode",
            "is_heuristic",
            "scenario",
            "seed",
            "provider_mix",
            "provider_subtype_count",
            "metrics_file",
            "provider_id",
            "provider_group_name",
            "provider_architecture",
            "provider_reasoning_mode",
            "provider_heuristic_strategy",
            "provider_heuristic_min_margin",
            "provider_heuristic_max_margin",
            "provider_subtype_label",
            "eligible_auctions",
            "participated_auctions",
            "wins",
            "bids_attempted",
            "bids_on_chain",
            "bid_participation_rate",
            "bid_acceptance_rate",
            "win_rate_per_eligible",
            "mean_attempted_bid_usdc",
            "mean_on_chain_bid_usdc",
            "revenue",
            "llm_costs",
            "gas_costs",
            "total_costs",
            "net_balance",
        ],
    )

    write_csv(
        out / "provider_type_run_summary.csv",
        provider_type_run_rows,
        [
            "experiment_id",
            "architecture",
            "reasoning_mode",
            "is_heuristic",
            "scenario",
            "seed",
            "provider_mix",
            "provider_subtype_count",
            "metrics_file",
            "provider_subtype_label",
            "provider_group_name",
            "provider_architecture",
            "provider_reasoning_mode",
            "provider_heuristic_strategy",
            "provider_count",
            "population_share",
            "eligible_auctions_total",
            "participated_auctions_total",
            "bids_attempted_total",
            "bids_on_chain_total",
            "wins_total",
            "win_share",
            "revenue_total",
            "revenue_share",
            "llm_costs_total",
            "gas_costs_total",
            "total_costs_total",
            "net_balance_total",
            "mean_net_balance",
            "mean_revenue",
            "mean_bid_participation_rate",
            "mean_bid_acceptance_rate",
            "mean_win_rate_per_eligible",
            "mean_attempted_bid_usdc",
            "mean_on_chain_bid_usdc",
        ],
    )

    write_csv(
        out / "provider_financials_evolution.csv",
        provider_fin_evolution_rows,
        [
            "experiment_id",
            "architecture",
            "reasoning_mode",
            "is_heuristic",
            "scenario",
            "seed",
            "provider_mix",
            "provider_subtype_count",
            "metrics_file",
            "timestamp",
            "provider_id",
            "provider_group_name",
            "provider_architecture",
            "provider_reasoning_mode",
            "provider_heuristic_strategy",
            "provider_heuristic_min_margin",
            "provider_heuristic_max_margin",
            "provider_subtype_label",
            "revenue",
            "llm_costs",
            "gas_costs",
            "total_costs",
            "net_balance",
        ],
    )

    write_csv(
        out / "reputation_evolution.csv",
        reputation_rows,
        [
            "experiment_id",
            "architecture",
            "reasoning_mode",
            "is_heuristic",
            "scenario",
            "seed",
            "provider_mix",
            "provider_subtype_count",
            "metrics_file",
            "sequence",
            "timestamp",
            "after_auction",
            "provider_id",
            "provider_group_name",
            "provider_architecture",
            "provider_reasoning_mode",
            "provider_heuristic_strategy",
            "provider_heuristic_min_margin",
            "provider_heuristic_max_margin",
            "provider_subtype_label",
            "reputation_score",
            "feedback_count",
            "source",
        ],
    )

    write_csv(
        out / "bid_market_data.csv",
        bid_market_rows,
        [
            "experiment_id",
            "architecture",
            "reasoning_mode",
            "is_heuristic",
            "scenario",
            "seed",
            "provider_mix",
            "provider_subtype_count",
            "metrics_file",
            "auction_id",
            "provider_id",
            "provider_group_name",
            "provider_architecture",
            "provider_reasoning_mode",
            "provider_heuristic_strategy",
            "provider_heuristic_min_margin",
            "provider_heuristic_max_margin",
            "provider_subtype_label",
            "bid_amount",
            "bid_amount_usdc",
            "budget",
            "budget_usdc",
            "normalized_bid",
            "reputation_before_auction",
            "won_auction",
            "has_on_chain_bid",
        ],
    )

    write_csv(
        out / "run_economics.csv",
        run_economics_rows,
        [
            "experiment_id",
            "architecture",
            "reasoning_mode",
            "is_heuristic",
            "scenario",
            "seed",
            "metrics_file",
            "provider_count",
            "system_total_revenue",
            "system_total_llm_costs",
            "system_total_gas_costs",
            "system_total_other_costs",
            "system_total_costs",
            "system_total_profit",
            "system_total_profit_excluding_gas",
            "system_total_profit_excluding_llm",
            "system_total_profit_excluding_llm_gas",
            "profit_per_completed_auction",
            "revenue_per_completed_auction",
            "cost_per_completed_auction",
            "llm_cost_per_completed_auction",
            "gas_cost_per_completed_auction",
            "other_cost_per_completed_auction",
            "profit_excluding_gas_per_completed_auction",
            "profit_excluding_llm_per_completed_auction",
            "profit_excluding_llm_gas_per_completed_auction",
            "cost_per_bid_attempted",
            "cost_per_on_chain_bid",
            "gas_cost_per_on_chain_bid",
            "llm_cost_per_bid_attempted",
            "efficiency_profit_to_cost",
            "profit_gini",
            "profit_fairness_index",
            "top1_profit_share",
        ],
    )

    # Generic multi-run summary (configurable grouping)
    group_fieldnames: List[str] = group_fields + ["n_runs"]
    for metric in metric_fields:
        group_fieldnames.extend(
            [
                f"mean_{metric}",
                f"std_{metric}",
                f"sem_{metric}",
                f"ci95_low_{metric}",
                f"ci95_high_{metric}",
                f"min_{metric}",
                f"max_{metric}",
            ]
        )
    write_csv(out / "group_summary.csv", group_summary, group_fieldnames)

    write_csv(
        out / "architecture_summary.csv",
        architecture_summary,
        [
            "architecture",
            "reasoning_mode",
            "scenario",
            "n_runs",
            "mean_success",
            "std_success",
            "mean_auction_completion_rate",
            "std_auction_completion_rate",
            "mean_bid_acceptance_rate",
            "std_bid_acceptance_rate",
            "mean_duration_seconds",
            "std_duration_seconds",
        ],
    )

    print(f"Aggregated {len(files)} experiment files into {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Run service-executor reliability analysis using ServiceEvaluator as the metric instrument.

Reliability question:
How much does ServiceExecutor output quality vary when the same requirement
is executed repeatedly?

Pipeline:
1. Build a fixed requirement set (default: 10) from PDF abstracts.
2. Execute ServiceExecutor multiple times per requirement.
3. Evaluate each executor output with ServiceEvaluator.
4. Persist raw runs and per-requirement/overall reliability summaries.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import csv
import json
import logging
import os
import random
import re
import shutil
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.config import Config
from agents.consumer_agent.evaluator import ServiceEvaluator
from agents.consumer_agent.service_generator import ServiceGenerator
from agents.provider_agent.service_executor import ServiceExecutor

logger = logging.getLogger("executor_reliability")


DEFAULT_QUALITY_CRITERIA = {
    "completeness": "All prompts must be answered thoroughly",
    "depth": "Answers should be detailed and well-supported with evidence from the paper",
    "citations": "Include relevant citations and specific references to the paper's content",
    "clarity": "Clear, well-structured responses that demonstrate deep understanding",
}


@dataclass
class ExecutorReliabilityConfig:
    analysis_id: str
    source_pdf_dir: str
    file_glob: str
    output_dir: str
    artifacts_dir: str
    workspace_dir: str
    provider_agent_id: str
    architecture: str
    effort_tier: Optional[str]
    requirement_count: int
    executor_repeats_per_requirement: int
    allow_pdf_reuse: bool
    shuffle: bool
    random_seed: int
    force_rebuild_rag: bool
    reset_executor_between_runs: bool
    include_partial_fallback_runs: bool
    evaluation_timeout_seconds: Optional[float]
    heartbeat_interval_seconds: float
    overwrite: bool


class ExecutorReliabilityAnalyzer:
    def __init__(self, cfg: ExecutorReliabilityConfig):
        self.cfg = cfg
        self.runtime_config = Config()

        self.source_pdf_dir = self._abs_path(cfg.source_pdf_dir)
        self.output_root = self._abs_path(cfg.output_dir) / cfg.analysis_id
        self.artifacts_root = self._abs_path(cfg.artifacts_dir) / cfg.analysis_id

        self.raw_runs_path = self.output_root / "raw_runs.jsonl"
        self.requirement_summary_json_path = self.output_root / "requirement_summary.json"
        self.requirement_summary_csv_path = self.output_root / "requirement_summary.csv"
        self.overall_summary_path = self.output_root / "overall_summary.json"

    @staticmethod
    def _abs_path(path_str: str) -> Path:
        path = Path(path_str)
        if path.is_absolute():
            return path
        return REPO_ROOT / path

    @staticmethod
    def _safe_id_fragment(text: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", text.strip())
        slug = slug.strip("_")
        return slug[:60] if slug else "sample"

    @staticmethod
    def _format_duration(seconds: float) -> str:
        total = max(0, int(round(seconds)))
        hours, rem = divmod(total, 3600)
        minutes, secs = divmod(rem, 60)
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        if minutes > 0:
            return f"{minutes}m {secs}s"
        return f"{secs}s"

    @staticmethod
    def _build_requirements(pdf_path: Path, prompts: List[str]) -> Dict[str, Any]:
        return {
            "title": f"Executor Reliability Task: {pdf_path.stem}",
            "description": f"Evaluate reliability for service execution on {pdf_path.stem}",
            "service_type": "literature_review",
            "prompts": prompts,
            "input_file": "input.pdf",
            "quality_criteria": dict(DEFAULT_QUALITY_CRITERIA),
        }

    @staticmethod
    def _write_json(path: Path, payload: Dict[str, Any]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    @staticmethod
    def _numeric_stats(values: List[float]) -> Dict[str, Optional[float]]:
        if not values:
            return {
                "count": 0,
                "mean": None,
                "variance": None,
                "std": None,
                "min": None,
                "max": None,
                "range": None,
            }

        if len(values) == 1:
            v = float(values[0])
            return {
                "count": 1,
                "mean": v,
                "variance": 0.0,
                "std": 0.0,
                "min": v,
                "max": v,
                "range": 0.0,
            }

        mean_v = statistics.mean(values)
        var_v = statistics.pvariance(values)
        std_v = statistics.pstdev(values)
        min_v = min(values)
        max_v = max(values)
        return {
            "count": len(values),
            "mean": float(mean_v),
            "variance": float(var_v),
            "std": float(std_v),
            "min": float(min_v),
            "max": float(max_v),
            "range": float(max_v - min_v),
        }

    def _discover_pdf_paths(self) -> List[Path]:
        if not self.source_pdf_dir.exists():
            raise FileNotFoundError(f"Source PDF directory does not exist: {self.source_pdf_dir}")

        pdf_paths = sorted(self.source_pdf_dir.glob(self.cfg.file_glob))
        pdf_paths = [p for p in pdf_paths if p.is_file() and p.suffix.lower() == ".pdf"]

        with_abstract = []
        skipped_missing_abstract = []
        for pdf in pdf_paths:
            if pdf.with_suffix(".txt").exists():
                with_abstract.append(pdf)
            else:
                skipped_missing_abstract.append(pdf.name)

        if skipped_missing_abstract:
            logger.warning(
                "Skipping %s PDFs with no matching abstract .txt: %s",
                len(skipped_missing_abstract),
                ", ".join(skipped_missing_abstract),
            )

        if not with_abstract:
            return []

        if self.cfg.shuffle:
            rng = random.Random(self.cfg.random_seed)
            rng.shuffle(with_abstract)

        target = self.cfg.requirement_count
        if target <= len(with_abstract):
            return with_abstract[:target]

        if not self.cfg.allow_pdf_reuse:
            logger.warning(
                "Requested %s requirements but only %s eligible PDFs are available. "
                "Set allow_pdf_reuse=true to permit repeated use of the same PDF.",
                target,
                len(with_abstract),
            )
            return with_abstract

        logger.info(
            "Reusing PDFs to build %s requirements from %s unique PDFs",
            target,
            len(with_abstract),
        )

        if self.cfg.shuffle:
            rng = random.Random(self.cfg.random_seed)
            return [rng.choice(with_abstract) for _ in range(target)]

        return [with_abstract[i % len(with_abstract)] for i in range(target)]

    def _prepare_output_dirs(self) -> None:
        if self.output_root.exists() and not self.cfg.overwrite:
            raise FileExistsError(
                f"Output already exists: {self.output_root}. Use overwrite=true or --overwrite."
            )

        if self.cfg.overwrite:
            if self.output_root.exists():
                shutil.rmtree(self.output_root)
            if self.artifacts_root.exists():
                shutil.rmtree(self.artifacts_root)

        self.output_root.mkdir(parents=True, exist_ok=True)
        self.artifacts_root.mkdir(parents=True, exist_ok=True)

    def _prepare_requirement_dir(self, requirement_id: str, pdf_path: Path) -> Path:
        req_dir = self.artifacts_root / requirement_id
        req_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy2(pdf_path, req_dir / "input.pdf")
        abstract_src = pdf_path.with_suffix(".txt")
        if abstract_src.exists():
            shutil.copy2(abstract_src, req_dir / "input.txt")

        return req_dir

    def _ensure_runtime_ready(self, dry_run: bool) -> None:
        if dry_run:
            return

        if not self.runtime_config.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY is required for executor reliability analysis")

    @staticmethod
    async def _evaluate_with_observability(
        evaluator: ServiceEvaluator,
        requirements: Dict[str, Any],
        service_output: Dict[str, Any],
        run_label: str,
        timeout: Optional[float],
        heartbeat_interval_seconds: float,
    ) -> Dict[str, Any]:
        started = time.time()
        heartbeat = max(1.0, float(heartbeat_interval_seconds))

        task = asyncio.create_task(evaluator.evaluate(requirements, service_output))
        while True:
            elapsed = time.time() - started
            remaining: Optional[float] = None
            if timeout is not None:
                remaining = timeout - elapsed
                if remaining <= 0:
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
                    return {
                        "rating": None,
                        "quality_scores": {},
                        "error": f"Timed out after {timeout:.1f}s",
                        "duration_seconds": round(time.time() - started, 3),
                    }

            wait_for = heartbeat if remaining is None else min(heartbeat, remaining)
            done, _ = await asyncio.wait({task}, timeout=wait_for)
            if task in done:
                try:
                    result = task.result()
                except Exception as exc:
                    return {
                        "rating": None,
                        "quality_scores": {},
                        "error": str(exc),
                        "duration_seconds": round(time.time() - started, 3),
                    }

                duration = round(time.time() - started, 3)
                if not isinstance(result, dict):
                    return {
                        "rating": None,
                        "quality_scores": {},
                        "error": "Evaluator returned non-dict result",
                        "duration_seconds": duration,
                    }

                result.setdefault("rating", None)
                result.setdefault("quality_scores", {})
                result.setdefault("error", None)
                result["duration_seconds"] = duration
                return result

            elapsed = time.time() - started
            if timeout is None:
                logger.info("Heartbeat: %s still running (elapsed=%.1fs)", run_label, elapsed)
            else:
                logger.info(
                    "Heartbeat: %s still running (elapsed=%.1fs, timeout=%.1fs)",
                    run_label,
                    elapsed,
                    timeout,
                )

    @staticmethod
    def _flatten_requirement_summary_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        dim_names = set()
        for row in rows:
            dim_names.update(row.get("dimension_stats", {}).keys())

        ordered_dims = sorted(dim_names)
        flat: List[Dict[str, Any]] = []
        for row in rows:
            base = {
                "requirement_id": row.get("requirement_id"),
                "source_pdf": row.get("source_pdf"),
                "runs_total": row.get("runs_total"),
                "runs_executor_success": row.get("runs_executor_success"),
                "runs_evaluator_success": row.get("runs_evaluator_success"),
                "runs_failed": row.get("runs_failed"),
                "partial_fallback_runs": row.get("partial_fallback_runs"),
                "rating_mean": row.get("rating_stats", {}).get("mean"),
                "rating_variance": row.get("rating_stats", {}).get("variance"),
                "rating_std": row.get("rating_stats", {}).get("std"),
                "rating_min": row.get("rating_stats", {}).get("min"),
                "rating_max": row.get("rating_stats", {}).get("max"),
                "rating_range": row.get("rating_stats", {}).get("range"),
                "executor_duration_mean_seconds": row.get("executor_duration_stats", {}).get("mean"),
                "evaluator_duration_mean_seconds": row.get("evaluator_duration_stats", {}).get("mean"),
            }

            dim_stats = row.get("dimension_stats", {})
            for dim in ordered_dims:
                ds = dim_stats.get(dim, {})
                base[f"{dim}_mean"] = ds.get("mean")
                base[f"{dim}_variance"] = ds.get("variance")
                base[f"{dim}_std"] = ds.get("std")
                base[f"{dim}_min"] = ds.get("min")
                base[f"{dim}_max"] = ds.get("max")

            flat.append(base)

        return flat

    @staticmethod
    def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return

        fieldnames = list(rows[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    async def run(self, dry_run: bool = False) -> Dict[str, Any]:
        self._ensure_runtime_ready(dry_run=dry_run)

        pdf_paths = self._discover_pdf_paths()
        if not pdf_paths:
            raise RuntimeError("No eligible PDF files found for executor reliability analysis")

        if dry_run:
            return {
                "analysis_id": self.cfg.analysis_id,
                "source_pdf_dir": str(self.source_pdf_dir),
                "requirement_count": len(pdf_paths),
                "executor_repeats_per_requirement": self.cfg.executor_repeats_per_requirement,
                "planned_total_runs": len(pdf_paths) * self.cfg.executor_repeats_per_requirement,
            }

        self._prepare_output_dirs()

        service_generator = ServiceGenerator(self.runtime_config)
        evaluator = ServiceEvaluator(self.runtime_config)

        service_executor = ServiceExecutor(
            agent_id=self.cfg.provider_agent_id,
            workspace_dir=str(self._abs_path(self.cfg.workspace_dir)),
            architecture=self.cfg.architecture,
        )

        logger.info(
            "Running executor reliability: requirements=%s, repeats=%s",
            len(pdf_paths),
            self.cfg.executor_repeats_per_requirement,
        )

        started = time.time()
        requirements_data: List[Dict[str, Any]] = []

        # Build requirements once so each repeated run gets identical prompts.
        for idx, pdf_path in enumerate(pdf_paths, start=1):
            requirement_id = f"{idx:04d}_{self._safe_id_fragment(pdf_path.stem)}"
            req_dir = self._prepare_requirement_dir(requirement_id, pdf_path)

            abstract = service_generator._read_abstract(pdf_path)
            prompts = await service_generator._generate_prompts_from_abstract(abstract, pdf_path.stem)
            requirements = self._build_requirements(pdf_path, prompts)

            self._write_json(req_dir / "requirements.json", requirements)
            requirements_data.append(
                {
                    "requirement_id": requirement_id,
                    "source_pdf": str(pdf_path.relative_to(REPO_ROOT)),
                    "pdf_path": pdf_path,
                    "req_dir": req_dir,
                    "requirements": requirements,
                }
            )

        total_planned_runs = len(requirements_data) * self.cfg.executor_repeats_per_requirement
        completed_runs = 0

        per_requirement_runs: Dict[str, List[Dict[str, Any]]] = {
            r["requirement_id"]: [] for r in requirements_data
        }

        for req_idx, req in enumerate(requirements_data, start=1):
            requirement_id = req["requirement_id"]
            source_pdf = req["source_pdf"]
            req_dir = req["req_dir"]
            requirements = req["requirements"]
            prompts = requirements["prompts"]

            logger.info(
                "[%s/%s] Running executor loops for %s",
                req_idx,
                len(requirements_data),
                requirement_id,
            )

            for repeat_idx in range(1, self.cfg.executor_repeats_per_requirement + 1):
                run_label = (
                    f"requirement {requirement_id} [{req_idx}/{len(requirements_data)}], "
                    f"repeat {repeat_idx}/{self.cfg.executor_repeats_per_requirement}"
                )
                logger.info("Starting %s", run_label)

                if self.cfg.reset_executor_between_runs:
                    service_executor = ServiceExecutor(
                        agent_id=self.cfg.provider_agent_id,
                        workspace_dir=str(self._abs_path(self.cfg.workspace_dir)),
                        architecture=self.cfg.architecture,
                    )

                exec_started = time.time()
                exec_result = service_executor.perform_analysis(
                    pdf_directory=str(req_dir),
                    prompts=prompts,
                    force_rebuild=self.cfg.force_rebuild_rag,
                    effort_tier=self.cfg.effort_tier,
                )
                exec_duration = round(time.time() - exec_started, 3)

                evaluator_payload: Dict[str, Any]
                if not exec_result.get("success", False) or not exec_result.get("responses"):
                    evaluator_payload = {
                        "rating": None,
                        "quality_scores": {},
                        "error": "Skipped evaluator due to failed/empty executor output",
                        "duration_seconds": 0.0,
                    }
                else:
                    evaluator_payload = await self._evaluate_with_observability(
                        evaluator=evaluator,
                        requirements=requirements,
                        service_output=exec_result,
                        run_label=run_label,
                        timeout=self.cfg.evaluation_timeout_seconds,
                        heartbeat_interval_seconds=self.cfg.heartbeat_interval_seconds,
                    )

                run_record = {
                    "analysis_id": self.cfg.analysis_id,
                    "requirement_id": requirement_id,
                    "source_pdf": source_pdf,
                    "repeat_idx": repeat_idx,
                    "executor_success": bool(exec_result.get("success", False)),
                    "executor_error": exec_result.get("error"),
                    "partial_fallback_used": bool(exec_result.get("partial_fallback_used", False)),
                    "response_count": len(exec_result.get("responses", [])),
                    "executor_duration_seconds": exec_duration,
                    "evaluator_rating": evaluator_payload.get("rating"),
                    "evaluator_quality_scores": evaluator_payload.get("quality_scores", {}),
                    "evaluator_error": evaluator_payload.get("error"),
                    "evaluator_duration_seconds": float(evaluator_payload.get("duration_seconds", 0.0)),
                }

                if run_record["partial_fallback_used"] and not self.cfg.include_partial_fallback_runs:
                    run_record["evaluator_error"] = (
                        run_record["evaluator_error"] or "Excluded: partial fallback run"
                    )
                    run_record["evaluator_rating"] = None
                    run_record["evaluator_quality_scores"] = {}

                per_requirement_runs[requirement_id].append(run_record)
                self._append_jsonl(self.raw_runs_path, run_record)

                completed_runs += 1
                avg_run_seconds = (time.time() - started) / completed_runs
                remaining_runs = total_planned_runs - completed_runs
                eta_seconds = remaining_runs * avg_run_seconds

                logger.info(
                    "Finished %s in %.1fs (executor_success=%s, rating=%s, evaluator_error=%s)",
                    run_label,
                    exec_duration + run_record["evaluator_duration_seconds"],
                    run_record["executor_success"],
                    run_record["evaluator_rating"],
                    run_record["evaluator_error"],
                )
                logger.info(
                    "Progress: %s/%s runs (%.1f%%), avg %.1fs/run, ETA ~%s",
                    completed_runs,
                    total_planned_runs,
                    (completed_runs / total_planned_runs) * 100.0,
                    avg_run_seconds,
                    self._format_duration(eta_seconds),
                )

        requirement_summaries: List[Dict[str, Any]] = []
        all_rating_values: List[float] = []
        within_requirement_vars: List[float] = []
        within_requirement_stds: List[float] = []
        total_executor_success_runs = 0
        total_evaluator_success_runs = 0
        total_failed_runs = 0
        total_fallback_runs = 0

        for req in requirements_data:
            requirement_id = req["requirement_id"]
            source_pdf = req["source_pdf"]
            runs = per_requirement_runs[requirement_id]

            executor_success_runs = [r for r in runs if r.get("executor_success")]
            evaluator_success_runs = [
                r
                for r in runs
                if r.get("evaluator_error") is None and r.get("evaluator_rating") is not None
            ]
            failed_runs = [r for r in runs if r.get("evaluator_error") is not None]
            fallback_runs = [r for r in runs if r.get("partial_fallback_used")]

            ratings = [float(r["evaluator_rating"]) for r in evaluator_success_runs]
            exec_durations = [float(r["executor_duration_seconds"]) for r in runs]
            eval_durations = [float(r["evaluator_duration_seconds"]) for r in runs]

            rating_stats = self._numeric_stats(ratings)
            exec_duration_stats = self._numeric_stats(exec_durations)
            eval_duration_stats = self._numeric_stats(eval_durations)

            if rating_stats["variance"] is not None:
                within_requirement_vars.append(float(rating_stats["variance"]))
            if rating_stats["std"] is not None:
                within_requirement_stds.append(float(rating_stats["std"]))
            all_rating_values.extend(ratings)

            dim_values: Dict[str, List[float]] = {}
            for r in evaluator_success_runs:
                for dim, score in (r.get("evaluator_quality_scores") or {}).items():
                    if isinstance(score, (int, float)):
                        dim_values.setdefault(dim, []).append(float(score))
            dimension_stats = {dim: self._numeric_stats(vals) for dim, vals in sorted(dim_values.items())}

            requirement_summaries.append(
                {
                    "requirement_id": requirement_id,
                    "source_pdf": source_pdf,
                    "runs_total": len(runs),
                    "runs_executor_success": len(executor_success_runs),
                    "runs_evaluator_success": len(evaluator_success_runs),
                    "runs_failed": len(failed_runs),
                    "partial_fallback_runs": len(fallback_runs),
                    "rating_stats": rating_stats,
                    "executor_duration_stats": exec_duration_stats,
                    "evaluator_duration_stats": eval_duration_stats,
                    "dimension_stats": dimension_stats,
                }
            )

            total_executor_success_runs += len(executor_success_runs)
            total_evaluator_success_runs += len(evaluator_success_runs)
            total_failed_runs += len(failed_runs)
            total_fallback_runs += len(fallback_runs)

        overall = {
            "analysis_id": self.cfg.analysis_id,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source_pdf_dir": str(self.source_pdf_dir),
            "requirement_count": len(requirements_data),
            "executor_repeats_per_requirement": self.cfg.executor_repeats_per_requirement,
            "total_runs": total_planned_runs,
            "executor_successful_runs": total_executor_success_runs,
            "evaluator_successful_runs": total_evaluator_success_runs,
            "failed_runs": total_failed_runs,
            "partial_fallback_runs": total_fallback_runs,
            "overall_rating_stats": self._numeric_stats(all_rating_values),
            "within_requirement_variance_stats": self._numeric_stats(within_requirement_vars),
            "within_requirement_std_stats": self._numeric_stats(within_requirement_stds),
            "duration_seconds_total": round(time.time() - started, 3),
            "config": {
                "architecture": self.cfg.architecture,
                "effort_tier": self.cfg.effort_tier,
                "requirement_count": self.cfg.requirement_count,
                "executor_repeats_per_requirement": self.cfg.executor_repeats_per_requirement,
                "allow_pdf_reuse": self.cfg.allow_pdf_reuse,
                "shuffle": self.cfg.shuffle,
                "random_seed": self.cfg.random_seed,
                "force_rebuild_rag": self.cfg.force_rebuild_rag,
                "reset_executor_between_runs": self.cfg.reset_executor_between_runs,
                "include_partial_fallback_runs": self.cfg.include_partial_fallback_runs,
                "evaluation_timeout_seconds": self.cfg.evaluation_timeout_seconds,
                "heartbeat_interval_seconds": self.cfg.heartbeat_interval_seconds,
            },
            "output_files": {
                "raw_runs_jsonl": str(self.raw_runs_path),
                "requirement_summary_json": str(self.requirement_summary_json_path),
                "requirement_summary_csv": str(self.requirement_summary_csv_path),
                "overall_summary_json": str(self.overall_summary_path),
            },
        }

        unstable = sorted(
            requirement_summaries,
            key=lambda s: (s.get("rating_stats", {}).get("std") or 0.0),
            reverse=True,
        )
        overall["top_unstable_requirements"] = [
            {
                "requirement_id": s["requirement_id"],
                "source_pdf": s.get("source_pdf"),
                "rating_std": s.get("rating_stats", {}).get("std"),
                "rating_variance": s.get("rating_stats", {}).get("variance"),
                "rating_mean": s.get("rating_stats", {}).get("mean"),
            }
            for s in unstable[: min(10, len(unstable))]
        ]

        with open(self.requirement_summary_json_path, "w", encoding="utf-8") as f:
            json.dump(requirement_summaries, f, indent=2, ensure_ascii=False)

        with open(self.overall_summary_path, "w", encoding="utf-8") as f:
            json.dump(overall, f, indent=2, ensure_ascii=False)

        flat_rows = self._flatten_requirement_summary_rows(requirement_summaries)
        self._write_csv(self.requirement_summary_csv_path, flat_rows)

        logger.info("Executor reliability analysis complete: %s", json.dumps(overall, indent=2))
        return overall


def _expand_env_vars(raw_text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        var_name = match.group(1)
        value = os.getenv(var_name)
        if value is None:
            raise ValueError(f"Environment variable {var_name} referenced in config but not set")
        return value

    return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", repl, raw_text)


def load_config(path: Path) -> ExecutorReliabilityConfig:
    raw = path.read_text(encoding="utf-8")
    expanded = _expand_env_vars(raw)
    data = yaml.safe_load(expanded) or {}

    section = data.get("executor_reliability_analysis", {})
    if not section:
        raise ValueError("Missing 'executor_reliability_analysis' section in config")

    requirement_count = int(section.get("requirement_count", 10))
    if requirement_count < 1:
        raise ValueError("requirement_count must be >= 1")

    repeats = int(section.get("executor_repeats_per_requirement", 5))
    if repeats < 1:
        raise ValueError("executor_repeats_per_requirement must be >= 1")

    heartbeat_interval_seconds = float(section.get("heartbeat_interval_seconds", 30))
    if heartbeat_interval_seconds <= 0:
        raise ValueError("heartbeat_interval_seconds must be > 0")

    timeout_raw = section.get("evaluation_timeout_seconds")
    evaluation_timeout_seconds: Optional[float]
    if timeout_raw is None:
        evaluation_timeout_seconds = None
    else:
        evaluation_timeout_seconds = float(timeout_raw)
        if evaluation_timeout_seconds <= 0:
            raise ValueError("evaluation_timeout_seconds must be > 0 when provided")

    return ExecutorReliabilityConfig(
        analysis_id=section.get("analysis_id", "executor_reliability_v1"),
        source_pdf_dir=section.get("source_pdf_dir", "utils/files"),
        file_glob=section.get("file_glob", "*.pdf"),
        output_dir=section.get("output_dir", "experiments/executor_analysis/data/reliability"),
        artifacts_dir=section.get("artifacts_dir", "experiments/executor_analysis/data/artifacts"),
        workspace_dir=section.get("workspace_dir", "workspaces/executor_analysis"),
        provider_agent_id=str(section.get("provider_agent_id", "executor-reliability-provider")),
        architecture=str(section.get("architecture", "1")),
        effort_tier=section.get("effort_tier"),
        requirement_count=requirement_count,
        executor_repeats_per_requirement=repeats,
        allow_pdf_reuse=bool(section.get("allow_pdf_reuse", True)),
        shuffle=bool(section.get("shuffle", False)),
        random_seed=int(section.get("random_seed", 42)),
        force_rebuild_rag=bool(section.get("force_rebuild_rag", True)),
        reset_executor_between_runs=bool(section.get("reset_executor_between_runs", False)),
        include_partial_fallback_runs=bool(section.get("include_partial_fallback_runs", True)),
        evaluation_timeout_seconds=evaluation_timeout_seconds,
        heartbeat_interval_seconds=heartbeat_interval_seconds,
        overwrite=bool(section.get("overwrite", False)),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run executor reliability analysis (executor repeated runs + evaluator scoring)"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "experiments/executor_analysis/configs/executor_reliability_analysis.yaml",
        help="Path to executor reliability YAML config",
    )
    parser.add_argument(
        "--requirements",
        type=int,
        default=None,
        help="Optional runtime override for requirement_count",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=None,
        help="Optional runtime override for executor repeats per requirement",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite previous outputs for this analysis_id",
    )
    parser.add_argument(
        "--evaluation-timeout-seconds",
        type=float,
        default=None,
        help="Optional timeout per evaluator call; timed-out calls are recorded as failed",
    )
    parser.add_argument(
        "--heartbeat-interval-seconds",
        type=float,
        default=None,
        help="Heartbeat log interval while waiting for evaluator calls",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only validate input/config and print planned run size",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


async def _async_main() -> int:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    cfg = load_config(args.config)
    if args.requirements is not None:
        if args.requirements < 1:
            raise ValueError("--requirements must be >= 1")
        cfg.requirement_count = int(args.requirements)
    if args.repeats is not None:
        if args.repeats < 1:
            raise ValueError("--repeats must be >= 1")
        cfg.executor_repeats_per_requirement = int(args.repeats)
    if args.overwrite:
        cfg.overwrite = True
    if args.evaluation_timeout_seconds is not None:
        if args.evaluation_timeout_seconds <= 0:
            raise ValueError("--evaluation-timeout-seconds must be > 0")
        cfg.evaluation_timeout_seconds = float(args.evaluation_timeout_seconds)
    if args.heartbeat_interval_seconds is not None:
        if args.heartbeat_interval_seconds <= 0:
            raise ValueError("--heartbeat-interval-seconds must be > 0")
        cfg.heartbeat_interval_seconds = float(args.heartbeat_interval_seconds)

    analyzer = ExecutorReliabilityAnalyzer(cfg)
    result = await analyzer.run(dry_run=args.dry_run)

    if args.dry_run:
        logger.info("Dry-run plan: %s", json.dumps(result, indent=2))

    return 0


def main() -> int:
    try:
        return asyncio.run(_async_main())
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return 130
    except Exception as exc:
        logger.error("Executor reliability analysis failed: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

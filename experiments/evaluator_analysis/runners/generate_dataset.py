#!/usr/bin/env python3
"""
Generate evaluator-analysis datasets offline using production agent logic.

This runner reproduces the relevant runtime path without marketplace plumbing:
1. Consumer-side requirement generation from PDF abstracts (ServiceGenerator).
2. Provider-side service execution on local job folders (ServiceExecutor).
3. Persist per-sample artifacts + consolidated dataset records.

No blockchain, IPFS pinning, or inter-agent communication is used.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Ensure repo root is importable when running this file directly.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.config import Config
from agents.consumer_agent.service_generator import ServiceGenerator
from agents.provider_agent.service_executor import ServiceExecutor

logger = logging.getLogger("evaluator_dataset")


DEFAULT_QUALITY_CRITERIA = {
    "completeness": "All prompts must be answered thoroughly",
    "depth": "Answers should be detailed and well-supported with evidence from the paper",
    "citations": "Include relevant citations and specific references to the paper's content",
    "clarity": "Clear, well-structured responses that demonstrate deep understanding",
}


@dataclass
class DatasetGenerationConfig:
    dataset_id: str
    source_pdf_dir: str
    file_glob: str
    output_dataset_dir: str
    output_artifacts_dir: str
    workspace_dir: str
    provider_agent_id: str
    architecture: str
    effort_tier: Optional[str]
    max_samples: Optional[int]
    allow_pdf_reuse: bool
    shuffle: bool
    random_seed: int
    force_rebuild_rag: bool
    overwrite: bool


class EvaluatorDatasetGenerator:
    def __init__(self, cfg: DatasetGenerationConfig):
        self.cfg = cfg
        self.runtime_config = Config()

        if not self.runtime_config.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY is required to generate prompts and service outputs")

        self.service_generator = ServiceGenerator(self.runtime_config)
        self.service_executor = ServiceExecutor(
            agent_id=cfg.provider_agent_id,
            workspace_dir=str(self._abs_path(cfg.workspace_dir)),
            architecture=cfg.architecture,
        )

        self.source_pdf_dir = self._abs_path(cfg.source_pdf_dir)
        self.dataset_dir = self._abs_path(cfg.output_dataset_dir)
        self.artifacts_root = self._abs_path(cfg.output_artifacts_dir) / cfg.dataset_id

        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_root.mkdir(parents=True, exist_ok=True)

        self.dataset_jsonl_path = self.dataset_dir / f"{cfg.dataset_id}.jsonl"
        self.manifest_path = self.dataset_dir / f"{cfg.dataset_id}.manifest.json"

    @staticmethod
    def _abs_path(path_str: str) -> Path:
        path = Path(path_str)
        if path.is_absolute():
            return path
        return REPO_ROOT / path

    def _discover_pdf_paths(self) -> List[Path]:
        if not self.source_pdf_dir.exists():
            raise FileNotFoundError(f"Source PDF directory does not exist: {self.source_pdf_dir}")

        pdf_paths = sorted(self.source_pdf_dir.glob(self.cfg.file_glob))
        pdf_paths = [p for p in pdf_paths if p.is_file() and p.suffix.lower() == ".pdf"]

        # Keep only PDFs that have matching abstract .txt files so requirement generation
        # follows the same path as consumer service generation.
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

        if self.cfg.max_samples is None:
            return with_abstract

        target = self.cfg.max_samples
        if target <= len(with_abstract):
            return with_abstract[:target]

        if not self.cfg.allow_pdf_reuse:
            logger.warning(
                "Requested %s samples but only %s eligible PDFs are available. "
                "Set allow_pdf_reuse=true to permit repeated use of the same PDF.",
                target,
                len(with_abstract),
            )
            return with_abstract

        logger.info(
            "Reusing PDFs to reach %s samples from %s unique PDFs",
            target,
            len(with_abstract),
        )

        if self.cfg.shuffle:
            rng = random.Random(self.cfg.random_seed)
            return [rng.choice(with_abstract) for _ in range(target)]

        # Deterministic round-robin reuse when shuffle is disabled.
        return [with_abstract[i % len(with_abstract)] for i in range(target)]

    @staticmethod
    def _safe_id_fragment(text: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", text.strip())
        slug = slug.strip("_")
        return slug[:60] if slug else "sample"

    def _build_requirements(self, pdf_path: Path, prompts: List[str]) -> Dict[str, Any]:
        return {
            "title": f"Literature Review: {pdf_path.stem}",
            "description": f"Comprehensive literature review and analysis of {pdf_path.stem}",
            "service_type": "literature_review",
            "prompts": prompts,
            # Kept as local trace metadata for offline experiments.
            "input_file": "input.pdf",
            "quality_criteria": dict(DEFAULT_QUALITY_CRITERIA),
        }

    async def _generate_prompts(self, pdf_path: Path) -> List[str]:
        abstract = self.service_generator._read_abstract(pdf_path)
        prompts = await self.service_generator._generate_prompts_from_abstract(abstract, pdf_path.stem)
        return prompts

    def _prepare_sample_dir(self, sample_id: str, pdf_path: Path) -> Path:
        sample_dir = self.artifacts_root / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        # Reproduce provider job input layout from runtime flow.
        dst_pdf = sample_dir / "input.pdf"
        shutil.copy2(pdf_path, dst_pdf)

        # Keep abstract for reproducibility/debugging.
        abstract_src = pdf_path.with_suffix(".txt")
        if abstract_src.exists():
            shutil.copy2(abstract_src, sample_dir / "input.txt")

        return sample_dir

    @staticmethod
    def _write_json(path: Path, payload: Dict[str, Any]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    async def run(self) -> Dict[str, Any]:
        if self.dataset_jsonl_path.exists() and not self.cfg.overwrite:
            raise FileExistsError(
                f"Dataset file already exists: {self.dataset_jsonl_path}. Use overwrite=true or a new dataset id."
            )

        if self.cfg.overwrite:
            if self.dataset_jsonl_path.exists():
                self.dataset_jsonl_path.unlink()
            if self.manifest_path.exists():
                self.manifest_path.unlink()
            if self.artifacts_root.exists():
                shutil.rmtree(self.artifacts_root)
                self.artifacts_root.mkdir(parents=True, exist_ok=True)

        pdf_paths = self._discover_pdf_paths()
        if not pdf_paths:
            raise RuntimeError("No eligible PDF files found for dataset generation")

        logger.info("Generating dataset '%s' from %s PDFs", self.cfg.dataset_id, len(pdf_paths))

        started = time.time()
        generated = 0
        failed = 0
        records: List[Dict[str, Any]] = []

        for idx, pdf_path in enumerate(pdf_paths, start=1):
            sample_id = f"{idx:04d}_{self._safe_id_fragment(pdf_path.stem)}"
            logger.info("[%s/%s] Building sample %s from %s", idx, len(pdf_paths), sample_id, pdf_path.name)

            sample_record: Dict[str, Any] = {
                "sample_id": sample_id,
                "source_pdf": str(pdf_path.relative_to(REPO_ROOT)),
                "success": False,
                "error": None,
            }

            sample_started = time.time()
            try:
                sample_dir = self._prepare_sample_dir(sample_id, pdf_path)
                prompts = await self._generate_prompts(pdf_path)
                requirements = self._build_requirements(pdf_path, prompts)

                self._write_json(sample_dir / "requirements.json", requirements)

                result = self.service_executor.perform_analysis(
                    pdf_directory=str(sample_dir),
                    prompts=prompts,
                    force_rebuild=self.cfg.force_rebuild_rag,
                    effort_tier=self.cfg.effort_tier,
                )
                self._write_json(sample_dir / "result.json", result)

                sample_record.update(
                    {
                        "success": bool(result.get("success", False)),
                        "requirements": requirements,
                        "result": result,
                        "prompt_count": len(prompts),
                        "response_count": len(result.get("responses", [])),
                        "sample_dir": str(sample_dir.relative_to(REPO_ROOT)),
                        "duration_seconds": round(time.time() - sample_started, 3),
                    }
                )

                if sample_record["success"]:
                    generated += 1
                else:
                    failed += 1
                    sample_record["error"] = result.get("error") or "Service execution returned success=false"

            except Exception as exc:
                failed += 1
                sample_record["error"] = str(exc)
                sample_record["duration_seconds"] = round(time.time() - sample_started, 3)
                logger.exception("Sample %s failed: %s", sample_id, exc)

            records.append(sample_record)
            self._append_jsonl(self.dataset_jsonl_path, sample_record)

        manifest = {
            "dataset_id": self.cfg.dataset_id,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source_pdf_dir": str(self.source_pdf_dir),
            "sample_count": len(records),
            "successful_samples": generated,
            "failed_samples": failed,
            "duration_seconds": round(time.time() - started, 3),
            "dataset_jsonl": str(self.dataset_jsonl_path),
            "artifacts_dir": str(self.artifacts_root),
            "config": {
                "architecture": self.cfg.architecture,
                "effort_tier": self.cfg.effort_tier,
                "force_rebuild_rag": self.cfg.force_rebuild_rag,
                "allow_pdf_reuse": self.cfg.allow_pdf_reuse,
                "shuffle": self.cfg.shuffle,
                "random_seed": self.cfg.random_seed,
                "max_samples": self.cfg.max_samples,
            },
        }

        self._write_json(self.manifest_path, manifest)
        logger.info("Dataset generation complete: %s", json.dumps(manifest, indent=2))
        return manifest


def _expand_env_vars(raw_text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        var_name = match.group(1)
        value = os.getenv(var_name)
        if value is None:
            raise ValueError(f"Environment variable {var_name} referenced in config but not set")
        return value

    return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", repl, raw_text)


def load_config(config_path: Path) -> DatasetGenerationConfig:
    raw = config_path.read_text(encoding="utf-8")
    expanded = _expand_env_vars(raw)
    data = yaml.safe_load(expanded) or {}

    section = data.get("dataset_generation", {})
    if not section:
        raise ValueError("Missing 'dataset_generation' section in config")

    return DatasetGenerationConfig(
        dataset_id=section.get("dataset_id", "service_outputs_v1"),
        source_pdf_dir=section.get("source_pdf_dir", "utils/files"),
        file_glob=section.get("file_glob", "*.pdf"),
        output_dataset_dir=section.get("output_dataset_dir", "experiments/evaluator_analysis/data/datasets"),
        output_artifacts_dir=section.get("output_artifacts_dir", "experiments/evaluator_analysis/data/artifacts"),
        workspace_dir=section.get("workspace_dir", "workspaces/evaluator_analysis"),
        provider_agent_id=str(section.get("provider_agent_id", "evaluator-dataset-provider")),
        architecture=str(section.get("architecture", "1")),
        effort_tier=section.get("effort_tier"),
        max_samples=section.get("max_samples"),
        allow_pdf_reuse=bool(section.get("allow_pdf_reuse", True)),
        shuffle=bool(section.get("shuffle", False)),
        random_seed=int(section.get("random_seed", 42)),
        force_rebuild_rag=bool(section.get("force_rebuild_rag", True)),
        overwrite=bool(section.get("overwrite", False)),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate offline evaluator-analysis dataset")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "experiments/evaluator_analysis/configs/dataset_generation.yaml",
        help="Path to dataset generation YAML config",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional runtime override for max sample count",
    )
    parser.add_argument(
        "--no-pdf-reuse",
        action="store_true",
        help="Do not reuse PDFs when max-samples exceeds available unique files",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing dataset artifacts for the configured dataset_id",
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
    if args.max_samples is not None:
        cfg.max_samples = args.max_samples
    if args.no_pdf_reuse:
        cfg.allow_pdf_reuse = False
    if args.overwrite:
        cfg.overwrite = True

    generator = EvaluatorDatasetGenerator(cfg)
    await generator.run()
    return 0


def main() -> int:
    try:
        return asyncio.run(_async_main())
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return 130
    except Exception as exc:
        logger.error("Dataset generation failed: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

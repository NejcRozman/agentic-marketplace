"""
Service Evaluator - evaluates completed service quality via iterative LLM calls.

Per-pair criteria (depth, clarity, …) are scored in parallel, one LLM call per
response.  Holistic criteria (completeness, …) require seeing all responses at
once and are scored in a single separate call.  Results are merged and averaged
to produce a 0-100 overall rating.
"""

import asyncio
import ast
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from langchain_openai import ChatOpenAI

try:
    from ..config import Config
except ImportError:
    from config import Config

logger = logging.getLogger(__name__)

# Criteria whose value can only be judged by looking at the full response set.
HOLISTIC_CRITERIA = {"completeness"}

# Retry settings for transient LLM API failures.
_MAX_RETRIES = 3
_RETRY_BACKOFF = 2.0  # seconds; doubled on each attempt


class ServiceEvaluator:

    def __init__(self, config: Config):
        self.config = config
        self._model = getattr(config, "openrouter_model", "openai/gpt-oss-20b")
        self.model = ChatOpenAI(
            model=self._model,
            api_key=config.openrouter_api_key,
            base_url=config.openrouter_base_url,
            temperature=0.2,
            max_tokens=1000,
        )
        logger.info("ServiceEvaluator initialized")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def evaluate(
        self,
        service_requirements: Dict[str, Any],
        result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Evaluate a completed service result.

        Per-pair criteria are scored in parallel (one LLM call per response).
        Holistic criteria are scored in one call over the full response set.
        Returns rating (0-100), per-dimension quality_scores, explanations, and error.
        """
        logger.info("Starting evaluation...")

        quality_criteria: Dict[str, str] = service_requirements.get("quality_criteria", {})
        required_prompts: List[str] = service_requirements.get("prompts", [])
        responses: List[Dict[str, str]] = result.get("responses", [])

        if not responses or not quality_criteria:
            logger.warning("No responses or quality criteria found; returning neutral scores.")
            return {"rating": 0, "quality_scores": {}, "explanations": [], "error": "No responses or quality criteria found."}

        per_pair_criteria = {k: v for k, v in quality_criteria.items() if k not in HOLISTIC_CRITERIA}
        holistic_criteria = {k: v for k, v in quality_criteria.items() if k in HOLISTIC_CRITERIA}

        # --- parallel per-pair scoring ---
        pair_tasks = [
            self._score_pair(pair, per_pair_criteria)
            for pair in responses
        ]
        pair_results: List[Optional[Dict]] = await asyncio.gather(*pair_tasks)

        # --- holistic scoring (single call, all responses) ---
        holistic_scores: Dict[str, int] = {}
        holistic_explanation: str = ""
        if holistic_criteria:
            holistic_scores, holistic_explanation = await self._score_holistic(
                responses, holistic_criteria
            )

        # If holistic parsing fails, avoid punishing completeness as zero.
        for dim in holistic_criteria:
            if dim in holistic_scores:
                continue
            if dim == "completeness":
                fallback = self._completeness_cap(required_prompts, responses)
                holistic_scores[dim] = fallback
                logger.warning(
                    "Holistic '%s' score missing; using prompt coverage fallback: %s",
                    dim,
                    fallback,
                )
            else:
                holistic_scores[dim] = 0
                logger.warning("Holistic '%s' score missing; defaulting to 0.", dim)

        # Prevent inflated completeness when required prompts were not all answered.
        if "completeness" in holistic_scores:
            prompt_coverage_cap = self._completeness_cap(required_prompts, responses)
            holistic_scores["completeness"] = min(holistic_scores["completeness"], prompt_coverage_cap)

        # Aggregate per-pair scores: average across pairs, skipping failed ones
        per_pair_dim_scores: Dict[str, List[int]] = {dim: [] for dim in per_pair_criteria}
        explanations: List[str] = []
        for r in pair_results:
            if r is None:
                continue
            for dim in per_pair_criteria:
                if dim in r.get("scores", {}):
                    per_pair_dim_scores[dim].append(r["scores"][dim])
            if r.get("explanation"):
                explanations.append(r["explanation"])

        agg_scores: Dict[str, int] = {}
        for dim, vals in per_pair_dim_scores.items():
            agg_scores[dim] = round(sum(vals) / len(vals)) if vals else 0

        # Merge holistic scores (they override per-pair for their dimensions)
        agg_scores.update(holistic_scores)
        if holistic_explanation:
            explanations.append(holistic_explanation)

        overall = max(0, min(100, round(sum(agg_scores.values()) / len(agg_scores)))) if agg_scores else 0

        logger.info(f"Aggregated scores: {agg_scores}, overall: {overall}")
        return {"rating": overall, "quality_scores": agg_scores, "explanations": explanations, "error": None}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _score_pair(
        self,
        pair: Dict[str, str],
        criteria: Dict[str, str],
    ) -> Optional[Dict]:
        """Score a single prompt-response pair. Returns None on unrecoverable failure."""
        if not criteria:
            return None

        prompt_text = pair.get("prompt", "")
        response_text = pair.get("response", "")
        criteria_str = "\n".join(f"- {dim}: {desc}" for dim, desc in criteria.items())
        dim_keys = list(criteria.keys())
        score_schema = ", ".join(f'"{d}": <int>' for d in dim_keys)

        eval_prompt = (
            "You are an expert evaluator. Score the following response for each quality dimension (0-100) "
            "and explain your reasoning.\n\n"
            f"Prompt: {prompt_text}\n"
            f"Response: {response_text}\n\n"
            f"Quality Criteria:\n{criteria_str}\n\n"
            f"Return JSON only: {{\"scores\": {{{score_schema}}}, \"explanation\": \"<str>\"}}"
        )

        parsed = await self._llm_call_with_retry(eval_prompt, label=f"pair '{prompt_text[:40]}'")
        if parsed is None or "scores" not in parsed:
            logger.warning(f"No valid scores returned for pair: {prompt_text[:60]}")
            return None

        normalized_scores: Dict[str, int] = {}
        for dim in criteria:
            if dim in parsed.get("scores", {}):
                coerced = self._coerce_score(parsed["scores"][dim])
                if coerced is not None:
                    normalized_scores[dim] = coerced

        if not normalized_scores:
            logger.warning(f"No valid numeric scores returned for pair: {prompt_text[:60]}")
            return None

        return {"scores": normalized_scores, "explanation": parsed.get("explanation", "")}

    async def _score_holistic(
        self,
        responses: List[Dict[str, str]],
        criteria: Dict[str, str],
    ) -> Tuple[Dict[str, int], str]:
        """Score holistic criteria (e.g. completeness) over the full response set."""
        responses_str = "\n\n".join(
            f"[{i+1}] Prompt: {r.get('prompt', '')}\n    Response: {r.get('response', '')}"
            for i, r in enumerate(responses)
        )
        criteria_str = "\n".join(f"- {dim}: {desc}" for dim, desc in criteria.items())
        dim_keys = list(criteria.keys())
        score_schema = ", ".join(f'"{d}": <int>' for d in dim_keys)

        eval_prompt = (
            "You are an expert evaluator. Given the full set of prompt-response pairs below, "
            "score the overall quality for each dimension (0-100) and explain your reasoning.\n\n"
            f"Responses:\n{responses_str}\n\n"
            f"Quality Criteria:\n{criteria_str}\n\n"
            f"Return JSON only: {{\"scores\": {{{score_schema}}}, \"explanation\": \"<str>\"}}"
        )

        parsed = await self._llm_call_with_retry(eval_prompt, label="holistic")
        if parsed is None or "scores" not in parsed:
            logger.warning("No valid holistic scores returned.")
            return {}, ""

        scores: Dict[str, int] = {}
        for dim in criteria:
            if dim in parsed.get("scores", {}):
                coerced = self._coerce_score(parsed["scores"][dim])
                if coerced is not None:
                    scores[dim] = coerced

        return scores, parsed.get("explanation", "")

    async def _llm_call_with_retry(
        self,
        user_prompt: str,
        label: str = "",
    ) -> Optional[Dict]:
        """
        Call the LLM API with up to _MAX_RETRIES attempts on transient failure.
        Returns the parsed JSON dict or None if all attempts fail.
        """
        llm_prompt = (
            "You are a strict, fair, and concise evaluator. Respond with JSON only.\n\n"
            f"{user_prompt}"
        )

        last_exc: Optional[Exception] = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = await self.model.ainvoke(llm_prompt)
                content = response.content if isinstance(response.content, str) else str(response.content)
                parsed = self._parse_llm_json(content)
                if parsed is not None:
                    return parsed

                last_exc = ValueError("LLM response did not contain parseable JSON object")
                preview = content.strip().replace("\n", " ")[:220]
                if attempt < _MAX_RETRIES:
                    wait = _RETRY_BACKOFF * (2 ** (attempt - 1))
                    logger.warning(
                        "LLM call returned unparseable JSON for %s (attempt %s/%s). "
                        "Preview: %r. Retrying in %ss...",
                        label,
                        attempt,
                        _MAX_RETRIES,
                        preview,
                        wait,
                    )
                    await asyncio.sleep(wait)
                continue
            except Exception as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES:
                    wait = _RETRY_BACKOFF * (2 ** (attempt - 1))
                    logger.warning(f"LLM call failed for {label} (attempt {attempt}/{_MAX_RETRIES}): {exc}. Retrying in {wait}s...")
                    await asyncio.sleep(wait)

        logger.error(f"LLM call for {label} failed after {_MAX_RETRIES} attempts: {last_exc}")
        return None

    @staticmethod
    def _parse_llm_json(content: str) -> Optional[Dict]:
        """Extract and parse a JSON dict from LLM output using robust fallbacks."""
        if not content:
            return None

        text = content.strip()
        candidates: List[str] = [text]

        # Prefer explicit fenced JSON blocks when present.
        fence_blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
        candidates.extend(block.strip() for block in fence_blocks if block.strip())

        # Then try balanced JSON-object slices extracted from the text.
        candidates.extend(ServiceEvaluator._extract_json_objects(text))

        seen: set[str] = set()
        for candidate in candidates:
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)

            parsed = ServiceEvaluator._try_parse_dict(candidate)
            if parsed is not None:
                return parsed

        return None

    @staticmethod
    def _try_parse_dict(candidate: str) -> Optional[Dict]:
        """Try parsing a string into a dict via JSON first, then Python literal."""
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        try:
            parsed = ast.literal_eval(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        return None

    @staticmethod
    def _extract_json_objects(text: str) -> List[str]:
        """Extract balanced `{...}` object substrings while respecting JSON strings."""
        objects: List[str] = []
        depth = 0
        start: Optional[int] = None
        in_string = False
        escape = False

        for i, ch in enumerate(text):
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
                continue

            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
                continue

            if ch == "}" and depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    objects.append(text[start:i + 1])
                    start = None

        return objects

    @staticmethod
    def _completeness_cap(required_prompts: List[str], responses: List[Dict[str, str]]) -> int:
        """Return a 0-100 cap based on coverage of required prompts with non-empty responses."""
        if not required_prompts:
            return 100

        required_set = {str(p).strip() for p in required_prompts if str(p).strip()}
        if not required_set:
            return 100

        answered_prompts = {
            str(item.get("prompt", "")).strip()
            for item in responses
            if str(item.get("response", "")).strip()
        }
        covered = len(required_set.intersection(answered_prompts))
        return round((covered / len(required_set)) * 100)

    @staticmethod
    def _coerce_score(value: Any) -> Optional[int]:
        """Convert an arbitrary score value to a bounded integer in [0, 100]."""
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None

        return max(0, min(100, round(numeric)))

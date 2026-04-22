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

import aiohttp

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
        self._api_url = config.openrouter_base_url.rstrip("/") + "/chat/completions"
        self._headers = {
            "Authorization": f"Bearer {config.openrouter_api_key}",
            "Content-Type": "application/json",
        }
        self._model = getattr(config, "openrouter_model", "openai/gpt-4o-mini")
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

        async with aiohttp.ClientSession() as session:
            # --- parallel per-pair scoring ---
            pair_tasks = [
                self._score_pair(session, pair, per_pair_criteria)
                for pair in responses
            ]
            pair_results: List[Optional[Dict]] = await asyncio.gather(*pair_tasks)

            # --- holistic scoring (single call, all responses) ---
            holistic_scores: Dict[str, int] = {}
            holistic_explanation: str = ""
            if holistic_criteria:
                holistic_scores, holistic_explanation = await self._score_holistic(
                    session, responses, holistic_criteria
                )

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
        session: aiohttp.ClientSession,
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

        parsed = await self._llm_call_with_retry(session, eval_prompt, label=f"pair '{prompt_text[:40]}'")
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
        session: aiohttp.ClientSession,
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

        parsed = await self._llm_call_with_retry(session, eval_prompt, label="holistic")
        if parsed is None or "scores" not in parsed:
            logger.warning("No valid holistic scores returned; defaulting to 0 for holistic dimensions.")
            return {dim: 0 for dim in criteria}, ""

        scores: Dict[str, int] = {}
        for dim in criteria:
            if dim in parsed.get("scores", {}):
                coerced = self._coerce_score(parsed["scores"][dim])
                if coerced is not None:
                    scores[dim] = coerced

        return scores, parsed.get("explanation", "")

    async def _llm_call_with_retry(
        self,
        session: aiohttp.ClientSession,
        user_prompt: str,
        label: str = "",
    ) -> Optional[Dict]:
        """
        Call the LLM API with up to _MAX_RETRIES attempts on transient failure.
        Returns the parsed JSON dict or None if all attempts fail.
        """
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": "You are a strict, fair, and concise evaluator. Respond with JSON only."},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 512,
        }

        last_exc: Optional[Exception] = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                async with session.post(self._api_url, json=payload, headers=self._headers, timeout=60) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    content = data["choices"][0]["message"]["content"]
                    return self._parse_llm_json(content)
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
        """Extract and parse the first JSON object from LLM output."""
        match = re.search(r'\{[\s\S]*\}', content)
        if not match:
            return None
        block = match.group()
        try:
            return json.loads(block)
        except Exception:
            try:
                return ast.literal_eval(block)
            except Exception:
                return None

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

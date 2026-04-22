"""
Tests for ServiceEvaluator.

Unit tests mock the aiohttp HTTP call to the LLM API.

Prerequisites for integration tests:
1. OPENROUTER_API_KEY in agents/.env

Run with: python -m unittest agents.tests.test_consumer_evaluator
"""

import unittest
import asyncio
import json
import logging
from unittest.mock import Mock, AsyncMock, patch, MagicMock

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from agents.consumer_agent.evaluator import ServiceEvaluator
from agents.config import Config


def run_async(coro):
    """Helper to run async functions in sync tests."""
    return asyncio.get_event_loop().run_until_complete(coro)


class TestServiceEvaluatorUnit(unittest.TestCase):
    """Unit tests for ServiceEvaluator with mocked LLM HTTP calls."""

    def setUp(self):
        self.mock_config = Mock(spec=Config)
        self.mock_config.openrouter_api_key = "test-api-key"
        self.mock_config.openrouter_base_url = "https://openrouter.ai/api/v1"
        self.mock_config.openrouter_model = "openai/gpt-4o-mini"
        self.sample_requirements = {
            "title": "Literature Review on Quantum Computing",
            "description": "Review recent papers on quantum algorithms",
            "prompts": [
                "What are the main quantum computing paradigms?",
                "Explain quantum entanglement."
            ],
            "quality_criteria": {
                "completeness": "All prompts answered comprehensively",
                "depth": "Detailed explanations with examples",
                "clarity": "Clear and well-structured responses"
            },
            "service_type": "literature_review"
        }
        self.sample_result_high_quality = {
            "success": True,
            "responses": [
                {
                    "prompt": "What are the main quantum computing paradigms?",
                    "response": "The main paradigms include gate-based quantum computing using quantum gates like Hadamard and CNOT, adiabatic quantum computing for optimization, and topological quantum computing using anyons for fault tolerance."
                },
                {
                    "prompt": "Explain quantum entanglement.",
                    "response": "Quantum entanglement is a phenomenon where quantum particles become correlated such that the state of one particle instantaneously affects the state of another, regardless of distance. This is described by Bell's theorem and EPR paradox."
                }
            ]
        }
        self.sample_result_low_quality = {
            "success": True,
            "responses": [
                {
                    "prompt": "What are the main quantum computing paradigms?",
                    "response": "Quantum computing."
                },
                {
                    "prompt": "Explain quantum entanglement.",
                    "response": "It's a quantum thing."
                }
            ]
        }
        self.evaluator = ServiceEvaluator(self.mock_config)

    # ========================================================================
    # Initialization
    # ========================================================================

    def test_evaluator_initialization(self):
        """Test ServiceEvaluator initializes correctly."""
        evaluator = ServiceEvaluator(self.mock_config)
        self.assertEqual(evaluator.config, self.mock_config)
        print("\n✓ ServiceEvaluator initialized")

    # ========================================================================
    # evaluate() — structure and happy path
    # ========================================================================

    @patch("aiohttp.ClientSession.post")
    def test_evaluate_returns_correct_structure(self, mock_post):
        """evaluate() returns rating, quality_scores, explanations, error=None."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = AsyncMock(return_value={
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "scores": {"completeness": 95, "depth": 90, "clarity": 92},
                        "explanation": "Excellent, detailed, and clear."
                    })
                }
            }]
        })
        mock_post.return_value.__aenter__.return_value = mock_response

        result = run_async(self.evaluator.evaluate(
            self.sample_requirements, self.sample_result_high_quality
        ))

        self.assertIn("rating", result)
        self.assertIn("quality_scores", result)
        self.assertIn("explanations", result)
        self.assertIsNone(result["error"])
        self.assertGreaterEqual(result["rating"], 0)
        self.assertLessEqual(result["rating"], 100)
        print("\n✓ evaluate() returns correct structure for high-quality input")

    @patch("aiohttp.ClientSession.post")
    def test_evaluate_aggregates_per_pair_scores(self, mock_post):
        """Overall rating is the average of per-pair per-dimension scores."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = AsyncMock(return_value={
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "scores": {"completeness": 80, "depth": 60},
                        "explanation": "ok"
                    })
                }
            }]
        })
        mock_post.return_value.__aenter__.return_value = mock_response

        requirements = {"quality_criteria": {"completeness": "desc", "depth": "desc"}}
        result_data = {
            "responses": [
                {"prompt": "q1", "response": "a1"},
                {"prompt": "q2", "response": "a2"},
            ]
        }
        result = run_async(self.evaluator.evaluate(requirements, result_data))

        # completeness avg = 80, depth avg = 60 -> overall = 70
        self.assertEqual(result["quality_scores"]["completeness"], 80)
        self.assertEqual(result["quality_scores"]["depth"], 60)
        self.assertEqual(result["rating"], 70)
        print("\n✓ evaluate() aggregates scores correctly")

    # ========================================================================
    # evaluate() — edge cases
    # ========================================================================

    @patch("aiohttp.ClientSession.post")
    def test_evaluate_empty_responses(self, mock_post):
        """evaluate() returns rating=0 and an error message for empty responses."""
        empty_result = {"success": True, "responses": []}
        result = run_async(self.evaluator.evaluate(self.sample_requirements, empty_result))
        self.assertEqual(result["rating"], 0)
        self.assertIsNotNone(result["error"])
        print("\n✓ evaluate() handles empty responses")

    @patch("aiohttp.ClientSession.post")
    def test_evaluate_missing_quality_criteria(self, mock_post):
        """evaluate() returns rating=0 and an error when quality_criteria is absent."""
        req = {k: v for k, v in self.sample_requirements.items() if k != "quality_criteria"}
        result = run_async(self.evaluator.evaluate(req, self.sample_result_high_quality))
        self.assertEqual(result["rating"], 0)
        self.assertIsNotNone(result["error"])
        print("\n✓ evaluate() handles missing quality_criteria")

    @patch("aiohttp.ClientSession.post")
    def test_evaluate_llm_returns_invalid_json(self, mock_post):
        """evaluate() skips a pair gracefully when LLM returns non-JSON."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = AsyncMock(return_value={
            "choices": [{"message": {"content": "Sorry, I cannot evaluate this."}}]
        })
        mock_post.return_value.__aenter__.return_value = mock_response

        result = run_async(self.evaluator.evaluate(
            self.sample_requirements, self.sample_result_high_quality
        ))

        # All pairs skipped -> all dimensions get 0 -> overall = 0
        self.assertEqual(result["rating"], 0)
        self.assertIsNone(result["error"])
        print("\n✓ evaluate() skips pairs when LLM returns unparseable output")

    @patch("aiohttp.ClientSession.post")
    def test_evaluate_rating_clamped_to_100(self, mock_post):
        """Overall and per-dimension scores are clamped to 100."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = AsyncMock(return_value={
            "choices": [{"message": {"content": json.dumps({
                "scores": {"completeness": 150, "depth": 200},
                "explanation": "unrealistically high"
            })}}]
        })
        mock_post.return_value.__aenter__.return_value = mock_response

        requirements = {"quality_criteria": {"completeness": "desc", "depth": "desc"}}
        result_data = {"responses": [{"prompt": "q", "response": "a"}]}
        result = run_async(self.evaluator.evaluate(requirements, result_data))

        self.assertLessEqual(result["rating"], 100)
        self.assertEqual(result["quality_scores"]["completeness"], 100)
        self.assertEqual(result["quality_scores"]["depth"], 100)
        print("\n✓ evaluate() clamps overall and per-dimension scores")

    @patch("aiohttp.ClientSession.post")
    def test_evaluate_llm_api_error_skips_pair(self, mock_post):
        """evaluate() continues when an HTTP error occurs for one pair."""
        mock_post.return_value.__aenter__.side_effect = Exception("Connection error")

        result = run_async(self.evaluator.evaluate(
            self.sample_requirements, self.sample_result_high_quality
        ))

        # Errors are caught per-pair; result should still return valid structure
        self.assertIn("rating", result)
        self.assertIn("quality_scores", result)
        print("\n✓ evaluate() handles LLM API errors gracefully")

    @patch("aiohttp.ClientSession.post")
    def test_evaluate_llm_json_with_markdown_fences(self, mock_post):
        """evaluate() correctly parses LLM response wrapped in markdown code fences."""
        content_with_fences = (
            "Here are the scores:\n```json\n"
            + json.dumps({"scores": {"completeness": 70}, "explanation": "ok"})
            + "\n```"
        )
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = AsyncMock(return_value={
            "choices": [{"message": {"content": content_with_fences}}]
        })
        mock_post.return_value.__aenter__.return_value = mock_response

        requirements = {"quality_criteria": {"completeness": "desc"}}
        result_data = {"responses": [{"prompt": "q", "response": "a"}]}
        result = run_async(self.evaluator.evaluate(requirements, result_data))

        self.assertEqual(result["quality_scores"].get("completeness"), 70)
        self.assertEqual(result["rating"], 70)
        print("\n✓ evaluate() parses JSON from markdown-fenced LLM output")

    @patch("aiohttp.ClientSession.post")
    def test_evaluate_explanations_collected(self, mock_post):
        """evaluate() collects pair explanations plus one holistic explanation."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = AsyncMock(return_value={
            "choices": [{"message": {"content": json.dumps({
                "scores": {"depth": 75, "completeness": 80},
                "explanation": "Good answer."
            })}}]
        })
        mock_post.return_value.__aenter__.return_value = mock_response

        requirements = {"quality_criteria": {"depth": "desc", "completeness": "desc"}}
        result_data = {
            "responses": [
                {"prompt": "q1", "response": "a1"},
                {"prompt": "q2", "response": "a2"},
            ]
        }
        result = run_async(self.evaluator.evaluate(requirements, result_data))

        # 2 pair calls + 1 holistic call
        self.assertEqual(len(result["explanations"]), 3)
        self.assertEqual(result["explanations"][0], "Good answer.")
        print("\n✓ evaluate() collects pair and holistic explanations")

    @patch("aiohttp.ClientSession.post")
    def test_evaluate_holistic_called_once(self, mock_post):
        """With holistic criteria present, evaluate() makes N pair calls + 1 holistic call."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = AsyncMock(return_value={
            "choices": [{"message": {"content": json.dumps({
                "scores": {"depth": 70, "clarity": 75, "completeness": 80},
                "explanation": "ok"
            })}}]
        })
        mock_post.return_value.__aenter__.return_value = mock_response

        requirements = {
            "quality_criteria": {
                "depth": "desc",
                "clarity": "desc",
                "completeness": "desc",
            }
        }
        result_data = {
            "responses": [
                {"prompt": "q1", "response": "a1"},
                {"prompt": "q2", "response": "a2"},
            ]
        }

        run_async(self.evaluator.evaluate(requirements, result_data))

        self.assertEqual(mock_post.call_count, 3)
        print("\n✓ evaluate() performs holistic scoring in a single extra call")

    @patch("aiohttp.ClientSession.post")
    def test_evaluate_no_holistic_criteria_makes_only_pair_calls(self, mock_post):
        """Without holistic criteria, evaluate() makes one call per response only."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = AsyncMock(return_value={
            "choices": [{"message": {"content": json.dumps({
                "scores": {"depth": 70, "clarity": 75},
                "explanation": "ok"
            })}}]
        })
        mock_post.return_value.__aenter__.return_value = mock_response

        requirements = {"quality_criteria": {"depth": "desc", "clarity": "desc"}}
        result_data = {
            "responses": [
                {"prompt": "q1", "response": "a1"},
                {"prompt": "q2", "response": "a2"},
                {"prompt": "q3", "response": "a3"},
            ]
        }

        run_async(self.evaluator.evaluate(requirements, result_data))

        self.assertEqual(mock_post.call_count, 3)
        print("\n✓ evaluate() skips holistic call when no holistic criteria exist")

    @patch("aiohttp.ClientSession.post")
    def test_evaluate_completeness_capped_by_prompt_coverage(self, mock_post):
        """Completeness is capped when fewer required prompts are answered."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = AsyncMock(return_value={
            "choices": [{"message": {"content": json.dumps({
                "scores": {"completeness": 100},
                "explanation": "Looks complete."
            })}}]
        })
        mock_post.return_value.__aenter__.return_value = mock_response

        requirements = {
            "prompts": ["q1", "q2", "q3"],
            "quality_criteria": {"completeness": "all prompts answered"},
        }
        result_data = {
            "responses": [
                {"prompt": "q1", "response": "a1"},
                {"prompt": "q2", "response": "a2"},
            ]
        }

        result = run_async(self.evaluator.evaluate(requirements, result_data))

        self.assertEqual(result["quality_scores"]["completeness"], 67)
        self.assertEqual(result["rating"], 67)
        print("\n✓ evaluate() caps completeness by answered prompt coverage")


class TestServiceEvaluatorIntegration(unittest.TestCase):
    """Integration tests that call the real OpenRouter API. Skipped if no key."""

    def setUp(self):
        self.config = Config()
        if not self.config.openrouter_api_key:
            self.skipTest("OPENROUTER_API_KEY not configured - skipping integration tests")
        self.evaluator = ServiceEvaluator(self.config)

    def test_real_llm_high_quality(self):
        """Real LLM scores high-quality responses above 50."""
        async def _test():
            requirements = {
                "prompts": ["What are convolutional neural networks?"],
                "quality_criteria": {
                    "completeness": "Fully answers the question",
                    "clarity": "Clear and well-structured"
                }
            }
            result = {
                "responses": [{
                    "prompt": "What are convolutional neural networks?",
                    "response": (
                        "Convolutional neural networks (CNNs) are a class of deep learning models designed "
                        "for grid-structured data like images. They use convolutional layers with learnable "
                        "filters to detect local features, pooling layers to reduce spatial dimensions, and "
                        "fully connected layers for classification. CNNs excel at image recognition, object "
                        "detection, and similar vision tasks."
                    )
                }]
            }
            evaluation = await self.evaluator.evaluate(requirements, result)
            self.assertGreater(evaluation["rating"], 50)
            print(f"\n✅ Real LLM high-quality rating: {evaluation['rating']}/100")
        run_async(_test())

    def test_real_llm_low_quality(self):
        """Real LLM scores low-quality responses; rating is within 0-100."""
        async def _test():
            requirements = {
                "prompts": ["What are convolutional neural networks?"],
                "quality_criteria": {
                    "completeness": "Fully answers the question",
                    "clarity": "Clear and well-structured"
                }
            }
            result = {
                "responses": [{
                    "prompt": "What are convolutional neural networks?",
                    "response": "They are neural networks."
                }]
            }
            evaluation = await self.evaluator.evaluate(requirements, result)
            self.assertGreaterEqual(evaluation["rating"], 0)
            self.assertLessEqual(evaluation["rating"], 100)
            print(f"\n✅ Real LLM low-quality rating: {evaluation['rating']}/100")
        run_async(_test())

    def test_real_llm_partial_completion(self):
        """Real LLM handles fewer responses than prompts."""
        async def _test():
            requirements = {
                "prompts": [
                    "What is machine learning?",
                    "What is deep learning?",
                    "What is reinforcement learning?"
                ],
                "quality_criteria": {
                    "completeness": "All prompts answered"
                }
            }
            result = {
                "responses": [
                    {
                        "prompt": "What is machine learning?",
                        "response": "Machine learning is a subset of AI that enables systems to learn from data."
                    },
                    {
                        "prompt": "What is deep learning?",
                        "response": "Deep learning uses neural networks with multiple layers."
                    }
                ]
            }
            evaluation = await self.evaluator.evaluate(requirements, result)
            self.assertGreaterEqual(evaluation["rating"], 0)
            self.assertLessEqual(evaluation["rating"], 100)
            print(f"\n✅ Real LLM partial completion rating: {evaluation['rating']}/100")
        run_async(_test())


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("RUNNING SERVICEEVALUATOR TESTS")
    print("=" * 80)
    unittest.main(verbosity=2)

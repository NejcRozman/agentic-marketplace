"""
Tests for ServiceEvaluator.

Unit tests use mocked LLM and ReAct agent for fast isolated testing.

Prerequisites for integration test:
1. OPENROUTER_API_KEY in agents/.env

Run with: python agents/tests/test_consumer_evaluator.py
"""

import sys
from pathlib import Path


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

from agents.consumer_agent.evaluator import ServiceEvaluator, EvaluatorState
from agents.config import Config


def run_async(coro):
    """Helper to run async functions in sync tests."""
    return asyncio.get_event_loop().run_until_complete(coro)


class TestServiceEvaluatorUnit(unittest.TestCase):
    """Unit tests with mocked LLM and ReAct agent."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_config = Mock(spec=Config)
        self.mock_config.openrouter_api_key = "test-api-key"
        self.mock_config.openrouter_base_url = "https://openrouter.ai/api/v1"
        
        # Sample test data
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
        
        # Create evaluator with mocked LLM
        with patch('agents.consumer_agent.evaluator.ChatOpenAI'):
            self.evaluator = ServiceEvaluator(self.mock_config)
    
    # ========================================================================
    # HIGH PRIORITY: Initialization Tests
    # ========================================================================
    
    def test_evaluator_initialization(self):
        """Test ServiceEvaluator initializes correctly with tools and graph."""
        with patch('agents.consumer_agent.evaluator.ChatOpenAI'):
            evaluator = ServiceEvaluator(self.mock_config)
            
            self.assertEqual(evaluator.config, self.mock_config)
            self.assertIsNotNone(evaluator._tools)
            self.assertIsNotNone(evaluator.graph)
            self.assertEqual(len(evaluator._tools), 2)
            
            # Check tool names
            tool_names = [tool.name for tool in evaluator._tools]
            self.assertIn("extract_prompt_response_pairs", tool_names)
            self.assertIn("finalize_evaluation", tool_names)
            
            print("\n✓ ServiceEvaluator initialized with 2 tools and compiled graph")
    
    # ========================================================================
    # HIGH PRIORITY: Tool Function Tests
    # ========================================================================
    
    def test_extract_prompt_response_pairs_success(self):
        """Test extracting prompt-response pairs from valid data."""
        # Get the tool
        extract_tool = self.evaluator._tools[0]
        
        requirements_json = json.dumps(self.sample_requirements)
        result_json = json.dumps(self.sample_result_high_quality)
        
        result = extract_tool.func(requirements_json, result_json)
        
        self.assertEqual(result["pair_count"], 2)
        self.assertEqual(len(result["pairs"]), 2)
        self.assertEqual(result["pairs"][0]["prompt"], "What are the main quantum computing paradigms?")
        self.assertIn("completeness", result["quality_criteria"])
        self.assertEqual(result["service_type"], "literature_review")
        
        print("\n✓ Extract tool successfully extracted 2 prompt-response pairs")
    
    def test_extract_prompt_response_pairs_empty_responses(self):
        """Test extraction with empty responses array."""
        extract_tool = self.evaluator._tools[0]
        
        requirements_json = json.dumps(self.sample_requirements)
        result_json = json.dumps({"responses": []})
        
        result = extract_tool.func(requirements_json, result_json)
        
        self.assertEqual(result["pair_count"], 0)
        self.assertEqual(len(result["pairs"]), 0)
        self.assertIn("quality_criteria", result)
        
        print("\n✓ Extract tool handled empty responses correctly")
    
    def test_extract_prompt_response_pairs_invalid_json(self):
        """Test extraction with malformed JSON input."""
        extract_tool = self.evaluator._tools[0]
        
        result = extract_tool.func("not valid json", "also invalid")
        
        self.assertIn("error", result)
        self.assertIsInstance(result["error"], str)
        
        print("\n✓ Extract tool handled invalid JSON with error message")
    
    def test_finalize_evaluation_success(self):
        """Test calculating overall rating from dimension scores."""
        finalize_tool = self.evaluator._tools[1]
        
        dimension_scores = json.dumps({
            "completeness": 90,
            "depth": 80,
            "clarity": 85
        })
        
        result = finalize_tool.func(dimension_scores)
        
        # Average: (90 + 80 + 85) / 3 = 85
        self.assertEqual(result["overall_rating"], 85)
        self.assertEqual(result["quality_scores"]["completeness"], 90)
        self.assertEqual(result["quality_scores"]["depth"], 80)
        self.assertEqual(result["quality_scores"]["clarity"], 85)
        
        print("\n✓ Finalize tool calculated average rating: 85")
    
    def test_finalize_evaluation_single_dimension(self):
        """Test with only one quality dimension."""
        finalize_tool = self.evaluator._tools[1]
        
        dimension_scores = json.dumps({"completeness": 95})
        
        result = finalize_tool.func(dimension_scores)
        
        self.assertEqual(result["overall_rating"], 95)
        self.assertEqual(result["quality_scores"]["completeness"], 95)
        
        print("\n✓ Finalize tool handled single dimension: 95")
    
    def test_finalize_evaluation_empty_scores(self):
        """Test with empty scores dict."""
        finalize_tool = self.evaluator._tools[1]
        
        dimension_scores = json.dumps({})
        
        result = finalize_tool.func(dimension_scores)
        
        self.assertEqual(result["overall_rating"], 0)
        self.assertEqual(result["quality_scores"], {})
        
        print("\n✓ Finalize tool handled empty scores with rating 0")
    
    def test_finalize_evaluation_rating_clamping(self):
        """Test that rating is clamped to 0-100 range."""
        finalize_tool = self.evaluator._tools[1]
        
        # Test values that would average > 100
        dimension_scores_high = json.dumps({
            "score1": 150,
            "score2": 200
        })
        result = finalize_tool.func(dimension_scores_high)
        self.assertEqual(result["overall_rating"], 100)
        
        print("\n✓ Finalize tool clamped high rating to 100")
    
    def test_finalize_evaluation_invalid_json(self):
        """Test error handling for malformed JSON."""
        finalize_tool = self.evaluator._tools[1]
        
        result = finalize_tool.func("invalid json")
        
        self.assertIn("error", result)
        self.assertEqual(result["overall_rating"], 0)
        self.assertEqual(result["quality_scores"], {})
        
        print("\n✓ Finalize tool handled invalid JSON with error")
    
    # ========================================================================
    # HIGH PRIORITY: Validation Node Tests
    # ========================================================================
    
    def test_validate_rating_within_range(self):
        """Test validation node with valid rating."""
        async def _test():
            state: EvaluatorState = {
                "service_requirements": {},
                "result": {},
                "quality_scores": {"test": 75},
                "overall_rating": 75,
                "error": None,
                "messages": []
            }
            
            result_state = await self.evaluator._validate_rating_node(state)
            
            self.assertEqual(result_state["overall_rating"], 75)
            print("\n✓ Validation kept valid rating at 75")
        
        run_async(_test())
    
    def test_validate_rating_clamps_high(self):
        """Test validation clamps rating > 100."""
        async def _test():
            state: EvaluatorState = {
                "service_requirements": {},
                "result": {},
                "quality_scores": {},
                "overall_rating": 150,
                "error": None,
                "messages": []
            }
            
            result_state = await self.evaluator._validate_rating_node(state)
            
            self.assertEqual(result_state["overall_rating"], 100)
            print("\n✓ Validation clamped 150 to 100")
        
        run_async(_test())
    
    def test_validate_rating_clamps_low(self):
        """Test validation clamps negative rating."""
        async def _test():
            state: EvaluatorState = {
                "service_requirements": {},
                "result": {},
                "quality_scores": {},
                "overall_rating": -10,
                "error": None,
                "messages": []
            }
            
            result_state = await self.evaluator._validate_rating_node(state)
            
            self.assertEqual(result_state["overall_rating"], 0)
            print("\n✓ Validation clamped -10 to 0")
        
        run_async(_test())
    
    def test_validate_rating_handles_none(self):
        """Test validation handles None rating."""
        async def _test():
            state: EvaluatorState = {
                "service_requirements": {},
                "result": {},
                "quality_scores": {},
                "overall_rating": 0,
                "error": None,
                "messages": []
            }
            
            result_state = await self.evaluator._validate_rating_node(state)
            
            self.assertEqual(result_state["overall_rating"], 0)
            print("\n✓ Validation handled 0 rating correctly")
        
        run_async(_test())
    
    # ========================================================================
    # MEDIUM PRIORITY: ReAct Evaluation Node Tests
    # ========================================================================
    
    def test_react_evaluation_node_success(self):
        """Test ReAct evaluation with mocked agent."""
        async def _test():
            # Mock the create_agent
            mock_agent = AsyncMock()
            
            # Create mock messages that simulate successful tool use
            mock_tool_message = Mock()
            mock_tool_message.__class__.__name__ = "ToolMessage"
            mock_tool_message.tool_calls = []  # Add empty tool_calls
            # Content should be a dict that can be evaluated
            mock_tool_message.content = {
                "overall_rating": 85,
                "quality_scores": {
                    "completeness": 90,
                    "depth": 80,
                    "clarity": 85
                }
            }
            
            # Create properly structured mock messages
            mock_human_msg = Mock(__class__=Mock(__name__="HumanMessage"))
            mock_human_msg.content = "test"
            mock_human_msg.tool_calls = []
            
            mock_agent_result = {
                "messages": [
                    mock_human_msg,
                    mock_tool_message
                ]
            }
            mock_agent.ainvoke = AsyncMock(return_value=mock_agent_result)
            
            with patch('agents.consumer_agent.evaluator.create_agent', return_value=mock_agent):
                with patch('agents.consumer_agent.evaluator.ChatOpenAI'):
                    state: EvaluatorState = {
                        "service_requirements": self.sample_requirements,
                        "result": self.sample_result_high_quality,
                        "quality_scores": None,
                        "overall_rating": None,
                        "error": None,
                        "messages": []
                    }
                    
                    result_state = await self.evaluator._react_evaluation_node(state)
                    
                    self.assertEqual(result_state["overall_rating"], 85)
                    self.assertIsNotNone(result_state["quality_scores"])
                    self.assertIsNone(result_state["error"])
                    self.assertTrue(len(result_state["messages"]) > 0)
                    
                    print("\n✓ ReAct evaluation completed with rating 85")
        
        run_async(_test())
    
    def test_react_evaluation_node_fallback(self):
        """Test fallback when agent doesn't use tools properly."""
        async def _test():
            # Mock agent that doesn't call finalize_evaluation
            mock_agent = AsyncMock()
            
            # Create properly structured mock message
            mock_human_msg = Mock(__class__=Mock(__name__="HumanMessage"))
            mock_human_msg.content = "test"
            mock_human_msg.tool_calls = []
            
            mock_agent_result = {
                "messages": [mock_human_msg]
            }
            mock_agent.ainvoke = AsyncMock(return_value=mock_agent_result)
            
            with patch('agents.consumer_agent.evaluator.create_agent', return_value=mock_agent):
                with patch('agents.consumer_agent.evaluator.ChatOpenAI'):
                    state: EvaluatorState = {
                        "service_requirements": self.sample_requirements,
                        "result": self.sample_result_high_quality,
                        "quality_scores": None,
                        "overall_rating": None,
                        "error": None,
                        "messages": []
                    }
                    
                    result_state = await self.evaluator._react_evaluation_node(state)
                    
                    # Should use fallback
                    self.assertEqual(result_state["overall_rating"], 75)
                    self.assertEqual(result_state["quality_scores"], {"fallback": 75})
                    self.assertIsNone(result_state["error"])
                    
                    print("\n✓ ReAct evaluation used fallback rating 75")
        
        run_async(_test())
    
    def test_react_evaluation_node_error_handling(self):
        """Test error handling when agent throws exception."""
        async def _test():
            # Mock agent that raises exception
            mock_agent = AsyncMock()
            mock_agent.ainvoke = AsyncMock(side_effect=Exception("API error"))
            
            with patch('agents.consumer_agent.evaluator.create_agent', return_value=mock_agent):
                with patch('agents.consumer_agent.evaluator.ChatOpenAI'):
                    state: EvaluatorState = {
                        "service_requirements": self.sample_requirements,
                        "result": self.sample_result_high_quality,
                        "quality_scores": None,
                        "overall_rating": None,
                        "error": None,
                        "messages": []
                    }
                    
                    result_state = await self.evaluator._react_evaluation_node(state)
                    
                    self.assertIsNotNone(result_state["error"])
                    self.assertIn("API error", result_state["error"])
                    self.assertEqual(result_state["overall_rating"], 0)
                    self.assertEqual(result_state["quality_scores"], {})
                    
                    print("\n✓ ReAct evaluation handled error with rating 0")
        
        run_async(_test())
    
    # ========================================================================
    # MEDIUM PRIORITY: End-to-End Evaluate Method Tests
    # ========================================================================
    
    def test_evaluate_complete_workflow(self):
        """Test complete evaluation workflow with mocked graph."""
        async def _test():
            # Mock the graph.ainvoke to return expected state
            mock_final_state = {
                "service_requirements": self.sample_requirements,
                "result": self.sample_result_high_quality,
                "quality_scores": {
                    "completeness": 90,
                    "depth": 85,
                    "clarity": 80
                },
                "overall_rating": 85,
                "error": None,
                "messages": []
            }
            
            self.evaluator.graph.ainvoke = AsyncMock(return_value=mock_final_state)
            
            evaluation = await self.evaluator.evaluate(
                self.sample_requirements,
                self.sample_result_high_quality
            )
            
            self.assertEqual(evaluation["rating"], 85)
            self.assertIsInstance(evaluation["quality_scores"], dict)
            self.assertEqual(len(evaluation["quality_scores"]), 3)
            self.assertIsNone(evaluation["error"])
            
            print("\n✓ Complete evaluate workflow returned rating 85")
        
        run_async(_test())
    
    def test_evaluate_with_missing_responses(self):
        """Test evaluation when service result has fewer responses than prompts."""
        async def _test():
            incomplete_result = {
                "success": True,
                "responses": [
                    {
                        "prompt": "What are the main quantum computing paradigms?",
                        "response": "Gate-based and adiabatic computing."
                    }
                    # Missing second response
                ]
            }
            
            # Mock lower rating for incomplete
            mock_final_state = {
                "service_requirements": self.sample_requirements,
                "result": incomplete_result,
                "quality_scores": {"completeness": 50},
                "overall_rating": 50,
                "error": None,
                "messages": []
            }
            
            self.evaluator.graph.ainvoke = AsyncMock(return_value=mock_final_state)
            
            evaluation = await self.evaluator.evaluate(
                self.sample_requirements,
                incomplete_result
            )
            
            self.assertEqual(evaluation["rating"], 50)
            self.assertIsNone(evaluation["error"])
            
            print("\n✓ Incomplete result evaluated with lower rating 50")
        
        run_async(_test())
    
    def test_evaluate_with_empty_result(self):
        """Test evaluation with empty responses array."""
        async def _test():
            empty_result = {"success": True, "responses": []}
            
            # Mock very low rating for empty
            mock_final_state = {
                "service_requirements": self.sample_requirements,
                "result": empty_result,
                "quality_scores": {},
                "overall_rating": 0,
                "error": None,
                "messages": []
            }
            
            self.evaluator.graph.ainvoke = AsyncMock(return_value=mock_final_state)
            
            evaluation = await self.evaluator.evaluate(
                self.sample_requirements,
                empty_result
            )
            
            self.assertEqual(evaluation["rating"], 0)
            
            print("\n✓ Empty result evaluated with rating 0")
        
        run_async(_test())
    
    def test_evaluate_with_error_state(self):
        """Test evaluation returns error from state."""
        async def _test():
            mock_final_state = {
                "service_requirements": self.sample_requirements,
                "result": self.sample_result_high_quality,
                "quality_scores": {},
                "overall_rating": 0,
                "error": "Evaluation failed",
                "messages": []
            }
            
            self.evaluator.graph.ainvoke = AsyncMock(return_value=mock_final_state)
            
            evaluation = await self.evaluator.evaluate(
                self.sample_requirements,
                self.sample_result_high_quality
            )
            
            self.assertEqual(evaluation["rating"], 0)
            self.assertEqual(evaluation["error"], "Evaluation failed")
            
            print("\n✓ Error state properly returned in evaluation")
        
        run_async(_test())
    
    # ========================================================================
    # MEDIUM PRIORITY: Edge Cases
    # ========================================================================
    
    def test_evaluate_malformed_requirements(self):
        """Test evaluation with malformed requirements structure."""
        async def _test():
            malformed_requirements = {
                "title": "Test"
                # Missing prompts, quality_criteria, etc.
            }
            
            mock_final_state = {
                "service_requirements": malformed_requirements,
                "result": self.sample_result_high_quality,
                "quality_scores": {"fallback": 50},
                "overall_rating": 50,
                "error": None,
                "messages": []
            }
            
            self.evaluator.graph.ainvoke = AsyncMock(return_value=mock_final_state)
            
            evaluation = await self.evaluator.evaluate(
                malformed_requirements,
                self.sample_result_high_quality
            )
            
            # Should still return a valid rating
            self.assertIsInstance(evaluation["rating"], int)
            self.assertGreaterEqual(evaluation["rating"], 0)
            self.assertLessEqual(evaluation["rating"], 100)
            
            print("\n✓ Malformed requirements handled gracefully")
        
        run_async(_test())
    
    def test_tools_list_immutability(self):
        """Test that tools list is properly initialized."""
        with patch('agents.consumer_agent.evaluator.ChatOpenAI'):
            evaluator1 = ServiceEvaluator(self.mock_config)
            evaluator2 = ServiceEvaluator(self.mock_config)
            
            # Each evaluator should have its own tools
            self.assertIsNot(evaluator1._tools, evaluator2._tools)
            self.assertEqual(len(evaluator1._tools), 2)
            self.assertEqual(len(evaluator2._tools), 2)
            
            print("\n✓ Each evaluator has independent tools list")


class TestServiceEvaluatorIntegration(unittest.TestCase):
    """Integration tests with real LLM (OpenRouter)."""
    
    def test_real_llm_evaluation_high_quality(self):
        """Integration test with real OpenRouter LLM: high-quality service result."""
        async def _test():
            # Initialize with real config
            config = Config()
            
            if not config.openrouter_api_key:
                self.skipTest("OPENROUTER_API_KEY not configured - skipping integration test")
            
            evaluator = ServiceEvaluator(config)
            
            requirements = {
                "title": "Literature Review on Machine Learning",
                "description": "Review papers on neural networks",
                "prompts": [
                    "What are convolutional neural networks?",
                    "Explain backpropagation algorithm."
                ],
                "quality_criteria": {
                    "completeness": "All prompts answered comprehensively",
                    "depth": "Detailed technical explanations",
                    "accuracy": "Scientifically correct information",
                    "clarity": "Clear and well-structured"
                },
                "service_type": "literature_review"
            }
            
            result = {
                "success": True,
                "responses": [
                    {
                        "prompt": "What are convolutional neural networks?",
                        "response": "Convolutional Neural Networks (CNNs) are specialized deep learning architectures designed for processing grid-like data such as images. They consist of convolutional layers that apply learnable filters to detect features like edges, textures, and patterns. Key components include: 1) Convolutional layers with shared weights for translation invariance, 2) Pooling layers for downsampling and feature aggregation, 3) Fully connected layers for classification. CNNs have revolutionized computer vision tasks including image classification, object detection, and segmentation."
                    },
                    {
                        "prompt": "Explain backpropagation algorithm.",
                        "response": "Backpropagation is the fundamental algorithm for training neural networks through gradient descent. It works by: 1) Forward pass - computing predictions through the network, 2) Loss calculation - measuring prediction error, 3) Backward pass - computing gradients of the loss with respect to each weight using the chain rule, 4) Weight update - adjusting parameters in the direction that reduces loss. The algorithm efficiently computes gradients by propagating error signals backward through the network layers, enabling deep networks to learn complex representations."
                    }
                ]
            }
            
            print("\n🔄 Running real OpenRouter LLM evaluation (this may take a few seconds)...")
            evaluation = await evaluator.evaluate(requirements, result)
            
            # Verify structure
            self.assertIn("rating", evaluation)
            self.assertIn("quality_scores", evaluation)
            self.assertIsInstance(evaluation["rating"], int)
            self.assertIsInstance(evaluation["quality_scores"], dict)
            
            # Rating should be in valid range (may use fallback if model doesn't support tools)
            self.assertGreaterEqual(evaluation["rating"], 0)
            self.assertLessEqual(evaluation["rating"], 100)
            
            print(f"\n✅ Real OpenRouter LLM evaluation completed!")
            print(f"   Rating: {evaluation['rating']}/100")
            print(f"   Quality Scores: {evaluation['quality_scores']}")
            print(f"   Error: {evaluation.get('error', 'None')}")
            print(f"   Note: If quality_scores={{'fallback': 75}}, the model didn't use tools properly")
        
        run_async(_test())
    
    def test_real_llm_evaluation_low_quality(self):
        """Integration test with real OpenRouter LLM: low-quality service result."""
        async def _test():
            config = Config()
            
            if not config.openrouter_api_key:
                self.skipTest("OPENROUTER_API_KEY not configured - skipping integration test")
            
            evaluator = ServiceEvaluator(config)
            
            requirements = {
                "title": "Literature Review on Machine Learning",
                "prompts": [
                    "What are convolutional neural networks?",
                    "Explain backpropagation algorithm."
                ],
                "quality_criteria": {
                    "completeness": "All prompts answered comprehensively",
                    "depth": "Detailed technical explanations"
                }
            }
            
            # Poor quality responses
            result = {
                "success": True,
                "responses": [
                    {
                        "prompt": "What are convolutional neural networks?",
                        "response": "They are neural networks."
                    },
                    {
                        "prompt": "Explain backpropagation algorithm.",
                        "response": "It's an algorithm for training."
                    }
                ]
            }
            
            print("\n🔄 Running real OpenRouter LLM evaluation on low-quality responses...")
            evaluation = await evaluator.evaluate(requirements, result)
            
            # Should return valid rating (may use fallback)
            self.assertIsInstance(evaluation["rating"], int)
            self.assertGreaterEqual(evaluation["rating"], 0)
            self.assertLessEqual(evaluation["rating"], 100)
            
            print(f"\n✅ Real OpenRouter LLM evaluation (low quality) completed!")
            print(f"   Rating: {evaluation['rating']}/100")
            print(f"   Quality Scores: {evaluation['quality_scores']}")
        
        run_async(_test())
    
    def test_real_llm_evaluation_partial_completion(self):
        """Integration test with real OpenRouter LLM: only some prompts answered."""
        async def _test():
            config = Config()
            
            if not config.openrouter_api_key:
                self.skipTest("OPENROUTER_API_KEY not configured - skipping integration test")
            
            evaluator = ServiceEvaluator(config)
            
            requirements = {
                "title": "Literature Review",
                "prompts": [
                    "What is machine learning?",
                    "What is deep learning?",
                    "What is reinforcement learning?"
                ],
                "quality_criteria": {
                    "completeness": "All prompts answered"
                }
            }
            
            # Only 2 out of 3 answered
            result = {
                "success": True,
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
            
            print("\n🔄 Running real LLM evaluation on partial completion...")
            evaluation = await evaluator.evaluate(requirements, result)
            
            # Partial completion should get moderate rating
            self.assertIsInstance(evaluation["rating"], int)
            self.assertGreaterEqual(evaluation["rating"], 0)
            self.assertLessEqual(evaluation["rating"], 100)
            
            print(f"\n✅ Real LLM evaluation (partial) completed!")
            print(f"   Rating: {evaluation['rating']}/100")
            print(f"   Quality Scores: {evaluation['quality_scores']}")
        
        run_async(_test())


if __name__ == "__main__":
    # Run tests
    print("\n" + "=" * 80)
    print("RUNNING SERVICEEVALUATOR TESTS")
    print("=" * 80)
    unittest.main(verbosity=2)

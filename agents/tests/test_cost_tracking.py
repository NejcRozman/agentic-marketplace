"""
Tests for cost tracking functionality.

This test suite validates:
1. CostTracker class: Tracking LLM costs, gas costs, and revenue
2. LLMCostCallback: Intercepting LLM calls and calculating costs
3. Integration: Cost tracking across ServiceExecutor and BlockchainHandler
4. Cost summary logging and balance calculations

Prerequisites:
1. Valid OpenRouter API key in agents/.env (OPENROUTER_API_KEY)
2. PDF files in utils/files/ directory

Run with: python agents/tests/test_cost_tracking.py
"""

import sys
import os
from pathlib import Path
import unittest
import logging
from unittest.mock import Mock, patch, MagicMock

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from agents.infrastructure.cost_tracker import CostTracker, LLMCostCallback
from agents.provider_agent.service_executor import ServiceExecutor
from agents.config import Config
from langchain_core.messages import HumanMessage
from langchain_core.outputs import LLMResult, ChatGeneration


class TestCostTracker(unittest.TestCase):
    """Test CostTracker class for tracking costs and revenue."""
    
    def setUp(self):
        """Create a fresh CostTracker for each test."""
        self.config = Config()
        self.tracker = CostTracker(agent_id="test_agent", config=self.config)
    
    def test_initialization(self):
        """Test CostTracker initializes with zero costs."""
        self.assertEqual(self.tracker.agent_id, "test_agent")
        self.assertEqual(self.tracker.total_llm_costs, 0.0)
        self.assertEqual(self.tracker.total_gas_costs, 0.0)
        self.assertEqual(self.tracker.total_revenue, 0.0)
        print("\n✓ CostTracker initializes with zero costs")
    
    def test_add_llm_cost(self):
        """Test adding LLM costs."""
        self.tracker.add_llm_cost(1.5, "test reasoning")
        
        self.assertEqual(self.tracker.total_llm_costs, 1.5)
        
        print(f"\n✓ Added LLM cost: ${1.5}")
    
    def test_add_multiple_llm_costs(self):
        """Test adding multiple LLM costs accumulates correctly."""
        self.tracker.add_llm_cost(1.0, "reasoning 1")
        self.tracker.add_llm_cost(2.5, "reasoning 2")
        self.tracker.add_llm_cost(0.5, "reasoning 3")
        
        self.assertEqual(self.tracker.total_llm_costs, 4.0)
        
        print(f"\n✓ Multiple LLM costs accumulated: ${self.tracker.total_llm_costs}")
    
    def test_add_gas_cost(self):
        """Test adding gas costs."""
        gas_price_wei = int(self.config.gas_price_gwei * 1e9)
        self.tracker.add_gas_cost(
            gas_used=50000,
            gas_price_wei=gas_price_wei,
            context="place_bid transaction"
        )
        
        # Calculate expected cost
        gas_price_gwei = gas_price_wei / 1e9
        cost_eth = (50000 * gas_price_gwei) / 1e9
        expected_cost = cost_eth * self.config.eth_price_usd
        
        self.assertAlmostEqual(self.tracker.total_gas_costs, expected_cost, places=6)
        
        print(f"\n✓ Added gas cost: ${expected_cost:.6f}")
        print(f"  Gas used: 50000, Price: {self.config.gas_price_gwei} gwei")
    
    def test_add_revenue(self):
        """Test adding revenue from completed services."""
        self.tracker.add_revenue(
            revenue_usd=100,
            context="Service #123 completed"
        )
        
        self.assertEqual(self.tracker.total_revenue, 100.0)
        
        print(f"\n✓ Added revenue: ${100.0}")
    
    def test_get_net_balance(self):
        """Test net balance calculation."""
        # Add some costs and revenue
        self.tracker.add_llm_cost(5.0, "reasoning")
        gas_price_wei = int(self.config.gas_price_gwei * 1e9)
        self.tracker.add_gas_cost(100000, gas_price_wei, "transaction")
        self.tracker.add_revenue(200, "service completed")
        
        net_balance = self.tracker.get_net_balance()
        expected = 200.0 - (5.0 + self.tracker.total_gas_costs)
        
        self.assertAlmostEqual(net_balance, expected, places=6)
        print(f"\n✓ Net balance: ${net_balance:.6f}")
        print(f"  Revenue: ${200.0}")
        print(f"  LLM costs: ${5.0}")
        print(f"  Gas costs: ${self.tracker.total_gas_costs:.6f}")
    
    def test_negative_balance(self):
        """Test net balance when costs exceed revenue."""
        self.tracker.add_llm_cost(100.0, "expensive reasoning")
        gas_price_wei = int(self.config.gas_price_gwei * 1e9)
        self.tracker.add_gas_cost(1000000, gas_price_wei, "expensive transaction")
        self.tracker.add_revenue(10.0, "small payment")
        
        net_balance = self.tracker.get_net_balance()
        
        self.assertLess(net_balance, 0)
        print(f"\n✓ Negative balance correctly calculated: ${net_balance:.6f}")


class TestLLMCostCallback(unittest.TestCase):
    """Test LLMCostCallback for tracking LLM costs from LangChain."""
    
    def setUp(self):
        """Create test fixtures."""
        self.config = Config()
        self.tracker = CostTracker(agent_id="test_agent", config=self.config)
        self.callback = LLMCostCallback(
            cost_tracker=self.tracker,
            model="openai/gpt-oss-20b",
            config=self.config
        )
    
    def test_callback_initialization(self):
        """Test LLMCostCallback initializes correctly."""
        self.assertEqual(self.callback.cost_tracker, self.tracker)
        self.assertEqual(self.callback.model, "openai/gpt-oss-20b")
        print("\n✓ LLMCostCallback initialized")
    
    def test_on_llm_end_calculates_cost(self):
        """Test callback calculates cost from token usage."""
        # Mock LLMResult with token usage
        mock_result = LLMResult(
            generations=[[ChatGeneration(message=HumanMessage(content="test"))]],
            llm_output={
                "token_usage": {
                    "prompt_tokens": 1000,
                    "completion_tokens": 500,
                    "total_tokens": 1500
                }
            }
        )
        
        # Call the callback
        self.callback.on_llm_end(response=mock_result)
        
        # Calculate expected cost
        expected_cost = (
            (1000 / 1000) * self.config.llm_input_price_per_1k +
            (500 / 1000) * self.config.llm_output_price_per_1k
        )
        
        self.assertAlmostEqual(self.tracker.total_llm_costs, expected_cost, places=6)
        
        print(f"\n✓ Callback tracked LLM cost: ${expected_cost:.6f}")
        print(f"  Input tokens: 1000, Output tokens: 500")
    
    def test_on_llm_end_multiple_calls(self):
        """Test callback tracks multiple LLM calls."""
        # First call
        mock_result_1 = LLMResult(
            generations=[[ChatGeneration(message=HumanMessage(content="test"))]],
            llm_output={"token_usage": {"prompt_tokens": 500, "completion_tokens": 300}}
        )
        self.callback.on_llm_end(response=mock_result_1)
        
        cost_after_first = self.tracker.total_llm_costs
        
        # Second call
        mock_result_2 = LLMResult(
            generations=[[ChatGeneration(message=HumanMessage(content="test"))]],
            llm_output={"token_usage": {"prompt_tokens": 800, "completion_tokens": 400}}
        )
        self.callback.on_llm_end(response=mock_result_2)
        
        self.assertGreater(self.tracker.total_llm_costs, cost_after_first)
        
        print(f"\n✓ Multiple LLM calls tracked: ${self.tracker.total_llm_costs:.6f}")
    
    def test_on_llm_end_no_token_usage(self):
        """Test callback handles missing token usage gracefully."""
        # Mock LLMResult without token usage
        mock_result = LLMResult(
            generations=[[ChatGeneration(message=HumanMessage(content="test"))]],
            llm_output={}
        )
        
        initial_cost = self.tracker.total_llm_costs
        self.callback.on_llm_end(response=mock_result)
        
        # Should not add cost if no token usage
        self.assertEqual(self.tracker.total_llm_costs, initial_cost)
        print("\n✓ Missing token usage handled gracefully")


class TestServiceExecutorCostTracking(unittest.TestCase):
    """Test cost tracking integration with ServiceExecutor."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures once."""
        cls.config = Config()
        
        # Skip if API key not configured
        if not cls.config.openrouter_api_key:
            raise unittest.SkipTest("OPENROUTER_API_KEY not configured")
        
        # Skip if PDF directory doesn't exist
        cls.pdf_dir = Path("utils/files")
        if not cls.pdf_dir.exists() or not list(cls.pdf_dir.glob("*.pdf")):
            raise unittest.SkipTest("No PDF files found in utils/files/")
    
    def setUp(self):
        """Create fresh instances for each test."""
        self.tracker = CostTracker(agent_id="test_agent", config=self.config)
        self.executor = ServiceExecutor(
            agent_id="test_agent",
            cost_tracker=self.tracker
        )
    
    def test_service_executor_tracks_costs(self):
        """Test that ServiceExecutor tracks LLM costs during review."""
        print("\n" + "="*80)
        print("Testing ServiceExecutor cost tracking")
        print("="*80)
        
        # Execute a simple service
        result = self.executor.perform_review(
            pdf_directory=str(self.pdf_dir),
            prompts=["What is the main topic of these papers?"],
            force_rebuild=False
        )
        
        # Verify service succeeded
        self.assertTrue(result["success"], f"Service failed: {result.get('error')}")
        
        # Verify LLM costs were tracked
        self.assertGreater(
            self.tracker.total_llm_costs,
            0.0,
            "LLM costs should be tracked during service execution"
        )
        
        print(f"\n✓ Service completed with cost tracking:")
        print(f"  Total LLM cost: ${self.tracker.total_llm_costs:.6f}")
    
    def test_multiple_prompts_accumulate_costs(self):
        """Test that multiple prompts accumulate costs correctly."""
        prompts = [
            "What are the main findings?",
            "Summarize the methodology.",
            "What are the limitations?"
        ]
        
        result = self.executor.perform_review(
            pdf_directory=str(self.pdf_dir),
            prompts=prompts,
            force_rebuild=False
        )
        
        self.assertTrue(result["success"])
        
        # Verify costs were tracked
        self.assertGreater(self.tracker.total_llm_costs, 0.0, "Should have tracked LLM costs")
        
        print(f"\n✓ Multiple prompts tracked:")
        print(f"  Prompts: {len(prompts)}")
        print(f"  Total cost: ${self.tracker.total_llm_costs:.6f}")


class TestCostSummaryLogging(unittest.TestCase):
    """Test cost summary logging functionality."""
    
    def test_log_summary(self):
        """Test that log_summary produces correct output."""
        config = Config()
        tracker = CostTracker(agent_id="test_agent", config=config)
        
        # Add some test data
        tracker.add_llm_cost(5.0, "reasoning")
        gas_price_wei = int(config.gas_price_gwei * 1e9)
        tracker.add_gas_cost(100000, gas_price_wei, "transaction")
        tracker.add_revenue(150.0, "service completed")
        
        # Capture log output
        with self.assertLogs(level='INFO') as log_context:
            tracker.log_summary()
        
        # Verify log contains expected information
        log_output = ' '.join(log_context.output)
        self.assertIn("Financial Summary", log_output)
        self.assertIn("LLM Costs", log_output)
        self.assertIn("Gas Costs", log_output)
        self.assertIn("Revenue", log_output)
        self.assertIn("NET BALANCE", log_output)
        
        print("\n✓ Cost summary logged successfully")


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)

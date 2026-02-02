"""
Tests for Adversarial Provider Strategies.

Unit tests for adversarial behavior controller and strategies.
Tests low-quality response generation, completion decisions, and bidding multipliers.

Run with: python agents/tests/test_adversarial_strategies.py
"""

import sys
from pathlib import Path

import unittest
import asyncio
import logging
from unittest.mock import AsyncMock, patch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from agents.adversarial_provider_agent.adversarial_strategies import (
    AdversarialStrategy,
    AdversarialBehaviorController
)
from agents.adversarial_provider_agent.orchestrator import AdversarialOrchestrator
from agents.config import Config
from unittest.mock import Mock


class TestAdversarialStrategiesUnit(unittest.TestCase):
    """Unit tests for adversarial strategies."""
    
    def test_strategy_enum_values(self):
        """Test that all strategy types are defined."""
        self.assertEqual(AdversarialStrategy.LOW_QUALITY.value, "low_quality")
        self.assertEqual(AdversarialStrategy.SELECTIVE_DEFECTION.value, "selective_defection")
        self.assertEqual(AdversarialStrategy.NON_COMPLETION.value, "non_completion")
        self.assertEqual(AdversarialStrategy.PRICE_MANIPULATION.value, "price_manipulation")
        
        print("\n✓ All adversarial strategy types defined")
    
    def test_controller_initialization(self):
        """Test controller initializes with correct defaults."""
        controller = AdversarialBehaviorController(
            strategy=AdversarialStrategy.LOW_QUALITY,
            config={}
        )
        
        self.assertEqual(controller.strategy, AdversarialStrategy.LOW_QUALITY)
        self.assertEqual(controller.auction_count, 0)
        self.assertEqual(controller.service_count, 0)
        self.assertEqual(controller.current_reputation, 50)
        
        print("\n✓ Controller initialized with correct defaults")


class TestLowQualityStrategy(unittest.TestCase):
    """Tests specifically for LOW_QUALITY strategy (E4.1)."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.controller = AdversarialBehaviorController(
            strategy=AdversarialStrategy.LOW_QUALITY,
            config={}
        )
    
    def test_should_complete_service(self):
        """Test LOW_QUALITY strategy completes service (doesn't refuse)."""
        result = self.controller.should_complete_service()
        
        self.assertTrue(result)
        print("\n✓ LOW_QUALITY strategy completes service")
    
    def test_should_use_low_quality_response(self):
        """Test LOW_QUALITY strategy always uses low-quality responses."""
        result = self.controller.should_use_low_quality_response()
        
        self.assertTrue(result)
        print("\n✓ LOW_QUALITY strategy always uses low-quality responses")
    
    def test_generate_low_quality_response(self):
        """Test low-quality response generation returns hardcoded responses."""
        prompt = "What are the main findings of this research?"
        
        response = self.controller.generate_low_quality_response(prompt)
        
        # Response should be a non-empty string
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)
        
        # Response should be one of the hardcoded bad responses
        bad_responses = [
            "I don't know.",
            "No information available.",
            "Unable to provide analysis.",
            "This topic is unclear.",
            "Insufficient data to answer.",
            "Request cannot be processed.",
            "Content not accessible.",
            "Analysis unavailable at this time.",
        ]
        self.assertIn(response, bad_responses)
        
        print(f"\n✓ Generated low-quality response: '{response}'")
    
    def test_generate_low_quality_response_rotation(self):
        """Test that responses rotate through different bad responses."""
        responses = set()
        
        # Generate multiple responses
        for _ in range(10):
            response = self.controller.generate_low_quality_response("test prompt")
            responses.add(response)
            self.controller.on_service_executed()
        
        # Should get multiple different responses
        self.assertGreater(len(responses), 1)
        print(f"\n✓ Response rotation works ({len(responses)} unique responses in 10 calls)")
    
    def test_get_bidding_multiplier(self):
        """Test LOW_QUALITY strategy uses normal bidding (1.0x)."""
        multiplier = self.controller.get_bidding_multiplier()
        
        self.assertEqual(multiplier, 1.0)
        print("\n✓ LOW_QUALITY strategy uses 1.0x bidding multiplier")
    
    def test_auction_tracking(self):
        """Test auction count tracking."""
        self.assertEqual(self.controller.auction_count, 0)
        
        self.controller.on_auction_detected()
        self.assertEqual(self.controller.auction_count, 1)
        
        self.controller.on_auction_detected()
        self.assertEqual(self.controller.auction_count, 2)
        
        print("\n✓ Auction count tracking works")
    
    def test_service_tracking(self):
        """Test service count tracking."""
        self.assertEqual(self.controller.service_count, 0)
        
        self.controller.on_service_executed()
        self.assertEqual(self.controller.service_count, 1)
        
        self.controller.on_service_executed()
        self.assertEqual(self.controller.service_count, 2)
        
        print("\n✓ Service count tracking works")
    
    def test_reputation_update(self):
        """Test reputation tracking updates correctly."""
        self.assertEqual(self.controller.current_reputation, 50)
        
        self.controller.update_reputation(30)
        self.assertEqual(self.controller.current_reputation, 30)
        
        self.controller.update_reputation(72)
        self.assertEqual(self.controller.current_reputation, 72)
        
        print("\n✓ Reputation tracking updates correctly")
    
    def test_get_stats(self):
        """Test statistics retrieval."""
        self.controller.on_auction_detected()
        self.controller.on_service_executed()
        self.controller.update_reputation(35)
        
        stats = self.controller.get_stats()
        
        self.assertEqual(stats["strategy"], "low_quality")
        self.assertEqual(stats["auction_count"], 1)
        self.assertEqual(stats["service_count"], 1)
        self.assertEqual(stats["current_reputation"], 35)
        
        print(f"\n✓ Statistics retrieved correctly: {stats}")


class TestNonCompletionStrategy(unittest.TestCase):
    """Tests for NON_COMPLETION strategy (E4.3)."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.controller = AdversarialBehaviorController(
            strategy=AdversarialStrategy.NON_COMPLETION,
            config={}
        )
    
    def test_should_complete_service(self):
        """Test NON_COMPLETION strategy refuses to complete service."""
        result = self.controller.should_complete_service()
        
        self.assertFalse(result)
        print("\n✓ NON_COMPLETION strategy refuses service completion")
    
    def test_get_bidding_multiplier(self):
        """Test NON_COMPLETION strategy uses very low bids (0.1x)."""
        multiplier = self.controller.get_bidding_multiplier()
        
        self.assertEqual(multiplier, 0.1)
        print("\n✓ NON_COMPLETION strategy uses 0.1x bidding multiplier")


class TestPriceManipulationStrategy(unittest.TestCase):
    """Tests for PRICE_MANIPULATION strategy (E4.4)."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.controller = AdversarialBehaviorController(
            strategy=AdversarialStrategy.PRICE_MANIPULATION,
            config={}
        )
    
    def test_should_use_low_quality_response(self):
        """Test PRICE_MANIPULATION strategy always uses low-quality responses."""
        result = self.controller.should_use_low_quality_response()
        
        self.assertTrue(result)
        print("\n✓ PRICE_MANIPULATION strategy uses low-quality responses")
    
    def test_get_bidding_multiplier(self):
        """Test PRICE_MANIPULATION strategy underbids (0.5x)."""
        multiplier = self.controller.get_bidding_multiplier()
        
        self.assertEqual(multiplier, 0.5)
        print("\n✓ PRICE_MANIPULATION strategy uses 0.5x bidding multiplier")


class TestSelectiveDefectionStrategy(unittest.TestCase):
    """Tests for SELECTIVE_DEFECTION strategy (E4.2 - bait-and-switch)."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.controller = AdversarialBehaviorController(
            strategy=AdversarialStrategy.SELECTIVE_DEFECTION,
            config={}
        )
    
    def test_bait_phase_uses_high_quality(self):
        """Test BAIT phase (reputation < 70) uses high-quality responses."""
        self.controller.current_reputation = 50
        
        result = self.controller.should_use_low_quality_response()
        
        self.assertFalse(result)
        print("\n✓ BAIT phase (rep=50) uses high-quality responses")
    
    def test_switch_phase_uses_low_quality(self):
        """Test SWITCH phase (reputation >= 70) uses low-quality responses."""
        self.controller.current_reputation = 70
        
        result = self.controller.should_use_low_quality_response()
        
        self.assertTrue(result)
        print("\n✓ SWITCH phase (rep=70) uses low-quality responses")
    
    def test_bait_phase_bidding(self):
        """Test BAIT phase uses low bids (0.8x) to win frequently."""
        self.controller.current_reputation = 50
        
        multiplier = self.controller.get_bidding_multiplier()
        
        self.assertEqual(multiplier, 0.8)
        print("\n✓ BAIT phase uses 0.8x bidding multiplier")
    
    def test_switch_phase_bidding(self):
        """Test SWITCH phase uses high bids (2.0x) to maximize profit."""
        self.controller.current_reputation = 75
        
        multiplier = self.controller.get_bidding_multiplier()
        
        self.assertEqual(multiplier, 2.0)
        print("\n✓ SWITCH phase uses 2.0x bidding multiplier")
    
    def test_reputation_threshold_boundary(self):
        """Test behavior at reputation threshold boundary (70)."""
        # Just below threshold - bait phase
        self.controller.current_reputation = 69
        self.assertFalse(self.controller.should_use_low_quality_response())
        self.assertEqual(self.controller.get_bidding_multiplier(), 0.8)
        
        # At threshold - switch phase
        self.controller.current_reputation = 70
        self.assertTrue(self.controller.should_use_low_quality_response())
        self.assertEqual(self.controller.get_bidding_multiplier(), 2.0)
        
        # Above threshold - switch phase
        self.controller.current_reputation = 85
        self.assertTrue(self.controller.should_use_low_quality_response())
        self.assertEqual(self.controller.get_bidding_multiplier(), 2.0)
        
        print("\n✓ Reputation threshold boundary (70) works correctly")


class TestCrossStrategyBehavior(unittest.TestCase):
    """Tests comparing behavior across different strategies."""
    
    def test_completion_behavior_varies(self):
        """Test that only NON_COMPLETION refuses service."""
        strategies_complete = {
            AdversarialStrategy.LOW_QUALITY: True,
            AdversarialStrategy.SELECTIVE_DEFECTION: True,
            AdversarialStrategy.PRICE_MANIPULATION: True,
            AdversarialStrategy.NON_COMPLETION: False,
        }
        
        for strategy, expected in strategies_complete.items():
            controller = AdversarialBehaviorController(strategy=strategy, config={})
            result = controller.should_complete_service()
            self.assertEqual(result, expected, f"{strategy.value} completion mismatch")
        
        print("\n✓ Completion behavior varies correctly across strategies")
    
    def test_bidding_multipliers_vary(self):
        """Test that bidding multipliers are strategy-specific."""
        expected_multipliers = {
            AdversarialStrategy.LOW_QUALITY: 1.0,
            AdversarialStrategy.PRICE_MANIPULATION: 0.5,
            AdversarialStrategy.NON_COMPLETION: 0.1,
        }
        
        for strategy, expected in expected_multipliers.items():
            controller = AdversarialBehaviorController(strategy=strategy, config={})
            multiplier = controller.get_bidding_multiplier()
            self.assertEqual(multiplier, expected, f"{strategy.value} multiplier mismatch")
        
        print("\n✓ Bidding multipliers vary correctly across strategies")


class TestReputationRefresh(unittest.TestCase):
    """Test reputation refresh functionality in adversarial orchestrator."""
    
    def test_reputation_refresh_updates_controller(self):
        """Test that reputation refresh updates the behavior controller."""
        mock_config = Mock(spec=Config)
        mock_config.agent_id = 123
        mock_config.llm_model = "gpt-4"
        mock_config.llm_temperature = 0.7
        
        orchestrator = AdversarialOrchestrator(
            config=mock_config,
            adversarial_strategy="selective_defection"
        )
        
        # Mock blockchain client
        mock_client = AsyncMock()
        mock_client.call_contract_method = AsyncMock(return_value=(5, 75))
        orchestrator.blockchain_handler.client = mock_client
        
        # Initial reputation should be 50 (default)
        self.assertEqual(orchestrator.current_reputation, 50)
        self.assertEqual(orchestrator.behavior_controller.current_reputation, 50)
        
        # Refresh reputation
        asyncio.run(orchestrator._refresh_reputation())
        
        # Verify reputation was updated
        self.assertEqual(orchestrator.current_reputation, 75)
        self.assertEqual(orchestrator.behavior_controller.current_reputation, 75)
        
        # Verify blockchain was queried
        mock_client.call_contract_method.assert_called_once_with(
            "ReputationRegistry",
            "getSummary",
            123,  # agent_id
            [],   # empty addresses
            b'\x00' * 32,  # zero bytes32
            b'\x00' * 32   # zero bytes32
        )
        
        print("\n✓ Reputation refresh updates controller correctly")
    
    def test_reputation_refresh_triggers_switch(self):
        """Test that crossing reputation threshold triggers bait-and-switch."""
        mock_config = Mock(spec=Config)
        mock_config.agent_id = 123
        mock_config.llm_model = "gpt-4"
        mock_config.llm_temperature = 0.7
        
        orchestrator = AdversarialOrchestrator(
            config=mock_config,
            adversarial_strategy="selective_defection"
        )
        
        # Mock blockchain client
        mock_client = AsyncMock()
        orchestrator.blockchain_handler.client = mock_client
        
        # Simulate reputation progression: 50 → 68 → 72
        mock_client.call_contract_method = AsyncMock(return_value=(3, 68))
        asyncio.run(orchestrator._refresh_reputation())
        self.assertEqual(orchestrator.current_reputation, 68)
        
        # Now cross the threshold
        mock_client.call_contract_method = AsyncMock(return_value=(4, 72))
        
        with patch('agents.adversarial_provider_agent.orchestrator.logger') as mock_logger:
            asyncio.run(orchestrator._refresh_reputation())
            
            # Verify warning was logged about threshold crossing
            warning_calls = [call for call in mock_logger.warning.call_args_list 
                           if 'BAIT-AND-SWITCH TRIGGERED' in str(call)]
            self.assertGreater(len(warning_calls), 0, "Should log warning when threshold crossed")
        
        self.assertEqual(orchestrator.current_reputation, 72)
        
        print("\n✓ Reputation threshold crossing triggers switch behavior")
    
    def test_reputation_refresh_handles_errors(self):
        """Test that reputation refresh handles blockchain errors gracefully."""
        mock_config = Mock(spec=Config)
        mock_config.agent_id = 123
        mock_config.llm_model = "gpt-4"
        mock_config.llm_temperature = 0.7
        
        orchestrator = AdversarialOrchestrator(
            config=mock_config,
            adversarial_strategy="selective_defection"
        )
        
        # Mock blockchain client to raise error
        mock_client = AsyncMock()
        mock_client.call_contract_method = AsyncMock(
            side_effect=Exception("Blockchain connection error")
        )
        orchestrator.blockchain_handler.client = mock_client
        
        # Refresh should not crash
        asyncio.run(orchestrator._refresh_reputation())
        
        # Reputation should remain at default
        self.assertEqual(orchestrator.current_reputation, 50)
        
        print("\n✓ Reputation refresh handles errors gracefully")


if __name__ == '__main__':
    print("\n" + "="*80)
    print("Running Adversarial Strategy Tests")
    print("="*80 + "\n")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestAdversarialStrategiesUnit))
    suite.addTests(loader.loadTestsFromTestCase(TestLowQualityStrategy))
    suite.addTests(loader.loadTestsFromTestCase(TestNonCompletionStrategy))
    suite.addTests(loader.loadTestsFromTestCase(TestPriceManipulationStrategy))
    suite.addTests(loader.loadTestsFromTestCase(TestSelectiveDefectionStrategy))
    suite.addTests(loader.loadTestsFromTestCase(TestCrossStrategyBehavior))
    suite.addTests(loader.loadTestsFromTestCase(TestReputationRefresh))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*80 + "\n")
    
    sys.exit(0 if result.wasSuccessful() else 1)

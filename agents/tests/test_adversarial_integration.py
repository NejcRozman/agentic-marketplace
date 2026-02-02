"""
Integration Tests for Adversarial Provider Strategies.

Tests full adversarial orchestrator workflow with real components:
- Blockchain interaction (bidding, service submission, reputation tracking)
- IPFS client (fetching service requirements, uploading results)
- LiteratureReviewAgent (for SELECTIVE_DEFECTION bait phase)
- Multi-agent scenarios (adversarial + honest providers)

Prerequisites:
1. Anvil running with contracts deployed
2. Agent registered and auction created
3. IPFS daemon running locally
4. Environment variables configured in agents/.env

Run with: python agents/tests/test_adversarial_integration.py
"""

import sys
from pathlib import Path
import unittest
import asyncio
import json
import os
import logging
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime
from tempfile import TemporaryDirectory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from agents.adversarial_provider_agent.orchestrator import (
    AdversarialOrchestrator,
    Job,
    JobStatus
)
from agents.adversarial_provider_agent.adversarial_strategies import (
    AdversarialStrategy,
    AdversarialBehaviorController
)
from agents.adversarial_provider_agent.blockchain_handler import BlockchainHandler
from agents.config import Config


def run_async(coro):
    """Helper to run async functions in sync tests."""
    return asyncio.get_event_loop().run_until_complete(coro)


class TestAdversarialOrchestratorUnit(unittest.TestCase):
    """Unit tests with mocked components for adversarial orchestrator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_config = Mock(spec=Config)
        self.mock_config.agent_id = 9999
        
        self.sample_auction_details = {
            "auction_id": 1,
            "buyer_address": "0x1234567890123456789012345678901234567890",
            "service_cid": "QmTestServiceCID123",
            "max_price": 100000000,
            "winning_bid": 50000000
        }
        
        self.sample_service_requirements = {
            "title": "Test Literature Review",
            "description": "Analyze research papers",
            "prompts": [
                "What is the main topic?",
                "What are the key findings?"
            ],
            "input_files_cid": "QmTestPDFCID456",
            "complexity": "medium"
        }
    
    def test_low_quality_orchestrator_initialization(self):
        """Test LOW_QUALITY orchestrator initializes correctly."""
        orchestrator = AdversarialOrchestrator(
            config=self.mock_config,
            adversarial_strategy="low_quality"
        )
        
        self.assertEqual(orchestrator.config, self.mock_config)
        self.assertEqual(orchestrator.behavior_controller.strategy, AdversarialStrategy.LOW_QUALITY)
        self.assertIsNotNone(orchestrator.blockchain_handler)
        self.assertIsNone(orchestrator.literature_agent)  # Not initialized for LOW_QUALITY
        self.assertIsNotNone(orchestrator.ipfs_client)
        self.assertEqual(orchestrator.active_jobs, {})
        
        print("\n✓ LOW_QUALITY orchestrator initialized correctly (no LLM agent)")
    
    def test_selective_defection_orchestrator_initialization(self):
        """Test SELECTIVE_DEFECTION orchestrator initializes LLM agent."""
        orchestrator = AdversarialOrchestrator(
            config=self.mock_config,
            adversarial_strategy="selective_defection"
        )
        
        self.assertEqual(orchestrator.behavior_controller.strategy, AdversarialStrategy.SELECTIVE_DEFECTION)
        self.assertIsNotNone(orchestrator.literature_agent)  # Initialized for SELECTIVE_DEFECTION
        
        print("\n✓ SELECTIVE_DEFECTION orchestrator initialized with LLM agent")
    
    def test_low_quality_uses_hardcoded_responses(self):
        """Test LOW_QUALITY strategy uses hardcoded responses instead of LLM."""
        orchestrator = AdversarialOrchestrator(
            config=self.mock_config,
            adversarial_strategy="low_quality"
        )
        orchestrator.blockchain_handler = AsyncMock()
        orchestrator.ipfs_client = AsyncMock()
        
        # Mock IPFS responses
        orchestrator.ipfs_client.get_json.return_value = self.sample_service_requirements
        
        # Start job
        run_async(orchestrator._start_job(self.sample_auction_details))
        
        job = orchestrator.active_jobs[1]
        self.assertEqual(job.status, JobStatus.WON)
        
        # Manually set up job for execution (skip fetch stages for unit test)
        job.service_requirements = self.sample_service_requirements
        job.prompts = self.sample_service_requirements["prompts"]
        job.pdf_directory = Path("/tmp/test")
        
        # Execute service
        run_async(orchestrator._execute_service(job))
        
        # Verify result contains hardcoded low-quality responses
        self.assertEqual(job.status, JobStatus.PROCESSING)
        self.assertIn("responses", job.result)
        for response_obj in job.result["responses"]:
            response = response_obj["response"]
            # Should be one of the hardcoded bad responses
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
        
        # Verify no LLM cost
        self.assertEqual(job.llm_cost, 0.0)
        
        print(f"\n✓ LOW_QUALITY generated hardcoded responses: {[r['response'] for r in job.result['responses']]}")
    
    def test_non_completion_refuses_service(self):
        """Test NON_COMPLETION strategy refuses to complete service."""
        orchestrator = AdversarialOrchestrator(
            config=self.mock_config,
            adversarial_strategy="non_completion"
        )
        orchestrator.blockchain_handler = AsyncMock()
        orchestrator.ipfs_client = AsyncMock()
        
        # Mock IPFS
        orchestrator.ipfs_client.get_json.return_value = self.sample_service_requirements
        
        # Start job
        run_async(orchestrator._start_job(self.sample_auction_details))
        
        job = orchestrator.active_jobs[1]
        
        # Set up job for execution
        job.service_requirements = self.sample_service_requirements
        job.prompts = self.sample_service_requirements["prompts"]
        job.pdf_directory = Path("/tmp/test")
        
        # Execute service - should refuse
        run_async(orchestrator._execute_service(job))
        
        # Verify service was refused
        self.assertEqual(job.status, JobStatus.REFUSED)
        self.assertIsNone(job.result)
        self.assertEqual(job.llm_cost, 0.0)
        
        # Verify nothing was submitted to blockchain
        orchestrator.blockchain_handler.submit_result.assert_not_called()
        
        print("\n✓ NON_COMPLETION strategy refused service")
    
    def test_selective_defection_bait_phase(self):
        """Test SELECTIVE_DEFECTION bait phase uses LLM (reputation < 70)."""
        orchestrator = AdversarialOrchestrator(
            config=self.mock_config,
            adversarial_strategy="selective_defection"
        )
        orchestrator.behavior_controller.current_reputation = 50  # BAIT phase
        orchestrator.blockchain_handler = AsyncMock()
        orchestrator.ipfs_client = AsyncMock()
        
        # Mock LLM agent
        orchestrator.literature_agent = Mock()
        orchestrator.literature_agent.perform_review = Mock(return_value={
            "success": True,
            "responses": [
                {"prompt": "What is the main topic?", "response": "High quality LLM answer 1"},
                {"prompt": "What are the key findings?", "response": "High quality LLM answer 2"}
            ],
            "total_cost": 0.05
        })
        
        # Mock IPFS
        orchestrator.ipfs_client.get_json.return_value = self.sample_service_requirements
        
        # Start job
        run_async(orchestrator._start_job(self.sample_auction_details))
        job = orchestrator.active_jobs[1]
        
        # Set up job for execution
        job.service_requirements = self.sample_service_requirements
        job.prompts = self.sample_service_requirements["prompts"]
        job.pdf_directory = Path("/tmp/test")
        
        # Execute service
        run_async(orchestrator._execute_service(job))
        
        # Verify LLM agent was called
        orchestrator.literature_agent.perform_review.assert_called_once()
        
        # Verify results are high-quality
        self.assertEqual(job.status, JobStatus.PROCESSING)
        self.assertIsNotNone(job.result)
        responses = job.result["responses"]
        
        # High-quality responses should be longer and more detailed
        for response_obj in responses:
            response = response_obj["response"]
            self.assertGreater(len(response), 20)  # Not just "I don't know"
        
        # Verify LLM cost was tracked
        self.assertEqual(job.llm_cost, 0.05)
        
        print(f"\n✓ SELECTIVE_DEFECTION BAIT phase used LLM: {[r['response'] for r in responses]}")
    
    def test_selective_defection_switch_phase(self):
        """Test SELECTIVE_DEFECTION switch phase uses hardcoded responses (reputation >= 70)."""
        orchestrator = AdversarialOrchestrator(
            config=self.mock_config,
            adversarial_strategy="selective_defection"
        )
        orchestrator.behavior_controller.current_reputation = 75  # SWITCH phase
        orchestrator.blockchain_handler = AsyncMock()
        orchestrator.ipfs_client = AsyncMock()
        
        # Mock LLM agent (should NOT be called)
        orchestrator.literature_agent = Mock()
        orchestrator.literature_agent.perform_review = Mock()
        
        # Mock IPFS
        orchestrator.ipfs_client.get_json.return_value = self.sample_service_requirements
        
        # Start job
        run_async(orchestrator._start_job(self.sample_auction_details))
        job = orchestrator.active_jobs[1]
        
        # Set up job for execution
        job.service_requirements = self.sample_service_requirements
        job.prompts = self.sample_service_requirements["prompts"]
        job.pdf_directory = Path("/tmp/test")
        
        # Execute service
        run_async(orchestrator._execute_service(job))
        
        # Verify LLM was NOT called (low-quality mode)
        orchestrator.literature_agent.perform_review.assert_not_called()
        
        # Verify results are low-quality hardcoded responses
        self.assertEqual(job.status, JobStatus.PROCESSING)
        responses = job.result["responses"]
        
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
        
        for response_obj in responses:
            response = response_obj["response"]
            self.assertIn(response, bad_responses)
        
        # Verify no LLM cost
        self.assertEqual(job.llm_cost, 0.0)
        
        print(f"\n✓ SELECTIVE_DEFECTION SWITCH phase used hardcoded responses: {[r['response'] for r in responses]}")


class TestAdversarialBlockchainIntegration(unittest.TestCase):
    """Integration tests with real blockchain (requires Anvil running)."""
    
    @classmethod
    def setUpClass(cls):
        """Check if blockchain is available."""
        try:
            from agents.config import config
            cls.config = config
            cls.blockchain_available = bool(config.rpc_url and config.reverse_auction_address)
        except Exception as e:
            logger.warning(f"Blockchain not available: {e}")
            cls.blockchain_available = False
    
    def setUp(self):
        """Skip tests if blockchain not available."""
        if not self.blockchain_available:
            self.skipTest("Blockchain not configured (need Anvil running + .env configured)")
    
    def test_bidding_multiplier_low_quality(self):
        """Test LOW_QUALITY strategy bids at normal price (1.0x)."""
        handler = BlockchainHandler(agent_id=9999)
        controller = AdversarialBehaviorController(
            strategy=AdversarialStrategy.LOW_QUALITY,
            config={}
        )
        
        multiplier = controller.get_bidding_multiplier()
        self.assertEqual(multiplier, 1.0)
        
        # If we bid 100 tokens normally, adversarial should bid 100
        base_bid = 100000000
        adversarial_bid = int(base_bid * multiplier)
        self.assertEqual(adversarial_bid, 100000000)
        
        print("\n✓ LOW_QUALITY bids at 1.0x (normal price)")
    
    def test_bidding_multiplier_non_completion(self):
        """Test NON_COMPLETION strategy underbids aggressively (0.1x)."""
        controller = AdversarialBehaviorController(
            strategy=AdversarialStrategy.NON_COMPLETION,
            config={}
        )
        
        multiplier = controller.get_bidding_multiplier()
        self.assertEqual(multiplier, 0.1)
        
        # If we bid 100 tokens normally, adversarial should bid 10
        base_bid = 100000000
        adversarial_bid = int(base_bid * multiplier)
        self.assertEqual(adversarial_bid, 10000000)
        
        print("\n✓ NON_COMPLETION bids at 0.1x (10% of normal price)")
    
    def test_bidding_multiplier_price_manipulation(self):
        """Test PRICE_MANIPULATION strategy underbids (0.5x)."""
        controller = AdversarialBehaviorController(
            strategy=AdversarialStrategy.PRICE_MANIPULATION,
            config={}
        )
        
        multiplier = controller.get_bidding_multiplier()
        self.assertEqual(multiplier, 0.5)
        
        base_bid = 100000000
        adversarial_bid = int(base_bid * multiplier)
        self.assertEqual(adversarial_bid, 50000000)
        
        print("\n✓ PRICE_MANIPULATION bids at 0.5x (50% of normal price)")
    
    def test_bidding_multiplier_selective_defection_bait(self):
        """Test SELECTIVE_DEFECTION bait phase underbids to win (0.8x)."""
        controller = AdversarialBehaviorController(
            strategy=AdversarialStrategy.SELECTIVE_DEFECTION,
            config={}
        )
        controller.current_reputation = 50  # BAIT phase
        
        multiplier = controller.get_bidding_multiplier()
        self.assertEqual(multiplier, 0.8)
        
        base_bid = 100000000
        adversarial_bid = int(base_bid * multiplier)
        self.assertEqual(adversarial_bid, 80000000)
        
        print("\n✓ SELECTIVE_DEFECTION bait phase bids at 0.8x (80% to win frequently)")
    
    def test_bidding_multiplier_selective_defection_switch(self):
        """Test SELECTIVE_DEFECTION switch phase overbids to maximize profit (2.0x)."""
        controller = AdversarialBehaviorController(
            strategy=AdversarialStrategy.SELECTIVE_DEFECTION,
            config={}
        )
        controller.current_reputation = 75  # SWITCH phase
        
        multiplier = controller.get_bidding_multiplier()
        self.assertEqual(multiplier, 2.0)
        
        base_bid = 100000000
        adversarial_bid = int(base_bid * multiplier)
        self.assertEqual(adversarial_bid, 200000000)
        
        print("\n✓ SELECTIVE_DEFECTION switch phase bids at 2.0x (200% to maximize profit)")


class TestMultiAgentAdversarialScenarios(unittest.TestCase):
    """Test scenarios with multiple agents (adversarial + honest)."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_config = Mock(spec=Config)
        self.mock_config.agent_id = 9999
        self.mock_config.llm_model = "gpt-4"
        self.mock_config.llm_temperature = 0.7
    
    def test_reputation_tracking_across_services(self):
        """Test that behavior controller tracks reputation updates correctly."""
        controller = AdversarialBehaviorController(
            strategy=AdversarialStrategy.LOW_QUALITY,
            config={}
        )
        
        # Simulate multiple service executions
        initial_rep = 50
        controller.update_reputation(initial_rep)
        
        # Execute services (should degrade reputation over time)
        controller.on_service_executed()
        controller.update_reputation(45)  # Downvoted
        
        controller.on_service_executed()
        controller.update_reputation(38)  # Downvoted again
        
        controller.on_service_executed()
        controller.update_reputation(30)  # Continues to degrade
        
        stats = controller.get_stats()
        self.assertEqual(stats["service_count"], 3)
        self.assertEqual(stats["current_reputation"], 30)
        self.assertLess(stats["current_reputation"], initial_rep)
        
        print(f"\n✓ Reputation tracking: {initial_rep} → {stats['current_reputation']} over {stats['service_count']} services")
    
    def test_selective_defection_reputation_threshold_crossing(self):
        """Test SELECTIVE_DEFECTION behavior changes when crossing reputation threshold."""
        controller = AdversarialBehaviorController(
            strategy=AdversarialStrategy.SELECTIVE_DEFECTION,
            config={}
        )
        
        # Start in BAIT phase
        controller.update_reputation(50)
        self.assertFalse(controller.should_use_low_quality_response())
        self.assertEqual(controller.get_bidding_multiplier(), 0.8)
        bait_phase_services = 0
        
        # Simulate building reputation through good service
        while controller.current_reputation < 70:
            controller.on_service_executed()
            bait_phase_services += 1
            # Simulate upvotes
            new_rep = min(70, controller.current_reputation + 5)
            controller.update_reputation(new_rep)
        
        # Now in SWITCH phase
        self.assertTrue(controller.should_use_low_quality_response())
        self.assertEqual(controller.get_bidding_multiplier(), 2.0)
        
        print(f"\n✓ SELECTIVE_DEFECTION threshold crossed: {bait_phase_services} services in bait phase, now in switch phase")
    
    def test_low_quality_detection_rate_simulation(self):
        """Simulate consumer detection of low-quality responses."""
        controller = AdversarialBehaviorController(
            strategy=AdversarialStrategy.LOW_QUALITY,
            config={}
        )
        
        # Simulate 20 service executions
        detections = 0
        services = 20
        
        for i in range(services):
            controller.on_service_executed()
            response = controller.generate_low_quality_response("test prompt")
            
            # Simulate consumer evaluation (low-quality responses are obvious)
            # In real scenario, consumer would downvote
            is_low_quality = len(response) < 50  # All hardcoded responses are short
            if is_low_quality:
                detections += 1
                # Downvote
                new_rep = max(0, controller.current_reputation - 3)
                controller.update_reputation(new_rep)
        
        detection_rate = detections / services
        self.assertGreaterEqual(detection_rate, 0.9)  # Should detect >90% of time
        self.assertLess(controller.current_reputation, 50)  # Reputation should degrade
        
        print(f"\n✓ LOW_QUALITY detection simulation: {detection_rate:.1%} detection rate, reputation: {controller.current_reputation}")
    
    def test_reputation_refresh_with_blockchain_mock(self):
        """Test reputation refresh in orchestrator with mocked blockchain."""
        orchestrator = AdversarialOrchestrator(
            config=self.mock_config,
            adversarial_strategy="selective_defection"
        )
        
        # Mock blockchain client to return increasing reputation
        mock_client = AsyncMock()
        reputation_values = [50, 60, 68, 72, 75]
        call_count = [0]
        
        async def mock_get_summary(*args, **kwargs):
            rep = reputation_values[min(call_count[0], len(reputation_values) - 1)]
            call_count[0] += 1
            return (call_count[0], rep)
        
        mock_client.call_contract_method = mock_get_summary
        orchestrator.blockchain_handler.client = mock_client
        
        # Initial state
        self.assertEqual(orchestrator.current_reputation, 50)
        self.assertEqual(orchestrator.behavior_controller.current_reputation, 50)
        
        # Refresh multiple times
        async def run_refreshes():
            for _ in range(5):
                await orchestrator._refresh_reputation()
                await asyncio.sleep(0.01)
        
        run_async(run_refreshes())
        
        # Reputation should have increased and crossed threshold
        self.assertGreaterEqual(orchestrator.current_reputation, 72)
        self.assertEqual(
            orchestrator.current_reputation,
            orchestrator.behavior_controller.current_reputation
        )
        
        # Behavior should have switched (high rep = switch phase)
        self.assertTrue(orchestrator.behavior_controller.should_use_low_quality_response())
        self.assertEqual(orchestrator.behavior_controller.get_bidding_multiplier(), 2.0)
        
        print(f"\n✓ Reputation refresh integration: {call_count[0]} refreshes, final rep: {orchestrator.current_reputation}, switch triggered")


if __name__ == '__main__':
    print("\n" + "="*80)
    print("Running Adversarial Integration Tests")
    print("="*80 + "\n")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestAdversarialOrchestratorUnit))
    suite.addTests(loader.loadTestsFromTestCase(TestAdversarialBlockchainIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestMultiAgentAdversarialScenarios))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*80)
    print("Integration Test Summary")
    print("="*80)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print("="*80 + "\n")
    
    sys.exit(0 if result.wasSuccessful() else 1)

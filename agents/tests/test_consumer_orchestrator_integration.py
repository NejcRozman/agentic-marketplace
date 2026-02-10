"""
Tests for Consumer Orchestrator - Integration Tests.

Integration tests use real components:
- Real LLM (ServiceGenerator and ServiceEvaluator)
- Real IPFS (Pinata)
- Real blockchain (Anvil with deployed contracts)

Prerequisites:
1. Anvil running locally on port 8545
2. Contracts deployed (ReverseAuction, IdentityRegistry, ReputationRegistry)
3. Consumer agent registered in IdentityRegistry
4. At least one provider agent registered
5. agents/.env configured with:
   - GOOGLE_API_KEY=<key>
   - PINATA_JWT or (PINATA_API_KEY + PINATA_API_SECRET)
   - BLOCKCHAIN_RPC_URL=http://localhost:8545
   - BLOCKCHAIN_REVERSE_AUCTION_ADDRESS=0x...
   - BLOCKCHAIN_IDENTITY_REGISTRY_ADDRESS=0x...
   - BLOCKCHAIN_REPUTATION_REGISTRY_ADDRESS=0x...
   - CONSUMER_AGENT_ID=<id>
   - ELIGIBLE_PROVIDERS=<id1>,<id2>
6. Test PDFs with abstracts in utils/files/

Run with: python agents/tests/test_consumer_orchestrator_integration.py
"""

import sys
from pathlib import Path


import unittest
import asyncio
import json
import logging
from tempfile import TemporaryDirectory
from datetime import datetime
from unittest.mock import patch, AsyncMock, Mock

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from agents.consumer_agent.consumer_orchestrator import (
    Consumer,
    AuctionTracker,
    AuctionStatus
)
from agents.config import config


def run_async(coro):
    """Helper to run async functions in sync tests."""
    return asyncio.get_event_loop().run_until_complete(coro)


class TestConsumerIntegrationSetup(unittest.TestCase):
    """Verify integration test prerequisites."""
    
    def test_config_loaded(self):
        """Verify all required configuration is loaded."""
        self.assertIsNotNone(config.google_api_key, "GOOGLE_API_KEY not set")
        self.assertTrue(
            config.pinata_jwt or (config.pinata_api_key and config.pinata_api_secret),
            "PINATA credentials not set"
        )
        self.assertIsNotNone(config.rpc_url, "BLOCKCHAIN_RPC_URL not set")
        self.assertIsNotNone(config.reverse_auction_address, "BLOCKCHAIN_REVERSE_AUCTION_ADDRESS not set")
        self.assertIsNotNone(config.consumer_agent_id, "CONSUMER_AGENT_ID not set")
        self.assertTrue(len(config.eligible_providers) > 0, "ELIGIBLE_PROVIDERS not set")
        
        print("\n✓ All configuration loaded:")
        print(f"  Consumer Agent ID: {config.consumer_agent_id}")
        print(f"  Eligible Providers: {config.eligible_providers}")
        print(f"  RPC URL: {config.rpc_url}")


class TestLoadAndCreateAuction(unittest.TestCase):
    """Test service loading with real LLM and auction creation."""
    
    @unittest.skipUnless(
        (Path(__file__).parent.parent.parent / "utils" / "files").exists(),
        "Test PDF directory not found in utils/files/"
    )
    def test_integration_load_services_and_create_auction(self):
        """Integration test: Load services with real LLM, create auction with blockchain."""
        async def _test():
            # Skip if API keys not configured
            if not config.google_api_key or not (
                config.pinata_jwt or (config.pinata_api_key and config.pinata_api_secret)
            ):
                print("\n⏭️  Skipping: API keys not configured")
                return
            
            print("\n🔄 Loading services with real LLM and IPFS...")
            
            consumer = Consumer(config)
            
            # Initialize with PDF directory
            test_dir = Path(__file__).parent.parent.parent / "utils" / "files"
            await consumer.initialize(pdf_dir=test_dir)
            
            # Verify services loaded
            self.assertGreater(len(consumer.available_services), 0)
            
            print(f"\n✅ Loaded {len(consumer.available_services)} services with real LLM")
            
            # Verify service structure
            service = consumer.available_services[0]
            self.assertIn("service_cid", service)
            self.assertIn("pdf_cid", service)
            self.assertIn("title", service)
            self.assertIn("prompts", service)
            self.assertTrue(service["service_cid"].startswith("Qm"))
            self.assertTrue(service["pdf_cid"].startswith("Qm"))
            
            print(f"  Service Title: {service['title']}")
            print(f"  Service CID: {service['service_cid']}")
            print(f"  Prompts: {len(service['prompts'])}")
            
            # Now create auction with real blockchain
            print("\n🔄 Creating auction on blockchain...")
            
            auction_id = await consumer.create_auction(
                service_index=0,
                max_budget=100_000_000,  # 100 USDC
                duration=1800,  # 30 minutes
                eligible_providers=config.eligible_providers
            )
            
            self.assertIsInstance(auction_id, int)
            self.assertGreater(auction_id, 0)
            self.assertIn(auction_id, consumer.active_auctions)
            
            tracker = consumer.active_auctions[auction_id]
            self.assertEqual(tracker.status, AuctionStatus.CREATED)
            self.assertEqual(tracker.service_cid, service["service_cid"])
            
            print(f"\n✅ Auction created on blockchain:")
            print(f"  Auction ID: {auction_id}")
            print(f"  Service CID: {tracker.service_cid}")
            print(f"  Max Budget: {tracker.max_budget / 1_000_000} USDC")
            print(f"  Duration: {tracker.duration}s")
            print(f"  Eligible Providers: {tracker.eligible_providers}")
        
        run_async(_test())


class TestEvaluateRealResult(unittest.TestCase):
    """Test evaluation with real LLM."""
    
    def test_integration_evaluate_with_real_llm(self):
        """Integration test: Evaluate result using real LLM."""
        async def _test():
            if not config.google_api_key:
                print("\n⏭️  Skipping: GOOGLE_API_KEY not configured")
                return
            
            print("\n🔄 Evaluating result with real LLM...")
            
            consumer = Consumer(config)
            await consumer.initialize()
            
            # Mock IPFS to return service requirements
            consumer.ipfs_client.fetch_json = AsyncMock(return_value={
                "title": "Machine Learning Literature Review",
                "description": "Comprehensive review of ML papers",
                "prompts": [
                    "What are the main machine learning paradigms?",
                    "Explain supervised vs unsupervised learning.",
                    "What are neural networks?"
                ],
                "quality_criteria": {
                    "completeness": "All prompts answered comprehensively",
                    "depth": "Detailed explanations with examples",
                    "clarity": "Clear and well-structured"
                }
            })
            
            # High quality result
            result = {
                "success": True,
                "responses": [
                    {
                        "prompt": "What are the main machine learning paradigms?",
                        "response": "The main machine learning paradigms include supervised learning (learning from labeled data), unsupervised learning (finding patterns in unlabeled data), reinforcement learning (learning through trial and error with rewards), and semi-supervised learning (combining labeled and unlabeled data). Each paradigm has specific use cases and algorithms."
                    },
                    {
                        "prompt": "Explain supervised vs unsupervised learning.",
                        "response": "Supervised learning uses labeled training data where inputs are paired with correct outputs (e.g., spam detection, image classification). The algorithm learns to map inputs to outputs. Unsupervised learning works with unlabeled data to discover hidden patterns (e.g., clustering customers, dimensionality reduction). It doesn't have predefined correct answers."
                    },
                    {
                        "prompt": "What are neural networks?",
                        "response": "Neural networks are computational models inspired by biological neural networks. They consist of interconnected nodes (neurons) organized in layers: input layer, hidden layers, and output layer. Each connection has a weight that's adjusted during training. They excel at pattern recognition, classification, and complex non-linear mappings. Deep learning uses neural networks with many hidden layers."
                    }
                ]
            }
            
            tracker = AuctionTracker(
                auction_id=999,
                status=AuctionStatus.COMPLETED,
                created_at=datetime.now(),
                service_cid="QmTestServiceCID",
                max_budget=100_000_000,
                duration=1800,
                eligible_providers=[1],
                result=result
            )
            
            # Mock feedback submission to avoid blockchain interaction
            consumer._submit_feedback = AsyncMock()
            
            # Evaluate with real LLM
            await consumer._evaluate_result(tracker)
            
            # Verify evaluation completed
            self.assertEqual(tracker.status, AuctionStatus.EVALUATED)
            self.assertIsNotNone(tracker.evaluation)
            self.assertIn("rating", tracker.evaluation)
            self.assertIn("quality_scores", tracker.evaluation)
            
            rating = tracker.evaluation["rating"]
            self.assertIsInstance(rating, int)
            self.assertGreaterEqual(rating, 0)
            self.assertLessEqual(rating, 100)
            
            # High quality result should get good rating
            self.assertGreater(rating, 60, "High quality result should get >60 rating")
            
            print(f"\n✅ Real LLM evaluation completed:")
            print(f"  Rating: {rating}/100")
            print(f"  Quality Scores: {tracker.evaluation['quality_scores']}")
            
            # Verify feedback submission was triggered
            consumer._submit_feedback.assert_called_once_with(tracker)
        
        run_async(_test())


class TestMonitoringWithBlockchain(unittest.TestCase):
    """Test monitoring with real blockchain."""
    
    def test_integration_monitor_auction_state_changes(self):
        """Integration test: Monitor auction state changes on real blockchain."""
        async def _test():
            print("\n🔄 Testing auction monitoring with blockchain...")
            
            consumer = Consumer(config)
            await consumer.initialize()
            
            # Create a real auction first
            # Mock service loading to avoid regenerating
            consumer.available_services = [{
                "service_cid": "QmTestService123",
                "pdf_cid": "QmPDF123",
                "pdf_name": "test.pdf",
                "title": "Test Service",
                "prompts": ["Q1"]
            }]
            
            auction_id = await consumer.create_auction(
                service_index=0,
                max_budget=50_000_000,
                duration=1800
            )
            
            print(f"  Created auction {auction_id}")
            
            # Monitor - should detect CREATED → ACTIVE or remain CREATED
            await consumer.monitor_auctions()
            
            tracker = consumer.active_auctions[auction_id]
            
            # Should be either CREATED or ACTIVE (depends on provider bids)
            self.assertIn(tracker.status, [AuctionStatus.CREATED, AuctionStatus.ACTIVE])
            
            print(f"  Auction status after monitoring: {tracker.status}")
            print(f"\n✅ Monitoring successfully queried blockchain state")
        
        run_async(_test())


class TestRunLoopIntegration(unittest.TestCase):
    """Test run loop with mocked iterations."""
    
    @patch('asyncio.sleep')
    def test_integration_run_loop_iterations(self, mock_sleep):
        """Test run loop executes N iterations correctly."""
        async def _test():
            mock_sleep.return_value = None  # No actual sleeping
            
            consumer = Consumer(config)
            await consumer.initialize()
            
            # Track monitor calls
            monitor_call_count = 0
            original_monitor = consumer.monitor_auctions
            
            async def mock_monitor():
                nonlocal monitor_call_count
                monitor_call_count += 1
                await original_monitor()
                # Stop after 3 iterations
                if monitor_call_count >= 3:
                    consumer.stop()
            
            consumer.monitor_auctions = mock_monitor
            
            print("\n🔄 Running consumer loop for 3 iterations...")
            
            # Run the loop
            await consumer.run()
            
            # Verify monitor was called 3 times
            self.assertEqual(monitor_call_count, 3)
            self.assertFalse(consumer.running)
            
            print(f"\n✅ Run loop completed 3 monitoring iterations")
            print(f"  Monitor called: {monitor_call_count} times")
        
        run_async(_test())


class TestMultipleServicesSequential(unittest.TestCase):
    """Test creating multiple auctions with real services."""
    
    @unittest.skipUnless(
        (Path(__file__).parent.parent.parent / "utils" / "files").exists(),
        "Test PDF directory not found"
    )
    def test_integration_multiple_auctions_from_real_services(self):
        """Integration test: Create multiple auctions from real generated services."""
        async def _test():
            if not config.google_api_key or not (
                config.pinata_jwt or (config.pinata_api_key and config.pinata_api_secret)
            ):
                print("\n⏭️  Skipping: API keys not configured")
                return
            
            print("\n🔄 Creating multiple auctions from real services...")
            
            consumer = Consumer(config)
            test_dir = Path(__file__).parent.parent.parent / "utils" / "files"
            
            await consumer.initialize(pdf_dir=test_dir)
            
            num_services = len(consumer.available_services)
            self.assertGreater(num_services, 0)
            
            print(f"  Loaded {num_services} services")
            
            # Create auctions for first 2 services (or all if less than 2)
            auctions_to_create = min(2, num_services)
            auction_ids = []
            
            for i in range(auctions_to_create):
                auction_id = await consumer.create_auction(
                    service_index=None,  # Use next in order
                    max_budget=50_000_000,
                    duration=1800
                )
                auction_ids.append(auction_id)
                
                # Verify correct service used
                tracker = consumer.active_auctions[auction_id]
                expected_cid = consumer.available_services[i]["service_cid"]
                self.assertEqual(tracker.service_cid, expected_cid)
                
                print(f"  Auction {auction_id}: {consumer.available_services[i]['title']}")
            
            # Verify service_index incremented correctly
            self.assertEqual(consumer.service_index, auctions_to_create)
            
            print(f"\n✅ Created {auctions_to_create} auctions sequentially")
            print(f"  Service index: {consumer.service_index}/{num_services}")
        
        run_async(_test())


if __name__ == "__main__":
    print("=" * 70)
    print("CONSUMER ORCHESTRATOR INTEGRATION TESTS")
    print("=" * 70)
    print()
    print("These tests use:")
    print("  - Real LLM (Google Gemini)")
    print("  - Real IPFS (Pinata)")
    print("  - Real blockchain (Anvil)")
    print()
    print("Prerequisites:")
    print("  1. Anvil running on localhost:8545")
    print("  2. Contracts deployed")
    print("  3. Agents registered")
    print("  4. API keys configured in .env")
    print("=" * 70)
    print()
    
    unittest.main(verbosity=2)

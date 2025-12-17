"""
Tests for ConsumerBlockchainHandler.

These tests require:
1. Anvil running locally on port 8545
2. Contracts deployed (ReverseAuction, IdentityRegistry, ReputationRegistry)
3. Consumer agent registered in IdentityRegistry
4. At least one provider agent registered (for eligible_agent_ids)
5. agents/.env configured with:
   - BLOCKCHAIN_RPC_URL=http://localhost:8545
   - BLOCKCHAIN_PRIVATE_KEY=0x...
   - BLOCKCHAIN_REVERSE_AUCTION_ADDRESS=0x...
   - BLOCKCHAIN_IDENTITY_REGISTRY_ADDRESS=0x...
   - BLOCKCHAIN_REPUTATION_REGISTRY_ADDRESS=0x...
   - CONSUMER_AGENT_ID=<id>
   - ELIGIBLE_PROVIDERS=<id1>,<id2>

Run with: python agents/tests/test_consumer_blockchain_handler.py
"""

import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import unittest
import asyncio
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from agents.config import config
from agents.consumer_agent.blockchain_handler import ConsumerBlockchainHandler
from agents.infrastructure.blockchain_client import BlockchainClient


def run_async(coro):
    """Helper to run async functions in sync tests."""
    return asyncio.get_event_loop().run_until_complete(coro)


class TestConsumerBlockchainHandlerSetup(unittest.TestCase):
    """Test ConsumerBlockchainHandler initialization and configuration."""
    
    def test_config_loaded(self):
        """Verify configuration is loaded from .env."""
        self.assertIsNotNone(config.rpc_url, "BLOCKCHAIN_RPC_URL not set")
        self.assertIsNotNone(config.reverse_auction_address, "BLOCKCHAIN_REVERSE_AUCTION_ADDRESS not set")
        self.assertIsNotNone(config.identity_registry_address, "BLOCKCHAIN_IDENTITY_REGISTRY_ADDRESS not set")
        self.assertIsNotNone(config.reputation_registry_address, "BLOCKCHAIN_REPUTATION_REGISTRY_ADDRESS not set")
        self.assertIsNotNone(config.consumer_agent_id, "CONSUMER_AGENT_ID not set")
        self.assertTrue(len(config.eligible_providers) > 0, "ELIGIBLE_PROVIDERS not set")
        print(f"\n✓ Config loaded:")
        print(f"  RPC URL: {config.rpc_url}")
        print(f"  ReverseAuction: {config.reverse_auction_address}")
        print(f"  Consumer Agent ID: {config.consumer_agent_id}")
        print(f"  Eligible Providers: {config.eligible_providers}")
    
    def test_handler_initialization(self):
        """Test ConsumerBlockchainHandler initializes correctly."""
        handler = ConsumerBlockchainHandler()
        
        self.assertEqual(handler.consumer_agent_id, config.consumer_agent_id)
        self.assertIsNotNone(handler.client)
        self.assertFalse(handler._initialized)
        print(f"\n✓ Handler created for consumer agent {config.consumer_agent_id}")
    
    def test_handler_initialize(self):
        """Test handler initialization with contracts."""
        async def _test():
            handler = ConsumerBlockchainHandler()
            result = await handler.initialize()
            
            self.assertTrue(result)
            self.assertTrue(handler._initialized)
            print(f"\n✓ Handler initialized successfully")
            
            # Test idempotent initialization
            result2 = await handler.initialize()
            self.assertTrue(result2)
        
        run_async(_test())


class TestCreateAuction(unittest.TestCase):
    """Test auction creation functionality."""
    
    def setUp(self):
        """Create handler instance."""
        self.handler = ConsumerBlockchainHandler()
    
    def test_create_auction_success(self):
        """Test creating a new auction successfully."""
        async def _test():
            await self.handler.initialize()
            
            # Create auction
            result = await self.handler.create_auction(
                service_cid="QmTest123",
                max_price=1000000000000000000,  # 1 ETH in wei
                duration=3600,  # 1 hour
                eligible_agent_ids=config.eligible_providers,
                reputation_weight=50
            )
            
            self.assertIsNone(result['error'])
            self.assertIsNotNone(result['auction_id'])
            self.assertIsNotNone(result['tx_hash'])
            self.assertIsInstance(result['auction_id'], int)
            self.assertGreater(result['auction_id'], 0)
            
            print(f"\n✓ Auction created:")
            print(f"  Auction ID: {result['auction_id']}")
            print(f"  Tx Hash: {result['tx_hash']}")
            
            return result['auction_id']
        
        auction_id = run_async(_test())
        self.auction_id = auction_id  # Store for other tests
    
    def test_create_auction_with_defaults(self):
        """Test creating auction with default eligible providers from config."""
        async def _test():
            await self.handler.initialize()
            
            # Create auction without specifying eligible_agent_ids
            result = await self.handler.create_auction(
                service_cid="QmTest456",
                max_price=500000000000000000,  # 0.5 ETH
                duration=1800  # 30 minutes
            )
            
            self.assertIsNone(result['error'])
            self.assertIsNotNone(result['auction_id'])
            print(f"\n✓ Auction created with default eligible providers")
            print(f"  Auction ID: {result['auction_id']}")
        
        run_async(_test())
    
    def test_create_auction_without_initialization(self):
        """Test that create_auction auto-initializes if needed."""
        async def _test():
            handler = ConsumerBlockchainHandler()
            # Don't call initialize()
            
            result = await handler.create_auction(
                service_cid="QmTest789",
                max_price=2000000000000000000,
                duration=7200
            )
            
            self.assertIsNone(result['error'])
            self.assertIsNotNone(result['auction_id'])
            self.assertTrue(handler._initialized)
            print(f"\n✓ Auto-initialization works")
        
        run_async(_test())


class TestGetAuctionStatus(unittest.TestCase):
    """Test auction status retrieval."""
    
    def setUp(self):
        """Create handler and auction."""
        self.handler = ConsumerBlockchainHandler()
        
        async def _setup():
            await self.handler.initialize()
            result = await self.handler.create_auction(
                service_cid="QmStatusTest",
                max_price=1000000000000000000,
                duration=3600,
                reputation_weight=50
            )
            return result['auction_id']
        
        self.auction_id = run_async(_setup())
    
    def test_get_auction_status_active(self):
        """Test getting status of an active auction."""
        async def _test():
            status = await self.handler.get_auction_status(self.auction_id)
            
            self.assertIsNone(status.get('error'))
            self.assertEqual(status['id'], self.auction_id)
            self.assertEqual(status['service_cid'], "QmStatusTest")
            self.assertEqual(status['max_price'], 1000000000000000000)
            self.assertEqual(status['duration'], 3600)
            self.assertEqual(status['reputation_weight'], 50)
            self.assertTrue(status['active'])
            self.assertFalse(status['completed'])
            self.assertEqual(status['winning_agent_id'], 0)
            self.assertIn('buyer', status)
            self.assertIn('start_time', status)
            
            print(f"\n✓ Auction status retrieved:")
            print(f"  ID: {status['id']}")
            print(f"  Active: {status['active']}")
            print(f"  Completed: {status['completed']}")
            print(f"  Buyer: {status['buyer']}")
        
        run_async(_test())
    
    def test_get_auction_status_invalid_id(self):
        """Test getting status with invalid auction ID."""
        async def _test():
            # Use a very high ID that doesn't exist
            status = await self.handler.get_auction_status(999999)
            
            # Should return error or default values
            self.assertIsNotNone(status.get('error'))
            print(f"\n✓ Invalid auction ID handled correctly")
        
        run_async(_test())


class TestEndAuction(unittest.TestCase):
    """Test auction ending functionality."""
    
    def setUp(self):
        """Create handler and auction."""
        self.handler = ConsumerBlockchainHandler()
        
        async def _setup():
            await self.handler.initialize()
            result = await self.handler.create_auction(
                service_cid="QmEndTest",
                max_price=1000000000000000000,
                duration=60,  # 1 minute for faster testing
                reputation_weight=50
            )
            return result['auction_id']
        
        self.auction_id = run_async(_setup())
    
    def test_end_auction_before_expiry(self):
        """Test ending auction before natural expiry."""
        async def _test():
            # Wait a few seconds for potential bids
            await asyncio.sleep(5)
            
            result = await self.handler.end_auction(self.auction_id)
            
            self.assertIsNone(result['error'])
            self.assertIsNotNone(result['tx_hash'])
            # winning_agent_id and winning_bid may be 0 if no bids
            self.assertIn('winning_agent_id', result)
            self.assertIn('winning_bid', result)
            
            print(f"\n✓ Auction ended:")
            print(f"  Tx Hash: {result['tx_hash']}")
            print(f"  Winner: {result['winning_agent_id']}")
            print(f"  Winning Bid: {result['winning_bid']}")
            
            # Verify auction is now completed
            status = await self.handler.get_auction_status(self.auction_id)
            self.assertFalse(status['active'])
            self.assertTrue(status['completed'])
        
        run_async(_test())
    
    def test_end_auction_invalid_id(self):
        """Test ending auction with invalid ID."""
        async def _test():
            result = await self.handler.end_auction(999999)
            
            self.assertIsNotNone(result['error'])
            self.assertIsNone(result['tx_hash'])
            print(f"\n✓ Invalid auction end handled correctly")
        
        run_async(_test())


class TestGetFeedbackAuth(unittest.TestCase):
    """Test feedback authorization retrieval."""
    
    def setUp(self):
        """Create handler."""
        self.handler = ConsumerBlockchainHandler()
    
    def test_get_feedback_auth_not_found(self):
        """Test querying for non-existent feedback auth."""
        async def _test():
            await self.handler.initialize()
            
            # Query for auction that doesn't have FeedbackAuthProvided event
            feedback_auth = await self.handler.get_feedback_auth(
                auction_id=999999,
                from_block=0,
                lookback_blocks=100
            )
            
            self.assertIsNone(feedback_auth)
            print(f"\n✓ No feedback auth found (expected)")
        
        run_async(_test())
    
    def test_get_feedback_auth_with_from_block(self):
        """Test querying with specific from_block."""
        async def _test():
            await self.handler.initialize()
            
            # Get current block
            current_block = await self.handler.client.get_block_number()
            
            # Query from recent block
            feedback_auth = await self.handler.get_feedback_auth(
                auction_id=1,
                from_block=max(0, current_block - 100),
                lookback_blocks=50
            )
            
            # May or may not find auth depending on test state
            print(f"\n✓ Queried from block {current_block - 100}")
            print(f"  Found: {feedback_auth is not None}")
        
        run_async(_test())
    
    def test_get_feedback_auth_default_lookback(self):
        """Test querying with default lookback_blocks."""
        async def _test():
            await self.handler.initialize()
            
            # Query without specifying from_block (uses lookback)
            feedback_auth = await self.handler.get_feedback_auth(
                auction_id=1,
                lookback_blocks=1000
            )
            
            print(f"\n✓ Queried with default lookback")
            print(f"  Found: {feedback_auth is not None}")
        
        run_async(_test())


class TestSubmitFeedback(unittest.TestCase):
    """Test feedback submission functionality."""
    
    def setUp(self):
        """Create handler."""
        self.handler = ConsumerBlockchainHandler()
    
    def test_submit_feedback_rating_validation(self):
        """Test that invalid ratings are rejected."""
        async def _test():
            await self.handler.initialize()
            
            # Mock feedback auth (32 bytes)
            mock_auth = b'\x00' * 32
            
            # Test rating too high
            result = await self.handler.submit_feedback(
                auction_id=1,
                agent_id=1,
                rating=101,  # Invalid: >100
                feedback_text="Test",
                feedback_auth=mock_auth
            )
            
            self.assertIsNotNone(result['error'])
            self.assertIn("0-100", result['error'])
            print(f"\n✓ Rating >100 rejected")
            
            # Test negative rating
            result2 = await self.handler.submit_feedback(
                auction_id=1,
                agent_id=1,
                rating=-1,  # Invalid: negative
                feedback_text="Test",
                feedback_auth=mock_auth
            )
            
            self.assertIsNotNone(result2['error'])
            print(f"✓ Negative rating rejected")
        
        run_async(_test())
    
    def test_submit_feedback_valid_rating(self):
        """Test that valid ratings (0-100) are accepted."""
        async def _test():
            await self.handler.initialize()
            
            # Create mock feedback auth (would come from FeedbackAuthProvided event)
            # In real scenario, this would be signed by the contract
            mock_auth = b'\x01' * 100  # Simulate signed auth
            
            # Test boundary values
            for rating in [0, 50, 100]:
                result = await self.handler.submit_feedback(
                    auction_id=1,
                    agent_id=config.eligible_providers[0],
                    rating=rating,
                    feedback_text=f"Test feedback with rating {rating}",
                    feedback_auth=mock_auth
                )
                
                # May fail due to invalid auth signature, but should pass validation
                if result['error']:
                    # Check it's not a validation error
                    self.assertNotIn("0-100", result['error'])
                    print(f"\n✓ Rating {rating} passed validation (failed on auth/execution)")
                else:
                    print(f"\n✓ Rating {rating} submitted successfully")
                    print(f"  Tx Hash: {result['tx_hash']}")
        
        run_async(_test())
    
    def test_submit_feedback_with_tags(self):
        """Test submitting feedback with custom tags."""
        async def _test():
            await self.handler.initialize()
            
            mock_auth = b'\x01' * 100
            
            # Create custom tags (bytes32)
            tag1 = b'quality' + b'\x00' * 25  # Pad to 32 bytes
            tag2 = b'speed' + b'\x00' * 27
            
            result = await self.handler.submit_feedback(
                auction_id=1,
                agent_id=config.eligible_providers[0],
                rating=85,
                feedback_text="Good service",
                feedback_auth=mock_auth,
                tag1=tag1,
                tag2=tag2
            )
            
            # Check validation passed (may fail on execution)
            if result['error']:
                self.assertNotIn("0-100", result['error'])
            print(f"\n✓ Feedback with tags processed")
        
        run_async(_test())
    
    def test_submit_feedback_with_uri(self):
        """Test submitting feedback with feedback_uri."""
        async def _test():
            await self.handler.initialize()
            
            mock_auth = b'\x01' * 100
            
            result = await self.handler.submit_feedback(
                auction_id=1,
                agent_id=config.eligible_providers[0],
                rating=90,
                feedback_text="Excellent work",
                feedback_auth=mock_auth,
                feedback_uri="ipfs://QmFeedbackHash123"
            )
            
            if result['error']:
                self.assertNotIn("0-100", result['error'])
            print(f"\n✓ Feedback with URI processed")
        
        run_async(_test())


class TestEventExtraction(unittest.TestCase):
    """Test event extraction methods."""
    
    def setUp(self):
        """Create handler and auction."""
        self.handler = ConsumerBlockchainHandler()
        
        async def _setup():
            await self.handler.initialize()
            result = await self.handler.create_auction(
                service_cid="QmEventTest",
                max_price=1000000000000000000,
                duration=3600
            )
            return result['auction_id'], result['tx_hash']
        
        self.auction_id, self.tx_hash = run_async(_setup())
    
    def test_extract_auction_id_from_receipt(self):
        """Test extracting auction ID from transaction receipt."""
        async def _test():
            # Get receipt
            receipt = await self.handler.client.wait_for_transaction(self.tx_hash)
            
            # Extract auction ID
            auction_id = await self.handler._extract_auction_id_from_receipt(receipt)
            
            self.assertEqual(auction_id, self.auction_id)
            print(f"\n✓ Extracted auction ID: {auction_id}")
        
        run_async(_test())
    
    def test_extract_auction_ended_event(self):
        """Test extracting auction ended event data."""
        async def _test():
            # End the auction
            end_result = await self.handler.end_auction(self.auction_id)
            
            # Get receipt
            receipt = await self.handler.client.wait_for_transaction(end_result['tx_hash'])
            
            # Extract event data
            event_data = await self.handler._extract_auction_ended_event(receipt, self.auction_id)
            
            self.assertIn('winning_agent_id', event_data)
            self.assertIn('winning_bid', event_data)
            self.assertEqual(end_result['winning_agent_id'], event_data['winning_agent_id'])
            self.assertEqual(end_result['winning_bid'], event_data['winning_bid'])
            
            print(f"\n✓ Extracted auction ended event:")
            print(f"  Winner: {event_data['winning_agent_id']}")
            print(f"  Bid: {event_data['winning_bid']}")
        
        run_async(_test())
    
    def test_extract_missing_event(self):
        """Test extraction fails gracefully when event not found."""
        async def _test():
            # Create a dummy receipt with wrong block number
            dummy_receipt = {
                'blockNumber': 1,
                'transactionHash': b'\x00' * 32
            }
            
            # Should raise exception
            with self.assertRaises(Exception) as context:
                await self.handler._extract_auction_id_from_receipt(dummy_receipt)
            
            self.assertIn("not found", str(context.exception))
            print(f"\n✓ Missing event handled correctly")
        
        run_async(_test())


class TestIntegrationFullFlow(unittest.TestCase):
    """Integration test for complete auction flow."""
    
    def test_complete_auction_lifecycle(self):
        """Test creating, monitoring, and ending an auction."""
        async def _test():
            handler = ConsumerBlockchainHandler()
            await handler.initialize()
            
            print("\n=== Starting Complete Auction Lifecycle Test ===")
            
            # 1. Create auction
            print("\n1. Creating auction...")
            create_result = await handler.create_auction(
                service_cid="QmIntegrationTest",
                max_price=2000000000000000000,
                duration=120,  # 2 minutes
                reputation_weight=60
            )
            
            self.assertIsNone(create_result['error'])
            auction_id = create_result['auction_id']
            print(f"✓ Auction {auction_id} created")
            
            # 2. Check initial status
            print("\n2. Checking initial status...")
            status = await handler.get_auction_status(auction_id)
            self.assertTrue(status['active'])
            self.assertFalse(status['completed'])
            print(f"✓ Auction is active")
            
            # 3. Wait for potential bids
            print("\n3. Waiting 10 seconds for potential bids...")
            await asyncio.sleep(10)
            
            # 4. End auction
            print("\n4. Ending auction...")
            end_result = await handler.end_auction(auction_id)
            self.assertIsNone(end_result['error'])
            print(f"✓ Auction ended, winner: {end_result['winning_agent_id']}")
            
            # 5. Check final status
            print("\n5. Checking final status...")
            final_status = await handler.get_auction_status(auction_id)
            self.assertFalse(final_status['active'])
            self.assertTrue(final_status['completed'])
            print(f"✓ Auction is completed")
            
            # 6. Try to get feedback auth (may not exist in this test)
            print("\n6. Querying for feedback auth...")
            current_block = await handler.client.get_block_number()
            feedback_auth = await handler.get_feedback_auth(
                auction_id,
                from_block=current_block - 100
            )
            print(f"✓ Feedback auth found: {feedback_auth is not None}")
            
            print("\n=== Complete Auction Lifecycle Test PASSED ===")
        
        run_async(_test())


if __name__ == '__main__':
    print("\n" + "="*70)
    print("CONSUMER BLOCKCHAIN HANDLER TESTS")
    print("="*70)
    print("\nPrerequisites:")
    print("  1. Anvil running on localhost:8545")
    print("  2. Contracts deployed")
    print("  3. Consumer agent registered")
    print("  4. Provider agents registered")
    print("  5. .env configured with consumer settings")
    print("="*70 + "\n")
    
    unittest.main(verbosity=2)

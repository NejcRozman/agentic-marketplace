"""
Tests for BlockchainHandler.

These tests require:
1. Anvil running with forked Sepolia: 
   anvil --fork-url https://sepolia.infura.io/v3/${INFURA_API_KEY}
2. ReverseAuction and MockUSDC deployed (run Deploy.s.sol)
3. Agent registered (run RegisterAgent.s.sol)
4. Auction created with the agent as eligible (run CreateAuction.s.sol)
5. agents/.env configured with:
   - BLOCKCHAIN_RPC_URL=http://localhost:8545
   - BLOCKCHAIN_PRIVATE_KEY=0x...
   - BLOCKCHAIN_REVERSE_AUCTION_ADDRESS=0x...
   - BLOCKCHAIN_IDENTITY_REGISTRY_ADDRESS=0x...
   - BLOCKCHAIN_REPUTATION_REGISTRY_ADDRESS=0x...
   - GOOGLE_API_KEY=...

Run with: python agents/tests/test_blockchain_handler.py
"""

import unittest
import asyncio
import os
import logging

# Configure logging to see all output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Use absolute imports from agents package
from agents.config import config
from agents.infrastructure.blockchain_client import BlockchainClient
from agents.infrastructure.contract_abis import get_reverse_auction_abi
from agents.provider_agent.blockchain_handler import BlockchainHandler


def run_async(coro):
    """Helper to run async functions in sync tests."""
    return asyncio.get_event_loop().run_until_complete(coro)


class TestBlockchainHandlerSetup(unittest.TestCase):
    """Test BlockchainHandler initialization and configuration."""
    
    def test_config_loaded(self):
        """Verify configuration is loaded from .env."""
        self.assertIsNotNone(config.rpc_url, "BLOCKCHAIN_RPC_URL not set")
        self.assertIsNotNone(config.reverse_auction_address, "BLOCKCHAIN_REVERSE_AUCTION_ADDRESS not set")
        self.assertIsNotNone(config.identity_registry_address, "BLOCKCHAIN_IDENTITY_REGISTRY_ADDRESS not set")
        self.assertIsNotNone(config.google_api_key, "GOOGLE_API_KEY not set")
        print(f"\n✓ Config loaded:")
        print(f"  RPC URL: {config.rpc_url}")
        print(f"  ReverseAuction: {config.reverse_auction_address}")
    
    def test_handler_initialization(self):
        """Test BlockchainHandler initializes correctly."""
        handler = BlockchainHandler(agent_id=config.agent_id)
        
        self.assertEqual(handler.agent_id, config.agent_id)
        self.assertIsNotNone(handler.client)
        self.assertIsNotNone(handler.graph)
        print(f"\n✓ Handler initialized for agent {config.agent_id}")


class TestBlockchainClient(unittest.TestCase):
    """Test BlockchainClient directly."""
    
    def setUp(self):
        """Create a BlockchainClient instance."""
        self.client = BlockchainClient()
    
    def tearDown(self):
        """Close the BlockchainClient."""
        run_async(self.client.close())
    
    def test_get_block_number(self):
        """Test getting current block number."""
        async def _test():
            await self.client._initialize()
            block = await self.client.get_block_number()
            self.assertGreater(block, 0)
            print(f"\n✓ Current block: {block}")
        
        run_async(_test())
    
    def test_load_contract(self):
        """Test loading ReverseAuction contract."""
        async def _test():
            await self.client._initialize()
            contract = await self.client.load_contract(
                "ReverseAuction",
                address=config.reverse_auction_address,
                abi=get_reverse_auction_abi()
            )
            self.assertIsNotNone(contract)
            print(f"\n✓ Contract loaded: {config.reverse_auction_address}")
        
        run_async(_test())
    
    def test_get_auction_count(self):
        """Test fetching auction counter from contract."""
        async def _test():
            await self.client._initialize()
            await self.client.load_contract(
                "ReverseAuction",
                address=config.reverse_auction_address,
                abi=get_reverse_auction_abi()
            )
            
            counter = await self.client.call_contract_method(
                "ReverseAuction",
                "auctionIdCounter"
            )
            print(f"\n📊 Total auctions created: {counter}")
            self.assertGreater(counter, 0)
        
        run_async(_test())
    
    def test_get_auction_details(self):
        """Test getting details of auction 2."""
        async def _test():
            await self.client._initialize()
            await self.client.load_contract(
                "ReverseAuction",
                address=config.reverse_auction_address,
                abi=get_reverse_auction_abi()
            )
            
            auction = await self.client.call_contract_method(
                "ReverseAuction",
                "getAuctionDetails",
                2
            )
            print(f"\n📋 Auction 2 details:")
            print(f"  Buyer: {auction[1]}")
            print(f"  Service CID: {auction[2]}")
            print(f"  Max Price: {auction[3] / 1e6} USDC")
            print(f"  Is Active: {auction[9]}")
            
            self.assertTrue(auction[9], "Auction should be active")
            print(f"\n✓ Auction 2 is active")
        
        run_async(_test())


class TestBlockchainHandlerMonitor(unittest.TestCase):
    """Test the monitor_auctions functionality."""
    
    def setUp(self):
        """Create a BlockchainHandler instance."""
        self.handler = BlockchainHandler(agent_id=config.agent_id)
    
    def tearDown(self):
        """Close the BlockchainClient."""
        run_async(self.handler.client.close())
    
    def test_handler_initialize(self):
        """Test that handler can initialize contracts."""
        async def _test():
            result = await self.handler.initialize()
            self.assertTrue(result, "Handler initialization failed")
            self.assertTrue(self.handler.contracts_loaded)
            print(f"\n✓ Handler contracts initialized")
        
        run_async(_test())
    
    def test_monitor_auctions_discovers_auction(self):
        """Test that monitor_auctions discovers the active auction."""
        async def _test():
            # Initialize first
            await self.handler.initialize()
            
            print(f"\n🔍 Running monitor_auctions for agent {self.handler.agent_id}...")
            result = await self.handler.monitor_auctions()
            
            print(f"\n📊 Monitor result:")
            print(f"  Won auctions: {result.get('won_auctions', [])}")
            print(f"  Eligible active auctions count: {result.get('eligible_active_auctions_count', 0)}")
            print(f"  Bids placed: {result.get('bids_placed', [])}")
            print(f"  Error: {result.get('error')}")
            
            self.assertIsNone(result.get('error'), f"Monitor failed: {result.get('error')}")
            self.assertGreater(result.get('eligible_active_auctions_count', 0), 0, "No eligible auctions found")
            
            print(f"\n✓ Found {result['eligible_active_auctions_count']} eligible auction(s)")
        
        run_async(_test())


class TestEndToEndWorkflow(unittest.TestCase):
    """Test the complete workflow: discover -> bid -> (win) -> complete."""
    
    def setUp(self):
        """Create a BlockchainHandler instance."""
        self.handler = BlockchainHandler(agent_id=config.agent_id)
    
    def tearDown(self):
        """Close the BlockchainClient."""
        run_async(self.handler.client.close())
    
    def test_full_workflow(self):
        """
        Test the full workflow:
        1. Initialize handler
        2. Monitor auctions and discover eligible ones
        3. Place a bid (if LLM decides to)
        """
        async def _test():
            print("\n" + "="*60)
            print("🚀 FULL WORKFLOW TEST")
            print("="*60)
            
            # Step 1: Initialize
            print("\n⚙️ Step 1: Initialize handler...")
            init_result = await self.handler.initialize()
            self.assertTrue(init_result, "Handler initialization failed")
            print("  ✓ Handler initialized")
            
            # Step 2: Monitor and discover auctions
            print("\n📡 Step 2: Monitor auctions...")
            result = await self.handler.monitor_auctions()
            
            self.assertIsNone(result.get('error'), f"Monitor failed: {result.get('error')}")
            print(f"  ✓ Found {result.get('eligible_active_auctions_count', 0)} eligible auction(s)")
            
            if result.get('bids_placed'):
                print(f"  ✓ Placed {len(result['bids_placed'])} bid(s)")
                for bid in result['bids_placed']:
                    print(f"    Auction {bid.get('auction_id')}: {bid.get('bid_amount')} (raw)")
            else:
                print("  ℹ No bids placed (LLM may have decided not to bid)")
            
            print("\n✅ Workflow test completed successfully!")
            print("="*60)
        
        run_async(_test())


if __name__ == "__main__":
    # Run with unittest
    unittest.main(verbosity=2)

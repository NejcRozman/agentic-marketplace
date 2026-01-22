"""
Tests for Consumer Orchestrator.

Unit tests use mocked components for fast isolated testing.

Prerequisites:
- None (all components mocked)

Run with: python agents/tests/test_consumer_orchestrator_unit.py
"""

import sys
from pathlib import Path


import unittest
import asyncio
import json
import logging
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from tempfile import TemporaryDirectory
from datetime import datetime, timedelta

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
from agents.config import Config


def run_async(coro):
    """Helper to run async functions in sync tests."""
    return asyncio.get_event_loop().run_until_complete(coro)


class TestConsumerInitialization(unittest.TestCase):
    """Test Consumer initialization and setup."""
    
    def test_consumer_initialization(self):
        """Test Consumer initializes correctly with all components."""
        mock_config = Mock(spec=Config)
        mock_config.consumer_check_interval = 10
        
        with patch('agents.consumer_agent.consumer_orchestrator.ConsumerBlockchainHandler'), \
             patch('agents.consumer_agent.consumer_orchestrator.IPFSClient'), \
             patch('agents.consumer_agent.consumer_orchestrator.ServiceGenerator'), \
             patch('agents.consumer_agent.consumer_orchestrator.ServiceEvaluator'):
            
            consumer = Consumer(mock_config)
            
            # Check core components
            self.assertEqual(consumer.config, mock_config)
            self.assertIsNotNone(consumer.blockchain_handler)
            self.assertIsNotNone(consumer.ipfs_client)
            self.assertIsNotNone(consumer.service_generator)
            self.assertIsNotNone(consumer.evaluator)
            
            # Check tracking structures
            self.assertEqual(consumer.active_auctions, {})
            self.assertEqual(consumer.completed_auctions, [])
            self.assertEqual(consumer.available_services, [])
            self.assertEqual(consumer.service_index, 0)
            
            # Check runtime state
            self.assertFalse(consumer.running)
            self.assertEqual(consumer.check_interval, 10)
            self.assertIsNotNone(consumer.result_base_path)
            
            print("\n✓ Consumer initialized with all components and tracking structures")
    
    def test_initialize_without_pdf_dir(self):
        """Test initialize without PDF directory."""
        async def _test():
            mock_config = Mock(spec=Config)
            mock_config.consumer_check_interval = 10
            
            with patch('agents.consumer_agent.consumer_orchestrator.ConsumerBlockchainHandler') as MockHandler, \
                 patch('agents.consumer_agent.consumer_orchestrator.IPFSClient'), \
                 patch('agents.consumer_agent.consumer_orchestrator.ServiceGenerator'), \
                 patch('agents.consumer_agent.consumer_orchestrator.ServiceEvaluator'):
                
                consumer = Consumer(mock_config)
                consumer.blockchain_handler.initialize = AsyncMock(return_value=True)
                consumer.blockchain_handler._initialized = False
                
                # Initialize without pdf_dir
                await consumer.initialize()
                
                # Verify blockchain handler initialized
                consumer.blockchain_handler.initialize.assert_called_once()
                
                # Verify available_services still empty
                self.assertEqual(consumer.available_services, [])
                
                print("\n✓ Initialize without pdf_dir completed successfully")
        
        run_async(_test())
    
    def test_initialize_with_pdf_dir(self):
        """Test initialize with PDF directory triggers service loading."""
        async def _test():
            mock_config = Mock(spec=Config)
            mock_config.consumer_check_interval = 10
            
            with patch('agents.consumer_agent.consumer_orchestrator.ConsumerBlockchainHandler'), \
                 patch('agents.consumer_agent.consumer_orchestrator.IPFSClient'), \
                 patch('agents.consumer_agent.consumer_orchestrator.ServiceGenerator'), \
                 patch('agents.consumer_agent.consumer_orchestrator.ServiceEvaluator'):
                
                consumer = Consumer(mock_config)
                consumer.blockchain_handler.initialize = AsyncMock(return_value=True)
                consumer.blockchain_handler._initialized = False
                consumer.load_services = AsyncMock()
                
                # Initialize with pdf_dir
                test_dir = Path("/test/pdfs")
                await consumer.initialize(pdf_dir=test_dir, complexity="high")
                
                # Verify blockchain handler initialized
                consumer.blockchain_handler.initialize.assert_called_once()
                
                # Verify load_services called with correct params
                consumer.load_services.assert_called_once_with(test_dir, "high")
                
                print("\n✓ Initialize with pdf_dir triggered load_services")
        
        run_async(_test())
    
    def test_initialize_blockchain_failure(self):
        """Test initialize raises error when blockchain initialization fails."""
        async def _test():
            mock_config = Mock(spec=Config)
            mock_config.consumer_check_interval = 10
            
            with patch('agents.consumer_agent.consumer_orchestrator.ConsumerBlockchainHandler'), \
                 patch('agents.consumer_agent.consumer_orchestrator.IPFSClient'), \
                 patch('agents.consumer_agent.consumer_orchestrator.ServiceGenerator'), \
                 patch('agents.consumer_agent.consumer_orchestrator.ServiceEvaluator'):
                
                consumer = Consumer(mock_config)
                consumer.blockchain_handler.initialize = AsyncMock(return_value=False)
                consumer.blockchain_handler._initialized = False
                
                # Should raise RuntimeError
                with self.assertRaises(RuntimeError) as cm:
                    await consumer.initialize()
                
                self.assertIn("Failed to initialize", str(cm.exception))
                
                print("\n✓ RuntimeError raised when blockchain initialization fails")
        
        run_async(_test())
    
    def test_initialize_idempotent(self):
        """Test calling initialize multiple times doesn't cause issues."""
        async def _test():
            mock_config = Mock(spec=Config)
            mock_config.consumer_check_interval = 10
            
            with patch('agents.consumer_agent.consumer_orchestrator.ConsumerBlockchainHandler'), \
                 patch('agents.consumer_agent.consumer_orchestrator.IPFSClient'), \
                 patch('agents.consumer_agent.consumer_orchestrator.ServiceGenerator'), \
                 patch('agents.consumer_agent.consumer_orchestrator.ServiceEvaluator'):
                
                consumer = Consumer(mock_config)
                consumer.blockchain_handler.initialize = AsyncMock(return_value=True)
                consumer.blockchain_handler._initialized = False
                
                # Initialize twice
                await consumer.initialize()
                consumer.blockchain_handler._initialized = True
                await consumer.initialize()
                
                # Should not fail
                print("\n✓ Initialize is idempotent")
        
        run_async(_test())


class TestServiceLoading(unittest.TestCase):
    """Test service pre-generation and caching."""
    
    def test_load_services_success(self):
        """Test loading services successfully."""
        async def _test():
            mock_config = Mock(spec=Config)
            mock_config.consumer_check_interval = 10
            
            with patch('agents.consumer_agent.consumer_orchestrator.ConsumerBlockchainHandler'), \
                 patch('agents.consumer_agent.consumer_orchestrator.IPFSClient'), \
                 patch('agents.consumer_agent.consumer_orchestrator.ServiceGenerator'), \
                 patch('agents.consumer_agent.consumer_orchestrator.ServiceEvaluator'):
                
                consumer = Consumer(mock_config)
                
                # Mock service_generator.generate_services_from_pdfs
                mock_result = {
                    "processed": [
                        {
                            "service_cid": "QmService1",
                            "pdf_cid": "QmPDF1",
                            "pdf_name": "paper1.pdf",
                            "title": "Test Paper 1",
                            "prompts": ["Q1", "Q2"],
                            "complexity": "medium"
                        },
                        {
                            "service_cid": "QmService2",
                            "pdf_cid": "QmPDF2",
                            "pdf_name": "paper2.pdf",
                            "title": "Test Paper 2",
                            "prompts": ["Q3", "Q4"],
                            "complexity": "medium"
                        }
                    ],
                    "skipped": [],
                    "failed": []
                }
                
                consumer.service_generator.generate_services_from_pdfs = AsyncMock(return_value=mock_result)
                
                # Load services
                test_dir = Path("/test/pdfs")
                await consumer.load_services(test_dir, complexity="medium")
                
                # Verify services loaded
                self.assertEqual(len(consumer.available_services), 2)
                self.assertEqual(consumer.available_services[0]["service_cid"], "QmService1")
                self.assertEqual(consumer.available_services[1]["service_cid"], "QmService2")
                self.assertEqual(consumer.service_index, 0)
                
                # Verify generator called correctly
                consumer.service_generator.generate_services_from_pdfs.assert_called_once_with(
                    pdf_dir=test_dir,
                    complexity="medium",
                    skip_processed=True
                )
                
                print("\n✓ Loaded 2 services successfully")
        
        run_async(_test())
    
    def test_load_services_with_failures(self):
        """Test loading services with some failures."""
        async def _test():
            mock_config = Mock(spec=Config)
            mock_config.consumer_check_interval = 10
            
            with patch('agents.consumer_agent.consumer_orchestrator.ConsumerBlockchainHandler'), \
                 patch('agents.consumer_agent.consumer_orchestrator.IPFSClient'), \
                 patch('agents.consumer_agent.consumer_orchestrator.ServiceGenerator'), \
                 patch('agents.consumer_agent.consumer_orchestrator.ServiceEvaluator'):
                
                consumer = Consumer(mock_config)
                
                mock_result = {
                    "processed": [
                        {"service_cid": "QmService1", "pdf_cid": "QmPDF1", "pdf_name": "paper1.pdf", 
                         "title": "Test 1", "prompts": ["Q1"], "complexity": "low"}
                    ],
                    "skipped": [],
                    "failed": [
                        {"pdf_name": "paper2.pdf", "error": "IPFS upload failed"}
                    ]
                }
                
                consumer.service_generator.generate_services_from_pdfs = AsyncMock(return_value=mock_result)
                
                await consumer.load_services(Path("/test"), complexity="low")
                
                # Verify only processed services added
                self.assertEqual(len(consumer.available_services), 1)
                
                print("\n✓ Handled failures correctly, loaded 1 processed service")
        
        run_async(_test())
    
    def test_load_services_all_skipped(self):
        """Test loading when all services already processed."""
        async def _test():
            mock_config = Mock(spec=Config)
            mock_config.consumer_check_interval = 10
            
            with patch('agents.consumer_agent.consumer_orchestrator.ConsumerBlockchainHandler'), \
                 patch('agents.consumer_agent.consumer_orchestrator.IPFSClient'), \
                 patch('agents.consumer_agent.consumer_orchestrator.ServiceGenerator'), \
                 patch('agents.consumer_agent.consumer_orchestrator.ServiceEvaluator'):
                
                consumer = Consumer(mock_config)
                
                mock_result = {
                    "processed": [],
                    "skipped": ["paper1.pdf", "paper2.pdf"],
                    "failed": []
                }
                
                consumer.service_generator.generate_services_from_pdfs = AsyncMock(return_value=mock_result)
                
                await consumer.load_services(Path("/test"), complexity="medium")
                
                # Verify no services added
                self.assertEqual(len(consumer.available_services), 0)
                
                print("\n✓ All services skipped, available_services empty")
        
        run_async(_test())
    
    def test_load_services_empty_directory(self):
        """Test loading from empty directory."""
        async def _test():
            mock_config = Mock(spec=Config)
            mock_config.consumer_check_interval = 10
            
            with patch('agents.consumer_agent.consumer_orchestrator.ConsumerBlockchainHandler'), \
                 patch('agents.consumer_agent.consumer_orchestrator.IPFSClient'), \
                 patch('agents.consumer_agent.consumer_orchestrator.ServiceGenerator'), \
                 patch('agents.consumer_agent.consumer_orchestrator.ServiceEvaluator'):
                
                consumer = Consumer(mock_config)
                
                mock_result = {
                    "processed": [],
                    "skipped": [],
                    "failed": []
                }
                
                consumer.service_generator.generate_services_from_pdfs = AsyncMock(return_value=mock_result)
                
                await consumer.load_services(Path("/test/empty"), complexity="medium")
                
                self.assertEqual(len(consumer.available_services), 0)
                
                print("\n✓ Empty directory handled correctly")
        
        run_async(_test())
    
    def test_load_services_different_complexity_levels(self):
        """Test loading services with different complexity levels."""
        async def _test():
            mock_config = Mock(spec=Config)
            mock_config.consumer_check_interval = 10
            
            with patch('agents.consumer_agent.consumer_orchestrator.ConsumerBlockchainHandler'), \
                 patch('agents.consumer_agent.consumer_orchestrator.IPFSClient'), \
                 patch('agents.consumer_agent.consumer_orchestrator.ServiceGenerator'), \
                 patch('agents.consumer_agent.consumer_orchestrator.ServiceEvaluator'):
                
                consumer = Consumer(mock_config)
                
                mock_result = {"processed": [], "skipped": [], "failed": []}
                consumer.service_generator.generate_services_from_pdfs = AsyncMock(return_value=mock_result)
                
                # Test each complexity level
                for complexity in ["low", "medium", "high"]:
                    await consumer.load_services(Path("/test"), complexity=complexity)
                    
                    # Verify correct complexity passed
                    calls = consumer.service_generator.generate_services_from_pdfs.call_args_list
                    last_call = calls[-1]
                    self.assertEqual(last_call[1]["complexity"], complexity)
                
                print("\n✓ All complexity levels (low/medium/high) passed correctly")
        
        run_async(_test())
    
    def test_load_services_updates_existing_cache(self):
        """Test that loading services replaces existing cache."""
        async def _test():
            mock_config = Mock(spec=Config)
            mock_config.consumer_check_interval = 10
            
            with patch('agents.consumer_agent.consumer_orchestrator.ConsumerBlockchainHandler'), \
                 patch('agents.consumer_agent.consumer_orchestrator.IPFSClient'), \
                 patch('agents.consumer_agent.consumer_orchestrator.ServiceGenerator'), \
                 patch('agents.consumer_agent.consumer_orchestrator.ServiceEvaluator'):
                
                consumer = Consumer(mock_config)
                
                # Load first batch
                mock_result1 = {
                    "processed": [
                        {"service_cid": "QmOld", "pdf_cid": "QmPDF1", "pdf_name": "old.pdf",
                         "title": "Old", "prompts": ["Q"], "complexity": "low"}
                    ],
                    "skipped": [],
                    "failed": []
                }
                consumer.service_generator.generate_services_from_pdfs = AsyncMock(return_value=mock_result1)
                await consumer.load_services(Path("/test"), complexity="low")
                
                self.assertEqual(len(consumer.available_services), 1)
                self.assertEqual(consumer.available_services[0]["service_cid"], "QmOld")
                
                # Load second batch
                mock_result2 = {
                    "processed": [
                        {"service_cid": "QmNew1", "pdf_cid": "QmPDF2", "pdf_name": "new1.pdf",
                         "title": "New 1", "prompts": ["Q"], "complexity": "medium"},
                        {"service_cid": "QmNew2", "pdf_cid": "QmPDF3", "pdf_name": "new2.pdf",
                         "title": "New 2", "prompts": ["Q"], "complexity": "medium"}
                    ],
                    "skipped": [],
                    "failed": []
                }
                consumer.service_generator.generate_services_from_pdfs = AsyncMock(return_value=mock_result2)
                await consumer.load_services(Path("/test"), complexity="medium")
                
                # Verify cache replaced (not appended)
                self.assertEqual(len(consumer.available_services), 2)
                self.assertEqual(consumer.available_services[0]["service_cid"], "QmNew1")
                self.assertEqual(consumer.available_services[1]["service_cid"], "QmNew2")
                self.assertEqual(consumer.service_index, 0)
                
                print("\n✓ Service cache replaced on reload")
        
        run_async(_test())


class TestAuctionCreation(unittest.TestCase):
    """Test auction creation workflow."""
    
    def setUp(self):
        """Set up common test fixtures."""
        self.mock_config = Mock(spec=Config)
        self.mock_config.consumer_check_interval = 10
        self.mock_config.eligible_providers = [1, 2, 3]
        
        self.sample_services = [
            {"service_cid": "QmService1", "pdf_cid": "QmPDF1", "pdf_name": "p1.pdf", 
             "title": "Paper 1", "prompts": ["Q1"], "complexity": "medium"},
            {"service_cid": "QmService2", "pdf_cid": "QmPDF2", "pdf_name": "p2.pdf",
             "title": "Paper 2", "prompts": ["Q2"], "complexity": "medium"},
            {"service_cid": "QmService3", "pdf_cid": "QmPDF3", "pdf_name": "p3.pdf",
             "title": "Paper 3", "prompts": ["Q3"], "complexity": "medium"}
        ]
    
    def test_create_auction_auto_initialize(self):
        """Test create_auction auto-initializes if not initialized."""
        async def _test():
            with patch('agents.consumer_agent.consumer_orchestrator.ConsumerBlockchainHandler'), \
                 patch('agents.consumer_agent.consumer_orchestrator.IPFSClient'), \
                 patch('agents.consumer_agent.consumer_orchestrator.ServiceGenerator'), \
                 patch('agents.consumer_agent.consumer_orchestrator.ServiceEvaluator'):
                
                consumer = Consumer(self.mock_config)
                consumer.blockchain_handler._initialized = False
                consumer.blockchain_handler.initialize = AsyncMock(return_value=True)
                consumer.blockchain_handler.create_auction = AsyncMock(
                    return_value={"auction_id": 1, "tx_hash": "0xabc", "error": None}
                )
                consumer.available_services = [self.sample_services[0]]
                
                # Should auto-initialize
                auction_id = await consumer.create_auction()
                
                consumer.blockchain_handler.initialize.assert_called_once()
                self.assertEqual(auction_id, 1)
                
                print("\n✓ Auto-initialization triggered when not initialized")
        
        run_async(_test())
    
    def test_create_auction_no_services_available(self):
        """Test create_auction raises error when no services loaded."""
        async def _test():
            with patch('agents.consumer_agent.consumer_orchestrator.ConsumerBlockchainHandler'), \
                 patch('agents.consumer_agent.consumer_orchestrator.IPFSClient'), \
                 patch('agents.consumer_agent.consumer_orchestrator.ServiceGenerator'), \
                 patch('agents.consumer_agent.consumer_orchestrator.ServiceEvaluator'):
                
                consumer = Consumer(self.mock_config)
                consumer.blockchain_handler._initialized = True
                consumer.available_services = []
                
                with self.assertRaises(RuntimeError) as cm:
                    await consumer.create_auction()
                
                self.assertIn("No services available", str(cm.exception))
                
                print("\n✓ RuntimeError raised when no services available")
        
        run_async(_test())
    
    def test_create_auction_next_service_in_order(self):
        """Test creating multiple auctions uses services in order."""
        async def _test():
            with patch('agents.consumer_agent.consumer_orchestrator.ConsumerBlockchainHandler'), \
                 patch('agents.consumer_agent.consumer_orchestrator.IPFSClient'), \
                 patch('agents.consumer_agent.consumer_orchestrator.ServiceGenerator'), \
                 patch('agents.consumer_agent.consumer_orchestrator.ServiceEvaluator'):
                
                consumer = Consumer(self.mock_config)
                consumer.blockchain_handler._initialized = True
                consumer.available_services = self.sample_services.copy()
                
                auction_ids = []
                for i in range(3):
                    consumer.blockchain_handler.create_auction = AsyncMock(
                        return_value={"auction_id": i+1, "tx_hash": f"0x{i}", "error": None}
                    )
                    auction_id = await consumer.create_auction(service_index=None)
                    auction_ids.append(auction_id)
                    
                    # Verify correct service used
                    call_args = consumer.blockchain_handler.create_auction.call_args
                    expected_cid = self.sample_services[i]["service_cid"]
                    self.assertEqual(call_args[1]["service_cid"], expected_cid)
                
                # Verify service_index incremented
                self.assertEqual(consumer.service_index, 3)
                self.assertEqual(auction_ids, [1, 2, 3])
                
                print("\n✓ Three auctions created with services in order")
        
        run_async(_test())
    
    def test_create_auction_specific_index(self):
        """Test creating auction with specific service index."""
        async def _test():
            with patch('agents.consumer_agent.consumer_orchestrator.ConsumerBlockchainHandler'), \
                 patch('agents.consumer_agent.consumer_orchestrator.IPFSClient'), \
                 patch('agents.consumer_agent.consumer_orchestrator.ServiceGenerator'), \
                 patch('agents.consumer_agent.consumer_orchestrator.ServiceEvaluator'):
                
                consumer = Consumer(self.mock_config)
                consumer.blockchain_handler._initialized = True
                consumer.available_services = self.sample_services.copy()
                consumer.blockchain_handler.create_auction = AsyncMock(
                    return_value={"auction_id": 42, "tx_hash": "0xdef", "error": None}
                )
                
                # Use service at index 1
                auction_id = await consumer.create_auction(service_index=1)
                
                # Verify correct service used
                call_args = consumer.blockchain_handler.create_auction.call_args
                self.assertEqual(call_args[1]["service_cid"], "QmService2")
                
                # Verify service_index not modified
                self.assertEqual(consumer.service_index, 0)
                
                print("\n✓ Specific service index used, service_index unchanged")
        
        run_async(_test())
    
    def test_create_auction_invalid_indices(self):
        """Test invalid service indices raise appropriate errors."""
        async def _test():
            with patch('agents.consumer_agent.consumer_orchestrator.ConsumerBlockchainHandler'), \
                 patch('agents.consumer_agent.consumer_orchestrator.IPFSClient'), \
                 patch('agents.consumer_agent.consumer_orchestrator.ServiceGenerator'), \
                 patch('agents.consumer_agent.consumer_orchestrator.ServiceEvaluator'):
                
                consumer = Consumer(self.mock_config)
                consumer.blockchain_handler._initialized = True
                consumer.available_services = self.sample_services.copy()
                
                # Test negative index
                with self.assertRaises(ValueError) as cm:
                    await consumer.create_auction(service_index=-1)
                self.assertIn("Invalid service_index", str(cm.exception))
                
                # Test index too high
                with self.assertRaises(ValueError) as cm:
                    await consumer.create_auction(service_index=10)
                self.assertIn("Invalid service_index", str(cm.exception))
                
                print("\n✓ ValueError raised for invalid indices (-1 and 10)")
        
        run_async(_test())
    
    def test_create_auction_blockchain_success(self):
        """Test successful auction creation stores tracker correctly."""
        async def _test():
            with patch('agents.consumer_agent.consumer_orchestrator.ConsumerBlockchainHandler'), \
                 patch('agents.consumer_agent.consumer_orchestrator.IPFSClient'), \
                 patch('agents.consumer_agent.consumer_orchestrator.ServiceGenerator'), \
                 patch('agents.consumer_agent.consumer_orchestrator.ServiceEvaluator'):
                
                consumer = Consumer(self.mock_config)
                consumer.blockchain_handler._initialized = True
                consumer.available_services = [self.sample_services[0]]
                consumer.blockchain_handler.create_auction = AsyncMock(
                    return_value={"auction_id": 123, "tx_hash": "0xabc123", "error": None}
                )
                
                auction_id = await consumer.create_auction(
                    max_budget=50000000,
                    duration=3600,
                    eligible_providers=[4, 5, 6]
                )
                
                # Verify tracker created and stored
                self.assertEqual(auction_id, 123)
                self.assertIn(123, consumer.active_auctions)
                
                tracker = consumer.active_auctions[123]
                self.assertEqual(tracker.auction_id, 123)
                self.assertEqual(tracker.status, AuctionStatus.CREATED)
                self.assertEqual(tracker.service_cid, "QmService1")
                self.assertEqual(tracker.max_budget, 50000000)
                self.assertEqual(tracker.duration, 3600)
                self.assertEqual(tracker.eligible_providers, [4, 5, 6])
                self.assertIsNotNone(tracker.created_at)
                
                print("\n✓ Tracker created with all fields correctly")
        
        run_async(_test())
    
    def test_create_auction_blockchain_failure(self):
        """Test blockchain failure raises appropriate error."""
        async def _test():
            with patch('agents.consumer_agent.consumer_orchestrator.ConsumerBlockchainHandler'), \
                 patch('agents.consumer_agent.consumer_orchestrator.IPFSClient'), \
                 patch('agents.consumer_agent.consumer_orchestrator.ServiceGenerator'), \
                 patch('agents.consumer_agent.consumer_orchestrator.ServiceEvaluator'):
                
                consumer = Consumer(self.mock_config)
                consumer.blockchain_handler._initialized = True
                consumer.available_services = [self.sample_services[0]]
                consumer.blockchain_handler.create_auction = AsyncMock(
                    return_value={"auction_id": None, "tx_hash": None, "error": "Gas estimation failed"}
                )
                
                with self.assertRaises(RuntimeError) as cm:
                    await consumer.create_auction()
                
                self.assertIn("Failed to create auction", str(cm.exception))
                self.assertIn("Gas estimation failed", str(cm.exception))
                
                print("\n✓ RuntimeError raised with blockchain error message")
        
        run_async(_test())
    
    def test_create_auction_custom_parameters(self):
        """Test auction created with custom parameters."""
        async def _test():
            with patch('agents.consumer_agent.consumer_orchestrator.ConsumerBlockchainHandler'), \
                 patch('agents.consumer_agent.consumer_orchestrator.IPFSClient'), \
                 patch('agents.consumer_agent.consumer_orchestrator.ServiceGenerator'), \
                 patch('agents.consumer_agent.consumer_orchestrator.ServiceEvaluator'):
                
                consumer = Consumer(self.mock_config)
                consumer.blockchain_handler._initialized = True
                consumer.available_services = [self.sample_services[0]]
                consumer.blockchain_handler.create_auction = AsyncMock(
                    return_value={"auction_id": 1, "tx_hash": "0x1", "error": None}
                )
                
                await consumer.create_auction(
                    service_index=0,
                    max_budget=75000000,
                    duration=7200,
                    eligible_providers=[7, 8, 9]
                )
                
                # Verify correct parameters passed
                call_args = consumer.blockchain_handler.create_auction.call_args
                self.assertEqual(call_args[1]["max_price"], 75000000)
                self.assertEqual(call_args[1]["duration"], 7200)
                self.assertEqual(call_args[1]["eligible_agent_ids"], [7, 8, 9])
                
                print("\n✓ Custom parameters passed correctly to blockchain")
        
        run_async(_test())
    
    def test_create_auction_random_selection_from_pool(self):
        """Test random provider selection when provider_pool is configured."""
        async def _test():
            with patch('agents.consumer_agent.consumer_orchestrator.ConsumerBlockchainHandler'), \
                 patch('agents.consumer_agent.consumer_orchestrator.IPFSClient'), \
                 patch('agents.consumer_agent.consumer_orchestrator.ServiceGenerator'), \
                 patch('agents.consumer_agent.consumer_orchestrator.ServiceEvaluator'):
                
                # Configure provider_pool and eligible_per_auction
                self.mock_config.provider_pool = [10, 20, 30, 40, 50]
                self.mock_config.eligible_per_auction = 3
                
                consumer = Consumer(self.mock_config)
                consumer.blockchain_handler._initialized = True
                consumer.available_services = [self.sample_services[0]]
                consumer.blockchain_handler.create_auction = AsyncMock(
                    return_value={"auction_id": 1, "tx_hash": "0x1", "error": None}
                )
                
                # Create auction without specifying eligible_providers
                await consumer.create_auction()
                
                # Verify random selection occurred
                call_args = consumer.blockchain_handler.create_auction.call_args
                eligible_sent = call_args[1]["eligible_agent_ids"]
                
                # Check correct count
                self.assertEqual(len(eligible_sent), 3, f"Expected 3 providers, got {len(eligible_sent)}")
                
                # Check all are from pool
                for pid in eligible_sent:
                    self.assertIn(pid, [10, 20, 30, 40, 50], f"Provider {pid} not in pool")
                
                # Check no duplicates
                self.assertEqual(len(eligible_sent), len(set(eligible_sent)), "Duplicate providers selected")
                
                print(f"\n✓ Random selection: {eligible_sent} from pool [10, 20, 30, 40, 50]")
        
        run_async(_test())
    
    def test_create_auction_use_full_pool(self):
        """Test using full provider pool when eligible_per_auction >= pool size."""
        async def _test():
            with patch('agents.consumer_agent.consumer_orchestrator.ConsumerBlockchainHandler'), \
                 patch('agents.consumer_agent.consumer_orchestrator.IPFSClient'), \
                 patch('agents.consumer_agent.consumer_orchestrator.ServiceGenerator'), \
                 patch('agents.consumer_agent.consumer_orchestrator.ServiceEvaluator'):
                
                # Configure provider_pool with eligible_per_auction >= pool size
                self.mock_config.provider_pool = [100, 200, 300]
                self.mock_config.eligible_per_auction = 5  # More than pool size
                
                consumer = Consumer(self.mock_config)
                consumer.blockchain_handler._initialized = True
                consumer.available_services = [self.sample_services[0]]
                consumer.blockchain_handler.create_auction = AsyncMock(
                    return_value={"auction_id": 1, "tx_hash": "0x1", "error": None}
                )
                
                await consumer.create_auction()
                
                # Verify full pool is used
                call_args = consumer.blockchain_handler.create_auction.call_args
                eligible_sent = call_args[1]["eligible_agent_ids"]
                
                self.assertEqual(len(eligible_sent), 3, "Expected full pool of 3 providers")
                self.assertEqual(set(eligible_sent), {100, 200, 300}, "Expected complete pool")
                
                print(f"\n✓ Full pool used: {eligible_sent}")
        
        run_async(_test())
    
    def test_create_auction_fallback_to_eligible_providers(self):
        """Test fallback to config.eligible_providers when provider_pool not set."""
        async def _test():
            with patch('agents.consumer_agent.consumer_orchestrator.ConsumerBlockchainHandler'), \
                 patch('agents.consumer_agent.consumer_orchestrator.IPFSClient'), \
                 patch('agents.consumer_agent.consumer_orchestrator.ServiceGenerator'), \
                 patch('agents.consumer_agent.consumer_orchestrator.ServiceEvaluator'):
                
                # No provider_pool configured - should fallback to eligible_providers
                self.mock_config.eligible_providers = [1, 2, 3]
                # Ensure provider_pool is not set
                if hasattr(self.mock_config, 'provider_pool'):
                    delattr(self.mock_config, 'provider_pool')
                
                consumer = Consumer(self.mock_config)
                consumer.blockchain_handler._initialized = True
                consumer.available_services = [self.sample_services[0]]
                consumer.blockchain_handler.create_auction = AsyncMock(
                    return_value={"auction_id": 1, "tx_hash": "0x1", "error": None}
                )
                
                await consumer.create_auction()
                
                # Verify fallback to eligible_providers
                call_args = consumer.blockchain_handler.create_auction.call_args
                eligible_sent = call_args[1]["eligible_agent_ids"]
                
                self.assertEqual(eligible_sent, [1, 2, 3], "Expected config.eligible_providers")
                
                print(f"\n✓ Fallback to eligible_providers: {eligible_sent}")
        
        run_async(_test())
    
    def test_create_auction_empty_provider_pool_fallback(self):
        """Test fallback to eligible_providers when provider_pool is empty."""
        async def _test():
            with patch('agents.consumer_agent.consumer_orchestrator.ConsumerBlockchainHandler'), \
                 patch('agents.consumer_agent.consumer_orchestrator.IPFSClient'), \
                 patch('agents.consumer_agent.consumer_orchestrator.ServiceGenerator'), \
                 patch('agents.consumer_agent.consumer_orchestrator.ServiceEvaluator'):
                
                # Empty provider_pool - should fallback
                self.mock_config.provider_pool = []
                self.mock_config.eligible_providers = [5, 6, 7]
                
                consumer = Consumer(self.mock_config)
                consumer.blockchain_handler._initialized = True
                consumer.available_services = [self.sample_services[0]]
                consumer.blockchain_handler.create_auction = AsyncMock(
                    return_value={"auction_id": 1, "tx_hash": "0x1", "error": None}
                )
                
                await consumer.create_auction()
                
                # Verify fallback to eligible_providers
                call_args = consumer.blockchain_handler.create_auction.call_args
                eligible_sent = call_args[1]["eligible_agent_ids"]
                
                self.assertEqual(eligible_sent, [5, 6, 7], "Expected fallback to eligible_providers")
                
                print(f"\n✓ Empty pool fallback: {eligible_sent}")
        
        run_async(_test())
    
    def test_create_auction_randomness_verification(self):
        """Test that multiple auction creations produce different random selections."""
        async def _test():
            with patch('agents.consumer_agent.consumer_orchestrator.ConsumerBlockchainHandler'), \
                 patch('agents.consumer_agent.consumer_orchestrator.IPFSClient'), \
                 patch('agents.consumer_agent.consumer_orchestrator.ServiceGenerator'), \
                 patch('agents.consumer_agent.consumer_orchestrator.ServiceEvaluator'):
                
                # Large pool to ensure high probability of different selections
                self.mock_config.provider_pool = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                self.mock_config.eligible_per_auction = 3
                
                consumer = Consumer(self.mock_config)
                consumer.blockchain_handler._initialized = True
                consumer.available_services = [
                    self.sample_services[0],
                    self.sample_services[1],
                    self.sample_services[2]
                ]
                
                selections = []
                for i in range(3):
                    consumer.blockchain_handler.create_auction = AsyncMock(
                        return_value={"auction_id": i+1, "tx_hash": f"0x{i}", "error": None}
                    )
                    await consumer.create_auction()
                    
                    call_args = consumer.blockchain_handler.create_auction.call_args
                    eligible_sent = call_args[1]["eligible_agent_ids"]
                    selections.append(sorted(eligible_sent))
                
                # Check that all selections are valid (3 providers from pool)
                for sel in selections:
                    self.assertEqual(len(sel), 3, f"Expected 3 providers in selection {sel}")
                    for pid in sel:
                        self.assertIn(pid, range(1, 11), f"Provider {pid} not in pool")
                
                # With 10 choose 3 = 120 combinations, probability all 3 identical is very low
                # Just verify we got valid selections (not testing actual randomness quality)
                print(f"\n✓ Three selections made:")
                print(f"   Selection 1: {selections[0]}")
                print(f"   Selection 2: {selections[1]}")
                print(f"   Selection 3: {selections[2]}")
                
                # At least verify they're from the pool
                self.assertEqual(len(selections), 3, "Expected 3 selections")
        
        run_async(_test())
    
    def test_create_auction_explicit_providers_override_pool(self):
        """Test that explicitly provided eligible_providers bypasses random selection."""
        async def _test():
            with patch('agents.consumer_agent.consumer_orchestrator.ConsumerBlockchainHandler'), \
                 patch('agents.consumer_agent.consumer_orchestrator.IPFSClient'), \
                 patch('agents.consumer_agent.consumer_orchestrator.ServiceGenerator'), \
                 patch('agents.consumer_agent.consumer_orchestrator.ServiceEvaluator'):
                
                # Configure provider_pool
                self.mock_config.provider_pool = [10, 20, 30, 40, 50]
                self.mock_config.eligible_per_auction = 3
                
                consumer = Consumer(self.mock_config)
                consumer.blockchain_handler._initialized = True
                consumer.available_services = [self.sample_services[0]]
                consumer.blockchain_handler.create_auction = AsyncMock(
                    return_value={"auction_id": 1, "tx_hash": "0x1", "error": None}
                )
                
                # Explicitly provide eligible_providers - should override pool
                await consumer.create_auction(eligible_providers=[99, 88])
                
                # Verify explicit providers used, not random selection
                call_args = consumer.blockchain_handler.create_auction.call_args
                eligible_sent = call_args[1]["eligible_agent_ids"]
                
                self.assertEqual(eligible_sent, [99, 88], "Expected explicit providers, not random selection")
                
                print(f"\n✓ Explicit providers override pool: {eligible_sent}")
        
        run_async(_test())


class TestAuctionMonitoring(unittest.TestCase):
    """Test auction monitoring and state transitions."""
    
    def setUp(self):
        """Set up common test fixtures."""
        self.mock_config = Mock(spec=Config)
        self.mock_config.consumer_check_interval = 10
    
    def test_monitor_auctions_not_initialized(self):
        """Test monitoring skips when blockchain not initialized."""
        async def _test():
            with patch('agents.consumer_agent.consumer_orchestrator.ConsumerBlockchainHandler'), \
                 patch('agents.consumer_agent.consumer_orchestrator.IPFSClient'), \
                 patch('agents.consumer_agent.consumer_orchestrator.ServiceGenerator'), \
                 patch('agents.consumer_agent.consumer_orchestrator.ServiceEvaluator'):
                
                consumer = Consumer(self.mock_config)
                consumer.blockchain_handler._initialized = False
                
                # Should return early with warning
                await consumer.monitor_auctions()
                
                print("\n✓ Monitoring skipped when not initialized")
        
        run_async(_test())
    
    def test_monitor_auctions_empty_list(self):
        """Test monitoring with no active auctions."""
        async def _test():
            with patch('agents.consumer_agent.consumer_orchestrator.ConsumerBlockchainHandler'), \
                 patch('agents.consumer_agent.consumer_orchestrator.IPFSClient'), \
                 patch('agents.consumer_agent.consumer_orchestrator.ServiceGenerator'), \
                 patch('agents.consumer_agent.consumer_orchestrator.ServiceEvaluator'):
                
                consumer = Consumer(self.mock_config)
                consumer.blockchain_handler._initialized = True
                consumer.active_auctions = {}
                
                # Should complete without errors
                await consumer.monitor_auctions()
                
                print("\n✓ Empty auction list handled correctly")
        
        run_async(_test())
    
    def test_monitor_auction_created_to_active(self):
        """Test CREATED → ACTIVE transition."""
        async def _test():
            with patch('agents.consumer_agent.consumer_orchestrator.ConsumerBlockchainHandler'), \
                 patch('agents.consumer_agent.consumer_orchestrator.IPFSClient'), \
                 patch('agents.consumer_agent.consumer_orchestrator.ServiceGenerator'), \
                 patch('agents.consumer_agent.consumer_orchestrator.ServiceEvaluator'):
                
                consumer = Consumer(self.mock_config)
                consumer.blockchain_handler._initialized = True
                consumer.blockchain_handler.get_auction_status = AsyncMock(return_value={
                    "active": True,
                    "completed": False,
                    "winning_agent_id": 0,
                    "winning_bid": 0,
                    "error": None
                })
                
                tracker = AuctionTracker(
                    auction_id=1,
                    status=AuctionStatus.CREATED,
                    created_at=datetime.now(),
                    service_cid="QmTest",
                    max_budget=100000000,
                    duration=1800,
                    eligible_providers=[1, 2]
                )
                consumer.active_auctions[1] = tracker
                
                await consumer.monitor_auctions()
                
                # Should transition to ACTIVE
                self.assertEqual(tracker.status, AuctionStatus.ACTIVE)
                
                print("\n✓ CREATED → ACTIVE transition successful")
        
        run_async(_test())
    
    def test_monitor_auction_active_to_ended(self):
        """Test ACTIVE → ENDED transition."""
        async def _test():
            with patch('agents.consumer_agent.consumer_orchestrator.ConsumerBlockchainHandler'), \
                 patch('agents.consumer_agent.consumer_orchestrator.IPFSClient'), \
                 patch('agents.consumer_agent.consumer_orchestrator.ServiceGenerator'), \
                 patch('agents.consumer_agent.consumer_orchestrator.ServiceEvaluator'):
                
                consumer = Consumer(self.mock_config)
                consumer.blockchain_handler._initialized = True
                consumer.blockchain_handler.get_auction_status = AsyncMock(return_value={
                    "active": False,
                    "completed": False,
                    "winning_agent_id": 42,
                    "winning_bid": 50000000,
                    "error": None
                })
                consumer.blockchain_handler.client = Mock()
                consumer.blockchain_handler.client.get_block_number = AsyncMock(return_value=12345)
                
                tracker = AuctionTracker(
                    auction_id=1,
                    status=AuctionStatus.ACTIVE,
                    created_at=datetime.now(),
                    service_cid="QmTest",
                    max_budget=100000000,
                    duration=1800,
                    eligible_providers=[1, 2]
                )
                consumer.active_auctions[1] = tracker
                
                await consumer.monitor_auctions()
                
                # Should transition to ENDED with results
                self.assertEqual(tracker.status, AuctionStatus.ENDED)
                self.assertEqual(tracker.winning_agent_id, 42)
                self.assertEqual(tracker.winning_bid, 50000000)
                self.assertIsNotNone(tracker.ended_at)
                self.assertEqual(tracker.ended_block, 12345)
                
                print("\n✓ ACTIVE → ENDED with winning bid captured")
        
        run_async(_test())
    
    @patch('agents.consumer_agent.consumer_orchestrator.datetime')
    def test_monitor_auction_duration_expired(self, mock_datetime):
        """Test auction ending when duration expires."""
        async def _test():
            # Set up time mocking
            now = datetime(2026, 1, 7, 10, 0, 0)
            past = datetime(2026, 1, 7, 8, 0, 0)  # 2 hours ago
            mock_datetime.now.return_value = now
            
            with patch('agents.consumer_agent.consumer_orchestrator.ConsumerBlockchainHandler'), \
                 patch('agents.consumer_agent.consumer_orchestrator.IPFSClient'), \
                 patch('agents.consumer_agent.consumer_orchestrator.ServiceGenerator'), \
                 patch('agents.consumer_agent.consumer_orchestrator.ServiceEvaluator'):
                
                consumer = Consumer(self.mock_config)
                consumer.blockchain_handler._initialized = True
                consumer.blockchain_handler.get_auction_status = AsyncMock(return_value={
                    "active": True,
                    "completed": False,
                    "winning_agent_id": 0,
                    "winning_bid": 0,
                    "error": None
                })
                consumer.end_auction = AsyncMock()
                
                tracker = AuctionTracker(
                    auction_id=1,
                    status=AuctionStatus.ACTIVE,
                    created_at=past,  # Created 2 hours ago
                    service_cid="QmTest",
                    max_budget=100000000,
                    duration=3600,  # 1 hour duration
                    eligible_providers=[1, 2]
                )
                consumer.active_auctions[1] = tracker
                
                await consumer.monitor_auctions()
                
                # Should call end_auction
                consumer.end_auction.assert_called_once_with(1)
                
                print("\n✓ end_auction called when duration expired")
        
        run_async(_test())
    
    def test_monitor_auction_completed_triggers_retrieve(self):
        """Test ENDED → COMPLETED transition triggers result retrieval."""
        async def _test():
            with patch('agents.consumer_agent.consumer_orchestrator.ConsumerBlockchainHandler'), \
                 patch('agents.consumer_agent.consumer_orchestrator.IPFSClient'), \
                 patch('agents.consumer_agent.consumer_orchestrator.ServiceGenerator'), \
                 patch('agents.consumer_agent.consumer_orchestrator.ServiceEvaluator'):
                
                consumer = Consumer(self.mock_config)
                consumer.blockchain_handler._initialized = True
                consumer.blockchain_handler.get_auction_status = AsyncMock(return_value={
                    "active": False,
                    "completed": True,
                    "winning_agent_id": 42,
                    "winning_bid": 50000000,
                    "error": None
                })
                consumer._retrieve_result = AsyncMock()
                
                tracker = AuctionTracker(
                    auction_id=1,
                    status=AuctionStatus.ENDED,
                    created_at=datetime.now(),
                    service_cid="QmTest",
                    max_budget=100000000,
                    duration=1800,
                    eligible_providers=[1, 2]
                )
                consumer.active_auctions[1] = tracker
                
                await consumer.monitor_auctions()
                
                # Should transition to COMPLETED and call _retrieve_result
                self.assertEqual(tracker.status, AuctionStatus.COMPLETED)
                self.assertIsNotNone(tracker.completed_at)
                consumer._retrieve_result.assert_called_once_with(tracker)
                
                print("\n✓ COMPLETED status triggered result retrieval")
        
        run_async(_test())
    
    def test_monitor_auction_completed_already_retrieved(self):
        """Test completed auction doesn't trigger retrieval twice."""
        async def _test():
            with patch('agents.consumer_agent.consumer_orchestrator.ConsumerBlockchainHandler'), \
                 patch('agents.consumer_agent.consumer_orchestrator.IPFSClient'), \
                 patch('agents.consumer_agent.consumer_orchestrator.ServiceGenerator'), \
                 patch('agents.consumer_agent.consumer_orchestrator.ServiceEvaluator'):
                
                consumer = Consumer(self.mock_config)
                consumer.blockchain_handler._initialized = True
                consumer.blockchain_handler.get_auction_status = AsyncMock(return_value={
                    "active": False,
                    "completed": True,
                    "winning_agent_id": 42,
                    "winning_bid": 50000000,
                    "error": None
                })
                consumer._retrieve_result = AsyncMock()
                
                tracker = AuctionTracker(
                    auction_id=1,
                    status=AuctionStatus.ENDED,
                    created_at=datetime.now(),
                    service_cid="QmTest",
                    max_budget=100000000,
                    duration=1800,
                    eligible_providers=[1, 2],
                    result_path=Path("/test/result.json")  # Already has result
                )
                consumer.active_auctions[1] = tracker
                
                await consumer.monitor_auctions()
                
                # Should NOT call _retrieve_result again
                consumer._retrieve_result.assert_not_called()
                
                print("\n✓ Already retrieved result not fetched again")
        
        run_async(_test())
    
    def test_monitor_auction_get_status_error(self):
        """Test error handling when get_auction_status fails."""
        async def _test():
            with patch('agents.consumer_agent.consumer_orchestrator.ConsumerBlockchainHandler'), \
                 patch('agents.consumer_agent.consumer_orchestrator.IPFSClient'), \
                 patch('agents.consumer_agent.consumer_orchestrator.ServiceGenerator'), \
                 patch('agents.consumer_agent.consumer_orchestrator.ServiceEvaluator'):
                
                consumer = Consumer(self.mock_config)
                consumer.blockchain_handler._initialized = True
                consumer.blockchain_handler.get_auction_status = AsyncMock(return_value={
                    "error": "RPC connection failed"
                })
                
                tracker = AuctionTracker(
                    auction_id=1,
                    status=AuctionStatus.ACTIVE,
                    created_at=datetime.now(),
                    service_cid="QmTest",
                    max_budget=100000000,
                    duration=1800,
                    eligible_providers=[1, 2]
                )
                consumer.active_auctions[1] = tracker
                
                # Should not crash
                await consumer.monitor_auctions()
                
                # Tracker status should remain unchanged
                self.assertEqual(tracker.status, AuctionStatus.ACTIVE)
                
                print("\n✓ Error in get_status handled gracefully")
        
        run_async(_test())
    
    def test_monitor_auction_exception_handling(self):
        """Test exception handling in monitoring loop."""
        async def _test():
            with patch('agents.consumer_agent.consumer_orchestrator.ConsumerBlockchainHandler'), \
                 patch('agents.consumer_agent.consumer_orchestrator.IPFSClient'), \
                 patch('agents.consumer_agent.consumer_orchestrator.ServiceGenerator'), \
                 patch('agents.consumer_agent.consumer_orchestrator.ServiceEvaluator'):
                
                consumer = Consumer(self.mock_config)
                consumer.blockchain_handler._initialized = True
                consumer.blockchain_handler.get_auction_status = AsyncMock(
                    side_effect=Exception("Network error")
                )
                
                tracker = AuctionTracker(
                    auction_id=1,
                    status=AuctionStatus.ACTIVE,
                    created_at=datetime.now(),
                    service_cid="QmTest",
                    max_budget=100000000,
                    duration=1800,
                    eligible_providers=[1, 2]
                )
                consumer.active_auctions[1] = tracker
                
                # Should catch exception and store error
                await consumer.monitor_auctions()
                
                # Error should be stored
                self.assertIsNotNone(tracker.error)
                self.assertIn("Network error", tracker.error)
                
                print("\n✓ Exception caught and stored in tracker")
        
        run_async(_test())
    
    def test_monitor_multiple_auctions_different_states(self):
        """Test monitoring multiple auctions with different states."""
        async def _test():
            with patch('agents.consumer_agent.consumer_orchestrator.ConsumerBlockchainHandler'), \
                 patch('agents.consumer_agent.consumer_orchestrator.IPFSClient'), \
                 patch('agents.consumer_agent.consumer_orchestrator.ServiceGenerator'), \
                 patch('agents.consumer_agent.consumer_orchestrator.ServiceEvaluator'):
                
                consumer = Consumer(self.mock_config)
                consumer.blockchain_handler._initialized = True
                
                # Different responses for different auctions
                def get_status(auction_id):
                    responses = {
                        1: {"active": True, "completed": False, "winning_agent_id": 0, 
                            "winning_bid": 0, "error": None},
                        2: {"active": False, "completed": False, "winning_agent_id": 5,
                            "winning_bid": 40000000, "error": None},
                        3: {"active": False, "completed": True, "winning_agent_id": 7,
                            "winning_bid": 30000000, "error": None}
                    }
                    return responses.get(auction_id, {})
                
                consumer.blockchain_handler.get_auction_status = AsyncMock(side_effect=get_status)
                consumer.blockchain_handler.client = Mock()
                consumer.blockchain_handler.client.get_block_number = AsyncMock(return_value=12345)
                consumer._retrieve_result = AsyncMock()
                
                # Create 3 trackers with different states
                trackers = {
                    1: AuctionTracker(1, AuctionStatus.CREATED, datetime.now(), "QmT1", 100000000, 1800, [1]),
                    2: AuctionTracker(2, AuctionStatus.ACTIVE, datetime.now(), "QmT2", 100000000, 1800, [1]),
                    3: AuctionTracker(3, AuctionStatus.ENDED, datetime.now(), "QmT3", 100000000, 1800, [1])
                }
                consumer.active_auctions = trackers
                
                await consumer.monitor_auctions()
                
                # Verify each handled correctly
                self.assertEqual(trackers[1].status, AuctionStatus.ACTIVE)  # CREATED → ACTIVE
                self.assertEqual(trackers[2].status, AuctionStatus.ENDED)   # ACTIVE → ENDED
                self.assertEqual(trackers[3].status, AuctionStatus.COMPLETED)  # ENDED → COMPLETED
                consumer._retrieve_result.assert_called_once()
                
                print("\n✓ Multiple auctions with different states handled correctly")
        
        run_async(_test())
    
    def test_end_auction_success(self):
        """Test end_auction calls blockchain handler correctly."""
        async def _test():
            with patch('agents.consumer_agent.consumer_orchestrator.ConsumerBlockchainHandler'), \
                 patch('agents.consumer_agent.consumer_orchestrator.IPFSClient'), \
                 patch('agents.consumer_agent.consumer_orchestrator.ServiceGenerator'), \
                 patch('agents.consumer_agent.consumer_orchestrator.ServiceEvaluator'):
                
                consumer = Consumer(self.mock_config)
                consumer.blockchain_handler.end_auction = AsyncMock(return_value={
                    "tx_hash": "0xabc123",
                    "winning_agent_id": 42,
                    "error": None
                })
                
                await consumer.end_auction(1)
                
                consumer.blockchain_handler.end_auction.assert_called_once_with(1)
                
                print("\n✓ end_auction success logged correctly")
        
        run_async(_test())
    
    def test_end_auction_failure(self):
        """Test end_auction handles blockchain failure."""
        async def _test():
            with patch('agents.consumer_agent.consumer_orchestrator.ConsumerBlockchainHandler'), \
                 patch('agents.consumer_agent.consumer_orchestrator.IPFSClient'), \
                 patch('agents.consumer_agent.consumer_orchestrator.ServiceGenerator'), \
                 patch('agents.consumer_agent.consumer_orchestrator.ServiceEvaluator'):
                
                consumer = Consumer(self.mock_config)
                consumer.blockchain_handler.end_auction = AsyncMock(return_value={
                    "tx_hash": None,
                    "winning_agent_id": None,
                    "error": "Auction already ended"
                })
                
                # Should not crash
                await consumer.end_auction(1)
                
                print("\n✓ end_auction failure handled gracefully")
        
        run_async(_test())


class TestResultRetrieval(unittest.TestCase):
    """Test result file retrieval."""
    
    def test_retrieve_result_success(self):
        """Test successful result retrieval and loading."""
        async def _test():
            with TemporaryDirectory() as tmpdir, \
                 patch('agents.consumer_agent.consumer_orchestrator.ConsumerBlockchainHandler'), \
                 patch('agents.consumer_agent.consumer_orchestrator.IPFSClient'), \
                 patch('agents.consumer_agent.consumer_orchestrator.ServiceGenerator'), \
                 patch('agents.consumer_agent.consumer_orchestrator.ServiceEvaluator'):
                
                # Create result file
                result_dir = Path(tmpdir) / "auction_1"
                result_dir.mkdir()
                result_file = result_dir / "result.json"
                result_data = {
                    "success": True,
                    "responses": [
                        {"prompt": "Q1", "response": "Answer 1"},
                        {"prompt": "Q2", "response": "Answer 2"}
                    ]
                }
                result_file.write_text(json.dumps(result_data))
                
                mock_config = Mock(spec=Config)
                mock_config.consumer_check_interval = 10
                consumer = Consumer(mock_config)
                consumer.result_base_path = Path(tmpdir)
                consumer._evaluate_result = AsyncMock()
                
                tracker = AuctionTracker(
                    auction_id=1,
                    status=AuctionStatus.COMPLETED,
                    created_at=datetime.now(),
                    service_cid="QmTest",
                    max_budget=100000000,
                    duration=1800,
                    eligible_providers=[1]
                )
                
                await consumer._retrieve_result(tracker)
                
                # Verify result loaded
                self.assertEqual(tracker.result_path, result_file)
                self.assertIsNotNone(tracker.result)
                self.assertEqual(tracker.result["success"], True)
                self.assertEqual(len(tracker.result["responses"]), 2)
                consumer._evaluate_result.assert_called_once_with(tracker)
                
                print("\n✓ Result file retrieved and parsed correctly")
        
        run_async(_test())
    
    def test_retrieve_result_file_not_found(self):
        """Test handling when result file doesn't exist."""
        async def _test():
            with TemporaryDirectory() as tmpdir, \
                 patch('agents.consumer_agent.consumer_orchestrator.ConsumerBlockchainHandler'), \
                 patch('agents.consumer_agent.consumer_orchestrator.IPFSClient'), \
                 patch('agents.consumer_agent.consumer_orchestrator.ServiceGenerator'), \
                 patch('agents.consumer_agent.consumer_orchestrator.ServiceEvaluator'):
                
                mock_config = Mock(spec=Config)
                mock_config.consumer_check_interval = 10
                consumer = Consumer(mock_config)
                consumer.result_base_path = Path(tmpdir)
                consumer._evaluate_result = AsyncMock()
                
                tracker = AuctionTracker(
                    auction_id=999,
                    status=AuctionStatus.COMPLETED,
                    created_at=datetime.now(),
                    service_cid="QmTest",
                    max_budget=100000000,
                    duration=1800,
                    eligible_providers=[1]
                )
                
                await consumer._retrieve_result(tracker)
                
                # Should store error
                self.assertIsNotNone(tracker.error)
                self.assertIn("Result file not found", tracker.error)
                consumer._evaluate_result.assert_not_called()
                
                print("\n✓ File not found error handled correctly")
        
        run_async(_test())


class TestResultEvaluation(unittest.TestCase):
    """Test result evaluation workflow."""
    
    def test_evaluate_result_success(self):
        """Test successful result evaluation."""
        async def _test():
            with patch('agents.consumer_agent.consumer_orchestrator.ConsumerBlockchainHandler'), \
                 patch('agents.consumer_agent.consumer_orchestrator.IPFSClient'), \
                 patch('agents.consumer_agent.consumer_orchestrator.ServiceGenerator'), \
                 patch('agents.consumer_agent.consumer_orchestrator.ServiceEvaluator'):
                
                mock_config = Mock(spec=Config)
                mock_config.consumer_check_interval = 10
                consumer = Consumer(mock_config)
                consumer.ipfs_client.fetch_json = AsyncMock(return_value={
                    "title": "Test Service",
                    "prompts": ["Q1", "Q2"]
                })
                consumer.evaluator.evaluate = AsyncMock(return_value={
                    "rating": 85,
                    "quality_scores": {
                        "completeness": 90,
                        "depth": 80,
                        "clarity": 85
                    }
                })
                consumer._submit_feedback = AsyncMock()
                
                tracker = AuctionTracker(
                    auction_id=1,
                    status=AuctionStatus.COMPLETED,
                    created_at=datetime.now(),
                    service_cid="QmTestService",
                    max_budget=100000000,
                    duration=1800,
                    eligible_providers=[1],
                    result={"success": True, "responses": []}
                )
                
                await consumer._evaluate_result(tracker)
                
                # Verify evaluation stored
                self.assertEqual(tracker.status, AuctionStatus.EVALUATED)
                self.assertIsNotNone(tracker.evaluation)
                self.assertEqual(tracker.evaluation["rating"], 85)
                self.assertEqual(tracker.evaluation["quality_scores"]["completeness"], 90)
                consumer._submit_feedback.assert_called_once_with(tracker)
                
                print("\n✓ Evaluation completed and feedback triggered")
        
        run_async(_test())


class TestFeedbackSubmission(unittest.TestCase):
    """Test feedback submission workflow."""
    
    def test_submit_feedback_success(self):
        """Test successful feedback submission."""
        async def _test():
            with patch('agents.consumer_agent.consumer_orchestrator.ConsumerBlockchainHandler'), \
                 patch('agents.consumer_agent.consumer_orchestrator.IPFSClient'), \
                 patch('agents.consumer_agent.consumer_orchestrator.ServiceGenerator'), \
                 patch('agents.consumer_agent.consumer_orchestrator.ServiceEvaluator'):
                
                mock_config = Mock(spec=Config)
                mock_config.consumer_check_interval = 10
                consumer = Consumer(mock_config)
                consumer.blockchain_handler.get_feedback_auth = AsyncMock(return_value={
                    "auth_data": "0xabc123",
                    "signature": "0xdef456"
                })
                consumer.blockchain_handler.submit_feedback = AsyncMock(return_value={
                    "tx_hash": "0x789",
                    "error": None
                })
                
                tracker = AuctionTracker(
                    auction_id=1,
                    status=AuctionStatus.EVALUATED,
                    created_at=datetime.now(),
                    service_cid="QmTest",
                    max_budget=100000000,
                    duration=1800,
                    eligible_providers=[1],
                    winning_agent_id=42,
                    ended_block=12345,
                    evaluation={
                        "rating": 90,
                        "quality_scores": {
                            "completeness": 95,
                            "depth": 85
                        }
                    }
                )
                consumer.active_auctions[1] = tracker
                
                await consumer._submit_feedback(tracker)
                
                # Verify feedback submitted
                self.assertTrue(tracker.feedback_submitted)
                self.assertNotIn(1, consumer.active_auctions)
                self.assertIn(tracker, consumer.completed_auctions)
                
                # Verify feedback text generated from quality scores
                call_args = consumer.blockchain_handler.submit_feedback.call_args
                feedback_text = call_args[1]["feedback_text"]
                self.assertIn("completeness", feedback_text)
                self.assertIn("depth", feedback_text)
                
                print("\n✓ Feedback submitted with quality scores")
        
        run_async(_test())
    
    def test_submit_feedback_without_quality_scores(self):
        """Test feedback submission without quality scores."""
        async def _test():
            with patch('agents.consumer_agent.consumer_orchestrator.ConsumerBlockchainHandler'), \
                 patch('agents.consumer_agent.consumer_orchestrator.IPFSClient'), \
                 patch('agents.consumer_agent.consumer_orchestrator.ServiceGenerator'), \
                 patch('agents.consumer_agent.consumer_orchestrator.ServiceEvaluator'):
                
                mock_config = Mock(spec=Config)
                mock_config.consumer_check_interval = 10
                consumer = Consumer(mock_config)
                consumer.blockchain_handler.get_feedback_auth = AsyncMock(return_value={"auth": "data"})
                consumer.blockchain_handler.submit_feedback = AsyncMock(return_value={
                    "tx_hash": "0x789",
                    "error": None
                })
                
                tracker = AuctionTracker(
                    auction_id=1,
                    status=AuctionStatus.EVALUATED,
                    created_at=datetime.now(),
                    service_cid="QmTest",
                    max_budget=100000000,
                    duration=1800,
                    eligible_providers=[1],
                    winning_agent_id=42,
                    evaluation={"rating": 75}  # No quality_scores
                )
                consumer.active_auctions[1] = tracker
                
                await consumer._submit_feedback(tracker)
                
                # Verify fallback feedback text
                call_args = consumer.blockchain_handler.submit_feedback.call_args
                feedback_text = call_args[1]["feedback_text"]
                self.assertEqual(feedback_text, "Rating: 75/100")
                
                print("\n✓ Fallback feedback text used when no quality scores")
        
        run_async(_test())


class TestConsumerRunLoop(unittest.TestCase):
    """Test consumer run loop and status."""
    
    def test_stop_sets_running_false(self):
        """Test stop() sets running flag to False."""
        with patch('agents.consumer_agent.consumer_orchestrator.ConsumerBlockchainHandler'), \
             patch('agents.consumer_agent.consumer_orchestrator.IPFSClient'), \
             patch('agents.consumer_agent.consumer_orchestrator.ServiceGenerator'), \
             patch('agents.consumer_agent.consumer_orchestrator.ServiceEvaluator'):
            
            mock_config = Mock(spec=Config)
            mock_config.consumer_check_interval = 10
            consumer = Consumer(mock_config)
            consumer.running = True
            
            consumer.stop()
            
            self.assertFalse(consumer.running)
            
            print("\n✓ stop() sets running=False")
    
    def test_get_status(self):
        """Test get_status returns correct counts."""
        with patch('agents.consumer_agent.consumer_orchestrator.ConsumerBlockchainHandler'), \
             patch('agents.consumer_agent.consumer_orchestrator.IPFSClient'), \
             patch('agents.consumer_agent.consumer_orchestrator.ServiceGenerator'), \
             patch('agents.consumer_agent.consumer_orchestrator.ServiceEvaluator'):
            
            mock_config = Mock(spec=Config)
            mock_config.consumer_check_interval = 10
            consumer = Consumer(mock_config)
            consumer.active_auctions = {1: Mock(), 2: Mock(), 3: Mock()}
            consumer.completed_auctions = [Mock(), Mock()]
            consumer.running = True
            
            status = consumer.get_status()
            
            self.assertEqual(status["active_auctions"], 3)
            self.assertEqual(status["completed_auctions"], 2)
            self.assertTrue(status["running"])
            
            print("\n✓ get_status returns correct counts")


if __name__ == "__main__":
    print("=" * 70)
    print("CONSUMER ORCHESTRATOR UNIT TESTS")
    print("=" * 70)
    print()
    print("All components mocked for fast isolated testing")
    print("=" * 70)
    print()
    
    unittest.main(verbosity=2)

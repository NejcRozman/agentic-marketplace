"""
Tests for the Orchestrator.

Unit tests mock components for fast isolated testing.
Integration test uses real blockchain and IPFS.

Prerequisites for integration test:
1. Anvil running with contracts deployed
2. Agent registered and has won an auction
3. SERVICE_DESCRIPTION_CID in contracts/.env

Run with: python agents/tests/test_orchestrator.py
"""

import sys
from pathlib import Path


import unittest
import asyncio
import json
import os
import logging
from unittest.mock import Mock, AsyncMock
from datetime import datetime
from tempfile import TemporaryDirectory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from agents.provider_agent.orchestrator import Orchestrator, Job, JobStatus
from agents.config import Config


def run_async(coro):
    """Helper to run async functions in sync tests."""
    return asyncio.get_event_loop().run_until_complete(coro)


class TestOrchestratorUnit(unittest.TestCase):
    """Unit tests with mocked components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_config = Mock(spec=Config)
        self.mock_config.agent_id = 4427
        
        self.sample_auction_details = {
            "auction_id": 1,
            "buyer_address": "0x1234567890123456789012345678901234567890",
            "service_cid": "QmTestServiceCID123",
            "max_price": 100000000,
            "winning_bid": 50000000
        }
        
        self.sample_service_requirements = {
            "title": "Test Literature Review",
            "description": "Test service",
            "prompts": [
                "What is the main topic?",
                "What are the key findings?"
            ],
            "input_files_cid": "QmTestPDFCID456"
        }
    
    def test_orchestrator_initialization(self):
        """Test that orchestrator initializes correctly."""
        orchestrator = Orchestrator(config=self.mock_config)
        
        self.assertEqual(orchestrator.config, self.mock_config)
        self.assertIsNotNone(orchestrator.blockchain_handler)
        self.assertIsNotNone(orchestrator.service_executor)
        self.assertIsNotNone(orchestrator.ipfs_client)
        self.assertEqual(orchestrator.active_jobs, {})
        self.assertEqual(orchestrator.completed_jobs, [])
        self.assertFalse(orchestrator.running)
    
    def test_start_job_creates_job(self):
        """Test that starting a job creates a Job object."""
        orchestrator = Orchestrator(config=self.mock_config)
        orchestrator.blockchain_handler = AsyncMock()
        orchestrator.service_executor = Mock()
        orchestrator.ipfs_client = AsyncMock()
        
        run_async(orchestrator._start_job(self.sample_auction_details))
        
        self.assertIn(1, orchestrator.active_jobs)
        job = orchestrator.active_jobs[1]
        self.assertEqual(job.auction_id, 1)
        self.assertEqual(job.status, JobStatus.WON)
        self.assertEqual(job.buyer_address, self.sample_auction_details["buyer_address"])
        self.assertEqual(job.service_cid, self.sample_auction_details["service_cid"])
    
    def test_fetch_requirements(self):
        """Test fetching service requirements from IPFS."""
        orchestrator = Orchestrator(config=self.mock_config)
        orchestrator.ipfs_client = AsyncMock()
        orchestrator.ipfs_client.fetch_json = AsyncMock(return_value=self.sample_service_requirements)
        
        job = Job(
            auction_id=1,
            status=JobStatus.WON,
            started_at=datetime.now(),
            buyer_address=self.sample_auction_details["buyer_address"],
            service_cid=self.sample_auction_details["service_cid"]
        )
        
        run_async(orchestrator._fetch_requirements(job))
        
        self.assertEqual(job.status, JobStatus.FETCHING_REQUIREMENTS)
        self.assertEqual(job.service_requirements, self.sample_service_requirements)
        self.assertEqual(job.prompts, self.sample_service_requirements["prompts"])
    
    def test_execute_service(self):
        """Test executing literature review service."""
        with TemporaryDirectory() as tmpdir:
            orchestrator = Orchestrator(config=self.mock_config)
            orchestrator.service_executor = Mock()
            orchestrator.service_executor.perform_review = Mock(return_value={
                "success": True,
                "responses": ["Answer 1", "Answer 2"],
                "agent_id": "4427"
            })
            
            pdf_dir = Path(tmpdir) / "auction_1"
            pdf_dir.mkdir()
            
            job = Job(
                auction_id=1,
                status=JobStatus.FETCHING_FILES,
                started_at=datetime.now(),
                buyer_address="0x123",
                service_cid="QmTest",
                pdf_directory=pdf_dir,
                prompts=["Q1", "Q2"]
            )
            
            run_async(orchestrator._execute_service(job))
            
            self.assertEqual(job.status, JobStatus.PROCESSING)
            self.assertIsNotNone(job.result)
            self.assertTrue(job.result["success"])
    
    def test_deliver_result(self):
        """Test delivering results to local file."""
        with TemporaryDirectory() as tmpdir:
            orchestrator = Orchestrator(config=self.mock_config)
            
            pdf_dir = Path(tmpdir) / "auction_1"
            pdf_dir.mkdir()
            
            job = Job(
                auction_id=1,
                status=JobStatus.PROCESSING,
                started_at=datetime.now(),
                buyer_address="0x123",
                service_cid="QmTest",
                pdf_directory=pdf_dir,
                result={"success": True, "responses": ["Answer 1"]}
            )
            
            run_async(orchestrator._deliver_result(job))
            
            self.assertEqual(job.status, JobStatus.DELIVERING)
            result_file = pdf_dir / "result.json"
            self.assertTrue(result_file.exists())
            
            with open(result_file) as f:
                saved_result = json.load(f)
            self.assertEqual(saved_result, job.result)
    
    def test_handle_job_error_increments_retry(self):
        """Test that error handling increments retry count."""
        orchestrator = Orchestrator(config=self.mock_config)
        
        job = Job(
            auction_id=1,
            status=JobStatus.FETCHING_FILES,
            started_at=datetime.now(),
            buyer_address="0x123",
            service_cid="QmTest"
        )
        
        run_async(orchestrator._handle_job_error(job, "Test error"))
        
        self.assertEqual(job.retry_count, 1)
        self.assertEqual(job.error, "Test error")
        self.assertEqual(job.status, JobStatus.FETCHING_FILES)  # Keep status for retry
    
    def test_handle_job_error_fails_after_max_retries(self):
        """Test that job fails after max retries."""
        orchestrator = Orchestrator(config=self.mock_config)
        
        job = Job(
            auction_id=1,
            status=JobStatus.FETCHING_FILES,
            started_at=datetime.now(),
            buyer_address="0x123",
            service_cid="QmTest",
            retry_count=2,
            max_retries=3
        )
        
        run_async(orchestrator._handle_job_error(job, "Test error"))
        
        self.assertEqual(job.retry_count, 3)
        self.assertEqual(job.status, JobStatus.FAILED)


class TestOrchestratorIntegration(unittest.TestCase):
    """Integration test with real blockchain and IPFS."""
    
    def test_full_workflow(self):
        """
        Integration test: Full orchestrator workflow.
        
        Prerequisites:
        1. Anvil running with contracts deployed
        2. Agent registered and has winning bid on an auction
        """
        # Initialize orchestrator with real components
        config = Config()
        orchestrator = Orchestrator(config=config)
        run_async(orchestrator.initialize())
        
        # Mock the literature agent to avoid long LLM calls
        mock_result = {
            "success": True,
            "agent_id": str(config.agent_id),
            "responses": ["Mock answer 1", "Mock answer 2"],
            "pdf_directory": ""
        }
        orchestrator.service_executor.perform_review = Mock(return_value=mock_result)
        
        # Track iterations
        max_iterations = 10
        iteration = 0
        job_completed = False
        
        try:
            while iteration < max_iterations and not job_completed:
                iteration += 1
                print(f"\n--- Iteration {iteration} ---")
                
                # Run one cycle
                run_async(orchestrator._check_won_auctions())
                run_async(orchestrator._process_active_jobs())
                orchestrator._cleanup_completed_jobs()
                
                # Check status
                status = orchestrator.get_status()
                print(f"Active jobs: {status['active_jobs']}")
                print(f"Completed jobs: {status['completed_jobs']}")
                
                if status['active_jobs'] > 0:
                    for job in orchestrator.active_jobs.values():
                        print(f"  Job {job.auction_id}: {job.status.value}")
                
                # Check if job completed
                if status['completed_jobs'] > 0:
                    completed_job = orchestrator.completed_jobs[-1]
                    print(f"\n✅ Job {completed_job.auction_id} completed!")
                    print(f"   Final status: {completed_job.status.value}")
                    print(f"   Buyer: {completed_job.buyer_address}")
                    print(f"   Service CID: {completed_job.service_cid}")
                    
                    if completed_job.pdf_directory:
                        print(f"   PDF directory: {completed_job.pdf_directory}")
                        result_file = completed_job.pdf_directory / "result.json"
                        if result_file.exists():
                            print(f"   Result file created: {result_file}")
                    
                    job_completed = True
                    
                    # Assertions
                    self.assertEqual(completed_job.status, JobStatus.COMPLETED)
                    self.assertIsNotNone(completed_job.service_requirements)
                    self.assertGreater(len(completed_job.prompts), 0)
                    self.assertIsNotNone(completed_job.result)
                    self.assertTrue(completed_job.result["success"])
                    break
                
                # Small delay between iterations
                asyncio.get_event_loop().run_until_complete(asyncio.sleep(0.5))
            
            if not job_completed:
                status = orchestrator.get_status()
                print(f"\nNo job completed after {max_iterations} iterations")
                print(f"Final state: {status['active_jobs']} active, {status['completed_jobs']} completed")
                self.skipTest("No won auctions detected - create an auction with winning bid first")
        
        finally:
            orchestrator.stop()


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)

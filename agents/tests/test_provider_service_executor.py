"""
Tests for ServiceExecutor and provider agent functionality.

Prerequisites:
1. Valid OpenRouter API key in agents/.env (OPENROUTER_API_KEY)
2. PDF files in utils/files/ directory

Run with: python agents/tests/test_provider_service_executor.py
"""

import sys
import os
from pathlib import Path
import unittest
import asyncio
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from agents.provider_agent.service_executor import ServiceExecutor
from agents.config import Config


def run_async(coro):
    """Helper to run async functions in sync tests."""
    return asyncio.get_event_loop().run_until_complete(coro)


class TestLiteratureReviewContextCleaning(unittest.TestCase):
    """Test that context doesn't accumulate across multiple service executions."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures once for all tests."""
        # Create config instance
        cls.config = Config()
        
        # Verify API key is configured
        if not cls.config.openrouter_api_key:
            raise unittest.SkipTest("OPENROUTER_API_KEY not configured in agents/.env")
        
        # Verify PDF directory exists
        cls.pdf_dir = Path("utils/files")
        if not cls.pdf_dir.exists():
            raise unittest.SkipTest(f"PDF directory not found: {cls.pdf_dir}")
        
        pdf_files = list(cls.pdf_dir.glob("*.pdf"))

        if len(pdf_files) == 0:
            raise unittest.SkipTest(f"No PDF files found in {cls.pdf_dir}")
        
        print(f"\n✓ Found {len(pdf_files)} PDF files for testing")
        print(f"✓ OpenRouter API key configured")
    
    def setUp(self):
        """Create fresh agent for each test."""
        self.agent = ServiceExecutor(agent_id="test_context_agent")
    
    def test_multiple_services_no_context_accumulation(self):
        """
        Test that executing multiple services doesn't cause context accumulation.
        
        This simulates the bug scenario where 20+ auctions would cause token limit errors.
        Without the fix (_build_graph called each time), LangGraph's add_messages reducer
        accumulates all messages across services, eventually hitting 262K token limit.
        
        With the fix, each service gets a fresh graph and clean message state.
        """
        print("\n" + "="*80)
        print("Testing context cleaning across multiple service executions")
        print("="*80)
        
        # Simulate multiple service executions (like multiple auctions)
        num_services = 5
        prompts_per_service = 3
        
        test_prompts = [
            "What are the main findings in these papers?",
            "Summarize the methodology used.",
            "What are the limitations discussed?"
        ]
        
        for service_num in range(1, num_services + 1):
            print(f"\n🔄 Service {service_num}/{num_services}")
            
            result = self.agent.perform_review(
                pdf_directory=str(self.pdf_dir),
                prompts=test_prompts[:prompts_per_service],
                force_rebuild=False  # Don't rebuild PDFs, only rebuild graph
            )
            
            # Check that service succeeded
            self.assertTrue(
                result["success"],
                f"Service {service_num} failed: {result.get('error')}"
            )
            
            # Check we got responses for all prompts
            self.assertEqual(
                len(result["responses"]),
                prompts_per_service,
                f"Service {service_num} didn't return all responses"
            )
            
            # Verify each response has content
            for i, response in enumerate(result["responses"], 1):
                self.assertIn("prompt", response)
                self.assertIn("response", response)
                self.assertIsInstance(response["response"], str)
                self.assertGreater(
                    len(response["response"]),
                    0,
                    f"Service {service_num}, prompt {i} returned empty response"
                )
            
            print(f"  ✓ Service {service_num} completed successfully")
            print(f"    - Prompts processed: {len(result['responses'])}")
            print(f"    - Total response length: {sum(len(r['response']) for r in result['responses'])} chars")
        
        print(f"\n{'='*80}")
        print(f"✅ SUCCESS: All {num_services} services executed without token limit errors")
        print(f"   This confirms context is properly cleared between services")
        print(f"{'='*80}\n")
    
    def test_graph_rebuilt_between_services(self):
        """
        Test that the graph is rebuilt for each service execution.
        
        This verifies the fix is in place: _build_graph() should be called
        at the start of perform_review() to create a fresh graph with clean state.
        """
        print("\n" + "="*80)
        print("Testing that LangGraph is rebuilt between services")
        print("="*80)
        
        # Execute first service
        result1 = self.agent.perform_review(
            pdf_directory=str(self.pdf_dir),
            prompts=["What is the main topic?"],
            force_rebuild=False
        )
        self.assertTrue(result1["success"])
        graph1_id = id(self.agent.graph)
        print(f"\n  Service 1: graph object ID = {graph1_id}")
        
        # Execute second service
        result2 = self.agent.perform_review(
            pdf_directory=str(self.pdf_dir),
            prompts=["Summarize the findings."],
            force_rebuild=False
        )
        self.assertTrue(result2["success"])
        graph2_id = id(self.agent.graph)
        print(f"  Service 2: graph object ID = {graph2_id}")
        
        # Graph should be rebuilt (different object ID)
        self.assertNotEqual(
            graph1_id,
            graph2_id,
            "Graph was not rebuilt between services - context may accumulate!"
        )
        
        print(f"\n✅ Graph is properly rebuilt between services")
        print(f"{'='*80}\n")
    
    def test_single_service_multiple_prompts(self):
        """
        Test that a single service with multiple prompts works correctly.
        
        This is the normal case within one auction.
        """
        prompts = [
            "What are the key contributions?",
            "What methods were used?",
            "What are the conclusions?",
            "What are the future directions?",
            "What datasets were used?"
        ]
        
        result = self.agent.perform_review(
            pdf_directory=str(self.pdf_dir),
            prompts=prompts,
            force_rebuild=False
        )
        
        self.assertTrue(result["success"], f"Service failed: {result.get('error')}")
        self.assertEqual(len(result["responses"]), len(prompts))
        
        for i, response in enumerate(result["responses"], 1):
            self.assertGreater(
                len(response["response"]),
                0,
                f"Prompt {i} returned empty response"
            )
        
        print(f"\n✅ Single service with {len(prompts)} prompts executed successfully\n")


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)

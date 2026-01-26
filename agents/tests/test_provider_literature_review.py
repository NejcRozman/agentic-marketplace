"""
Tests for LiteratureReviewAgent and provider agent functionality.

This test suite validates:
1. Quality profile configuration: Different quality tiers set parameters correctly
2. Bug #14 fix: LangGraph doesn't accumulate messages across service executions

Prerequisites:
1. Valid OpenRouter API key in agents/.env (OPENROUTER_API_KEY)
2. PDF files in utils/files/ directory

Run with: python agents/tests/test_provider_literature_review.py
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

from agents.provider_agent.literature_review import LiteratureReviewAgent
from agents.config import Config


def run_async(coro):
    """Helper to run async functions in sync tests."""
    return asyncio.get_event_loop().run_until_complete(coro)


class TestQualityProfiles(unittest.TestCase):
    """Test that quality profiles configure parameters correctly."""
    
    def test_high_quality_profile(self):
        """Test high quality profile sets correct parameters."""
        os.environ['QUALITY_PROFILE'] = 'high'
        config = Config()
        
        self.assertEqual(config.rag_temperature, 0.0, "High quality should have temp=0.0")
        self.assertEqual(config.rag_retrieval_k, 5, "High quality should have k=5")
        self.assertEqual(config.rag_chunk_size, 8000, "High quality should have chunk_size=8000")
        self.assertEqual(config.rag_chunk_overlap, 400, "High quality should have chunk_overlap=400")
        self.assertEqual(config.rag_system_prompt_type, "detailed", "High quality should use detailed prompt")
        
        print(f"✅ HIGH quality: temp={config.rag_temperature}, k={config.rag_retrieval_k}, "
              f"chunk={config.rag_chunk_size}, overlap={config.rag_chunk_overlap}, "
              f"prompt={config.rag_system_prompt_type}")
    
    def test_medium_quality_profile(self):
        """Test medium quality profile sets correct parameters."""
        os.environ['QUALITY_PROFILE'] = 'medium'
        config = Config()
        
        self.assertEqual(config.rag_temperature, 0.3, "Medium quality should have temp=0.3")
        self.assertEqual(config.rag_retrieval_k, 3, "Medium quality should have k=3")
        self.assertEqual(config.rag_chunk_size, 10000, "Medium quality should have chunk_size=10000")
        self.assertEqual(config.rag_chunk_overlap, 200, "Medium quality should have chunk_overlap=200")
        self.assertEqual(config.rag_system_prompt_type, "standard", "Medium quality should use standard prompt")
        
        print(f"✅ MEDIUM quality: temp={config.rag_temperature}, k={config.rag_retrieval_k}, "
              f"chunk={config.rag_chunk_size}, overlap={config.rag_chunk_overlap}, "
              f"prompt={config.rag_system_prompt_type}")
    
    def test_low_quality_profile(self):
        """Test low quality profile sets correct parameters."""
        os.environ['QUALITY_PROFILE'] = 'low'
        config = Config()
        
        self.assertEqual(config.rag_temperature, 0.7, "Low quality should have temp=0.7")
        self.assertEqual(config.rag_retrieval_k, 1, "Low quality should have k=1")
        self.assertEqual(config.rag_chunk_size, 15000, "Low quality should have chunk_size=15000")
        self.assertEqual(config.rag_chunk_overlap, 0, "Low quality should have chunk_overlap=0")
        self.assertEqual(config.rag_system_prompt_type, "minimal", "Low quality should use minimal prompt")
        
        print(f"✅ LOW quality: temp={config.rag_temperature}, k={config.rag_retrieval_k}, "
              f"chunk={config.rag_chunk_size}, overlap={config.rag_chunk_overlap}, "
              f"prompt={config.rag_system_prompt_type}")
    
    def test_invalid_quality_profile(self):
        """Test invalid quality profile raises error."""
        os.environ['QUALITY_PROFILE'] = 'invalid'
        
        with self.assertRaises(ValueError) as context:
            Config()
        
        self.assertIn("Unknown quality profile", str(context.exception))
        print("✅ Invalid quality profile correctly raises ValueError")
    
    def test_default_quality_profile(self):
        """Test default quality profile is medium."""
        # Remove QUALITY_PROFILE to test default
        if 'QUALITY_PROFILE' in os.environ:
            del os.environ['QUALITY_PROFILE']
        
        config = Config()
        
        # Should default to medium
        self.assertEqual(config.quality_profile, "medium")
        self.assertEqual(config.rag_temperature, 0.3)
        self.assertEqual(config.rag_retrieval_k, 3)
        
        print("✅ Default quality profile is MEDIUM")
    
    def test_bidding_base_cost_high(self):
        """Test high quality profile has highest base cost."""
        os.environ['QUALITY_PROFILE'] = 'high'
        config = Config()
        
        self.assertEqual(config.bidding_base_cost, 60, "High quality should have base_cost=60")
        print("✅ High quality base cost: 60 USDC")
    
    def test_bidding_base_cost_medium(self):
        """Test medium quality profile has medium base cost."""
        os.environ['QUALITY_PROFILE'] = 'medium'
        config = Config()
        
        self.assertEqual(config.bidding_base_cost, 40, "Medium quality should have base_cost=40")
        print("✅ Medium quality base cost: 40 USDC")
    
    def test_bidding_base_cost_low(self):
        """Test low quality profile has lowest base cost."""
        os.environ['QUALITY_PROFILE'] = 'low'
        config = Config()
        
        self.assertEqual(config.bidding_base_cost, 20, "Low quality should have base_cost=20")
        print("✅ Low quality base cost: 20 USDC")


class TestLiteratureReviewContextCleaning(unittest.TestCase):
    """Test that context doesn't accumulate across multiple service executions."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures once for all tests."""
        # Reset to default quality profile for these tests
        os.environ['QUALITY_PROFILE'] = 'medium'
        
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
        self.agent = LiteratureReviewAgent(agent_id="test_context_agent")
    
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

"""
Tests for ServiceGenerator.

Unit tests use mocked LLM and IPFS for fast isolated testing.
Integration test uses real LLM and IPFS (requires API keys).

Prerequisites for integration test:
1. GEMINI_API_KEY in agents/.env
2. PINATA_JWT or (PINATA_API_KEY + PINATA_API_SECRET) in agents/.env
3. Test PDFs with abstracts in utils/files/test/

Run with: python agents/tests/test_service_generator.py
"""

import sys
from pathlib import Path


import unittest
import asyncio
import json
import logging
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from tempfile import TemporaryDirectory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from agents.consumer_agent.service_generator import ServiceGenerator
from agents.config import Config
from agents.infrastructure.ipfs_client import IPFSClient, IPFSUploadResult


def run_async(coro):
    """Helper to run async functions in sync tests."""
    return asyncio.get_event_loop().run_until_complete(coro)


class TestServiceGeneratorUnit(unittest.TestCase):
    """Unit tests with mocked LLM and IPFS."""
    
    def setUp(self):
        """Set up test fixtures with mocks."""
        self.mock_config = Mock(spec=Config)
        self.mock_config.openrouter_api_key = "test-api-key"
        self.mock_config.openrouter_base_url = "https://openrouter.ai/api/v1"
        
        # Create generator with mocked LLM
        with patch('agents.consumer_agent.service_generator.ChatOpenAI'):
            self.generator = ServiceGenerator(self.mock_config)
            self.generator.model = Mock()
            self.generator.ipfs_client = AsyncMock(spec=IPFSClient)
    
    def test_initialization(self):
        """Test ServiceGenerator initializes correctly."""
        with patch('agents.consumer_agent.service_generator.ChatOpenAI'):
            generator = ServiceGenerator(self.mock_config)
            
            self.assertEqual(generator.config, self.mock_config)
            self.assertIsNotNone(generator.ipfs_client)
            self.assertIsInstance(generator.processed_pdfs, set)
            self.assertEqual(len(generator.processed_pdfs), 0)
            
            print("\n✓ ServiceGenerator initialized with config and empty tracking")
    
    def test_read_abstract_success(self):
        """Test reading abstract from .txt file."""
        with TemporaryDirectory() as tmpdir:
            # Create test PDF and abstract
            pdf_path = Path(tmpdir) / "test_paper.pdf"
            abstract_path = Path(tmpdir) / "test_paper.txt"
            
            pdf_path.write_text("dummy pdf content")
            abstract_path.write_text("This is a test abstract about quantum computing.")
            
            # Read abstract
            abstract = self.generator._read_abstract(pdf_path)
            
            self.assertEqual(abstract, "This is a test abstract about quantum computing.")
            print("\n✓ Abstract read successfully from .txt file")
    
    def test_read_abstract_missing_file(self):
        """Test error when abstract file is missing."""
        with TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / "test_paper.pdf"
            pdf_path.write_text("dummy pdf")
            
            with self.assertRaises(FileNotFoundError) as cm:
                self.generator._read_abstract(pdf_path)
            
            self.assertIn("Abstract file not found", str(cm.exception))
            print("\n✓ FileNotFoundError raised for missing abstract")
    
    def test_read_abstract_empty_file(self):
        """Test error when abstract file is empty."""
        with TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / "test_paper.pdf"
            abstract_path = Path(tmpdir) / "test_paper.txt"
            
            pdf_path.write_text("dummy pdf")
            abstract_path.write_text("   \n  ")  # Only whitespace
            
            with self.assertRaises(ValueError) as cm:
                self.generator._read_abstract(pdf_path)
            
            self.assertIn("Abstract file is empty", str(cm.exception))
            print("\n✓ ValueError raised for empty abstract")
    
    def test_get_default_prompts(self):
        """Test default prompts fallback."""
        prompts = self.generator._get_default_prompts()
        
        self.assertIsInstance(prompts, list)
        self.assertEqual(len(prompts), 4)
        self.assertTrue(all(isinstance(p, str) for p in prompts))
        
        print(f"\n✓ Default prompts returned: {len(prompts)} prompts")
    
    def test_generate_prompts_success(self):
        """Test LLM prompt generation with valid JSON response."""
        async def _test():
            # Mock LLM response
            mock_response = Mock()
            mock_response.content = '["What is the main contribution?", "What methodology is used?", "What are the key findings?"]'
            self.generator.model.invoke = Mock(return_value=mock_response)
            
            prompts = await self.generator._generate_prompts_from_abstract(
                "This paper presents a novel approach to quantum computing.",
                "quantum_paper"
            )
            
            self.assertIsInstance(prompts, list)
            self.assertEqual(len(prompts), 3)
            self.assertEqual(prompts[0], "What is the main contribution?")
            
            print(f"\n✓ LLM generated {len(prompts)} prompts successfully")
        
        run_async(_test())
    
    def test_generate_prompts_with_markdown_wrapper(self):
        """Test LLM response with markdown code blocks."""
        async def _test():
            # Mock LLM response with markdown
            mock_response = Mock()
            mock_response.content = '```json\n["Question 1?", "Question 2?"]\n```'
            self.generator.model.invoke = Mock(return_value=mock_response)
            
            prompts = await self.generator._generate_prompts_from_abstract(
                "Test abstract",
                "test_paper"
            )
            
            self.assertIsInstance(prompts, list)
            self.assertEqual(len(prompts), 2)
            
            print("\n✓ Markdown code blocks stripped successfully")
        
        run_async(_test())
    
    def test_generate_prompts_fallback_on_invalid_json(self):
        """Test fallback to default prompts on JSON parse error."""
        async def _test():
            # Mock LLM response with invalid JSON
            mock_response = Mock()
            mock_response.content = 'This is not valid JSON'
            self.generator.model.invoke = Mock(return_value=mock_response)
            
            prompts = await self.generator._generate_prompts_from_abstract(
                "Test abstract",
                "test_paper"
            )
            
            # Should return default prompts
            self.assertEqual(prompts, self.generator._get_default_prompts())
            
            print("\n✓ Fallback to default prompts on JSON error")
        
        run_async(_test())
    
    def test_generate_prompts_fallback_on_llm_error(self):
        """Test fallback to default prompts on LLM error."""
        async def _test():
            # Mock LLM error
            self.generator.model.invoke = Mock(side_effect=Exception("API error"))
            
            prompts = await self.generator._generate_prompts_from_abstract(
                "Test abstract",
                "test_paper"
            )
            
            # Should return default prompts
            self.assertEqual(prompts, self.generator._get_default_prompts())
            
            print("\n✓ Fallback to default prompts on LLM error")
        
        run_async(_test())
    
    def test_generate_service_for_pdf_success(self):
        """Test complete service generation for single PDF."""
        async def _test():
            with TemporaryDirectory() as tmpdir:
                # Create test files
                pdf_path = Path(tmpdir) / "test.pdf"
                abstract_path = Path(tmpdir) / "test.txt"
                
                pdf_path.write_text("dummy pdf")
                abstract_path.write_text("Test abstract about AI.")
                
                # Mock LLM and IPFS
                mock_llm_response = Mock()
                mock_llm_response.content = '["Question 1?", "Question 2?"]'
                self.generator.model.invoke = Mock(return_value=mock_llm_response)
                
                self.generator.ipfs_client.pin_file = AsyncMock(
                    return_value=IPFSUploadResult(success=True, cid="QmPDF123", error=None)
                )
                self.generator.ipfs_client.pin_json = AsyncMock(
                    return_value=IPFSUploadResult(success=True, cid="QmService456", error=None)
                )
                
                result = await self.generator._generate_service_for_pdf(pdf_path)
                
                # Verify result structure
                self.assertEqual(result["service_cid"], "QmService456")
                self.assertEqual(result["pdf_cid"], "QmPDF123")
                self.assertEqual(result["pdf_name"], "test.pdf")
                self.assertEqual(len(result["prompts"]), 2)
                
                # Verify IPFS was called
                self.generator.ipfs_client.pin_file.assert_called_once()
                self.generator.ipfs_client.pin_json.assert_called_once()
                
                print("\n✓ Complete service generation successful")
                print(f"  PDF CID: {result['pdf_cid']}")
                print(f"  Service CID: {result['service_cid']}")
        
        run_async(_test())
    
    def test_generate_service_for_pdf_upload_failure(self):
        """Test error handling when PDF upload fails."""
        async def _test():
            with TemporaryDirectory() as tmpdir:
                pdf_path = Path(tmpdir) / "test.pdf"
                abstract_path = Path(tmpdir) / "test.txt"
                
                pdf_path.write_text("dummy pdf")
                abstract_path.write_text("Test abstract")
                
                # Mock LLM success but IPFS failure
                mock_llm_response = Mock()
                mock_llm_response.content = '["Question?"]'
                self.generator.model.invoke = Mock(return_value=mock_llm_response)
                
                self.generator.ipfs_client.pin_file = AsyncMock(
                    return_value=IPFSUploadResult(success=False, cid=None, error="Upload failed")
                )
                
                with self.assertRaises(Exception) as cm:
                    await self.generator._generate_service_for_pdf(pdf_path)
                
                self.assertIn("Failed to upload PDF", str(cm.exception))
                print("\n✓ PDF upload failure handled correctly")
        
        run_async(_test())
    
    def test_generate_services_from_pdfs_success(self):
        """Test batch processing multiple PDFs."""
        async def _test():
            with TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                
                # Create 3 test PDFs with abstracts
                for i in range(1, 4):
                    pdf_path = tmpdir_path / f"{i}.pdf"
                    abstract_path = tmpdir_path / f"{i}.txt"
                    
                    pdf_path.write_text(f"dummy pdf {i}")
                    abstract_path.write_text(f"Abstract for paper {i}")
                
                # Mock LLM and IPFS
                mock_llm_response = Mock()
                mock_llm_response.content = '["Q1?", "Q2?"]'
                self.generator.model.invoke = Mock(return_value=mock_llm_response)
                
                self.generator.ipfs_client.pin_file = AsyncMock(
                    return_value=IPFSUploadResult(success=True, cid="QmPDF", error=None)
                )
                self.generator.ipfs_client.pin_json = AsyncMock(
                    return_value=IPFSUploadResult(success=True, cid="QmService", error=None)
                )
                
                results = await self.generator.generate_services_from_pdfs(tmpdir_path)
                
                # Verify results
                self.assertEqual(len(results["processed"]), 3)
                self.assertEqual(len(results["skipped"]), 0)
                self.assertEqual(len(results["failed"]), 0)
                self.assertEqual(len(self.generator.processed_pdfs), 3)
                
                print(f"\n✓ Batch processing successful:")
                print(f"  Processed: {len(results['processed'])} PDFs")
                print(f"  Skipped: {len(results['skipped'])} PDFs")
                print(f"  Failed: {len(results['failed'])} PDFs")
        
        run_async(_test())
    
    def test_generate_services_skip_processed(self):
        """Test skipping already processed PDFs."""
        async def _test():
            with TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                
                # Create test files
                pdf_path = tmpdir_path / "test.pdf"
                abstract_path = tmpdir_path / "test.txt"
                
                pdf_path.write_text("dummy pdf")
                abstract_path.write_text("Test abstract")
                
                # Mark as already processed
                self.generator.processed_pdfs.add("test.pdf")
                
                results = await self.generator.generate_services_from_pdfs(tmpdir_path)
                
                # Should be skipped
                self.assertEqual(len(results["processed"]), 0)
                self.assertEqual(len(results["skipped"]), 1)
                self.assertEqual(results["skipped"][0], "test.pdf")
                
                print("\n✓ Already processed PDFs skipped correctly")
        
        run_async(_test())
    
    def test_generate_services_no_skip(self):
        """Test reprocessing when skip_processed=False."""
        async def _test():
            with TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                
                # Create test files
                pdf_path = tmpdir_path / "test.pdf"
                abstract_path = tmpdir_path / "test.txt"
                
                pdf_path.write_text("dummy pdf")
                abstract_path.write_text("Test abstract")
                
                # Mark as already processed
                self.generator.processed_pdfs.add("test.pdf")
                
                # Mock LLM and IPFS
                mock_llm_response = Mock()
                mock_llm_response.content = '["Q?"]'
                self.generator.model.invoke = Mock(return_value=mock_llm_response)
                
                self.generator.ipfs_client.pin_file = AsyncMock(
                    return_value=IPFSUploadResult(success=True, cid="QmPDF", error=None)
                )
                self.generator.ipfs_client.pin_json = AsyncMock(
                    return_value=IPFSUploadResult(success=True, cid="QmService", error=None)
                )
                
                results = await self.generator.generate_services_from_pdfs(
                    tmpdir_path,
                    skip_processed=False
                )
                
                # Should be processed again
                self.assertEqual(len(results["processed"]), 1)
                self.assertEqual(len(results["skipped"]), 0)
                
                print("\n✓ Reprocessing works with skip_processed=False")
        
        run_async(_test())
    
    def test_reset_processed(self):
        """Test clearing processed PDFs tracking."""
        self.generator.processed_pdfs.add("file1.pdf")
        self.generator.processed_pdfs.add("file2.pdf")
        
        self.assertEqual(len(self.generator.processed_pdfs), 2)
        
        self.generator.reset_processed()
        
        self.assertEqual(len(self.generator.processed_pdfs), 0)
        print("\n✓ Processed PDFs tracking cleared")
    
    def test_sorted_pdf_processing(self):
        """Test PDFs are processed in sorted order."""
        async def _test():
            with TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                
                # Create PDFs in non-sorted order
                for name in ["3.pdf", "1.pdf", "2.pdf"]:
                    pdf_path = tmpdir_path / name
                    abstract_path = tmpdir_path / name.replace(".pdf", ".txt")
                    
                    pdf_path.write_text(f"dummy {name}")
                    abstract_path.write_text(f"Abstract {name}")
                
                # Mock LLM and IPFS
                mock_llm_response = Mock()
                mock_llm_response.content = '["Q?"]'
                self.generator.model.invoke = Mock(return_value=mock_llm_response)
                
                call_order = []
                
                async def track_pdf_upload(file_path, **kwargs):
                    call_order.append(file_path.name)
                    return IPFSUploadResult(success=True, cid="QmPDF", error=None)
                
                self.generator.ipfs_client.pin_file = AsyncMock(side_effect=track_pdf_upload)
                self.generator.ipfs_client.pin_json = AsyncMock(
                    return_value=IPFSUploadResult(success=True, cid="QmService", error=None)
                )
                
                await self.generator.generate_services_from_pdfs(tmpdir_path)
                
                # Verify sorted order
                self.assertEqual(call_order, ["1.pdf", "2.pdf", "3.pdf"])
                print("\n✓ PDFs processed in sorted order: 1.pdf, 2.pdf, 3.pdf")
        
        run_async(_test())


class TestServiceGeneratorIntegration(unittest.TestCase):
    """Integration tests with real LLM and IPFS (requires API keys)."""
    
    @unittest.skipUnless(
        (Path(__file__).parent.parent.parent / "utils" / "files").exists(),
        "Test PDF directory not found in utils/files/"
    )
    def test_real_service_generation(self):
        """Test with real LLM and IPFS (slow, requires API keys)."""
        async def _test():
            from agents.config import config
            
            # Skip if API keys not configured
            if not config.openrouter_api_key or not (
                config.pinata_jwt or (config.pinata_api_key and config.pinata_api_secret)
            ):
                print("\n⏭️  Skipping integration test: API keys not configured")
                return
            
            generator = ServiceGenerator(config)
            
            test_dir = Path(__file__).parent.parent.parent / "utils" / "files"
            
            results = await generator.generate_services_from_pdfs(test_dir)
            
            self.assertGreater(len(results["processed"]), 0)
            
            for result in results["processed"]:
                self.assertIn("service_cid", result)
                self.assertIn("pdf_cid", result)
                self.assertTrue(result["service_cid"].startswith("Qm"))
                self.assertTrue(result["pdf_cid"].startswith("Qm"))
                
                print(f"\n✓ Generated service: {result['pdf_name']}")
                print(f"  Service CID: {result['service_cid']}")
                print(f"  PDF CID: {result['pdf_cid']}")
                print(f"  Prompts: {len(result['prompts'])}")
        
        run_async(_test())


if __name__ == "__main__":
    print("=" * 70)
    print("SERVICE GENERATOR TESTS")
    print("=" * 70)
    print()
    print("Unit tests: Mock LLM and IPFS for fast testing")
    print("Integration test: Real LLM and IPFS (requires API keys)")
    print("=" * 70)
    print()
    
    unittest.main(verbosity=2)

"""
Service Generator - LLM-based generator for service requirements.

Uses LLM to generate custom prompts for literature review services based on paper abstracts.
"""

import logging
from typing import Dict, Any, List, Set
from pathlib import Path
import json
from langchain_openai import ChatOpenAI

try:
    from ..config import Config
    from ..infrastructure.ipfs_client import IPFSClient
except ImportError:
    from config import Config
    from infrastructure.ipfs_client import IPFSClient

logger = logging.getLogger(__name__)


class ServiceGenerator:
    """
    LLM-based service generator for literature review requirements.
    
    Generates custom prompts for each paper based on its abstract.
    """
    
    def __init__(self, config: Config):
        """Initialize service generator with LLM and IPFS clients."""
        self.config = config
        self.ipfs_client = IPFSClient()
        self.processed_pdfs: Set[str] = set()  # Track processed PDFs in memory
        
        # Initialize OpenRouter LLM
        self.model = ChatOpenAI(
            model="openai/gpt-oss-20b",
            api_key=config.openrouter_api_key,
            base_url=config.openrouter_base_url,
            temperature=0.7
        )
        
        logger.info("ServiceGenerator initialized with OpenRouter LLM")
    
    async def generate_services_from_pdfs(
        self,
        pdf_dir: Path,
        complexity: str = "medium",
        skip_processed: bool = True
    ) -> Dict[str, Any]:
        """
        Generate service descriptions for all PDFs in directory.
        
        For each PDF:
        1. Check if already processed (skip if skip_processed=True)
        2. Read abstract from .txt file
        3. Generate custom prompts using LLM
        4. Upload PDF to IPFS (reuses upload_pdfs logic)
        5. Create service description with PDF CID (reuses create_and_upload_service)
        6. Upload service description to IPFS
        
        Args:
            pdf_dir: Directory containing PDF files and their abstracts
            complexity: Service complexity level (low/medium/high)
            skip_processed: Skip PDFs already processed in this session
            
        Returns:
            Dict with 'processed', 'skipped', and 'failed' lists
        """
        logger.info(f"Generating services from PDFs in {pdf_dir}")
        
        # Find all PDFs
        pdf_paths = sorted(pdf_dir.glob("*.pdf"))
        
        if not pdf_paths:
            logger.warning(f"No PDF files found in {pdf_dir}")
            return {"processed": [], "skipped": [], "failed": []}
        
        logger.info(f"Found {len(pdf_paths)} PDF files")
        
        processed = []
        skipped = []
        failed = []
        
        for pdf_path in pdf_paths:
            pdf_name = pdf_path.name
            
            # Check if already processed
            if skip_processed and pdf_name in self.processed_pdfs:
                logger.info(f"⏭️  Skipping {pdf_name} (already processed)")
                skipped.append(pdf_name)
                continue
            
            try:
                result = await self._generate_service_for_pdf(pdf_path, complexity)
                processed.append(result)
                self.processed_pdfs.add(pdf_name)
                logger.info(f"✓ Generated service for {pdf_name}")
            except Exception as e:
                logger.error(f"❌ Failed to generate service for {pdf_name}: {e}")
                failed.append({"pdf_name": pdf_name, "error": str(e)})
                continue
        
        logger.info(
            f"✓ Results: {len(processed)} processed, "
            f"{len(skipped)} skipped, {len(failed)} failed"
        )
        
        return {
            "processed": processed,
            "skipped": skipped,
            "failed": failed
        }
    
    async def _generate_service_for_pdf(
        self,
        pdf_path: Path,
        complexity: str
    ) -> Dict[str, Any]:
        """
        Generate service description for a single PDF.
        
        Reuses logic from create_service_with_pdfs.py:
        - upload_pdfs() for PDF upload
        - create_and_upload_service() for service creation
        """
        
        # 1. Read abstract
        abstract = self._read_abstract(pdf_path)
        
        # 2. Generate custom prompts using LLM
        prompts = await self._generate_prompts_from_abstract(abstract, pdf_path.stem)
        
        # 3. Upload PDF to IPFS (equivalent to upload_pdfs)
        logger.info(f"  Uploading {pdf_path.name}...")
        pdf_result = await self.ipfs_client.pin_file(
            file_path=pdf_path,
            name=pdf_path.name,
            metadata={"type": "input_file", "format": "pdf"}
        )
        
        if not pdf_result.success:
            raise Exception(f"Failed to upload PDF: {pdf_result.error}")
        
        pdf_cid = pdf_result.cid
        logger.info(f"  ✅ PDF: {pdf_cid}")
        
        # 4. Create service description (equivalent to create_and_upload_service)
        service_desc = {
            "title": f"Literature Review: {pdf_path.stem}",
            "description": f"Comprehensive literature review and analysis of {pdf_path.stem}",
            "service_type": "literature_review",
            "prompts": prompts,
            "input_files_cid": pdf_cid,
            "complexity": complexity,
            "expected_duration_minutes": self._estimate_duration(complexity),
            "quality_criteria": {
                "completeness": "All prompts must be answered thoroughly",
                "depth": "Answers should be detailed and well-supported with evidence from the paper",
                "citations": "Include relevant citations and specific references to the paper's content",
                "clarity": "Clear, well-structured responses that demonstrate deep understanding"
            }
        }
        
        # 5. Upload service description to IPFS
        service_result = await self.ipfs_client.pin_json(
            data=service_desc,
            name=f"service-{pdf_path.stem[:30]}",
            metadata={"type": "service_description", "complexity": complexity}
        )
        
        if not service_result.success:
            raise Exception(f"Failed to upload service: {service_result.error}")
        
        service_cid = service_result.cid
        logger.info(f"  ✅ Service: {service_cid}")
        
        return {
            "service_cid": service_cid,
            "pdf_cid": pdf_cid,
            "pdf_name": pdf_path.name,
            "title": service_desc["title"],
            "prompts": prompts,
            "complexity": complexity
        }
    
    def _read_abstract(self, pdf_path: Path) -> str:
        """Read abstract from .txt file alongside PDF."""
        abstract_path = pdf_path.with_suffix('.txt')
        
        if not abstract_path.exists():
            raise FileNotFoundError(
                f"Abstract file not found: {abstract_path}. "
                f"Each PDF must have a corresponding .txt file with the abstract."
            )
        
        with open(abstract_path, 'r', encoding='utf-8') as f:
            abstract = f.read().strip()
        
        if not abstract:
            raise ValueError(f"Abstract file is empty: {abstract_path}")
        
        logger.info(f"  Read abstract ({len(abstract)} chars)")
        return abstract
    
    async def _generate_prompts_from_abstract(
        self,
        abstract: str,
        paper_name: str
    ) -> List[str]:
        """Generate literature review prompts using LLM."""
        
        system_prompt = """You are an expert at creating insightful questions for literature reviews.

Given a research paper abstract, generate 3-5 specific, thoughtful questions that would help someone thoroughly understand and analyze the paper.

Your questions should cover:
- Main research question and contribution
- Methodology and approach
- Key findings and results
- Limitations and future directions
- Broader implications or applications

Make questions specific to the paper's content, not generic.

Output ONLY a valid JSON array of strings with no additional text or formatting.
Example: ["What is the main...", "How does the paper...", "What are the key..."]"""
        
        user_prompt = f"""Paper: {paper_name}

Abstract:
{abstract}

Generate 3-5 specific questions for this paper's literature review:"""
        
        try:
            response = self.model.invoke(
                f"{system_prompt}\n\n{user_prompt}"
            )
            
            # Parse JSON response
            response_text = response.content.strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
                response_text = response_text.strip()
            
            prompts = json.loads(response_text)
            
            if not isinstance(prompts, list) or not prompts:
                raise ValueError("LLM did not return a valid list of prompts")
            
            # Ensure all prompts are strings
            prompts = [str(p) for p in prompts]
            
            logger.info(f"  Generated {len(prompts)} prompts")
            return prompts
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.error(f"Response: {response.content}")
            # Fallback to default prompts
            return self._get_default_prompts()
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return self._get_default_prompts()
    
    
    def _estimate_duration(self, complexity: str) -> int:
        """Estimate expected duration based on complexity."""
        durations = {
            "low": 20,
            "medium": 30,
            "high": 45
        }
        return durations.get(complexity, 30)
    
    def _get_default_prompts(self) -> List[str]:
        """Return default prompts as fallback."""
        return [
            "What is the main research question or contribution of this paper?",
            "What methodology or approach does the paper use?",
            "What are the key findings and results?",
            "What are the limitations and potential areas for future work?"
        ]
    
    def reset_processed(self) -> None:
        """Clear the set of processed PDFs."""
        self.processed_pdfs.clear()
        logger.info("Cleared processed PDFs tracking")

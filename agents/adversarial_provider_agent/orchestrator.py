"""
Adversarial Orchestrator for Byzantine robustness testing.

This orchestrator coordinates adversarial provider behavior to test the
reputation system's ability to detect and penalize malicious actors.

Key differences from honest orchestrator:
- Initializes AdversarialBehaviorController with specified strategy
- Modifies service execution to inject low-quality responses
- May refuse to complete service (NON_COMPLETION strategy)
- Adjusts bidding based on strategy
"""

import asyncio
import logging
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime

from agents.adversarial_provider_agent.blockchain_handler import BlockchainHandler
from agents.provider_agent.literature_review import LiteratureReviewAgent
from agents.adversarial_provider_agent.adversarial_strategies import (
    AdversarialStrategy,
    AdversarialBehaviorController
)
from agents.infrastructure.ipfs_client import IPFSClient
from agents.config import Config

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    """Status of a service execution job."""
    WON = "won"
    FETCHING_REQUIREMENTS = "fetching_requirements"
    FETCHING_FILES = "fetching_files"
    PROCESSING = "processing"
    DELIVERING = "delivering"
    COMPLETING = "completing"
    COMPLETED = "completed"
    FAILED = "failed"
    REFUSED = "refused"  # NEW: For NON_COMPLETION strategy


@dataclass
class Job:
    """Represents a service execution job."""
    auction_id: int
    status: JobStatus
    started_at: datetime
    buyer_address: str
    service_cid: str
    service_requirements: Optional[Dict[str, Any]] = None
    pdf_directory: Optional[Path] = None
    prompts: List[str] = field(default_factory=list)
    result: Optional[Dict[str, Any]] = None
    llm_cost: float = 0.0
    retry_count: int = 0
    max_retries: int = 3
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary for logging/storage."""
        return {
            "auction_id": self.auction_id,
            "status": self.status.value,
            "started_at": self.started_at.isoformat(),
            "buyer_address": self.buyer_address,
            "service_cid": self.service_cid,
            "retry_count": self.retry_count,
            "llm_cost": self.llm_cost,
            "error": self.error
        }


class AdversarialOrchestrator:
    """
    Adversarial orchestrator for testing Byzantine robustness.
    
    Coordinates malicious provider behavior based on selected strategy:
    - LOW_QUALITY: Delivers hardcoded low-quality responses
    - SELECTIVE_DEFECTION: Bait-and-switch (high quality → low quality)
    - NON_COMPLETION: Wins but refuses to complete service
    - PRICE_MANIPULATION: Underbids + delivers low quality
    """
    
    def __init__(
        self, 
        config: Optional[Config] = None, 
        adversarial_strategy: str = "low_quality"
    ):
        """
        Initialize the adversarial orchestrator.
        
        Args:
            config: Configuration object
            adversarial_strategy: Strategy name (low_quality, selective_defection, etc.)
        """
        self.config = config or Config()
        
        # Initialize adversarial behavior controller
        try:
            strategy_enum = AdversarialStrategy(adversarial_strategy)
        except ValueError:
            logger.error(f"Invalid adversarial strategy: {adversarial_strategy}")
            raise ValueError(f"Unknown strategy: {adversarial_strategy}. "
                           f"Valid options: {[s.value for s in AdversarialStrategy]}")
        
        self.behavior_controller = AdversarialBehaviorController(
            strategy=strategy_enum,
            config={}
        )
        
        logger.warning(f"⚠️  [ADVERSARIAL] Initialized with strategy: {adversarial_strategy}")
        
        # Initialize agents (same as honest provider)
        self.blockchain_handler = BlockchainHandler(self.config.agent_id)
        
        # Literature agent only needed for SELECTIVE_DEFECTION bait phase
        # For pure LOW_QUALITY, we skip LLM entirely
        if strategy_enum == AdversarialStrategy.SELECTIVE_DEFECTION:
            self.literature_agent = LiteratureReviewAgent(str(self.config.agent_id))
        else:
            self.literature_agent = None
            
        self.ipfs_client = IPFSClient()
        
        # Job tracking
        self.active_jobs: Dict[int, Job] = {}  # auction_id -> Job
        self.completed_jobs: List[Job] = []
        
        # Working directory for downloaded files
        self.work_dir = Path(__file__).parent.parent / "data" / "jobs"
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        # Runtime state
        self.running = False
        self.check_interval = 5  # Check blockchain every 5 seconds
        
        # Reputation tracking for SELECTIVE_DEFECTION strategy
        self.current_reputation = 50  # Default starting reputation
        self.reputation_refresh_interval = 10  # Refresh every 10 seconds
        self.last_reputation_refresh = 0
        
        logger.info(f"Adversarial Orchestrator initialized for agent {self.config.agent_id}")
    
    async def initialize(self):
        """Initialize all components."""
        logger.info("Initializing adversarial orchestrator components...")
        await self.blockchain_handler.initialize()
        
        # Fetch initial reputation
        await self._refresh_reputation()
        
        logger.info("✓ Adversarial orchestrator ready")
    
    async def _refresh_reputation(self):
        """
        Refresh current reputation from blockchain.
        
        Critical for SELECTIVE_DEFECTION strategy to detect when reputation
        crosses threshold (e.g., >= 70) and trigger switch from bait to defection.
        """
        try:
            result = await self.blockchain_handler.client.call_contract_method(
                "ReputationRegistry",
                "getSummary",
                self.config.agent_id,
                [],  # empty addresses
                b'\x00' * 32,  # zero bytes32
                b'\x00' * 32   # zero bytes32
            )
            
            feedback_count = result[0]
            average_score = result[1] if feedback_count > 0 else 50
            
            old_reputation = self.current_reputation
            self.current_reputation = average_score
            
            # Update behavior controller with new reputation
            self.behavior_controller.current_reputation = average_score
            
            if old_reputation != average_score:
                logger.info(f"[ADVERSARIAL] Reputation updated: {old_reputation} → {average_score} (feedback_count={feedback_count})")
                
                # Check if SELECTIVE_DEFECTION threshold crossed
                if (self.behavior_controller.strategy == AdversarialStrategy.SELECTIVE_DEFECTION and
                    old_reputation < 70 and average_score >= 70):
                    logger.warning(f"⚠️  [ADVERSARIAL] BAIT-AND-SWITCH TRIGGERED! Reputation crossed threshold: {average_score} >= 70")
            
            self.last_reputation_refresh = asyncio.get_event_loop().time()
            
        except Exception as e:
            logger.error(f"Error refreshing reputation: {e}", exc_info=True)
    
    async def run(self):
        """
        Main orchestrator loop.
        
        Identical to honest provider loop:
        1. Check for new won auctions
        2. Process active jobs
        3. Clean up completed jobs
        """
        self.running = True
        logger.info("🚀 [ADVERSARIAL] Orchestrator started")
        
        try:
            while self.running:
                try:
                    # Refresh reputation periodically (important for SELECTIVE_DEFECTION)
                    current_time = asyncio.get_event_loop().time()
                    if current_time - self.last_reputation_refresh >= self.reputation_refresh_interval:
                        await self._refresh_reputation()
                    
                    # Check for new won auctions
                    await self._check_won_auctions()
                    
                    # Process active jobs
                    await self._process_active_jobs()
                    
                    # Clean up old completed jobs
                    self._cleanup_completed_jobs()
                    
                except Exception as e:
                    logger.error(f"Error in orchestrator loop: {e}", exc_info=True)
                
                # Wait before next check
                await asyncio.sleep(self.check_interval)
                
        except asyncio.CancelledError:
            logger.info("[ADVERSARIAL] Orchestrator cancelled")
        finally:
            self.running = False
            logger.info("[ADVERSARIAL] Orchestrator stopped")
    
    def stop(self):
        """Stop the orchestrator."""
        logger.info("Stopping adversarial orchestrator...")
        self.running = False
    
    async def _check_won_auctions(self):
        """Check blockchain for newly won auctions."""
        try:
            result = await self.blockchain_handler.monitor_auctions()
            won_auctions = result.get("won_auctions", [])
            
            for auction_details in won_auctions:
                auction_id = auction_details["auction_id"]
                if auction_id not in self.active_jobs:
                    await self._start_job(auction_details)
                    self.behavior_controller.on_auction_detected()
                    
        except Exception as e:
            logger.error(f"Error checking won auctions: {e}")
    
    async def _start_job(self, auction_details: Dict[str, Any]):
        """Start a new job for a won auction."""
        auction_id = auction_details["auction_id"]
        logger.info(f"🎉 [ADVERSARIAL] Starting job for won auction {auction_id}")
        
        try:
            job = Job(
                auction_id=auction_id,
                status=JobStatus.WON,
                started_at=datetime.now(),
                buyer_address=auction_details["buyer_address"],
                service_cid=auction_details["service_cid"]
            )
            
            self.active_jobs[auction_id] = job
            logger.info(f"✓ Job {auction_id} created: {job.to_dict()}")
            
        except Exception as e:
            logger.error(f"Error starting job {auction_id}: {e}", exc_info=True)
    
    async def _process_active_jobs(self):
        """Process all active jobs through their lifecycle."""
        for auction_id in list(self.active_jobs.keys()):
            job = self.active_jobs[auction_id]
            
            try:
                # Process job based on current status
                if job.status == JobStatus.WON:
                    await self._fetch_requirements(job)
                elif job.status == JobStatus.FETCHING_REQUIREMENTS:
                    await self._fetch_files(job)
                elif job.status == JobStatus.FETCHING_FILES:
                    await self._execute_service(job)
                elif job.status == JobStatus.PROCESSING:
                    await self._deliver_result(job)
                elif job.status == JobStatus.DELIVERING:
                    await self._complete_service(job)
                elif job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.REFUSED]:
                    # Move to completed jobs
                    self.completed_jobs.append(job)
                    del self.active_jobs[auction_id]
                    logger.info(f"✓ Job {auction_id} finished with status: {job.status.value}")
                    
            except Exception as e:
                logger.error(f"Error processing job {auction_id}: {e}", exc_info=True)
                await self._handle_job_error(job, str(e))
    
    async def _fetch_requirements(self, job: Job):
        """Fetch service requirements from IPFS."""
        logger.info(f"📥 Fetching requirements for job {job.auction_id} from {job.service_cid}")
        
        try:
            # Fetch service description from IPFS
            requirements = await self.ipfs_client.fetch_json(job.service_cid)
            
            if not requirements:
                raise Exception(f"Failed to fetch service requirements from IPFS: {job.service_cid}")
            
            job.service_requirements = requirements
            
            # Extract prompts for literature review
            job.prompts = requirements.get("prompts", requirements.get("requirements", []))
            
            logger.info(f"✓ Requirements fetched: {len(job.prompts)} prompts")
            job.status = JobStatus.FETCHING_REQUIREMENTS
            
        except Exception as e:
            raise Exception(f"Failed to fetch requirements: {e}")
    
    async def _fetch_files(self, job: Job):
        """
        Fetch input files (PDFs) for the service.
        
        For LOW_QUALITY strategy, we still fetch files to maintain appearance
        of legitimate behavior, but won't actually use them.
        """
        logger.info(f"📥 Fetching input files for job {job.auction_id}")
        
        try:
            # Create job-specific directory
            job.pdf_directory = self.work_dir / f"auction_{job.auction_id}"
            job.pdf_directory.mkdir(exist_ok=True)
            
            downloaded_count = 0
            
            # Check for single file CID
            input_cid = job.service_requirements.get("input_files_cid")
            if input_cid:
                logger.info(f"Downloading file from IPFS: {input_cid}")
                output_path = job.pdf_directory / "input.pdf"
                success = await self.ipfs_client.download_file(input_cid, output_path)
                if success:
                    downloaded_count += 1
                else:
                    logger.warning(f"Failed to download file from {input_cid}")
            
            # Check for multiple files (dict or list of CIDs)
            input_files = job.service_requirements.get("input_files")
            if input_files:
                if isinstance(input_files, dict):
                    # Format: {"file1.pdf": "Qm...", "file2.pdf": "Qm..."}
                    for filename, cid in input_files.items():
                        logger.info(f"Downloading {filename} from IPFS: {cid}")
                        output_path = job.pdf_directory / filename
                        success = await self.ipfs_client.download_file(cid, output_path)
                        if success:
                            downloaded_count += 1
                        else:
                            logger.warning(f"Failed to download {filename} from {cid}")
                            
                elif isinstance(input_files, list):
                    # Format: ["Qm...", "Qm..."]
                    for i, cid in enumerate(input_files):
                        logger.info(f"Downloading file {i+1} from IPFS: {cid}")
                        output_path = job.pdf_directory / f"input_{i+1}.pdf"
                        success = await self.ipfs_client.download_file(cid, output_path)
                        if success:
                            downloaded_count += 1
                        else:
                            logger.warning(f"Failed to download file from {cid}")
            
            if downloaded_count == 0 and not input_cid and not input_files:
                logger.warning(f"No input files specified for job {job.auction_id}")
            elif downloaded_count > 0:
                logger.info(f"✓ Downloaded {downloaded_count} files to {job.pdf_directory}")
            
            job.status = JobStatus.FETCHING_FILES
            
        except Exception as e:
            raise Exception(f"Failed to fetch files: {e}")
    
    async def _execute_service(self, job: Job):
        """
        Execute service with adversarial modifications.
        
        This is the key method where adversarial behavior is injected.
        """
        logger.info(f"⚙️  [ADVERSARIAL] Executing service for job {job.auction_id}")
        
        try:
            # Check if we should even complete the service
            if not self.behavior_controller.should_complete_service():
                logger.warning(f"[ADVERSARIAL] Refusing to complete service {job.auction_id} (NON_COMPLETION)")
                job.status = JobStatus.REFUSED
                job.error = "Adversarial non-completion"
                return
            
            # Check if we should use low-quality response
            if self.behavior_controller.should_use_low_quality_response():
                # Generate hardcoded low-quality responses (no LLM)
                logger.info(f"[ADVERSARIAL] Generating low-quality responses for {job.auction_id}")
                
                responses = []
                for i, prompt in enumerate(job.prompts):
                    low_quality_resp = self.behavior_controller.generate_low_quality_response(prompt)
                    responses.append({
                        "prompt": prompt,
                        "response": low_quality_resp,
                        "adversarial": True
                    })
                
                job.result = {
                    "success": True,
                    "responses": responses,
                    "adversarial_modified": True,
                    "strategy": self.behavior_controller.strategy.value
                }
                job.llm_cost = 0.0  # No LLM used
                
            else:
                # Use honest LLM execution (for SELECTIVE_DEFECTION bait phase)
                logger.info(f"[ADVERSARIAL] Using honest execution for {job.auction_id} (bait phase)")
                
                if self.literature_agent is None:
                    raise Exception("Literature agent not initialized for honest execution")
                
                result = self.literature_agent.perform_review(
                    pdf_directory=str(job.pdf_directory),
                    prompts=job.prompts,
                    force_rebuild=True
                )
                
                if not result["success"]:
                    raise Exception(f"Literature review failed: {result.get('error')}")
                
                job.result = result
                job.llm_cost = result.get("total_cost", 0.0)
            
            self.behavior_controller.on_service_executed()
            logger.info(f"✓ [ADVERSARIAL] Service executed: {len(job.result['responses'])} responses")
            job.status = JobStatus.PROCESSING
            
        except Exception as e:
            raise Exception(f"Failed to execute service: {e}")
    
    async def _deliver_result(self, job: Job):
        """
        Deliver results to customer.
        
        For v1: Save to local file
        For v2: Use A2A protocol to POST to customer endpoint
        """
        logger.info(f"📤 Delivering result for job {job.auction_id}")
        
        try:
            # For v1: Save result to file
            result_file = job.pdf_directory / "result.json"
            with open(result_file, "w") as f:
                json.dump(job.result, f, indent=2)
            
            logger.info(f"✓ Result saved to {result_file}")
            
            # TODO: For v2, send via A2A protocol
            
            job.status = JobStatus.DELIVERING
            
        except Exception as e:
            raise Exception(f"Failed to deliver result: {e}")
    
    async def _complete_service(self, job: Job):
        """Complete the service on blockchain."""
        logger.info(f"✅ Completing service for job {job.auction_id}")
        
        try:
            # Call blockchain handler to complete service
            completion_result = await self.blockchain_handler.complete_service(
                auction_id=job.auction_id,
                client_address=job.buyer_address
            )
            
            if completion_result.get("error"):
                raise Exception(f"Blockchain completion failed: {completion_result['error']}")
            
            logger.info(f"✓ Service completed on-chain: tx={completion_result.get('tx_hash')}")
            
            job.status = JobStatus.COMPLETED
            
        except Exception as e:
            raise Exception(f"Failed to complete service: {e}")
    
    async def _handle_job_error(self, job: Job, error: str):
        """Handle job errors with retry logic."""
        job.error = error
        job.retry_count += 1
        
        logger.warning(f"⚠️  Job {job.auction_id} error (attempt {job.retry_count}/{job.max_retries}): {error}")
        
        if job.retry_count >= job.max_retries:
            logger.error(f"❌ Job {job.auction_id} failed after {job.retry_count} attempts")
            job.status = JobStatus.FAILED
        else:
            # Retry: Keep current status to retry the same step
            logger.info(f"🔄 Retrying job {job.auction_id}...")
    
    def _cleanup_completed_jobs(self, max_completed: int = 100):
        """Clean up old completed jobs to prevent memory bloat."""
        if len(self.completed_jobs) > max_completed:
            removed = len(self.completed_jobs) - max_completed
            self.completed_jobs = self.completed_jobs[-max_completed:]
            logger.debug(f"Cleaned up {removed} old completed jobs")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current orchestrator status including adversarial stats."""
        return {
            "running": self.running,
            "adversarial": True,
            "strategy": self.behavior_controller.strategy.value,
            "behavior_stats": self.behavior_controller.get_stats(),
            "active_jobs": len(self.active_jobs),
            "completed_jobs": len(self.completed_jobs),
            "jobs": {
                "active": [job.to_dict() for job in self.active_jobs.values()],
                "completed": [job.to_dict() for job in self.completed_jobs[-10:]]
            }
        }


async def main(args):
    """Run the adversarial orchestrator with CLI arguments."""
    # Override config from CLI arguments
    if args.agent_id:
        Config.agent_id = args.agent_id
    if args.private_key:
        Config.blockchain_private_key = args.private_key
    if args.check_interval:
        Config.check_interval = args.check_interval
    
    orchestrator = AdversarialOrchestrator(
        adversarial_strategy=args.strategy
    )
    await orchestrator.initialize()
    
    # Create status file path if provided
    status_file = Path(args.status_file) if args.status_file else None
    
    try:
        # Write initial status
        if status_file:
            with open(status_file, 'w') as f:
                json.dump(orchestrator.get_status(), f)
        
        # Start orchestrator main loop
        orchestrator_task = asyncio.create_task(orchestrator.run())
        
        # Periodically write status while orchestrator is running
        while not orchestrator_task.done():
            if status_file:
                with open(status_file, 'w') as f:
                    json.dump(orchestrator.get_status(), f)
            await asyncio.sleep(10)  # Update status every 10 seconds
        
        # Await the orchestrator task to catch any exceptions
        await orchestrator_task
            
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        orchestrator.stop()
        if status_file:
            with open(status_file, 'w') as f:
                json.dump(orchestrator.get_status(), f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adversarial Provider Agent Orchestrator")
    parser.add_argument("--agent-id", type=int, required=True, help="Provider agent ID")
    parser.add_argument("--private-key", type=str, required=True, help="Blockchain private key")
    parser.add_argument("--strategy", type=str, required=True, 
                       choices=["low_quality", "selective_defection", "non_completion", "price_manipulation"],
                       help="Adversarial strategy to use")
    parser.add_argument("--check-interval", type=int, default=5, help="Blockchain check interval in seconds")
    parser.add_argument("--status-file", type=str, help="Path to write status JSON file")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    asyncio.run(main(args))

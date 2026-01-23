"""
Consumer Agent - Main orchestrator for service consumers.

Responsibilities:
- Generate service requirements
- Create auctions with eligible providers
- Monitor auction progress and end when appropriate
- Retrieve completed service results
- Evaluate service quality
- Submit feedback to reputation registry
"""

import asyncio
import logging
import argparse
import json
import sys
import random
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from agents.config import Config
from agents.infrastructure.ipfs_client import IPFSClient
from agents.consumer_agent.blockchain_handler import ConsumerBlockchainHandler
from agents.consumer_agent.service_generator import ServiceGenerator
from agents.consumer_agent.evaluator import ServiceEvaluator

# Create config instance
config = Config()

logger = logging.getLogger(__name__)


class AuctionStatus(str, Enum):
    """Status of consumer's auction tracking."""
    CREATED = "created"
    ACTIVE = "active"
    ENDED = "ended"
    COMPLETED = "completed"
    EVALUATED = "evaluated"
    FAILED = "failed"


@dataclass
class AuctionTracker:
    """Tracks a single auction lifecycle from consumer perspective."""
    auction_id: int
    status: AuctionStatus
    created_at: datetime
    service_cid: str
    max_budget: int
    duration: int
    eligible_providers: List[int]
    
    # Auction results
    winning_agent_id: Optional[int] = None
    winning_bid: Optional[int] = None
    ended_at: Optional[datetime] = None
    ended_block: Optional[int] = None  # Block number when auction ended
    
    # Service delivery
    result_path: Optional[Path] = None
    result: Optional[Dict[str, Any]] = None  # Actual result data
    completed_at: Optional[datetime] = None
    
    # Evaluation
    evaluation: Optional[Dict[str, Any]] = None
    feedback_submitted: bool = False
    
    # Error tracking
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3


class Consumer:
    """
    Consumer agent for the decentralized AI services marketplace.
    
    Manages the full lifecycle of service requests:
    1. Generate service requirements (via ReAct)
    2. Create auctions with eligible providers
    3. Monitor and end auctions
    4. Retrieve and evaluate completed services
    5. Submit feedback to reputation registry
    """
    
    def __init__(self, config: Config):
        """Initialize consumer agent."""
        self.config = config
        
        # Core components
        self.blockchain_handler = ConsumerBlockchainHandler(config)
        self.ipfs_client = IPFSClient(config)
        self.service_generator = ServiceGenerator(config)
        self.evaluator = ServiceEvaluator(config)
        
        # Auction tracking
        self.active_auctions: Dict[int, AuctionTracker] = {}
        self.completed_auctions: List[AuctionTracker] = []
        
        # Service generation cache
        self.available_services: List[Dict[str, Any]] = []  # Pre-generated services
        self.service_index: int = 0  # Track which service to use next
        
        # Runtime state
        self.running = False
        self.check_interval = config.consumer_check_interval
        self.result_base_path = Path(config.result_base_path) if hasattr(config, 'result_base_path') else Path(__file__).parent.parent / "data" / "jobs"
        
        logger.info(f"Consumer initialized")
    
    async def initialize(self, pdf_dir: Optional[Path] = None, complexity: str = "medium"):
        """Initialize blockchain connections and contracts. Pre-generate services if PDF directory provided.
        
        Args:
            pdf_dir: Optional directory containing PDFs to generate services from
            complexity: Service complexity level (low/medium/high)
        """
        logger.info("Initializing consumer components...")
        
        success = await self.blockchain_handler.initialize()
        if not success:
            raise RuntimeError("Failed to initialize consumer blockchain handler")
        
        # Pre-generate services from PDFs if directory provided
        if pdf_dir:
            await self.load_services(pdf_dir, complexity)
        
        logger.info("✅ Consumer initialized successfully")
    
    async def load_services(self, pdf_dir: Path, complexity: str = "medium"):
        """Pre-generate all services from PDF directory.
        
        Args:
            pdf_dir: Directory containing PDF files with corresponding .txt abstracts
            complexity: Service complexity level (low/medium/high)
        """
        logger.info(f"📚 Loading services from {pdf_dir}...")
        
        result = await self.service_generator.generate_services_from_pdfs(
            pdf_dir=pdf_dir,
            complexity=complexity,
            skip_processed=True
        )
        
        self.available_services = result["processed"]
        self.service_index = 0
        
        logger.info(f"✅ Loaded {len(self.available_services)} services")
        if result["failed"]:
            logger.warning(f"⚠️  {len(result['failed'])} services failed to generate")
        if result["skipped"]:
            logger.info(f"⏭️  {len(result['skipped'])} services skipped (already processed)")
    
    async def create_auction(
        self,
        service_index: Optional[int] = None,
        max_budget: int = 100_000_000,  # 100 USDC
        duration: int = 1800,  # 30 minutes
        eligible_providers: Optional[List[int]] = None
    ) -> int:
        """Create a new auction for a pre-generated service.
        
        Args:
            service_index: Index of service to use from available_services, or None to use next in order
            max_budget: Maximum budget in USDC (with 6 decimals)
            duration: Auction duration in seconds
            eligible_providers: List of provider agent IDs, or None to use from config
            
        Returns:
            Auction ID
        """
        # Ensure initialized
        if not self.blockchain_handler._initialized:
            await self.initialize()
        
        logger.info("📝 Creating new auction...")
        
        # Check if services are available
        if not self.available_services:
            raise RuntimeError(
                "No services available. Call load_services(pdf_dir) or initialize(pdf_dir) first."
            )
        
        # Select service to use
        if service_index is None:
            # Use next service in order (cycling through available services)
            idx = self.service_index % len(self.available_services)
            self.service_index += 1
        else:
            # Use specific index
            if service_index < 0 or service_index >= len(self.available_services):
                raise ValueError(
                    f"Invalid service_index {service_index}. "
                    f"Must be between 0 and {len(self.available_services) - 1}"
                )
            idx = service_index
        
        service_data = self.available_services[idx]
        service_cid = service_data["service_cid"]
        
        logger.info(f"Using service: {service_data['title']}")
        logger.info(f"Service CID: {service_cid}")
        
        # Use eligible providers from config if not specified
        if eligible_providers is None:
            # Check if random provider selection is enabled
            if hasattr(self.config, 'provider_pool') and self.config.provider_pool:
                # Randomly select from provider pool
                pool = self.config.provider_pool
                count = getattr(self.config, 'eligible_per_auction', len(pool))
                
                if count >= len(pool):
                    eligible_providers = pool
                    logger.info(f"Using all {len(pool)} providers from pool")
                else:
                    eligible_providers = random.sample(pool, count)
                    logger.info(f"Randomly selected {count} providers from pool of {len(pool)}: {eligible_providers}")
            else:
                # Use static eligible_providers from config
                eligible_providers = self.config.eligible_providers
                logger.info(f"Using static eligible providers: {eligible_providers}")
        else:
            logger.info(f"Using explicitly provided eligible providers: {eligible_providers}")
        
        # Create auction on blockchain
        logger.info(f"Creating auction on blockchain (budget: {max_budget}, duration: {duration}s)...")
        
        result = await self.blockchain_handler.create_auction(
            service_cid=service_cid,
            max_price=max_budget,
            duration=duration,
            eligible_agent_ids=eligible_providers,
            reputation_weight=self.config.reputation_weight
        )
        
        if result['error']:
            raise RuntimeError(f"Failed to create auction: {result['error']}")
        
        auction_id = result['auction_id']
        tx_hash = result['tx_hash']
        
        logger.info(f"✓ Auction created on blockchain: ID={auction_id}, tx={tx_hash}")
        
        # Track the auction
        tracker = AuctionTracker(
            auction_id=auction_id,
            status=AuctionStatus.CREATED,
            created_at=datetime.now(),
            service_cid=service_cid,
            max_budget=max_budget,
            duration=duration,
            eligible_providers=eligible_providers
        )
        
        self.active_auctions[auction_id] = tracker
        
        logger.info(f"✅ Auction {auction_id} created")
        return auction_id
    
    async def monitor_auctions(self):
        """Monitor active auctions and update their status."""
        # Ensure initialized
        if not self.blockchain_handler._initialized:
            logger.warning("Blockchain handler not initialized, skipping monitoring")
            return
        
        for auction_id, tracker in list(self.active_auctions.items()):
            try:
                # Get auction details from blockchain
                auction_data = await self.blockchain_handler.get_auction_status(auction_id)
                
                if auction_data.get('error'):
                    logger.error(f"Error getting auction status: {auction_data['error']}")
                    continue
                
                is_active = auction_data['active']
                
                if not is_active and tracker.status in [AuctionStatus.CREATED, AuctionStatus.ACTIVE]:
                    # Auction has ended
                    current_block = await self.blockchain_handler.client.get_block_number()
                    logger.info(f"🏁 Auction {auction_id} has ended")
                    tracker.status = AuctionStatus.ENDED
                    tracker.ended_at = datetime.now()
                    tracker.ended_block = current_block
                    tracker.winning_agent_id = auction_data['winning_agent_id']
                    tracker.winning_bid = auction_data['winning_bid']
                    
                elif is_active:
                    tracker.status = AuctionStatus.ACTIVE
                    
                    # Check if we should end the auction
                    elapsed = (datetime.now() - tracker.created_at).total_seconds()
                    if elapsed >= tracker.duration:
                        logger.info(f"⏰ Ending auction {auction_id} (duration expired)")
                        await self.end_auction(auction_id)
                
                # Check if service is completed
                is_completed = auction_data.get('completed', False)
                if is_completed and tracker.status == AuctionStatus.ENDED and not tracker.result_path:
                    logger.info(f"✅ Service {auction_id} completed by provider")
                    tracker.status = AuctionStatus.COMPLETED
                    tracker.completed_at = datetime.now()
                    await self._retrieve_result(tracker)
                    
            except Exception as e:
                logger.error(f"Error monitoring auction {auction_id}: {e}")
                tracker.error = str(e)
    
    async def end_auction(self, auction_id: int):
        """End an auction."""
        logger.info(f"Ending auction {auction_id}...")
        
        result = await self.blockchain_handler.end_auction(auction_id)
        
        if result['error']:
            logger.error(f"Failed to end auction: {result['error']}")
            return
        
        logger.info(f"✓ Auction ended: tx={result['tx_hash']}, winner={result['winning_agent_id']}")
    
    async def _retrieve_result(self, tracker: AuctionTracker):
        """Retrieve completed service result from provider's local storage."""
        logger.info(f"📥 Retrieving result for auction {tracker.auction_id}...")
        
        # Result is stored in provider's local directory
        # Path: {result_base_path}/auction_{id}/result.json
        result_path = self.result_base_path / f"auction_{tracker.auction_id}" / "result.json"
        
        if not result_path.exists():
            logger.warning(f"Result file not found: {result_path}")
            tracker.error = "Result file not found"
            return
        
        tracker.result_path = result_path
        logger.info(f"✓ Result retrieved from {result_path}")
        
        # Load result data
        import json
        with open(result_path, 'r') as f:
            tracker.result = json.load(f)
        
        # Evaluate the result
        await self._evaluate_result(tracker)
    
    async def _evaluate_result(self, tracker: AuctionTracker):
        """Evaluate the service result using ReAct evaluator."""
        logger.info(f"🔍 Evaluating result for auction {tracker.auction_id}...")
        
        # Get original service requirements from IPFS
        service_requirements = await self.ipfs_client.fetch_json(tracker.service_cid)
        
        # Evaluate using ReAct agent
        evaluation = await self.evaluator.evaluate(
            service_requirements=service_requirements,
            result=tracker.result
        )
        
        tracker.evaluation = evaluation
        tracker.status = AuctionStatus.EVALUATED
        
        logger.info(f"✓ Evaluation complete: rating={evaluation['rating']}/100")
        if evaluation.get('quality_scores'):
            logger.info(f"  Quality scores: {evaluation['quality_scores']}")
        
        # Submit feedback
        await self._submit_feedback(tracker)
    
    async def _submit_feedback(self, tracker: AuctionTracker):
        """Submit feedback to reputation registry."""
        logger.info(f"📤 Submitting feedback for auction {tracker.auction_id}...")
        
        try:
            # Query for FeedbackAuthProvided event (already emitted by provider)
            # Use ended_block as starting point to minimize search range
            from_block = tracker.ended_block if tracker.ended_block else None
            
            feedback_auth = await self.blockchain_handler.get_feedback_auth(
                auction_id=tracker.auction_id,
                from_block=from_block,
                lookback_blocks=1000  # Search last ~1000 blocks if no ended_block
            )
            
            if not feedback_auth:
                logger.error(f"FeedbackAuthProvided event not found for auction {tracker.auction_id}")
                tracker.error = "FeedbackAuthProvided event not found"
                return
            
            # Generate feedback text from quality scores
            quality_scores = tracker.evaluation.get('quality_scores', {})
            if quality_scores:
                feedback_text = ", ".join(
                    f"{k}: {v}" for k, v in quality_scores.items()
                )
            else:
                feedback_text = f"Rating: {tracker.evaluation['rating']}/100"
            
            # Submit feedback to reputation registry
            result = await self.blockchain_handler.submit_feedback(
                auction_id=tracker.auction_id,
                agent_id=tracker.winning_agent_id,
                rating=tracker.evaluation['rating'],
                feedback_text=feedback_text,
                feedback_auth=feedback_auth
            )
            
            if result['error']:
                logger.error(f"Failed to submit feedback: {result['error']}")
                tracker.error = result['error']
                return
            
            tracker.feedback_submitted = True
            logger.info(f"✓ Feedback submitted: tx={result['tx_hash']}")
            
            # Mark as fully completed and move to completed list
            tracker.status = AuctionStatus.COMPLETED
            self.completed_auctions.append(tracker)
            del self.active_auctions[tracker.auction_id]
            
        except Exception as e:
            logger.error(f"Error submitting feedback: {e}")
            tracker.error = str(e)
    
    async def run(self):
        """Run the consumer agent main loop."""
        logger.info("Starting consumer agent...")
        self.running = True
        
        try:
            while self.running:
                # Monitor active auctions
                await self.monitor_auctions()
                
                # Sleep before next cycle
                await asyncio.sleep(self.check_interval)
                
        except Exception as e:
            logger.error(f"Consumer agent error: {e}", exc_info=True)
        finally:
            logger.info("Consumer agent stopped")
    
    def stop(self):
        """Stop the consumer agent."""
        logger.info("Stopping consumer agent...")
        self.running = False
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of consumer agent."""
        return {
            "active_auctions": len(self.active_auctions),
            "completed_auctions": len(self.completed_auctions),
            "running": self.running
        }
    
    def write_status(self, status_file: Path):
        """Write status to JSON file for external monitoring."""
        try:
            with open(status_file, 'w') as f:
                json.dump(self.get_status(), f)
        except Exception as e:
            logger.error(f"Failed to write status file: {e}")


async def main(args):
    """Run the consumer orchestrator with CLI arguments."""
    # Override config from CLI arguments
    if args.agent_id:
        config.agent_id = args.agent_id
    if args.private_key:
        config.blockchain_private_key = args.private_key
    if args.eligible_providers:
        config.eligible_providers = args.eligible_providers
    if args.provider_pool:
        config.provider_pool = args.provider_pool
    if args.eligible_per_auction:
        config.eligible_per_auction = args.eligible_per_auction
    if args.max_budget:
        config.max_budget = args.max_budget
    if args.auction_duration:
        config.auction_duration = args.auction_duration
    if args.reputation_weight:
        config.reputation_weight = args.reputation_weight
    if args.check_interval:
        config.check_interval = args.check_interval
    if args.auto_create_auction:
        config.auto_create_auction = True
    if args.num_auctions:
        config.num_auctions = args.num_auctions
    if args.pdf_directory:
        config.pdf_directory = args.pdf_directory
    if args.service_complexity:
        config.service_complexity = args.service_complexity
    if args.auction_creation_delay:
        config.auction_creation_delay = args.auction_creation_delay
    if args.inter_auction_delay:
        config.inter_auction_delay = args.inter_auction_delay
    
    consumer = Consumer(config)
    
    # Initialize with PDF directory if auto-create is enabled
    pdf_dir = Path(config.pdf_directory) if config.pdf_directory and config.auto_create_auction else None
    await consumer.initialize(pdf_dir=pdf_dir, complexity=config.service_complexity)
    
    # Create status file path if provided
    status_file = Path(args.status_file) if args.status_file else None
    
    # Track auction creation
    auctions_created = 0
    
    try:
        # Auto-create first auction if configured
        if config.auto_create_auction:
            # Optional delay before first auction
            if config.auction_creation_delay > 0:
                logger.info(f"⏳ Waiting {config.auction_creation_delay}s before creating first auction...")
                await asyncio.sleep(config.auction_creation_delay)
            
            # Create initial auction
            logger.info("🎬 Creating initial auction...")
            try:
                auction_id = await consumer.create_auction(
                    max_budget=config.max_budget,
                    duration=config.auction_duration
                )
                auctions_created = 1
                logger.info(f"✅ Created auction {auctions_created}/{config.num_auctions}: ID={auction_id}")
            except Exception as e:
                logger.error(f"❌ Failed to create initial auction: {e}")
                raise
        
        # Write initial status
        if status_file:
            consumer.write_status(status_file)
        
        # Start monitoring loop
        consumer.running = True
        while consumer.running:
            await consumer.monitor_auctions()
            
            # Sequential creation: Create next auction when previous completes
            if config.auto_create_auction and auctions_created < config.num_auctions:
                # Check if we should create next auction
                completed_count = len([a for a in consumer.completed_auctions 
                                     if a.status == AuctionStatus.COMPLETED])
                
                logger.debug(f"Status check: completed={completed_count}, created={auctions_created}, target={config.num_auctions}")
                
                if completed_count >= auctions_created:
                    logger.info(f"✅ Auction {auctions_created} completed. Creating next auction...")
                    
                    # Optional delay between auctions
                    if config.inter_auction_delay > 0:
                        logger.info(f"⏳ Waiting {config.inter_auction_delay}s before next auction...")
                        await asyncio.sleep(config.inter_auction_delay)
                    
                    try:
                        auction_id = await consumer.create_auction(
                            max_budget=config.max_budget,
                            duration=config.auction_duration
                        )
                        auctions_created += 1
                        logger.info(f"✅ Created auction {auctions_created}/{config.num_auctions}: ID={auction_id}")
                    except Exception as e:
                        logger.error(f"❌ Failed to create auction {auctions_created + 1}: {e}")
            
            # Stop when all target auctions completed (if auto-create enabled)
            if (config.auto_create_auction and 
                len(consumer.completed_auctions) >= config.num_auctions):
                logger.info(f"🎯 All {config.num_auctions} auctions completed!")
                consumer.stop()
            
            # Update status file
            if status_file:
                consumer.write_status(status_file)
            
            await asyncio.sleep(consumer.check_interval)
            
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        consumer.stop()
        if status_file:
            consumer.write_status(status_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Consumer Agent Orchestrator")
    parser.add_argument("--agent-id", type=int, help="Consumer agent ID")
    parser.add_argument("--private-key", type=str, help="Blockchain private key")
    parser.add_argument("--eligible-providers", type=int, nargs="+", help="List of eligible provider agent IDs")
    parser.add_argument("--provider-pool", type=int, nargs="+", help="Provider pool for random selection")
    parser.add_argument("--eligible-per-auction", type=int, help="Number of providers to randomly select per auction")
    parser.add_argument("--max-budget", type=int, help="Maximum budget for auctions (in token units)")
    parser.add_argument("--auction-duration", type=int, help="Auction duration in seconds")
    parser.add_argument("--reputation-weight", type=int, help="Weight of reputation in bid scoring (0-100)")
    parser.add_argument("--check-interval", type=int, help="Blockchain check interval in seconds")
    parser.add_argument("--status-file", type=str, help="Path to write status JSON file")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    # Auto-auction arguments
    parser.add_argument("--auto-create-auction", action="store_true", help="Automatically create auction(s) on startup")
    parser.add_argument("--num-auctions", type=int, help="Number of auctions to create sequentially")
    parser.add_argument("--pdf-directory", type=str, help="Directory containing PDFs for service generation")
    parser.add_argument("--service-complexity", type=str, choices=["low", "medium", "high"], help="Service complexity level")
    parser.add_argument("--auction-creation-delay", type=int, help="Seconds to wait before creating first auction")
    parser.add_argument("--inter-auction-delay", type=int, help="Seconds to wait between sequential auctions")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    asyncio.run(main(args))

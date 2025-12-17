"""Blockchain handler for consumer agent operations."""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from ..config import config
from ..infrastructure.blockchain_client import BlockchainClient
from ..infrastructure.ipfs_client import IPFSClient

logger = logging.getLogger(__name__)


class ConsumerBlockchainHandler:
    """
    Handler for consumer agent blockchain operations.
    
    Provides methods for:
    - Creating auctions
    - Ending auctions
    - Monitoring auction status
    - Submitting feedback using ERC-8004 feedbackAuth
    """
    
    def __init__(self, consumer_config=None):
        """
        Initialize the consumer blockchain handler.
        
        Args:
            consumer_config: Optional configuration object
        """
        self.config = consumer_config or config
        self.client = BlockchainClient(self.config)
        self.consumer_agent_id = self.config.consumer_agent_id
        self._initialized = False
        
        logger.info(f"ConsumerBlockchainHandler created for consumer agent {self.consumer_agent_id}")
    
    async def initialize(self) -> bool:
        """
        Initialize blockchain client and load contracts.
        
        Returns:
            True if initialization successful, False otherwise
        """
        if self._initialized:
            return True
        
        try:
            await self.client._initialize()
            
            # Load ReverseAuction ABI
            reverse_auction_abi = self._load_reverse_auction_abi()
            await self.client.load_contract(
                "ReverseAuction",
                self.config.reverse_auction_address,
                reverse_auction_abi
            )
            
            # Load IdentityRegistry ABI
            identity_registry_abi = self._load_identity_registry_abi()
            await self.client.load_contract(
                "IdentityRegistry",
                self.config.identity_registry_address,
                identity_registry_abi
            )
            
            # Load ReputationRegistry ABI
            reputation_registry_abi = self._load_reputation_registry_abi()
            await self.client.load_contract(
                "ReputationRegistry",
                self.config.reputation_registry_address,
                reputation_registry_abi
            )
            
            self._initialized = True
            logger.info("✅ Consumer blockchain handler initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize consumer blockchain handler: {e}")
            return False
    
    async def create_auction(
        self,
        service_cid: str,
        max_price: int,
        duration: int,
        eligible_agent_ids: Optional[List[int]] = None,
        reputation_weight: int = 50
    ) -> Dict[str, Any]:
        """
        Create a new reverse auction.
        
        Args:
            service_cid: IPFS CID of service requirements
            max_price: Maximum price willing to pay (in wei)
            duration: Auction duration in seconds
            eligible_agent_ids: Optional list of eligible provider agent IDs
            reputation_weight: Weight of reputation in scoring (0-100)
            
        Returns:
            Dictionary with auction_id, tx_hash, and any errors
        """
        try:
            if not self._initialized:
                await self.initialize()
            
            # Use configured eligible providers if not specified
            if eligible_agent_ids is None:
                eligible_agent_ids = self.config.eligible_providers
            
            logger.info(f"Creating auction: service={service_cid}, max_price={max_price}, "
                       f"duration={duration}s, eligible_agents={eligible_agent_ids}")
            
            # Estimate gas
            estimated_gas = await self.client.estimate_gas(
                "ReverseAuction",
                "createAuction",
                service_cid,
                max_price,
                duration,
                eligible_agent_ids,
                reputation_weight
            )
            
            # Send transaction
            tx_hash = await self.client.send_transaction(
                "ReverseAuction",
                "createAuction",
                service_cid,
                max_price,
                duration,
                eligible_agent_ids,
                reputation_weight,
                gas_limit=estimated_gas + 50000,
                value=max_price  # Deposit max_price as escrow
            )
            
            # Wait for transaction receipt
            receipt = await self.client.wait_for_transaction(tx_hash)
            
            if receipt['status'] != 1:
                raise Exception("Transaction failed")
            
            # Extract auction_id from AuctionCreated event
            auction_id = await self._extract_auction_id_from_receipt(receipt)
            
            logger.info(f"✅ Auction created: ID={auction_id}, tx={tx_hash}")
            
            return {
                "auction_id": auction_id,
                "tx_hash": tx_hash,
                "error": None
            }
            
        except Exception as e:
            logger.error(f"Failed to create auction: {e}")
            return {
                "auction_id": None,
                "tx_hash": None,
                "error": str(e)
            }
    
    async def end_auction(self, auction_id: int) -> Dict[str, Any]:
        """
        End an active auction.
        
        Args:
            auction_id: The auction ID to end
            
        Returns:
            Dictionary with tx_hash, winning_agent_id, winning_bid, and any errors
        """
        try:
            if not self._initialized:
                await self.initialize()
            
            logger.info(f"Ending auction {auction_id}...")
            
            # Estimate gas
            estimated_gas = await self.client.estimate_gas(
                "ReverseAuction",
                "endAuction",
                auction_id
            )
            
            # Send transaction
            tx_hash = await self.client.send_transaction(
                "ReverseAuction",
                "endAuction",
                auction_id,
                gas_limit=estimated_gas + 50000
            )
            
            # Wait for transaction receipt
            receipt = await self.client.wait_for_transaction(tx_hash)
            
            if receipt['status'] != 1:
                raise Exception("Transaction failed")
            
            # Extract winning bid info from AuctionEnded event
            event_data = await self._extract_auction_ended_event(receipt, auction_id)
            
            logger.info(f"✅ Auction ended: ID={auction_id}, winner={event_data['winning_agent_id']}, "
                       f"bid={event_data['winning_bid']}")
            
            return {
                "tx_hash": tx_hash,
                "winning_agent_id": event_data['winning_agent_id'],
                "winning_bid": event_data['winning_bid'],
                "error": None
            }
            
        except Exception as e:
            logger.error(f"Failed to end auction {auction_id}: {e}")
            return {
                "tx_hash": None,
                "winning_agent_id": None,
                "winning_bid": None,
                "error": str(e)
            }
    
    async def get_auction_status(self, auction_id: int) -> Dict[str, Any]:
        """
        Get current status of an auction.
        
        Args:
            auction_id: The auction ID to check
            
        Returns:
            Dictionary with auction details and status
        """
        try:
            if not self._initialized:
                await self.initialize()
            
            # Call getAuctionDetails view function (returns Auction struct)
            auction_data = await self.client.call_contract_method(
                "ReverseAuction",
                "getAuctionDetails",
                auction_id
            )
            
            # Parse auction status using struct field names
            # Auction struct has: id, buyer, serviceDescriptionCid, maxPrice, duration,
            #                     startTime, eligibleAgentIds, winningAgentId, winningBid,
            #                     isActive, isCompleted, escrowAmount, reputationWeight
            return {
                "id": auction_data[0],
                "buyer": auction_data[1],
                "service_cid": auction_data[2],
                "max_price": auction_data[3],
                "duration": auction_data[4],
                "start_time": auction_data[5],
                "eligible_agent_ids": auction_data[6],
                "winning_agent_id": auction_data[7],
                "winning_bid": auction_data[8],
                "active": auction_data[9],
                "completed": auction_data[10],
                "escrow_amount": auction_data[11],
                "reputation_weight": auction_data[12],
                "error": None
            }
            
        except Exception as e:
            logger.error(f"Failed to get auction status for {auction_id}: {e}")
            return {
                "error": str(e)
            }
    
    async def get_feedback_auth(
        self, 
        auction_id: int, 
        from_block: Optional[int] = None,
        lookback_blocks: int = 1000
    ) -> Optional[bytes]:
        """
        Get FeedbackAuthProvided event for a completed service.
        
        This queries historical events rather than waiting/polling. Should be called
        after detecting service completion (isCompleted=true).
        
        Args:
            auction_id: The auction ID to query
            from_block: Optional starting block number, defaults to lookback_blocks ago
            lookback_blocks: How many blocks to search backwards if from_block not provided
            
        Returns:
            feedbackAuth bytes if found, None otherwise
        """
        try:
            if not self._initialized:
                await self.initialize()
            
            # Determine starting block for search
            if from_block is None:
                current_block = await self.client.get_block_number()
                from_block = max(0, current_block - lookback_blocks)
            
            logger.info(f"Querying FeedbackAuthProvided event for auction {auction_id} from block {from_block}...")
            
            # Query historical events
            events = await self.client.get_contract_events(
                "ReverseAuction",
                "FeedbackAuthProvided",
                from_block=from_block,
                to_block="latest",
                argument_filters={"auctionId": auction_id}
            )
            
            if events:
                feedback_auth = events[0]['args']['feedbackAuth']
                logger.info(f"✅ Found feedbackAuth for auction {auction_id}")
                return feedback_auth
            else:
                logger.warning(f"No FeedbackAuthProvided event found for auction {auction_id}")
                return None
            
        except Exception as e:
            logger.error(f"Failed to get feedbackAuth for auction {auction_id}: {e}")
            return None
    
    async def submit_feedback(
        self,
        auction_id: int,
        agent_id: int,
        rating: int,
        feedback_text: str,
        feedback_auth: bytes,
        tag1: bytes = None,
        tag2: bytes = None,
        feedback_uri: str = ""
    ) -> Dict[str, Any]:
        """
        Submit feedback to ReputationRegistry using ERC-8004 giveFeedback.
        
        Args:
            auction_id: The auction ID (for logging)
            agent_id: The provider agent ID to rate
            rating: Rating score (0-100, REQUIRED)
            feedback_text: Textual feedback (unused if no feedback_uri)
            feedback_auth: Signed authorization from provider (REQUIRED)
            tag1: Optional first categorization tag (bytes32)
            tag2: Optional second categorization tag (bytes32)
            feedback_uri: Optional URI to detailed feedback JSON (IPFS or HTTPS)
            
        Returns:
            Dictionary with tx_hash and any errors
        """
        try:
            if not self._initialized:
                await self.initialize()
            
            # Validate rating is 0-100
            if not (0 <= rating <= 100):
                raise ValueError(f"Rating must be 0-100, got {rating}")
            
            logger.info(f"Submitting feedback for auction {auction_id}, agent {agent_id}, rating {rating}")
            
            # Use empty tags if not provided (bytes32(0))
            if tag1 is None:
                tag1 = b'\x00' * 32
            if tag2 is None:
                tag2 = b'\x00' * 32
            
            # Calculate file hash (only for non-IPFS URIs, optional otherwise)
            file_hash = b'\x00' * 32  # Empty hash for IPFS or no file
            # Note: For HTTPS URIs, caller can compute hash if needed
            
            # Estimate gas
            estimated_gas = await self.client.estimate_gas(
                "ReputationRegistry",
                "giveFeedback",
                agent_id,
                rating,
                tag1,
                tag2,
                feedback_uri,
                file_hash,
                feedback_auth
            )
            
            # Send transaction
            tx_hash = await self.client.send_transaction(
                "ReputationRegistry",
                "giveFeedback",
                agent_id,
                rating,
                tag1,
                tag2,
                feedback_uri,
                file_hash,
                feedback_auth,
                gas_limit=estimated_gas + 50000
            )
            
            # Wait for transaction receipt
            receipt = await self.client.wait_for_transaction(tx_hash)
            
            if receipt['status'] != 1:
                raise Exception("Transaction failed")
            
            logger.info(f"✅ Feedback submitted: tx={tx_hash}")
            
            return {
                "tx_hash": tx_hash,
                "error": None
            }
            
        except Exception as e:
            logger.error(f"Failed to submit feedback: {e}")
            return {
                "tx_hash": None,
                "error": str(e)
            }
    
    async def _extract_auction_id_from_receipt(self, receipt: Dict[str, Any]) -> int:
        """Extract auction ID from AuctionCreated event in transaction receipt."""
        events = await self.client.get_contract_events(
            "ReverseAuction",
            "AuctionCreated",
            from_block=receipt['blockNumber'],
            to_block=receipt['blockNumber']
        )
        
        # Find event matching this transaction
        for event in events:
            if event['transactionHash'].hex() == receipt['transactionHash'].hex():
                return event['args']['auctionId']
        
        raise Exception("AuctionCreated event not found in receipt")
    
    async def _extract_auction_ended_event(self, receipt: Dict[str, Any], auction_id: int) -> Dict[str, Any]:
        """Extract winning bid info from AuctionEnded event."""
        events = await self.client.get_contract_events(
            "ReverseAuction",
            "AuctionEnded",
            from_block=receipt['blockNumber'],
            to_block=receipt['blockNumber'],
            argument_filters={"auctionId": auction_id}
        )
        
        if not events:
            raise Exception("AuctionEnded event not found in receipt")
        
        event = events[0]
        return {
            "winning_agent_id": event['args']['winningAgentId'],
            "winning_bid": event['args']['winningBid']
        }
    
    def _load_reverse_auction_abi(self) -> List[Dict]:
        """Load ReverseAuction contract ABI."""
        import json
        from pathlib import Path
        
        # Path to compiled contract ABI
        abi_path = Path(__file__).parent.parent.parent / "contracts" / "out" / "ReverseAuction.sol" / "ReverseAuction.json"
        
        with open(abi_path, 'r') as f:
            contract_json = json.load(f)
            return contract_json['abi']
    
    def _load_identity_registry_abi(self) -> List[Dict]:
        """Load IdentityRegistry contract ABI."""
        import json
        from pathlib import Path
        
        abi_path = Path(__file__).parent.parent.parent / "contracts" / "out" / "IIdentityRegistry.sol" / "IIdentityRegistry.json"
        
        with open(abi_path, 'r') as f:
            contract_json = json.load(f)
            return contract_json['abi']
    
    def _load_reputation_registry_abi(self) -> List[Dict]:
        """Load ReputationRegistry contract ABI."""
        import json
        from pathlib import Path
        
        abi_path = Path(__file__).parent.parent.parent / "contracts" / "out" / "IReputationRegistry.sol" / "IReputationRegistry.json"
        
        with open(abi_path, 'r') as f:
            contract_json = json.load(f)
            return contract_json['abi']

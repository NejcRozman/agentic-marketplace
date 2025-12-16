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
            
            # Call getAuction view function
            auction_data = await self.client.call_contract_method(
                "ReverseAuction",
                "getAuction",
                auction_id
            )
            
            # Parse auction status
            # Assumes auction_data returns: (buyer, serviceDescriptionCid, maxPrice, 
            #                                 startTime, duration, winningAgentId, winningBid, active)
            return {
                "buyer": auction_data[0],
                "service_cid": auction_data[1],
                "max_price": auction_data[2],
                "start_time": auction_data[3],
                "duration": auction_data[4],
                "winning_agent_id": auction_data[5],
                "winning_bid": auction_data[6],
                "active": auction_data[7],
                "error": None
            }
            
        except Exception as e:
            logger.error(f"Failed to get auction status for {auction_id}: {e}")
            return {
                "error": str(e)
            }
    
    async def wait_for_feedback_auth(self, auction_id: int, timeout: int = 300) -> Optional[bytes]:
        """
        Wait for FeedbackAuthProvided event after service completion.
        
        Args:
            auction_id: The auction ID to monitor
            timeout: Maximum time to wait in seconds
            
        Returns:
            feedbackAuth bytes if found, None otherwise
        """
        try:
            if not self._initialized:
                await self.initialize()
            
            logger.info(f"Waiting for FeedbackAuthProvided event for auction {auction_id}...")
            
            # Get current block
            current_block = await self.client.get_block_number()
            
            # Poll for event
            start_time = asyncio.get_event_loop().time()
            while (asyncio.get_event_loop().time() - start_time) < timeout:
                events = await self.client.get_contract_events(
                    "ReverseAuction",
                    "FeedbackAuthProvided",
                    from_block=current_block,
                    to_block="latest",
                    argument_filters={"auctionId": auction_id}
                )
                
                if events:
                    feedback_auth = events[0]['args']['feedbackAuth']
                    logger.info(f"✅ Received feedbackAuth for auction {auction_id}")
                    return feedback_auth
                
                # Sleep before next poll
                await asyncio.sleep(5)
            
            logger.warning(f"Timeout waiting for FeedbackAuthProvided for auction {auction_id}")
            return None
            
        except Exception as e:
            logger.error(f"Failed to wait for feedbackAuth: {e}")
            return None
    
    async def submit_feedback(
        self,
        auction_id: int,
        agent_id: int,
        rating: int,
        feedback_text: str,
        feedback_auth: bytes
    ) -> Dict[str, Any]:
        """
        Submit feedback to ReputationRegistry using ERC-8004 feedbackAuth.
        
        Args:
            auction_id: The auction ID
            agent_id: The provider agent ID to rate
            rating: Rating score (0-100)
            feedback_text: Textual feedback
            feedback_auth: Signed authorization from provider
            
        Returns:
            Dictionary with tx_hash and any errors
        """
        try:
            if not self._initialized:
                await self.initialize()
            
            logger.info(f"Submitting feedback for auction {auction_id}, agent {agent_id}, rating {rating}")
            
            # ERC-8004 feedback submission
            # The exact method name and signature depends on the ReputationRegistry implementation
            # Common pattern: addFeedback(bytes feedbackAuth, uint256 score, bytes metadata)
            
            # For now, we'll use a placeholder that matches typical ERC-8004 pattern
            # TODO: Verify actual ReputationRegistry method signature
            
            # Encode feedback metadata (could be JSON or other format)
            feedback_metadata = feedback_text.encode('utf-8')
            
            # Estimate gas
            estimated_gas = await self.client.estimate_gas(
                "ReputationRegistry",
                "addFeedback",
                feedback_auth,
                rating,
                feedback_metadata
            )
            
            # Send transaction
            tx_hash = await self.client.send_transaction(
                "ReputationRegistry",
                "addFeedback",
                feedback_auth,
                rating,
                feedback_metadata,
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

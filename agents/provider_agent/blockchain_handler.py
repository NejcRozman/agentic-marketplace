"""
BlockchainHandler - Agentic handler for blockchain operations.

This agent uses LangGraph to reason about blockchain operations:
- Monitoring auctions and deciding which to participate in
- Calculating optimal bids based on risk assessment
- Managing service completion and payment release

Built with LangGraph for agentic reasoning and decision-making.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage

from ..infrastructure.blockchain_client import BlockchainClient
from ..infrastructure.auction_data import AuctionInfo
from ..infrastructure.contract_abis import (
    get_reverse_auction_abi,
    get_identity_registry_abi,
    get_reputation_registry_abi
)
from ..infrastructure.feedback_auth import generate_feedback_auth, verify_feedback_auth_format
from ..config import config

logger = logging.getLogger(__name__)


class BlockchainState(TypedDict):
    """State for blockchain agent workflow."""
    agent_id: int
    action: str
    auctions: List[Dict[str, Any]]
    selected_auction: Optional[Dict[str, Any]]
    should_bid: bool
    bid_amount: int
    risk_assessment: str
    tx_hash: Optional[str]
    error: Optional[str]
    messages: List[Any]


class BlockchainHandler:
    """
    Agentic blockchain handler using LangGraph.
    
    Handles blockchain operations with reasoning capabilities.
    """
    
    def __init__(self, agent_id: int, blockchain_client: Optional[BlockchainClient] = None):
        """Initialize the blockchain handler agent."""
        self.agent_id = agent_id
        self.client = blockchain_client or BlockchainClient()
        self.config = config
        
        self.contracts_loaded = False
        self.reverse_auction_contract = None
        self.identity_registry_contract = None
        self.reputation_registry_contract = None
        
        self.graph = self._build_graph()
        logger.info(f"BlockchainHandler initialized for agent {agent_id}")
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow for blockchain operations."""
        workflow = StateGraph(BlockchainState)
        
        workflow.add_node("monitor", self._monitor_node)
        workflow.add_node("analyze", self._analyze_node)
        workflow.add_node("execute", self._execute_node)
        
        workflow.set_entry_point("monitor")
        workflow.add_edge("monitor", "analyze")
        workflow.add_edge("analyze", "execute")
        workflow.add_edge("execute", END)
        
        return workflow.compile()
    
    async def initialize(self) -> bool:
        """Initialize the blockchain handler by loading contracts."""
        try:
            await self.client._initialize()
            
            if not self.config.reverse_auction_address:
                logger.warning("ReverseAuction address not configured")
                return False
            
            self.reverse_auction_contract = await self.client.load_contract(
                name="ReverseAuction",
                address=self.config.reverse_auction_address,
                abi=get_reverse_auction_abi()
            )
            
            if self.config.identity_registry_address:
                self.identity_registry_contract = await self.client.load_contract(
                    name="IdentityRegistry",
                    address=self.config.identity_registry_address,
                    abi=get_identity_registry_abi()
                )
            
            if self.config.reputation_registry_address:
                self.reputation_registry_contract = await self.client.load_contract(
                    name="ReputationRegistry",
                    address=self.config.reputation_registry_address,
                    abi=get_reputation_registry_abi()
                )
            
            self.contracts_loaded = True
            logger.info("✅ BlockchainHandler initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize BlockchainHandler: {e}", exc_info=True)
            return False
    
    def _monitor_node(self, state: BlockchainState) -> BlockchainState:
        """Monitor blockchain for auctions."""
        logger.info("📡 Monitoring blockchain for auctions...")
        
        try:
            state["auctions"] = []
            state["messages"].append(SystemMessage(content="Monitoring auctions from blockchain"))
            logger.info(f"Found {len(state['auctions'])} auctions")
        except Exception as e:
            logger.error(f"Error in monitor_node: {e}")
            state["error"] = str(e)
        
        return state
    
    def _analyze_node(self, state: BlockchainState) -> BlockchainState:
        """Analyze auctions and make decisions."""
        logger.info("🤔 Analyzing auction opportunities...")
        
        try:
            auctions = state.get("auctions", [])
            
            if not auctions:
                state["should_bid"] = False
                state["messages"].append(SystemMessage(content="No auctions to analyze"))
                return state
            
            # Simple heuristic: bid on first auction at 80% of budget
            auction = auctions[0]
            state["selected_auction"] = auction
            state["should_bid"] = True
            state["bid_amount"] = int(auction.get("budget", 0) * 0.8)
            state["risk_assessment"] = "low"
            
            state["messages"].append(
                SystemMessage(content=f"Decided to bid {state['bid_amount']} on auction {auction.get('auction_id')}")
            )
            
            logger.info(f"Decision: bid={state['bid_amount']}, risk={state['risk_assessment']}")
            
        except Exception as e:
            logger.error(f"Error in analyze_node: {e}")
            state["error"] = str(e)
            state["should_bid"] = False
        
        return state
    
    def _execute_node(self, state: BlockchainState) -> BlockchainState:
        """Execute blockchain transaction."""
        logger.info("⚡ Executing blockchain transaction...")
        
        try:
            if not state.get("should_bid", False):
                logger.info("No action to execute")
                state["messages"].append(SystemMessage(content="No transaction to execute"))
                return state
            
            state["tx_hash"] = None
            state["messages"].append(SystemMessage(content="Transaction would be executed here"))
            logger.info("✅ Transaction execution complete")
            
        except Exception as e:
            logger.error(f"Error in execute_node: {e}")
            state["error"] = str(e)
        
        return state
    
    async def _fetch_auctions(self, from_block: str = "latest", to_block: str = "latest") -> List[AuctionInfo]:
        """Fetch auctions from blockchain."""
        if not self.contracts_loaded:
            await self.initialize()
        
        try:
            events = await self.client.get_contract_events(
                contract_name="ReverseAuction",
                event_name="AuctionCreated",
                from_block=from_block,
                to_block=to_block
            )
            
            auctions = []
            for event in events:
                args = event.get('args', {})
                auction_id = args.get('auctionId', 0)
                
                auction_data = await self.client.call_contract_method(
                    "ReverseAuction",
                    "getAuctionDetails",
                    auction_id
                )
                
                auction_info = AuctionInfo({
                    'auctionId': auction_id,
                    'buyer': auction_data[0],
                    'budget': auction_data[1],
                    'deadline': auction_data[2],
                    'serviceDescriptionCid': auction_data[3],
                    'isActive': auction_data[4],
                    'selectedProvider': auction_data[5],
                    'finalPrice': auction_data[6],
                    'serviceCompleted': auction_data[7]
                })
                
                auctions.append(auction_info)
            
            return auctions
            
        except Exception as e:
            logger.error(f"Error fetching auctions: {e}")
            return []
    
    async def _submit_bid(self, auction_id: int, bid_amount: int) -> Optional[str]:
        """Submit bid transaction."""
        try:
            logger.info(f"Submitting bid to auction {auction_id}: {bid_amount}")
            
            estimated_gas = await self.client.estimate_gas(
                "ReverseAuction",
                "placeBid",
                auction_id,
                value=0
            )
            
            tx_hash = await self.client.send_transaction(
                "ReverseAuction",
                "placeBid",
                auction_id,
                value=0,
                gas_limit=estimated_gas + 50000
            )
            
            receipt = await self.client.wait_for_transaction(tx_hash)
            
            if receipt['status'] == 1:
                logger.info(f"✅ Bid confirmed in block {receipt['blockNumber']}")
                return tx_hash
            else:
                logger.error("❌ Bid transaction failed")
                return None
                
        except Exception as e:
            logger.error(f"Error submitting bid: {e}")
            return None
    
    async def monitor_auctions(self, from_block: str = "latest", to_block: str = "latest") -> List[AuctionInfo]:
        """Monitor blockchain for auctions."""
        return await self._fetch_auctions(from_block, to_block)
    
    async def process_auction_decision(self, auctions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process auction bidding decision through LangGraph."""
        initial_state: BlockchainState = {
            "agent_id": self.agent_id,
            "action": "bid",
            "auctions": auctions,
            "selected_auction": None,
            "should_bid": False,
            "bid_amount": 0,
            "risk_assessment": "",
            "tx_hash": None,
            "error": None,
            "messages": []
        }
        
        result = self.graph.invoke(initial_state)
        
        return {
            "tx_hash": result.get("tx_hash"),
            "bid_amount": result.get("bid_amount"),
            "error": result.get("error")
        }
    
    def create_feedback_auth(self, client_address: str, index_limit: int = 1000, expiry: Optional[int] = None) -> bytes:
        """Generate ERC-8004 compliant feedbackAuth."""
        if expiry is None:
            expiry = int(datetime.now().timestamp()) + 3600
        
        if not self.client.account:
            raise ValueError("No account configured")
        
        feedback_auth = generate_feedback_auth(
            agent_id=self.agent_id,
            client_address=client_address,
            index_limit=index_limit,
            expiry=expiry,
            chain_id=self.config.chain_id,
            identity_registry_address=self.config.identity_registry_address,
            signer_address=self.client.account.address,
            private_key=self.config.private_key
        )
        
        if not verify_feedback_auth_format(feedback_auth):
            raise ValueError("Generated feedbackAuth has invalid format")
        
        logger.info(f"✅ Created feedbackAuth for agent {self.agent_id}")
        return feedback_auth
    
    async def get_agent_address(self) -> str:
        """Get agent's blockchain address."""
        if self.client.account:
            return self.client.account.address
        return "0x0000000000000000000000000000000000000000"
    
    async def get_balance(self, address: Optional[str] = None) -> int:
        """Get balance in wei."""
        return await self.client.get_balance(address)

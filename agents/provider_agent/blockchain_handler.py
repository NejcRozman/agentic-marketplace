"""
BlockchainHandler - Agentic handler for blockchain operations.

This agent uses LangGraph to handle blockchain operations:
- Complete service path: Deterministic workflow for completing auctions
- Monitor path: ReAct agent with tools for intelligent bidding decisions

Built with LangGraph for agentic reasoning and decision-making.
"""

import asyncio
import logging
import json
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_google_genai import ChatGoogleGenerativeAI

from ..infrastructure.blockchain_client import BlockchainClient
from ..infrastructure.contract_abis import (
    get_reverse_auction_abi,
    get_identity_registry_abi,
    get_reputation_registry_abi
)
from ..infrastructure.feedback_auth import generate_feedback_auth, verify_feedback_auth_format
from ..infrastructure.ipfs_client import IPFSClient
from ..config import config

logger = logging.getLogger(__name__)


class BlockchainState(TypedDict):
    """State for blockchain agent workflow."""
    agent_id: int
    action: str  # "complete_service" or "monitor"
    
    # Service completion fields
    auction_id: Optional[int]
    client_address: Optional[str]
    feedback_auth: Optional[bytes]
    
    # Monitor path - gathered state (deterministic)
    won_auctions: List[int]  # Auction IDs we won (from AuctionEnded events)
    eligible_active_auctions: List[Dict[str, Any]]  # Auctions we can bid on
    
    # Monitor path - ReAct results
    bids_placed: List[Dict[str, Any]]  # Bids submitted this invocation
    
    # Transaction results
    tx_hash: Optional[str]
    error: Optional[str]
    messages: List[Any]


class BlockchainHandler:
    """
    Agentic blockchain handler using LangGraph.
    
    - complete_service: Deterministic path for service completion
    - monitor: ReAct agent for intelligent bidding decisions
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
        
        # Block tracking - persists across invocations
        self._last_processed_block: int = 0
        
        # Bids placed tracker (for ReAct tool results)
        self._bids_placed: List[Dict[str, Any]] = []
        
        # Build tools for ReAct agent
        self._tools = self._build_tools()
        
        # Build the graph
        self.graph = self._build_graph()
        logger.info(f"BlockchainHandler initialized for agent {agent_id}")
    
    def _build_tools(self) -> List:
        """Build tools for the ReAct agent."""
        handler = self
        
        @tool
        def get_ipfs_data(cid: str) -> Dict[str, Any]:
            """Fetch service requirements from IPFS using a CID.
            
            Args:
                cid: The IPFS content identifier (from auction's service_description_cid)
                
            Returns:
                Dictionary containing service requirements from IPFS
            """
            try:
                if not cid:
                    return {"error": "No IPFS CID provided"}
                
                ipfs_client = IPFSClient()
                result = asyncio.run(ipfs_client.fetch_json(cid))
                
                if result is None:
                    return {"error": "Failed to fetch from IPFS"}
                
                return result
            except Exception as e:
                return {"error": str(e)}
        
        @tool
        def get_reputation(agent_id: int) -> Dict[str, Any]:
            """Get reputation score for an agent from ERC-8004 ReputationRegistry.
            
            Args:
                agent_id: The agent ID to get reputation for
                
            Returns:
                Dictionary with rating and feedback count
            """
            try:
                if not handler.reputation_registry_contract:
                    return {"rating": 50, "feedback_count": 0, "note": "ReputationRegistry not configured"}
                
                result = asyncio.run(handler.client.call_contract_method(
                    "ReputationRegistry",
                    "getSummary",
                    agent_id,
                    [],
                    0,
                    0
                ))
                feedback_count = result[0]
                average_score = result[1] if feedback_count > 0 else 50
                
                return {"rating": average_score, "feedback_count": feedback_count}
            except Exception as e:
                return {"error": str(e), "rating": 50, "feedback_count": 0}
        
        @tool
        def estimate_cost(service_requirements: str) -> Dict[str, Any]:
            """Estimate cost to deliver a service based on requirements.
            
            Args:
                service_requirements: JSON string of service requirements from IPFS
                
            Returns:
                Dictionary with estimated cost and confidence
            """
            try:
                requirements = json.loads(service_requirements) if isinstance(service_requirements, str) else service_requirements
                
                complexity = requirements.get("complexity", "medium")
                base_cost = 100
                
                multipliers = {"low": 0.7, "medium": 1.0, "high": 1.5}
                multiplier = multipliers.get(complexity, 1.0)
                
                estimated = int(base_cost * multiplier * 1e6)
                
                return {"estimated_cost": estimated, "confidence": 0.7, "complexity": complexity}
            except Exception as e:
                return {"error": str(e), "estimated_cost": 0, "confidence": 0}
        
        @tool
        def place_bid(auction_id: int, bid_amount: int) -> Dict[str, Any]:
            """Submit a bid for an auction on the blockchain.
            
            Args:
                auction_id: The auction ID to bid on
                bid_amount: The bid amount in USDC (with decimals)
                
            Returns:
                Dictionary with transaction result
            """
            try:
                logger.info(f"📤 Placing bid: auction={auction_id}, amount={bid_amount}")
                
                estimated_gas = asyncio.run(handler.client.estimate_gas(
                    "ReverseAuction",
                    "placeBid",
                    auction_id,
                    bid_amount,
                    handler.agent_id
                ))
                
                tx_hash = asyncio.run(handler.client.send_transaction(
                    "ReverseAuction",
                    "placeBid",
                    auction_id,
                    bid_amount,
                    handler.agent_id,
                    gas_limit=estimated_gas + 50000
                ))
                
                receipt = asyncio.run(handler.client.wait_for_transaction(tx_hash))
                
                if receipt['status'] == 1:
                    logger.info(f"✅ Bid placed successfully: {tx_hash}")
                    result = {
                        "success": True,
                        "tx_hash": tx_hash,
                        "auction_id": auction_id,
                        "bid_amount": bid_amount,
                        "block_number": receipt['blockNumber']
                    }
                    handler._bids_placed.append(result)
                    return result
                else:
                    return {"success": False, "error": "Transaction failed"}
                    
            except Exception as e:
                logger.error(f"Error placing bid: {e}")
                return {"success": False, "error": str(e)}
        
        return [get_ipfs_data, get_reputation, estimate_cost, place_bid]
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow for blockchain operations."""
        workflow = StateGraph(BlockchainState)
        
        workflow.add_node("router", self._router_node)
        workflow.add_node("generate_feedback_auth", self._generate_feedback_auth_node)
        workflow.add_node("call_complete_service", self._call_complete_service_node)
        workflow.add_node("gather_state", self._gather_state_node)
        workflow.add_node("react_reasoning", self._react_reasoning_node)
        
        workflow.set_entry_point("router")
        
        workflow.add_conditional_edges(
            "router",
            self._route_action,
            {"complete_service": "generate_feedback_auth", "monitor": "gather_state"}
        )
        
        workflow.add_edge("generate_feedback_auth", "call_complete_service")
        workflow.add_edge("call_complete_service", END)
        workflow.add_edge("gather_state", "react_reasoning")
        workflow.add_edge("react_reasoning", END)
        
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
    
    def _router_node(self, state: BlockchainState) -> BlockchainState:
        """Router node - entry point that examines action field."""
        logger.info(f"🔀 Router: action={state['action']}")
        return state
    
    def _route_action(self, state: BlockchainState) -> str:
        """Routing function for action field."""
        if state.get("action") == "complete_service":
            return "complete_service"
        return "monitor"
    
    async def _generate_feedback_auth_node(self, state: BlockchainState) -> BlockchainState:
        """Generate ERC-8004 feedbackAuth for service completion."""
        logger.info("🔐 Generating feedbackAuth...")
        
        try:
            client_address = state.get("client_address")
            if not client_address:
                raise ValueError("client_address is required")
            
            expiry = int(datetime.now().timestamp()) + 3600
            feedback_auth = generate_feedback_auth(
                agent_id=self.agent_id,
                client_address=client_address,
                index_limit=1000,
                expiry=expiry,
                chain_id=self.config.chain_id,
                identity_registry_address=self.config.identity_registry_address,
                signer_address=self.client.account.address,
                private_key=self.config.private_key
            )
            
            if not verify_feedback_auth_format(feedback_auth):
                raise ValueError("Generated feedbackAuth has invalid format")
            
            state["feedback_auth"] = feedback_auth
            logger.info("✅ FeedbackAuth generated")
            
        except Exception as e:
            logger.error(f"Error generating feedbackAuth: {e}")
            state["error"] = str(e)
        
        return state
    
    async def _call_complete_service_node(self, state: BlockchainState) -> BlockchainState:
        """Call completeService on ReverseAuction contract."""
        logger.info("📝 Calling completeService...")
        
        try:
            auction_id = state.get("auction_id")
            feedback_auth = state.get("feedback_auth")
            
            if auction_id is None:
                raise ValueError("auction_id is required")
            if not feedback_auth:
                raise ValueError("feedback_auth is required")
            
            estimated_gas = await self.client.estimate_gas(
                "ReverseAuction",
                "completeService",
                auction_id,
                feedback_auth
            )
            
            tx_hash = await self.client.send_transaction(
                "ReverseAuction",
                "completeService",
                auction_id,
                feedback_auth,
                gas_limit=estimated_gas + 50000
            )
            
            receipt = await self.client.wait_for_transaction(tx_hash)
            
            if receipt['status'] == 1:
                state["tx_hash"] = tx_hash
                logger.info(f"✅ Service completed: {tx_hash}")
            else:
                raise Exception("Transaction failed")
            
        except Exception as e:
            logger.error(f"Error calling completeService: {e}")
            state["error"] = str(e)
        
        return state
    
    async def _gather_state_node(self, state: BlockchainState) -> BlockchainState:
        """Gather blockchain state: won auctions and active eligible auctions."""
        logger.info("📡 Gathering blockchain state...")
        
        try:
            current_block = await self.client.get_block_number()
            from_block = self._last_processed_block + 1 if self._last_processed_block > 0 else max(0, current_block - 100)
            
            # 1. Fetch AuctionEnded events to find won auctions
            won_auctions = []
            try:
                ended_events = await self.client.get_contract_events(
                    contract_name="ReverseAuction",
                    event_name="AuctionEnded",
                    from_block=from_block,
                    to_block=current_block
                )
                
                for event in ended_events:
                    args = event.get("args", {})
                    if args.get("winningAgentId") == self.agent_id:
                        auction_id = args.get("auctionId")
                        won_auctions.append(auction_id)
                        logger.info(f"🎉 Won auction {auction_id}!")
            except Exception as e:
                logger.warning(f"Error fetching AuctionEnded events: {e}")
            
            state["won_auctions"] = won_auctions
            
            # 2. Fetch all active auctions we're eligible for
            eligible_active_auctions = []
            try:
                auction_count = await self.client.call_contract_method(
                    "ReverseAuction",
                    "auctionIdCounter"
                )
                
                logger.info(f"Checking {auction_count} auctions for eligibility...")
                
                for auction_id in range(1, auction_count + 1):
                    try:
                        is_eligible = await self.client.call_contract_method(
                            "ReverseAuction",
                            "isEligibleAgent",
                            auction_id,
                            self.agent_id
                        )
                        
                        if not is_eligible:
                            continue
                        
                        auction = await self.client.call_contract_method(
                            "ReverseAuction",
                            "getAuctionDetails",
                            auction_id
                        )
                        
                        # Auction struct indices (based on contract):
                        # 0: id, 1: buyer, 2: serviceDescriptionCid, 3: maxPrice
                        # 4: duration, 5: startTime, 6: eligibleAgentIds (array)
                        # 7: winningAgentId, 8: winningBid, 9: isActive
                        # 10: isCompleted, 11: escrowAmount, 12: reputationWeight
                        
                        is_active = auction[9]
                        start_time = auction[5]
                        duration = auction[4]
                        
                        # Use blockchain timestamp, not real-world time
                        # This is important for Anvil forks where time is frozen
                        block = await self.client.get_block('latest')
                        current_time = block['timestamp']
                        end_time = start_time + duration
                        time_remaining = max(0, end_time - current_time)
                        
                        if not is_active or time_remaining == 0:
                            continue
                        
                        auction_info = {
                            "auction_id": auction_id,
                            "buyer": auction[1],
                            "service_description_cid": auction[2],
                            "max_price": auction[3],
                            "duration": duration,
                            "start_time": start_time,
                            "end_time": end_time,
                            "time_remaining": time_remaining,
                            "winning_agent_id": auction[7],
                            "winning_bid": auction[8],
                            "is_active": is_active,
                            "reputation_weight": auction[12]
                        }
                        
                        eligible_active_auctions.append(auction_info)
                        logger.info(f"📋 Eligible auction {auction_id}: max_price={auction[3]}, time_remaining={time_remaining}s")
                        
                    except Exception as e:
                        logger.debug(f"Error checking auction {auction_id}: {e}")
                        continue
                        
            except Exception as e:
                logger.error(f"Error fetching active auctions: {e}")
            
            state["eligible_active_auctions"] = eligible_active_auctions
            self._last_processed_block = current_block
            
            logger.info(f"✅ State gathered: {len(won_auctions)} won, {len(eligible_active_auctions)} active eligible")
            
        except Exception as e:
            logger.error(f"Error gathering state: {e}")
            state["error"] = str(e)
            state["won_auctions"] = []
            state["eligible_active_auctions"] = []
        
        return state
    
    async def _react_reasoning_node(self, state: BlockchainState) -> BlockchainState:
        """Use ReAct agent to reason about bidding decisions."""
        logger.info("🤔 ReAct reasoning about auctions...")
        
        eligible_auctions = state.get("eligible_active_auctions", [])
        
        if not eligible_auctions:
            logger.info("No eligible auctions to consider")
            state["bids_placed"] = []
            return state
        
        # Reset bids tracker
        self._bids_placed = []
        
        try:
            auctions_context = json.dumps(eligible_auctions, indent=2)
            
            system_prompt = f"""You are a bidding agent (ID: {self.agent_id}) for a decentralized AI service marketplace.

Available tools:
- get_ipfs_data(cid): Fetch service requirements from IPFS using the service_description_cid
- get_reputation(agent_id): Get reputation score for an agent  
- estimate_cost(service_requirements): Estimate cost to deliver a service
- place_bid(auction_id, bid_amount): Submit a bid for an auction

BIDDING GUIDELINES:
1. Analyze each auction before bidding
2. Only bid if profitable (bid_amount > estimated_cost)
3. Consider time remaining - urgent auctions may need immediate bids
4. Check current winning bid - you need a better score to win
5. Your reputation affects your score
6. You can bid on multiple auctions if profitable
7. If no auctions are profitable, don't bid

Current eligible auctions:
{auctions_context}

Analyze these auctions and decide which to bid on."""

            rate_limiter = InMemoryRateLimiter(
                requests_per_second=0.2, 
                check_every_n_seconds=0.1,
                max_bucket_size=1
            )
            
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash-lite",
                google_api_key=self.config.google_api_key,
                temperature=0.3,
                rate_limiter=rate_limiter
            )
            
            react_agent = create_react_agent(llm, self._tools)
            
            await react_agent.ainvoke({"messages": [HumanMessage(content=system_prompt)]})
            
            state["bids_placed"] = self._bids_placed
            logger.info(f"✅ ReAct completed: {len(self._bids_placed)} bids placed")
            
        except Exception as e:
            logger.error(f"Error in ReAct reasoning: {e}")
            state["error"] = str(e)
            state["bids_placed"] = []
        
        return state
    
    async def monitor_auctions(self) -> Dict[str, Any]:
        """
        Monitor blockchain for auctions and make bidding decisions.
        
        Returns:
            Dictionary with won_auctions, bids_placed, and any errors
        """
        initial_state: BlockchainState = {
            "agent_id": self.agent_id,
            "action": "monitor",
            "auction_id": None,
            "client_address": None,
            "feedback_auth": None,
            "won_auctions": [],
            "eligible_active_auctions": [],
            "bids_placed": [],
            "tx_hash": None,
            "error": None,
            "messages": []
        }
        
        result = await self.graph.ainvoke(initial_state)
        
        return {
            "won_auctions": result.get("won_auctions", []),
            "eligible_active_auctions_count": len(result.get("eligible_active_auctions", [])),
            "bids_placed": result.get("bids_placed", []),
            "error": result.get("error")
        }
    
    async def complete_service(self, auction_id: int, client_address: str) -> Dict[str, Any]:
        """
        Complete service for an auction.
        
        Args:
            auction_id: The auction ID to complete
            client_address: The buyer's address
            
        Returns:
            Dictionary with tx_hash, feedback_auth, and any errors
        """
        initial_state: BlockchainState = {
            "agent_id": self.agent_id,
            "action": "complete_service",
            "auction_id": auction_id,
            "client_address": client_address,
            "feedback_auth": None,
            "won_auctions": [],
            "eligible_active_auctions": [],
            "bids_placed": [],
            "tx_hash": None,
            "error": None,
            "messages": []
        }
        
        result = await self.graph.ainvoke(initial_state)
        
        return {
            "tx_hash": result.get("tx_hash"),
            "feedback_auth": result.get("feedback_auth"),
            "error": result.get("error")
        }
    
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
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_openai import ChatOpenAI

try:
    from ..infrastructure.blockchain_client import BlockchainClient
    from ..infrastructure.contract_abis import (
        get_reverse_auction_abi,
        get_identity_registry_abi,
        get_reputation_registry_abi
    )
    from ..infrastructure.feedback_auth import generate_feedback_auth, verify_feedback_auth_format
    from ..infrastructure.ipfs_client import IPFSClient
    from ..config import config
except ImportError:
    from infrastructure.blockchain_client import BlockchainClient
    from infrastructure.contract_abis import (
        get_reverse_auction_abi,
        get_identity_registry_abi,
        get_reputation_registry_abi
    )
    from infrastructure.feedback_auth import generate_feedback_auth, verify_feedback_auth_format
    from infrastructure.ipfs_client import IPFSClient
    from config import Config
    config = Config()

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
    won_auctions: List[Dict[str, Any]]  # Auctions we won with full details
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
                
                empty_addresses = []
                zero_bytes32 = b'\x00' * 32
                
                result = asyncio.run(handler.client.call_contract_method(
                    "ReputationRegistry",
                    "getSummary",
                    agent_id,
                    empty_addresses,
                    zero_bytes32,
                    zero_bytes32
                ))
                feedback_count = result[0]
                average_score = result[1] if feedback_count > 0 else 50
                
                return {"rating": average_score, "feedback_count": feedback_count}
            except Exception as e:
                return {"error": str(e), "rating": 50, "feedback_count": 0}
        
        @tool
        def estimate_cost(complexity: str = "medium") -> Dict[str, Any]:
            """Estimate cost to deliver a service based on complexity.
            
            Args:
                complexity: Service complexity level - 'low', 'medium', or 'high'
                
            Returns:
                Dictionary with estimated cost and confidence
            """
            try:
                # Normalize complexity string
                complexity = str(complexity).lower().strip()
                if complexity not in ["low", "medium", "high"]:
                    logger.warning(f"Invalid complexity '{complexity}', defaulting to 'medium'")
                    complexity = "medium"
                
                base_cost = 50  # Base cost in USDC
                
                multipliers = {"low": 0.7, "medium": 1.0, "high": 1.5}
                multiplier = multipliers.get(complexity, 1.0)
                
                estimated = int(base_cost * multiplier * 1e6)
                
                return {"estimated_cost": estimated, "confidence": 0.7, "complexity": complexity}
            except Exception as e:
                return {"error": str(e), "estimated_cost": 0, "confidence": 0}
        
        @tool
        def validate_bid_profitability(estimated_cost: int, proposed_bid: int) -> Dict[str, Any]:
            """Check if a proposed bid is profitable compared to estimated cost.
            
            CRITICAL: Always use this tool before placing a bid to ensure profitability.
            
            IMPORTANT: Loss-leading strategy for reputation building:
            - If you have LOW reputation (< 3 feedback), accepting UNPROFITABLE bids 
              can be a strategic investment to build your reputation
            - Once you build reputation, you gain competitive advantage in future auctions
            - Consider: Would you rather make 0 profit with 0 reputation, or lose some money 
              now to build reputation and win more profitable jobs later?
            
            Args:
                estimated_cost: Your estimated cost to deliver the service (in USDC with 6 decimals, e.g., 50000000 = 50 USDC)
                proposed_bid: The bid amount you're considering (in USDC with 6 decimals, e.g., 55000000 = 55 USDC)
            
            Returns:
                Profitability analysis with verdict and strategic recommendation considering reputation
            """
            try:
                # Get current reputation - call the function directly, not as a tool
                rep_result = get_reputation.func(handler.agent_id)
                feedback_count = rep_result.get("feedback_count", 0)
                current_rating = rep_result.get("rating", 50)
                
                profit = proposed_bid - estimated_cost
                is_profitable = proposed_bid > estimated_cost
                
                if estimated_cost > 0:
                    margin_percent = round((profit / estimated_cost) * 100, 2)
                    loss_percent = -margin_percent if not is_profitable else 0
                else:
                    margin_percent = 0
                    loss_percent = 0
                
                # Calculate suggested minimum bid
                suggested_min_bid = estimated_cost + 1000000  # Add 1 USDC profit
                
                # Strategic analysis based on reputation
                is_newcomer = feedback_count < 3
                has_low_reputation = current_rating < 70 and feedback_count > 0
                
                # Create clear verdict with strategic context
                if is_profitable:
                    if margin_percent < 5:
                        verdict = "⚠️ MARGINALLY PROFITABLE - Low profit margin"
                    else:
                        verdict = "✅ PROFITABLE - Good margin"
                    strategic_note = "Direct profit. No reputation investment needed."
                else:
                    # Unprofitable - provide strategic analysis
                    if is_newcomer:
                        if loss_percent <= 30:
                            verdict = "💡 STRATEGIC LOSS-LEADING - Acceptable for reputation building"
                            strategic_note = f"You have {feedback_count} reviews. Investing {abs(profit/1e6):.1f} USDC (loss) to build reputation could win future profitable auctions. CONSIDER ACCEPTING."
                        elif loss_percent <= 50:
                            verdict = "⚠️ RISKY LOSS-LEADING - High loss for reputation building"
                            strategic_note = f"Loss of {abs(profit/1e6):.1f} USDC is significant. Only accept if you believe reputation will unlock much higher-paying jobs."
                        else:
                            verdict = "❌ UNPROFITABLE - Loss too large even for reputation building"
                            strategic_note = f"Loss of {abs(profit/1e6):.1f} USDC ({loss_percent:.1f}%) is excessive. Not worth it even for reputation."
                    elif has_low_reputation:
                        if loss_percent <= 20:
                            verdict = "💡 REPUTATION RECOVERY - Small loss may help improve low rating"
                            strategic_note = f"Your rating is {current_rating}/100. Small loss of {abs(profit/1e6):.1f} USDC could help recover reputation."
                        else:
                            verdict = "❌ UNPROFITABLE - Loss too large for recovery strategy"
                            strategic_note = f"Loss of {abs(profit/1e6):.1f} USDC is too much for reputation recovery."
                    else:
                        verdict = "❌ UNPROFITABLE - You have good reputation, don't accept losses"
                        strategic_note = f"With {feedback_count} reviews at {current_rating}/100 rating, you should not accept unprofitable jobs."
                
                return {
                    "is_profitable": is_profitable,
                    "estimated_cost": estimated_cost,
                    "proposed_bid": proposed_bid,
                    "profit": profit,
                    "profit_margin_percent": margin_percent if is_profitable else -loss_percent,
                    "verdict": verdict,
                    "strategic_note": strategic_note,
                    "reputation_context": {
                        "feedback_count": feedback_count,
                        "current_rating": current_rating,
                        "is_newcomer": is_newcomer,
                        "has_low_reputation": has_low_reputation
                    },
                    "recommendation": f"Bid at least {suggested_min_bid/1e6:.1f} USDC to ensure profit" if is_profitable else f"Current bid results in {abs(profit/1e6):.1f} USDC loss. {strategic_note}",
                    "math_check": f"{proposed_bid/1e6:.1f} (bid) - {estimated_cost/1e6:.1f} (cost) = {profit/1e6:.1f} USDC ({'profit' if profit >= 0 else 'LOSS'})"
                }
            except Exception as e:
                logger.error(f"validate_bid_profitability failed: {e}")
                return {
                    "error": str(e),
                    "verdict": "ERROR - Could not validate profitability",
                    "recommendation": "Do not bid until validation succeeds"
                }
        
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
                
                # Decode common smart contract errors for better agent understanding
                error_msg = str(e)
                
                # BidScoreNotCompetitive - 0x29e8399d
                if "0x29e8399d" in error_msg or "BidScoreNotCompetitive" in error_msg:
                    return {
                        "success": False,
                        "error": "BidScoreNotCompetitive",
                        "error_code": "0x29e8399d",
                        "explanation": "Your bid score (combining bid amount and reputation) is not better than the current winning bid. In a reverse auction, LOWER bids win, but the bid score also factors in your reputation.",
                        "suggestion": "Try a significantly lower bid amount (10-30% less) to improve your competitiveness. Check the current winning bid and aim to beat it by a meaningful margin.",
                        "retry_recommended": True
                    }
                
                # BidTooHigh - 0xc9b80cd4
                elif "0xc9b80cd4" in error_msg or "BidTooHigh" in error_msg:
                    return {
                        "success": False,
                        "error": "BidTooHigh",
                        "error_code": "0xc9b80cd4",
                        "explanation": "Your bid amount exceeds the maximum price set by the consumer for this auction.",
                        "suggestion": "Bid below the max_price shown in the auction details.",
                        "retry_recommended": True
                    }
                
                # AuctionNotActive - 0x15e5e7f5
                elif "0x15e5e7f5" in error_msg or "AuctionNotActive" in error_msg:
                    return {
                        "success": False,
                        "error": "AuctionNotActive",
                        "error_code": "0x15e5e7f5",
                        "explanation": "The auction has already ended or been cancelled. You can no longer place bids.",
                        "suggestion": "Look for other active auctions to bid on.",
                        "retry_recommended": False
                    }
                
                # AgentNotEligible - 0x5c427cd9
                elif "0x5c427cd9" in error_msg or "AgentNotEligible" in error_msg:
                    return {
                        "success": False,
                        "error": "AgentNotEligible",
                        "error_code": "0x5c427cd9",
                        "explanation": "You are not eligible to bid on this auction. The consumer may have restricted bidding to specific agents.",
                        "suggestion": "This auction is not available to you. Look for other auctions without eligibility restrictions.",
                        "retry_recommended": False
                    }
                
                # Generic error
                else:
                    return {
                        "success": False,
                        "error": str(e),
                        "explanation": "An unexpected error occurred while placing the bid.",
                        "suggestion": "Review the error message and check if the auction is still active and you meet all requirements."
                    }
        
        return [get_ipfs_data, get_reputation, estimate_cost, validate_bid_profitability, place_bid]
    
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
            
            # Use blockchain time for expiry to handle time-travel scenarios (e.g., in tests)
            current_block = await self.client.w3.eth.get_block('latest')
            block_timestamp = current_block['timestamp']
            expiry = block_timestamp + 3600  # 1 hour from current block time
            
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
                        
                        # Fetch full auction details
                        try:
                            auction = await self.client.call_contract_method(
                                "ReverseAuction",
                                "getAuctionDetails",
                                auction_id
                            )
                            
                            auction_info = {
                                "auction_id": auction_id,
                                "buyer_address": auction[1],
                                "service_cid": auction[2],
                                "max_price": auction[3],
                                "duration": auction[4],
                                "start_time": auction[5],
                                "winning_agent_id": auction[7],
                                "winning_bid": auction[8],
                                "is_active": auction[9],
                                "is_completed": auction[10],
                                "escrow_amount": auction[11],
                                "reputation_weight": auction[12]
                            }
                            
                            won_auctions.append(auction_info)
                            logger.info(f"🎉 Won auction {auction_id}! Buyer: {auction[1]}, Service CID: {auction[2]}")
                        except Exception as e:
                            logger.error(f"Error fetching details for won auction {auction_id}: {e}")
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
- estimate_cost(complexity): Estimate cost to deliver a service. Pass 'low', 'medium', or 'high' based on service complexity from IPFS data
- validate_bid_profitability(estimated_cost, proposed_bid): MANDATORY tool to check if a bid is profitable before placing it
- place_bid(auction_id, bid_amount): Submit a bid for an auction

BIDDING GUIDELINES:
1. Analyze each auction before bidding
2. Fetch service requirements from IPFS using get_ipfs_data
3. Extract the 'complexity' field from service data and use it to estimate cost
4. CRITICAL: ALWAYS use validate_bid_profitability to check if your proposed bid is profitable
   - Pass your estimated_cost and proposed_bid to this tool
   - Only proceed if verdict is PROFITABLE
   - NEVER bid if verdict is UNPROFITABLE - you will lose money!
5. Only bid if profitable (bid_amount > estimated_cost) as confirmed by validation
6. Consider time remaining - urgent auctions may need immediate bids
7. Check current winning bid - you need a better score to win
8. This is a reverse auction where LOWER bids win, but your bid score is also weighted by your reputation
9. You can bid on multiple auctions if profitable
10. If no auctions are profitable, don't bid

IMPORTANT: The validate_bid_profitability tool will show you the math clearly. Trust its verdict.

Current eligible auctions:
{auctions_context}

Analyze these auctions and decide which to bid on."""

            # Create ReAct agent without rate limiting (use API defaults)
            llm = ChatOpenAI(
                model="xiaomi/mimo-v2-flash:free",
                api_key=self.config.openrouter_api_key,
                base_url=self.config.openrouter_base_url,
                temperature=0.3
            )
            
            react_agent = create_agent(llm, self._tools)
            
            result = await react_agent.ainvoke({"messages": [HumanMessage(content=system_prompt)]})
            
            # Log the agent's reasoning trace
            logger.info("\n" + "=" * 80)
            logger.info("🤖 AGENT REASONING TRACE")
            logger.info("=" * 80)
            for i, msg in enumerate(result["messages"], 1):
                msg_class = msg.__class__.__name__
                
                # Human/System messages
                if hasattr(msg, 'content') and msg.content:
                    content = str(msg.content)
                    if len(content) > 500:
                        content = content[:500] + "..."
                    logger.info(f"\n[{i}] {msg_class}:\n{content}")
                
                # AI messages with tool calls
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    logger.info(f"\n[{i}] {msg_class} - Tool Calls:")
                    for tc in msg.tool_calls:
                        tool_name = tc.get('name', 'unknown')
                        tool_args = tc.get('args', {})
                        logger.info(f"  🔧 {tool_name}({tool_args})")
                
                # Tool messages (results)
                if msg_class == "ToolMessage":
                    tool_output = str(msg.content)[:300]
                    logger.info(f"\n[{i}] ToolMessage:\n  ✅ {tool_output}")
            
            logger.info("=" * 80 + "\n")
            
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
    
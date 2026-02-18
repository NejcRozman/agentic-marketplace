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
    from ..infrastructure.cost_tracker import CostTracker, LLMCostCallback
    from ..infrastructure.prompts import get_prompt
    from ..config import config, get_architecture
except ImportError:
    from infrastructure.blockchain_client import BlockchainClient
    from infrastructure.contract_abis import (
        get_reverse_auction_abi,
        get_identity_registry_abi,
        get_reputation_registry_abi
    )
    from infrastructure.feedback_auth import generate_feedback_auth, verify_feedback_auth_format
    from infrastructure.ipfs_client import IPFSClient
    from infrastructure.cost_tracker import CostTracker, LLMCostCallback
    from infrastructure.prompts import get_prompt
    from config import Config, get_architecture
    config = Config()

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """
    State for blockchain agent workflow.
    
    Organized by temporal dimension (current/history/future) and parameter type
    (performance/market/decision). Not all fields need to be populated - simpler
    architectures can leave history/future fields empty without affecting reasoning.
    """
    
    # ============================================================================
    # WORKFLOW CONTROL (always populated)
    # ============================================================================
    agent_id: int
    action: str  # "complete_service" or "monitor"
    
    # Transaction results
    tx_hash: Optional[str]
    error: Optional[str]
    messages: List[Any]
    
    # ============================================================================
    # CURRENT STATE - Present moment snapshot
    # ============================================================================

    # Active auctions and market conditions
    eligible_active_auctions: List[Dict[str, Any]]  # Auctions we can bid on (BC + IPFS data)
    won_auctions: List[Dict[str, Any]]  # Auctions we won (for service completion) 

    # Current reputation context
    competitors_reputation: List[Dict[str, Any]]  # All other agents bidding in auctions with their reputation
    agent_reputation: Dict[str, Any]  # Agent's current reputation score and feedback count

    # Agent's current performance context
    estimated_service_cost: Optional[float]  # Estimated cost of service execution (USD)
    
    # Results of current decision cycle
    bids_placed: List[Dict[str, Any]]  # Bids submitted this invocation

    # Service completion workflow (only for complete_service action)
    auction_id: Optional[int]
    client_address: Optional[str]
    feedback_auth: Optional[bytes]
    
    # ============================================================================
    # HISTORY STATE - Past observations and outcomes
    # ============================================================================
    
    # --- History Performance Parameters ---
    # past_execution_costs: Optional[List[float]]  # Historical service execution costs (USD)
    # past_auctions: Optional[List[Dict[str, Any]]]  # Last N auctions participated
    # past_win_rate: Optional[float]  # Win rate over last N auctions
    # past_avg_profit: Optional[float]  # Average profit per won auction
    # past_reputation_trajectory: Optional[List[float]]  # Reputation over time
    
    # --- History Market Parameters ---
    # past_market_prices: Optional[List[float]]  # Historical winning bids
    # past_competition_levels: Optional[List[float]]  # Competition over time
    
    # --- History Decision Parameters ---
    # past_bidding_strategy: Optional[List[str]]  # What strategies were used
    # past_bid_outcomes: Optional[List[Dict[str, Any]]]  # Win/loss with details
    
    # ============================================================================
    # FUTURE STATE - Predictions and forward-looking parameters
    # ============================================================================
    
    # --- Future Market Parameters ---
    # predicted_competition: Optional[float]  # Expected future competition
    # predicted_price_trend: Optional[str]  # "increasing", "stable", "decreasing"
    
    # --- Future Performance Parameters ---
    # expected_reputation_change: Optional[float]  # If we win/lose this auction
    # expected_capacity_utilization: Optional[float]  # Future workload
    
    # --- Future Decision Parameters ---
    # recommended_bid_range: Optional[Dict[str, float]]  # Min/max profitable bids
    # risk_assessment: Optional[str]  # "low", "medium", "high"


class BlockchainHandler:
    """
    Agentic blockchain handler using LangGraph.
    
    - complete_service: Deterministic path for service completion
    - monitor: ReAct agent for intelligent bidding decisions
    """
    
    def __init__(
        self, 
        agent_id: int, 
        blockchain_client: Optional[BlockchainClient] = None,
        architecture: Optional[str] = None,
        system_prompt: Optional[str] = None,
        llm_model: Optional[str] = None,
        llm_temperature: Optional[float] = None,
        llm_base_url: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        enabled_tools: Optional[List[str]] = None,
        reasoning_mode: Optional[str] = None,
        cost_tracker: Optional[CostTracker] = None
    ):
        """
        Initialize the blockchain handler agent.
        
        Args:
            agent_id: The agent's ID
            blockchain_client: Optional blockchain client instance
            architecture: Architecture name. If None, uses config.architecture
            system_prompt: System prompt for the ReAct agent (if None, uses architecture's template)
            llm_model: LLM model identifier (if None, uses architecture's default)
            llm_temperature: LLM temperature (if None, uses architecture's default)
            llm_base_url: LLM API base URL (default: from config)
            llm_api_key: LLM API key (default: from config)
            enabled_tools: List of tool names to enable (if None, uses architecture's tools)
            reasoning_mode: "deterministic" or "llm_react" (if None, uses architecture's mode)
            cost_tracker: Optional CostTracker instance (creates new if None)
        """
        self.agent_id = agent_id
        self.client = blockchain_client or BlockchainClient()
        self.config = config
        
        # Load architecture configuration
        arch_name = architecture or self.config.architecture
        self.arch_config = get_architecture(arch_name)
        
        # Apply architecture settings (can be overridden by explicit kwargs)
        self.reasoning_mode = reasoning_mode or self.arch_config.reasoning_mode
        if self.reasoning_mode not in ["deterministic", "llm_react", "llm_strategic"]:
            raise ValueError(f"Invalid reasoning_mode: {self.reasoning_mode}")
        
        # System prompt configuration
        self.system_prompt = system_prompt
        
        # LLM configuration (with architecture defaults)
        self.llm_model = llm_model or self.arch_config.llm_model
        self.llm_temperature = llm_temperature if llm_temperature is not None else self.arch_config.llm_temperature
        self.llm_base_url = llm_base_url or self.config.openrouter_base_url
        self.llm_api_key = llm_api_key or self.config.openrouter_api_key
        
        # Tools configuration (from architecture)
        self.enabled_tools = enabled_tools or self.arch_config.enabled_tools
        
        # Store coupling mode from architecture (used in state population)
        self.coupling_mode = self.arch_config.coupling_mode
        
        self.contracts_loaded = False
        self.reverse_auction_contract = None
        self.identity_registry_contract = None
        self.reputation_registry_contract = None
        
        # Cost tracking (create new if not provided)
        self.cost_tracker = cost_tracker or CostTracker(agent_id=agent_id, config=self.config)
        
        # Block tracking - persists across invocations
        self._last_processed_block: int = 0
        
        # Bids placed tracker (for ReAct tool results)
        self._bids_placed: List[Dict[str, Any]] = []
        
        # Build tools for ReAct agent
        self._tools = self._build_tools()
        
        # Build the graph
        self.graph = self._build_graph()
        
        prompt_info = "custom" if system_prompt else self.arch_config.prompt_template
        logger.info(
            f"BlockchainHandler initialized for agent {agent_id} "
            f"(architecture={self.arch_config.name}, reasoning_mode={self.reasoning_mode}, "
            f"coupling_mode={self.coupling_mode}, prompt={prompt_info}, "
            f"model={self.llm_model}, temp={self.llm_temperature}, tools={len(self._tools)})"
        )
    
    def _build_tools(self) -> List:
        """Build tools for the ReAct agent."""
        handler = self

        # ============================================================================
        # COMPUTATION TOOLS - perform operation LLM is bad at
        # ============================================================================
        
        @tool
        def validate_bid_profitability(estimated_cost: int, proposed_bid: int) -> Dict[str, Any]:
            """Check if a proposed bid is profitable compared to estimated cost.
            
            Args:
                estimated_cost: Your estimated cost to deliver the service (in USDC with 6 decimals, e.g., 50000000 = 50 USDC)
                proposed_bid: The bid amount you're considering (in USDC with 6 decimals, e.g., 55000000 = 55 USDC)
            
            Returns:
                Profitability analysis with verdict and profit margin
            """
            try:
                profit = proposed_bid - estimated_cost
                is_profitable = proposed_bid > estimated_cost
                
                if estimated_cost > 0:
                    margin_percent = round((profit / estimated_cost) * 100, 2)
                    loss_percent = -margin_percent if not is_profitable else 0
                else:
                    margin_percent = 0
                    loss_percent = 0
                
                return {
                    "is_profitable": is_profitable,
                    "estimated_cost": estimated_cost,
                    "proposed_bid": proposed_bid,
                    "profit": profit,
                    "profit_margin_percent": margin_percent if is_profitable else -loss_percent,
                    "summary": f"{proposed_bid/1e6:.1f} USDC bid - {estimated_cost/1e6:.1f} USDC cost = {profit/1e6:.1f} USDC {'profit' if profit >= 0 else 'LOSS'} ({margin_percent if is_profitable else -loss_percent:.1f}%)"
                }
            except Exception as e:
                logger.error(f"validate_bid_profitability failed: {e}")
                return {
                    "error": str(e),
                    "is_profitable": False
                }
        
        @tool
        def calculate_bid_score(bid_amount: int, agent_reputation: int) -> Dict[str, Any]:
            """Calculate the bid score that will be used in auction ranking.
            
            The contract calculates bid score as: bid * (100 + reputation) / 100
            Lower scores win in reverse auctions. Higher reputation gives better (lower) scores.
            
            Args:
                bid_amount: The bid amount in USDC (with 6 decimals, e.g., 50000000 = 50 USDC)
                agent_reputation: The reputation score (0-100, where 50 is neutral)
            
            Returns:
                Dictionary with bid_score and explanation
            """
            try:
                # Contract formula: bidScore = bid * (100 + reputation) / 100
                bid_score = (bid_amount * (100 + agent_reputation)) // 100
                
                reputation_effect = agent_reputation - 50  # How much better/worse than neutral
                if reputation_effect > 0:
                    advantage = f"{reputation_effect} points above neutral gives you a {reputation_effect}% advantage"
                elif reputation_effect < 0:
                    advantage = f"{-reputation_effect} points below neutral gives you a {-reputation_effect}% disadvantage"
                else:
                    advantage = "neutral reputation (no advantage or disadvantage)"
                
                return {
                    "bid_amount": bid_amount,
                    "agent_reputation": agent_reputation,
                    "bid_score": bid_score,
                    "bid_amount_usdc": round(bid_amount / 1e6, 2),
                    "bid_score_usdc": round(bid_score / 1e6, 2),
                    "reputation_effect": advantage,
                    "summary": f"Bid {bid_amount/1e6:.2f} USDC with reputation {agent_reputation} = score {bid_score/1e6:.2f} USDC ({advantage})"
                }
            except Exception as e:
                logger.error(f"calculate_bid_score failed: {e}")
                return {
                    "error": str(e),
                    "bid_score": bid_amount  # Fallback to bid amount
                }
        
        @tool
        def simulate_bid_outcome(
            proposed_bid: int,
            your_reputation: int,
            current_winning_bid: int = 0,
            current_winner_reputation: int = 50
        ) -> Dict[str, Any]:
            """Simulate whether a proposed bid would win against the current winner.
            
            Checks if your bid score (bid + reputation weight) would beat the current winning bid.
            Prevents wasting gas on bids that will revert with BidScoreNotCompetitive.
            
            Args:
                proposed_bid: Your proposed bid amount (in USDC with 6 decimals)
                your_reputation: Your reputation score (0-100, from agent_reputation in state)
                current_winning_bid: Current winning bid amount (0 if no bids yet, in USDC with 6 decimals)
                current_winner_reputation: Current winner's reputation (default 50 if unknown)
            
            Returns:
                Analysis of whether you would win and by what margin
            """
            try:
                # Calculate our bid score: bid * (100 + reputation) / 100
                our_score = (proposed_bid * (100 + your_reputation)) // 100
                
                # If there's a current winner, compare scores
                if current_winning_bid > 0:
                    current_winner_score = (current_winning_bid * (100 + current_winner_reputation)) // 100
                    
                    # In reverse auction, LOWER score wins
                    will_win = our_score < current_winner_score
                    margin = current_winner_score - our_score  # Positive = we're better
                    margin_percent = (margin / current_winner_score * 100) if current_winner_score > 0 else 0
                    
                    return {
                        "proposed_bid": proposed_bid,
                        "proposed_bid_usdc": round(proposed_bid / 1e6, 2),
                        "your_score": our_score,
                        "your_score_usdc": round(our_score / 1e6, 2),
                        "your_reputation": your_reputation,
                        "current_winning_bid": current_winning_bid,
                        "current_winning_bid_usdc": round(current_winning_bid / 1e6, 2),
                        "current_winner_score": current_winner_score,
                        "current_winner_score_usdc": round(current_winner_score / 1e6, 2),
                        "current_winner_reputation": current_winner_reputation,
                        "will_win": will_win,
                        "margin": margin,
                        "margin_usdc": round(margin / 1e6, 2),
                        "margin_percent": round(margin_percent, 2),
                        "summary": f"{'✅ WILL WIN' if will_win else '❌ WILL LOSE'}: Your score {our_score/1e6:.2f} vs current {current_winner_score/1e6:.2f} (margin: {margin/1e6:.2f} USDC, {margin_percent:.1f}%)"
                    }
                else:
                    # No current winner - we'll be first bid
                    return {
                        "proposed_bid": proposed_bid,
                        "proposed_bid_usdc": round(proposed_bid / 1e6, 2),
                        "your_score": our_score,
                        "your_score_usdc": round(our_score / 1e6, 2),
                        "your_reputation": your_reputation,
                        "current_winning_bid": 0,
                        "will_win": True,
                        "summary": "✅ WILL WIN: No current bids, you'll be the first bidder"
                    }
                    
            except Exception as e:
                logger.error(f"simulate_bid_outcome failed: {e}")
                return {
                    "error": str(e),
                    "will_win": False,
                    "explanation": "Failed to simulate bid outcome."
                }

        # ============================================================================
        # ACTUATION TOOLS - perform actions that affect the world (e.g., placing bids)
        # ============================================================================
                
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
                    # Track gas cost
                    gas_used = receipt.get('gasUsed', 0)
                    gas_price_wei = receipt.get('effectiveGasPrice', 0)
                    handler.cost_tracker.add_gas_cost(
                        gas_used=gas_used,
                        gas_price_wei=gas_price_wei,
                        context="place_bid"
                    )
                    
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
        
        # Build tool list based on enabled_tools configuration
        all_tools = {
            "validate_bid_profitability": validate_bid_profitability,
            "calculate_bid_score": calculate_bid_score,
            "simulate_bid_outcome": simulate_bid_outcome,
            "place_bid": place_bid
        }
        
        enabled = [all_tools[name] for name in self.enabled_tools if name in all_tools]
        
        if len(enabled) < len(self.enabled_tools):
            missing = set(self.enabled_tools) - set(all_tools.keys())
            logger.warning(f"Some requested tools not found: {missing}")
        
        logger.info(f"Enabled tools: {[t.name for t in enabled]}")
        return enabled
    
    def _get_default_system_prompt(self, auctions_context: str) -> str:
        """
        Generate default system prompt for the ReAct agent.
        
        This is used as fallback when no custom system_prompt is provided.
        In experiments, custom prompts should be injected via config.
        
        Args:
            auctions_context: JSON string of eligible auctions
            
        Returns:
            Default system prompt string
        """
        # Dynamically generate tool descriptions based on enabled tools
        tool_descriptions = {
            "validate_bid_profitability": "validate_bid_profitability(estimated_cost, proposed_bid): Check if a bid is profitable (returns profit/loss calculations)",
            "calculate_bid_score": "calculate_bid_score(bid_amount, agent_reputation): Calculate bid score used in auction ranking",
            "simulate_bid_outcome": "simulate_bid_outcome(proposed_bid, your_reputation, current_winning_bid, current_winner_reputation): Check if your bid would win against current winner",
            "place_bid": "place_bid(auction_id, bid_amount): Submit a bid for an auction"
        }
        
        enabled_tool_descs = [f"- {tool_descriptions[name]}" for name in self.enabled_tools if name in tool_descriptions]
        tools_section = "\n".join(enabled_tool_descs)
        
        return f"""You are a bidding agent (ID: {self.agent_id}) for a decentralized AI service marketplace.

Available tools:
{tools_section}

BIDDING GUIDELINES:
1. Analyze each auction before bidding
2. Service requirements are already embedded in auction details
3. Use validate_bid_profitability to check if your proposed bid is profitable
4. Use calculate_bid_score to understand how reputation affects your bid score
5. Use simulate_bid_outcome to check if your bid will actually win (prevents BidScoreNotCompetitive reverts)
6. Consider time remaining - urgent auctions may need immediate bids
7. This is a reverse auction where LOWER bids win, and your bid score is weighted by reputation
8. You can bid on multiple auctions if profitable
9. If no auctions are profitable, don't bid

Current eligible auctions:
{auctions_context}

Analyze these auctions and decide which to bid on."""
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow for blockchain operations."""
        workflow = StateGraph(AgentState)
        
        workflow.add_node("router", self._router_node)
        workflow.add_node("generate_feedback_auth", self._generate_feedback_auth_node)
        workflow.add_node("call_complete_service", self._call_complete_service_node)
        workflow.add_node("gather_state", self._gather_state_node)
        workflow.add_node("reasoning", self._reasoning_node)
        
        workflow.set_entry_point("router")
        
        workflow.add_conditional_edges(
            "router",
            self._route_action,
            {"complete_service": "generate_feedback_auth", "monitor": "gather_state"}
        )
        
        workflow.add_edge("generate_feedback_auth", "call_complete_service")
        workflow.add_edge("call_complete_service", END)
        workflow.add_edge("gather_state", "reasoning")
        workflow.add_edge("reasoning", END)
        
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
    
    def _router_node(self, state: AgentState) -> AgentState:
        """Router node - entry point that examines action field."""
        logger.info(f"🔀 Router: action={state['action']}")
        return state
    
    async def _fetch_reputation(self, agent_id: int) -> Dict[str, Any]:
        """Fetch reputation for an agent.
        
        Args:
            agent_id: Agent ID to fetch reputation for
            
        Returns:
            Dictionary with rating and feedback_count
        """
        try:
            if not self.reputation_registry_contract:
                return {"rating": 50, "feedback_count": 0}
            
            empty_addresses = []
            zero_bytes32 = b'\x00' * 32
            
            result = await self.client.call_contract_method(
                "ReputationRegistry",
                "getSummary",
                agent_id,
                empty_addresses,
                zero_bytes32,
                zero_bytes32
            )
            feedback_count = result[0]
            average_score = result[1] if feedback_count > 0 else 50
            
            return {"rating": average_score, "feedback_count": feedback_count}
        except Exception as e:
            logger.warning(f"Error fetching reputation for agent {agent_id}: {e}")
            return {"rating": 50, "feedback_count": 0}
    
    def _route_action(self, state: AgentState) -> str:
        """Routing function for action field."""
        if state.get("action") == "complete_service":
            return "complete_service"
        return "monitor"
    
    async def _generate_feedback_auth_node(self, state: AgentState) -> AgentState:
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
    
    async def _call_complete_service_node(self, state: AgentState) -> AgentState:
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
                # Track gas cost
                gas_used = receipt.get('gasUsed', 0)
                gas_price_wei = receipt.get('effectiveGasPrice', 0)
                self.cost_tracker.add_gas_cost(
                    gas_used=gas_used,
                    gas_price_wei=gas_price_wei,
                    context="complete_service"
                )
                
                # Track revenue from winning auction
                auction_data = await self.client.call_contract(
                    "ReverseAuction",
                    "auctions",
                    auction_id
                )
                winning_bid = auction_data[3]  # winningBid field
                revenue_usd = winning_bid / 1e6  # Convert from 6 decimals to USD
                
                self.cost_tracker.add_revenue(
                    revenue_usd=revenue_usd,
                    context=f"auction_{auction_id}"
                )
                
                logger.info(f"💰 Revenue from auction {auction_id}: ${revenue_usd:.2f} USD")
                
                state["tx_hash"] = tx_hash
                logger.info(f"✅ Service completed: {tx_hash}")
            else:
                raise Exception("Transaction failed")
            
        except Exception as e:
            logger.error(f"Error calling completeService: {e}")
            state["error"] = str(e)
        
        return state
    
    async def _gather_state_node(self, state: AgentState) -> AgentState:
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
                                "reputation_weight": auction[12],
                                "effort_tier": None  # To be populated by bidding strategy
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
            
            # Pre-fetch agent's own reputation
            agent_reputation = await self._fetch_reputation(self.agent_id)
            state["agent_reputation"] = agent_reputation
            logger.info(f"📊 Agent reputation: rating={agent_reputation['rating']}, feedback_count={agent_reputation['feedback_count']}")
            
            # Pre-fetch IPFS service requirements and embed in auction data
            ipfs_client = IPFSClient()
            for auction in eligible_active_auctions:
                cid = auction.get("service_description_cid")
                if cid:
                    try:
                        service_reqs = await ipfs_client.fetch_json(cid)
                        auction["service_requirements"] = service_reqs if service_reqs else {}
                        logger.debug(f"📥 Fetched service requirements for auction {auction['auction_id']}")
                    except Exception as e:
                        logger.warning(f"Failed to fetch IPFS data for auction {auction['auction_id']}: {e}")
                        auction["service_requirements"] = {}
                else:
                    auction["service_requirements"] = {}
            
            # Pre-fetch competitors' reputation (winning agents in active auctions)
            competitors_reputation = []
            seen_agents = set()
            for auction in eligible_active_auctions:
                winning_agent_id = auction.get("winning_agent_id")
                if winning_agent_id and winning_agent_id != self.agent_id and winning_agent_id not in seen_agents:
                    seen_agents.add(winning_agent_id)
                    comp_rep = await self._fetch_reputation(winning_agent_id)
                    competitors_reputation.append({
                        "agent_id": winning_agent_id,
                        "rating": comp_rep["rating"],
                        "feedback_count": comp_rep["feedback_count"]
                    })
            
            state["competitors_reputation"] = competitors_reputation
            logger.info(f"👥 Fetched reputation for {len(competitors_reputation)} competitors")
            
            # Populate state based on architecture's state_level and coupling_mode
            if self.arch_config.state_level == 0:
                # State Level 0: Current only (no history) - Architecture 1
                state["estimated_service_cost"] = self.config.bidding_base_cost
            elif self.arch_config.state_level >= 1:
                # State Level 1+: Performance history (if coupling allows) - Future architectures
                if self.coupling_mode in ["one_way", "two_way"]:
                    execution_history = self.cost_tracker.get_execution_cost_history()
                    # For now, use most recent cost or base cost as fallback
                    state["estimated_service_cost"] = execution_history[-1] if execution_history else self.config.bidding_base_cost
                else:
                    # Isolated coupling: no history available
                    state["estimated_service_cost"] = self.config.bidding_base_cost
            
            # State Level 2+ (market history) - to be implemented for future architectures
            # if self.arch_config.state_level >= 2:
            #     state["past_winning_bids"] = await self._fetch_market_history()
            
            logger.info(f"✅ State gathered: {len(won_auctions)} won, {len(eligible_active_auctions)} active eligible")
            
        except Exception as e:
            logger.error(f"Error gathering state: {e}")
            state["error"] = str(e)
            state["won_auctions"] = []
            state["eligible_active_auctions"] = []
        
        return state
    
    async def _reasoning_node(self, state: AgentState) -> AgentState:
        """Route to appropriate reasoning strategy based on reasoning_mode."""
        if self.reasoning_mode == "deterministic":
            return await self._deterministic_reasoning(state)
        else:
            return await self._llm_react_reasoning(state)
    
    async def _deterministic_reasoning(self, state: AgentState) -> AgentState:
        """Deterministic bidding strategy: bid = cost × markup."""
        logger.info("🎯 Using deterministic reasoning (fixed strategy)")
        
        eligible_auctions = state.get("eligible_active_auctions", [])
        
        if not eligible_auctions:
            logger.info("No eligible auctions to consider")
            state["bids_placed"] = []
            return state
        
        # Reset bids tracker
        self._bids_placed = []
        
        try:
            # Fixed markup strategy: bid = cost × 1.3 (30% profit margin)
            markup = 1.3
            
            for auction in eligible_auctions:
                auction_id = auction["auction_id"]
                max_price = auction["max_price"]
                
                # Estimate cost using base cost
                base_cost = self.config.bidding_base_cost
                estimated_cost = int(base_cost * 1e6)  # Convert to 6 decimals
                
                # Calculate bid with fixed markup
                bid_amount = int(estimated_cost * markup)
                
                # Check if bid is within max_price
                if bid_amount > max_price:
                    logger.info(
                        f"⏭️ Skipping auction {auction_id}: bid {bid_amount} exceeds max_price {max_price}"
                    )
                    continue
                
                # Check profitability (should always be profitable with fixed markup)
                profit = bid_amount - estimated_cost
                profit_margin = (profit / estimated_cost) * 100
                
                logger.info(
                    f"📊 Auction {auction_id}: cost={estimated_cost/1e6:.1f} USDC, "
                    f"bid={bid_amount/1e6:.1f} USDC, margin={profit_margin:.1f}%"
                )
                
                # Place bid
                try:
                    estimated_gas = await self.client.estimate_gas(
                        "ReverseAuction",
                        "placeBid",
                        auction_id,
                        bid_amount,
                        self.agent_id
                    )
                    
                    tx_hash = await self.client.send_transaction(
                        "ReverseAuction",
                        "placeBid",
                        auction_id,
                        bid_amount,
                        self.agent_id,
                        gas_limit=estimated_gas + 50000
                    )
                    
                    receipt = await self.client.wait_for_transaction(tx_hash)
                    
                    if receipt['status'] == 1:
                        # Track gas cost
                        gas_used = receipt.get('gasUsed', 0)
                        gas_price_wei = receipt.get('effectiveGasPrice', 0)
                        self.cost_tracker.add_gas_cost(
                            gas_used=gas_used,
                            gas_price_wei=gas_price_wei,
                            context="place_bid"
                        )
                        
                        logger.info(f"✅ Bid placed successfully: {tx_hash}")
                        result = {
                            "success": True,
                            "tx_hash": tx_hash,
                            "auction_id": auction_id,
                            "bid_amount": bid_amount,
                            "block_number": receipt['blockNumber']
                        }
                        self._bids_placed.append(result)
                    else:
                        logger.warning(f"❌ Bid transaction failed for auction {auction_id}")
                        
                except Exception as e:
                    logger.error(f"Error placing bid on auction {auction_id}: {e}")
                    continue
            
            state["bids_placed"] = self._bids_placed
            logger.info(f"✅ Deterministic reasoning completed: {len(self._bids_placed)} bids placed")
            
        except Exception as e:
            logger.error(f"Error in deterministic reasoning: {e}")
            state["error"] = str(e)
            state["bids_placed"] = []
        
        return state
    
    async def _llm_react_reasoning(self, state: AgentState) -> AgentState:
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
            
            # Use injected system_prompt or generate from architecture's template
            if self.system_prompt is not None:
                # Use custom prompt from config
                prompt = self.system_prompt.format(
                    agent_id=self.agent_id,
                    auctions_context=auctions_context
                )
                logger.info("Using custom system prompt from config")
            else:
                # Use architecture's prompt template
                prompt = get_prompt(
                    architecture=self.arch_config.prompt_template,
                    agent_id=self.agent_id,
                    auctions_context=auctions_context,
                    past_execution_costs=state.get("past_execution_costs"),
                    # past_winning_bids=state.get("past_winning_bids"),  # For future architectures
                )
                logger.info(f"Using {self.arch_config.name} prompt")
            
            # Create ReAct agent with configured LLM and cost tracking
            callback = LLMCostCallback(
                cost_tracker=self.cost_tracker,
                model=self.llm_model,
                config=self.config
            )
            
            llm = ChatOpenAI(
                model=self.llm_model,
                api_key=self.llm_api_key,
                base_url=self.llm_base_url,
                temperature=self.llm_temperature,
                callbacks=[callback]
            )
            
            react_agent = create_agent(llm, self._tools)
            
            result = await react_agent.ainvoke({"messages": [HumanMessage(content=prompt)]})
            
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
        initial_state: AgentState = {
            "agent_id": self.agent_id,
            "action": "monitor",
            "auction_id": None,
            "client_address": None,
            "feedback_auth": None,
            "won_auctions": [],
            "eligible_active_auctions": [],
            "competitors_reputation": [],
            "agent_reputation": {"rating": 50, "feedback_count": 0},
            "estimated_service_cost": None,
            "bids_placed": [],
            "tx_hash": None,
            "error": None,
            "messages": []
        }
        
        result = await self.graph.ainvoke(initial_state)
        
        # Log cost summary after monitoring cycle
        self.cost_tracker.log_summary()
        
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
        initial_state: AgentState = {
            "agent_id": self.agent_id,
            "action": "complete_service",
            "auction_id": auction_id,
            "client_address": client_address,
            "feedback_auth": None,
            "won_auctions": [],
            "eligible_active_auctions": [],
            "competitors_reputation": [],
            "agent_reputation": {"rating": 50, "feedback_count": 0},
            "estimated_service_cost": None,
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
    
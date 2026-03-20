"""
BlockchainHandler - Agentic handler for blockchain operations.

This agent uses LangGraph to handle blockchain operations:
- Complete service path: Deterministic workflow for completing auctions
- Monitor path: Heuristic or ReAct agent with tools for intelligent bidding decisions

Built with LangGraph for agentic reasoning and decision-making.
"""

import asyncio
import logging
import json
import time
import random
import re
from typing import Dict, Any, List, Optional
from datetime import datetime
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
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
    - monitor: Heuristic or ReAct agent for intelligent bidding decisions
    """

    # Keep weight semantics in percent (0-100) while increasing score precision.
    SCORE_PRECISION = 10_000
    WEIGHT_SCALE = 100
    
    def __init__(
        self, 
        agent_id: int, 
        blockchain_client: Optional[BlockchainClient] = None,
        architecture: Optional[str] = None,
        system_prompt: Optional[str] = None,
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
            llm_temperature: LLM temperature (if None, uses architecture's default)
            llm_base_url: LLM API base URL (default: from config)
            llm_api_key: LLM API key (default: from config)
            enabled_tools: List of tool names to enable (if None, uses architecture's tools)
            reasoning_mode: "heuristic" or "llm_react" (if None, uses architecture's mode)
            cost_tracker: Optional CostTracker instance (creates new if None)
        """
        self.agent_id = agent_id
        self.client = blockchain_client or BlockchainClient()
        self.config = config
        self.initial_reputation_default = self.config.initial_reputation_default
        
        # Load architecture configuration
        arch_name = architecture or self.config.architecture
        self.arch_config = get_architecture(arch_name)
        
        # Apply architecture settings (can be overridden by explicit kwargs)
        self.reasoning_mode = reasoning_mode or self.arch_config.reasoning_mode
        if self.reasoning_mode not in ["deterministic", "heuristic", "llm_react", "llm_strategic"]:
            raise ValueError(f"Invalid reasoning_mode: {self.reasoning_mode}")
        
        # System prompt configuration
        self.system_prompt = system_prompt
        
        # LLM configuration (with architecture defaults)
        self.llm_model = self.arch_config.blockchain_llm
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
        self._react_stop_after_bid_attempt: bool = False

        # Per-cycle state snapshot used by tool-call logging
        self._tool_audit_context: Dict[str, Any] = {}
        
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
        logger.info("Tool-call logging enabled (state snapshot + raw tool args)")
    
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
            raw_args = {
                "estimated_cost": estimated_cost,
                "proposed_bid": proposed_bid,
            }
            try:
                profit = proposed_bid - estimated_cost
                is_profitable = proposed_bid > estimated_cost
                
                if estimated_cost > 0:
                    margin_percent = round((profit / estimated_cost) * 100, 2)
                    loss_percent = -margin_percent if not is_profitable else 0
                else:
                    margin_percent = 0
                    loss_percent = 0
                
                result = {
                    "is_profitable": is_profitable,
                    "estimated_cost": estimated_cost,
                    "proposed_bid": proposed_bid,
                    "profit": profit,
                    "profit_margin_percent": margin_percent if is_profitable else -loss_percent,
                    "summary": f"{proposed_bid/1e6:.6f} USDC bid - {estimated_cost/1e6:.6f} USDC cost = {profit/1e6:.6f} USDC {'profit' if profit >= 0 else 'LOSS'} ({margin_percent if is_profitable else -loss_percent:.3f}%)"
                }
                handler._audit_tool_call(
                    tool_name="validate_bid_profitability",
                    llm_args=raw_args,
                    normalized_args={
                        "estimated_cost": estimated_cost,
                        "proposed_bid": proposed_bid,
                    },
                    result=result,
                )
                return result
            except Exception as e:
                logger.error(f"validate_bid_profitability failed: {e}")
                result = {
                    "error": str(e),
                    "is_profitable": False
                }
                handler._audit_tool_call(
                    tool_name="validate_bid_profitability",
                    llm_args=raw_args,
                    normalized_args=raw_args,
                    result=result,
                    error=str(e),
                )
                return result
        
        @tool
        def calculate_bid_score(
            bid_amount: int,
            agent_reputation: int,
            max_price: int = 0,
            reputation_weight: int = 20
        ) -> Dict[str, Any]:
            """Calculate the bid score that will be used in auction ranking.
            
            Contract scoring (higher score wins):
            - normalized_reputation = reputation (0-100)
            - normalized_bid_score = ((max_price - bid_amount) * SCORE_PRECISION) / max_price
            - score = (reputation_weight * normalized_reputation +
                      (100 - reputation_weight) * normalized_bid_score) / 100
            
            Args:
                bid_amount: The bid amount in USDC (with 6 decimals)
                agent_reputation: The reputation score (0-100, where 50 is neutral)
                max_price: Auction max price (required for exact score)
                reputation_weight: Auction reputation weight (0-100)
            
            Returns:
                Dictionary with bid_score and explanation
            """
            raw_args = {
                "bid_amount": bid_amount,
                "agent_reputation": agent_reputation,
                "max_price": max_price,
                "reputation_weight": reputation_weight,
            }
            try:
                if max_price <= 0:
                    result = {
                        "error": "max_price is required for exact contract score",
                        "bid_amount": bid_amount,
                        "agent_reputation": agent_reputation,
                        "max_price": max_price,
                        "reputation_weight": reputation_weight,
                        "summary": "Provide max_price to calculate exact on-chain score."
                    }
                    handler._audit_tool_call(
                        tool_name="calculate_bid_score",
                        llm_args=raw_args,
                        normalized_args=raw_args,
                        result=result,
                        error=result["error"],
                    )
                    return result

                if bid_amount > max_price:
                    result = {
                        "error": "bid exceeds max_price",
                        "bid_amount": bid_amount,
                        "max_price": max_price,
                        "summary": "Bid exceeds auction max price and will revert (BidTooHigh)."
                    }
                    handler._audit_tool_call(
                        tool_name="calculate_bid_score",
                        llm_args=raw_args,
                        normalized_args=raw_args,
                        result=result,
                        error=result["error"],
                    )
                    return result

                rw = max(0, min(handler.WEIGHT_SCALE, int(reputation_weight)))
                normalized_reputation = int(agent_reputation) * (handler.SCORE_PRECISION // handler.WEIGHT_SCALE)
                normalized_bid_score = ((max_price - bid_amount) * handler.SCORE_PRECISION) // max_price
                bid_score = (
                    rw * normalized_reputation + (handler.WEIGHT_SCALE - rw) * normalized_bid_score
                ) // handler.WEIGHT_SCALE

                result = {
                    "bid_amount": bid_amount,
                    "agent_reputation": agent_reputation,
                    "max_price": max_price,
                    "reputation_weight": rw,
                    "normalized_reputation": normalized_reputation,
                    "normalized_bid_score": normalized_bid_score,
                    "bid_score": bid_score,
                    "bid_amount_usdc": round(bid_amount / 1e6, 6),
                    "higher_is_better": True,
                    "summary": (
                        f"Score={bid_score} (higher is better): rep_component={normalized_reputation} "
                        f"@w={rw}%, bid_component={normalized_bid_score} @w={100-rw}%"
                    )
                }
                handler._audit_tool_call(
                    tool_name="calculate_bid_score",
                    llm_args=raw_args,
                    normalized_args={
                        "bid_amount": bid_amount,
                        "agent_reputation": agent_reputation,
                        "max_price": max_price,
                        "reputation_weight": rw,
                    },
                    result=result,
                )
                return result
            except Exception as e:
                logger.error(f"calculate_bid_score failed: {e}")
                result = {
                    "error": str(e),
                    "bid_score": 0
                }
                handler._audit_tool_call(
                    tool_name="calculate_bid_score",
                    llm_args=raw_args,
                    normalized_args=raw_args,
                    result=result,
                    error=str(e),
                )
                return result
        
        @tool
        def simulate_bid_outcome(
            proposed_bid: int,
            your_reputation: int,
            current_winning_bid: int = 0,
            current_winner_reputation: Optional[int] = None,
            max_price: int = 0,
            reputation_weight: int = 25
        ) -> Dict[str, Any]:
            """Simulate whether a proposed bid would win against the current winner.
            
            Uses the contract score formula and checks if your score beats current best score.
            Prevents wasting gas on bids that will revert with BidScoreNotCompetitive.
            
            Args:
                proposed_bid: Your proposed bid amount (in USDC with 6 decimals)
                your_reputation: Your reputation score (0-100, from agent_reputation in state)
                current_winning_bid: Current winning bid amount (0 if no bids yet, in USDC with 6 decimals)
                current_winner_reputation: Current winner's reputation (uses configured default if unknown)
                max_price: Auction max price (required for exact simulation)
                reputation_weight: Auction reputation weight (0-100)
            
            Returns:
                Analysis of whether you would win and by what margin
            """
            raw_args = {
                "proposed_bid": proposed_bid,
                "your_reputation": your_reputation,
                "current_winning_bid": current_winning_bid,
                "current_winner_reputation": current_winner_reputation,
                "max_price": max_price,
                "reputation_weight": reputation_weight,
            }
            try:
                if max_price <= 0:
                    result = {
                        "error": "max_price is required for exact simulation",
                        "will_win": False,
                        "summary": "Provide max_price to simulate on-chain outcome accurately."
                    }
                    handler._audit_tool_call(
                        tool_name="simulate_bid_outcome",
                        llm_args=raw_args,
                        normalized_args=raw_args,
                        result=result,
                        error=result["error"],
                    )
                    return result

                if proposed_bid > max_price:
                    result = {
                        "proposed_bid": proposed_bid,
                        "proposed_bid_usdc": round(proposed_bid / 1e6, 6),
                        "max_price": max_price,
                        "will_win": False,
                        "would_revert": True,
                        "revert_reason": "BidTooHigh",
                        "summary": "❌ WILL REVERT: proposed bid exceeds auction max_price"
                    }
                    handler._audit_tool_call(
                        tool_name="simulate_bid_outcome",
                        llm_args=raw_args,
                        normalized_args=raw_args,
                        result=result,
                        error=result.get("revert_reason"),
                    )
                    return result

                rw = max(0, min(handler.WEIGHT_SCALE, int(reputation_weight)))
                our_rep_scaled = int(your_reputation) * (handler.SCORE_PRECISION // handler.WEIGHT_SCALE)
                our_bid_component = ((max_price - proposed_bid) * handler.SCORE_PRECISION) // max_price
                our_score = (
                    rw * our_rep_scaled + (handler.WEIGHT_SCALE - rw) * our_bid_component
                ) // handler.WEIGHT_SCALE
                
                # If there's a current winner, compare scores
                if current_winning_bid > 0:
                    if current_winning_bid > max_price:
                        result = {
                            "error": "current_winning_bid exceeds max_price",
                            "will_win": False,
                            "summary": "Invalid auction data: current_winning_bid > max_price"
                        }
                        handler._audit_tool_call(
                            tool_name="simulate_bid_outcome",
                            llm_args=raw_args,
                            normalized_args=raw_args,
                            result=result,
                            error=result["error"],
                        )
                        return result

                    winner_rep = self.initial_reputation_default if current_winner_reputation is None else int(current_winner_reputation)
                    winner_rep_scaled = winner_rep * (handler.SCORE_PRECISION // handler.WEIGHT_SCALE)
                    current_bid_component = ((max_price - current_winning_bid) * handler.SCORE_PRECISION) // max_price
                    current_winner_score = (
                        rw * winner_rep_scaled + (handler.WEIGHT_SCALE - rw) * current_bid_component
                    ) // handler.WEIGHT_SCALE
                    
                    # Higher score wins in this contract
                    will_win = our_score > current_winner_score
                    margin = our_score - current_winner_score  # Positive = we're better
                    margin_percent = (margin / current_winner_score * 100) if current_winner_score > 0 else 0
                    
                    result = {
                        "proposed_bid": proposed_bid,
                        "proposed_bid_usdc": round(proposed_bid / 1e6, 6),
                        "your_score": our_score,
                        "your_bid_component": our_bid_component,
                        "your_reputation": your_reputation,
                        "current_winning_bid": current_winning_bid,
                        "current_winning_bid_usdc": round(current_winning_bid / 1e6, 6),
                        "current_winner_score": current_winner_score,
                        "current_winner_bid_component": current_bid_component,
                        "current_winner_reputation": winner_rep,
                        "max_price": max_price,
                        "reputation_weight": rw,
                        "higher_is_better": True,
                        "will_win": will_win,
                        "margin": margin,
                        "margin_percent": round(margin_percent, 3),
                        "summary": f"{'✅ WILL WIN' if will_win else '❌ WILL LOSE'}: Your score {our_score} vs current {current_winner_score} (margin: {margin}, {margin_percent:.3f}%)"
                    }
                    handler._audit_tool_call(
                        tool_name="simulate_bid_outcome",
                        llm_args=raw_args,
                        normalized_args={
                            "proposed_bid": proposed_bid,
                            "your_reputation": int(your_reputation),
                            "current_winning_bid": int(current_winning_bid),
                            "current_winner_reputation": winner_rep,
                            "max_price": max_price,
                            "reputation_weight": rw,
                        },
                        result=result,
                    )
                    return result
                else:
                    # No current winner - we'll be first bid
                    result = {
                        "proposed_bid": proposed_bid,
                        "proposed_bid_usdc": round(proposed_bid / 1e6, 6),
                        "your_score": our_score,
                        "your_bid_component": our_bid_component,
                        "your_reputation": your_reputation,
                        "max_price": max_price,
                        "reputation_weight": rw,
                        "higher_is_better": True,
                        "current_winning_bid": 0,
                        "will_win": True,
                        "summary": "✅ WILL WIN: No current bids, you'll be the first bidder"
                    }
                    handler._audit_tool_call(
                        tool_name="simulate_bid_outcome",
                        llm_args=raw_args,
                        normalized_args={
                            "proposed_bid": proposed_bid,
                            "your_reputation": int(your_reputation),
                            "current_winning_bid": int(current_winning_bid),
                            "current_winner_reputation": current_winner_reputation,
                            "max_price": max_price,
                            "reputation_weight": rw,
                        },
                        result=result,
                    )
                    return result
                    
            except Exception as e:
                logger.error(f"simulate_bid_outcome failed: {e}")
                result = {
                    "error": str(e),
                    "will_win": False,
                    "explanation": "Failed to simulate bid outcome."
                }
                handler._audit_tool_call(
                    tool_name="simulate_bid_outcome",
                    llm_args=raw_args,
                    normalized_args=raw_args,
                    result=result,
                    error=str(e),
                )
                return result

        # ============================================================================
        # ACTUATION TOOLS - perform actions that affect the world (e.g., placing bids)
        # ============================================================================
                
        @tool(return_direct=True)
        def place_bid(auction_id: int, bid_amount: int) -> Dict[str, Any]:
            """Submit a bid for an auction on the blockchain."""
            raw_args = {"auction_id": auction_id, "bid_amount": bid_amount}
            handler._react_stop_after_bid_attempt = True
            try:
                logger.info(
                    f"📤 Placing bid: auction={auction_id}, amount={bid_amount} ({bid_amount/1e6:.6f} USDC)"
                )

                fresh_auction = asyncio.run(
                    handler.client.call_contract_method("ReverseAuction", "getAuctionDetails", auction_id)
                )
                max_price = int(fresh_auction[3])
                duration = int(fresh_auction[4])
                start_time = int(fresh_auction[5])
                current_winning_agent = int(fresh_auction[7])
                current_winning_bid = int(fresh_auction[8])
                is_active = bool(fresh_auction[9])
                reputation_weight = int(fresh_auction[12])

                latest_block = asyncio.run(handler.client.get_block("latest"))
                current_time = int(latest_block["timestamp"])
                end_time = start_time + duration

                if (not is_active) or current_time >= end_time:
                    result = {
                        "success": False,
                        "error": "AuctionNotActive",
                        "error_code": "0x69b8d0fe",
                        "explanation": "Auction ended before bid submission. State refreshed and tx skipped.",
                        "retry_recommended": False,
                    }
                    handler._audit_tool_call("place_bid", raw_args, raw_args, result, result.get("error_code"))
                    return result

                if bid_amount > max_price:
                    result = {
                        "success": False,
                        "error": "BidTooHigh",
                        "error_code": "0xc9b80cd4",
                        "explanation": "Bid exceeds current auction max_price. State refreshed and tx skipped.",
                        "retry_recommended": True,
                    }
                    handler._audit_tool_call("place_bid", raw_args, raw_args, result, result.get("error_code"))
                    return result

                if current_winning_agent == handler.agent_id:
                    result = {
                        "success": False,
                        "error": "AlreadyWinning",
                        "explanation": "You are already the current winner. No tx sent.",
                        "retry_recommended": False,
                    }
                    handler._audit_tool_call("place_bid", raw_args, raw_args, result, result.get("error"))
                    return result

                rw = max(0, min(handler.WEIGHT_SCALE, reputation_weight))
                our_rep = (handler._tool_audit_context or {}).get("agent_reputation")
                if not isinstance(our_rep, int):
                    our_rep = handler.initial_reputation_default
                our_rep_scaled = int(our_rep) * (handler.SCORE_PRECISION // handler.WEIGHT_SCALE)
                our_bid_component = ((max_price - bid_amount) * handler.SCORE_PRECISION) // max_price
                our_score = (
                    rw * our_rep_scaled + (handler.WEIGHT_SCALE - rw) * our_bid_component
                ) // handler.WEIGHT_SCALE

                if current_winning_bid > 0:
                    winner_rep = handler.initial_reputation_default
                    if current_winning_agent and current_winning_agent != handler.agent_id:
                        try:
                            winner_rep = int(
                                asyncio.run(handler._fetch_reputation(current_winning_agent)).get(
                                    "rating", handler.initial_reputation_default
                                )
                            )
                        except Exception:
                            winner_rep = handler.initial_reputation_default
                    winner_rep_scaled = winner_rep * (handler.SCORE_PRECISION // handler.WEIGHT_SCALE)
                    current_bid_component = ((max_price - current_winning_bid) * handler.SCORE_PRECISION) // max_price
                    current_score = (
                        rw * winner_rep_scaled + (handler.WEIGHT_SCALE - rw) * current_bid_component
                    ) // handler.WEIGHT_SCALE
                    if our_score <= current_score:
                        result = {
                            "success": False,
                            "error": "BidScoreNotCompetitive",
                            "error_code": "0x29e8399d",
                            "explanation": "Refreshed on-chain state shows bid is not competitive; tx skipped.",
                            "current_winning_bid": current_winning_bid,
                            "current_winning_agent": current_winning_agent,
                            "retry_recommended": False,
                        }
                        handler._audit_tool_call("place_bid", raw_args, raw_args, result, result.get("error_code"))
                        return result

                estimated_gas = asyncio.run(
                    handler.client.estimate_gas("ReverseAuction", "placeBid", auction_id, bid_amount, handler.agent_id)
                )
                tx_hash = asyncio.run(
                    handler.client.send_transaction(
                        "ReverseAuction",
                        "placeBid",
                        auction_id,
                        bid_amount,
                        handler.agent_id,
                        gas_limit=estimated_gas + 50000,
                    )
                )
                receipt = asyncio.run(handler.client.wait_for_transaction(tx_hash))

                if receipt["status"] == 1:
                    handler.cost_tracker.add_gas_cost(
                        gas_used=receipt.get("gasUsed", 0),
                        gas_price_wei=receipt.get("effectiveGasPrice", 0),
                        context="place_bid",
                    )
                    logger.info(f"✅ Bid placed successfully: {tx_hash}")
                    result = {
                        "success": True,
                        "tx_hash": tx_hash,
                        "auction_id": auction_id,
                        "bid_amount": bid_amount,
                        "block_number": receipt["blockNumber"],
                    }
                    handler._bids_placed.append(result)
                    handler._audit_tool_call("place_bid", raw_args, raw_args, result)
                    return result

                result = {"success": False, "error": "Transaction failed"}
                handler._audit_tool_call("place_bid", raw_args, raw_args, result, result["error"])
                return result

            except Exception as e:
                logger.error(f"Error placing bid: {e}")
                error_msg = str(e)

                if "0x29e8399d" in error_msg or "BidScoreNotCompetitive" in error_msg:
                    result = {
                        "success": False,
                        "error": "BidScoreNotCompetitive",
                        "error_code": "0x29e8399d",
                        "explanation": "Your weighted score is not better than the current winning bid.",
                        "suggestion": "Try a lower bid amount to improve competitiveness.",
                        "retry_recommended": False,
                    }
                    handler._audit_tool_call("place_bid", raw_args, raw_args, result, result.get("error_code"))
                    return result
                if "0xc9b80cd4" in error_msg or "BidTooHigh" in error_msg:
                    result = {
                        "success": False,
                        "error": "BidTooHigh",
                        "error_code": "0xc9b80cd4",
                        "explanation": "Your bid amount exceeds auction max price.",
                        "retry_recommended": True,
                    }
                    handler._audit_tool_call("place_bid", raw_args, raw_args, result, result.get("error_code"))
                    return result
                if "0x69b8d0fe" in error_msg or "AuctionNotActive" in error_msg:
                    result = {
                        "success": False,
                        "error": "AuctionNotActive",
                        "error_code": "0x69b8d0fe",
                        "explanation": "Auction is no longer active.",
                        "retry_recommended": False,
                    }
                    handler._audit_tool_call("place_bid", raw_args, raw_args, result, result.get("error_code"))
                    return result
                if "0x5c427cd9" in error_msg or "AgentNotEligible" in error_msg:
                    result = {
                        "success": False,
                        "error": "AgentNotEligible",
                        "error_code": "0x5c427cd9",
                        "explanation": "Agent is not eligible for this auction.",
                        "retry_recommended": False,
                    }
                    handler._audit_tool_call("place_bid", raw_args, raw_args, result, result.get("error_code"))
                    return result

                result = {
                    "success": False,
                    "error": str(e),
                    "explanation": "An unexpected error occurred while placing the bid.",
                    "suggestion": "Check auction status and eligibility before retrying.",
                }
                handler._audit_tool_call("place_bid", raw_args, raw_args, result, result.get("error"))
                return result

        # Build tool list based on enabled_tools configuration
        all_tools = {
            "validate_bid_profitability": validate_bid_profitability,
            "calculate_bid_score": calculate_bid_score,
            "simulate_bid_outcome": simulate_bid_outcome,
            "place_bid": place_bid
        }
        
        enabled = []
        for name in self.enabled_tools:
            if name not in all_tools:
                continue
            enabled.append(all_tools[name])
        
        if len(enabled) < len(self.enabled_tools):
            missing = set(self.enabled_tools) - set(all_tools.keys())
            logger.warning(f"Some requested tools not found: {missing}")
        
        logger.info(f"Enabled tools: {[t.name for t in enabled]}")
        return enabled

    @staticmethod
    def _normalize_tool_name(raw_name: str) -> str:
        """Normalize malformed tool names emitted by the model.

        Keeps canonical names intact while stripping known transport artifacts.
        """
        if not raw_name:
            return raw_name

        name = str(raw_name).strip().strip("`").strip('"').strip("'")
        name = re.sub(r"\s+", "", name)

        # Common malformed suffix pattern: tool_name<|channel|>commentary
        if "<|" in name:
            name = name.split("<|", 1)[0]

        return name

    class _ToolNameCanonicalizationMiddleware(AgentMiddleware):
        """Canonicalize malformed tool names before ToolNode dispatch."""

        def __init__(self, tool_lookup: Dict[str, Any], normalize_fn):
            self._tool_lookup = tool_lookup
            self._normalize_fn = normalize_fn

        def _rewrite_request_if_needed(self, request):
            call = request.tool_call or {}
            raw_name = call.get("name", "")
            canonical_name = self._normalize_fn(raw_name)

            if not canonical_name or canonical_name == raw_name:
                return request

            tool_obj = self._tool_lookup.get(canonical_name)
            if tool_obj is None:
                return request

            updated_call = dict(call)
            updated_call["name"] = canonical_name
            updated_request = request.override(tool_call=updated_call)

            # ToolCallRequest.override does not include `tool`; set it explicitly
            # so ToolNode can execute the resolved canonical tool.
            object.__setattr__(updated_request, "tool", tool_obj)

            logger.warning(f"Normalized malformed tool name '{raw_name}' -> '{canonical_name}'")
            return updated_request

        def wrap_tool_call(self, request, handler):
            return handler(self._rewrite_request_if_needed(request))

        async def awrap_tool_call(self, request, handler):
            return await handler(self._rewrite_request_if_needed(request))

    def _build_tool_audit_context(self, state: AgentState) -> Dict[str, Any]:
        """Build per-cycle state snapshot to include in each tool-call audit event."""
        auctions = state.get("eligible_active_auctions", []) or []
        auction_map = {}
        for a in auctions:
            auction_id = a.get("auction_id")
            if auction_id is None:
                continue
            auction_map[str(auction_id)] = {
                "max_price": a.get("max_price"),
                "winning_bid": a.get("winning_bid"),
                "reputation_weight": a.get("reputation_weight"),
                "time_remaining": a.get("time_remaining"),
            }

        estimated_cost_usd = state.get("estimated_service_cost")
        estimated_cost_micro = None
        if isinstance(estimated_cost_usd, (int, float)):
            estimated_cost_micro = int(float(estimated_cost_usd) * 1e6)

        return {
            "agent_id": self.agent_id,
            "agent_reputation": (state.get("agent_reputation") or {}).get("rating"),
            "estimated_service_cost_usd": estimated_cost_usd,
            "estimated_service_cost_micro": estimated_cost_micro,
            "auctions": auction_map,
            "auction_count": len(auctions),
        }

    def _audit_tool_call(
        self,
        tool_name: str,
        llm_args: Dict[str, Any],
        normalized_args: Optional[Dict[str, Any]] = None,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        """Record a structured tool-call event with state snapshot and raw LLM args."""
        event = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "agent_id": self.agent_id,
            "tool": tool_name,
            "llm_args": llm_args,
            "current_state": self._tool_audit_context or {},
        }
        logger.info(f"TOOL_AUDIT {json.dumps(event, default=str)}")
    
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
            "calculate_bid_score": "calculate_bid_score(bid_amount, agent_reputation, max_price, reputation_weight): Calculate exact contract bid score (higher score wins)",
            "simulate_bid_outcome": "simulate_bid_outcome(proposed_bid, your_reputation, current_winning_bid, current_winner_reputation, max_price, reputation_weight): Check if your bid would beat current winner",
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
7. This contract ranks bids by weighted score where HIGHER score wins (lower bids improve score via bid component, reputation also improves score)
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
                return {"rating": self.initial_reputation_default, "feedback_count": 0}
            
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
            average_score = result[1] if feedback_count > 0 else self.initial_reputation_default
            
            return {"rating": average_score, "feedback_count": feedback_count}
        except Exception as e:
            logger.warning(f"Error fetching reputation for agent {agent_id}: {e}")
            return {"rating": self.initial_reputation_default, "feedback_count": 0}
    
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
                auction_data = await self.client.call_contract_method(
                    "ReverseAuction",
                    "getAuctionDetails",
                    auction_id
                )
                # getAuctionDetails struct index 8 = winningBid
                winning_bid = auction_data[8]
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
        if self.reasoning_mode in ["deterministic", "heuristic"]:
            return await self._heuristic_reasoning(state)
        else:
            return await self._llm_react_reasoning(state)
    
    async def _heuristic_reasoning(self, state: AgentState) -> AgentState:
        """Heuristic bidding strategy with selectable approaches."""
        strategy = (self.config.heuristic_strategy or "random_markup").lower()
        min_margin = max(0.0, self.config.heuristic_min_margin)
        max_margin = max(min_margin, self.config.heuristic_max_margin)

        logger.info(
            f"🎯 Using heuristic reasoning (strategy={strategy}, "
            f"min_margin={min_margin:.2f}, max_margin={max_margin:.2f})"
        )
        
        eligible_auctions = state.get("eligible_active_auctions", [])
        
        if not eligible_auctions:
            logger.info("No eligible auctions to consider")
            state["bids_placed"] = []
            return state
        
        # Reset bids tracker
        self._bids_placed = []
        
        try:
            for auction in eligible_auctions:
                auction_id = auction["auction_id"]
                max_price = auction["max_price"]
                winning_agent_id = auction.get("winning_agent_id", 0) or 0
                time_remaining = auction.get("time_remaining", 0) or 0

                # Skip rebidding if we are already current winner
                if winning_agent_id == self.agent_id:
                    logger.info(f"⏭️ Skipping auction {auction_id}: already winning")
                    continue

                # Skip last-second bids that are likely to expire before inclusion
                if time_remaining <= 10:
                    logger.info(
                        f"⏭️ Skipping auction {auction_id}: only {time_remaining}s remaining"
                    )
                    continue
                
                # Estimate cost using base cost
                base_cost = self.config.bidding_base_cost
                estimated_cost = int(base_cost * 1e6)  # Convert to 6 decimals

                # Strategy 1: Random markup within configured margin bounds
                if strategy == "random_markup":
                    markup = 1.0 + random.uniform(min_margin, max_margin)
                    bid_amount = int(estimated_cost * markup)
                # Strategy 2: Feasible random within profitable and competitive region
                elif strategy == "feasible_random":
                    min_profitable_bid = int(estimated_cost * (1.0 + min_margin))

                    winning_bid = auction.get("winning_bid", 0) or 0
                    reputation_weight = int(auction.get("reputation_weight", 50) or 50)
                    our_reputation = int(
                        (state.get("agent_reputation") or {}).get(
                            "rating", self.initial_reputation_default
                        )
                    )
                    winner_reputation = self.initial_reputation_default
                    if winning_agent_id and winning_agent_id != self.agent_id:
                        for rep in state.get("competitors_reputation", []):
                            if int(rep.get("agent_id", 0) or 0) == int(winning_agent_id):
                                winner_reputation = int(rep.get("rating", self.initial_reputation_default))
                                break
                        else:
                            rep_data = await self._fetch_reputation(int(winning_agent_id))
                            winner_reputation = int(rep_data.get("rating", self.initial_reputation_default))

                    if winning_bid > 0:
                        rw = max(0, min(self.WEIGHT_SCALE, reputation_weight))
                        our_reputation_scaled = our_reputation * (self.SCORE_PRECISION // self.WEIGHT_SCALE)
                        winner_reputation_scaled = winner_reputation * (self.SCORE_PRECISION // self.WEIGHT_SCALE)
                        current_bid_component = ((max_price - winning_bid) * self.SCORE_PRECISION) // max_price
                        current_winner_score = (
                            rw * winner_reputation_scaled + (self.WEIGHT_SCALE - rw) * current_bid_component
                        ) // self.WEIGHT_SCALE

                        if rw >= 100:
                            # Bid component has zero effect when reputation is fully weighted.
                            max_competitive_bid = max_price if our_reputation > winner_reputation else -1
                        else:
                            required_numerator = self.WEIGHT_SCALE * (current_winner_score + 1) - (rw * our_reputation_scaled)
                            if required_numerator <= 0:
                                min_required_bid_component = 0
                            else:
                                min_required_bid_component = (
                                    required_numerator + (self.WEIGHT_SCALE - rw) - 1
                                ) // (self.WEIGHT_SCALE - rw)

                            if min_required_bid_component > self.SCORE_PRECISION:
                                max_competitive_bid = -1
                            else:
                                # Enforce ((max_price - bid) * SCORE_PRECISION // max_price) >= min_required_bid_component
                                min_delta = (
                                    min_required_bid_component * max_price + self.SCORE_PRECISION - 1
                                ) // self.SCORE_PRECISION
                                max_competitive_bid = max_price - min_delta
                    else:
                        max_competitive_bid = max_price

                    # Cap to auction max price
                    max_competitive_bid = min(max_competitive_bid, max_price)

                    if max_competitive_bid < min_profitable_bid:
                        logger.info(
                            f"⏭️ Skipping auction {auction_id}: no profitable competitive region "
                            f"(min_profitable={min_profitable_bid}, max_competitive={max_competitive_bid})"
                        )
                        continue

                    bid_amount = random.randint(min_profitable_bid, max_competitive_bid)
                else:
                    logger.warning(f"Unknown heuristic strategy '{strategy}', defaulting to random_markup")
                    markup = 1.0 + random.uniform(min_margin, max_margin)
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
                    f"📊 Auction {auction_id}: cost={estimated_cost/1e6:.6f} USDC, "
                    f"bid={bid_amount/1e6:.6f} USDC, margin={profit_margin:.3f}%"
                )

                # Strict pre-bid refresh gate: if refresh fails, skip bid.
                try:
                    fresh_auction = await self.client.call_contract_method(
                        "ReverseAuction",
                        "getAuctionDetails",
                        auction_id,
                    )
                    if not isinstance(fresh_auction, (list, tuple)) or len(fresh_auction) < 13:
                        raise ValueError("invalid getAuctionDetails payload")

                    fresh_max_price = int(fresh_auction[3])
                    fresh_duration = int(fresh_auction[4])
                    fresh_start_time = int(fresh_auction[5])
                    fresh_winning_agent = int(fresh_auction[7])
                    fresh_winning_bid = int(fresh_auction[8])
                    fresh_is_active = bool(fresh_auction[9])
                    fresh_reputation_weight = int(fresh_auction[12])

                    latest_block = await self.client.get_block("latest")
                    current_time = int(latest_block["timestamp"])
                    end_time = fresh_start_time + fresh_duration
                except Exception as e:
                    logger.warning(
                        f"Pre-bid refresh check failed for auction {auction_id}: {e}. Skipping bid."
                    )
                    continue

                if (not fresh_is_active) or (current_time >= end_time):
                    logger.info(
                        f"⏭️ Skipping auction {auction_id}: auction no longer active"
                    )
                    continue

                if bid_amount > fresh_max_price:
                    logger.info(
                        f"⏭️ Skipping auction {auction_id}: bid {bid_amount} exceeds max_price {fresh_max_price}"
                    )
                    continue

                if fresh_winning_agent == self.agent_id:
                    logger.info(
                        f"⏭️ Skipping auction {auction_id}: already winning"
                    )
                    continue

                rw = max(0, min(self.WEIGHT_SCALE, fresh_reputation_weight))
                our_rep = int(
                    (state.get("agent_reputation") or {}).get(
                        "rating", self.initial_reputation_default
                    )
                )
                our_rep_scaled = our_rep * (self.SCORE_PRECISION // self.WEIGHT_SCALE)
                our_bid_component = ((fresh_max_price - bid_amount) * self.SCORE_PRECISION) // fresh_max_price
                our_score = (
                    rw * our_rep_scaled + (self.WEIGHT_SCALE - rw) * our_bid_component
                ) // self.WEIGHT_SCALE

                if fresh_winning_bid > 0:
                    winner_rep = self.initial_reputation_default
                    if fresh_winning_agent and fresh_winning_agent != self.agent_id:
                        for rep in state.get("competitors_reputation", []):
                            if int(rep.get("agent_id", 0) or 0) == int(fresh_winning_agent):
                                winner_rep = int(rep.get("rating", self.initial_reputation_default))
                                break
                        else:
                            rep_data = await self._fetch_reputation(fresh_winning_agent)
                            winner_rep = int(rep_data.get("rating", self.initial_reputation_default))

                    winner_rep_scaled = winner_rep * (self.SCORE_PRECISION // self.WEIGHT_SCALE)
                    current_bid_component = ((fresh_max_price - fresh_winning_bid) * self.SCORE_PRECISION) // fresh_max_price
                    current_score = (
                        rw * winner_rep_scaled + (self.WEIGHT_SCALE - rw) * current_bid_component
                    ) // self.WEIGHT_SCALE

                    if our_score <= current_score:
                        logger.info(
                            f"⏭️ Skipping auction {auction_id}: non-competitive "
                            f"(our_score={our_score}, current_score={current_score})"
                        )
                        continue
                
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
            logger.info(f"✅ Heuristic reasoning completed: {len(self._bids_placed)} bids placed")
            
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
        self._react_stop_after_bid_attempt = False
        self._tool_audit_context = self._build_tool_audit_context(state)
        
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
                    agent_state=state,
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
            
            tool_lookup = {tool.name: tool for tool in self._tools}
            middleware = [
                self._ToolNameCanonicalizationMiddleware(
                    tool_lookup=tool_lookup,
                    normalize_fn=self._normalize_tool_name,
                )
            ]
            react_agent = create_agent(llm, self._tools, middleware=middleware)
            
            recursion_limit = int(getattr(self.config, "react_recursion_limit", 20))
            result = await react_agent.ainvoke(
                {"messages": [HumanMessage(content=prompt)]},
                config={"recursion_limit": recursion_limit}
            )
            
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

            if self._react_stop_after_bid_attempt:
                logger.info("ReAct cycle ended after place_bid attempt; next monitor cycle will use refreshed blockchain state.")

            state["bids_placed"] = self._bids_placed
            logger.info(f"✅ ReAct completed: {len(self._bids_placed)} bids placed")
            
        except Exception as e:
            err_text = str(e)
            if "GRAPH_RECURSION_LIMIT" in err_text or "Recursion limit" in err_text:
                logger.warning(f"ReAct recursion limit reached; returning partial bidding decisions: {e}")
                state["error"] = None
                state["bids_placed"] = self._bids_placed
                return state

            logger.error(f"Error in ReAct reasoning: {e}")
            state["error"] = err_text
            state["bids_placed"] = []
        finally:
            # Avoid stale cross-cycle context
            self._tool_audit_context = {}
        
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
            "agent_reputation": {"rating": self.initial_reputation_default, "feedback_count": 0},
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
            "agent_reputation": {"rating": self.initial_reputation_default, "feedback_count": 0},
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
    
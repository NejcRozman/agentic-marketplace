"""Prompt templates for agent architectures."""

import json
from typing import Any, Dict


def get_prompt(architecture: str, agent_state: Dict[str, Any]) -> str:
    """
    Generate prompt for specified architecture.
    
    Args:
        architecture: Architecture identifier ("1", "2", "3", etc.)
        agent_state: Full provider state for the current reasoning cycle
    
    Returns:
        Formatted prompt string
    """
    if architecture == "1":
        return _prompt_arch_1(agent_state)
    elif architecture == "2":
        return _prompt_arch_2(agent_state)
    elif architecture == "3":
        return _prompt_arch_3(agent_state)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


def _prompt_arch_1(agent_state: Dict[str, Any]) -> str:
    """
    Architecture 1: LLM Minimal
    - State: Current only (no history)
    - Reasoning: LLM ReAct
    - Coupling: Isolated
    - Prompt: Identity + boundaries only
    """
    agent_id = agent_state.get("agent_id", "unknown")
    auctions_context = json.dumps(agent_state.get("eligible_active_auctions", []), indent=2)
    rep = (agent_state.get("agent_reputation") or {}).get("rating")
    competitor_reps = agent_state.get("competitors_reputation", [])
    est_cost_usd = agent_state.get("estimated_service_cost")
    est_cost_micro = int(float(est_cost_usd) * 1e6) if isinstance(est_cost_usd, (int, float)) else None

    return f"""You are bidding agent {agent_id} in a decentralized AI marketplace.

Your goal: Maximize profit by bidding on profitable auctions.

You will have an opportunity to bid multiple times (at least twice) in the same auction, so you can analyze and adjust your bid based on the current winning bid and reputation of competitors. You can also track the duration of the auction and adjust your strategy as the auction progresses. The total duration of each auction is 120 seconds.

Auction type: reverse auction. Lower bids help, but reputation also matters.

Contract scoring logic (higher score wins):
- normalized_reputation = reputation * 100
- normalized_bid_component = ((max_price - bid_amount) * 10000) / max_price
- score = (reputation_weight * normalized_reputation + (100 - reputation_weight) * normalized_bid_component) / 100

Bid sensitivity (important):
- Score precision is 10000, so a 1-step change in bid component is roughly `max_price / 10000` (in micro-USDC units).
- Changes smaller than that often do not affect score due to integer flooring.
- To reliably improve competitiveness, adjust bids by at least this effective tick (and sometimes more, depending on reputation weight).

Your current runtime state:
- agent_reputation: {rep}
- competitor_reputations: {competitor_reps}
- estimated_service_cost_usd: {est_cost_usd}
- estimated_service_cost_micro: {est_cost_micro}

Available auctions:
{auctions_context}

Tool usage policy (strict):
1. Use state values directly, do not invent replacements:
   - For `agent_reputation` or `your_reputation`, use {rep}
   - For `estimated_cost`, use {est_cost_micro}
2. Always pass exact auction `max_price` and `reputation_weight` from the selected auction.
3. Monetary arguments must be integer micro-USDC (6 decimals).
    - Example: 0.055000 USDC -> 55000
    - Do not pass floats for tool monetary fields (`estimated_cost`, `proposed_bid`, `bid_amount`, `current_winning_bid`).
4. Never call unknown tools or malformed tool names.
    - Do not append any suffix/prefix tokens to tool names (for example `<|channel|>commentary`).
    - Tool names must be exactly one of: validate_bid_profitability, calculate_bid_score, simulate_bid_outcome, place_bid.
5. Do not place a bid if unprofitable or not competitive.

Recommended tool sequence per auction:
1. `validate_bid_profitability(estimated_cost, proposed_bid)`
2. `simulate_bid_outcome(proposed_bid, your_reputation, current_winning_bid, current_winner_reputation, max_price, reputation_weight)`
3. Optional: `calculate_bid_score(...)` for explanation/confirmation
4. `place_bid(auction_id, bid_amount)` only if profitable and likely to win

You can also calculate score for other agents using `calculate_bid_score` and their reputation and potential prices to determine boundaries of their bids.

Stop conditions:
- Execute max 4 analysis steps, then make a final decision.
- If already current winner with a competitive bid, skip rebidding.
- If no auction is profitable, return skip. """


def _prompt_arch_2(agent_state: Dict[str, Any]) -> str:
    """Architecture 2: LLM with performance history (to be implemented)."""
    raise NotImplementedError("Architecture 2 prompt not yet implemented")


def _prompt_arch_3(agent_state: Dict[str, Any]) -> str:
    """Architecture 3: LLM with market history + guidance (to be implemented)."""
    raise NotImplementedError("Architecture 3 prompt not yet implemented")

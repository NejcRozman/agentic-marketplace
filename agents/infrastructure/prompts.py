"""Prompt templates for agent architectures."""

import json
from typing import Any, Dict


def _build_shared_strategic_core_prompt(
    agent_state: Dict[str, Any],
    architecture_mode: str,
    architecture_data_policy: str,
    include_history: bool,
    include_balance: bool,
    max_reasoning_steps: int = 6,
) -> str:
    """Build a shared strategic prompt core with architecture-specific data visibility."""
    agent_id = agent_state.get("agent_id", "unknown")
    auctions_context = json.dumps(agent_state.get("eligible_active_auctions", []), indent=2)
    rep = (agent_state.get("agent_reputation") or {}).get("rating")
    competitor_reps = agent_state.get("competitors_reputation", [])

    est_cost_usd = agent_state.get("estimated_service_cost")
    est_cost_micro = int(float(est_cost_usd) * 1e6) if isinstance(est_cost_usd, (int, float)) else None

    history = agent_state.get("past_execution_costs") or []
    history = [float(v) for v in history if isinstance(v, (int, float))]
    history_count = len(history)
    recent_history = history[-5:]
    avg_cost = (sum(history) / history_count) if history_count > 0 else None
    min_cost = min(history) if history_count > 0 else None
    max_cost = max(history) if history_count > 0 else None

    current_balance = agent_state.get("current_balance")
    if isinstance(current_balance, (int, float)):
        if current_balance < 0:
            balance_regime = "negative"
        elif current_balance > 0:
            balance_regime = "positive"
        else:
            balance_regime = "neutral"
    else:
        balance_regime = "unknown"

    visibility_lines = [
        f"- agent_reputation: {rep}",
        f"- competitor_reputations: {competitor_reps}",
        f"- estimated_service_cost_usd: {est_cost_usd}",
        f"- estimated_service_cost_micro: {est_cost_micro}",
    ]

    if include_balance:
        visibility_lines.extend([
            f"- current_balance_usd: {current_balance}",
            f"- balance_regime: {balance_regime}",
        ])
    else:
        visibility_lines.append("- current_balance_usd: unavailable in this architecture (do not invent)")

    if include_history:
        visibility_lines.extend([
            f"- past_execution_costs_count: {history_count}",
            f"- past_execution_costs_recent: {recent_history}",
            f"- past_execution_costs_min_usd: {min_cost}",
            f"- past_execution_costs_max_usd: {max_cost}",
            f"- past_execution_costs_avg_usd: {avg_cost}",
        ])
    else:
        visibility_lines.append("- past_execution_costs: unavailable in this architecture (do not invent)")

    visibility_block = "\n".join(visibility_lines)

    return f"""You are bidding agent {agent_id} in a decentralized AI marketplace.

Architecture mode: {architecture_mode}

Primary objective:
Maximize long-term utility, not just single-auction immediate margin.

Long-term utility should consider:
- Immediate PnL of a bid
- Competitiveness / chance to win this auction
- Reputation trajectory from winning and delivering services
- Execution-cost uncertainty and risk of underpricing
- Financial runway and sustainability

Auction type: reverse auction. Lower bids help, but reputation also matters.

Contract scoring logic (higher score wins):
- normalized_reputation = reputation * 100
- normalized_bid_component = ((max_price - bid_amount) * 10000) / max_price
- score = (reputation_weight * normalized_reputation + (100 - reputation_weight) * normalized_bid_component) / 100

Bid sensitivity:
- Score precision is 10000, so a 1-step change in bid component is roughly `max_price / 10000` (in micro-USDC).
- Changes smaller than that often do not affect score due to integer flooring.

Architecture-specific data policy:
{architecture_data_policy}

Your current runtime state:
{visibility_block}

Available auctions:
{auctions_context}

Strategic policy:
1. Evaluate trade-offs explicitly for each serious auction candidate:
   - expected short-term profit/loss
   - competitiveness against current winner
   - long-term reputation/value effect
2. Generate at least two candidate bids for important auctions when possible:
   - one conservative (profit-preserving)
   - one aggressive (win-seeking)
3. You MAY choose a controlled short-term loss if and only if it materially improves winning odds and serves long-term recovery.
4. Do not place loss-making bids that are still not competitive.
5. Avoid repetitive high-loss behavior across cycles; adapt using available evidence.

Tool usage policy (strict):
1. Use state values directly, do not invent replacements.
2. For `agent_reputation` or `your_reputation`, use {rep}.
3. For `estimated_cost`, use {est_cost_micro}.
4. Always pass exact auction `max_price` and `reputation_weight` from the selected auction.
5. Monetary arguments must be integer micro-USDC (6 decimals).
6. Never call unknown tools or malformed tool names.
   Tool names must be exactly one of: validate_bid_profitability, calculate_bid_score, simulate_bid_outcome, place_bid.

Recommended reasoning sequence per auction:
1. Propose candidate bids (conservative/aggressive).
2. Run `validate_bid_profitability` for each candidate.
3. Run `simulate_bid_outcome` for each candidate.
4. Use `calculate_bid_score` where needed for deeper comparisons.
5. Choose the bid with best long-term utility; place at most one bid per auction in this cycle.

Stop conditions:
- Execute max {max_reasoning_steps} analysis steps, then decide.
- If already current winner with a competitive bid, skip rebidding.
- If no strategically sound auction is found, return skip.
"""


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
    return _build_shared_strategic_core_prompt(
        agent_state=agent_state,
        architecture_mode="A1 strategic reasoning (isolated)",
        architecture_data_policy=(
            "This architecture is isolated (no coupling to execution history). "
            "You must not assume hidden historical cost or balance signals. "
            "Reason strategically using current auction state, current cost estimate, and available tools only."
        ),
        include_history=False,
        include_balance=False,
        max_reasoning_steps=6,
    )


def _prompt_arch_2(agent_state: Dict[str, Any]) -> str:
    """
    Architecture 2: Strategic ReAct with performance history and balance context.

    Differences from A1:
    - Uses historical execution costs
    - Uses current net balance
    - Optimizes long-term utility, not only immediate per-auction profit
    """
    return _build_shared_strategic_core_prompt(
        agent_state=agent_state,
        architecture_mode="A2 strategic reasoning (one-way coupled)",
        architecture_data_policy=(
            "This architecture has one-way coupling from service execution to bidding. "
            "You can use real past execution costs and current balance to calibrate risk and recovery strategy."
        ),
        include_history=True,
        include_balance=True,
        max_reasoning_steps=6,
    )


def _prompt_arch_3(agent_state: Dict[str, Any]) -> str:
    """Architecture 3: LLM with market history + guidance (to be implemented)."""
    raise NotImplementedError("Architecture 3 prompt not yet implemented")

"""Prompt templates for agent architectures."""


def get_prompt(architecture: str, agent_id: int, auctions_context: str, **kwargs) -> str:
    """
    Generate prompt for specified architecture.
    
    Args:
        architecture: Architecture identifier ("1", "2", "3", etc.)
        agent_id: Agent ID
        auctions_context: JSON string of eligible auctions
        **kwargs: Additional context (past_execution_costs, past_winning_bids, etc.)
    
    Returns:
        Formatted prompt string
    """
    if architecture == "1":
        return _prompt_arch_1(agent_id, auctions_context)
    elif architecture == "2":
        return _prompt_arch_2(agent_id, auctions_context, **kwargs)
    elif architecture == "3":
        return _prompt_arch_3(agent_id, auctions_context, **kwargs)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


def _prompt_arch_1(agent_id: int, auctions_context: str) -> str:
    """
    Architecture 1: LLM Minimal
    - State: Current only (no history)
    - Reasoning: LLM ReAct
    - Coupling: Isolated
    - Prompt: Identity + boundaries only
    """
    return f"""You are bidding agent {agent_id} in a decentralized AI marketplace.

Your goal: Maximize profit by bidding on profitable auctions.

Available auctions:
{auctions_context}

Use your tools to analyze auctions and place bids where profitable."""


def _prompt_arch_2(agent_id: int, auctions_context: str, **kwargs) -> str:
    """Architecture 2: LLM with performance history (to be implemented)."""
    raise NotImplementedError("Architecture 2 prompt not yet implemented")


def _prompt_arch_3(agent_id: int, auctions_context: str, **kwargs) -> str:
    """Architecture 3: LLM with market history + guidance (to be implemented)."""
    raise NotImplementedError("Architecture 3 prompt not yet implemented")

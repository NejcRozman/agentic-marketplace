"""
Cost tracking infrastructure for agent economic simulation.

Provides:
- CostTracker: Simple in-memory accounting for LLM costs, gas fees, and revenue
- LLMCostCallback: LangChain callback handler to automatically track LLM token usage
"""

import logging
from langchain_core.callbacks import BaseCallbackHandler

logger = logging.getLogger(__name__)


class LLMCostCallback(BaseCallbackHandler):
    """
    Callback handler to track LLM token usage and costs.
    
    Integrates with CostTracker to record actual LLM expenses.
    """
    
    def __init__(self, cost_tracker: "CostTracker", model: str, config):
        super().__init__()
        self.cost_tracker = cost_tracker
        self.model = model
        self.config = config
        self.total_tokens = 0
        self.total_cost = 0.0
    
    def on_llm_end(self, response, **kwargs):
        """Called when LLM finishes - calculate cost from token usage."""
        try:
            # Extract token usage from response
            if hasattr(response, "llm_output") and response.llm_output:
                token_usage = response.llm_output.get("token_usage", {})
                input_tokens = token_usage.get("prompt_tokens", 0)
                output_tokens = token_usage.get("completion_tokens", 0)
                
                # Use config-based pricing
                input_cost = (input_tokens / 1000) * self.config.llm_input_price_per_1k
                output_cost = (output_tokens / 1000) * self.config.llm_output_price_per_1k
                total_cost = input_cost + output_cost
                
                self.total_tokens += (input_tokens + output_tokens)
                self.total_cost += total_cost
                
                # Record in cost tracker
                self.cost_tracker.add_llm_cost(
                    cost_usd=total_cost,
                    context=f"{self.model} ({input_tokens}+{output_tokens} tokens)"
                )
                
        except Exception as e:
            logger.warning(f"Failed to track LLM cost: {e}")


class CostTracker:
    """
    Tracks all operational costs for an agent.
    
    Simple in-memory accounting:
    - LLM API calls (reasoning + service execution)
    - Blockchain gas fees
    - Revenue from completed services
    
    No persistence - tracks during agent lifetime only.
    """
    
    def __init__(self, agent_id: int, config):
        self.agent_id = agent_id
        self.config = config
        
        # Cost accumulators (in USD)
        self.total_llm_costs = 0.0
        self.total_gas_costs = 0.0
        self.total_revenue = 0.0
    
    def add_llm_cost(self, cost_usd: float, context: str = "llm_call"):
        """
        Record LLM API cost.
        
        Args:
            cost_usd: Cost in USD
            context: Description of what the LLM call was for
        """
        self.total_llm_costs += cost_usd
        logger.debug(f"💰 LLM cost: ${cost_usd:.4f} ({context})")
    
    def add_gas_cost(self, gas_used: int, gas_price_wei: int, context: str = "transaction"):
        """
        Record blockchain gas cost using config constants.
        
        Args:
            gas_used: Gas units consumed
            gas_price_wei: Gas price in wei (from transaction receipt)
            context: Description of transaction type
        """
        # Convert wei to gwei
        gas_price_gwei = gas_price_wei / 1e9
        cost_eth = (gas_used * gas_price_gwei) / 1e9
        cost_usd = cost_eth * self.config.eth_price_usd
        
        self.total_gas_costs += cost_usd
        logger.debug(f"⛽ Gas cost: ${cost_usd:.4f} ({context}, {gas_used} gas @ {gas_price_gwei:.2f} gwei)")
    
    def add_revenue(self, revenue_usd: float, context: str = "service_completion"):
        """
        Record revenue from winning an auction.
        
        Args:
            revenue_usd: Revenue in USD
            context: Description of revenue source
        """
        self.total_revenue += revenue_usd
        logger.info(f"💵 Revenue: ${revenue_usd:.2f} USD ({context})")
    
    def get_net_balance(self) -> float:
        """
        Get current net balance (revenue - costs).
        
        Returns:
            Net balance in USD
        """
        total_costs = self.total_llm_costs + self.total_gas_costs
        return self.total_revenue - total_costs
    
    def log_summary(self):
        """Log a summary of current financial state."""
        total_costs = self.total_llm_costs + self.total_gas_costs
        net_balance = self.get_net_balance()
        
        logger.info(f"""
╔══════════════════════════════════════════════════════════╗
║  Agent #{self.agent_id} - Financial Summary     
╠══════════════════════════════════════════════════════════╣
║  Revenue:      ${self.total_revenue:>10.2f} USD         
║  LLM Costs:    ${self.total_llm_costs:>10.2f} USD       
║  Gas Costs:    ${self.total_gas_costs:>10.2f} USD       
║  ────────────────────────────────────────────────        
║  Total Costs:  ${total_costs:>10.2f} USD                
║  NET BALANCE:  ${net_balance:>10.2f} USD                
╚══════════════════════════════════════════════════════════╝
        """)

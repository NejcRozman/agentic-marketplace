"""
Adversarial Provider Strategies

This module defines various adversarial behaviors for testing reputation system robustness.
Each strategy represents a different type of attack on the marketplace.
"""

from enum import Enum
from typing import Dict, Any
import logging


class AdversarialStrategy(Enum):
    """Types of adversarial behaviors for Byzantine robustness testing"""
    LOW_QUALITY = "low_quality"                      # E4.1: Always deliver poor service
    SELECTIVE_DEFECTION = "selective_defection"      # E4.2: Bait-and-switch (build rep, then degrade)
    NON_COMPLETION = "non_completion"                # E4.3: Win but don't complete service
    PRICE_MANIPULATION = "price_manipulation"        # E4.4: Underbid + low quality


class AdversarialBehaviorController:
    """
    Controls adversarial provider behavior based on selected strategy.
    
    This controller modifies provider behavior to simulate various attacks
    on the marketplace reputation system.
    """
    
    def __init__(self, strategy: AdversarialStrategy, config: Dict[str, Any]):
        """
        Initialize adversarial behavior controller.
        
        Args:
            strategy: The adversarial strategy to employ
            config: Configuration dictionary with strategy-specific parameters
        """
        self.strategy = strategy
        self.config = config
        self.auction_count = 0
        self.service_count = 0
        self.current_reputation = 50  # Track own reputation for adaptive strategies
        
        logging.info(f"[ADVERSARIAL] Initialized with strategy: {strategy.value}")
        
    def should_complete_service(self) -> bool:
        """
        Decide whether to complete service after winning auction.
        
        Returns:
            bool: False for NON_COMPLETION strategy, True otherwise
        """
        if self.strategy == AdversarialStrategy.NON_COMPLETION:
            logging.info("[ADVERSARIAL] Refusing to complete service (NON_COMPLETION strategy)")
            return False
        return True
        
    def generate_low_quality_response(self, prompt: str) -> str:
        """
        Generate intentionally low-quality response without using LLM.
        
        This bypasses actual service execution to save costs while collecting payment.
        Used for LOW_QUALITY and PRICE_MANIPULATION strategies.
        
        Args:
            prompt: The service request prompt (ignored)
            
        Returns:
            str: Hardcoded low-quality response
        """
        # Rotate through different types of bad responses
        bad_responses = [
            "I don't know.",
            "No information available.",
            "Unable to provide analysis.",
            "This topic is unclear.",
            "Insufficient data to answer.",
            "Request cannot be processed.",
            "Content not accessible.",
            "Analysis unavailable at this time.",
        ]
        
        response_index = self.service_count % len(bad_responses)
        low_quality_response = bad_responses[response_index]
        
        logging.info(f"[ADVERSARIAL] Generated low-quality response: '{low_quality_response}'")
        return low_quality_response
        
    def should_use_low_quality_response(self) -> bool:
        """
        Determine if this service should use low-quality response.
        
        Returns:
            bool: True if should skip LLM and use hardcoded response
        """
        if self.strategy == AdversarialStrategy.LOW_QUALITY:
            # Always deliver low quality
            return True
            
        elif self.strategy == AdversarialStrategy.PRICE_MANIPULATION:
            # Always deliver low quality
            return True
            
        elif self.strategy == AdversarialStrategy.SELECTIVE_DEFECTION:
            # Bait-and-switch: deliver high quality until reputation >= 70, then switch
            if self.current_reputation >= 70:
                logging.info(f"[ADVERSARIAL] SWITCH triggered (reputation={self.current_reputation})")
                return True
            else:
                logging.info(f"[ADVERSARIAL] BAIT phase (reputation={self.current_reputation})")
                return False
                
        return False
        
    def get_bidding_multiplier(self) -> float:
        """
        Adjust bidding base cost for adversarial strategy.
        
        Returns:
            float: Multiplier to apply to base bidding cost
        """
        if self.strategy == AdversarialStrategy.PRICE_MANIPULATION:
            # Underbid aggressively to win auctions
            logging.info("[ADVERSARIAL] Applying 0.5x bidding multiplier (PRICE_MANIPULATION)")
            return 0.5
            
        elif self.strategy == AdversarialStrategy.SELECTIVE_DEFECTION:
            # During bait phase, bid low to win frequently and build reputation
            if self.current_reputation < 70:
                logging.info("[ADVERSARIAL] BAIT phase: bidding at cost (0.8x multiplier)")
                return 0.8
            else:
                # After switch, bid high to maximize profit per service
                logging.info("[ADVERSARIAL] SWITCH phase: bidding high (2.0x multiplier)")
                return 2.0
                
        elif self.strategy == AdversarialStrategy.NON_COMPLETION:
            # Bid extremely low (near zero) to always win
            logging.info("[ADVERSARIAL] Applying 0.1x bidding multiplier (NON_COMPLETION)")
            return 0.1
            
        # LOW_QUALITY uses normal bidding (1.0x) to stay competitive
        return 1.0
        
    def on_auction_detected(self):
        """Track auction participation."""
        self.auction_count += 1
        logging.debug(f"[ADVERSARIAL] Auction detected (total: {self.auction_count})")
        
    def on_service_executed(self):
        """Track service execution."""
        self.service_count += 1
        logging.debug(f"[ADVERSARIAL] Service executed (total: {self.service_count})")
        
    def update_reputation(self, new_reputation: int):
        """
        Update tracked reputation for adaptive strategies.
        
        Args:
            new_reputation: Current reputation score (0-100)
        """
        old_reputation = self.current_reputation
        self.current_reputation = new_reputation
        logging.info(f"[ADVERSARIAL] Reputation updated: {old_reputation} → {new_reputation}")
        
    def get_stats(self) -> Dict[str, Any]:
        """
        Get current statistics for this adversarial agent.
        
        Returns:
            dict: Statistics including strategy, counts, reputation
        """
        return {
            "strategy": self.strategy.value,
            "auction_count": self.auction_count,
            "service_count": self.service_count,
            "current_reputation": self.current_reputation,
        }

"""Data classes for auction information."""

from typing import Dict, Any


class AuctionInfo:
    """Data class for auction information from ReverseAuction contract."""
    
    def __init__(self, auction_data: Dict[str, Any]):
        """
        Initialize from contract data.
        
        Args:
            auction_data: Dictionary with auction fields from contract
        """
        self.auction_id = auction_data.get('auctionId', 0)
        self.buyer = auction_data.get('buyer', '')
        self.budget = auction_data.get('budget', 0)
        self.deadline = auction_data.get('deadline', 0)
        self.service_description_cid = auction_data.get('serviceDescriptionCid', '')
        self.is_active = auction_data.get('isActive', False)
        self.selected_provider = auction_data.get('selectedProvider', '')
        self.final_price = auction_data.get('finalPrice', 0)
        self.service_completed = auction_data.get('serviceCompleted', False)
    
    def __repr__(self):
        return f"AuctionInfo(id={self.auction_id}, buyer={self.buyer[:10]}..., budget={self.budget}, active={self.is_active})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'auction_id': self.auction_id,
            'buyer': self.buyer,
            'budget': self.budget,
            'deadline': self.deadline,
            'service_description_cid': self.service_description_cid,
            'is_active': self.is_active,
            'selected_provider': self.selected_provider,
            'final_price': self.final_price,
            'service_completed': self.service_completed
        }

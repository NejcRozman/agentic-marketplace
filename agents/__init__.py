"""
Agentic Marketplace - AI Agents Package

This package contains AI agents for the agentic marketplace, built with LangGraph.
It includes service provider agents, consumer agents, and the core infrastructure
for interacting with the blockchain-based marketplace.
"""

__version__ = "0.1.0"

from .core.base_agent import BaseAgent
from .core.blockchain_client import BlockchainClient

__all__ = ["BaseAgent", "BlockchainClient"]
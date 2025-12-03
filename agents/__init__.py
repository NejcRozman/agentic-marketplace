"""
Agentic Marketplace - AI Agents Package

This package contains AI agents for the agentic marketplace, built with LangGraph.

Structure:
- infrastructure/: Shared blockchain utilities (client, ABIs, feedbackAuth)
- provider_agent/: Complete service provider agent (literature review + blockchain + orchestration)
- consumer_agent/: Complete service consumer agent (future)
- config.py: Shared configuration
"""

__version__ = "0.1.0"

from .config import config
from .infrastructure import (
    BlockchainClient,
    IPFSClient,
    IPFSUploadResult,
    ServiceDescription,
    create_ipfs_client,
    AuctionInfo,
    contract_abis,
    get_reverse_auction_abi,
    get_identity_registry_abi,
    get_reputation_registry_abi,
    generate_feedback_auth,
    parse_feedback_auth,
    verify_feedback_auth_format
)
from .provider_agent import (
    LiteratureReviewAgent,
    BlockchainHandler,
    BlockchainState
)

__all__ = [
    "config",
    "BlockchainClient",
    "IPFSClient",
    "IPFSUploadResult",
    "ServiceDescription",
    "create_ipfs_client",
    "AuctionInfo",
    "contract_abis",
    "get_reverse_auction_abi",
    "get_identity_registry_abi",
    "get_reputation_registry_abi",
    "generate_feedback_auth",
    "parse_feedback_auth",
    "verify_feedback_auth_format",
    "LiteratureReviewAgent",
    "BlockchainHandler",
    "BlockchainState",
    "__version__"
]
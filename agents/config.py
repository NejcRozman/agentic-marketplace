"""Configuration management for agentic marketplace agents."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Get the path to the agents directory where .env file is located
AGENTS_DIR = Path(__file__).parent
ENV_FILE_PATH = AGENTS_DIR / ".env"

# Load environment variables from .env file (use as defaults, don't override existing)
# This allows experiment runner to override contract addresses for each run
load_dotenv(ENV_FILE_PATH, override=False)


class Config:
    """Simple configuration class for marketplace agents."""
    
    def __init__(self):
        """Load configuration from environment variables."""
        
        # LLM API (OpenRouter)
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        self.openrouter_base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        
        # Pinata IPFS
        self.pinata_api_key = os.getenv("PINATA_API_KEY")
        self.pinata_api_secret = os.getenv("PINATA_API_SECRET")
        self.pinata_jwt = os.getenv("PINATA_JWT")
        
        # Blockchain connection
        self.rpc_url = os.getenv("BLOCKCHAIN_RPC_URL", "http://localhost:8545")
        self.chain_id = int(os.getenv("BLOCKCHAIN_CHAIN_ID", "31337"))
        self.private_key = os.getenv("BLOCKCHAIN_PRIVATE_KEY")
        self.gas_limit = int(os.getenv("BLOCKCHAIN_GAS_LIMIT", "3000000"))
        
        # Contract addresses (set these after deployment)
        self.reverse_auction_address = os.getenv("BLOCKCHAIN_REVERSE_AUCTION_ADDRESS")
        self.identity_registry_address = os.getenv("BLOCKCHAIN_IDENTITY_REGISTRY_ADDRESS")
        self.reputation_registry_address = os.getenv("BLOCKCHAIN_REPUTATION_REGISTRY_ADDRESS")
        self.payment_token_address = os.getenv("BLOCKCHAIN_PAYMENT_TOKEN_ADDRESS")  # USDC or mock USDC
        
        # Provider Agent ID
        self.agent_id = int(os.getenv("BLOCKCHAIN_AGENT_ID", "4427"))
        
        # Consumer Agent Configuration
        self.consumer_agent_id = int(os.getenv("CONSUMER_AGENT_ID", "0"))  # Set to 0 if not a consumer
        eligible_str = os.getenv("ELIGIBLE_PROVIDERS", "")
        self.eligible_providers = [int(id.strip()) for id in eligible_str.split(",") if id.strip()]
        self.consumer_check_interval = int(os.getenv("CONSUMER_CHECK_INTERVAL", "5"))
        
        # Random provider selection configuration
        pool_str = os.getenv("PROVIDER_POOL", "")
        self.provider_pool = [int(id.strip()) for id in pool_str.split(",") if id.strip()]
        eligible_per_auction_str = os.getenv("ELIGIBLE_PER_AUCTION", "")
        self.eligible_per_auction = int(eligible_per_auction_str) if eligible_per_auction_str else None
        
        # Consumer Auto-Auction Configuration
        self.auto_create_auction = os.getenv("AUTO_CREATE_AUCTION", "false").lower() == "true"
        self.num_auctions = int(os.getenv("NUM_AUCTIONS", "1"))
        self.auction_creation_delay = int(os.getenv("AUCTION_CREATION_DELAY", "0"))  # Delay before first auction
        self.inter_auction_delay = int(os.getenv("INTER_AUCTION_DELAY", "30"))  # Delay between auctions
        self.pdf_directory = os.getenv("PDF_DIRECTORY", "")
        self.service_complexity = os.getenv("SERVICE_COMPLEXITY", "medium")
        self.max_budget = int(os.getenv("MAX_BUDGET", "100000000"))  # 100 USDC with 6 decimals
        self.auction_duration = int(os.getenv("AUCTION_DURATION", "1800"))  # 30 minutes
        self.reputation_weight = int(os.getenv("REPUTATION_WEIGHT", "30"))  # 0-100, weight of reputation in bid scoring
        
        # Agent workspace
        self.workspace_dir = os.getenv("WORKSPACE_DIR", "./workspaces")
        
        # Provider quality profile configuration
        self.quality_profile = os.getenv("QUALITY_PROFILE", "medium")  # high, medium, or low
        
        # RAG parameters (will be set based on quality_profile)
        self.rag_temperature = None
        self.rag_retrieval_k = None
        self.rag_chunk_size = None
        self.rag_chunk_overlap = None
        self.rag_system_prompt_type = None  # "detailed", "standard", or "minimal"
        
        # Provider bidding configuration
        # Check if BIDDING_BASE_COST was explicitly set, otherwise use quality profile default
        self.bidding_base_cost_override = os.getenv("BIDDING_BASE_COST")
        self.bidding_base_cost = None  # Will be set by _set_quality_profile_params()
        
        # Set quality profile parameters
        self._set_quality_profile_params()
        
        # Apply override if provided
        if self.bidding_base_cost_override is not None:
            self.bidding_base_cost = int(self.bidding_base_cost_override)
        
        # Environment
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.debug = os.getenv("DEBUG", "true").lower() == "true"
    
    def _set_quality_profile_params(self):
        """Set RAG and bidding parameters based on quality profile.
        
        Quality profiles are justified by literature:
        - High: Deterministic (temp=0), more context (k=5), better chunking → higher quality, higher cost
        - Medium: Balanced parameters → standard quality, moderate cost
        - Low: More random (temp=0.7), less context (k=1), poor chunking → lower quality, lower cost
        """
        profile = self.quality_profile.lower()
        
        if profile == "high":
            # High quality: Precise, comprehensive, well-structured
            self.rag_temperature = 0.0
            self.rag_retrieval_k = 5
            self.rag_chunk_size = 8000
            self.rag_chunk_overlap = 400
            self.rag_system_prompt_type = "detailed"
            self.bidding_base_cost = 60  # Higher operational costs
            
        elif profile == "medium":
            # Medium quality: Standard configuration
            self.rag_temperature = 0.3
            self.rag_retrieval_k = 3
            self.rag_chunk_size = 10000
            self.rag_chunk_overlap = 200
            self.rag_system_prompt_type = "standard"
            self.bidding_base_cost = 40  # Standard operational costs
            
        elif profile == "low":
            # Low quality: Minimal effort, less precise
            self.rag_temperature = 0.7
            self.rag_retrieval_k = 1
            self.rag_chunk_size = 15000
            self.rag_chunk_overlap = 0
            self.rag_system_prompt_type = "minimal"
            self.bidding_base_cost = 20  # Lower operational costs
            
        else:
            raise ValueError(f"Unknown quality profile: {profile}. Must be 'high', 'medium', or 'low'")
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() == "production"
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment.lower() == "development"
    
    def validate(self) -> bool:
        """
        Validate that required configuration is present.
        
        Returns:
            True if valid, False otherwise
        """
        errors = []
        
        if not self.openrouter_api_key:
            errors.append("OPENROUTER_API_KEY is required")
        
        if not self.private_key:
            errors.append("BLOCKCHAIN_PRIVATE_KEY is required for transactions")
        
        if not self.reverse_auction_address:
            errors.append("BLOCKCHAIN_REVERSE_AUCTION_ADDRESS is required")
        
        if errors:
            print("⚠️  Configuration errors:")
            for error in errors:
                print(f"   - {error}")
            return False
        
        return True
    
    def display(self):
        """Display current configuration (hiding sensitive data)."""
        print("Configuration:")
        print(f"  Google API Key: {'✓ Set' if self.google_api_key else '✗ Not set'}")
        print(f"  Pinata JWT: {'✓ Set' if self.pinata_jwt else '✗ Not set'}")
        print(f"  Pinata API Key: {'✓ Set' if self.pinata_api_key else '✗ Not set'}")
        print(f"  RPC URL: {self.rpc_url}")
        print(f"  Chain ID: {self.chain_id}")
        print(f"  Private Key: {'✓ Set' if self.private_key else '✗ Not set'}")
        print(f"  Gas Limit: {self.gas_limit}")
        print(f"  ReverseAuction: {self.reverse_auction_address or '✗ Not set'}")
        print(f"  IdentityRegistry: {self.identity_registry_address or '✗ Not set'}")
        print(f"  ReputationRegistry: {self.reputation_registry_address or '✗ Not set'}")
        print(f"  Agent ID: {self.agent_id}")
        print(f"  Workspace: {self.workspace_dir}")
        print(f"  Environment: {self.environment}")
        print(f"  Debug: {self.debug}")


# Global configuration instance
config = Config()
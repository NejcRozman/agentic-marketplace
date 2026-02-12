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
        self.llm_model = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
        
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
        
        # RAG parameters (fixed configuration)
        self.rag_temperature = float(os.getenv("RAG_TEMPERATURE", "0.3"))
        self.rag_retrieval_k = int(os.getenv("RAG_RETRIEVAL_K", "3"))
        self.rag_chunk_size = int(os.getenv("RAG_CHUNK_SIZE", "10000"))
        self.rag_chunk_overlap = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))
        
        # Provider bidding configuration
        self.bidding_base_cost = int(os.getenv("BIDDING_BASE_COST", "40"))
        
        # Cost tracking constants (for economic simulation)
        self.gas_price_gwei = float(os.getenv("GAS_PRICE_GWEI", "20.0"))
        self.eth_price_usd = float(os.getenv("ETH_PRICE_USD", "3000.0"))
        self.llm_input_price_per_1k = float(os.getenv("LLM_INPUT_PRICE_PER_1K", "0.0002"))
        self.llm_output_price_per_1k = float(os.getenv("LLM_OUTPUT_PRICE_PER_1K", "0.0002"))
        self.service_cost_multiplier = float(os.getenv("SERVICE_COST_MULTIPLIER", "100.0"))
        
        # Coupling mode configuration
        self.coupling_mode = os.getenv("COUPLING_MODE", "isolated")  # "isolated", "one_way", "two_way"
        
        # Effort tier configuration (model selection based on quality/cost tradeoff)
        # Maps effort tier name to LLM model identifier
        self.effort_tiers = {
            "minimal": "meta-llama/llama-3.2-3b-instruct",      # ~$0.06/M tokens
            "low": "meta-llama/llama-3.1-8b-instruct",          # ~$0.055/M tokens
            "standard": "openai/gpt-4o-mini",                   # ~$0.15/M tokens (default)
            "high": "anthropic/claude-3.5-sonnet",              # ~$3/M tokens
            "premium": "openai/o1"                              # ~$15/M tokens (reasoning model)
        }
        self.default_effort_tier = "standard"
        
        # Environment
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.debug = os.getenv("DEBUG", "true").lower() == "true"
    
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
"""Configuration management for agentic marketplace agents."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Get the path to the agents directory where .env file is located
AGENTS_DIR = Path(__file__).parent.parent
ENV_FILE_PATH = AGENTS_DIR / ".env"

# Load environment variables from .env file
load_dotenv(ENV_FILE_PATH)


class Config:
    """Simple configuration class for marketplace agents."""
    
    def __init__(self):
        """Load configuration from environment variables."""
        
        # Google Gemini API
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        
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
        
        # Agent workspace
        self.workspace_dir = os.getenv("WORKSPACE_DIR", "./workspaces")
        
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
        
        if not self.google_api_key:
            errors.append("GOOGLE_API_KEY is required")
        
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
        print(f"  Workspace: {self.workspace_dir}")
        print(f"  Environment: {self.environment}")
        print(f"  Debug: {self.debug}")


# Global configuration instance
config = Config()
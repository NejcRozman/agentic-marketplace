"""Configuration management for agentic marketplace agents."""

import os
from typing import Optional
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Get the path to the agents directory where .env file is located
AGENTS_DIR = Path(__file__).parent.parent
ENV_FILE_PATH = AGENTS_DIR / ".env"


class BlockchainConfig(BaseSettings):
    """Blockchain connection configuration."""
    
    model_config = SettingsConfigDict(env_prefix="BLOCKCHAIN_")
    
    rpc_url: str = Field(default="http://localhost:8545", description="RPC endpoint URL")
    chain_id: int = Field(default=31337, description="Chain ID (31337 for local Anvil)")
    private_key: Optional[str] = Field(default=None, description="Private key for transactions")
    gas_limit: int = Field(default=3000000, description="Default gas limit")


class MarketplaceConfig(BaseSettings):
    """Main configuration class for marketplace agents."""
    
    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE_PATH),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Google Gemini API
    google_api_key: Optional[str] = Field(default=None, description="Google API key for Gemini")
    
    # Blockchain configuration
    blockchain: BlockchainConfig = Field(default_factory=BlockchainConfig)
    
    # Agent workspace
    workspace_dir: str = Field(default="./workspaces", description="Directory for agent workspaces")
    
    # Environment
    environment: str = Field(default="development", description="Environment (development, production)")
    debug: bool = Field(default=True, description="Debug mode")
    
    @classmethod
    def from_env(cls) -> "MarketplaceConfig":
        """Create configuration from environment variables."""
        return cls()
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() == "production"
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment.lower() == "development"


# Global configuration instance
config = MarketplaceConfig.from_env()
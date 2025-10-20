"""Configuration management for agentic marketplace agents."""

import os
from typing import Optional, Dict, Any
from pydantic import BaseSettings, Field
from pydantic_settings import SettingsConfigDict


class BlockchainConfig(BaseSettings):
    """Blockchain connection configuration."""
    
    model_config = SettingsConfigDict(env_prefix="BLOCKCHAIN_")
    
    rpc_url: str = Field(default="http://localhost:8545", description="RPC endpoint URL")
    chain_id: int = Field(default=31337, description="Chain ID")
    private_key: Optional[str] = Field(default=None, description="Private key for transactions")
    contract_addresses: Dict[str, str] = Field(default_factory=dict, description="Contract addresses")
    gas_limit: int = Field(default=3000000, description="Default gas limit")
    gas_price: Optional[int] = Field(default=None, description="Gas price in wei")


class LLMConfig(BaseSettings):
    """Large Language Model configuration."""
    
    model_config = SettingsConfigDict(env_prefix="LLM_")
    
    provider: str = Field(default="openai", description="LLM provider (openai, anthropic, etc.)")
    model_name: str = Field(default="gpt-4-turbo-preview", description="Model name")
    api_key: Optional[str] = Field(default=None, description="API key")
    temperature: float = Field(default=0.7, description="Temperature for sampling")
    max_tokens: int = Field(default=4096, description="Maximum tokens in response")
    timeout: int = Field(default=60, description="Request timeout in seconds")


class AgentConfig(BaseSettings):
    """Agent-specific configuration."""
    
    model_config = SettingsConfigDict(env_prefix="AGENT_")
    
    name: str = Field(default="marketplace-agent", description="Agent name")
    max_concurrent_tasks: int = Field(default=10, description="Maximum concurrent tasks")
    task_timeout: int = Field(default=300, description="Task timeout in seconds")
    retry_attempts: int = Field(default=3, description="Number of retry attempts")
    log_level: str = Field(default="INFO", description="Logging level")


class MarketplaceConfig(BaseSettings):
    """Main configuration class combining all settings."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Environment
    environment: str = Field(default="development", description="Environment (development, staging, production)")
    debug: bool = Field(default=True, description="Debug mode")
    
    # Sub-configurations
    blockchain: BlockchainConfig = Field(default_factory=BlockchainConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    
    # Database (optional)
    database_url: Optional[str] = Field(default=None, description="Database connection URL")
    
    # API configuration
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")
    
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
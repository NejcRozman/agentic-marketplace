"""Basic test to verify the agent setup."""

import pytest
import asyncio
from unittest.mock import Mock, patch

from agents.core.config import MarketplaceConfig
from agents.service_providers.data_analysis_agent import DataAnalysisAgent


class TestAgentSetup:
    """Test basic agent functionality."""
    
    def test_config_creation(self):
        """Test configuration creation."""
        config = MarketplaceConfig()
        assert config.environment == "development"
        assert config.debug is True
        assert config.blockchain.chain_id == 31337
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self):
        """Test agent initialization."""
        # Mock the blockchain client to avoid actual connections
        with patch('agents.core.base_agent.BlockchainClient') as mock_bc:
            mock_bc.return_value.is_connected = Mock(return_value=True)
            
            agent = DataAnalysisAgent()
            assert agent.agent_id == "data-analysis-agent"
            assert agent.service_type == "data_analysis"
            assert "statistical_analysis" in agent.capabilities
    
    @pytest.mark.asyncio
    async def test_agent_status(self):
        """Test agent status retrieval."""
        with patch('agents.core.base_agent.BlockchainClient') as mock_bc:
            mock_bc.return_value.is_connected = Mock(return_value=True)
            
            agent = DataAnalysisAgent()
            status = await agent.get_status()
            
            assert "agent_id" in status
            assert "state" in status
            assert status["agent_id"] == "data-analysis-agent"


if __name__ == "__main__":
    pytest.main([__file__])
# Agentic Marketplace - AI Agents

This package contains AI agents built with LangGraph for the decentralized agentic marketplace. The agents can function as both service providers and consumers, interacting with smart contracts on the blockchain.

## Features

- **LangGraph Integration**: Complex multi-step workflows using state graphs
- **Blockchain Interaction**: Direct integration with Ethereum-compatible networks
- **Modular Architecture**: Easy to extend with new agent types
- **Async Support**: High-performance asynchronous operations
- **Type Safety**: Full type hints and validation with Pydantic
- **Testing**: Comprehensive test suite with pytest

## Quick Start

### 1. Setup Environment

```bash
# Navigate to the project root
cd /home/nejc/Projects/agentic-marketplace

# Activate the project virtual environment
source .venv/bin/activate  # Linux/macOS
# or
.\.venv\Scripts\activate  # Windows

# Navigate to agents for development
cd agents/
```

### 2. Configure Environment

Copy and edit the environment file:

```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

Required environment variables:
- `LLM_API_KEY`: OpenAI or Anthropic API key
- `BLOCKCHAIN_RPC_URL`: Ethereum RPC endpoint
- `BLOCKCHAIN_PRIVATE_KEY`: Private key for transactions (optional)

### 3. Install Dependencies

```bash
pip install -r requirements.txt

# For development
pip install -r requirements-dev.txt
```

### 4. Run Your First Agent

```python
import asyncio
from agents.service_providers.data_analysis_agent import DataAnalysisAgent

async def main():
    agent = DataAnalysisAgent()
    
    result = await agent.process_request(
        "I want to register as a data analysis service provider",
        task_id="test-001"
    )
    
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
```

## Architecture

### Core Components

1. **BaseAgent**: Abstract base class for all agents using LangGraph
2. **BlockchainClient**: Handles all blockchain interactions
3. **Configuration**: Centralized configuration management

### Agent Types

#### Service Providers
- `DataAnalysisAgent`: Provides data analysis services
- `TradingAgent`: Automated trading services
- `ContentCreatorAgent`: Content generation services
- `ResearchAgent`: Research and information gathering

#### Consumers
- `ConsumerAgent`: Purchases services from providers
- `ServiceBuyer`: Automated service procurement

### LangGraph Workflow

Each agent follows a standard workflow:

```
Input Processing → Analysis → Blockchain Interaction → Response Generation
       ↓               ↓              ↓                      ↓
   Validate &      Use LLM to     Smart Contract        Generate final
   Prepare Data    Understand     Transactions          Response
                   Request
```

## Development

### Creating a New Agent

1. Inherit from `BaseAgent`:

```python
from agents.core.base_agent import BaseAgent, GraphState

class MyAgent(BaseAgent):
    def __init__(self, agent_id: str = "my-agent"):
        super().__init__(agent_id)
    
    async def _custom_input_processing(self, state: GraphState) -> None:
        # Custom input processing logic
        pass
    
    # Implement other abstract methods...
```

2. Implement required methods:
   - `_custom_input_processing()`
   - `_get_analysis_prompt()`
   - `_parse_analysis()`
   - `_perform_blockchain_action()`
   - `_get_response_prompt()`

### Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_base_agent.py

# Run with coverage
pytest --cov=agents tests/
```

### Code Quality

We use several tools to maintain code quality:

```bash
# Format code
black agents/
isort agents/

# Lint code
flake8 agents/

# Type checking
mypy agents/

# All quality checks
pre-commit run --all-files
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ENVIRONMENT` | Environment (dev/staging/prod) | `development` |
| `DEBUG` | Enable debug mode | `true` |
| `LLM_PROVIDER` | LLM provider (openai/anthropic) | `openai` |
| `LLM_MODEL_NAME` | Model name | `gpt-4-turbo-preview` |
| `LLM_API_KEY` | LLM API key | Required |
| `BLOCKCHAIN_RPC_URL` | Blockchain RPC endpoint | `http://localhost:8545` |
| `BLOCKCHAIN_CHAIN_ID` | Chain ID | `31337` |
| `BLOCKCHAIN_PRIVATE_KEY` | Private key for transactions | Optional |

### Blockchain Networks

Supported networks:
- **Local Development**: Hardhat/Anvil (Chain ID: 31337)
- **Ethereum Mainnet**: Chain ID 1
- **Ethereum Sepolia**: Chain ID 11155111
- **Polygon**: Chain ID 137
- **Arbitrum**: Chain ID 42161
- **Optimism**: Chain ID 10
- **Base**: Chain ID 8453

## Smart Contract Integration

### Contract Loading

```python
from agents.core.blockchain_client import BlockchainClient

client = BlockchainClient()

# Load marketplace contract
await client.load_contract(
    name="marketplace",
    address="0x1234567890123456789012345678901234567890",
    abi=marketplace_abi  # Contract ABI
)
```

### Contract Interaction

```python
# Read-only call
result = await client.call_contract_method(
    "marketplace",
    "getServiceProvider",
    provider_address
)

# Send transaction
tx_hash = await client.send_transaction(
    "marketplace",
    "registerService",
    service_type,
    price,
    metadata
)
```

## Examples

### Service Provider Registration

```python
agent = DataAnalysisAgent()

result = await agent.process_request(
    "Register my data analysis service with pricing of 100 tokens per request",
    metadata={"capabilities": ["statistical_analysis", "visualization"]}
)
```

### Service Request Processing

```python
agent = DataAnalysisAgent()

result = await agent.process_request(
    "Analyze this sales data and provide insights on trends and seasonality",
    task_id="analysis-001",
    metadata={"data_url": "https://example.com/data.csv"}
)
```

### Service Completion

```python
result = await agent.process_request(
    "Complete the analysis task with findings: Revenue increased 15% YoY",
    task_id="analysis-001"
)
```

## Monitoring and Logging

### Structured Logging

```python
import structlog

logger = structlog.get_logger(__name__)

logger.info(
    "Agent processing request",
    agent_id=self.agent_id,
    task_id=task_id,
    request_type="data_analysis"
)
```

### Performance Monitoring

```python
# Agent status
status = await agent.get_status()
print(f"Agent {status['agent_id']} is {status['state']}")
```

## Security Considerations

1. **Private Keys**: Never commit private keys to version control
2. **API Keys**: Use environment variables for all API keys
3. **Input Validation**: All inputs are validated using Pydantic
4. **Rate Limiting**: Implement rate limiting for API calls
5. **Error Handling**: Comprehensive error handling and logging

## Troubleshooting

### Common Issues

1. **Connection Issues**:
   ```bash
   # Check blockchain connection
   python -c "from agents.core.blockchain_client import BlockchainClient; import asyncio; asyncio.run(BlockchainClient().is_connected())"
   ```

2. **LLM API Issues**:
   ```bash
   # Test LLM connection
   python -c "from langchain_openai import ChatOpenAI; ChatOpenAI().invoke('Hello')"
   ```

3. **Import Errors**:
   ```bash
   # Check Python path
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-agent`
3. Make your changes and add tests
4. Ensure all tests pass: `pytest`
5. Run code quality checks: `pre-commit run --all-files`
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions and support:
- Create an issue on GitHub
- Check the documentation in the `docs/` folder
- Review existing examples in the `examples/` folder
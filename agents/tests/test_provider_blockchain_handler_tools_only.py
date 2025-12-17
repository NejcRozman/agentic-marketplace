"""
Test individual blockchain tools without the agent
"""
import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.provider_agent.blockchain_handler import BlockchainHandler
from agents.infrastructure.blockchain_client import BlockchainClient
from agents.config import Config


async def test_tools():
    """Test each tool individually"""
    config = Config()
    
    # Initialize blockchain client with private key
    client = BlockchainClient()
    await client._initialize()  # Must initialize to load account
    
    handler = BlockchainHandler(config.agent_id, blockchain_client=client)
    await handler.initialize()  # Load contracts
    
    print("\n" + "="*60)
    print("Testing Blockchain Tools")
    print("="*60)
    
    # Get tools and build name->tool map
    tools = handler._tools
    tool_map = {tool.name: tool for tool in tools}
    
    print(f"\nFound {len(tools)} tools:")
    for tool in tools:
        print(f"  - {tool.name}")
    
    # Test 1: Get IPFS data
    print("\n1. Testing get_ipfs_data...")
    try:
        result = await tool_map["get_ipfs_data"].ainvoke({"cid": "Qmc2Xruh3gcKCxo8FiNXMox5jj3wQS12s5YQnxNkM542AS"})
        print(f"✅ Success!")
        print(f"   Title: {result.get('title')}")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Check reputation
    print("\n2. Testing get_reputation...")
    try:
        result = await tool_map["get_reputation"].ainvoke({"agent_id": config.agent_id})
        print(f"✅ Success: {result}")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Estimate cost
    print("\n3. Testing estimate_cost...")
    try:
        # Get IPFS data first
        ipfs_data = await tool_map["get_ipfs_data"].ainvoke({"cid": "Qmc2Xruh3gcKCxo8FiNXMox5jj3wQS12s5YQnxNkM542AS"})
        import json
        result = await tool_map["estimate_cost"].ainvoke({
            "service_requirements": json.dumps(ipfs_data)
        })
        print(f"✅ Success: {result}")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 4: Place bid
    print("\n4. Testing place_bid for auction 2...")
    try:
        result = await tool_map["place_bid"].ainvoke({
            "auction_id": 2,
            "bid_amount": 55,
            "agent_id": config.agent_id
        })
        print(f"✅ Success: {result}")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Cleanup - close the web3 connection
    try:
        if client and hasattr(client, 'w3') and client.w3 and hasattr(client.w3, 'provider'):
            if hasattr(client.w3.provider, '_provider'):
                await client.w3.provider._provider.disconnect()
    except:
        pass  # Ignore cleanup errors
    
    print("\n" + "="*60)
    print("Tool Testing Complete")
    print("="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(test_tools())

"""
Test individual blockchain tools without the agent
"""
import asyncio
import os
import sys
from pathlib import Path

# Add project root to path

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
    estimated_cost = None
    try:
        # Get IPFS data first to check complexity
        ipfs_data = await tool_map["get_ipfs_data"].ainvoke({"cid": "Qmc2Xruh3gcKCxo8FiNXMox5jj3wQS12s5YQnxNkM542AS"})
        complexity = ipfs_data.get("complexity", "medium")
        result = await tool_map["estimate_cost"].ainvoke({
            "complexity": complexity
        })
        print(f"✅ Success: {result}")
        # Extract cost for use in next test
        if isinstance(result, dict) and "estimated_cost" in result:
            estimated_cost = result["estimated_cost"]
        elif isinstance(result, str) and "estimated_cost" in result.lower():
            import re
            match = re.search(r'(\d+)\s*USDC', result)
            if match:
                estimated_cost = int(match.group(1)) * 1_000_000  # Convert to micro USDC
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 4: Validate bid profitability
    print("\n4. Testing validate_bid_profitability...")
    if estimated_cost:
        # Test 4a: Profitable bid (higher than cost)
        print("   4a. Testing profitable bid...")
        try:
            result = await tool_map["validate_bid_profitability"].ainvoke({
                "estimated_cost": estimated_cost,
                "proposed_bid": estimated_cost + 5_000_000  # 5 USDC profit
            })
            print(f"   ✅ Success: {result.get('verdict', result)}")
            assert result.get("is_profitable") == True, "Expected profitable bid to be marked as profitable"
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
        
        # Test 4b: Unprofitable bid (lower than cost)
        print("   4b. Testing unprofitable bid...")
        try:
            result = await tool_map["validate_bid_profitability"].ainvoke({
                "estimated_cost": estimated_cost,
                "proposed_bid": estimated_cost - 2_000_000  # 2 USDC loss
            })
            print(f"   ✅ Success: {result.get('verdict', result)}")
            assert result.get("is_profitable") == False, "Expected unprofitable bid to be marked as unprofitable"
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
        
        # Test 4c: Marginally profitable bid
        print("   4c. Testing marginally profitable bid...")
        try:
            result = await tool_map["validate_bid_profitability"].ainvoke({
                "estimated_cost": estimated_cost,
                "proposed_bid": estimated_cost + 500_000  # 0.5 USDC profit
            })
            print(f"   ✅ Success: {result.get('verdict', result)}")
            assert result.get("is_profitable") == True, "Expected marginally profitable bid to be marked as profitable"
            print(f"   Profit margin: {result.get('profit_margin_percent', 'N/A')}%")
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("   ⚠️  Skipped - no estimated cost available from previous test")
    
    # Test 5: Place bid
    print("\n5. Testing place_bid for auction 2...")
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

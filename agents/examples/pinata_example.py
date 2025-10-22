"""
Example usage of Pinata IPFS storage for service descriptions.

This script demonstrates how to:
1. Store service descriptions on IPFS via Pinata
2. Retrieve service descriptions from IPFS
3. Manage service metadata

Before running this example, make sure you have:
1. Set up your Pinata credentials in the .env file
2. Installed required dependencies: pip install httpx
"""

import asyncio
import logging
from pathlib import Path
import sys

# Add the agents directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from utils.pinata_storage import PinataStorage, PinataError
from utils.service_schemas import (
    ServiceDescription, 
    create_ai_analysis_service,
    create_data_processing_service
)
from core.config import MarketplaceConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_pinata_authentication():
    """Test Pinata authentication."""
    print("🔑 Testing Pinata authentication...")
    
    try:
        config = MarketplaceConfig.from_env()
        async with PinataStorage(config) as storage:
            is_authenticated = await storage.test_authentication()
            
            if is_authenticated:
                print("✅ Pinata authentication successful!")
                return True
            else:
                print("❌ Pinata authentication failed!")
                return False
                
    except PinataError as e:
        print(f"❌ Pinata error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


async def store_service_description_example():
    """Example of storing a service description on IPFS."""
    print("\n📦 Storing service description on IPFS...")
    
    try:
        # Create a sample service description
        service = create_ai_analysis_service()
        
        config = MarketplaceConfig.from_env()
        async with PinataStorage(config) as storage:
            # Store the service description
            cid = await storage.pin_json_to_ipfs(
                data=service.model_dump(mode='json'),
                metadata={
                    "name": f"Service: {service.title}",
                    "description": "AI Analysis Service Description",
                    "category": service.category.value,
                    "created_by": "marketplace_example"
                }
            )
            
            print(f"✅ Service description stored successfully!")
            print(f"📋 IPFS CID: {cid}")
            print(f"🔗 Gateway URL: https://gateway.pinata.cloud/ipfs/{cid}")
            
            return cid
            
    except Exception as e:
        print(f"❌ Error storing service description: {e}")
        return None


async def retrieve_service_description_example(cid: str):
    """Example of retrieving a service description from IPFS."""
    print(f"\n📥 Retrieving service description from IPFS...")
    print(f"📋 CID: {cid}")
    
    try:
        config = MarketplaceConfig.from_env()
        async with PinataStorage(config) as storage:
            # Retrieve the service description
            data = await storage.retrieve_json_from_ipfs(cid)
            
            # Parse back into ServiceDescription object
            service = ServiceDescription.model_validate(data)
            
            print(f"✅ Service description retrieved successfully!")
            print(f"📝 Title: {service.title}")
            print(f"📂 Category: {service.category.value}")
            print(f"� Deliverables: {', '.join(service.deliverables)}")
            print(f"⏱️  Max response time: {service.requirements.max_response_time}")
            print(f"📥 Input: {service.requirements.input_description}")
            print(f"📤 Output: {service.requirements.output_description}")
            
            return service
            
    except Exception as e:
        print(f"❌ Error retrieving service description: {e}")
        return None


async def list_pinned_files_example():
    """Example of listing pinned files on Pinata."""
    print(f"\n📋 Listing pinned files...")
    
    try:
        config = MarketplaceConfig.from_env()
        async with PinataStorage(config) as storage:
            files = await storage.list_pinned_files(limit=10)
            
            print(f"✅ Found {len(files)} pinned files:")
            
            for file_info in files:
                print(f"  📄 {file_info.ipfs_hash[:20]}... ({file_info.size} bytes)")
                if file_info.metadata:
                    name = file_info.metadata.get("name", "Unnamed")
                    print(f"      Name: {name}")
                print(f"      Timestamp: {file_info.timestamp}")
                
            return files
            
    except Exception as e:
        print(f"❌ Error listing files: {e}")
        return []


async def complete_workflow_example():
    """Complete workflow example: create, store, retrieve service description."""
    print("🚀 Running complete Pinata workflow example...")
    print("=" * 60)
    
    # Step 1: Test authentication
    auth_success = await test_pinata_authentication()
    if not auth_success:
        print("❌ Cannot proceed without valid authentication")
        return
    
    # Step 2: Store service description
    cid = await store_service_description_example()
    if not cid:
        print("❌ Cannot proceed without successful storage")
        return
    
    # Step 3: Retrieve service description
    service = await retrieve_service_description_example(cid)
    if not service:
        print("❌ Failed to retrieve stored service")
        return
    
    # Step 4: List pinned files
    await list_pinned_files_example()
    
    print("\n✅ Workflow completed successfully!")
    print(f"🎯 Your service description is available at:")
    print(f"   IPFS CID: {cid}")
    print(f"   Gateway: https://gateway.pinata.cloud/ipfs/{cid}")


async def marketplace_integration_example():
    """Example showing how this integrates with the marketplace."""
    print("\n🏪 Marketplace Integration Example...")
    print("=" * 50)
    
    try:
        # Create different types of services
        ai_service = create_ai_analysis_service()
        data_service = create_data_processing_service()
        
        config = MarketplaceConfig.from_env()
        async with PinataStorage(config) as storage:
            print("📦 Storing multiple service descriptions...")
            
            # Store AI service
            ai_cid = await storage.pin_json_to_ipfs(
                data=ai_service.model_dump(mode='json'),
                metadata={
                    "name": f"AI Service: {ai_service.title}",
                    "category": ai_service.category.value,
                    "marketplace_version": "1.0"
                }
            )
            
            # Store data service  
            data_cid = await storage.pin_json_to_ipfs(
                data=data_service.model_dump(mode='json'),
                metadata={
                    "name": f"Data Service: {data_service.title}",
                    "category": data_service.category.value,
                    "marketplace_version": "1.0"
                }
            )
            
            print(f"✅ AI Service CID: {ai_cid}")
            print(f"✅ Data Service CID: {data_cid}")
            
            # These CIDs would be used in the smart contract
            print(f"\n🔗 Smart Contract Integration:")
            print(f"   When creating auction, use serviceDescriptionCid: '{ai_cid}'")
            print(f"   Buyers can retrieve full service details from IPFS")
            print(f"   Providers can verify service requirements before bidding")
            
            return {"ai_service": ai_cid, "data_service": data_cid}
            
    except Exception as e:
        print(f"❌ Error in marketplace integration: {e}")
        return None


if __name__ == "__main__":
    print("🌐 Pinata IPFS Storage Example for Agentic Marketplace")
    print("=" * 60)
    
    try:
        # Run the complete workflow
        asyncio.run(complete_workflow_example())
        
        # Run marketplace integration example
        asyncio.run(marketplace_integration_example())
        
        print(f"\n🎉 All examples completed!")
        print(f"💡 Next steps:")
        print(f"   1. Integrate with smart contract deployment")
        print(f"   2. Create agent workflows that use IPFS storage")
        print(f"   3. Build marketplace frontend that displays service descriptions")
        
    except KeyboardInterrupt:
        print("\n⏹️  Example interrupted by user")
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        logger.exception("Detailed error information:")
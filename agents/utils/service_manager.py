#!/usr/bin/env python3
"""
Service Description Management CLI

A command-line utility for managing service descriptions on IPFS via Pinata.
This tool allows you to create, store, retrieve, and manage service descriptions
for the agentic marketplace.

Usage:
    python service_manager.py store --title "My AI Service" --category ai_ml
    python service_manager.py retrieve --cid QmXXXXXXX
    python service_manager.py list
    python service_manager.py validate --file service.json
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Optional

# Add the agents directory to the Python path for proper imports
current_dir = Path(__file__).parent
agents_dir = current_dir.parent
sys.path.insert(0, str(agents_dir))

from utils.pinata_storage import PinataStorage, PinataError
from utils.service_schemas import (
    ServiceDescription, 
    ServiceCategory,
    create_ai_analysis_service,
    create_data_processing_service,
    create_simple_service
)
from core.config import MarketplaceConfig

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class ServiceManager:
    """CLI tool for managing service descriptions."""
    
    def __init__(self):
        self.config = MarketplaceConfig.from_env()
    
    async def store_service(self, title: str, category: str, description: str = "", 
                          interactive: bool = False) -> Optional[str]:
        """Store a new service description on IPFS."""
        try:
            if interactive:
                service = await self._create_service_interactive()
            else:
                service = self._create_service_from_args(title, category, description)
            
            async with PinataStorage(self.config) as storage:
                cid = await storage.pin_json_to_ipfs(
                    data=service.model_dump(mode='json'),
                    metadata={
                        "name": f"Service: {service.title}",
                        "category": service.category.value,
                        "created_by": "service_manager_cli",
                        "version": "1.0"
                    }
                )
                
                print(f"✅ Service stored successfully!")
                print(f"📋 CID: {cid}")
                print(f"🔗 URL: https://gateway.pinata.cloud/ipfs/{cid}")
                
                return cid
                
        except Exception as e:
            print(f"❌ Error storing service: {e}")
            return None
    
    async def retrieve_service(self, cid: str) -> Optional[ServiceDescription]:
        """Retrieve a service description from IPFS."""
        try:
            async with PinataStorage(self.config) as storage:
                data = await storage.retrieve_json_from_ipfs(cid)
                service = ServiceDescription.model_validate(data)
                
                print(f"✅ Service retrieved successfully!")
                print(f"📝 Title: {service.title}")
                print(f"📂 Category: {service.category.value}")
                print(f"📄 Description: {service.description}")
                print(f"� Deliverables: {', '.join(service.deliverables)}")
                print(f"⏱️  Max Response Time: {service.requirements.max_response_time}")
                print(f"📥 Input: {service.requirements.input_description}")
                print(f"📤 Output: {service.requirements.output_description}")
                
                return service
                
        except Exception as e:
            print(f"❌ Error retrieving service: {e}")
            return None
    
    async def list_services(self, limit: int = 20) -> None:
        """List all stored services."""
        try:
            async with PinataStorage(self.config) as storage:
                files = await storage.list_pinned_files(limit=limit)
                
                print(f"📋 Found {len(files)} pinned files:")
                print("-" * 80)
                
                for i, file_info in enumerate(files, 1):
                    print(f"{i}. CID: {file_info.ipfs_hash}")
                    print(f"   Size: {file_info.size} bytes")
                    print(f"   Date: {file_info.timestamp}")
                    
                    if file_info.metadata:
                        name = file_info.metadata.get("name", "Unnamed")
                        category = file_info.metadata.get("category", "Unknown")
                        print(f"   Name: {name}")
                        print(f"   Category: {category}")
                    
                    print()
                    
        except Exception as e:
            print(f"❌ Error listing services: {e}")
    
    async def validate_service_file(self, file_path: str) -> bool:
        """Validate a service description JSON file."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            service = ServiceDescription.model_validate(data)
            print(f"✅ Service file is valid!")
            print(f"📝 Title: {service.title}")
            print(f"📂 Category: {service.category.value}")
            return True
            
        except FileNotFoundError:
            print(f"❌ File not found: {file_path}")
            return False
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON: {e}")
            return False
        except Exception as e:
            print(f"❌ Validation error: {e}")
            return False
    
    async def test_connection(self) -> bool:
        """Test connection to Pinata."""
        try:
            async with PinataStorage(self.config) as storage:
                is_authenticated = await storage.test_authentication()
                
                if is_authenticated:
                    print("✅ Pinata connection successful!")
                    return True
                else:
                    print("❌ Pinata authentication failed!")
                    return False
                    
        except PinataError as e:
            print(f"❌ Pinata error: {e}")
            return False
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False
    
    def _create_service_from_args(self, title: str, category: str, 
                                description: str) -> ServiceDescription:
        """Create a service from command line arguments."""
        try:
            category_enum = ServiceCategory(category.lower())
        except ValueError:
            category_enum = ServiceCategory.OTHER
        
        # Use predefined templates if available
        if category_enum == ServiceCategory.AI_ANALYSIS:
            service = create_ai_analysis_service()
            service.title = title
            if description:
                service.description = description
        elif category_enum == ServiceCategory.DATA_PROCESSING:
            service = create_data_processing_service()
            service.title = title
            if description:
                service.description = description
        else:
            # Create basic service
            service = create_simple_service(title, description or f"A {category} service", category_enum)
        
        return service
    
    async def _create_service_interactive(self) -> ServiceDescription:
        """Create a service interactively."""
        print("🔧 Creating service interactively...")
        
        # Get basic info
        title = input("Service title: ").strip()
        description = input("Service description: ").strip()
        
        # Get category
        print("\nAvailable categories:")
        for cat in ServiceCategory:
            print(f"  - {cat.value}")
        category_input = input("Category: ").strip().lower()
        try:
            category = ServiceCategory(category_input)
        except ValueError:
            category = ServiceCategory.OTHER
        
        return self._create_service_from_args(title, category.value, description)


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Service Description Management CLI for Agentic Marketplace"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Store command
    store_parser = subparsers.add_parser("store", help="Store a new service description")
    store_parser.add_argument("--title", required=True, help="Service title")
    store_parser.add_argument("--category", required=True, 
                            choices=[cat.value for cat in ServiceCategory],
                            help="Service category")
    store_parser.add_argument("--description", default="", help="Service description")
    store_parser.add_argument("--interactive", action="store_true",
                            help="Create service interactively")
    
    # Retrieve command
    retrieve_parser = subparsers.add_parser("retrieve", help="Retrieve a service description")
    retrieve_parser.add_argument("--cid", required=True, help="IPFS CID of the service")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List all stored services")
    list_parser.add_argument("--limit", type=int, default=20, help="Maximum number of results")
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate a service description file")
    validate_parser.add_argument("--file", required=True, help="Path to JSON file to validate")
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Test connection to Pinata")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    manager = ServiceManager()
    
    try:
        if args.command == "store":
            asyncio.run(manager.store_service(
                args.title, args.category, args.description, args.interactive
            ))
        elif args.command == "retrieve":
            asyncio.run(manager.retrieve_service(args.cid))
        elif args.command == "list":
            asyncio.run(manager.list_services(args.limit))
        elif args.command == "validate":
            asyncio.run(manager.validate_service_file(args.file))
        elif args.command == "test":
            asyncio.run(manager.test_connection())
    except KeyboardInterrupt:
        print("\n⏹️  Operation interrupted by user")
    except Exception as e:
        print(f"\n❌ Operation failed: {e}")
        logger.exception("Detailed error information:")


if __name__ == "__main__":
    main()
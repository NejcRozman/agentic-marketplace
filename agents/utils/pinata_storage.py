"""
Pinata IPFS Storage Utility for Agentic Marketplace

Provides utilities for storing and retrieving service descriptions
on IPFS using Pinata as the pinning service provider.
"""

import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import httpx
from pydantic import BaseModel
import sys

# Add path for config import
current_dir = Path(__file__).parent
agents_dir = current_dir.parent
sys.path.insert(0, str(agents_dir))

from core.config import MarketplaceConfig


class PinataError(Exception):
    """Custom exception for Pinata-related errors."""
    pass


class PinataFileInfo(BaseModel):
    """Information about a file stored on Pinata."""
    ipfs_hash: str
    size: int
    timestamp: str
    metadata: Dict[str, Any] = {}


class PinataStorage:
    """
    Utility class for interacting with Pinata IPFS service.
    
    Provides methods for pinning JSON data and files to IPFS,
    retrieving content, and managing pinned content.
    """
    
    BASE_URL = "https://api.pinata.cloud"
    GATEWAY_URL = "https://gateway.pinata.cloud/ipfs"
    
    def __init__(self, config: Optional[MarketplaceConfig] = None):
        """
        Initialize PinataStorage with configuration.
        
        Args:
            config: Marketplace configuration containing Pinata credentials
        """
        self.config = config or MarketplaceConfig.from_env()
        self.logger = logging.getLogger(__name__)
        
        # Pinata authentication
        self.api_key = self.config.pinata_api_key
        self.api_secret = self.config.pinata_api_secret
        self.jwt = self.config.pinata_jwt
        
        # Validate credentials
        if not (self.api_key and self.api_secret) and not self.jwt:
            raise PinataError(
                "Pinata credentials not found. Please set PINATA_API_KEY and "
                "PINATA_API_SECRET or PINATA_JWT in environment variables."
            )
        
        # Setup HTTP client
        self.client = httpx.AsyncClient(timeout=30.0)
        
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.client.aclose()
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for Pinata API."""
        if self.jwt:
            return {"Authorization": f"Bearer {self.jwt}"}
        else:
            return {
                "pinata_api_key": self.api_key,
                "pinata_secret_api_key": self.api_secret
            }
    
    async def test_authentication(self) -> bool:
        """
        Test Pinata authentication by making a simple API call.
        
        Returns:
            bool: True if authentication is successful
        """
        try:
            response = await self.client.get(
                f"{self.BASE_URL}/data/testAuthentication",
                headers=self._get_auth_headers()
            )
            return response.status_code == 200
        except Exception as e:
            self.logger.error(f"Authentication test failed: {e}")
            return False
    
    async def pin_json_to_ipfs(self, data: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Pin JSON data to IPFS via Pinata.
        
        Args:
            data: JSON data to pin
            metadata: Optional metadata for the pinned content
            
        Returns:
            IPFS hash (CID) of the pinned content
        """
        try:
            # Convert any Pydantic models to JSON-serializable dicts
            if hasattr(data, 'model_dump'):
                data = data.model_dump(mode='json')
            
            payload = {
                "pinataContent": data,
                "pinataMetadata": metadata or {}
            }
            
            response = await self.client.post(
                f"{self.BASE_URL}/pinning/pinJSONToIPFS",
                headers=self._get_auth_headers(),
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["IpfsHash"]
            else:
                raise PinataError(f"Failed to pin JSON: {response.status_code} - {response.text}")
                
        except Exception as e:
            self.logger.error(f"Error pinning JSON to IPFS: {e}")
            raise
    
    async def retrieve_json_from_ipfs(self, cid: str) -> Dict[str, Any]:
        """
        Retrieve JSON data from IPFS by CID.
        
        Args:
            cid: IPFS content identifier
            
        Returns:
            JSON data as dictionary
        """
        try:
            response = await self.client.get(f"{self.GATEWAY_URL}/{cid}")
            
            if response.status_code == 200:
                return response.json()
            else:
                raise PinataError(f"Failed to retrieve content: {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"Error retrieving JSON from IPFS: {e}")
            raise
    
    async def list_pinned_files(self, limit: int = 20) -> List[PinataFileInfo]:
        """
        List pinned files from Pinata.
        
        Args:
            limit: Maximum number of results
            
        Returns:
            List of pinned file information
        """
        try:
            params = {"pageLimit": limit}
            
            response = await self.client.get(
                f"{self.BASE_URL}/data/pinList",
                headers=self._get_auth_headers(),
                params=params
            )
            
            if response.status_code == 200:
                result = response.json()
                files = []
                
                for row in result.get("rows", []):
                    files.append(PinataFileInfo(
                        ipfs_hash=row["ipfs_pin_hash"],
                        size=row["size"],
                        timestamp=row["date_pinned"],
                        metadata=row.get("metadata", {})
                    ))
                
                return files
            else:
                raise PinataError(f"Failed to list files: {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"Error listing pinned files: {e}")
            return []
#!/usr/bin/env python3
"""
Create a test service description on IPFS via Pinata.

Usage:
    python scripts/create_service_description.py

Outputs the CID that can be used in CreateAuction.s.sol
"""

import asyncio
import os
import sys
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Any
from datetime import datetime, timezone

import aiohttp


@dataclass
class ServiceDescription:
    """Service description to be stored on IPFS."""
    title: str
    description: str
    prompts: list[str]  # Questions for the literature review agent
    input_files_cid: str | None = None  # CID of uploaded PDFs
    requirements: dict[str, Any] | None = None  # Optional technical requirements
    input_schema: dict[str, Any] | None = None  # Optional input schema
    output_schema: dict[str, Any] | None = None  # Optional output schema
    category: str = "research"
    tags: list[str] = field(default_factory=list)
    estimated_duration_seconds: int = 300
    version: str = "1.0.0"
    complexity: str = "medium"  # low, medium, high
    customer_endpoint: str | None = None  # A2A endpoint for result delivery
    
    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["created_at"] = datetime.now(timezone.utc).isoformat()
        # Remove None values for cleaner JSON
        return {k: v for k, v in data.items() if v is not None}


async def pin_json(data: dict, name: str, jwt: str = None, api_key: str = None, api_secret: str = None) -> str:
    """Pin JSON data to IPFS via Pinata."""
    url = "https://api.pinata.cloud/pinning/pinJSONToIPFS"
    
    if jwt:
        headers = {"Authorization": f"Bearer {jwt}"}
    elif api_key and api_secret:
        headers = {"pinata_api_key": api_key, "pinata_secret_api_key": api_secret}
    else:
        raise ValueError("Provide either jwt or api_key+api_secret")
    
    headers["Content-Type"] = "application/json"
    
    payload = {
        "pinataContent": data,
        "pinataMetadata": {"name": name},
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, headers=headers) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise Exception(f"Pinata error {resp.status}: {text}")
            result = await resp.json()
            return result["IpfsHash"]


async def fetch_json(cid: str) -> dict | None:
    """Fetch JSON from IPFS via public gateway."""
    gateways = [
        f"https://gateway.pinata.cloud/ipfs/{cid}",
        f"https://ipfs.io/ipfs/{cid}",
        f"https://cloudflare-ipfs.com/ipfs/{cid}",
    ]
    
    async with aiohttp.ClientSession() as session:
        for gateway in gateways:
            try:
                async with session.get(gateway, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        return await resp.json()
            except Exception:
                continue
    return None


async def main():
    # Check for Pinata credentials
    pinata_jwt = os.getenv("PINATA_JWT")
    pinata_api_key = os.getenv("PINATA_API_KEY")
    pinata_api_secret = os.getenv("PINATA_API_SECRET")
    
    if not pinata_jwt and not (pinata_api_key and pinata_api_secret):
        print("Error: Set PINATA_JWT or (PINATA_API_KEY + PINATA_API_SECRET) in environment")
        print("You can also add them to agents/.env")
        sys.exit(1)
    
    # Create a test service description
    # Note: You should upload PDFs first and set input_files_cid
    service_description = ServiceDescription(
        title="Literature Review on AI Agent Marketplaces",
        description="Comprehensive literature review analyzing the current state of AI agent marketplaces, "
                    "including decentralized approaches, reputation systems, and economic models.",
        prompts=[
            "What are the main approaches to building decentralized AI agent marketplaces?",
            "How do reputation systems work in agent marketplaces and what are their limitations?",
            "What economic models have been proposed for agent-to-agent transactions?",
            "What are the key technical challenges in implementing agent marketplaces?",
            "What research gaps exist in the current literature?",
        ],
        input_files_cid=None,  # Set this after uploading PDFs with upload_pdfs.py
        requirements={
            "min_papers": 10,
            "max_age_years": 3,
            "databases": ["arXiv", "Google Scholar", "Semantic Scholar"],
        },
        input_schema={
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "The research topic to review"},
                "prompts": {"type": "array", "items": {"type": "string"}, "description": "Questions to answer"},
            },
            "required": ["topic", "prompts"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "responses": {"type": "array", "items": {"type": "string"}, "description": "Answers to prompts"},
                "citations": {"type": "array", "items": {"type": "string"}, "description": "Sources cited"},
            },
        },
        category="research",
        tags=["literature-review", "ai-agents", "marketplace", "research"],
        estimated_duration_seconds=300,
        complexity="medium",
        version="1.0.0",
    )
    
    print("Creating service description on IPFS...")
    print(f"  Title: {service_description.title}")
    print(f"  Category: {service_description.category}")
    print()
    
    try:
        cid = await pin_json(
            data=service_description.to_dict(),
            name=f"service-description-{service_description.title[:30]}",
            jwt=pinata_jwt,
            api_key=pinata_api_key,
            api_secret=pinata_api_secret,
        )
        print("=" * 50)
        print(f"Service description created successfully!")
        print(f"CID: {cid}")
        print("=" * 50)
        print()
        print("Add this to contracts/.env:")
        print(f"  SERVICE_DESCRIPTION_CID={cid}")
        print()
        print("Or use directly in CreateAuction.s.sol")
        
        # Verify we can fetch it back
        print()
        print("Verifying CID is accessible...")
        fetched = await fetch_json(cid)
        if fetched and fetched.get("title") == service_description.title:
            print("✓ Successfully verified CID is accessible via IPFS gateway")
        else:
            print("⚠ Warning: Could not verify CID accessibility")
            
    except Exception as e:
        print(f"Error creating service description: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

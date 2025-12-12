#!/usr/bin/env python3
"""
Upload PDF files to IPFS via Pinata.

This script uploads multiple PDF files and returns a single CID that can be used
as input_files_cid in the service description.

Usage:
    python scripts/upload_pdfs.py path/to/pdfs/*.pdf
    python scripts/upload_pdfs.py path/to/pdfs/  # uploads all PDFs in directory

Outputs the CIDs for each file that can be downloaded later.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.infrastructure.ipfs_client import IPFSClient


async def upload_pdfs(pdf_paths: list[Path]) -> dict[str, str]:
    """
    Upload PDF files to IPFS.
    
    Args:
        pdf_paths: List of paths to PDF files
        
    Returns:
        Dict mapping filename to CID
    """
    client = IPFSClient()
    
    print(f"📤 Uploading {len(pdf_paths)} PDF files to IPFS...")
    print()
    
    results = {}
    
    for pdf_path in pdf_paths:
        if not pdf_path.exists():
            print(f"⚠️  File not found: {pdf_path}")
            continue
        
        if pdf_path.suffix.lower() != '.pdf':
            print(f"⚠️  Skipping non-PDF file: {pdf_path}")
            continue
        
        print(f"Uploading {pdf_path.name}...", end=" ")
        
        result = await client.pin_file(
            file_path=pdf_path,
            name=pdf_path.name,
            metadata={"type": "input_file", "format": "pdf"}
        )
        
        if result.success:
            print(f"✅ {result.cid}")
            results[pdf_path.name] = result.cid
        else:
            print(f"❌ {result.error}")
    
    return results


async def main():
    # Check for Pinata credentials
    pinata_jwt = os.getenv("PINATA_JWT")
    pinata_api_key = os.getenv("PINATA_API_KEY")
    pinata_api_secret = os.getenv("PINATA_API_SECRET")
    
    if not pinata_jwt and not (pinata_api_key and pinata_api_secret):
        print("Error: Set PINATA_JWT or (PINATA_API_KEY + PINATA_API_SECRET) in environment")
        print("You can also add them to agents/.env")
        sys.exit(1)
    
    if len(sys.argv) < 2:
        print("Usage: python scripts/upload_pdfs.py <pdf_files_or_directory>")
        print()
        print("Examples:")
        print("  python scripts/upload_pdfs.py data/pdfs/*.pdf")
        print("  python scripts/upload_pdfs.py data/pdfs/")
        sys.exit(1)
    
    # Collect PDF paths
    pdf_paths = []
    for arg in sys.argv[1:]:
        path = Path(arg)
        
        if path.is_file() and path.suffix.lower() == '.pdf':
            pdf_paths.append(path)
        elif path.is_dir():
            # Find all PDFs in directory
            pdf_paths.extend(path.glob("*.pdf"))
            pdf_paths.extend(path.glob("*.PDF"))
        else:
            print(f"⚠️  Invalid path: {arg}")
    
    if not pdf_paths:
        print("Error: No PDF files found")
        sys.exit(1)
    
    # Upload files
    results = await upload_pdfs(pdf_paths)
    
    if not results:
        print()
        print("❌ No files uploaded successfully")
        sys.exit(1)
    
    # Print summary
    print()
    print("=" * 70)
    print(f"✅ Successfully uploaded {len(results)} PDF files")
    print("=" * 70)
    print()
    print("File CIDs (save these for downloading later):")
    print()
    
    for filename, cid in results.items():
        print(f"  {filename}:")
        print(f"    CID: {cid}")
        print(f"    URL: https://gateway.pinata.cloud/ipfs/{cid}")
        print()
    
    # Create a mapping file
    mapping_file = Path("pdf_cids.json")
    import json
    with open(mapping_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"📝 CID mapping saved to: {mapping_file}")
    print()
    print("To use these in a service description:")
    print("  1. Note the CIDs you want to use as inputs")
    print("  2. Update create_service_description.py to set input_files_cid")
    print("  3. Or manually specify the CID list when creating the service")
    print()
    print("Note: For multiple files, you can store the CIDs in the service description")
    print("      as a JSON object or create a directory CID (requires Pinata SDK)")


if __name__ == "__main__":
    asyncio.run(main())

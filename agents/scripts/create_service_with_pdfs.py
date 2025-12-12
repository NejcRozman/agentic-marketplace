#!/usr/bin/env python3
"""
Complete workflow for creating a service description with PDFs.

This script:
1. Uploads PDF files to IPFS
2. Creates a service description with the PDF CIDs
3. Uploads the service description to IPFS
4. Outputs the CID for use in auction creation

Usage:
    python scripts/create_service_with_pdfs.py --pdfs data/pdfs/*.pdf
    python scripts/create_service_with_pdfs.py --pdf-dir data/pdfs/
"""

import asyncio
import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.infrastructure.ipfs_client import IPFSClient, ServiceDescription


async def upload_pdfs(client: IPFSClient, pdf_paths: list[Path]) -> dict[str, str]:
    """Upload PDF files and return filename -> CID mapping."""
    print(f"📤 Uploading {len(pdf_paths)} PDF files...")
    
    results = {}
    for pdf_path in pdf_paths:
        print(f"  {pdf_path.name}...", end=" ")
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


async def create_and_upload_service(
    client: IPFSClient,
    title: str,
    description: str,
    prompts: list[str],
    pdf_cids: dict[str, str],
    complexity: str = "medium"
) -> str:
    """Create service description and upload to IPFS."""
    print()
    print("📝 Creating service description...")
    
    # Use first PDF CID as single input_files_cid, or create dict for multiple
    if len(pdf_cids) == 1:
        input_files_cid = list(pdf_cids.values())[0]
        input_files = None
    else:
        input_files_cid = None
        input_files = pdf_cids
    
    # Create service description with the infrastructure's ServiceDescription
    service = ServiceDescription(
        title=title,
        description=description,
        prompts=prompts,
        input_files_cid=input_files_cid,
        complexity=complexity
    )
    
    # If we have multiple files, add them to requirements
    if input_files:
        service.requirements = {"input_files": input_files}
    
    # Upload to IPFS
    result = await client.pin_json(
        data=service.to_dict(),
        name=f"service-{title[:30]}",
        metadata={"type": "service_description", "complexity": complexity}
    )
    
    if result.success:
        print(f"✅ Service description uploaded: {result.cid}")
        return result.cid
    else:
        raise Exception(f"Failed to upload service description: {result.error}")


async def main():
    parser = argparse.ArgumentParser(
        description="Create a service description with PDF inputs on IPFS"
    )
    parser.add_argument(
        "--pdfs",
        nargs="+",
        help="PDF files to upload"
    )
    parser.add_argument(
        "--pdf-dir",
        help="Directory containing PDF files (default: ./utils/files)"
    )
    parser.add_argument(
        "--title",
        default="Literature Review on AI Agent Marketplaces",
        help="Service title"
    )
    parser.add_argument(
        "--complexity",
        choices=["low", "medium", "high"],
        default="medium",
        help="Service complexity"
    )
    
    args = parser.parse_args()
    
    # Check credentials
    if not os.getenv("PINATA_JWT") and not (os.getenv("PINATA_API_KEY") and os.getenv("PINATA_API_SECRET")):
        print("Error: Set PINATA_JWT or (PINATA_API_KEY + PINATA_API_SECRET)")
        sys.exit(1)
    
    # Collect PDF paths
    pdf_paths = []
    if args.pdfs:
        pdf_paths.extend([Path(p) for p in args.pdfs])
    elif args.pdf_dir:
        pdf_dir = Path(args.pdf_dir)
        pdf_paths.extend(pdf_dir.glob("*.pdf"))
        pdf_paths.extend(pdf_dir.glob("*.PDF"))
    else:
        # Default to ./utils/files directory
        default_dir = Path(__file__).parent.parent.parent / "utils" / "files"
        if default_dir.exists():
            pdf_paths.extend(default_dir.glob("*.pdf"))
            pdf_paths.extend(default_dir.glob("*.PDF"))
            print(f"Using default directory: {default_dir}")
        else:
            print(f"Error: Default directory not found: {default_dir}")
            print("Use --pdfs or --pdf-dir to specify files")
            sys.exit(1)
    
    if not pdf_paths:
        print("Error: No PDF files found")
        sys.exit(1)
    
    # Filter valid PDFs
    pdf_paths = [p for p in pdf_paths if p.exists() and p.suffix.lower() == '.pdf']
    
    if not pdf_paths:
        print("Error: No valid PDF files found")
        sys.exit(1)
    
    print("=" * 70)
    print(f"Creating service: {args.title}")
    print(f"PDF files: {len(pdf_paths)}")
    print("=" * 70)
    print()
    
    # Initialize client
    client = IPFSClient()
    
    # Upload PDFs
    pdf_cids = await upload_pdfs(client, pdf_paths)
    
    if not pdf_cids:
        print("❌ No PDFs uploaded successfully")
        sys.exit(1)
    
    # Define prompts for literature review
    prompts = [
        "What are the main approaches to building decentralized autonomous machines?",
        "Why blockchain technology is important for autonomous production units?",
        "What is the key architectural design of autonomous machines?",
        "What are the current capabilities of autonomous production units compared to human-operated systems?",
    ]
    
    # Create and upload service description
    service_cid = await create_and_upload_service(
        client=client,
        title=args.title,
        description=(
            "Comprehensive literature review analyzing the current state of autonomous machines, "
            "including architecture, capabilities, and future directions."
        ),
        prompts=prompts,
        pdf_cids=pdf_cids,
        complexity=args.complexity
    )
    
    # Print results
    print()
    print("=" * 70)
    print("✅ SERVICE CREATED SUCCESSFULLY")
    print("=" * 70)
    print()
    print(f"Service Description CID: {service_cid}")
    print(f"View at: https://gateway.pinata.cloud/ipfs/{service_cid}")
    print()
    print("PDF Files:")
    for filename, cid in pdf_cids.items():
        print(f"  {filename}: {cid}")
    print()
    print("=" * 70)
    print("Add this to contracts/.env:")
    print(f"  SERVICE_DESCRIPTION_CID={service_cid}")
    print()
    print("Or use in CreateAuction.s.sol script")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())

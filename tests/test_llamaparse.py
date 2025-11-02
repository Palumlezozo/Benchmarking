#!/usr/bin/env python3
"""
Test script for LlamaParse parsing API.

This script tests LlamaParse parsing using configuration from llm_v2/config
and parses the document(s) in data/documents/collection.
"""

import os
import sys
import asyncio
import time
import warnings
from pathlib import Path

# Suppress Pydantic warnings from dependencies
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from llama_parse import LlamaParse

# Import config
from llm_v2.config import config

# Load environment variables
load_dotenv()

# Create output directory for markdown files
MARKDOWN_OUTPUT_DIR = Path("tests/llamaparse_output")
MARKDOWN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

async def test_aparse():
    """Test LlamaParse parsing with config parameters."""
    print("🧪 Testing LlamaParse API\n")
    
    # Check for API key
    api_key = os.getenv("LLAMA_CLOUD_API_KEY")
    if not api_key:
        print("❌ LLAMA_CLOUD_API_KEY not found in environment")
        print("   Please set it in your .env file")
        return 1
    
    # Get base URL (optional)
    base_url = os.getenv("LLAMA_CLOUD_BASE_URL")
    
    # Get collection directory
    collection_dir = Path("data/documents/collection")
    if not collection_dir.exists():
        print(f"❌ Collection directory not found: {collection_dir}")
        return 1
    
    # Find PDF files
    pdf_files = list(collection_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"❌ No PDF files found in {collection_dir}")
        return 1
    
    print(f"📁 Found {len(pdf_files)} PDF file(s) in {collection_dir}")
    
    # Build parser configuration from config
    parser_config = {
        "api_key": api_key,
        "result_type": config.LLAMA_PARSE_RESULT_TYPE,
        "verbose": config.LLAMA_PARSE_VERBOSE,
        "language": config.LLAMA_PARSE_LANGUAGE,
        "parse_mode": config.LLAMA_PARSE_PARSE_MODE,
        "invalidate_cache": config.LLAMA_PARSE_INVALIDATE_CACHE,
        "do_not_cache": config.LLAMA_PARSE_DO_NOT_CACHE,
        "num_workers": config.LLAMA_PARSE_NUM_WORKERS,
        "skip_diagonal_text": config.LLAMA_PARSE_SKIP_DIAGONAL_TEXT,
        "spreadsheet_extract_sub_tables": config.LLAMA_PARSE_SPREADSHEET_EXTRACT_SUB_TABLES,
        "spreadsheet_force_formula_computation": config.LLAMA_PARSE_SPREADSHEET_FORCE_FORMULA_COMPUTATION,
    }
    
    # Add partition_pages if specified (for large document processing)
    if hasattr(config, 'LLAMA_PARSE_PARTITION_PAGES') and config.LLAMA_PARSE_PARTITION_PAGES is not None:
        parser_config["partition_pages"] = config.LLAMA_PARSE_PARTITION_PAGES
    
    # Add base_url if specified
    if base_url:
        parser_config["base_url"] = base_url
    
    # Create parser
    try:
        parser = LlamaParse(**parser_config)
        print(f"✅ LlamaParse parser created successfully")
    except Exception as e:
        print(f"❌ Failed to create parser: {e}")
        return 1
    
    # Test Method 1: Parse files individually
    async def parse_single_file(pdf_file: Path) -> tuple:
        """Parse a single file and save markdown."""
        file_name = pdf_file.name
        start_time = time.time()
        
        try:
            result = await parser.aparse(str(pdf_file))
            aparse_time = time.time() - start_time
            
            # Reconstruct markdown from result.pages with page separators
            if hasattr(result, 'pages') and result.pages:
                combined_markdown_parts = []
                for page in result.pages:
                    page_separator = f"<!-- PAGE: {page.page} -->\n"
                    combined_markdown_parts.append(page_separator)
                    # Handle None md (can happen in without_llm mode) - use text as fallback or empty string
                    page_content = page.md if page.md is not None else (page.text if hasattr(page, 'text') and page.text else "")
                    combined_markdown_parts.append(page_content)
                    combined_markdown_parts.append("\n\n")
                
                combined_markdown = "".join(combined_markdown_parts)
                
                # Save to file
                output_file = MARKDOWN_OUTPUT_DIR / f"method1_{file_name.replace('.pdf', '.md')}"
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(combined_markdown)
                
                print(f"   ✅ Saved: {output_file.name}")
            
            extract_time = time.time() - start_time
            return (file_name, True, None, None, aparse_time, extract_time)
            
        except Exception as e:
            error_time = time.time() - start_time
            print(f"   ❌ Failed: {file_name} - {e}")
            return (file_name, False, None, str(e), error_time, error_time)
    
    # Test Method 2: Parse files as a list (batch API call)
    async def parse_file_list(file_list: list) -> tuple:
        """Parse a list of files at once and save markdown."""
        start_time = time.time()
        
        try:
            result = await parser.aparse([str(f) for f in file_list])
            aparse_time = time.time() - start_time
            
            # Reconstruct markdown from result.pages for each JobResult in batch
            # Batch API returns a list of JobResults, one per file
            if isinstance(result, list):
                for i, job_result in enumerate(result):
                    if hasattr(job_result, 'pages') and job_result.pages:
                        file_name = file_list[i].name if i < len(file_list) else f"file_{i+1}"
                        
                        combined_markdown_parts = []
                        for page in job_result.pages:
                            page_separator = f"<!-- PAGE: {page.page} -->\n"
                            combined_markdown_parts.append(page_separator)
                            # Handle None md (can happen in without_llm mode) - use text as fallback or empty string
                            page_content = page.md if page.md is not None else (page.text if hasattr(page, 'text') and page.text else "")
                            combined_markdown_parts.append(page_content)
                            combined_markdown_parts.append("\n\n")
                        
                        combined_markdown = "".join(combined_markdown_parts)
                        
                        output_file = MARKDOWN_OUTPUT_DIR / f"method2_{file_name.replace('.pdf', '.md')}"
                        with open(output_file, 'w', encoding='utf-8') as f:
                            f.write(combined_markdown)
                        
                        print(f"   ✅ Saved: {output_file.name}")
            
            extract_time = time.time() - start_time
            return (True, None, None, aparse_time, extract_time)
            
        except Exception as e:
            error_time = time.time() - start_time
            print(f"   ❌ Batch parse failed: {e}")
            return (False, None, str(e), error_time, error_time)
    
    print(f"\n{'='*60}")
    print(f"📊 Performance Comparison Test")
    print(f"{'='*60}")
    print(f"Testing {len(pdf_files)} file(s): {', '.join(f.name for f in pdf_files)}")
    
    # METHOD 1: Individual sequential calls
    print(f"\n{'─'*60}")
    print(f"Method 1: Individual sequential calls")
    print(f"{'─'*60}")
    method1_start = time.time()
    
    for pdf_file in pdf_files:
        await parse_single_file(pdf_file)
    
    method1_elapsed = time.time() - method1_start
    
    # METHOD 2: Batch API call
    print(f"\n{'─'*60}")
    print(f"Method 2: Batch API call")
    print(f"{'─'*60}")
    method2_start = time.time()
    
    success2, _, _, aparse_time2, extract_time2 = await parse_file_list(pdf_files)
    
    method2_elapsed = time.time() - method2_start
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📊 Test Summary")
    print(f"{'='*60}")
    print(f"Method 1: {method1_elapsed:.2f}s")
    print(f"Method 2: {method2_elapsed:.2f}s")
    
    if method1_elapsed > 0 and method2_elapsed > 0:
        speedup = method1_elapsed / method2_elapsed
        if speedup > 1:
            print(f"\n🏆 Method 2 is {speedup:.2f}x FASTER")
        elif speedup < 1:
            print(f"\n🏆 Method 1 is {1/speedup:.2f}x FASTER")
    
    print(f"\n✅ Markdown files saved to: {MARKDOWN_OUTPUT_DIR}")
    
    if not success2:
        return 1
    
    return 0

def main():
    """Main entry point - runs async test."""
    return asyncio.run(test_aparse())

if __name__ == "__main__":
    sys.exit(main())


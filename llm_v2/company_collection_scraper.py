#!/usr/bin/env python3
"""
Company Collection Scraper

Script to scrape company websites using Crawl4AI and save markdown files to a collection.
Uses Tavily to search for company websites when company name is provided.
"""

import argparse
import asyncio
import hashlib
import os
import re
import sys
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

from dotenv import load_dotenv
from tavily import TavilyClient

# Add parent directory to path for imports when running as script
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "llm_v2"

try:
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
    from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
    CRAWL4AI_AVAILABLE = True
except ImportError:
    CRAWL4AI_AVAILABLE = False
    AsyncWebCrawler = None
    BrowserConfig = None
    CrawlerRunConfig = None
    CacheMode = None
    BFSDeepCrawlStrategy = None

try:
    from .config import (
        CRAWL4AI_BROWSER_TYPE,
        CRAWL4AI_HEADLESS,
        CRAWL4AI_PAGE_TIMEOUT,
        CRAWL4AI_WAIT_UNTIL,
        CRAWL4AI_MAX_DEPTH,
        CRAWL4AI_MAX_PAGES,
        CRAWL4AI_WORD_COUNT_THRESHOLD,
        CRAWL4AI_VERBOSE,
        DEFAULT_EMBEDDING_MODEL,
        DEFAULT_CHUNK_SIZE,
        DEFAULT_CHUNK_OVERLAP,
        RAG_MARKDOWNS_DIR,
        RAG_CHROMA_STORAGE_DIR,
        RAG_CHROMA_STATE_FILE,
        RAG_QDRANT_STATE_FILE,
        RAG_OPENAI_STATE_FILE,
        USE_QDRANT,
        QDRANT_HOST,
        QDRANT_PORT,
        QDRANT_API_KEY,
        QDRANT_HNSW_M,
        QDRANT_HNSW_EF_CONSTRUCT,
        QDRANT_HNSW_FULL_SCAN_THRESHOLD,
        QDRANT_ON_DISK,
        QDRANT_ON_DISK_PAYLOAD,
        QDRANT_DELETED_THRESHOLD,
        QDRANT_VACUUM_MIN_VECTOR_NUMBER,
        QDRANT_DEFAULT_SEGMENT_NUMBER,
        QDRANT_MAX_SEGMENT_SIZE,
        QDRANT_MEMMAP_THRESHOLD,
        QDRANT_INDEXING_THRESHOLD,
        QDRANT_FLUSH_INTERVAL_SEC,
        EMBEDDING_BATCH_SIZE,
    )
except ImportError:
    from config import (
        CRAWL4AI_BROWSER_TYPE,
        CRAWL4AI_HEADLESS,
        CRAWL4AI_PAGE_TIMEOUT,
        CRAWL4AI_WAIT_UNTIL,
        CRAWL4AI_MAX_DEPTH,
        CRAWL4AI_MAX_PAGES,
        CRAWL4AI_WORD_COUNT_THRESHOLD,
        CRAWL4AI_VERBOSE,
        DEFAULT_EMBEDDING_MODEL,
        DEFAULT_CHUNK_SIZE,
        DEFAULT_CHUNK_OVERLAP,
        RAG_MARKDOWNS_DIR,
        RAG_CHROMA_STORAGE_DIR,
        RAG_CHROMA_STATE_FILE,
        RAG_QDRANT_STATE_FILE,
        RAG_OPENAI_STATE_FILE,
        USE_QDRANT,
        QDRANT_HOST,
        QDRANT_PORT,
        QDRANT_API_KEY,
        QDRANT_HNSW_M,
        QDRANT_HNSW_EF_CONSTRUCT,
        QDRANT_HNSW_FULL_SCAN_THRESHOLD,
        QDRANT_ON_DISK,
        QDRANT_ON_DISK_PAYLOAD,
        QDRANT_DELETED_THRESHOLD,
        QDRANT_VACUUM_MIN_VECTOR_NUMBER,
        QDRANT_DEFAULT_SEGMENT_NUMBER,
        QDRANT_MAX_SEGMENT_SIZE,
        QDRANT_MEMMAP_THRESHOLD,
        QDRANT_INDEXING_THRESHOLD,
        QDRANT_FLUSH_INTERVAL_SEC,
        EMBEDDING_BATCH_SIZE,
    )

# Vector store imports
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance, VectorParams, HnswConfigDiff,
        OptimizersConfigDiff, VectorsConfig, PointStruct
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    QdrantClient = None
    Distance = None
    VectorParams = None
    HnswConfigDiff = None
    OptimizersConfigDiff = None
    VectorsConfig = None
    PointStruct = None

try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    chromadb = None

# OpenAI and chunking imports
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None

try:
    from llama_index.core.node_parser import SentenceSplitter
    LLAMA_INDEX_AVAILABLE = True
except ImportError:
    LLAMA_INDEX_AVAILABLE = False
    SentenceSplitter = None

# Embedding dimensions mapping
EMBEDDING_DIMENSIONS = {
    "text-embedding-3-large": 3072,
    "text-embedding-3-small": 1536,
    "text-embedding-ada-002": 1536,
}

# Load environment variables
load_dotenv()

# Constants
DEFAULT_DOCUMENTS_DIR = "data/documents"


def sanitize_filename(text: str, max_length: int = 100) -> str:
    """Sanitize text for use as filename."""
    # Remove special characters, keep alphanumeric, spaces, hyphens, underscores
    sanitized = re.sub(r'[^\w\s-]', '', text)
    # Replace spaces with underscores
    sanitized = re.sub(r'\s+', '_', sanitized)
    # Limit length
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    return sanitized


def extract_company_name_from_url(url: str) -> str:
    """Extract company name from URL domain."""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc
        # Remove www. and .com/.org etc.
        name = domain.replace('www.', '').split('.')[0]
        return name
    except Exception:
        return "company"


async def search_company_websites(company_name: str) -> List[dict]:
    """Search for company websites using Tavily."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY not found in environment variables")
    
    client = TavilyClient(api_key=api_key)
    
    # Search for company website
    query = f"{company_name} official website"
    print(f"🔍 Searching for '{company_name}' websites...")
    
    # Run synchronous Tavily call in thread pool to avoid blocking event loop
    response = await asyncio.to_thread(
        client.search,
        query=query,
        max_results=10,
        include_answer=False,
        include_raw_content=False
    )
    
    websites = []
    if response.get("results"):
        for result in response["results"]:
            url = result.get("url", "")
            title = result.get("title", "Untitled")
            if url:
                websites.append({
                    "url": url,
                    "title": title
                })
    
    return websites


def select_website(websites: List[dict]) -> Optional[str]:
    """Present website options to user and get selection."""
    if not websites:
        print("❌ No websites found")
        return None
    
    print(f"\n📋 Found {len(websites)} website(s):")
    for i, site in enumerate(websites, 1):
        print(f"  {i}. {site['title']}")
        print(f"     {site['url']}")
    
    print(f"\n  0. Enter custom URL")
    
    while True:
        try:
            choice = input("\nSelect website (number or 0 for custom): ").strip()
            
            if choice == "0":
                custom_url = input("Enter website URL: ").strip()
                if custom_url:
                    # Ensure URL has protocol
                    if not custom_url.startswith(('http://', 'https://')):
                        custom_url = f"https://{custom_url}"
                    return custom_url
                else:
                    print("⚠️  Empty URL, please try again")
                    continue
            
            choice_num = int(choice)
            if 1 <= choice_num <= len(websites):
                return websites[choice_num - 1]["url"]
            else:
                print(f"⚠️  Please enter a number between 1 and {len(websites)}")
        except ValueError:
            print("⚠️  Please enter a valid number")
        except KeyboardInterrupt:
            print("\n❌ Cancelled by user")
            return None


async def crawl_website(url: str, collection_dir: Path, verbose: bool = False) -> List[dict]:
    """Crawl website using Crawl4AI's built-in deep crawling strategy.
    
    Args:
        url: URL to crawl
        collection_dir: Directory for collection (not used for storage, kept for compatibility)
        verbose: If True, show detailed progress information
    """
    if not CRAWL4AI_AVAILABLE:
        raise ImportError("crawl4ai not available. Install with: pip install crawl4ai")
    
    if BFSDeepCrawlStrategy is None:
        raise ImportError("crawl4ai.deep_crawling not available. Please update crawl4ai: pip install --upgrade crawl4ai")
    
    print(f"\n🕷️  Crawling website: {url}")
    print(f"   Max depth: {CRAWL4AI_MAX_DEPTH}, Max pages: {CRAWL4AI_MAX_PAGES}")
    if verbose:
        print(f"   Word count threshold: {CRAWL4AI_WORD_COUNT_THRESHOLD} words")
        print(f"   Page timeout: {CRAWL4AI_PAGE_TIMEOUT}ms")
        print(f"   Wait until: {CRAWL4AI_WAIT_UNTIL}")
    
    # Configure browser
    browser_config = BrowserConfig(
        browser_type=CRAWL4AI_BROWSER_TYPE,
        headless=CRAWL4AI_HEADLESS,
    )
    
    # Configure deep crawling strategy
    # BFSDeepCrawlStrategy handles all link extraction and multi-page crawling automatically
    deep_crawl_strategy = BFSDeepCrawlStrategy(
        max_depth=CRAWL4AI_MAX_DEPTH,
        include_external=False,  # Stay within the same domain
        max_pages=CRAWL4AI_MAX_PAGES,
    )
    
    # Configure crawler run with deep crawling strategy
    run_config = CrawlerRunConfig(
        deep_crawl_strategy=deep_crawl_strategy,
        page_timeout=CRAWL4AI_PAGE_TIMEOUT,
        wait_until=CRAWL4AI_WAIT_UNTIL,
        word_count_threshold=CRAWL4AI_WORD_COUNT_THRESHOLD,
        verbose=CRAWL4AI_VERBOSE or verbose,  # Enable verbose if --info is set
        cache_mode=CacheMode.DISABLED,  # Don't cache for fresh content
        stream=False,  # Get all results at once (non-streaming mode)
    )
    
    crawled_pages = []
    skipped_pages = []
    failed_pages = []
    
    async with AsyncWebCrawler(config=browser_config) as crawler:
        if verbose:
            print(f"\n   🔄 Starting deep crawl...")
            print(f"   📍 Initial URL: {url}")
        else:
            print(f"   Starting deep crawl...")
        
        # arun() with deep_crawl_strategy returns a list of results
        results = await crawler.arun(url=url, config=run_config)
        
        # Handle both single result and list of results
        if not isinstance(results, list):
            results = [results]
        
        total_results = len(results)
        if verbose:
            print(f"\n   📊 Processing {total_results} page(s) from crawl...")
        else:
            print(f"   ✅ Crawled {total_results} page(s)")
        
        # Process each result
        for idx, result in enumerate(results, 1):
            if verbose:
                print(f"\n   [{idx}/{total_results}] Processing: {result.url}")
            
            # Skip failed results
            if not result.success:
                error_msg = result.error_message if hasattr(result, 'error_message') else 'Unknown error'
                failed_pages.append({"url": result.url, "reason": error_msg})
                if verbose:
                    print(f"      ❌ Failed: {error_msg}")
                elif CRAWL4AI_VERBOSE:
                    print(f"      ⚠️  Skipped {result.url}: {error_msg}")
                continue
            
            # Check if we have markdown content
            if not result.markdown:
                skipped_pages.append({"url": result.url, "reason": "No markdown content"})
                if verbose:
                    print(f"      ⚠️  Skipped: No markdown content")
                elif CRAWL4AI_VERBOSE:
                    print(f"      ⚠️  Skipped {result.url}: No markdown content")
                continue
            
            # Check word count
            word_count = len(result.markdown.split())
            if word_count < CRAWL4AI_WORD_COUNT_THRESHOLD:
                skipped_pages.append({
                    "url": result.url, 
                    "reason": f"Too few words ({word_count} < {CRAWL4AI_WORD_COUNT_THRESHOLD})"
                })
                if verbose:
                    print(f"      ⚠️  Skipped: Too few words ({word_count} < {CRAWL4AI_WORD_COUNT_THRESHOLD})")
                elif CRAWL4AI_VERBOSE:
                    print(f"      ⚠️  Skipped {result.url}: Too few words ({word_count})")
                continue
            
            # Extract title from metadata or HTML
            title = "Untitled"
            if result.metadata and isinstance(result.metadata, dict):
                title = result.metadata.get("title", "Untitled")
            elif hasattr(result, 'html') and result.html:
                # Try to extract title from HTML
                title_match = re.search(r'<title[^>]*>([^<]+)</title>', result.html, re.IGNORECASE)
                if title_match:
                    title = title_match.group(1).strip()
            
            # Get depth from metadata (added by deep crawling strategy)
            depth = result.metadata.get("depth", 0) if result.metadata and isinstance(result.metadata, dict) else 0
            
            crawled_pages.append({
                "url": result.url,
                "markdown": result.markdown,
                "title": title,
                "depth": depth
            })
            
            if verbose:
                print(f"      ✅ Success: {title}")
                print(f"         Depth: {depth}, Words: {word_count}, Chars: {len(result.markdown)}")
            else:
                print(f"      ✅ {result.url} (depth {depth}, {word_count} words)")
    
    # Print summary if verbose
    if verbose:
        print(f"\n   📈 Crawl Summary:")
        print(f"      ✅ Successfully crawled: {len(crawled_pages)} page(s)")
        if skipped_pages:
            print(f"      ⚠️  Skipped: {len(skipped_pages)} page(s)")
            for skipped in skipped_pages[:5]:  # Show first 5 skipped
                print(f"         - {skipped['url']}: {skipped['reason']}")
            if len(skipped_pages) > 5:
                print(f"         ... and {len(skipped_pages) - 5} more")
        if failed_pages:
            print(f"      ❌ Failed: {len(failed_pages)} page(s)")
            for failed in failed_pages[:5]:  # Show first 5 failed
                print(f"         - {failed['url']}: {failed['reason']}")
            if len(failed_pages) > 5:
                print(f"         ... and {len(failed_pages) - 5} more")
        
        # Depth distribution
        if crawled_pages:
            depth_dist = {}
            for page in crawled_pages:
                depth = page.get('depth', 0)
                depth_dist[depth] = depth_dist.get(depth, 0) + 1
            print(f"\n   📊 Depth Distribution:")
            for depth in sorted(depth_dist.keys()):
                print(f"      Depth {depth}: {depth_dist[depth]} page(s)")
    
    return crawled_pages


def save_markdown_files(pages: List[dict], collection_dir: Path):
    """Save crawled pages as markdown files."""
    collection_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving {len(pages)} markdown file(s) to {collection_dir}...")
    
    for i, page in enumerate(pages, 1):
        url = page["url"]
        markdown = page["markdown"]
        title = page["title"]
        
        # Create filename from title or URL
        if title and title != "Untitled":
            filename = sanitize_filename(title)
        else:
            # Extract from URL path
            parsed = urlparse(url)
            path_parts = [p for p in parsed.path.split('/') if p]
            if path_parts:
                filename = sanitize_filename('_'.join(path_parts[-2:]))  # Last 2 path parts
            else:
                filename = "index"
        
        # Add index if needed to avoid duplicates
        base_filename = filename
        counter = 1
        while (collection_dir / f"{filename}.md").exists():
            filename = f"{base_filename}_{counter}"
            counter += 1
        
        filepath = collection_dir / f"{filename}.md"
        
        # Add URL metadata to markdown header
        markdown_content = f"<!--\nSource URL: {url}\nCrawled: {page.get('depth', 0)} levels deep\n-->\n\n{markdown}"
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            print(f"   ✅ {filepath.name}")
        except Exception as e:
            print(f"   ❌ Failed to save {filename}.md: {e}")


def delete_collection(collection_name: str) -> bool:
    """Delete vector stores, markdown files, and state files for a collection."""
    # Ensure collection name has -web suffix
    if not collection_name.endswith("-web"):
        collection_name = f"{collection_name}-web"
    
    markdown_dir = Path(RAG_MARKDOWNS_DIR) / collection_name
    
    # Check if collection exists in any location
    collection_exists = (
        markdown_dir.exists() or
        (Path(RAG_CHROMA_STORAGE_DIR) / collection_name).exists() or
        Path(RAG_CHROMA_STATE_FILE).exists() or
        Path(RAG_QDRANT_STATE_FILE).exists()
    )
    
    if not collection_exists:
        print(f"❌ Collection '{collection_name}' does not exist")
        return False
    
    # Confirm deletion
    try:
        import shutil
        import json
        from datetime import datetime
        
        print(f"🗑️  Deleting collection '{collection_name}'...")
        
        # Count files in markdown directory if it exists
        if markdown_dir.exists():
            md_count = len(list(markdown_dir.glob("*.md")))
            print(f"   Markdown files: {md_count}")
        
        # Delete vector stores
        vector_collection_name = f"{collection_name}_vectors"
        
        # Try to delete Qdrant collection
        if QDRANT_AVAILABLE:
            try:
                client_kwargs = {
                    "host": QDRANT_HOST,
                    "port": QDRANT_PORT
                }
                if QDRANT_API_KEY:
                    client_kwargs["api_key"] = QDRANT_API_KEY
                
                qdrant_client = QdrantClient(**client_kwargs)
                collections = qdrant_client.get_collections().collections
                collection_exists = any(col.name == vector_collection_name for col in collections)
                
                if collection_exists:
                    qdrant_client.delete_collection(vector_collection_name)
                    print(f"   ✅ Deleted Qdrant collection '{vector_collection_name}'")
                else:
                    print(f"   ℹ️  Qdrant collection '{vector_collection_name}' not found")
            except Exception as e:
                print(f"   ⚠️  Warning: Could not delete Qdrant collection: {e}")
        
        # Try to delete ChromaDB collection
        if CHROMA_AVAILABLE:
            try:
                chroma_storage_dir = Path(RAG_CHROMA_STORAGE_DIR) / collection_name
                if chroma_storage_dir.exists():
                    chroma_client = chromadb.PersistentClient(path=str(chroma_storage_dir))
                    try:
                        chroma_client.delete_collection(name=vector_collection_name)
                        print(f"   ✅ Deleted ChromaDB collection '{vector_collection_name}'")
                    except Exception:
                        pass  # Collection might not exist
            except Exception as e:
                print(f"   ⚠️  Warning: Could not delete ChromaDB collection: {e}")
        
        # Delete markdown files from rag/markdowns
        if markdown_dir.exists():
            try:
                md_count = len(list(markdown_dir.glob("*.md")))
                shutil.rmtree(markdown_dir)
                print(f"   ✅ Deleted markdown directory: {markdown_dir} ({md_count} file(s))")
            except Exception as e:
                print(f"   ⚠️  Warning: Could not delete markdown directory {markdown_dir}: {e}")
        
        # Delete ChromaDB storage directory if it exists
        chroma_storage_dir = Path(RAG_CHROMA_STORAGE_DIR) / collection_name
        if chroma_storage_dir.exists():
            try:
                shutil.rmtree(chroma_storage_dir)
                print(f"   ✅ Deleted ChromaDB storage directory: {chroma_storage_dir}")
            except Exception as e:
                print(f"   ⚠️  Warning: Could not delete ChromaDB storage directory {chroma_storage_dir}: {e}")
        
        # Update global state files (remove collection from state)
        state_files = [
            (Path(RAG_CHROMA_STATE_FILE), "ChromaDB"),
            (Path(RAG_QDRANT_STATE_FILE), "Qdrant"),
            (Path(RAG_OPENAI_STATE_FILE), "OpenAI"),
        ]
        
        for state_file, backend_name in state_files:
            if state_file.exists():
                try:
                    with open(state_file, 'r', encoding='utf-8') as f:
                        state = json.load(f)
                    
                    # Remove collection from state if it exists
                    if "collections" in state and collection_name in state["collections"]:
                        del state["collections"][collection_name]
                        state["last_updated"] = datetime.now().isoformat()
                        
                        with open(state_file, 'w', encoding='utf-8') as f:
                            json.dump(state, f, indent=2)
                        print(f"   ✅ Updated {backend_name} state file (removed collection '{collection_name}')")
                except Exception as e:
                    print(f"   ⚠️  Warning: Could not update {backend_name} state file: {e}")
        
        print(f"\n✅ Successfully deleted collection '{collection_name}'")
        return True
        
    except Exception as e:
        print(f"❌ Error deleting collection: {e}")
        import traceback
        traceback.print_exc()
        return False


def _get_embedding_dimension(embedding_model: str) -> int:
    """Get the embedding dimension for a given model."""
    return EMBEDDING_DIMENSIONS.get(embedding_model, 1536)


def _chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[str]:
    """Chunk text into smaller pieces."""
    if not LLAMA_INDEX_AVAILABLE:
        raise ImportError("llama-index-core not available. Install with: pip install llama-index-core")
    
    node_parser = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    # Create a simple document-like object
    from llama_index.core import Document
    doc = Document(text=text)
    
    # Parse into nodes
    nodes = node_parser.get_nodes_from_documents([doc])
    
    # Extract text from nodes
    return [node.text for node in nodes]


async def _generate_embeddings(texts: List[str], embedding_model: str, client: OpenAI) -> List[List[float]]:
    """Generate embeddings for a list of texts using OpenAI."""
    if not OPENAI_AVAILABLE:
        raise ImportError("openai not available. Install with: pip install openai")
    
    # Process in batches to respect rate limits
    all_embeddings = []
    batch_size = EMBEDDING_BATCH_SIZE
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        if total_batches > 1:
            print(f"      Generating embeddings for batch {batch_num}/{total_batches} ({len(batch)} texts)...")
        
        # Generate embeddings for this batch
        response = await asyncio.to_thread(
            client.embeddings.create,
            model=embedding_model,
            input=batch
        )
        
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)
        
        # Add delay between batches (except for last batch)
        if i + batch_size < len(texts) and EMBEDDING_BATCH_SIZE > 0:
            await asyncio.sleep(0.1)  # Small delay to avoid rate limits
    
    return all_embeddings


def _index_to_qdrant(
    collection_name: str,
    chunks: List[str],
    embeddings: List[List[float]],
    metadata_list: List[dict],
    embedding_model: str
) -> bool:
    """Index chunks to Qdrant using native SDK."""
    if not QDRANT_AVAILABLE:
        raise ImportError("qdrant-client not available. Install with: pip install qdrant-client")
    
    try:
        # Initialize Qdrant client
        client_kwargs = {
            "host": QDRANT_HOST,
            "port": QDRANT_PORT
        }
        if QDRANT_API_KEY:
            client_kwargs["api_key"] = QDRANT_API_KEY
        
        client = QdrantClient(**client_kwargs)
        
        # Get embedding dimension
        embedding_dim = _get_embedding_dimension(embedding_model)
        
        # Check if collection exists
        collections = client.get_collections().collections
        collection_exists = any(col.name == collection_name for col in collections)
        
        if not collection_exists:
            # Create collection with optimized parameters
            hnsw_config = HnswConfigDiff(
                m=QDRANT_HNSW_M,
                ef_construct=QDRANT_HNSW_EF_CONSTRUCT,
                full_scan_threshold=QDRANT_HNSW_FULL_SCAN_THRESHOLD,
                on_disk=QDRANT_ON_DISK,
            )
            
            optimizer_config = OptimizersConfigDiff(
                deleted_threshold=QDRANT_DELETED_THRESHOLD,
                vacuum_min_vector_number=QDRANT_VACUUM_MIN_VECTOR_NUMBER,
                default_segment_number=QDRANT_DEFAULT_SEGMENT_NUMBER,
                max_segment_size=QDRANT_MAX_SEGMENT_SIZE,
                memmap_threshold=QDRANT_MEMMAP_THRESHOLD,
                indexing_threshold=QDRANT_INDEXING_THRESHOLD,
                flush_interval_sec=QDRANT_FLUSH_INTERVAL_SEC,
            )
            
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=embedding_dim,
                    distance=Distance.COSINE,
                    on_disk=QDRANT_ON_DISK,
                ),
                hnsw_config=hnsw_config,
                optimizers_config=optimizer_config,
                on_disk_payload=QDRANT_ON_DISK_PAYLOAD,
            )
            print(f"   Created Qdrant collection '{collection_name}'")
        
        # Prepare points for insertion
        points = []
        for idx, (chunk, embedding, metadata) in enumerate(zip(chunks, embeddings, metadata_list)):
            # Generate unique ID based on content hash and index
            content_hash = hashlib.md5(chunk.encode()).hexdigest()[:8]
            unique_id = int(hashlib.md5(f"{metadata.get('file_path', '')}_{idx}_{content_hash}".encode()).hexdigest()[:16], 16)
            points.append(
                PointStruct(
                    id=unique_id,
                    vector=embedding,
                    payload={
                        "text": chunk,
                        **metadata
                    }
                )
            )
        
        # Insert points in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            client.upsert(collection_name=collection_name, points=batch)
        
        print(f"   ✅ Indexed {len(chunks)} chunks to Qdrant collection '{collection_name}'")
        return True
        
    except Exception as e:
        print(f"   ❌ Error indexing to Qdrant: {e}")
        import traceback
        traceback.print_exc()
        return False


def _index_to_chroma(
    collection_name: str,
    chunks: List[str],
    embeddings: List[List[float]],
    metadata_list: List[dict],
    storage_dir: Path
) -> bool:
    """Index chunks to ChromaDB using native SDK."""
    if not CHROMA_AVAILABLE:
        raise ImportError("chromadb not available. Install with: pip install chromadb")
    
    try:
        # Initialize Chroma client
        client = chromadb.PersistentClient(path=str(storage_dir))
        
        # Get or create collection
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Prepare data for insertion with unique IDs
        ids = []
        for idx, (chunk, metadata) in enumerate(zip(chunks, metadata_list)):
            content_hash = hashlib.md5(chunk.encode()).hexdigest()[:8]
            unique_id = f"{metadata.get('file_path', '')}_{idx}_{content_hash}"
            ids.append(unique_id)
        
        documents = chunks
        metadatas = metadata_list
        
        # Insert in batches
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch_ids = ids[i:i + batch_size]
            batch_docs = documents[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]
            
            collection.add(
                ids=batch_ids,
                documents=batch_docs,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas
            )
        
        print(f"   ✅ Indexed {len(chunks)} chunks to ChromaDB collection '{collection_name}'")
        return True
        
    except Exception as e:
        print(f"   ❌ Error indexing to ChromaDB: {e}")
        import traceback
        traceback.print_exc()
        return False


async def index_markdown_files(collection_dir: Path, collection_name: str, use_chroma: bool = False) -> bool:
    """Chunk, embed, and index markdown files to vector store."""
    if not OPENAI_AVAILABLE:
        print("⚠️  OpenAI not available. Skipping indexing.")
        return False
    
    if not LLAMA_INDEX_AVAILABLE:
        print("⚠️  llama-index-core not available. Skipping indexing.")
        return False
    
    # Check vector store availability
    if use_chroma:
        if not CHROMA_AVAILABLE:
            print("⚠️  ChromaDB not available. Install with: pip install chromadb")
            return False
    else:
        if not QDRANT_AVAILABLE:
            print("⚠️  Qdrant not available. Install with: pip install qdrant-client")
            return False
    
    # Get all markdown files
    md_files = list(collection_dir.glob("*.md"))
    if not md_files:
        print("⚠️  No markdown files found to index")
        return False
    
    print(f"\n📚 Indexing {len(md_files)} markdown file(s) to vector store...")
    print(f"   Collection: {collection_name}")
    print(f"   Vector Store: {'ChromaDB' if use_chroma else 'Qdrant'}")
    
    # Initialize OpenAI client
    azure_base_url = os.getenv("AZURE_OPENAI_BASE_URL")
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    
    if azure_base_url and azure_api_key:
        client = OpenAI(api_key=azure_api_key, base_url=azure_base_url)
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("❌ OPENAI_API_KEY or AZURE_OPENAI_API_KEY not found")
            return False
        client = OpenAI(api_key=api_key)
    
    # Process all files: chunk, collect metadata
    all_chunks = []
    all_metadata = []
    
    for md_file in md_files:
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract URL from markdown header if present
            url = None
            depth = 0
            if content.startswith("<!--"):
                header_end = content.find("-->")
                if header_end > 0:
                    header = content[:header_end + 3]
                    url_match = re.search(r'Source URL: ([^\n]+)', header)
                    if url_match:
                        url = url_match.group(1).strip()
                    depth_match = re.search(r'Crawled: (\d+)', header)
                    if depth_match:
                        depth = int(depth_match.group(1))
                    # Remove header from content
                    content = content[header_end + 3:].strip()
            
            # Chunk the content
            chunks = _chunk_text(content, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP)
            
            # Create metadata for each chunk
            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadata.append({
                    "file_name": md_file.name,
                    "file_path": str(md_file),
                    "chunk_index": chunk_idx,
                    "total_chunks": len(chunks),
                    "url": url or "",
                    "depth": depth,
                })
            
            # Log URL if found
            url_info = f" (URL: {url})" if url else ""
            print(f"   ✅ {md_file.name}: {len(chunks)} chunks{url_info}")
            
        except Exception as e:
            print(f"   ⚠️  Error processing {md_file.name}: {e}")
            continue
    
    if not all_chunks:
        print("❌ No chunks created from markdown files")
        return False
    
    print(f"\n   Generating embeddings for {len(all_chunks)} chunks...")
    
    # Generate embeddings
    try:
        embeddings = await _generate_embeddings(all_chunks, DEFAULT_EMBEDDING_MODEL, client)
    except Exception as e:
        print(f"❌ Error generating embeddings: {e}")
        return False
    
    # Index to vector store
    vector_collection_name = f"{collection_name}_vectors"
    
    if use_chroma:
        # Determine storage directory for ChromaDB
        storage_dir = Path(RAG_CHROMA_STORAGE_DIR) / collection_name
        storage_dir.mkdir(parents=True, exist_ok=True)
        success = _index_to_chroma(vector_collection_name, all_chunks, embeddings, all_metadata, storage_dir)
    else:
        success = _index_to_qdrant(vector_collection_name, all_chunks, embeddings, all_metadata, DEFAULT_EMBEDDING_MODEL)
    
    # Update global state file to mark index as created
    if success:
        try:
            import json
            from datetime import datetime
            
            # Determine which state file to update based on vector store type
            if use_chroma:
                state_file = Path(RAG_CHROMA_STATE_FILE)
            else:
                state_file = Path(RAG_QDRANT_STATE_FILE)
            
            # Ensure state file directory exists
            state_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Load existing state or create new
            if state_file.exists():
                with open(state_file, 'r', encoding='utf-8') as f:
                    global_state = json.load(f)
            else:
                global_state = {"collections": {}, "last_updated": None}
            
            # Initialize collection state if it doesn't exist
            if "collections" not in global_state:
                global_state["collections"] = {}
            
            if collection_name not in global_state["collections"]:
                global_state["collections"][collection_name] = {
                    "documents": {},
                    "last_updated": None,
                    "total_pages": 0,
                    "indexed_pages": 0,
                    "failed_pages": 0,
                    "index_created": False,
                    "vector_store_type": "chroma" if use_chroma else "qdrant"
                }
            
            # Update collection state
            collection_state = global_state["collections"][collection_name]
            collection_state["index_created"] = True
            collection_state["last_updated"] = datetime.now().isoformat()
            collection_state["vector_store_type"] = "chroma" if use_chroma else "qdrant"
            
            # Count documents (markdown files) and chunks
            collection_state["total_pages"] = len(md_files)
            collection_state["indexed_pages"] = len(md_files)
            collection_state["failed_pages"] = 0
            
            # Add document metadata for each markdown file
            for md_file in md_files:
                file_path_str = str(md_file)
                collection_state["documents"][file_path_str] = {
                    "size": md_file.stat().st_size if md_file.exists() else 0,
                    "modified": md_file.stat().st_mtime if md_file.exists() else 0,
                    "extension": md_file.suffix.lower(),
                    "indexed": True,
                    "indexed_at": datetime.now().isoformat(),
                    "chunk_count": sum(1 for m in all_metadata if m.get("file_path") == str(md_file))
                }
            
            # Update global state timestamp
            global_state["last_updated"] = datetime.now().isoformat()
            
            # Save state file
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(global_state, f, indent=2)
            
            print(f"   ✅ Updated state file: {state_file}")
            
        except Exception as e:
            print(f"   ⚠️  Warning: Could not update state file: {e}")
            import traceback
            traceback.print_exc()
    
    return success


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Scrape company website and save markdown files to collection, or delete collection content"
    )
    parser.add_argument(
        "--collection",
        type=str,
        required=True,
        help="Collection name (accepts both <name> or <name>-web; will add '-web' suffix if missing)"
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete all content from the specified collection"
    )
    parser.add_argument(
        "--company",
        type=str,
        help="Company name (will search for website using Tavily). Required unless --delete is used."
    )
    parser.add_argument(
        "--website",
        type=str,
        help="Website URL (direct URL to crawl). Required unless --delete is used."
    )
    parser.add_argument(
        "--chroma",
        action="store_true",
        help="Use ChromaDB instead of Qdrant for vector storage"
    )
    parser.add_argument(
        "--store-md",
        action="store_true",
        help=f"Store markdown files in {RAG_MARKDOWNS_DIR}/<collection-name> (default: files are not saved, only indexed)"
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show detailed progress information during crawling (verbose mode)"
    )
    
    args = parser.parse_args()
    
    # Handle delete operation
    if args.delete:
        success = delete_collection(args.collection)
        sys.exit(0 if success else 1)
    
    # For crawling operations, validate that we have company or website
    if not args.company and not args.website:
        parser.error("Either --company or --website must be provided for crawling operations")
    
    # Determine company name and website URL
    company_name = args.company
    website_url = args.website
    
    if args.website:
        # Extract company name from URL if not provided (for informational purposes only)
        if not company_name:
            company_name = extract_company_name_from_url(website_url)
            print(f"📌 Extracted company name from URL: {company_name}")
    else:
        # Search for websites using Tavily
        try:
            websites = await search_company_websites(company_name)
            website_url = select_website(websites)
            if not website_url:
                print("❌ No website selected")
                return
        except Exception as e:
            print(f"❌ Error searching for websites: {e}")
            return
    
    # Use collection name as provided (mandatory)
    # Always append "-web" suffix to avoid confusion with other collections
    # Accept both <name> and <name>-web, add -web if missing
    base_collection_name = sanitize_filename(args.collection)
    if base_collection_name.endswith("-web"):
        collection_name = base_collection_name
    else:
        collection_name = f"{base_collection_name}-web"
    
    # Always save markdown files to rag/markdowns directory
    # If --store-md is not set, they will be deleted after indexing
    markdown_dir = Path(RAG_MARKDOWNS_DIR) / collection_name
    
    # Crawl website (collection_dir is not used for file storage anymore)
    collection_dir = Path(DEFAULT_DOCUMENTS_DIR) / collection_name
    
    # Crawl website
    try:
        pages = await crawl_website(website_url, collection_dir, verbose=args.info)
        
        if not pages:
            print("❌ No pages crawled successfully")
            return
        
        # Save markdown files
        save_markdown_files(pages, markdown_dir)
        
        # Index markdown files to vector store
        use_chroma = args.chroma
        indexing_success = await index_markdown_files(markdown_dir, collection_name, use_chroma=use_chroma)
        
        # If --store-md was not set, delete the markdown files after indexing
        if not args.store_md:
            try:
                import shutil
                for md_file in markdown_dir.glob("*.md"):
                    md_file.unlink()
                # Remove directory if empty
                if markdown_dir.exists() and not any(markdown_dir.iterdir()):
                    markdown_dir.rmdir()
                print(f"   🗑️  Removed temporary markdown files (use --store-md to keep them)")
            except Exception as e:
                print(f"   ⚠️  Warning: Could not remove temporary markdown files: {e}")
        
        print(f"\n✅ Successfully created collection '{collection_name}' with {len(pages)} page(s)")
        if args.store_md:
            print(f"   Markdown files: {markdown_dir}")
        if indexing_success:
            vector_store = "ChromaDB" if use_chroma else "Qdrant"
            print(f"   ✅ Indexed to {vector_store} collection '{collection_name}_vectors'")
        else:
            print(f"   ⚠️  Indexing skipped or failed")
        print(f"\n💡 Next steps:")
        if not indexing_success:
            print(f"   To index these documents, run:")
            print(f"   python -m llm_v2.document_manager --collection {collection_name} --update")
        print(f"\n   To delete this collection's content, run:")
        print(f"   python -m llm_v2.company_collection_scraper --collection {collection_name} --delete")
        
    except Exception as e:
        print(f"❌ Error crawling website: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    asyncio.run(main())


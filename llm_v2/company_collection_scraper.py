#!/usr/bin/env python3
"""
Company Collection Scraper

Script to scrape company websites using Crawl4AI and save markdown files to a collection.
Uses Tavily to search for company websites when company name is provided.
"""

import argparse
import asyncio
import os
import re
import sys
from pathlib import Path
from typing import List, Optional
from urllib.parse import urljoin, urlparse

from dotenv import load_dotenv
from tavily import TavilyClient

# Add parent directory to path for imports when running as script
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "llm_v2"

try:
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
    CRAWL4AI_AVAILABLE = True
except ImportError:
    CRAWL4AI_AVAILABLE = False
    AsyncWebCrawler = None
    BrowserConfig = None
    CrawlerRunConfig = None
    CacheMode = None

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
    )

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
    
    response = client.search(
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


async def crawl_website(url: str, collection_dir: Path) -> List[dict]:
    """Crawl website using Crawl4AI and return page data."""
    if not CRAWL4AI_AVAILABLE:
        raise ImportError("crawl4ai not available. Install with: pip install crawl4ai")
    
    print(f"\n🕷️  Crawling website: {url}")
    print(f"   Max depth: {CRAWL4AI_MAX_DEPTH}, Max pages: {CRAWL4AI_MAX_PAGES}")
    
    # Configure browser
    browser_config = BrowserConfig(
        browser_type=CRAWL4AI_BROWSER_TYPE,
        headless=CRAWL4AI_HEADLESS,
    )
    
    # Configure crawler run
    run_config = CrawlerRunConfig(
        page_timeout=CRAWL4AI_PAGE_TIMEOUT,
        wait_until=CRAWL4AI_WAIT_UNTIL,
        word_count_threshold=CRAWL4AI_WORD_COUNT_THRESHOLD,
        verbose=CRAWL4AI_VERBOSE,
        cache_mode=CacheMode.DISABLED,  # Don't cache for fresh content
    )
    
    crawled_pages = []
    parsed_base = urlparse(url)
    base_domain = parsed_base.netloc
    # Normalize base URL (remove trailing slash)
    base_url = f"{parsed_base.scheme}://{parsed_base.netloc}{parsed_base.path}".rstrip('/')
    
    async with AsyncWebCrawler(config=browser_config) as crawler:
        # Start with the base URL
        visited_urls = set()
        urls_to_crawl = [(base_url, 0)]  # (url, depth)
        
        while urls_to_crawl and len(crawled_pages) < CRAWL4AI_MAX_PAGES:
            current_url, depth = urls_to_crawl.pop(0)
            
            # Skip if already visited or max depth reached
            if current_url in visited_urls or depth > CRAWL4AI_MAX_DEPTH:
                continue
            
            visited_urls.add(current_url)
            
            try:
                print(f"   Crawling: {current_url} (depth {depth})...")
                result = await crawler.arun(url=current_url, config=run_config)
                
                if result.success and result.markdown:
                    # Check word count
                    word_count = len(result.markdown.split())
                    if word_count >= CRAWL4AI_WORD_COUNT_THRESHOLD:
                        # Extract title from metadata or HTML
                        title = "Untitled"
                        if result.metadata and isinstance(result.metadata, dict):
                            title = result.metadata.get("title", "Untitled")
                        elif hasattr(result, 'html') and result.html:
                            # Try to extract title from HTML
                            title_match = re.search(r'<title[^>]*>([^<]+)</title>', result.html, re.IGNORECASE)
                            if title_match:
                                title = title_match.group(1).strip()
                        
                        crawled_pages.append({
                            "url": current_url,
                            "markdown": result.markdown,
                            "title": title,
                            "depth": depth
                        })
                        print(f"      ✅ Saved ({word_count} words)")
                    else:
                        print(f"      ⚠️  Skipped (too few words: {word_count})")
                    
                    # Extract links for further crawling (if not at max depth)
                    if depth < CRAWL4AI_MAX_DEPTH:
                        links = []
                        # Try different ways to get links from Crawl4AI result
                        # First check if Crawl4AI provides links directly
                        if hasattr(result, 'links') and result.links:
                            try:
                                links = list(result.links) if not isinstance(result.links, list) else result.links
                            except Exception:
                                pass
                        
                        # Try extracted_links attribute
                        if not links and hasattr(result, 'extracted_links') and result.extracted_links:
                            try:
                                links = list(result.extracted_links) if not isinstance(result.extracted_links, list) else result.extracted_links
                            except Exception:
                                pass
                        
                        # Fallback: extract from HTML if available
                        if not links and hasattr(result, 'html') and result.html:
                            try:
                                # Extract links from HTML (more comprehensive pattern)
                                link_pattern = r'href=["\']([^"\']+)["\']'
                                found_links = re.findall(link_pattern, result.html, re.IGNORECASE)
                                # Filter out empty and malformed links
                                links = [l for l in found_links if l and not l.startswith('#') and not l.startswith('javascript:')]
                            except Exception:
                                pass
                        
                        # Ensure links is a list and limit to 20
                        if isinstance(links, list):
                            links = links[:20]
                        else:
                            links = []
                        
                        # Debug: print how many links found
                        if links:
                            print(f"      🔗 Found {len(links)} links, processing...")
                        
                        links_added = 0
                        for link in links:
                            try:
                                # Convert relative URLs to absolute
                                absolute_url = urljoin(current_url, link)
                                parsed = urlparse(absolute_url)
                                
                                # Normalize URL: remove trailing slash, lowercase domain
                                absolute_url = absolute_url.rstrip('/')
                                normalized_domain = parsed.netloc.lower()
                                base_domain_normalized = base_domain.lower()
                                
                                # Only crawl same domain, skip fragments and non-http(s) URLs
                                # Also skip common non-content URLs
                                skip_extensions = {'.pdf', '.jpg', '.jpeg', '.png', '.gif', '.svg', '.css', '.js', '.zip', '.doc', '.docx', '.xls', '.xlsx'}
                                url_path_lower = parsed.path.lower()
                                
                                if (normalized_domain == base_domain_normalized and 
                                    absolute_url not in visited_urls and
                                    parsed.scheme in ('http', 'https') and
                                    not parsed.fragment and
                                    not any(url_path_lower.endswith(ext) for ext in skip_extensions) and
                                    'javascript:' not in absolute_url.lower() and
                                    'mailto:' not in absolute_url.lower()):
                                    if absolute_url not in [u[0] for u in urls_to_crawl]:
                                        urls_to_crawl.append((absolute_url, depth + 1))
                                        links_added += 1
                            except Exception as e:
                                # Skip malformed URLs
                                continue
                        
                        if links_added > 0:
                            print(f"      ✅ Added {links_added} new URL(s) to crawl queue")
                        elif links:
                            print(f"      ⚠️  Found {len(links)} links but none were added (filtered or duplicates)")
                
                elif not result.success:
                    print(f"      ❌ Failed: {result.error_message if hasattr(result, 'error_message') else 'Unknown error'}")
            
            except Exception as e:
                print(f"      ❌ Error crawling {current_url}: {e}")
                continue
    
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
    """Delete all files in a collection directory."""
    collection_dir = Path(DEFAULT_DOCUMENTS_DIR) / collection_name
    
    if not collection_dir.exists():
        print(f"❌ Collection '{collection_name}' does not exist at {collection_dir}")
        return False
    
    if not collection_dir.is_dir():
        print(f"❌ '{collection_dir}' is not a directory")
        return False
    
    # Confirm deletion
    try:
        import shutil
        file_count = len(list(collection_dir.glob("*")))
        print(f"🗑️  Collection '{collection_name}' contains {file_count} file(s)")
        
        # Delete all files in the directory
        deleted_count = 0
        for file_path in collection_dir.glob("*"):
            if file_path.is_file():
                file_path.unlink()
                deleted_count += 1
            elif file_path.is_dir():
                shutil.rmtree(file_path)
                deleted_count += 1
        
        print(f"✅ Deleted {deleted_count} item(s) from collection '{collection_name}'")
        print(f"   Directory '{collection_dir}' is now empty (but kept for future use)")
        return True
        
    except Exception as e:
        print(f"❌ Error deleting collection: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Scrape company website and save markdown files to collection, or delete collection content"
    )
    parser.add_argument(
        "--collection",
        type=str,
        required=True,
        help="Collection name (directory name where crawl results are stored)"
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
    collection_name = sanitize_filename(args.collection)
    
    # Create collection directory
    collection_dir = Path(DEFAULT_DOCUMENTS_DIR) / collection_name
    
    # Crawl website
    try:
        pages = await crawl_website(website_url, collection_dir)
        
        if not pages:
            print("❌ No pages crawled successfully")
            return
        
        # Save markdown files
        save_markdown_files(pages, collection_dir)
        
        print(f"\n✅ Successfully created collection '{collection_name}' with {len(pages)} markdown file(s)")
        print(f"   Location: {collection_dir}")
        print(f"\n💡 Next steps:")
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


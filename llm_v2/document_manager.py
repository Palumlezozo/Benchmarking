#!/usr/bin/env python3
"""
Document Manager - Manage document collections with pluggable backends

A flexible script that manages document collections with support for multiple backends:
- RAG (default): Uses LlamaIndex + Qdrant (default) or Chroma for local RAG with LlamaParse
- OpenAI: Uses OpenAI's file search tool and vector stores (with --openai-tools flag)

Features:
- Multiple backend support (RAG, OpenAI)
- Vector store selection: Qdrant (default) or ChromaDB (--chroma flag)
- Document upload and indexing
- Query capabilities with semantic search
- Collection management
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Add parent directory to path for imports when running as script
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "llm_v2"

from dotenv import load_dotenv

try:
    from .document_store_factory import create_document_store, OPENAI_BACKEND, RAG_BACKEND
    from .openai_document_store import OpenAIDocumentStore
    from .config import (
        add_common_arguments, parse_model_from_args,
        DEFAULT_MODEL, DEFAULT_REASONING_EFFORT, DEFAULT_TEXT_VERBOSITY
    )
except ImportError:
    from document_store_factory import create_document_store, OPENAI_BACKEND, RAG_BACKEND
    from openai_document_store import OpenAIDocumentStore
    from config import (
        add_common_arguments, parse_model_from_args,
        DEFAULT_MODEL, DEFAULT_REASONING_EFFORT, DEFAULT_TEXT_VERBOSITY
    )

# Load environment variables
load_dotenv()

# Constants
DEFAULT_COLLECTION = "collection"


def main():
    """Main function for the Document Manager."""
    parser = argparse.ArgumentParser(
        description="Document Manager - Manage document collections (uses RAG backend with Qdrant by default)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Update documents for default collection (uses RAG backend with Qdrant and LlamaParse)
  python document_manager.py --update
  
  # Update documents for specific collection (RAG backend)
  python document_manager.py --collection "my_docs" --update
  python document_manager.py --collections "my_docs" --update  # Both work!
  
  # Query documents in default collection (RAG backend)
  python document_manager.py --query "What are the main products?"
  
  # Query documents in specific collection
  python document_manager.py --collection "my_docs" --query "What are the main products?"
  
  # Use Chroma instead of Qdrant (default)
  python document_manager.py --chroma --collection "my_docs" --update
  
  # Use OpenAI backend instead of RAG
  python document_manager.py --openai-tools --collection "my_docs" --update
  python document_manager.py --openai-tools --query "What are the main products?"
  
  # List all collections
  python document_manager.py --list-collections
  python document_manager.py --openai-tools --list-collections
  
  # Show document store info for a collection
  python document_manager.py --collection "my_docs" --info
  
  # Delete all documents in a collection (with confirmation)
  python document_manager.py --collection "my_docs" --delete
        """
    )
    
    parser.add_argument(
        "--collection", "--collections",
        dest="collection",
        default=DEFAULT_COLLECTION, 
        help=f"Collection name (folder in data/documents/) (default: '{DEFAULT_COLLECTION}')"
    )
    
    parser.add_argument(
        "--openai-tools",
        action="store_true",
        help="Use OpenAI backend instead of RAG (default: RAG backend)"
    )
    
    # Add common configuration arguments
    add_common_arguments(parser, include_similarity=False, include_embedding_model=True)
    
    # Actions
    parser.add_argument("--update", action="store_true", help="Update documents for the collection")
    parser.add_argument("--query", help="Query the collection's documents")
    parser.add_argument("--list-collections", action="store_true", help="List all collections with document stores")
    parser.add_argument("--info", action="store_true", help="Show information about the collection (or enable verbose logging when combined with --update)")
    parser.add_argument("--delete", action="store_true", help="Delete all documents for the collection")
    
    # Query options (for RAG backend)
    parser.add_argument("--chunks", action="store_true", help="Show retrieved chunks preview (RAG backend only)")
    parser.add_argument("--store-md", action="store_true", help="Store markdown outputs from LlamaParse in data/rag/markdowns/ (RAG backend only)")
    parser.add_argument("--no-llama-parse", action="store_true", help="Use classical LlamaIndex parsing instead of LlamaParse (RAG backend only)")
    parser.add_argument("--chroma", action="store_true", help="Use Chroma instead of Qdrant (default) for vector store (RAG backend only)")
    
    parser.add_argument("--force", action="store_true", help="Skip confirmation prompts")
    
    args = parser.parse_args()
    
    # Determine backend based on flags (openai-tools > rag)
    if args.openai_tools:
        backend = OPENAI_BACKEND
    else:
        backend = RAG_BACKEND
    
    # Configure logging based on --info flag
    if args.info:
        # Show INFO level logs (API calls, HTTP requests, etc.)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    else:
        # Only show WARNING and above (suppress INFO and DEBUG)
        logging.basicConfig(
            level=logging.WARNING,
            format='%(levelname)s: %(message)s'
        )
        # Suppress specific noisy loggers
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("chromadb").setLevel(logging.WARNING)
        logging.getLogger("llama_index").setLevel(logging.WARNING)
    
    # Parse model configuration from arguments
    try:
        model, reasoning_effort, text_verbosity = parse_model_from_args(args)
    except Exception as e:
        print(f"❌ Error parsing model configuration: {e}")
        sys.exit(1)
    
    # Handle list-collections separately (needs to list all, not just one collection)
    if args.list_collections:
        if backend == OPENAI_BACKEND:
            collections = OpenAIDocumentStore.list_all_collections()
            if collections:
                print(f"📋 Collections with document stores (OpenAI backend):")
                for collection in collections:
                    # Create temporary store to get info
                    store = create_document_store(
                        backend=backend,
                        collection=collection,
                        model=model,
                        reasoning_effort=reasoning_effort,
                        text_verbosity=text_verbosity,
                        embedding_model=args.embedding_model
                    )
                    info = store.get_info()
                    doc_count = info.get("total_documents", 0) if info else 0
                    print(f"  - {collection} ({doc_count} documents)")
            else:
                print(f"📋 No collections with document stores found (OpenAI backend)")
        else:
            # RAG backend - read state files directly without initializing stores
            try:
                from .rag_client import RAGDocumentStore
            except ImportError:
                from rag_client import RAGDocumentStore
            import json
            
            collections = RAGDocumentStore.list_all_collections()
            if collections:
                print(f"📋 Collections with document stores (RAG backend):")
                for collection in collections:
                    # Read state file directly to get document count
                    # Try both vector_stores and chroma_stores for backward compatibility
                    state_file = Path("data/rag/vector_stores") / collection / "rag_state.json"
                    if not state_file.exists():
                        state_file = Path("data/rag/chroma_stores") / collection / "rag_state.json"
                    doc_count = 0
                    try:
                        if state_file.exists():
                            with open(state_file, 'r', encoding='utf-8') as f:
                                state = json.load(f)
                                # Count indexed documents from state
                                documents = state.get("documents", {})
                                doc_count = len([doc for doc in documents.values() if doc.get("indexed", False)])
                    except Exception:
                        pass
                    print(f"  - {collection} ({doc_count} documents)")
            else:
                print(f"📋 No collections with document stores found (RAG backend)")
        return
    
    # Initialize the document store using factory
    try:
        if backend == OPENAI_BACKEND:
            backend_name = "OpenAI"
        else:
            backend_name = "RAG"
        
        # Show which API provider is being used
        import os
        azure_base_url = os.getenv("AZURE_OPENAI_BASE_URL")
        if backend == RAG_BACKEND and azure_base_url:
            provider_info = " (Azure OpenAI)"
        else:
            provider_info = ""
        
        print(f"🔧 Using {backend_name} backend{provider_info} for collection '{args.collection}'")
        
        # Build store kwargs
        store_kwargs = {
            "backend": backend,
            "collection": args.collection,
            "model": model,
            "reasoning_effort": reasoning_effort,
            "text_verbosity": text_verbosity,
            "embedding_model": args.embedding_model,
        }
        
        # Add backend-specific parameters
        if backend == RAG_BACKEND:
            store_kwargs["store_md"] = args.store_md
            store_kwargs["use_llama_parse"] = (not args.no_llama_parse)
            store_kwargs["vector_store_type"] = "chroma" if args.chroma else "qdrant"
        
        store = create_document_store(**store_kwargs)
    except Exception as e:
        print(f"❌ Error initializing document store: {e}")
        sys.exit(1)
    
    # Execute requested action
    # If --info is specified without other actions, show info and exit
    if args.info and not any([args.update, args.query, args.delete]):
        info = store.get_info()
        if info:
            if backend == OPENAI_BACKEND:
                backend_name = "OpenAI"
            else:
                backend_name = "RAG"
            print(f"📊 Document store info for collection '{args.collection}':")
            print(f"  Backend: {backend_name}")
            print(f"  Total documents: {info.get('total_documents', 0)}")
            print(f"  Last Updated: {info.get('last_updated', 'Never')}")
            if 'vector_store_id' in info:
                print(f"  Vector Store ID: {info.get('vector_store_id')}")
            if 'storage_dir' in info:
                print(f"  Storage Directory: {info.get('storage_dir')}")
        else:
            print(f"❌ No document store found for collection '{args.collection}'")
        return
    
    if args.delete:
        # Handle confirmation for delete (unless --force is used)
        if not args.force:
            try:
                confirm = input(f"\n⚠️  Are you sure you want to delete all documents in collection '{args.collection}'? (yes/no): ")
                if confirm.lower() != 'yes':
                    print("❌ Deletion cancelled")
                    return
            except EOFError:
                print("❌ Cannot get confirmation in non-interactive mode. Use --force to skip confirmation.")
                return
        
        print(f"\n🗑️  Deleting documents for collection '{args.collection}'...")
        success = store.delete_documents()
        if success:
            print(f"✅ Successfully deleted documents for collection '{args.collection}'")
        else:
            print(f"❌ Failed to delete documents for collection '{args.collection}'")
            sys.exit(1)
        return
    
    if args.update:
        update_start = time.time()
        success = store.update_documents()
        update_elapsed = time.time() - update_start
        
        if not success:
            print(f"❌ Update failed for collection '{args.collection}' (took {update_elapsed:.1f}s)")
            sys.exit(1)
    
    if args.query:
        query_start = time.time()
        
        response = store.query_documents(args.query)
        
        query_elapsed = time.time() - query_start
        
        if response:
            print(f"\n📝 Response:\n{response}")
        else:
            print(f"❌ Query failed (took {query_elapsed:.2f}s)")
            sys.exit(1)
    
    if not any([args.update, args.query, args.info, args.list_collections, args.delete]):
        print("❌ No action specified. Use --help for available options.")
        parser.print_help()


if __name__ == "__main__":
    main()

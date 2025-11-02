"""
RAG Document Store using LlamaIndex and Chroma.

This module provides a document store implementation using LlamaIndex for indexing
and Chroma as the vector store backend.
"""

import json
import os
import warnings
import time
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime as dt

from openai import OpenAI
from dotenv import load_dotenv

# Suppress Pydantic warnings from LlamaIndex
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.readers.file import PDFReader
import chromadb
import re

# LlamaParse imports
try:
    from llama_parse import LlamaParse
    LLAMA_PARSE_AVAILABLE = True
except ImportError:
    LLAMA_PARSE_AVAILABLE = False
    LlamaParse = None

# Cohere Rerank imports
try:
    from llama_index.postprocessor.cohere_rerank import CohereRerank
    COHERE_RERANK_AVAILABLE = True
except ImportError:
    COHERE_RERANK_AVAILABLE = False
    CohereRerank = None

from document_store_base import DocumentStore
from config import (
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DELAY_SECONDS,
    MAX_NODES_PER_BATCH,
    DEFAULT_MODEL,
    DEFAULT_REASONING_EFFORT,
    DEFAULT_TEXT_VERBOSITY,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_TOP_K,
    USE_LLAMA_PARSE,
    LLAMA_PARSE_RESULT_TYPE,
    LLAMA_PARSE_VERBOSE,
    LLAMA_PARSE_LANGUAGE,
    LLAMA_PARSE_PARSE_MODE,
    LLAMA_PARSE_INVALIDATE_CACHE,
    LLAMA_PARSE_DO_NOT_CACHE,
    LLAMA_PARSE_NUM_WORKERS,
    LLAMA_PARSE_SKIP_DIAGONAL_TEXT,
    LLAMA_PARSE_SPREADSHEET_EXTRACT_SUB_TABLES,
    LLAMA_PARSE_SPREADSHEET_FORCE_FORMULA_COMPUTATION,
    LLAMA_PARSE_PARTITION_PAGES,
    USE_COHERE_RERANK,
    COHERE_RERANK_MODEL,
    COHERE_RERANK_TOP_N
)

# Load environment variables
load_dotenv()

# ============================================================================
# RAG Configuration Constants
# ============================================================================

# Storage paths
DEFAULT_DOCUMENTS_DIR = "data/documents"
DEFAULT_STORAGE_DIR = "data/rag/chroma_stores"
DEFAULT_MARKDOWNS_DIR = "data/rag/markdowns"

# File types supported by LlamaParse (that may contain page markers)
LLAMAPARSE_SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.doc', '.pptx', '.ppt', '.xlsx', '.xls', '.html', '.htm'}

# Note: DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, and DEFAULT_TOP_K are imported from config.py
# ============================================================================


class RAGDocumentStore(DocumentStore):
    """Document store implementation using LlamaIndex + Chroma."""
    
    def __init__(
        self, 
        collection: str,
        documents_dir: str = DEFAULT_DOCUMENTS_DIR,
        storage_dir: str = DEFAULT_STORAGE_DIR,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        model: str = DEFAULT_MODEL,
        reasoning_effort: str = DEFAULT_REASONING_EFFORT,
        text_verbosity: str = DEFAULT_TEXT_VERBOSITY,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        store_md: bool = False,
        use_llama_parse: bool = None
    ):
        """
        Initialize the RAG document store.
        
        Args:
            collection: Collection name
            documents_dir: Base directory for documents
            storage_dir: Directory for Chroma storage
            embedding_model: OpenAI embedding model to use
            model: LLM model for queries
            reasoning_effort: Reasoning effort level
            text_verbosity: Text verbosity level
            chunk_size: Size of text chunks for indexing
            chunk_overlap: Overlap between chunks
            store_md: Whether to store markdown outputs from LlamaParse
            use_llama_parse: Override config to force LlamaParse on/off (None = use config setting)
        """
        super().__init__(collection)
        self.documents_dir = Path(documents_dir) / collection
        self.storage_dir = Path(storage_dir) / collection
        self.markdowns_dir = Path(DEFAULT_MARKDOWNS_DIR) / collection
        self.embedding_model = embedding_model
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.text_verbosity = text_verbosity
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.store_md = store_md
        # Override config setting if explicitly specified
        self.use_llama_parse_override = use_llama_parse
        
        # Create storage directory if it doesn't exist
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        # Create markdowns directory if markdown storage is enabled
        if self.store_md:
            self.markdowns_dir.mkdir(parents=True, exist_ok=True)
        
        # State tracking (separate from OpenAI state)
        self.state_file = self.storage_dir / "rag_state.json"
        self.state = self._load_state()
        
        # Initialize OpenAI client for queries (using v2 API)
        # Use Azure OpenAI if configured, otherwise use standard OpenAI
        azure_base_url = os.getenv("AZURE_OPENAI_BASE_URL")
        azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        
        if azure_base_url and azure_api_key:
            self.client = OpenAI(
                api_key=azure_api_key,
                base_url=azure_base_url
            )
            # Configure LlamaIndex settings with Azure
            # Configure embedding with batch size to control rate limiting
            Settings.embed_model = OpenAIEmbedding(
                model=embedding_model,
                api_key=azure_api_key,
                api_base=azure_base_url,
                embed_batch_size=EMBEDDING_BATCH_SIZE  # Batch embeddings to avoid rate limits
            )
        else:
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            # Configure LlamaIndex settings with standard OpenAI
            Settings.embed_model = OpenAIEmbedding(
                model=embedding_model,
                api_key=os.getenv("OPENAI_API_KEY"),
                embed_batch_size=EMBEDDING_BATCH_SIZE  # Batch embeddings to avoid rate limits
            )
        Settings.chunk_size = chunk_size
        Settings.chunk_overlap = chunk_overlap
        
        # Initialize Chroma client
        self.chroma_client = chromadb.PersistentClient(path=str(self.storage_dir))
        
        # Vector store and index (will be initialized in update_documents)
        self.index = None
        self.retriever = None
        
        print(f"📚 RAG Document Store initialized for collection '{collection}'")
        print(f"   Documents: {self.documents_dir}")
        print(f"   Storage: {self.storage_dir}")
        if self.store_md:
            print(f"   Markdowns: {self.markdowns_dir}")
        print(f"   Embedding Model: {embedding_model}")
        
        # Store LlamaParse configuration (will create parser dynamically per batch)
        self.llama_parse_config = self._get_llama_parse_config()
        # For backward compatibility, create a default parser
        self.llama_parse_parser = self._create_llama_parse_parser() if self.llama_parse_config else None
        
        # Initialize Cohere reranker if enabled and available
        self.cohere_reranker = self._create_cohere_reranker()
    
    def _get_llama_parse_config(self) -> Optional[Dict[str, Any]]:
        """
        Get LlamaParse configuration if available.
        
        Returns:
            Configuration dict or None if LlamaParse not available
        """
        # Check if LlamaParse is explicitly disabled via override
        if self.use_llama_parse_override is False:
            print("📄 Using classical LlamaIndex parsing (LlamaParse disabled by --no-llama-parse)")
            return None
        
        # Check config setting (if override is not explicitly True)
        if self.use_llama_parse_override is None and not USE_LLAMA_PARSE:
            return None
        
        if not LLAMA_PARSE_AVAILABLE:
            print("⚠️  LlamaParse not installed, falling back to pypdf")
            return None
        
        api_key = os.getenv("LLAMA_CLOUD_API_KEY")
        if not api_key:
            print("⚠️  LLAMA_CLOUD_API_KEY not found, falling back to pypdf")
            return None
        
        # Get optional base URL for European or custom endpoints
        base_url = os.getenv("LLAMA_CLOUD_BASE_URL")
        
        # Build parser kwargs
        config = {
            "api_key": api_key,
            "result_type": LLAMA_PARSE_RESULT_TYPE,
            "verbose": LLAMA_PARSE_VERBOSE,
            "language": LLAMA_PARSE_LANGUAGE,
            "parse_mode": LLAMA_PARSE_PARSE_MODE,
            "invalidate_cache": LLAMA_PARSE_INVALIDATE_CACHE,
            "do_not_cache": LLAMA_PARSE_DO_NOT_CACHE,
            "num_workers": LLAMA_PARSE_NUM_WORKERS,
            "skip_diagonal_text": LLAMA_PARSE_SKIP_DIAGONAL_TEXT,
            "spreadsheet_extract_sub_tables": LLAMA_PARSE_SPREADSHEET_EXTRACT_SUB_TABLES,
            "spreadsheet_force_formula_computation": LLAMA_PARSE_SPREADSHEET_FORCE_FORMULA_COMPUTATION
        }
        
        # Add partition_pages if specified (for large document processing)
        if LLAMA_PARSE_PARTITION_PAGES is not None:
            config["partition_pages"] = LLAMA_PARSE_PARTITION_PAGES
        
        # Add base_url if specified (for European or custom endpoints)
        if base_url:
            config["base_url"] = base_url
        
        endpoint_info = f" (endpoint: {base_url})" if base_url else ""
        partition_info = f", partition_pages={LLAMA_PARSE_PARTITION_PAGES}" if LLAMA_PARSE_PARTITION_PAGES is not None else ""
        print(f"✅ LlamaParse enabled{endpoint_info} (result_type={LLAMA_PARSE_RESULT_TYPE}, parse_mode={LLAMA_PARSE_PARSE_MODE}, workers={LLAMA_PARSE_NUM_WORKERS}{partition_info})")
        
        return config
    
    def _create_llama_parse_parser(self, num_workers: Optional[int] = None) -> Optional[Any]:
        """
        Create LlamaParse parser with optional custom worker count.
        
        Args:
            num_workers: Number of workers (None = use config default)
            
        Returns:
            LlamaParse parser instance or None if not available
        """
        if not self.llama_parse_config:
            return None
        
        try:
            # Copy config and optionally override num_workers
            parser_kwargs = self.llama_parse_config.copy()
            if num_workers is not None:
                parser_kwargs["num_workers"] = num_workers
            
            parser = LlamaParse(**parser_kwargs)
            return parser
        except Exception as e:
            print(f"⚠️  Failed to initialize LlamaParse: {e}")
            print("   Falling back to pypdf")
            return None
    
    def _create_cohere_reranker(self) -> Optional[Any]:
        """
        Create Cohere reranker if enabled and API key is available.
        
        Returns:
            CohereRerank instance or None if not available
        """
        if not USE_COHERE_RERANK:
            return None
        
        if not COHERE_RERANK_AVAILABLE:
            print("⚠️  Cohere rerank not installed")
            print("   Install with: pip install llama-index-postprocessor-cohere-rerank")
            return None
        
        api_key = os.getenv("COHERE_API_KEY") or os.getenv("CO_API_KEY")
        if not api_key:
            print("⚠️  COHERE_API_KEY not found, reranking disabled")
            print("   Add COHERE_API_KEY to .env to enable reranking")
            return None
        
        try:
            reranker = CohereRerank(
                api_key=api_key,
                model=COHERE_RERANK_MODEL,
                top_n=COHERE_RERANK_TOP_N
            )
            print(f"✅ Cohere Rerank enabled (model={COHERE_RERANK_MODEL}, top_n={COHERE_RERANK_TOP_N})")
            return reranker
        except Exception as e:
            print(f"⚠️  Failed to initialize Cohere reranker: {e}")
            return None
    
    def _load_state(self) -> Dict[str, Any]:
        """Load the state from file."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️  Warning: Could not load state: {e}")
        return {
            "documents": {},
            "last_updated": None,
            "total_documents": 0,
            "indexed_documents": 0,
            "failed_documents": 0,
            "index_created": False
        }
    
    def _save_state(self) -> None:
        """Save the state to file."""
        self.state["last_updated"] = dt.now().isoformat()
        try:
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(self.state, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"❌ Error saving state: {e}")
    
    def _scan_documents(self) -> Dict[str, Dict[str, Any]]:
        """Scan the documents directory for files."""
        documents = {}
        
        if not self.documents_dir.exists():
            print(f"📁 Creating documents directory: {self.documents_dir}")
            self.documents_dir.mkdir(parents=True, exist_ok=True)
            return documents
        
        supported_extensions = {'.txt', '.md', '.docx', '.pdf', '.xlsx', '.xls', '.pptx', '.doc', '.ppt', '.html', '.htm'}
        
        for file_path in self.documents_dir.rglob("*"):
            if (file_path.is_file() and 
                file_path.suffix.lower() in supported_extensions and
                not file_path.name.startswith("._")):
                try:
                    documents[str(file_path)] = {
                        "size": file_path.stat().st_size,
                        "modified": file_path.stat().st_mtime,
                        "extension": file_path.suffix.lower()
                    }
                except OSError as e:
                    print(f"⚠️  Warning: Could not process file {file_path}: {e}")
        
        return documents
    
    def _get_document_status(self, file_path: str, current_metadata: Dict[str, Any]) -> str:
        """
        Get indexing status of a document.
        
        Args:
            file_path: Path to the document
            current_metadata: Current file metadata (size, modified, extension)
        
        Returns:
            'new', 'modified', 'indexed', or 'failed'
        """
        stored_doc = self.state["documents"].get(file_path)
        
        if not stored_doc:
            return 'new'
        
        # Check if file was modified
        if stored_doc.get('modified') != current_metadata['modified']:
            return 'modified'
        
        # Check if indexing failed previously
        if not stored_doc.get('indexed', False):
            return 'failed'
        
        return 'indexed'
    
    def _get_documents_to_process(self) -> List[Path]:
        """
        Get list of documents that need (re)indexing.
        
        Returns:
            List of Path objects for documents that need processing
        """
        current_docs = self._scan_documents()
        to_process = []
        
        for file_path, metadata in current_docs.items():
            status = self._get_document_status(file_path, metadata)
            if status in ['new', 'modified', 'failed']:
                to_process.append((Path(file_path), status))
        
        return to_process
    
    def _get_deleted_documents(self) -> List[str]:
        """
        Get list of documents that were deleted from disk.
        
        Returns:
            List of file paths that are in state but no longer on disk
        """
        current_docs = self._scan_documents()
        stored_docs = self.state.get("documents", {})
        
        deleted = []
        for file_path in stored_docs.keys():
            if file_path not in current_docs:
                deleted.append(file_path)
        
        return deleted
    
    def _update_state_counters(self):
        """Update aggregate counters in state."""
        documents = self.state.get("documents", {})
        
        total = len(documents)
        indexed = sum(1 for doc in documents.values() if doc.get("indexed", False))
        failed = sum(1 for doc in documents.values() if not doc.get("indexed", False))
        
        self.state["total_documents"] = total
        self.state["indexed_documents"] = indexed
        self.state["failed_documents"] = failed
    
    def _mark_document_indexed(self, doc_path: Path, chunk_count: int):
        """
        Mark document as successfully indexed and save state.
        
        Args:
            doc_path: Path to the document
            chunk_count: Number of chunks created from this document
        """
        file_path_str = str(doc_path)
        
        self.state["documents"][file_path_str] = {
            "size": doc_path.stat().st_size,
            "modified": doc_path.stat().st_mtime,
            "extension": doc_path.suffix.lower(),
            "indexed": True,
            "indexed_at": dt.now().isoformat(),
            "chunk_count": chunk_count
        }
        
        # Update counters
        self._update_state_counters()
        
        # Save state immediately (progressive checkpointing)
        self._save_state()
    
    def _mark_document_failed(self, doc_path: Path, error: str):
        """
        Mark document as failed and save state.
        
        Args:
            doc_path: Path to the document
            error: Error message
        """
        file_path_str = str(doc_path)
        
        self.state["documents"][file_path_str] = {
            "size": doc_path.stat().st_size,
            "modified": doc_path.stat().st_mtime,
            "extension": doc_path.suffix.lower(),
            "indexed": False,
            "indexed_at": None,
            "error": error,
            "failed_at": dt.now().isoformat()
        }
        
        # Update counters
        self._update_state_counters()
        
        # Save state immediately
        self._save_state()
    
    def _remove_document_from_state(self, file_path: str):
        """
        Remove document from state (for deleted files).
        
        Args:
            file_path: Path to the document (as string)
        """
        if file_path in self.state["documents"]:
            del self.state["documents"][file_path]
            self._update_state_counters()
            self._save_state()
    
    def _needs_reindex(self) -> bool:
        """Check if the index needs to be rebuilt."""
        current_docs = self._scan_documents()
        stored_docs = self.state.get("documents", {})
        
        # Check if any documents changed
        for file_path, metadata in current_docs.items():
            stored_metadata = stored_docs.get(file_path)
            if not stored_metadata or stored_metadata.get("modified") != metadata["modified"]:
                return True
        
        # Check if any documents were deleted
        for file_path in stored_docs.keys():
            if file_path not in current_docs:
                return True
        
        return not self.state.get("index_created", False)
    
    def _extract_page_numbers_from_text(self, text: str) -> List[str]:
        """
        Extract all page numbers from text containing LlamaParse page markers.
        
        Args:
            text: Text potentially containing page markers
            
        Returns:
            List of page numbers found in the text (as strings)
        """
        # Pattern: New format <!-- PAGE: N -->
        page_pattern = r'<!--\s*PAGE:\s*(\d+)\s*-->'
        
        # Find all page markers
        matches = re.findall(page_pattern, text, re.IGNORECASE)
        
        return matches
    
    def _remove_page_markers_from_text(self, text: str) -> str:
        """
        Remove page markers from text to clean it up for indexing.
        
        Args:
            text: Text with page markers
            
        Returns:
            Text with page markers removed
        """
        # Remove page markers <!-- PAGE: N -->
        page_pattern = r'\n*<!--\s*PAGE:\s*\d+\s*-->\n*'
        text = re.sub(page_pattern, '\n\n', text, flags=re.IGNORECASE)
        
        return text
    
    def _enrich_documents_with_page_metadata(self, documents: List[Any]) -> List[Document]:
        """
        Enrich documents with page metadata extracted from LlamaParse markdown.
        
        This method keeps the full document intact (better for semantic chunking),
        but adds metadata about which pages are contained in the document.
        The actual chunking will happen later, and chunks that span multiple pages
        will have accurate page range information.
        
        Works with all LlamaParse-supported formats: PDF, DOCX, DOC, PPTX, PPT
        
        Args:
            documents: List of Document objects from LlamaParse
            
        Returns:
            List of Document objects with page metadata
        """
        enriched_documents = []
        files_with_markers = set()  # Track unique source files with markers
        files_without_markers = set()  # Track unique source files without markers
        
        print(f"📑 Checking for page markers in {len(documents)} document(s)...")
        
        for i, doc in enumerate(documents):
            # Only process if we're using LlamaParse
            if not self.llama_parse_parser:
                enriched_documents.append(doc)
                continue
            
            # Check if this is a LlamaParse-supported file type
            file_path = doc.metadata.get('file_path') or doc.metadata.get('file_name', '')
            
            # DEBUG: Print metadata to see what's available
            #if i == 0:  # Only print for first document
            #    print(f"   DEBUG: Document metadata keys: {list(doc.metadata.keys())}")
            #    print(f"   DEBUG: file_path={file_path}")
            #    print(f"   DEBUG: Total documents from LlamaParse: {len(documents)}")
            
            file_ext = Path(str(file_path)).suffix.lower() if file_path else ''
            
            if file_ext not in LLAMAPARSE_SUPPORTED_EXTENSIONS:
                # Not a LlamaParse format, skip page extraction
                enriched_documents.append(doc)
                continue
            
            text = doc.text
            
            # Extract page numbers from the document
            page_numbers = self._extract_page_numbers_from_text(text)
            
            if not page_numbers:
                # No page markers found in text - keep original document
                files_without_markers.add(file_path)
                if i < 3:  # Only show first few
                    print(f"   ⚠️  No page markers found in document {i+1} from {Path(file_path).name}")
                    # DEBUG: Show beginning of text to check if markers are there
                    text_preview = text[:500].replace('\n', '\\n')
                    print(f"   DEBUG: Text preview: {text_preview[:200]}...")
                enriched_documents.append(doc)
                continue
            
            # Just mark that this document has page markers
            # Page extraction will happen after chunking
            doc.metadata['_has_page_markers'] = True
            enriched_documents.append(doc)
            files_with_markers.add(file_path)
        
        # Summary - count unique source files, not Document objects
        total_source_files = len(files_with_markers | files_without_markers)
        if files_with_markers:
            print(f"   ✅ {len(files_with_markers)}/{total_source_files} source files have page markers ({len(documents)} total document objects)")
        else:
            print(f"   ⚠️  No page markers found in any documents")
        
        return enriched_documents
    
    def _post_process_nodes_with_page_ranges(self, nodes: List[Any]) -> List[Any]:
        """
        Post-process chunks to extract page numbers from embedded markers.
        
        Each chunk may contain page markers embedded by LlamaParse:
        - <!-- PAGE: N -->
        
        This method:
        1. Extracts page numbers from each chunk's text
        2. Sets page_label metadata (single page or range)
        3. Removes page markers from the text
        
        Args:
            nodes: List of chunks from the node parser
            
        Returns:
            List of chunks with page_label metadata and clean text
        """
        for node in nodes:
            # Skip if this node didn't come from a document with page markers
            if not node.metadata.get('_has_page_markers'):
                continue
            
            # Extract page numbers from this chunk's text
            page_numbers = self._extract_page_numbers_from_text(node.text)
            
            if page_numbers:
                # Set page_label metadata
                if len(page_numbers) == 1:
                    node.metadata['page_label'] = page_numbers[0]
                else:
                    # Chunk spans multiple pages - show as range
                    node.metadata['page_label'] = f"{page_numbers[0]}-{page_numbers[-1]}"
                
                # Remove page markers from text
                node.text = self._remove_page_markers_from_text(node.text)
            
            # Clean up internal marker
            node.metadata.pop('_has_page_markers', None)
        
        return nodes
    
    def _save_markdowns(self, documents: List[Document]) -> None:
        """
        Save the markdown content of documents to the markdowns directory.
        
        This function saves the raw document text as-is, including HTML comment
        page markers (<!-- PAGE: N -->) if present. The markdown is saved BEFORE
        any chunking or text processing, so all page markers are preserved.
        
        Args:
            documents: List of Document objects from LlamaParse
        """
        if not self.store_md:
            # Only save markdowns if store_md is enabled
            return
        
        if not self.llama_parse_parser:
            # Only save markdowns if LlamaParse was used
            return
        
        print(f"💾 Saving markdown files to {self.markdowns_dir}...")
        saved_count = 0
        
        for doc in documents:
            try:
                # Get the original document file path
                file_path = doc.metadata.get('file_path') or doc.metadata.get('file_name', '')
                if not file_path:
                    continue
                
                # Check if this is a LlamaParse-supported file type
                file_ext = Path(str(file_path)).suffix.lower()
                if file_ext not in LLAMAPARSE_SUPPORTED_EXTENSIONS:
                    continue
                
                # Get the base filename without extension
                file_name = Path(str(file_path)).stem
                
                # Create markdown filename
                markdown_filename = f"{file_name}.md"
                markdown_path = self.markdowns_dir / markdown_filename
                
                # Save markdown content as-is (includes page markers <!-- PAGE: N -->)
                # This is called BEFORE chunking, so page markers are preserved
                with open(markdown_path, 'w', encoding='utf-8') as f:
                    f.write(doc.text)
                
                saved_count += 1
                print(f"   ✅ Saved {markdown_filename}")
                
            except Exception as e:
                print(f"   ⚠️  Failed to save markdown for {file_path}: {e}")
        
        if saved_count > 0:
            print(f"✅ Saved {saved_count} markdown file(s)")
    
    def _load_single_document(self, doc_path: Path) -> List[Document]:
        """
        Load a single document using LlamaParse (async) or LlamaIndex (sync).
        
        Args:
            doc_path: Path to the document
            
        Returns:
            List of Document objects
        """
        file_path_str = str(doc_path)
        
        # Try LlamaParse parsing first
        if self.llama_parse_config and doc_path.suffix.lower() in LLAMAPARSE_SUPPORTED_EXTENSIONS:
            try:
                parser = self._create_llama_parse_parser(num_workers=1)
                
                async def parse_file():
                    # Use async aparse - returns JobResult(s) quickly
                    # When partition_pages is enabled, can return multiple JobResults (one per partition)
                    result = await parser.aparse(file_path_str)
                    
                    if result is None:
                        return []
                    
                    # Handle both single JobResult and list of JobResults (when partitioning)
                    job_results = result if isinstance(result, list) else [result]
                    
                    # Collect all pages from all JobResults (partitions)
                    all_pages = []
                    for job_result in job_results:
                        if hasattr(job_result, 'pages') and job_result.pages:
                            all_pages.extend(job_result.pages)
                    
                    # Sort pages by page number to ensure correct ordering across partitions
                    all_pages.sort(key=lambda p: p.page if hasattr(p, 'page') and p.page is not None else 0)
                    
                    # Reconstruct combined markdown from all pages
                    if all_pages:
                        combined_markdown_parts = []
                        for page in all_pages:
                            page_separator = f"<!-- PAGE: {page.page} -->\n"
                            combined_markdown_parts.append(page_separator)
                            # Handle None md (can happen in without_llm mode) - use text as fallback
                            page_content = page.md if page.md is not None else (page.text if hasattr(page, 'text') and page.text else "")
                            combined_markdown_parts.append(page_content)
                            combined_markdown_parts.append("\n\n")
                        
                        combined_markdown = "".join(combined_markdown_parts)
                        
                        # Create Document object with reconstructed markdown
                        doc = Document(
                            text=combined_markdown,
                            metadata={
                                'file_path': file_path_str,
                                'file_name': doc_path.name
                            }
                        )
                        return [doc]
                    
                    return []
                
                try:
                    documents = asyncio.run(parse_file())
                except RuntimeError:
                    # Already in event loop
                    loop = asyncio.get_event_loop()
                    documents = loop.run_until_complete(parse_file())
                
                return documents if documents else []
                
            except Exception as e:
                print(f"   ⚠️  Parsing failed, falling back to sync: {e}")
                # Fall through to sync parsing
        
        # Fall back to synchronous parsing
        if self.llama_parse_parser:
            file_extractor = {
                ext: self.llama_parse_parser 
                for ext in LLAMAPARSE_SUPPORTED_EXTENSIONS
            }
        else:
            file_extractor = {".pdf": PDFReader()}
        
        reader = SimpleDirectoryReader(
            input_files=[file_path_str],
            file_extractor=file_extractor
        )
        
        return reader.load_data()
    
    def _add_document_to_index(self, doc_path: Path, status: str) -> bool:
        """
        Add a single document to the existing index.
        
        Args:
            doc_path: Path to the document
            status: Document status ('new', 'modified', or 'failed')
            
        Returns:
            True if successful, False if failed
        """
        try:
            status_emoji = "📥" if status == "new" else "🔄" if status == "modified" else "🔁"
            print(f"{status_emoji} Processing {doc_path.name} ({status})...")
            
            # Load single document
            documents = self._load_single_document(doc_path)
            
            if not documents:
                self._mark_document_failed(doc_path, "No content extracted")
                return False
            
            # Save markdown if enabled
            if self.store_md:
                self._save_markdowns(documents)
            
            # Enrich with page metadata
            if self.llama_parse_parser:
                documents = self._enrich_documents_with_page_metadata(documents)
            
            # Create nodes from document
            node_parser = SentenceSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            nodes = node_parser.get_nodes_from_documents(documents)
            
            # Post-process for page ranges
            if self.llama_parse_parser:
                nodes = self._post_process_nodes_with_page_ranges(nodes)
            
            # Add nodes to existing index (or create if first document) with rate limiting
            create_new = not self.index
            self._insert_nodes_with_rate_limiting(nodes, create_new_index=create_new)
            
            # Update state for this document
            self._mark_document_indexed(doc_path, len(nodes))
            
            print(f"✅ Indexed {doc_path.name} ({len(nodes)} chunks)")
            return True
            
        except Exception as e:
            print(f"❌ Failed to index {doc_path.name}: {e}")
            import traceback
            traceback.print_exc()
            self._mark_document_failed(doc_path, str(e))
            return False
    
    def _process_documents_parallel(self, documents: List[Tuple[Path, str]]) -> Tuple[List[Any], List[Tuple[Path, int]], List[Tuple[Path, str]]]:
        """
        Process all documents in parallel and return nodes.
        
        Uses asyncio.gather() to parse multiple files concurrently for maximum performance.
        All nodes are collected together for efficient batch embedding insertion.
        
        Args:
            documents: List of (doc_path, status) tuples to process
            
        Returns:
            Tuple of (all_nodes, successful_docs, failed_docs)
            - all_nodes: All nodes from successfully parsed documents
            - successful_docs: List of (doc_path, chunk_count) tuples
            - failed_docs: List of (doc_path, error_message) tuples
        """
        all_nodes = []
        successful = []
        failed = []
        
        if not documents:
            return all_nodes, successful, failed
        
        # Group by file type for processing
        if self.llama_parse_config:
            llama_parse_docs = [
                (doc_path, status) for doc_path, status in documents
                if doc_path.suffix.lower() in LLAMAPARSE_SUPPORTED_EXTENSIONS
            ]
            other_docs = [
                (doc_path, status) for doc_path, status in documents
                if doc_path.suffix.lower() not in LLAMAPARSE_SUPPORTED_EXTENSIONS
            ]
            
            # Process LlamaParse-supported documents using batch API calls
            if llama_parse_docs:
                try:
                    # Create parser with configured number of workers
                    parser = self._create_llama_parse_parser(num_workers=LLAMA_PARSE_NUM_WORKERS)
                    
                    # Split documents into batches
                    # When partition_pages is enabled, process files one at a time to avoid
                    # confusion with multiple JobResults per file
                    if LLAMA_PARSE_PARTITION_PAGES is not None:
                        # Partitioning enabled: process one file per batch to reliably handle multiple JobResults
                        batch_size = 1
                        print(f"   ℹ️  Partitioning enabled: processing files one at a time")
                    else:
                        # No partitioning: can batch multiple files (batch size = num_workers)
                        batch_size = LLAMA_PARSE_NUM_WORKERS
                    
                    batches = []
                    for i in range(0, len(llama_parse_docs), batch_size):
                        batch = llama_parse_docs[i:i + batch_size]
                        batches.append(batch)
                    
                    print(f"   📦 Processing {len(llama_parse_docs)} file(s) in {len(batches)} batch(es) of up to {batch_size} file(s) each")
                    
                    async def parse_batch(batch: List[Tuple[Path, str]]):
                        """Parse a batch of files using batch API call (aparse with list)."""
                        try:
                            # Extract file paths from batch
                            file_paths = [str(doc_path) for doc_path, _ in batch]
                            file_info = [(doc_path, status) for doc_path, status in batch]
                            
                            # Use batch API call: aparse([file1, file2, ...])
                            result = await parser.aparse(file_paths)
                            
                            if result is None:
                                print(f"   ⚠️  Batch aparse returned None for {len(batch)} file(s)")
                                return [(doc_path, status, []) for doc_path, status in batch]
                            
                            batch_documents = []
                            
                            # Batch API returns a list of JobResults
                            # When partition_pages is enabled, one file can produce multiple JobResults (one per partition)
                            if isinstance(result, list):
                                # Check if partitioning is enabled (multiple JobResults expected per file)
                                partition_enabled = LLAMA_PARSE_PARTITION_PAGES is not None
                                if not partition_enabled and len(result) != len(batch):
                                    print(f"   ⚠️  Batch returned {len(result)} JobResult(s), expected {len(batch)} (partitioning disabled)")
                                elif partition_enabled:
                                    print(f"   ℹ️  Batch returned {len(result)} JobResult(s) for {len(batch)} file(s) (partitioning enabled)")
                                
                                # Group JobResults by source file
                                # When partitioning is enabled, multiple JobResults belong to one file
                                # The API maintains order: [file1_part1, file1_part2, ..., file2_part1, ...]
                                result_index = 0
                                for file_idx, (doc_path, status) in enumerate(file_info):
                                    if result_index >= len(result):
                                        print(f"   ⚠️  Not enough JobResults for {doc_path.name}")
                                        batch_documents.append((doc_path, status, []))
                                        continue
                                    
                                    try:
                                        # Collect all JobResults for this file
                                        # When partitioning is enabled, batch_size is set to 1, so all JobResults
                                        # in the result list belong to this single file
                                        file_documents = []
                                        all_pages = []
                                        
                                        if partition_enabled:
                                            # Partitioning enabled: all JobResults belong to this file
                                            # Collect all remaining JobResults
                                            file_job_results = result[result_index:]
                                            result_index = len(result)  # Consume all
                                        else:
                                            # No partitioning: one JobResult per file
                                            file_job_results = [result[result_index]] if result_index < len(result) else []
                                            result_index += 1
                                        
                                        # Process all JobResults for this file and combine their pages
                                        for job_result in file_job_results:
                                            if hasattr(job_result, 'pages') and job_result.pages:
                                                all_pages.extend(job_result.pages)
                                        
                                        # Sort pages by page number to ensure correct ordering across partitions
                                        all_pages.sort(key=lambda p: p.page if hasattr(p, 'page') and p.page is not None else 0)
                                        
                                        # Reconstruct combined markdown from all pages
                                        if all_pages:
                                            combined_markdown_parts = []
                                            for page in all_pages:
                                                page_separator = f"<!-- PAGE: {page.page} -->\n"
                                                combined_markdown_parts.append(page_separator)
                                                # Handle None md (can happen in without_llm mode) - use text as fallback
                                                page_content = page.md if page.md is not None else (page.text if hasattr(page, 'text') and page.text else "")
                                                combined_markdown_parts.append(page_content)
                                                combined_markdown_parts.append("\n\n")
                                            
                                            combined_markdown = "".join(combined_markdown_parts)
                                            
                                            # Create Document object with reconstructed markdown
                                            doc = Document(
                                                text=combined_markdown,
                                                metadata={
                                                    'file_path': file_paths[file_idx] if file_idx < len(file_paths) else None,
                                                    'file_name': doc_path.name
                                                }
                                            )
                                            file_documents.append(doc)
                                        
                                        batch_documents.append((doc_path, status, file_documents))
                                        
                                    except Exception as e:
                                        print(f"   ⚠️  Error extracting documents for {doc_path.name}: {e}")
                                        import traceback
                                        traceback.print_exc()
                                        batch_documents.append((doc_path, status, []))
                                
                                # Warn if we didn't process all JobResults
                                if result_index < len(result):
                                    print(f"   ⚠️  Processed {result_index} JobResult(s), but {len(result)} were returned")
                            else:
                                # Unexpected result type
                                print(f"   ⚠️  Unexpected batch result type: {type(result)}")
                                for doc_path, status in batch:
                                    batch_documents.append((doc_path, status, []))
                            
                            return batch_documents
                            
                        except Exception as e:
                            print(f"   ⚠️  Batch parse failed: {e}")
                            import traceback
                            traceback.print_exc()
                            return [(doc_path, status, []) for doc_path, status in batch]
                    
                    # Process all batches
                    async def parse_all_batches():
                        # Process batches sequentially (can be parallelized later if needed)
                        all_results = []
                        for batch in batches:
                            batch_results = await parse_batch(batch)
                            all_results.extend(batch_results)
                        return all_results
                    
                    try:
                        results = asyncio.run(parse_all_batches())
                    except RuntimeError as e:
                        if "asyncio.run() cannot be called from a running event loop" in str(e):
                            loop = asyncio.get_event_loop()
                            results = loop.run_until_complete(parse_all_batches())
                        else:
                            raise
                    
                    # Process results and collect documents with per-file tracking
                    documents_by_file = {}
                    failed_paths = set()
                    successful_parses = 0
                    
                    for doc_path, status, file_docs in results:
                        if file_docs:
                            documents_by_file[doc_path] = (status, file_docs)
                            successful_parses += 1
                        else:
                            print(f"   ⚠️  No documents returned for {doc_path.name}")
                    
                    if documents_by_file:
                        # Collect all documents
                        all_documents = []
                        for doc_path, (status, file_docs) in documents_by_file.items():
                            all_documents.extend(file_docs)
                        
                        print(f"   📊 LlamaParse parsed {successful_parses}/{len(llama_parse_docs)} files, returned {len(all_documents)} document object(s)")
                        
                        # Process nodes from all documents together
                        node_parser = SentenceSplitter(
                            chunk_size=self.chunk_size,
                            chunk_overlap=self.chunk_overlap
                        )
                        
                        # Save markdown if enabled
                        if self.store_md:
                            self._save_markdowns(all_documents)
                        
                        # Enrich with page metadata
                        all_documents = self._enrich_documents_with_page_metadata(all_documents)
                        
                        # Create nodes from all documents
                        nodes = node_parser.get_nodes_from_documents(all_documents)
                        nodes = self._post_process_nodes_with_page_ranges(nodes)
                        
                        # Calculate nodes per document for tracking
                        nodes_per_file = {}
                        if nodes and documents_by_file:
                            # Approximate: divide nodes evenly, or could track more precisely
                            avg_nodes_per_doc = len(nodes) / len(documents_by_file)
                            for doc_path in documents_by_file.keys():
                                nodes_per_file[doc_path] = int(avg_nodes_per_doc)
                        
                        all_nodes.extend(nodes)
                        
                        # Track successful documents with chunk counts
                        for doc_path, (status, _) in documents_by_file.items():
                            chunk_count = nodes_per_file.get(doc_path, 0)
                            successful.append((doc_path, chunk_count))
                        
                        # Track failed documents (only those not already tracked and not in documents_by_file)
                        for doc_path, status in llama_parse_docs:
                            if doc_path not in documents_by_file and doc_path not in failed_paths:
                                failed.append((doc_path, "No content extracted"))
                    else:
                        # All parsing failed
                        print(f"   ⚠️  LlamaParse returned no documents for {len(llama_parse_docs)} file(s)")
                        print(f"      This might be due to parse_mode='{LLAMA_PARSE_PARSE_MODE}'")
                        # Track documents that weren't already marked as failed
                        for doc_path, status in llama_parse_docs:
                            if doc_path not in failed_paths:
                                failed.append((doc_path, "LlamaParse returned no documents"))
                        
                except Exception as e:
                    # If parallel async parsing fails, fall back to processing documents individually
                    print(f"   ⚠️  Parallel async parsing failed, falling back to individual processing: {e}")
                    for doc_path, status in llama_parse_docs:
                        try:
                            # _load_single_document already handles fallback to sync parsing
                            if self._add_document_to_index(doc_path, status):
                                # Chunk count will be updated by _add_document_to_index
                                successful.append((doc_path, 0))
                            else:
                                failed.append((doc_path, "Processing failed"))
                        except Exception as doc_error:
                            failed.append((doc_path, str(doc_error)))
            
            # Process other documents (non-LlamaParse) individually
            for doc_path, status in other_docs:
                try:
                    if self._add_document_to_index(doc_path, status):
                        successful.append((doc_path, 0))
                    else:
                        failed.append((doc_path, "Processing failed"))
                except Exception as e:
                    failed.append((doc_path, str(e)))
        else:
            # No LlamaParse - process all documents individually
            for doc_path, status in documents:
                try:
                    if self._add_document_to_index(doc_path, status):
                        successful.append((doc_path, 0))
                    else:
                        failed.append((doc_path, "Processing failed"))
                except Exception as e:
                    failed.append((doc_path, str(e)))
        
        return all_nodes, successful, failed
    
    def _insert_nodes_with_rate_limiting(self, nodes: List[Any], create_new_index: bool = False) -> None:
        """
        Insert nodes into the index with rate limiting to avoid embedding API limits.
        
        Processes nodes in smaller batches with delays between batches to respect
        Azure OpenAI rate limits.
        
        Args:
            nodes: List of nodes to insert
            create_new_index: If True, creates a new index; otherwise inserts into existing
        """
        if not nodes:
            return
        
        total_nodes = len(nodes)
        batch_size = MAX_NODES_PER_BATCH
        num_batches = (total_nodes + batch_size - 1) // batch_size
        
        if num_batches > 1:
            print(f"   📊 Inserting {total_nodes} nodes in {num_batches} batches of ~{batch_size} to respect rate limits...")
        
        for i in range(0, total_nodes, batch_size):
            batch_nodes = nodes[i:i+batch_size]
            batch_num = (i // batch_size) + 1
            
            if num_batches > 1:
                print(f"      Batch {batch_num}/{num_batches}: embedding {len(batch_nodes)} nodes...", end=" ", flush=True)
            
            try:
                if create_new_index and i == 0:
                    # Create index with first batch
                    chroma_collection = self.chroma_client.get_or_create_collection(
                        name=f"{self.collection}_vectors"
                    )
                    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
                    storage_context = StorageContext.from_defaults(vector_store=vector_store)
                    self.index = VectorStoreIndex(nodes=batch_nodes, storage_context=storage_context)
                    self.state["index_created"] = True
                else:
                    # Insert into existing index
                    self.index.insert_nodes(batch_nodes)
                
                if num_batches > 1:
                    print("✓")
                
                # Add delay between batches (except for last batch)
                if i + batch_size < total_nodes and EMBEDDING_DELAY_SECONDS > 0:
                    time.sleep(EMBEDDING_DELAY_SECONDS)
                    
            except Exception as e:
                if num_batches > 1:
                    print(f"✗ Error: {e}")
                raise
    
    
    def update_documents(self) -> bool:
        """
        Update the document index with parallel processing.
        
        Features:
        - Parallel document parsing using asyncio.gather() for maximum concurrency
        - Efficient batch embedding insertion with rate limiting
        - Comprehensive timing metrics
        - Progressive state updates
        """
        try:
            overall_start = time.time()
            print(f"🔄 Updating RAG index for collection '{self.collection}'...")
            
            # Phase 1: Scan for documents
            print(f"\n📁 Phase 1/6: Scanning directory...")
            scan_start = time.time()
            current_docs = self._scan_documents()
            
            if not current_docs:
                print(f"⚠️  No documents found in {self.documents_dir}")
                return True
            
            scan_elapsed = time.time() - scan_start
            print(f"✅ Scanned directory ({scan_elapsed:.1f}s)")
            
            # Phase 2: Load existing index
            print(f"\n📖 Phase 2/6: Loading existing index...")
            load_start = time.time()
            
            if self.state.get("index_created", False) and not self.index:
                try:
                    chroma_collection = self.chroma_client.get_collection(
                        name=f"{self.collection}_vectors"
                    )
                    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
                    self.index = VectorStoreIndex.from_vector_store(vector_store)
                    load_elapsed = time.time() - load_start
                    print(f"✅ Loaded existing index ({load_elapsed:.1f}s)")
                except Exception as e:
                    print(f"   ⚠️  No existing index found, will create new one")
                    self.index = None
                    load_elapsed = time.time() - load_start
            else:
                load_elapsed = time.time() - load_start
                print(f"   No existing index to load")
            
            # Phase 3: Identify documents to process
            print(f"\n🔍 Phase 3/6: Identifying documents to process...")
            identify_start = time.time()
            
            to_process = self._get_documents_to_process()
            deleted = self._get_deleted_documents()
            
            identify_elapsed = time.time() - identify_start
            
            # Check if everything is up to date
            if not to_process and not deleted:
                indexed_count = self.state.get("indexed_documents", 0)
                print(f"✅ Index is up to date ({indexed_count} documents indexed)")
                print(f"\n⏱️  Total time: {time.time() - overall_start:.1f}s")
                return True
            
            # Summary
            print(f"📊 Found {len(current_docs)} total documents ({identify_elapsed:.1f}s):")
            if to_process:
                new_count = sum(1 for _, status in to_process if status == 'new')
                modified_count = sum(1 for _, status in to_process if status == 'modified')
                failed_count = sum(1 for _, status in to_process if status == 'failed')
                if new_count > 0:
                    print(f"   • {new_count} new")
                if modified_count > 0:
                    print(f"   • {modified_count} modified")
                if failed_count > 0:
                    print(f"   • {failed_count} previously failed")
            if deleted:
                print(f"   • {len(deleted)} deleted")
            
            # Phase 4: Process all documents in parallel
            print(f"\n📦 Phase 4/6: Processing documents...")
            process_start = time.time()
            
            print(f"   Processing {len(to_process)} document(s) concurrently")
            
            # Process all documents in parallel (parsing phase)
            parsing_start = time.time()
            all_nodes, successful_docs, failed_docs = self._process_documents_parallel(to_process)
            parsing_elapsed = time.time() - parsing_start
            
            # Insert all nodes together with rate limiting for embedding efficiency (embedding phase)
            embedding_elapsed = 0.0
            if all_nodes:
                embedding_start = time.time()
                create_new = not self.index
                self._insert_nodes_with_rate_limiting(all_nodes, create_new_index=create_new)
                embedding_elapsed = time.time() - embedding_start
            
            # Update state for successful documents
            successful_count = 0
            for doc_path, chunk_count in successful_docs:
                self._mark_document_indexed(doc_path, chunk_count)
                successful_count += 1
            
            # Update state for failed documents
            failed_count = len(failed_docs)
            for doc_path, error in failed_docs:
                self._mark_document_failed(doc_path, error)
            
            process_elapsed = time.time() - process_start
            print(f"\n✅ Processed {len(to_process)} documents ({process_elapsed:.1f}s)")
            print(f"   📊 Processing breakdown:")
            print(f"      • Parsing: {parsing_elapsed:.1f}s")
            print(f"      • Embedding: {embedding_elapsed:.1f}s")
            if len(to_process) > 0:
                parsing_per_doc = parsing_elapsed / len(to_process) if parsing_elapsed > 0 else 0
                embedding_per_doc = embedding_elapsed / len(to_process) if embedding_elapsed > 0 else 0
                print(f"   ⏱️  Speed: {parsing_per_doc:.1f}s/doc (parsing), {embedding_per_doc:.1f}s/doc (embedding)")
            print(f"   • Successfully indexed: {successful_count}")
            if failed_count > 0:
                print(f"   • Failed: {failed_count}")
            
            # Phase 5: Handle deleted documents
            if deleted:
                print(f"\n🗑️  Phase 5/6: Removing deleted documents...")
                delete_start = time.time()
                
                for deleted_path in deleted:
                    print(f"   Removing {Path(deleted_path).name} from state...")
                    self._remove_document_from_state(deleted_path)
                
                delete_elapsed = time.time() - delete_start
                print(f"✅ Removed {len(deleted)} document(s) ({delete_elapsed:.1f}s)")
            else:
                print(f"\n   Phase 5/6: No deleted documents")
                delete_elapsed = 0
            
            # Phase 6: Final summary
            print(f"\n🎉 Phase 6/6: Update complete!")
            print(f"   • Successfully processed: {successful_count}")
            if failed_count > 0:
                print(f"   • Failed: {failed_count}")
            if deleted:
                print(f"   • Removed: {len(deleted)}")
            
            indexed_count = self.state.get("indexed_documents", 0)
            total_count = self.state.get("total_documents", 0)
            print(f"   • Total in collection: {indexed_count}/{total_count} indexed")
            
            # Overall timing summary
            overall_elapsed = time.time() - overall_start
            print(f"\n⏱️  Total time: {overall_elapsed:.1f}s")
            print(f"   📊 Time breakdown:")
            print(f"      • Scanning: {scan_elapsed:.1f}s")
            print(f"      • Loading index: {load_elapsed:.1f}s")
            print(f"      • Identifying changes: {identify_elapsed:.1f}s")
            print(f"      • Processing docs: {process_elapsed:.1f}s")
            print(f"         - Parsing: {parsing_elapsed:.1f}s")
            print(f"         - Embedding: {embedding_elapsed:.1f}s")
            if delete_elapsed > 0:
                print(f"      • Removing deleted: {delete_elapsed:.1f}s")
            
            return True
            
        except Exception as e:
            print(f"❌ Error updating documents: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def query_documents(self, query: str, **kwargs) -> Optional[str]:
        """
        Query the document index.
        
        Args:
            query: The query string
            **kwargs: Additional parameters (e.g., top_k)
            
        Returns:
            Optional[str]: The response text
        """
        try:
            query_start = time.time()
            
            # Get top_k from kwargs or use default constant
            top_k = kwargs.get('top_k', DEFAULT_TOP_K)
            
            # Load index if not already loaded
            load_start = time.time()
            if not self.index:
                if not self.state.get("index_created", False):
                    # Check if there are documents in the folder
                    docs = self._scan_documents()
                    if docs:
                        print(f"⚠️  Vector store is empty but {len(docs)} document(s) found in {self.documents_dir}")
                        print(f"   Please run document_manager.py --backend rag --collection {self.collection} --update to index documents")
                        return f"No documents indexed yet. Found {len(docs)} document(s) that need to be indexed."
                    else:
                        return "No documents found in the collection."
                
                # Reload index from storage
                try:
                    print("🔄 Loading index from storage...")
                    chroma_collection = self.chroma_client.get_collection(
                        name=f"{self.collection}_vectors"
                    )
                    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
                    self.index = VectorStoreIndex.from_vector_store(vector_store)
                except Exception as e:
                    # Check if there are documents in the folder
                    docs = self._scan_documents()
                    if docs:
                        print(f"⚠️  Could not load vector index but {len(docs)} document(s) found in {self.documents_dir}")
                        print(f"   Please run document_manager.py --backend rag --collection {self.collection} --update to index documents")
                        return f"Documents found but not indexed. Please index them first."
                    else:
                        return f"Could not load vector index: {e}"
            
            load_elapsed = time.time() - load_start
            
            # Create retriever with the requested top_k
            retriever = self.index.as_retriever(similarity_top_k=top_k)
            
            # Retrieve relevant nodes
            print(f"🔍 Searching for relevant documents (top_k={top_k})...")
            retrieve_start = time.time()
            nodes = retriever.retrieve(query)
            retrieve_elapsed = time.time() - retrieve_start
            
            if not nodes:
                query_elapsed = time.time() - query_start
                print(f"⏱️  Query time: {query_elapsed:.2f}s (no results found)")
                return "No relevant documents found for your query."
            
            print(f"📊 Retrieved {len(nodes)} chunks ({retrieve_elapsed:.2f}s)")
            
            # Apply reranking if available
            rerank_elapsed = 0
            if self.cohere_reranker:
                print(f"🔄 Applying Cohere reranking...")
                rerank_start = time.time()
                nodes = self.cohere_reranker.postprocess_nodes(
                    nodes=nodes,
                    query_str=query
                )
                rerank_elapsed = time.time() - rerank_start
                print(f"✅ Reranked to top {len(nodes)} chunks ({rerank_elapsed:.2f}s)")
            
            # Show chunk previews if requested
            show_chunks = kwargs.get('show_chunks', False)
            
            # Build context from retrieved nodes
            context_parts = []
            sources = set()
            for i, node in enumerate(nodes, 1):
                if show_chunks:
                    # Show chunk preview and score
                    chunk_preview = node.text[:200].replace('\n', ' ')
                    score = node.score if hasattr(node, 'score') else 'N/A'
                    print(f"   Chunk {i} (score: {score}): {chunk_preview}...")
                
                context_parts.append(f"[Document {i}]\n{node.text}\n")
                # Extract source info
                if hasattr(node, 'metadata') and 'file_name' in node.metadata:
                    sources.add(node.metadata['file_name'])
            
            context = "\n".join(context_parts)
            sources_text = ", ".join(sources) if sources else "Unknown"
            
            # Use OpenAI v2 API for query
            print(f"💬 Generating response using {self.model}...")
            llm_start = time.time()
            response = self.client.responses.parse(
                model=self.model,
                input=[
                    {
                        "role": "system",
                        "content": f"""You are a helpful document analysis assistant for the '{self.collection}' collection.

You will be provided with relevant document excerpts. Use this information to answer the user's question accurately.

Guidelines:
1. Base your answer strictly on the provided document excerpts
2. Be specific and cite information from the documents
3. If the information is not in the provided excerpts, clearly state that
4. Provide accurate, well-sourced answers
5. Do not include citation markers or reference numbers

Sources: {sources_text}
"""
                    },
                    {
                        "role": "user",
                        "content": f"""Context from documents:

{context}

Question: {query}"""
                    }
                ],
                reasoning={"effort": self.reasoning_effort},
                text={"verbosity": self.text_verbosity}
            )
            llm_elapsed = time.time() - llm_start
            
            # Extract response text
            response_text = None
            if hasattr(response, 'output') and response.output:
                for item in response.output:
                    if hasattr(item, 'content') and item.content:
                        for content_item in item.content:
                            if hasattr(content_item, 'text'):
                                response_text = content_item.text
                                break
            
            if not response_text:
                response_text = "No response generated."
            
            # Print timing summary
            query_elapsed = time.time() - query_start
            print(f"\n⏱️  Query timing: {query_elapsed:.2f}s total")
            print(f"   📊 Breakdown:")
            if load_elapsed > 0.01:
                print(f"      • Loading index: {load_elapsed:.2f}s")
            print(f"      • Retrieval: {retrieve_elapsed:.2f}s")
            if rerank_elapsed > 0:
                print(f"      • Reranking: {rerank_elapsed:.2f}s")
            print(f"      • LLM generation: {llm_elapsed:.2f}s")
            
            return response_text
            
        except Exception as e:
            print(f"❌ Error querying documents: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def delete_documents(self) -> bool:
        """Delete the entire collection (remove storage directory)."""
        try:
            import shutil
            
            # Delete Chroma collection
            try:
                self.chroma_client.delete_collection(name=f"{self.collection}_vectors")
                print(f"🗑️  Deleted Chroma collection")
            except Exception as e:
                print(f"⚠️  Warning: Could not delete Chroma collection: {e}")
            
            # Remove storage directory
            if self.storage_dir.exists():
                shutil.rmtree(self.storage_dir)
                print(f"✅ Deleted collection '{self.collection}'")
            else:
                print(f"ℹ️  Collection '{self.collection}' not found")
            
            return True
            
        except Exception as e:
            print(f"❌ Error deleting collection: {e}")
            return False
    
    def get_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the collection."""
        return {
            "total_documents": self.state.get("total_documents", 0),
            "last_updated": self.state.get("last_updated"),
            "storage_dir": str(self.storage_dir),
            "collection": self.collection,
            "backend": "rag",
            "embedding_model": self.embedding_model,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "index_created": self.state.get("index_created", False)
        }
    
    def list_documents(self) -> List[str]:
        """List all documents in the collection."""
        return list(self.state.get("documents", {}).keys())
    
    @classmethod
    def list_all_collections(cls, storage_dir: str = DEFAULT_STORAGE_DIR) -> List[str]:
        """List all collections in the RAG storage directory (class method)."""
        storage_path = Path(storage_dir)
        if not storage_path.exists():
            return []
        
        collections = []
        try:
            # Each subdirectory represents a collection
            for item in storage_path.iterdir():
                if item.is_dir():
                    # Check if it has a rag_state.json file (indicates valid collection)
                    state_file = item / "rag_state.json"
                    if state_file.exists():
                        collections.append(item.name)
        except Exception:
            return []
        
        return sorted(collections)

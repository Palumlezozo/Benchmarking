"""
RAG Document Store using LlamaIndex and Chroma.

This module provides a document store implementation using LlamaIndex for indexing
and Chroma as the vector store backend.
"""

import json
import os
import warnings
from pathlib import Path
from typing import Dict, Any, List, Optional
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
    LLAMA_PARSE_PAGE_PREFIX,
    LLAMA_PARSE_PAGE_SUFFIX,
    LLAMA_SPLIT_BY_PAGE,
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
            Settings.embed_model = OpenAIEmbedding(
                model=embedding_model,
                api_key=azure_api_key,
                api_base=azure_base_url
            )
        else:
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            # Configure LlamaIndex settings with standard OpenAI
            Settings.embed_model = OpenAIEmbedding(
                model=embedding_model,
                api_key=os.getenv("OPENAI_API_KEY")
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
        
        # Initialize LlamaParse if enabled and available
        self.llama_parse_parser = self._create_llama_parse_parser()
        
        # Initialize Cohere reranker if enabled and available
        self.cohere_reranker = self._create_cohere_reranker()
    
    def _create_llama_parse_parser(self) -> Optional[Any]:
        """
        Create LlamaParse parser if enabled and API key is available.
        
        Returns:
            LlamaParse parser instance or None if not available
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
        
        try:
            # Build parser kwargs
            parser_kwargs = {
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
                "spreadsheet_force_formula_computation": LLAMA_PARSE_SPREADSHEET_FORCE_FORMULA_COMPUTATION,
                "page_prefix": LLAMA_PARSE_PAGE_PREFIX,
                "page_suffix": LLAMA_PARSE_PAGE_SUFFIX,
                "split_by_page": LLAMA_SPLIT_BY_PAGE
            }
            
            # Add base_url if specified (for European or custom endpoints)
            if base_url:
                parser_kwargs["base_url"] = base_url
            
            parser = LlamaParse(**parser_kwargs)
            endpoint_info = f" (endpoint: {base_url})" if base_url else ""
            print(f"✅ LlamaParse enabled{endpoint_info} (result_type={LLAMA_PARSE_RESULT_TYPE}, parse_mode={LLAMA_PARSE_PARSE_MODE}, workers={LLAMA_PARSE_NUM_WORKERS})")
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
        # Pattern: Our custom format <!-- PAGE_START: N --> or <!-- PAGE_END: N -->
        page_pattern = r'<!--\s*PAGE_(?:START|END):\s*(\d+)\s*-->'
        
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
        # Remove our custom page markers <!-- PAGE_START: N --> and <!-- PAGE_END: N -->
        page_pattern = r'\n*<!--\s*PAGE_(?:START|END):\s*\d+\s*-->\n*'
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
        documents_with_markers = 0
        
        print(f"📑 Checking for page markers in {len(documents)} document(s)...")
        
        for doc in documents:
            # Only process if we're using LlamaParse
            if not self.llama_parse_parser:
                enriched_documents.append(doc)
                continue
            
            # Check if this is a LlamaParse-supported file type
            file_path = doc.metadata.get('file_path') or doc.metadata.get('file_name', '')
            file_ext = Path(str(file_path)).suffix.lower()
            
            if file_ext not in LLAMAPARSE_SUPPORTED_EXTENSIONS:
                # Not a LlamaParse format, skip page extraction
                enriched_documents.append(doc)
                continue
            
            text = doc.text
            
            # Extract page numbers from the document
            page_numbers = self._extract_page_numbers_from_text(text)
            
            if not page_numbers:
                # No page markers found in text - keep original document
                print(f"   ⚠️  No page markers found")
                enriched_documents.append(doc)
                continue
            
            # Just mark that this document has page markers
            # Page extraction will happen after chunking
            doc.metadata['_has_page_markers'] = True
            enriched_documents.append(doc)
            documents_with_markers += 1
        
        # Summary
        if documents_with_markers > 0:
            print(f"   ✅ {documents_with_markers}/{len(documents)} documents have page markers")
        else:
            print(f"   ⚠️  No page markers found in any documents")
        
        return enriched_documents
    
    def _post_process_nodes_with_page_ranges(self, nodes: List[Any]) -> List[Any]:
        """
        Post-process chunks to extract page numbers from embedded markers.
        
        Each chunk may contain page markers embedded by LlamaParse:
        - <!-- PAGE_START: N -->
        - <!-- PAGE_END: N -->
        
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
                
                # Save markdown content
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
        Load a single document using LlamaIndex.
        
        Args:
            doc_path: Path to the document
            
        Returns:
            List of Document objects
        """
        # Configure file extractor based on LlamaParse availability
        if self.llama_parse_parser:
            file_extractor = {
                ".pdf": self.llama_parse_parser,
                ".docx": self.llama_parse_parser,
                ".doc": self.llama_parse_parser,
                ".pptx": self.llama_parse_parser,
                ".ppt": self.llama_parse_parser,
                ".xlsx": self.llama_parse_parser,
                ".xls": self.llama_parse_parser,
                ".html": self.llama_parse_parser,
                ".htm": self.llama_parse_parser,
            }
        else:
            file_extractor = {
                ".pdf": PDFReader(),
            }
        
        reader = SimpleDirectoryReader(
            input_files=[str(doc_path)],
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
            
            # Add nodes to existing index (or create if first document)
            if not self.index:
                # First document - create new index
                chroma_collection = self.chroma_client.get_or_create_collection(
                    name=f"{self.collection}_vectors"
                )
                vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
                storage_context = StorageContext.from_defaults(vector_store=vector_store)
                
                self.index = VectorStoreIndex(
                    nodes,
                    storage_context=storage_context
                )
                
                self.state["index_created"] = True
            else:
                # Add to existing index
                self.index.insert_nodes(nodes)
            
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
    
    def update_documents(self) -> bool:
        """
        Update the document index incrementally.
        
        This will:
        1. Scan the documents directory
        2. Identify documents that need processing (new, modified, or failed)
        3. Process documents one at a time with progressive state updates
        4. Handle deletions
        5. Resume capability: can restart after interruption
        """
        try:
            print(f"🔄 Updating RAG index for collection '{self.collection}'...")
            
            # Scan for documents
            current_docs = self._scan_documents()
            
            if not current_docs:
                print(f"⚠️  No documents found in {self.documents_dir}")
                return True
            
            # Load existing index if available
            if self.state.get("index_created", False) and not self.index:
                try:
                    print("📖 Loading existing index...")
                    chroma_collection = self.chroma_client.get_collection(
                        name=f"{self.collection}_vectors"
                    )
                    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
                    self.index = VectorStoreIndex.from_vector_store(vector_store)
                    print("   ✅ Existing index loaded")
                except Exception as e:
                    print(f"   ⚠️  Could not load existing index: {e}")
                    print("   Will create new index")
                    self.index = None
            
            # Get documents that need processing
            to_process = self._get_documents_to_process()
            
            # Get deleted documents
            deleted = self._get_deleted_documents()
            
            # Check if everything is up to date
            if not to_process and not deleted:
                indexed_count = self.state.get("indexed_documents", 0)
                print(f"✅ Index is up to date ({indexed_count} documents indexed)")
                return True
            
            # Summary
            total_work = len(to_process) + len(deleted)
            print(f"📊 Found {len(current_docs)} total documents:")
            if to_process:
                new_count = sum(1 for _, status in to_process if status == 'new')
                modified_count = sum(1 for _, status in to_process if status == 'modified')
                failed_count = sum(1 for _, status in to_process if status == 'failed')
                print(f"   • {new_count} new")
                print(f"   • {modified_count} modified")
                print(f"   • {failed_count} previously failed")
            if deleted:
                print(f"   • {len(deleted)} deleted")
            print()
            
            # Process documents one at a time
            successful = 0
            failed = 0
            
            for i, (doc_path, status) in enumerate(to_process, 1):
                print(f"[{i}/{len(to_process)}] ", end="")
                if self._add_document_to_index(doc_path, status):
                    successful += 1
                else:
                    failed += 1
                print()  # Empty line between documents
            
            # Handle deleted documents
            for deleted_path in deleted:
                print(f"🗑️  Removing {Path(deleted_path).name} from state...")
                self._remove_document_from_state(deleted_path)
            
            # Final summary
            print("=" * 60)
            print(f"✅ Index update complete!")
            print(f"   • Successfully indexed: {successful}")
            if failed > 0:
                print(f"   • Failed: {failed}")
            if deleted:
                print(f"   • Removed: {len(deleted)}")
            
            indexed_count = self.state.get("indexed_documents", 0)
            total_count = self.state.get("total_documents", 0)
            print(f"   • Total in collection: {indexed_count}/{total_count} indexed")
            print("=" * 60)
            
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
            # Get top_k from kwargs or use default constant
            top_k = kwargs.get('top_k', DEFAULT_TOP_K)
            
            # Load index if not already loaded
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
            
            # Create retriever with the requested top_k
            retriever = self.index.as_retriever(similarity_top_k=top_k)
            
            # Retrieve relevant nodes
            print(f"🔍 Searching for relevant documents (top_k={top_k})...")
            nodes = retriever.retrieve(query)
            
            if not nodes:
                return "No relevant documents found for your query."
            
            print(f"📊 Retrieved {len(nodes)} chunks")
            
            # Apply reranking if available
            if self.cohere_reranker:
                print(f"🔄 Applying Cohere reranking...")
                nodes = self.cohere_reranker.postprocess_nodes(
                    nodes=nodes,
                    query_str=query
                )
                print(f"✅ Reranked to top {len(nodes)} chunks")
            
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
            
            # Extract response text
            if hasattr(response, 'output') and response.output:
                for item in response.output:
                    if hasattr(item, 'content') and item.content:
                        for content_item in item.content:
                            if hasattr(content_item, 'text'):
                                return content_item.text
            
            return "No response generated."
            
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

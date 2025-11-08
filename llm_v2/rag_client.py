"""
RAG Document Store using LlamaIndex with Qdrant or Chroma.

This module provides a document store implementation using LlamaIndex for indexing
with Qdrant (default) or Chroma as the vector store backend.
"""

import json
import os
import warnings
import time
import asyncio
import io
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime as dt

from openai import OpenAI
from dotenv import load_dotenv

# Suppress Pydantic warnings from LlamaIndex
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.readers.file import PDFReader
import re
from pypdf import PdfReader as PypdfReader, PdfWriter

# Vector store imports (conditional)
try:
    from llama_index.vector_stores.qdrant import QdrantVectorStore
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    QdrantVectorStore = None

try:
    from llama_index.vector_stores.chroma import ChromaVectorStore
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    ChromaVectorStore = None
    chromadb = None

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance, VectorParams, HnswConfigDiff, 
        OptimizersConfigDiff, VectorsConfig, PointStruct
    )
    QDRANT_CLIENT_AVAILABLE = True
except ImportError:
    QDRANT_CLIENT_AVAILABLE = False
    QdrantClient = None
    Distance = None
    VectorParams = None
    HnswConfigDiff = None
    OptimizersConfigDiff = None
    VectorsConfig = None
    PointStruct = None

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
    DEFAULT_MODEL,
    DEFAULT_REASONING_EFFORT,
    DEFAULT_TEXT_VERBOSITY,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_TOP_K,
    USE_QDRANT,
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_API_KEY,
    QDRANT_HNSW_M,
    QDRANT_HNSW_EF_CONSTRUCT,
    QDRANT_HNSW_EF,
    QDRANT_HNSW_FULL_SCAN_THRESHOLD,
    QDRANT_ON_DISK,
    QDRANT_ON_DISK_PAYLOAD,
    QDRANT_DEFAULT_SEGMENT_NUMBER,
    QDRANT_MAX_SEGMENT_SIZE,
    QDRANT_MEMMAP_THRESHOLD,
    QDRANT_DELETED_THRESHOLD,
    QDRANT_VACUUM_MIN_VECTOR_NUMBER,
    QDRANT_INDEXING_THRESHOLD,
    QDRANT_FLUSH_INTERVAL_SEC,
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
    LLAMA_PARSE_USE_VENDOR_MULTIMODAL_MODEL,
    LLAMA_PARSE_VENDOR_MULTIMODAL_MODEL_NAME,
    LLAMA_PARSE_AZURE_OPENAI_DEPLOYMENT_NAME,
    LLAMA_PARSE_AZURE_OPENAI_ENDPOINT,
    LLAMA_PARSE_AZURE_OPENAI_API_VERSION,
    USE_COHERE_RERANK,
    COHERE_RERANK_MODEL,
    COHERE_RERANK_TOP_N,
    DOCUMENT_BATCH_SIZE,
    RAG_MARKDOWNS_DIR,
    RAG_CHROMA_STORAGE_DIR,
    RAG_CHROMA_STATE_FILE,
    RAG_QDRANT_STATE_FILE,
)

# Load environment variables
load_dotenv()

# ============================================================================
# RAG Configuration Constants
# ============================================================================

# Storage paths
DEFAULT_DOCUMENTS_DIR = "data/documents"
# Use new simplified structure - storage_dir is now backend-specific
DEFAULT_STORAGE_DIR = RAG_CHROMA_STORAGE_DIR  # For ChromaDB, use data/rag/chroma
DEFAULT_MARKDOWNS_DIR = RAG_MARKDOWNS_DIR  # Use data/rag/markdowns

# File types supported by LlamaParse (that may contain page markers)
LLAMAPARSE_SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.doc', '.pptx', '.ppt', '.xlsx', '.xls', '.html', '.htm'}

# Note: DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, and DEFAULT_TOP_K are imported from config.py
# ============================================================================

# Embedding model dimensions mapping
EMBEDDING_DIMENSIONS = {
    "text-embedding-3-large": 3072,
    "text-embedding-3-small": 1536,
    "text-embedding-ada-002": 1536,
}

def _get_embedding_dimension(embedding_model: str) -> int:
    """
    Get the embedding dimension for a given model.
    
    Args:
        embedding_model: Name of the embedding model
        
    Returns:
        Dimension size (defaults to 1536 if unknown)
    """
    return EMBEDDING_DIMENSIONS.get(embedding_model, 1536)


class RAGDocumentStore(DocumentStore):
    """Document store implementation using LlamaIndex with Qdrant (default) or Chroma."""
    
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
        use_llama_parse: bool = None,
        vector_store_type: Optional[str] = None
    ):
        """
        Initialize the RAG document store.
        
        Args:
            collection: Collection name
            documents_dir: Base directory for documents
            storage_dir: Directory for vector store storage (local for Chroma)
            embedding_model: OpenAI embedding model to use
            model: LLM model for queries
            reasoning_effort: Reasoning effort level
            text_verbosity: Text verbosity level
            chunk_size: Size of text chunks for indexing
            chunk_overlap: Overlap between chunks
            store_md: Whether to store markdown outputs from LlamaParse
            use_llama_parse: Override config to force LlamaParse on/off (None = use config setting)
            vector_store_type: Vector store type ("qdrant" or "chroma"). None = use config default (Qdrant)
        """
        super().__init__(collection)
        self.documents_dir = Path(documents_dir) / collection
        self.markdowns_dir = Path(DEFAULT_MARKDOWNS_DIR) / collection
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.store_md = store_md
        # Override config setting if explicitly specified
        self.use_llama_parse_override = use_llama_parse
        
        # Determine vector store type
        if vector_store_type is None:
            self.vector_store_type = "qdrant" if USE_QDRANT else "chroma"
        else:
            self.vector_store_type = vector_store_type.lower()
        
        if self.vector_store_type not in ["qdrant", "chroma"]:
            raise ValueError(f"Unsupported vector_store_type: {self.vector_store_type}. Must be 'qdrant' or 'chroma'")
        
        # Set storage directory based on vector store type
        if self.vector_store_type == "chroma":
            # ChromaDB: per-collection subdirectory for database files
            self.storage_dir = Path(RAG_CHROMA_STORAGE_DIR) / collection
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            # Global state file for all ChromaDB collections
            self.state_file = Path(RAG_CHROMA_STATE_FILE)
        else:  # qdrant
            # Qdrant: no local storage (service-based), only state file
            self.storage_dir = None  # No local storage for Qdrant
            # Global state file for all Qdrant collections
            self.state_file = Path(RAG_QDRANT_STATE_FILE)
        
        # Ensure state file directory exists
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Create markdowns directory if markdown storage is enabled
        if self.store_md:
            self.markdowns_dir.mkdir(parents=True, exist_ok=True)
        
        # State tracking (global state file with per-collection data)
        self.global_state = self._load_global_state()
        self.state = self._get_collection_state()
        
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
        
        # Initialize vector store client
        self._initialize_vector_store_client()
        
        # Vector store and index (will be initialized in update_documents)
        self.index = None
        
        print(f"📚 RAG Document Store initialized for collection '{collection}'")
        print(f"   Vector Store: {self.vector_store_type.upper()}")
        print(f"   Documents: {self.documents_dir}")
        if self.storage_dir:
            print(f"   Storage: {self.storage_dir}")
        else:
            print(f"   Storage: Qdrant service (no local storage)")
        if self.store_md:
            print(f"   Markdowns: {self.markdowns_dir}")
        print(f"   Embedding Model: {embedding_model}")
        
        # Store LlamaParse configuration and create parser at class level
        self.llama_parse_config = self._get_llama_parse_config()
        # Create parser once at initialization (reused for all parsing operations)
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
            "invalidate_cache": LLAMA_PARSE_INVALIDATE_CACHE,
            "do_not_cache": LLAMA_PARSE_DO_NOT_CACHE,
            "num_workers": LLAMA_PARSE_NUM_WORKERS,
            "skip_diagonal_text": LLAMA_PARSE_SKIP_DIAGONAL_TEXT,
            "spreadsheet_extract_sub_tables": LLAMA_PARSE_SPREADSHEET_EXTRACT_SUB_TABLES,
            "spreadsheet_force_formula_computation": LLAMA_PARSE_SPREADSHEET_FORCE_FORMULA_COMPUTATION,
            "check_interval": 10
        }
        
        if LLAMA_PARSE_PARSE_MODE:
            config["parse_mode"] = LLAMA_PARSE_PARSE_MODE
        # Add Azure OpenAI configuration if specified
        if LLAMA_PARSE_USE_VENDOR_MULTIMODAL_MODEL:
            config["use_vendor_multimodal_model"] = True
            config["vendor_multimodal_model_name"] = LLAMA_PARSE_VENDOR_MULTIMODAL_MODEL_NAME
        
        if LLAMA_PARSE_AZURE_OPENAI_DEPLOYMENT_NAME:
            config["azure_openai_deployment_name"] = LLAMA_PARSE_AZURE_OPENAI_DEPLOYMENT_NAME
        
        if LLAMA_PARSE_AZURE_OPENAI_ENDPOINT:
            config["azure_openai_endpoint"] = LLAMA_PARSE_AZURE_OPENAI_ENDPOINT
        
        if LLAMA_PARSE_AZURE_OPENAI_API_VERSION:
            config["azure_openai_api_version"] = LLAMA_PARSE_AZURE_OPENAI_API_VERSION
        
        # Azure OpenAI key can come from config or environment variable
        azure_openai_key = os.getenv("LLAMA_PARSE_AZURE_OPENAI_KEY")
        if azure_openai_key:
            config["azure_openai_key"] = azure_openai_key
        
        # Add base_url if specified (for European or custom endpoints)
        if base_url:
            config["base_url"] = base_url
        
        endpoint_info = f" (endpoint: {base_url})" if base_url else ""
        
        print(f"   LlamaParse enabled{endpoint_info} (result_type={LLAMA_PARSE_RESULT_TYPE}, parse_mode={LLAMA_PARSE_PARSE_MODE}, workers={LLAMA_PARSE_NUM_WORKERS})")
        
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
            print(f"   Cohere Rerank enabled (model={COHERE_RERANK_MODEL}, top_n={COHERE_RERANK_TOP_N})")
            return reranker
        except Exception as e:
            print(f"⚠️  Failed to initialize Cohere reranker: {e}")
            return None
    
    def _initialize_vector_store_client(self):
        """Initialize the vector store client based on type."""
        if self.vector_store_type == "qdrant":
            if not QDRANT_CLIENT_AVAILABLE:
                raise ImportError("Qdrant client not available. Install with: pip install qdrant-client")
            if not QDRANT_AVAILABLE:
                raise ImportError("Qdrant vector store not available. Install with: pip install llama-index-vector-stores-qdrant")
            
            # Initialize Qdrant client
            client_kwargs = {
                "host": QDRANT_HOST,
                "port": QDRANT_PORT
            }
            if QDRANT_API_KEY:
                client_kwargs["api_key"] = QDRANT_API_KEY
            
            try:
                self.qdrant_client = QdrantClient(**client_kwargs)
                # Test connection by attempting to get collections
                self.qdrant_client.get_collections()
                self.vector_client = self.qdrant_client  # Alias for compatibility
                print(f"   Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
            except Exception as e:
                error_msg = (
                    f"❌ Error: Qdrant is configured but not responding.\n"
                    f"   Host: {QDRANT_HOST}:{QDRANT_PORT}\n"
                    f"   Error: {str(e)}\n"
                    f"   Please ensure Qdrant is running and accessible.\n"
                    f"   You can start Qdrant with: docker run -p {QDRANT_PORT}:6333 qdrant/qdrant"
                )
                raise ConnectionError(error_msg) from e
            
        elif self.vector_store_type == "chroma":
            if not CHROMA_AVAILABLE:
                raise ImportError("Chroma not available. Install with: pip install chromadb llama-index-vector-stores-chroma")
            
            # Initialize Chroma client with per-collection storage directory
            if self.storage_dir is None:
                raise ValueError("ChromaDB requires a storage directory")
            self.chroma_client = chromadb.PersistentClient(path=str(self.storage_dir))
            self.vector_client = self.chroma_client  # Alias for compatibility
            print(f"   Using Chroma (local storage at {self.storage_dir})")
        else:
            raise ValueError(f"Unsupported vector_store_type: {self.vector_store_type}")
    
    def _create_vector_store(self, collection_name: str) -> Any:
        """
        Create LlamaIndex vector store wrapper.
        Creates Qdrant collection with optimized parameters if it doesn't exist.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            LlamaIndex vector store instance
        """
        if self.vector_store_type == "qdrant":
            if not QDRANT_AVAILABLE:
                raise ImportError("Qdrant vector store not available")
            
            # Get embedding dimension
            embedding_dim = _get_embedding_dimension(self.embedding_model)
            
            # Check if collection exists
            try:
                collections = self.qdrant_client.get_collections().collections
            except Exception as e:
                error_msg = (
                    f"❌ Error: Qdrant is not responding when trying to access collections.\n"
                    f"   Host: {QDRANT_HOST}:{QDRANT_PORT}\n"
                    f"   Error: {str(e)}\n"
                    f"   Please ensure Qdrant is running and accessible."
                )
                raise ConnectionError(error_msg) from e
            
            collection_exists = any(col.name == collection_name for col in collections)
            
            if not collection_exists:
                # Create collection with optimized parameters
                # Build HNSW config
                hnsw_config = HnswConfigDiff(
                    m=QDRANT_HNSW_M,
                    ef_construct=QDRANT_HNSW_EF_CONSTRUCT,
                    full_scan_threshold=QDRANT_HNSW_FULL_SCAN_THRESHOLD,
                    on_disk=QDRANT_ON_DISK,
                )
                
                # Build optimizer config
                optimizer_config = OptimizersConfigDiff(
                    deleted_threshold=QDRANT_DELETED_THRESHOLD,
                    vacuum_min_vector_number=QDRANT_VACUUM_MIN_VECTOR_NUMBER,
                    default_segment_number=QDRANT_DEFAULT_SEGMENT_NUMBER,
                    max_segment_size=QDRANT_MAX_SEGMENT_SIZE,
                    memmap_threshold=QDRANT_MEMMAP_THRESHOLD,
                    indexing_threshold=QDRANT_INDEXING_THRESHOLD,
                    flush_interval_sec=QDRANT_FLUSH_INTERVAL_SEC,
                )
                
                # Create collection with optimized parameters
                try:
                    self.qdrant_client.create_collection(
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
                    print(f"   Created Qdrant collection '{collection_name}' with optimized parameters")
                    print(f"      Dimension: {embedding_dim}, HNSW M: {QDRANT_HNSW_M}, EF Construct: {QDRANT_HNSW_EF_CONSTRUCT}")
                except Exception as e:
                    error_msg = (
                        f"❌ Error: Qdrant is not responding when trying to create collection.\n"
                        f"   Host: {QDRANT_HOST}:{QDRANT_PORT}\n"
                        f"   Collection: {collection_name}\n"
                        f"   Error: {str(e)}\n"
                        f"   Please ensure Qdrant is running and accessible."
                    )
                    raise ConnectionError(error_msg) from e
            
            return QdrantVectorStore(
                client=self.qdrant_client,
                collection_name=collection_name
            )
        elif self.vector_store_type == "chroma":
            if not CHROMA_AVAILABLE:
                raise ImportError("Chroma vector store not available")
            
            chroma_collection = self.chroma_client.get_or_create_collection(
                name=collection_name
            )
            return ChromaVectorStore(chroma_collection=chroma_collection)
        else:
            raise ValueError(f"Unsupported vector_store_type: {self.vector_store_type}")
    
    def _get_vector_store_for_loading(self, collection_name: str) -> Any:
        """
        Get existing vector store for loading an index.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            LlamaIndex vector store instance
        """
        if self.vector_store_type == "qdrant":
            if not QDRANT_AVAILABLE:
                raise ImportError("Qdrant vector store not available")
            
            # Test connection before attempting to load
            try:
                self.qdrant_client.get_collections()
            except Exception as e:
                error_msg = (
                    f"❌ Error: Qdrant is not responding when trying to load index.\n"
                    f"   Host: {QDRANT_HOST}:{QDRANT_PORT}\n"
                    f"   Collection: {collection_name}\n"
                    f"   Error: {str(e)}\n"
                    f"   Please ensure Qdrant is running and accessible."
                )
                raise ConnectionError(error_msg) from e
            
            return QdrantVectorStore(
                client=self.qdrant_client,
                collection_name=collection_name
            )
        elif self.vector_store_type == "chroma":
            if not CHROMA_AVAILABLE:
                raise ImportError("Chroma vector store not available")
            
            chroma_collection = self.chroma_client.get_collection(
                name=collection_name
            )
            return ChromaVectorStore(chroma_collection=chroma_collection)
        else:
            raise ValueError(f"Unsupported vector_store_type: {self.vector_store_type}")
    
    def _delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection from the vector store.
        
        Args:
            collection_name: Name of the collection to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.vector_store_type == "qdrant":
                # Check if collection exists before deleting
                collections = self.qdrant_client.get_collections().collections
                collection_exists = any(col.name == collection_name for col in collections)
                if collection_exists:
                    self.qdrant_client.delete_collection(collection_name=collection_name)
                    print(f"🗑️  Deleted Qdrant collection '{collection_name}'")
                    return True
                else:
                    print(f"ℹ️  Qdrant collection '{collection_name}' not found (may have been already deleted)")
                    return True  # Return True since the goal is achieved (collection doesn't exist)
            elif self.vector_store_type == "chroma":
                # Check if collection exists before deleting
                try:
                    # Try to get the collection to check if it exists
                    self.chroma_client.get_collection(name=collection_name)
                    # If we get here, collection exists, so delete it
                    self.chroma_client.delete_collection(name=collection_name)
                    print(f"🗑️  Deleted Chroma collection '{collection_name}'")
                    return True
                except Exception as get_error:
                    # Collection doesn't exist
                    print(f"ℹ️  Chroma collection '{collection_name}' not found (may have been already deleted)")
                    return True  # Return True since the goal is achieved (collection doesn't exist)
            else:
                return False
        except Exception as e:
            print(f"⚠️  Warning: Could not delete {self.vector_store_type} collection '{collection_name}': {e}")
            return False
    
    def _load_global_state(self) -> Dict[str, Any]:
        """Load the global state from file (contains all collections)."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️  Warning: Could not load global state: {e}")
        return {
            "collections": {},
            "last_updated": None
        }
    
    def _get_collection_state(self) -> Dict[str, Any]:
        """Get state for the current collection from global state."""
        if "collections" not in self.global_state:
            self.global_state["collections"] = {}
        
        if self.collection not in self.global_state["collections"]:
            # Initialize collection state
            self.global_state["collections"][self.collection] = {
                "documents": {},
                "last_updated": None,
                "total_pages": 0,
                "indexed_pages": 0,
                "failed_pages": 0,
                "index_created": False,
                "vector_store_type": self.vector_store_type
            }
        
        return self.global_state["collections"][self.collection]
    
    def _save_state(self) -> None:
        """Save the global state to file (updates current collection's state)."""
        # Update collection state in global state
        self.state["last_updated"] = dt.now().isoformat()
        self.global_state["collections"][self.collection] = self.state
        self.global_state["last_updated"] = dt.now().isoformat()
        
        try:
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(self.global_state, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"❌ Error saving global state: {e}")
    
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
        """Update aggregate counters in state (per page)."""
        documents = self.state.get("documents", {})
        
        total_pages = sum(doc.get("page_count", 0) for doc in documents.values())
        indexed_pages = sum(doc.get("page_count", 0) for doc in documents.values() if doc.get("indexed", False))
        failed_pages = total_pages - indexed_pages
        
        self.state["total_pages"] = total_pages
        self.state["indexed_pages"] = indexed_pages
        self.state["failed_pages"] = failed_pages
    
    def _mark_document_indexed(self, doc_path: Path, page_count: int, chunk_count: int):
        """
        Mark document as successfully indexed and save state.
        
        Args:
            doc_path: Path to the document
            page_count: Number of pages in this document
            chunk_count: Number of chunks created from this document
        """
        file_path_str = str(doc_path)
        
        self.state["documents"][file_path_str] = {
            "size": doc_path.stat().st_size,
            "modified": doc_path.stat().st_mtime,
            "extension": doc_path.suffix.lower(),
            "indexed": True,
            "indexed_at": dt.now().isoformat(),
            "page_count": page_count,
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
        
        for i, doc in enumerate(documents):
            # Only process if we're using LlamaParse
            if not self.llama_parse_parser:
                enriched_documents.append(doc)
                continue
            
            # Check if this is a LlamaParse-supported file type
            file_path = doc.metadata.get('file_path') or doc.metadata.get('file_name', '')
            
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
                enriched_documents.append(doc)
                continue
            
            # Just mark that this document has page markers
            # Page extraction will happen after chunking
            doc.metadata['_has_page_markers'] = True
            enriched_documents.append(doc)
            files_with_markers.add(file_path)
        
        # Summary - count unique source files, not Document objects
        total_source_files = len(files_with_markers | files_without_markers)
        if not files_with_markers:
            print(f"   ⚠️  No page markers found in the documents")
        
        return enriched_documents
    
    def _post_process_nodes_with_page_ranges(self, nodes: List[Any]) -> List[Any]:
        """
        Post-process chunks to extract page numbers from embedded markers.
        
        Each chunk may contain page markers embedded by LlamaParse:
        - <!-- PAGE: N -->
        
        This method:
        1. Extracts page numbers from each chunk's text
        2. Sets page_label metadata according to rules:
           - Single marker (N): use "N-N+1" (e.g., "5-6")
           - Multiple markers (N to M): use "N-M" (e.g., "5-7")
           - No markers: use page number from next chunk with markers
        3. Removes page markers from the text
        
        Args:
            nodes: List of chunks from the node parser
            
        Returns:
            List of chunks with page_label metadata and clean text
        """
        # First pass: extract page numbers and convert to integers for all nodes
        page_numbers_by_node = []
        for node in nodes:
            if not node.metadata.get('_has_page_markers'):
                page_numbers_by_node.append(None)
                continue
            
            # Extract page numbers from this chunk's text
            page_str_list = self._extract_page_numbers_from_text(node.text)
            
            if page_str_list:
                # Convert to integers and sort
                page_ints = sorted([int(p) for p in page_str_list])
                page_numbers_by_node.append(page_ints)
            else:
                page_numbers_by_node.append([])
        
        # Second pass: assign page labels and remove markers
        for idx, node in enumerate(nodes):
            # Skip if this node didn't come from a document with page markers
            if not node.metadata.get('_has_page_markers'):
                continue
            
            page_numbers = page_numbers_by_node[idx]
            
            if page_numbers is None:
                # Shouldn't happen, but skip if it does
                continue
            
            if page_numbers:
                # Chunk has page markers
                first_page = page_numbers[0]
                last_page = page_numbers[-1]
                
                if len(page_numbers) == 1:
                    # Single marker: use N-N+1 format
                    node.metadata['page_label'] = f"{first_page}-{first_page + 1}"
                else:
                    # Multiple markers: use N-M format
                    node.metadata['page_label'] = f"{first_page}-{last_page + 1}"
            else:
                # No page markers in this chunk - look ahead to next chunk with markers
                next_page = None
                for next_idx in range(idx + 1, len(nodes)):
                    next_page_numbers = page_numbers_by_node[next_idx]
                    if next_page_numbers:
                        next_page = next_page_numbers[0]
                        break
                
                if next_page is not None:
                    # Use the page number from the next chunk
                    node.metadata['page_label'] = str(next_page)
                else:
                    # No subsequent chunks with markers - leave empty
                    node.metadata['page_label'] = None
            
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
                print(f"✅ Saved {markdown_filename}")
                
            except Exception as e:
                print(f"⚠️  Failed to save markdown for {file_path}: {e}")
        
        if saved_count > 0:
            print(f"✅ Saved {saved_count} markdown file(s)")
    
    def _split_pdf_into_batches(self, file_bytes: bytes, batch_size: int = 20) -> List[Tuple[bytes, int]]:
        """
        Split a PDF into batches of pages.
        
        Args:
            file_bytes: PDF file as bytes
            batch_size: Number of pages per batch (default: 20)
            
        Returns:
            List of tuples (pdf_bytes, start_page) for each batch
        """
        reader = PypdfReader(io.BytesIO(file_bytes))
        total_pages = len(reader.pages)
        
        if total_pages <= batch_size:
            return [(file_bytes, 1)]
        
        batches = []
        for start_page in range(0, total_pages, batch_size):
            writer = PdfWriter()
            end_page = min(start_page + batch_size, total_pages)
            
            for page_num in range(start_page, end_page):
                writer.add_page(reader.pages[page_num])
            
            batch_bytes = io.BytesIO()
            writer.write(batch_bytes)
            batch_bytes.seek(0)
            batches.append((batch_bytes.getvalue(), start_page + 1))
        
        return batches
    
    async def _parse_bytes_and_reconstruct(self, file_bytes: bytes, file_path_str: str, doc_name: str) -> Tuple[Optional[Document], int]:
        """
        Parse bytes using LlamaParse and reconstruct markdown with page markers.
        
        Args:
            file_bytes: File content as bytes
            file_path_str: Original file path (for metadata)
            doc_name: Document name (for metadata)
            
        Returns:
            Tuple of (Document object with reconstructed markdown, page_count), or (None, 0) if parsing failed
        """
        if not self.llama_parse_parser:
            return None, 0
        parser = self.llama_parse_parser
        file_obj = io.BytesIO(file_bytes)
        # LlamaParse requires extra_info with file_name when passing bytes
        extra_info = {"file_name": doc_name}
        result = await parser.aparse(file_obj, extra_info=extra_info)
        
        if result is None or not hasattr(result, 'pages') or not result.pages:
            return None, 0
        
        page_count = len(result.pages)
        
        # Reconstruct markdown from pages with page markers at the bottom
        combined_markdown_parts = []
        for page in result.pages:
            page_content = page.md if page.md is not None else (page.text if hasattr(page, 'text') and page.text else "")
            combined_markdown_parts.append(page_content)
            page_separator = f"\n<!-- PAGE: {page.page} -->\n"
            combined_markdown_parts.append(page_separator)
            combined_markdown_parts.append("\n")
        
        combined_markdown = "".join(combined_markdown_parts)
        
        doc = Document(
            text=combined_markdown,
            metadata={
                'file_path': file_path_str,
                'file_name': doc_name
            }
        )
        return doc, page_count
    
    def _ensure_qdrant_collection(self, collection_name: str) -> bool:
        """
        Ensure Qdrant collection exists with optimized parameters.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            True if collection exists or was created successfully
        """
        if not QDRANT_CLIENT_AVAILABLE:
            raise ImportError("Qdrant client not available")
        
        try:
            collections = self.qdrant_client.get_collections().collections
            collection_exists = any(col.name == collection_name for col in collections)
            
            if not collection_exists:
                # Get embedding dimension
                embedding_dim = _get_embedding_dimension(self.embedding_model)
                
                # Build HNSW config
                hnsw_config = HnswConfigDiff(
                    m=QDRANT_HNSW_M,
                    ef_construct=QDRANT_HNSW_EF_CONSTRUCT,
                    full_scan_threshold=QDRANT_HNSW_FULL_SCAN_THRESHOLD,
                    on_disk=QDRANT_ON_DISK,
                )
                
                # Build optimizer config
                optimizer_config = OptimizersConfigDiff(
                    deleted_threshold=QDRANT_DELETED_THRESHOLD,
                    vacuum_min_vector_number=QDRANT_VACUUM_MIN_VECTOR_NUMBER,
                    default_segment_number=QDRANT_DEFAULT_SEGMENT_NUMBER,
                    max_segment_size=QDRANT_MAX_SEGMENT_SIZE,
                    memmap_threshold=QDRANT_MEMMAP_THRESHOLD,
                    indexing_threshold=QDRANT_INDEXING_THRESHOLD,
                    flush_interval_sec=QDRANT_FLUSH_INTERVAL_SEC,
                )
                
                # Create collection
                self.qdrant_client.create_collection(
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
                print(f"   Created Qdrant collection '{collection_name}' with optimized parameters")
            
            return True
        except Exception as e:
            print(f"   ❌ Error ensuring Qdrant collection: {e}")
            return False
    
    async def _index_nodes_to_qdrant_direct(self, nodes: List[Any], collection_name: str) -> bool:
        """
        Index nodes directly to Qdrant using native SDK (consistent with retrieval).
        
        Args:
            nodes: List of LlamaIndex nodes to index
            collection_name: Name of the Qdrant collection
            
        Returns:
            True if successful, False otherwise
        """
        if not QDRANT_CLIENT_AVAILABLE or not PointStruct:
            raise ImportError("Qdrant client not available")
        
        try:
            # Ensure collection exists
            if not self._ensure_qdrant_collection(collection_name):
                return False
            
            # Extract text and metadata from nodes
            texts = []
            metadata_list = []
            for node in nodes:
                texts.append(node.text)
                # Extract metadata from node
                node_metadata = {}
                if hasattr(node, 'metadata') and node.metadata:
                    node_metadata = dict(node.metadata)
                # Add node_id if available
                if hasattr(node, 'node_id'):
                    node_metadata['node_id'] = node.node_id
                metadata_list.append(node_metadata)
            
            # Generate embeddings if nodes don't have them
            # Check if first node has embedding
            has_embeddings = hasattr(nodes[0], 'embedding') and nodes[0].embedding is not None
            
            if not has_embeddings:
                # Generate embeddings using OpenAI
                print(f"   Generating embeddings for {len(texts)} chunks...")
                embeddings = await self._generate_embeddings_batch(texts)
            else:
                # Extract embeddings from nodes
                embeddings = [node.embedding for node in nodes]
            
            # Prepare points for insertion
            import hashlib
            points = []
            for idx, (text, embedding, metadata) in enumerate(zip(texts, embeddings, metadata_list)):
                # Generate unique ID based on content hash and index
                content_hash = hashlib.md5(text.encode()).hexdigest()[:8]
                unique_id = int(hashlib.md5(f"{metadata.get('file_path', '')}_{idx}_{content_hash}".encode()).hexdigest()[:16], 16)
                
                points.append(
                    PointStruct(
                        id=unique_id,
                        vector=embedding,
                        payload={
                            "text": text,
                            **metadata
                        }
                    )
                )
            
            # Insert points in batches
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.qdrant_client.upsert(collection_name=collection_name, points=batch)
            
            return True
            
        except Exception as e:
            print(f"   ❌ Error indexing nodes to Qdrant: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def _generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors
        """
        # Use OpenAI client to generate embeddings
        # Handle batching to avoid rate limits
        from config import EMBEDDING_BATCH_SIZE
        
        all_embeddings = []
        batch_size = EMBEDDING_BATCH_SIZE
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                response = self.client.embeddings.create(
                    model=self.embedding_model,
                    input=batch
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"   ⚠️  Error generating embeddings for batch {i//batch_size + 1}: {e}")
                raise
        
        return all_embeddings
    
    async def _process_single_document_async(self, doc_path: Path, status: str) -> Tuple[bool, int, int, Optional[str], float, float]:
        """
        Process a single document asynchronously: parse → chunk → embed → insert.
        
        This function handles the complete pipeline for one document:
        - Parses the document (supports LlamaParse)
        - Chunks the document into nodes
        - Embeds and inserts nodes into index immediately
        
        Args:
            doc_path: Path to the document
            status: Document status ('new', 'modified', or 'failed')
            
        Returns:
            Tuple of (success: bool, page_count: int, chunk_count: int, error: Optional[str], parsing_time: float, embedding_time: float)
        """
        parsing_start = time.time()
        embedding_start = 0.0
        embedding_time = 0.0
        page_count = 0
        
        try:
            file_path_str = str(doc_path)
            documents = []
            
            # Parse document using LlamaParse if available
            if self.llama_parse_config and doc_path.suffix.lower() in LLAMAPARSE_SUPPORTED_EXTENSIONS:
                try:
                    # Check if this is a PDF and we should split it
                    if doc_path.suffix.lower() == '.pdf':
                        # Read PDF file as bytes
                        with open(doc_path, 'rb') as f:
                            file_bytes = f.read()
                        
                        # Split into 20-page batches
                        PAGE_BATCH_SIZE = 20
                        batches = self._split_pdf_into_batches(file_bytes, batch_size=PAGE_BATCH_SIZE)
                        
                        if len(batches) > 1:
                            print(f"   📄 Splitting PDF into {len(batches)} batch(es) of up to {PAGE_BATCH_SIZE} pages")
                            
                            # Parse all batches concurrently
                            async def parse_batch(batch_bytes: bytes, start_page: int, batch_idx: int) -> Tuple[Optional[Any], int]:
                                """Parse a single PDF batch directly from bytes."""
                                # Pass bytes directly to LlamaParse (no disk I/O needed)
                                if not self.llama_parse_parser:
                                    return None, start_page
                                parser = self.llama_parse_parser
                                # Create a file-like object from bytes for LlamaParse
                                file_obj = io.BytesIO(batch_bytes)
                                # LlamaParse requires extra_info with file_name when passing bytes
                                # Use batch-specific filename to help identify the batch
                                batch_filename = f"{doc_path.stem}_batch{batch_idx + 1}.pdf"
                                extra_info = {"file_name": batch_filename}
                                # Use aparse with bytes/file-like object
                                result = await parser.aparse(file_obj, extra_info=extra_info)
                                return result, start_page
                            
                            # Parse all batches concurrently
                            results = await asyncio.gather(*[
                                parse_batch(batch_bytes, start_page, idx)
                                for idx, (batch_bytes, start_page) in enumerate(batches)
                            ], return_exceptions=True)
                            
                            # Collect all pages from all batches, adjusting page numbers
                            all_pages = []
                            failed_batches = []
                            
                            for idx, result_data in enumerate(results):
                                if isinstance(result_data, Exception):
                                    failed_batches.append(f"Batch {idx + 1}: {result_data}")
                                    continue
                                
                                result, start_page = result_data
                                
                                if result is None or not hasattr(result, 'pages') or not result.pages:
                                    failed_batches.append(f"Batch {idx + 1}: No pages")
                                    continue
                                
                                # Adjust page numbers to absolute page numbers
                                for page in result.pages:
                                    # Page numbers from LlamaParse are relative to the batch (1-N)
                                    # We need to convert to absolute page numbers
                                    if hasattr(page, 'page') and page.page is not None:
                                        # Adjust: batch page 1 -> absolute page (start_page)
                                        # batch page 2 -> absolute page (start_page + 1), etc.
                                        absolute_page = start_page + (page.page - 1)
                                        # Store absolute page number for later use
                                        page._absolute_page = absolute_page
                                    all_pages.append(page)
                            
                            if failed_batches:
                                print(f"   ⚠️  {len(failed_batches)} batch(es) failed: {failed_batches[:3]}")
                                if len(failed_batches) > 3:
                                    print(f"      ... and {len(failed_batches) - 3} more")
                            
                            if not all_pages:
                                parsing_time = time.time() - parsing_start
                                return False, 0, 0, "No pages extracted from any batches", parsing_time, 0.0
                            
                            page_count = len(all_pages)
                            
                            # Sort pages by absolute page number
                            all_pages.sort(key=lambda p: getattr(p, '_absolute_page', p.page if hasattr(p, 'page') and p.page is not None else 0))
                            
                            # Reconstruct markdown with page markers at the bottom using absolute page numbers
                            combined_markdown_parts = []
                            for page in all_pages:
                                # Use absolute page number if available, otherwise use original
                                page_num = getattr(page, '_absolute_page', page.page if hasattr(page, 'page') and page.page is not None else 0)
                                page_content = page.md if page.md is not None else (page.text if hasattr(page, 'text') and page.text else "")
                                combined_markdown_parts.append(page_content)
                                page_separator = f"\n<!-- PAGE: {page_num} -->\n"
                                combined_markdown_parts.append(page_separator)
                                combined_markdown_parts.append("\n")
                            
                            combined_markdown = "".join(combined_markdown_parts)
                            
                            doc = Document(
                                text=combined_markdown,
                                metadata={
                                    'file_path': file_path_str,
                                    'file_name': doc_path.name
                                }
                            )
                            documents = [doc]
                        else:
                            # Single batch - parse from bytes
                            batch_bytes, start_page = batches[0]
                            doc, page_count = await self._parse_bytes_and_reconstruct(batch_bytes, file_path_str, doc_path.name)
                            if doc is None:
                                parsing_time = time.time() - parsing_start
                                return False, 0, 0, "No content extracted from LlamaParse", parsing_time, 0.0
                            documents = [doc]
                    else:
                        # Non-PDF document - read as bytes and parse
                        with open(doc_path, 'rb') as f:
                            file_bytes = f.read()
                        
                        doc, page_count = await self._parse_bytes_and_reconstruct(file_bytes, file_path_str, doc_path.name)
                        if doc is None:
                            parsing_time = time.time() - parsing_start
                            return False, 0, 0, "No content extracted from LlamaParse", parsing_time, 0.0
                        documents = [doc]
                        
                except Exception as e:
                    # Log the error for debugging
                    error_details = str(e)
                    if hasattr(e, '__cause__') and e.__cause__:
                        error_details += f" (caused by: {e.__cause__})"
                    print(f"   ⚠️  LlamaParse async parsing failed: {error_details}")
                    
                    # Fall back to sync parsing
                    try:
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
                        documents = reader.load_data()
                        # For sync-parsed documents, estimate page count from document text length
                        # (rough estimate: ~2000 chars per page)
                        if documents:
                            total_chars = sum(len(doc.text) for doc in documents)
                            page_count = max(1, total_chars // 2000)
                    except Exception as sync_error:
                        parsing_time = time.time() - parsing_start
                        return False, 0, 0, f"Parsing failed (async: {e}, sync: {sync_error})", parsing_time, 0.0
            
            else:
                # Non-LlamaParse document - use sync parsing
                try:
                    file_extractor = {".pdf": PDFReader()} if doc_path.suffix.lower() == '.pdf' else None
                    reader = SimpleDirectoryReader(
                        input_files=[file_path_str],
                        file_extractor=file_extractor
                    )
                    documents = reader.load_data()
                    # For non-LlamaParse documents, estimate page count from document text length
                    # (rough estimate: ~2000 chars per page)
                    if documents:
                        total_chars = sum(len(doc.text) for doc in documents)
                        page_count = max(1, total_chars // 2000)
                except Exception as e:
                    parsing_time = time.time() - parsing_start
                    return False, 0, 0, f"Sync parsing failed: {e}", parsing_time, 0.0
            
            # Parsing complete
            parsing_time = time.time() - parsing_start
            
            if not documents:
                return False, page_count, 0, "No content extracted", parsing_time, 0.0
            
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
            
            if not nodes:
                return False, page_count, 0, "No nodes created from document", parsing_time, 0.0
            
            # Insert nodes into vector store using direct SDK (consistent with retrieval)
            # Track embedding time
            embedding_start = time.time()
            collection_name = f"{self.collection}_vectors"
            
            try:
                if self.vector_store_type == "qdrant":
                    # Use direct Qdrant SDK for indexing (consistent with retrieval)
                    success = await self._index_nodes_to_qdrant_direct(nodes, collection_name)
                    if not success:
                        raise Exception("Failed to index nodes to Qdrant")
                    
                    # Mark index as created in state
                    if not self.state.get("index_created", False):
                        self.state["index_created"] = True
                        self.state["vector_store_type"] = self.vector_store_type
                        self._save_state()
                else:
                    # For ChromaDB, still use LlamaIndex wrapper (for now)
                    create_new = not self.index
                    if create_new:
                        vector_store = self._create_vector_store(collection_name)
                        storage_context = StorageContext.from_defaults(vector_store=vector_store)
                        self.index = VectorStoreIndex(nodes=nodes, storage_context=storage_context)
                        self.state["index_created"] = True
                        self.state["vector_store_type"] = self.vector_store_type
                        self._save_state()
                    else:
                        self.index.insert_nodes(nodes)
            except Exception as index_error:
                # If index creation failed (e.g., race condition), try again
                if self.vector_store_type == "qdrant":
                    # For Qdrant, try once more
                    success = await self._index_nodes_to_qdrant_direct(nodes, collection_name)
                    if not success:
                        raise
                else:
                    # For ChromaDB, try inserting if index exists
                    if self.index:
                        self.index.insert_nodes(nodes)
                    else:
                        raise
            
            embedding_time = time.time() - embedding_start
            
            # Update state for this document
            self._mark_document_indexed(doc_path, page_count, len(nodes))
            
            return True, page_count, len(nodes), None, parsing_time, embedding_time
            
        except Exception as e:
            error_msg = str(e)
            parsing_time = time.time() - parsing_start
            self._mark_document_failed(doc_path, error_msg)
            return False, page_count, 0, error_msg, parsing_time, embedding_time
    
    def _process_documents_parallel(self, documents: List[Tuple[Path, str]]) -> Tuple[List[Any], List[Tuple[Path, int, int]], List[Tuple[Path, str]]]:
        """
        Process all documents in batches with concurrent processing per batch.
        
        Documents are split into batches of DOCUMENT_BATCH_SIZE. For each batch,
        documents are processed concurrently: parse → chunk → embed → insert.
        
        Args:
            documents: List of (doc_path, status) tuples to process
            
        Returns:
            Tuple of (all_nodes, successful_docs, failed_docs)
            - all_nodes: Always empty list (nodes are inserted immediately per document, not collected)
            - successful_docs: List of (doc_path, page_count, chunk_count) tuples
            - failed_docs: List of (doc_path, error_message) tuples
        """
        successful = []
        failed = []
        
        if not documents:
            return [], successful, failed
        
        # Split documents into batches
        batch_size = DOCUMENT_BATCH_SIZE
        batches = []
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batches.append(batch)
        
        print(f"   📦 Processing {len(documents)} document(s) in {len(batches)} batch(es) of up to {batch_size} document(s) each")
        
        # Process each batch concurrently
        async def process_batch(batch: List[Tuple[Path, str]]):
            """Process a batch of documents concurrently."""
            tasks = [
                self._process_single_document_async(doc_path, status)
                for doc_path, status in batch
            ]
            return await asyncio.gather(*tasks)
        
        # Process all batches
        async def process_all_batches():
            """Process all batches sequentially (batches are processed sequentially, documents within batch concurrently)."""
            all_results = []
            for batch_num, batch in enumerate(batches, 1):
                if len(batches) > 1:
                    print(f"   Processing batch {batch_num}/{len(batches)} ({len(batch)} document(s))...")
                batch_results = await process_batch(batch)
                all_results.extend(batch_results)
            return all_results
        
        try:
            results = asyncio.run(process_all_batches())
        except RuntimeError as e:
            if "asyncio.run() cannot be called from a running event loop" in str(e):
                loop = asyncio.get_event_loop()
                results = loop.run_until_complete(process_all_batches())
            else:
                raise
        
        # Process results and aggregate timing
        total_parsing_time = 0.0
        total_embedding_time = 0.0
        
        for (doc_path, status), result in zip(documents, results):
            success, page_count, chunk_count, error, parsing_time, embedding_time = result
            if success:
                successful.append((doc_path, page_count, chunk_count))
                total_parsing_time += parsing_time
                total_embedding_time += embedding_time
            else:
                failed.append((doc_path, error or "Unknown error"))
                total_parsing_time += parsing_time
        
        # Store aggregated timing for use in update_documents
        self._last_parsing_time = total_parsing_time
        self._last_embedding_time = total_embedding_time
        
        # Return empty nodes list since nodes are inserted immediately per document
        return [], successful, failed
    
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
            print(f"🔄 Updating RAG index for collection '{self.collection}'...")
            
            # Phase 1: Scan for documents
            print(f"\n📁 Phase 1/6: Scanning directory...")
            current_docs = self._scan_documents()
            
            if not current_docs:
                print(f"⚠️  No documents found in {self.documents_dir}")
                return True
            
            print(f"✅ Scanned directory")
            
            # Phase 2: Load existing index
            print(f"\n📖 Phase 2/6: Loading existing index...")
            
            # Check if vector store type changed (migration scenario)
            stored_vector_store_type = self.state.get("vector_store_type")
            if stored_vector_store_type and stored_vector_store_type != self.vector_store_type:
                print(f"   ⚠️  Vector store type changed from {stored_vector_store_type} to {self.vector_store_type}")
                print(f"   ⚠️  Documents will be re-indexed into the new vector store")
                # Clear index_created flag to force re-indexing
                self.state["index_created"] = False
                # Mark all documents as needing re-indexing
                for doc_path in self.state.get("documents", {}):
                    self.state["documents"][doc_path]["indexed"] = False
                self._save_state()
            
            if self.state.get("index_created", False) and not self.index:
                try:
                    collection_name = f"{self.collection}_vectors"
                    vector_store = self._get_vector_store_for_loading(collection_name)
                    self.index = VectorStoreIndex.from_vector_store(vector_store)
                    print(f"✅ Loaded existing index")
                except Exception as e:
                    print(f"   ⚠️  No existing index found, will create new one")
                    self.index = None
            else:
                print(f"   No existing index to load")
            
            # Phase 3: Identify documents to process
            print(f"\n🔍 Phase 3/6: Identifying documents to process...")
            
            to_process = self._get_documents_to_process()
            deleted = self._get_deleted_documents()
            
            # Check if everything is up to date
            if not to_process and not deleted:
                indexed_count = self.state.get("indexed_pages", 0)
                print(f"✅ Index is up to date ({indexed_count} pages indexed)")
                return True
            
            # Summary
            print(f"📊 Found {len(current_docs)} total documents:")
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
            
            print(f"   Processing {len(to_process)} document(s) concurrently")
            
            # Process all documents in parallel (parsing + embedding happen per-document)
            all_nodes, successful_docs, failed_docs = self._process_documents_parallel(to_process)
            
            # Get aggregated timing from _process_documents_parallel
            parsing_elapsed = getattr(self, '_last_parsing_time', 0.0)
            embedding_elapsed = getattr(self, '_last_embedding_time', 0.0)
            
            # Update state for successful documents (already updated in _process_single_document_async)
            successful_pages = sum(page_count for _, page_count, _ in successful_docs)
            successful_count = len(successful_docs)
            
            # Update state for failed documents
            failed_count = len(failed_docs)
            for doc_path, error in failed_docs:
                # State already updated in _process_single_document_async
                pass
            
            print(f"\n✅ Processed {len(to_process)} documents")
            
            # Phase 5: Handle deleted documents
            if deleted:
                print(f"\n🗑️  Phase 5/6: Removing deleted documents...")
                
                for deleted_path in deleted:
                    print(f"   Removing {Path(deleted_path).name} from state...")
                    self._remove_document_from_state(deleted_path)
                
                print(f"✅ Removed {len(deleted)} document(s)")
            else:
                print(f"\n   Phase 5/6: No deleted documents")
            
            # Phase 6: Final summary
            print(f"\n🎉 Phase 6/6: Update complete!")
            print(f"   • Successfully indexed: {successful_pages} pages ({successful_count} documents)")
            if failed_count > 0:
                print(f"   • Failed: {failed_count} documents")
            if deleted:
                print(f"   • Removed: {len(deleted)}")
            
            indexed_count = self.state.get("indexed_pages", 0)
            total_count = self.state.get("total_pages", 0)
            print(f"   • Total in collection: {indexed_count}/{total_count} pages indexed")
            
            # Timing summary (only parsing and embedding)
            print(f"\n⏱️  Timing:")
            print(f"   • Parsing: {parsing_elapsed:.1f}s")
            print(f"   • Embedding: {embedding_elapsed:.1f}s")
            if successful_pages > 0:
                parsing_per_page = parsing_elapsed / successful_pages if parsing_elapsed > 0 else 0
                embedding_per_page = embedding_elapsed / successful_pages if embedding_elapsed > 0 else 0
                print(f"   • Speed: {parsing_per_page:.1f}s/page (parsing), {embedding_per_page:.1f}s/page (embedding)")
            
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
            if not self.index:
                # Check if vector store type matches
                stored_vector_store_type = self.state.get("vector_store_type")
                if stored_vector_store_type and stored_vector_store_type != self.vector_store_type:
                    return f"Vector store type mismatch. Documents were indexed with {stored_vector_store_type}, but current store is {self.vector_store_type}. Please re-index with: python document_manager.py --collection {self.collection} --update"
                
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
                    collection_name = f"{self.collection}_vectors"
                    vector_store = self._get_vector_store_for_loading(collection_name)
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
            retrieve_start = time.time()
            nodes = retriever.retrieve(query)
            retrieve_elapsed = time.time() - retrieve_start
            
            if not nodes:
                query_elapsed = time.time() - query_start
                print(f"⏱️ Query time: {query_elapsed:.2f}s (no results found)")
                return "No relevant documents found for your query."
            
            print(f"📊 Retrieved {len(nodes)} chunks ({retrieve_elapsed:.2f}s)")
            
            # Apply reranking if enabled
            if self.cohere_reranker and len(nodes) > 1:
                print(f"🔄 Reranking {len(nodes)} chunks...")
                rerank_start = time.time()
                nodes = self.cohere_reranker.postprocess_nodes(nodes, query_str=query)
                rerank_elapsed = time.time() - rerank_start
                print(f"✅ Reranked to {len(nodes)} chunks ({rerank_elapsed:.2f}s)")
            
            # Format response
            response_parts = []
            for i, node in enumerate(nodes, 1):
                metadata = node.metadata or {}
                file_name = metadata.get('file_name', 'Unknown')
                page_label = metadata.get('page_label', '')
                
                response_parts.append(f"\n--- Chunk {i} (from {file_name}")
                if page_label:
                    response_parts.append(f", {page_label}")
                response_parts.append(") ---\n")
                response_parts.append(node.text)
                response_parts.append("\n")
            
            query_elapsed = time.time() - query_start
            print(f"⏱️  Total query time: {query_elapsed:.2f}s")
            
            return "".join(response_parts)
            
        except Exception as e:
            print(f"❌ Error querying documents: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def delete_documents(self) -> bool:
        """Delete the entire collection (remove storage directory and update global state)."""
        try:
            import shutil
            
            # Delete vector store collection
            collection_name = f"{self.collection}_vectors"
            self._delete_collection(collection_name)
            
            # Remove collection from global state
            if "collections" in self.global_state and self.collection in self.global_state["collections"]:
                del self.global_state["collections"][self.collection]
                self.global_state["last_updated"] = dt.now().isoformat()
                # Save updated global state
                try:
                    with open(self.state_file, 'w', encoding='utf-8') as f:
                        json.dump(self.global_state, f, indent=2, ensure_ascii=False)
                    print(f"🗑️  Removed collection '{self.collection}' from global state")
                except Exception as e:
                    print(f"⚠️  Warning: Could not update global state: {e}")
            
            # Remove storage directory (for ChromaDB only, Qdrant has no local storage)
            if self.storage_dir and self.storage_dir.exists():
                shutil.rmtree(self.storage_dir)
                print(f"🗑️  Deleted collection storage directory: {self.storage_dir}")
            elif self.storage_dir is None:
                print(f"ℹ️  No local storage directory for Qdrant (service-based)")
            else:
                print(f"ℹ️  Collection storage directory '{self.collection}' not found")
            
            print(f"✅ Deleted collection '{self.collection}'")
            return True
            
        except Exception as e:
            print(f"❌ Error deleting collection: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the collection."""
        documents = self.state.get("documents", {})
        total_documents = len([doc for doc in documents.values() if doc.get("indexed", False)])
        
        return {
            "total_documents": total_documents,
            "total_pages": self.state.get("total_pages", 0),
            "last_updated": self.state.get("last_updated"),
            "storage_dir": str(self.storage_dir) if self.storage_dir else None,
            "vector_store_type": self.vector_store_type,
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
    def list_all_collections(cls, storage_dir: str = None) -> List[str]:
        """List all collections from global state files (class method)."""
        collections = set()
        
        # Read from ChromaDB global state file
        chroma_state_file = Path(RAG_CHROMA_STATE_FILE)
        if chroma_state_file.exists():
            try:
                with open(chroma_state_file, 'r', encoding='utf-8') as f:
                    chroma_state = json.load(f)
                    if "collections" in chroma_state:
                        collections.update(chroma_state["collections"].keys())
            except Exception:
                pass
        
        # Read from Qdrant global state file
        qdrant_state_file = Path(RAG_QDRANT_STATE_FILE)
        if qdrant_state_file.exists():
            try:
                with open(qdrant_state_file, 'r', encoding='utf-8') as f:
                    qdrant_state = json.load(f)
                    if "collections" in qdrant_state:
                        collections.update(qdrant_state["collections"].keys())
            except Exception:
                pass
        
        return sorted(collections)

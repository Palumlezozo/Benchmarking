"""
Azure AI Search Document Store Implementation.

This module provides a document store implementation using Azure AI Search
with vector embeddings for semantic search.
"""

import os
import hashlib
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SimpleField,
    SearchableField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
)
from azure.search.documents.models import VectorizedQuery
from openai import OpenAI

from .document_store_base import DocumentStore
from .config import DEFAULT_MODEL, DEFAULT_REASONING_EFFORT, DEFAULT_TEXT_VERBOSITY

# Load environment variables
load_dotenv()

# Constants
DOCUMENTS_DIR = "data/documents"
DEFAULT_INDEX_NAME_PREFIX = "documents-index"
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".json"}
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSIONS = 3072  # text-embedding-3-large dimension
DEFAULT_CHUNK_SIZE = 1024
DEFAULT_CHUNK_OVERLAP = 100


class AzureDocumentStore(DocumentStore):
    """Document store implementation using Azure AI Search."""
    
    def __init__(
        self,
        collection: str,
        model: str = DEFAULT_MODEL,
        reasoning_effort: str = DEFAULT_REASONING_EFFORT,
        text_verbosity: str = DEFAULT_TEXT_VERBOSITY,
        embedding_model: str = EMBEDDING_MODEL,
        **kwargs
    ):
        """
        Initialize the Azure Document Store.
        
        Args:
            collection: Collection name
            model: Model to use for queries (for future LLM integration)
            reasoning_effort: Reasoning effort level (for future LLM integration)
            text_verbosity: Text verbosity level (for future LLM integration)
            embedding_model: Embedding model to use
            **kwargs: Additional arguments (ignored)
        """
        super().__init__(collection)
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.text_verbosity = text_verbosity
        self.embedding_model = embedding_model
        
        # Azure configuration
        self.azure_search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
        self.azure_search_api_key = os.getenv("AZURE_SEARCH_API_KEY")
        self.azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_openai_base_url = os.getenv("AZURE_OPENAI_BASE_URL")
        
        # Validate environment
        self._check_environment()
        
        # Index name
        self.index_name = f"{DEFAULT_INDEX_NAME_PREFIX}-{collection}"
        
        # Initialize clients
        self.index_client = SearchIndexClient(
            endpoint=self.azure_search_endpoint,
            credential=AzureKeyCredential(self.azure_search_api_key)
        )
        
        self.search_client = SearchClient(
            endpoint=self.azure_search_endpoint,
            index_name=self.index_name,
            credential=AzureKeyCredential(self.azure_search_api_key)
        )
        
        self.openai_client = OpenAI(
            api_key=self.azure_openai_api_key,
            base_url=self.azure_openai_base_url
        )
    
    def _check_environment(self):
        """Check if all required environment variables are set."""
        missing = []
        if not self.azure_search_endpoint:
            missing.append("AZURE_SEARCH_ENDPOINT")
        if not self.azure_search_api_key:
            missing.append("AZURE_SEARCH_API_KEY")
        if not self.azure_openai_api_key:
            missing.append("AZURE_OPENAI_API_KEY")
        if not self.azure_openai_base_url:
            missing.append("AZURE_OPENAI_BASE_URL")
        
        if missing:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing)}\n"
                "Please add these to your .env file."
            )
    
    def _get_collection_dir(self) -> Path:
        """Get the documents directory for the collection."""
        return Path(DOCUMENTS_DIR) / self.collection
    
    def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using Azure OpenAI."""
        try:
            # Truncate text if too long
            max_chars = 30000
            if len(text) > max_chars:
                text = text[:max_chars]
            
            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"⚠️  Warning: Failed to generate embedding: {e}")
            return [0.0] * EMBEDDING_DIMENSIONS
    
    def _get_embedding_batch(self, texts: List[str], max_batch_size: int = 100) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batch (10-20x faster!).
        
        Args:
            texts: List of texts to embed
            max_batch_size: Maximum batch size (Azure OpenAI limit)
        
        Returns:
            List of embeddings (same order as input texts)
        """
        if not texts:
            return []
        
        # Truncate texts if needed
        max_chars = 30000
        truncated_texts = [t[:max_chars] if len(t) > max_chars else t for t in texts]
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(truncated_texts), max_batch_size):
            batch = truncated_texts[i:i + max_batch_size]
            
            try:
                response = self.openai_client.embeddings.create(
                    model=self.embedding_model,
                    input=batch
                )
                # Extract embeddings in order
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"⚠️  Warning: Batch embedding failed, falling back to individual: {e}")
                # Fallback to individual embeddings
                for text in batch:
                    all_embeddings.append(self._get_embedding(text))
        
        return all_embeddings
    
    def _chunk_text(self, text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, 
                   chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[str]:
        """
        Split text into overlapping chunks using sliding window approach.
        Same as Azure Indexer approach for consistency.
        
        Args:
            text: Text to chunk
            chunk_size: Size of each chunk in characters
            chunk_overlap: Number of overlapping characters between chunks
            
        Returns:
            List of text chunks
        """
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        stride = chunk_size - chunk_overlap
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            if chunk.strip():
                chunks.append(chunk)
            
            start += stride
            
            if start < len(text) and start + chunk_size > len(text):
                pass
        
        return chunks
    
    def _extract_text_from_pdf(self, filepath: Path) -> str:
        """Extract text from PDF file."""
        try:
            from pypdf import PdfReader
            
            reader = PdfReader(str(filepath))
            text_parts = []
            
            for page_num, page in enumerate(reader.pages, 1):
                text = page.extract_text()
                if text.strip():
                    text_parts.append(f"[Page {page_num}]\n{text}")
            
            return "\n\n".join(text_parts)
        except Exception as e:
            print(f"⚠️  Warning: Failed to extract text from {filepath.name}: {e}")
            return ""
    
    def _extract_text_from_file(self, filepath: Path) -> str:
        """Extract text content from a file based on its type."""
        if filepath.suffix.lower() == ".pdf":
            return self._extract_text_from_pdf(filepath)
        elif filepath.suffix.lower() in {".txt", ".md"}:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                print(f"⚠️  Warning: Failed to read {filepath.name}: {e}")
                return ""
        elif filepath.suffix.lower() == ".json":
            try:
                import json
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return json.dumps(data, indent=2)
            except Exception as e:
                print(f"⚠️  Warning: Failed to read {filepath.name}: {e}")
                return ""
        return ""
    
    def _collect_documents(self, directory: Path) -> List[Path]:
        """Collect all supported documents from directory."""
        if not directory.exists():
            return []
        
        documents = []
        for file_path in directory.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                documents.append(file_path)
        
        return sorted(documents)
    
    def _create_or_update_index(self):
        """Create or update the Azure AI Search index."""
        fields = [
            SimpleField(
                name="id",
                type=SearchFieldDataType.String,
                key=True,
                filterable=True,
            ),
            SearchableField(
                name="title",
                type=SearchFieldDataType.String,
                searchable=True,
                filterable=True,
                sortable=True,
            ),
            SearchableField(
                name="content",
                type=SearchFieldDataType.String,
                searchable=True,
            ),
            SearchableField(
                name="filepath",
                type=SearchFieldDataType.String,
                searchable=True,
                filterable=True,
            ),
            SimpleField(
                name="file_type",
                type=SearchFieldDataType.String,
                filterable=True,
            ),
            SimpleField(
                name="file_size",
                type=SearchFieldDataType.Int64,
                filterable=True,
                sortable=True,
            ),
            SimpleField(
                name="indexed_at",
                type=SearchFieldDataType.DateTimeOffset,
                filterable=True,
                sortable=True,
            ),
            SimpleField(
                name="chunk_number",
                type=SearchFieldDataType.Int32,
                filterable=True,
                sortable=True,
            ),
            SimpleField(
                name="total_chunks",
                type=SearchFieldDataType.Int32,
                filterable=True,
                sortable=True,
            ),
            SimpleField(
                name="chunk_id",
                type=SearchFieldDataType.String,
                filterable=True,
            ),
            SearchField(
                name="content_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=EMBEDDING_DIMENSIONS,
                vector_search_profile_name="vector-profile",
            ),
        ]
        
        vector_search = VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(
                    name="hnsw-config",
                    parameters={
                        "m": 4,
                        "efConstruction": 400,
                        "efSearch": 500,
                        "metric": "cosine"
                    }
                )
            ],
            profiles=[
                VectorSearchProfile(
                    name="vector-profile",
                    algorithm_configuration_name="hnsw-config",
                )
            ],
        )
        
        semantic_config = SemanticConfiguration(
            name="semantic-config",
            prioritized_fields=SemanticPrioritizedFields(
                title_field=SemanticField(field_name="title"),
                content_fields=[SemanticField(field_name="content")],
            ),
        )
        
        semantic_search = SemanticSearch(configurations=[semantic_config])
        
        index = SearchIndex(
            name=self.index_name,
            fields=fields,
            vector_search=vector_search,
            semantic_search=semantic_search,
        )
        
        try:
            self.index_client.create_or_update_index(index)
            print(f"✅ Index '{self.index_name}' created/updated successfully")
        except Exception as e:
            print(f"❌ Error creating index: {e}")
            raise
    
    def update_documents(self, directory: Optional[str] = None, batch_size: int = 10) -> bool:
        """
        Update documents in Azure AI Search with batch embedding generation.
        
        Args:
            directory: Optional custom directory path (defaults to collection directory)
            batch_size: Batch size for uploading documents to Azure (default: 10)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            overall_start = time.time()
            
            # Determine directory
            if directory:
                collection_dir = Path(directory)
            else:
                collection_dir = self._get_collection_dir()
            
            # Phase 1: Scan for documents
            print(f"\n📁 Phase 1/4: Scanning directory: {collection_dir}")
            scan_start = time.time()
            documents = self._collect_documents(collection_dir)
            
            if not documents:
                print(f"⚠️  No supported documents found in {collection_dir}")
                print(f"   Supported extensions: {', '.join(SUPPORTED_EXTENSIONS)}")
                return False
            
            scan_elapsed = time.time() - scan_start
            print(f"✅ Found {len(documents)} document(s) ({scan_elapsed:.1f}s)")
            
            # Create/update index
            print(f"\n📋 Phase 2/4: Creating/updating search index...")
            index_start = time.time()
            self._create_or_update_index()
            index_elapsed = time.time() - index_start
            print(f"   ⏱️  Index setup: {index_elapsed:.1f}s")
            
            # Phase 3: Extract text from all documents
            print(f"\n📄 Phase 3/4: Extracting text from {len(documents)} document(s)...")
            extract_start = time.time()
            
            doc_metadata = []  # Store (doc_path, content) for batch processing
            skipped_count = 0
            
            for idx, doc_path in enumerate(documents, 1):
                print(f"   [{idx}/{len(documents)}] Extracting: {doc_path.name}...", end=" ", flush=True)
                
                try:
                    content = self._extract_text_from_file(doc_path)
                    
                    if not content or len(content.strip()) < 10:
                        print("⚠️  Skipped (no content)")
                        skipped_count += 1
                        continue
                    
                    print(f"✅ ({len(content)} chars)")
                    doc_metadata.append((doc_path, content))
                    
                except Exception as e:
                    print(f"❌ Error: {e}")
                    skipped_count += 1
            
            extract_elapsed = time.time() - extract_start
            print(f"\n✅ Extracted {len(doc_metadata)} document(s) ({extract_elapsed:.1f}s)")
            if skipped_count > 0:
                print(f"   ⚠️  Skipped {skipped_count} document(s)")
            
            if not doc_metadata:
                print("❌ No documents to index")
                return False
            
            # Phase 4: Chunk all documents
            print(f"\n✂️  Phase 4/6: Chunking {len(doc_metadata)} document(s)...")
            chunk_start = time.time()
            
            chunk_metadata = []  # Store (doc_path, chunk_idx, chunk_content, total_chunks)
            
            for doc_idx, (doc_path, content) in enumerate(doc_metadata, 1):
                # Chunk the content
                chunks = self._chunk_text(content)
                print(f"   [{doc_idx}/{len(doc_metadata)}] {doc_path.name}: {len(chunks)} chunks")
                
                # Store chunk metadata for batch processing
                for chunk_idx, chunk_content in enumerate(chunks):
                    chunk_metadata.append((doc_path, chunk_idx, chunk_content, len(chunks)))
            
            chunk_elapsed = time.time() - chunk_start
            print(f"✅ Chunked {len(doc_metadata)} docs into {len(chunk_metadata)} chunks ({chunk_elapsed:.1f}s)")
            
            # Phase 5: Generate embeddings in batch (MAJOR SPEEDUP!)
            print(f"\n🔢 Phase 5/6: Generating {len(chunk_metadata)} embeddings in batch...")
            embed_start = time.time()
            
            # Extract just the chunk text for batch embedding
            chunk_texts = [chunk_content for _, _, chunk_content, _ in chunk_metadata]
            
            # Generate all embeddings at once
            embeddings = self._get_embedding_batch(chunk_texts, max_batch_size=100)
            
            embed_elapsed = time.time() - embed_start
            print(f"✅ Generated {len(embeddings)} embeddings ({embed_elapsed:.1f}s, {len(embeddings)/embed_elapsed:.1f} emb/sec)")
            
            # Phase 6: Create and upload chunk documents
            print(f"\n⬆️  Phase 6/6: Uploading {len(chunk_metadata)} chunks to Azure Search...")
            upload_start = time.time()
            
            # Prepare all documents first
            all_documents = []
            for (doc_path, chunk_idx, chunk_content, total_chunks), embedding in zip(chunk_metadata, embeddings):
                try:
                    # Create unique chunk ID
                    path_hash = hashlib.md5(str(doc_path).encode()).hexdigest()[:16]
                    base_id = f"{doc_path.stem}_{path_hash}".replace(".", "_").replace(" ", "_")[:40]
                    chunk_doc_id = f"{base_id}_chunk_{chunk_idx}"
                    
                    # Create chunk document with metadata
                    document = {
                        "id": chunk_doc_id,
                        "title": doc_path.stem,
                        "content": chunk_content,
                        "filepath": str(doc_path),
                        "file_type": doc_path.suffix.lower(),
                        "file_size": doc_path.stat().st_size,
                        "indexed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                        "chunk_number": chunk_idx,
                        "total_chunks": total_chunks,
                        "chunk_id": base_id,  # Reference to original document
                        "content_vector": embedding,
                    }
                    all_documents.append(document)
                except Exception as e:
                    print(f"\n   ❌ Error processing chunk {chunk_idx} of {doc_path.name}: {e}")
            
            # Split into batches
            batches = []
            for i in range(0, len(all_documents), batch_size):
                batches.append(all_documents[i:i + batch_size])
            
            print(f"   Uploading {len(batches)} batches in parallel (batch size: {batch_size})...")
            
            # Upload batches in parallel
            def upload_batch(batch_idx_and_docs):
                batch_idx, docs = batch_idx_and_docs
                try:
                    result = self.search_client.upload_documents(documents=docs)
                    succeeded = sum(1 for r in result if r.succeeded)
                    failed = len(docs) - succeeded
                    return succeeded, failed
                except Exception as e:
                    print(f"\n   ❌ Batch {batch_idx + 1} upload error: {e}")
                    return 0, len(docs)
            
            # Use ThreadPoolExecutor for parallel uploads
            indexed_count = 0
            failed_count = 0
            
            with ThreadPoolExecutor(max_workers=4) as executor:
                batch_results = list(executor.map(upload_batch, enumerate(batches)))
                
                for succeeded, failed in batch_results:
                    indexed_count += succeeded
                    failed_count += failed
            
            upload_elapsed = time.time() - upload_start
            print(f"\n✅ Uploaded {indexed_count} chunk(s) ({upload_elapsed:.1f}s)")
            
            if failed_count > 0:
                print(f"❌ Failed: {failed_count} chunk(s)")
            
            # Overall summary
            overall_elapsed = time.time() - overall_start
            print(f"\n🎉 All Done! Total time: {overall_elapsed:.1f}s")
            print(f"   📊 Time breakdown:")
            print(f"      • Scanning: {scan_elapsed:.1f}s")
            print(f"      • Index setup: {index_elapsed:.1f}s")
            print(f"      • Text extraction: {extract_elapsed:.1f}s")
            print(f"      • Chunking: {chunk_elapsed:.1f}s")
            print(f"      • Embeddings: {embed_elapsed:.1f}s ({len(embeddings)/embed_elapsed:.1f} emb/sec)")
            print(f"      • Upload: {upload_elapsed:.1f}s")
            print(f"   📈 Stats:")
            print(f"      • Documents: {len(doc_metadata)}")
            print(f"      • Total chunks: {len(chunk_metadata)}")
            print(f"      • Avg chunks/doc: {len(chunk_metadata) / len(doc_metadata):.1f}")
            
            return indexed_count > 0
            
        except Exception as e:
            print(f"❌ Error updating documents: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def query_documents(self, query: str, **kwargs) -> Optional[str]:
        """Query documents using Azure AI Search."""
        try:
            query_start = time.time()
            
            # Increase default to return more results and more content
            top = kwargs.get('top', 10)  # Return more results
            use_semantic = kwargs.get('use_semantic', False)
            use_vector_search = kwargs.get('use_vector_search', True)
            max_content_length = kwargs.get('max_content_length', 8000)  # Much more content per result
            
            # Generate query embedding if using vector search
            if use_vector_search:
                embed_start = time.time()
                query_vector = self._get_embedding(query)
                embed_elapsed = time.time() - embed_start
                
                # Vector search
                vector_query = VectorizedQuery(
                    vector=query_vector,
                    k_nearest_neighbors=top,
                    fields="content_vector"
                )
            else:
                vector_query = None
                embed_elapsed = 0
            
            # Perform search
            search_start = time.time()
            if use_semantic:
                if vector_query:
                    results = self.search_client.search(
                        search_text=query,
                        vector_queries=[vector_query],
                        query_type="semantic",
                        semantic_configuration_name="semantic-config",
                        top=top,
                    )
                else:
                    results = self.search_client.search(
                        search_text=query,
                        query_type="semantic",
                        semantic_configuration_name="semantic-config",
                        top=top,
                    )
            else:
                if vector_query:
                    results = self.search_client.search(
                        search_text=query,
                        vector_queries=[vector_query],
                        top=top,
                    )
                else:
                    results = self.search_client.search(
                        search_text=query,
                        top=top,
                    )
            
            search_elapsed = time.time() - search_start
            
            # Format results with chunk information
            response_parts = []
            for idx, result in enumerate(results, 1):
                score = result.get('@search.score', 0)
                title = result['title']
                filepath = result['filepath']
                
                # Get chunk information if available
                chunk_number = result.get('chunk_number')
                total_chunks = result.get('total_chunks')
                
                # Return much more content (up to max_content_length)
                content = result['content'][:max_content_length]
                
                # Format result with chunk info
                if chunk_number is not None and total_chunks is not None:
                    result_header = f"[Result {idx}] {title} (Chunk {chunk_number + 1}/{total_chunks}) (Score: {score:.4f})"
                else:
                    result_header = f"[Result {idx}] {title} (Score: {score:.4f})"
                
                response_parts.append(
                    f"{result_header}\n"
                    f"File: {filepath}\n"
                    f"Content: {content}\n"
                    f"{'...(truncated)' if len(result['content']) > max_content_length else ''}\n"
                )
            
            query_elapsed = time.time() - query_start
            
            # Add timing info
            timing_info = f"\n⏱️  Query timing: {query_elapsed:.2f}s total"
            if use_vector_search:
                timing_info += f" (embedding: {embed_elapsed:.2f}s, search: {search_elapsed:.2f}s)"
            else:
                timing_info += f" (search: {search_elapsed:.2f}s)"
            
            if response_parts:
                return "\n".join(response_parts) + timing_info
            else:
                return "No results found." + timing_info
                
        except Exception as e:
            print(f"❌ Error querying documents: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def delete_documents(self) -> bool:
        """Delete the entire index for this collection."""
        try:
            self.index_client.delete_index(self.index_name)
            print(f"✅ Index '{self.index_name}' deleted successfully")
            return True
        except Exception as e:
            print(f"❌ Error deleting index: {e}")
            return False
    
    def get_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the document store."""
        try:
            # Try to get index statistics
            index = self.index_client.get_index(self.index_name)
            
            # Count documents
            results = self.search_client.search(
                search_text="*",
                select=["id"],
                top=0,
                include_total_count=True
            )
            
            doc_count = results.get_count() if hasattr(results, 'get_count') else 0
            
            return {
                "total_documents": doc_count,
                "index_name": self.index_name,
                "collection": self.collection,
                "last_updated": "N/A",  # Azure doesn't track this automatically
            }
        except Exception as e:
            # Index doesn't exist
            return None
    
    def list_documents(self) -> List[str]:
        """List all documents in the collection (with pagination support)."""
        try:
            all_filepaths = []
            skip = 0
            page_size = 1000
            
            while True:
                results = list(self.search_client.search(
                    search_text="*",
                    select=["filepath"],
                    top=page_size,
                    skip=skip
                ))
                
                if not results:
                    break
                
                all_filepaths.extend([result['filepath'] for result in results])
                skip += page_size
                
                # If we got fewer results than page_size, we're done
                if len(results) < page_size:
                    break
            
            return all_filepaths
        except Exception as e:
            print(f"❌ Error listing documents: {e}")
            return []
    
    @staticmethod
    def list_all_collections() -> List[str]:
        """List all collections with Azure AI Search indices."""
        try:
            azure_search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
            azure_search_api_key = os.getenv("AZURE_SEARCH_API_KEY")
            
            if not azure_search_endpoint or not azure_search_api_key:
                return []
            
            index_client = SearchIndexClient(
                endpoint=azure_search_endpoint,
                credential=AzureKeyCredential(azure_search_api_key)
            )
            
            indices = index_client.list_indexes()
            collections = []
            
            for index in indices:
                # Extract collection name from index name
                if index.name.startswith(DEFAULT_INDEX_NAME_PREFIX + "-"):
                    collection = index.name[len(DEFAULT_INDEX_NAME_PREFIX) + 1:]
                    collections.append(collection)
            
            return sorted(collections)
        except Exception:
            return []


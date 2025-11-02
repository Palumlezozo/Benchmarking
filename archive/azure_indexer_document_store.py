"""
Azure AI Search Indexer Document Store Implementation.

This module provides a document store implementation using Azure AI Search
with Azure Blob Storage and Indexers for automatic document processing.

Key Differences from azure_document_store.py:
- Documents stored in Azure Blob Storage
- Azure Indexer automatically extracts and indexes content
- Supports scheduled automatic updates
- Can use skillsets for advanced processing (OCR, chunking, embeddings)
- No manual text extraction required
"""

import os
import hashlib
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional
import time

from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient, ContainerClient
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient, SearchIndexerClient
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
    SearchIndexer,
    SearchIndexerDataSourceConnection,
    SearchIndexerDataContainer,
    IndexingSchedule,
)
from azure.search.documents.models import VectorizedQuery
from openai import OpenAI

from .document_store_base import DocumentStore
from .config import DEFAULT_MODEL, DEFAULT_REASONING_EFFORT, DEFAULT_TEXT_VERBOSITY

# Load environment variables
load_dotenv()

# Constants
DOCUMENTS_DIR = "data/documents"
DEFAULT_INDEX_NAME_PREFIX = "documents-indexer-index"
DEFAULT_DATASOURCE_PREFIX = "documents-indexer-datasource"
DEFAULT_INDEXER_PREFIX = "documents-indexer"
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".json", ".docx", ".xlsx", ".pptx", ".html", ".xml"}
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSIONS = 3072  # text-embedding-3-large dimension

# Chunking configuration (same as manual approach)
DEFAULT_CHUNK_SIZE = 1024
DEFAULT_CHUNK_OVERLAP = 100


class AzureIndexerDocumentStore(DocumentStore):
    """Document store implementation using Azure AI Search Indexers with Blob Storage."""
    
    def __init__(
        self,
        collection: str,
        model: str = DEFAULT_MODEL,
        reasoning_effort: str = DEFAULT_REASONING_EFFORT,
        text_verbosity: str = DEFAULT_TEXT_VERBOSITY,
        embedding_model: str = EMBEDDING_MODEL,
        container_name: str = None,
        use_skillset: bool = False,
        schedule_interval_minutes: int = None,
        **kwargs
    ):
        """
        Initialize the Azure Indexer Document Store.
        
        Args:
            collection: Collection name
            model: Model to use for queries (for future LLM integration)
            reasoning_effort: Reasoning effort level
            text_verbosity: Text verbosity level
            embedding_model: Embedding model to use
            container_name: Blob container name (default: collection name)
            use_skillset: Whether to use skillsets for advanced processing
            schedule_interval_minutes: Indexer schedule interval (None = manual only)
            **kwargs: Additional arguments (ignored)
        """
        super().__init__(collection)
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.text_verbosity = text_verbosity
        self.embedding_model = embedding_model
        self.use_skillset = use_skillset
        self.schedule_interval_minutes = schedule_interval_minutes
        
        # Azure configuration
        self.azure_search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
        self.azure_search_api_key = os.getenv("AZURE_SEARCH_API_KEY")
        self.azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_openai_base_url = os.getenv("AZURE_OPENAI_BASE_URL")
        self.azure_storage_connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        
        # Validate environment
        self._check_environment()
        
        # Names
        self.container_name = container_name or f"documents-{collection}"
        self.index_name = f"{DEFAULT_INDEX_NAME_PREFIX}-{collection}"
        self.datasource_name = f"{DEFAULT_DATASOURCE_PREFIX}-{collection}"
        self.indexer_name = f"{DEFAULT_INDEXER_PREFIX}-{collection}"
        
        # Initialize clients
        self.index_client = SearchIndexClient(
            endpoint=self.azure_search_endpoint,
            credential=AzureKeyCredential(self.azure_search_api_key)
        )
        
        self.indexer_client = SearchIndexerClient(
            endpoint=self.azure_search_endpoint,
            credential=AzureKeyCredential(self.azure_search_api_key)
        )
        
        self.search_client = SearchClient(
            endpoint=self.azure_search_endpoint,
            index_name=self.index_name,
            credential=AzureKeyCredential(self.azure_search_api_key)
        )
        
        self.blob_service_client = BlobServiceClient.from_connection_string(
            self.azure_storage_connection_string
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
        if not self.azure_storage_connection_string:
            missing.append("AZURE_STORAGE_CONNECTION_STRING")
        
        if missing:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing)}\n"
                "Please add these to your .env file."
            )
    
    def _get_collection_dir(self) -> Path:
        """Get the local documents directory for the collection."""
        return Path(DOCUMENTS_DIR) / self.collection
    
    def _ensure_container_exists(self) -> ContainerClient:
        """Ensure blob container exists and return the client."""
        try:
            container_client = self.blob_service_client.get_container_client(self.container_name)
            # Try to get properties to check if exists
            container_client.get_container_properties()
            return container_client
        except:
            # Container doesn't exist, create it
            container_client = self.blob_service_client.create_container(self.container_name)
            print(f"✅ Created blob container: {self.container_name}")
            return container_client
    
    def _upload_to_blob(self, local_path: Path, blob_name: str = None) -> str:
        """
        Upload a file to blob storage.
        
        Args:
            local_path: Local file path
            blob_name: Blob name (default: use filename)
            
        Returns:
            Blob URL
        """
        container_client = self._ensure_container_exists()
        
        if blob_name is None:
            blob_name = local_path.name
        
        blob_client = container_client.get_blob_client(blob_name)
        
        with open(local_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)
        
        return blob_client.url
    
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
        Same as manual approach for consistency.
        
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
    
    def _collect_documents(self, directory: Path) -> List[Path]:
        """Collect all supported documents from directory."""
        if not directory.exists():
            return []
        
        documents = []
        for file_path in directory.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                # Skip hidden files and system files
                if not file_path.name.startswith('.') and not file_path.name.startswith('~'):
                    documents.append(file_path)
        
        return sorted(documents)
    
    def _create_index(self):
        """Create or update the Azure AI Search index."""
        fields = [
            SimpleField(
                name="id",
                type=SearchFieldDataType.String,
                key=True,
                filterable=True,
            ),
            SearchableField(
                name="metadata_storage_name",
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
                name="metadata_storage_path",
                type=SearchFieldDataType.String,
                searchable=True,
                filterable=True,
            ),
            SimpleField(
                name="metadata_storage_file_extension",
                type=SearchFieldDataType.String,
                filterable=True,
            ),
            SimpleField(
                name="metadata_storage_size",
                type=SearchFieldDataType.Int64,
                filterable=True,
                sortable=True,
            ),
            SimpleField(
                name="metadata_storage_last_modified",
                type=SearchFieldDataType.DateTimeOffset,
                filterable=True,
                sortable=True,
            ),
            # Chunking fields
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
            # Vector field for embeddings
            SearchField(
                name="content_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=EMBEDDING_DIMENSIONS,
                vector_search_profile_name="vector-profile",
            ),
        ]
        
        # Vector search configuration (now actively used!)
        
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
                title_field=SemanticField(field_name="metadata_storage_name"),
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
    
    def _create_datasource(self):
        """Create or update the data source connection to blob storage."""
        container = SearchIndexerDataContainer(name=self.container_name)
        
        data_source = SearchIndexerDataSourceConnection(
            name=self.datasource_name,
            type="azureblob",
            connection_string=self.azure_storage_connection_string,
            container=container
        )
        
        try:
            self.indexer_client.create_or_update_data_source_connection(data_source)
            print(f"✅ Data source '{self.datasource_name}' created/updated successfully")
        except Exception as e:
            print(f"❌ Error creating data source: {e}")
            raise
    
    def _create_indexer(self):
        """Create or update the indexer."""
        indexer_params = {
            "name": self.indexer_name,
            "data_source_name": self.datasource_name,
            "target_index_name": self.index_name,
            "parameters": {
                "configuration": {
                    "dataToExtract": "contentAndMetadata",
                    "parsingMode": "default",
                    "imageAction": "none",  # Set to "generateNormalizedImages" for OCR
                }
            }
        }
        
        # Add schedule if specified
        if self.schedule_interval_minutes:
            indexer_params["schedule"] = IndexingSchedule(
                interval=f"PT{self.schedule_interval_minutes}M"
            )
        
        indexer = SearchIndexer(**indexer_params)
        
        try:
            self.indexer_client.create_or_update_indexer(indexer)
            schedule_info = f" with {self.schedule_interval_minutes}min schedule" if self.schedule_interval_minutes else ""
            print(f"✅ Indexer '{self.indexer_name}' created/updated successfully{schedule_info}")
        except Exception as e:
            print(f"❌ Error creating indexer: {e}")
            raise
    
    def _enrich_with_embeddings(self) -> bool:
        """
        Generate and add vector embeddings to indexed documents WITH CHUNKING.
        
        This is called after the indexer finishes to add semantic search capabilities.
        The indexer extracts text, then we:
        1. Chunk the content (1024 chars, 100 overlap)
        2. Generate embeddings for ALL chunks in batch (10-20x faster!)
        3. Create multiple index entries (one per chunk)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            import time
            start_time = time.time()
            
            print("\n✂️  Step 7: Chunking documents and generating vector embeddings...")
            
            # Get all documents from the index (with pagination)
            all_docs = []
            skip = 0
            page_size = 1000
            
            while True:
                results = list(self.search_client.search(
                    search_text="*",
                    select=["id", "content", "metadata_storage_name", "metadata_storage_path", 
                           "metadata_storage_file_extension", "metadata_storage_size", 
                           "metadata_storage_last_modified", "chunk_id"],
                    top=page_size,
                    skip=skip
                ))
                
                if not results:
                    break
                
                # Filter out chunks (only keep original documents)
                original_docs = [r for r in results if not r.get('chunk_id')]
                all_docs.extend(original_docs)
                
                skip += page_size
                
                # If we got fewer results than page_size, we're done
                if len(results) < page_size:
                    break
            
            if not all_docs:
                print("   ℹ️  No documents to chunk (all already chunked)")
                return True
            
            print(f"   Found {len(all_docs)} document(s) to chunk")
            
            # Phase 1: Chunk all documents and collect metadata
            print("\n   Phase 1/3: Chunking documents...")
            chunk_time = time.time()
            
            chunk_metadata = []  # Store (doc, chunk_idx, chunk_content, doc_id)
            docs_to_delete = []
            
            for doc_idx, doc in enumerate(all_docs, 1):
                doc_id = doc.get('id')
                content = doc.get('content', '')
                doc_name = doc.get('metadata_storage_name', 'unknown')
                
                if not content or len(content.strip()) < 10:
                    print(f"      [{doc_idx}/{len(all_docs)}] Skipping {doc_name} (no content)")
                    continue
                
                # Chunk the content
                chunks = self._chunk_text(content)
                print(f"      [{doc_idx}/{len(all_docs)}] {doc_name}: {len(chunks)} chunks")
                
                # Mark original document for deletion
                docs_to_delete.append(doc_id)
                
                # Store chunk metadata for batch processing
                for chunk_idx, chunk_content in enumerate(chunks):
                    chunk_metadata.append((doc, chunk_idx, chunk_content, doc_id, len(chunks)))
            
            chunk_elapsed = time.time() - chunk_time
            print(f"   ✅ Chunked {len(all_docs)} docs into {len(chunk_metadata)} chunks ({chunk_elapsed:.1f}s)")
            
            # Phase 2: Generate embeddings in batch (MAJOR SPEEDUP!)
            print(f"\n   Phase 2/3: Generating {len(chunk_metadata)} embeddings in batch...")
            embed_time = time.time()
            
            # Extract just the text content for batch embedding
            chunk_texts = [metadata[2] for metadata in chunk_metadata]
            
            # Generate all embeddings at once
            embeddings = self._get_embedding_batch(chunk_texts, max_batch_size=100)
            
            embed_elapsed = time.time() - embed_time
            print(f"   ✅ Generated {len(embeddings)} embeddings ({embed_elapsed:.1f}s, {len(embeddings)/embed_elapsed:.1f} emb/sec)")
            
            # Phase 3: Create and upload chunk documents
            print(f"\n   Phase 3/3: Uploading {len(chunk_metadata)} chunks to index...")
            upload_time = time.time()
            
            # Prepare all chunk documents first
            all_chunks = []
            for idx, (doc, chunk_idx, chunk_content, doc_id, total_chunks) in enumerate(chunk_metadata):
                chunk_doc_id = f"{doc_id}_chunk_{chunk_idx}"
                
                # Create chunk document (copy all metadata from original)
                chunk_doc = {
                    "id": chunk_doc_id,
                    "metadata_storage_name": doc.get('metadata_storage_name'),
                    "content": chunk_content,
                    "metadata_storage_path": doc.get('metadata_storage_path'),
                    "metadata_storage_file_extension": doc.get('metadata_storage_file_extension'),
                    "metadata_storage_size": doc.get('metadata_storage_size'),
                    "metadata_storage_last_modified": doc.get('metadata_storage_last_modified'),
                    "chunk_number": chunk_idx,
                    "total_chunks": total_chunks,
                    "chunk_id": doc_id,  # Reference to original document
                    "content_vector": embeddings[idx],  # Use pre-generated embedding
                }
                all_chunks.append(chunk_doc)
            
            # Split into batches of 100
            batch_size = 100
            batches = []
            for i in range(0, len(all_chunks), batch_size):
                batches.append(all_chunks[i:i + batch_size])
            
            print(f"      Uploading {len(batches)} batches in parallel (batch size: {batch_size})...")
            
            # Upload batches in parallel
            def upload_batch(batch_idx_and_docs):
                batch_idx, docs = batch_idx_and_docs
                try:
                    self.search_client.upload_documents(documents=docs)
                    return len(docs), 0  # succeeded, failed
                except Exception as e:
                    print(f"\n      ❌ Batch {batch_idx + 1} upload error: {e}")
                    return 0, len(docs)
            
            # Use ThreadPoolExecutor for parallel uploads (4 workers)
            with ThreadPoolExecutor(max_workers=4) as executor:
                batch_results = list(executor.map(upload_batch, enumerate(batches)))
                
                succeeded_count = sum(s for s, _ in batch_results)
                failed_count = sum(f for _, f in batch_results)
            
            upload_elapsed = time.time() - upload_time
            print(f"\n   ✅ Uploaded {succeeded_count} chunks ({upload_elapsed:.1f}s)")
            
            if failed_count > 0:
                print(f"   ❌ Failed: {failed_count} chunks")
            
            # Delete original un-chunked documents
            if docs_to_delete:
                print(f"\n   🗑️  Removing {len(docs_to_delete)} original documents (replaced with chunks)...", end=" ", flush=True)
                delete_docs = [{"id": doc_id} for doc_id in docs_to_delete]
                self.search_client.delete_documents(documents=delete_docs)
                print("✅")
            
            total_elapsed = time.time() - start_time
            print(f"\n✅ Chunking & Embedding Complete!")
            print(f"   📊 Stats:")
            print(f"      • Documents: {len(all_docs)}")
            print(f"      • Chunks created: {len(chunk_metadata)}")
            print(f"      • Average chunks/doc: {len(chunk_metadata) / len(all_docs):.1f}")
            print(f"      • Total time: {total_elapsed:.1f}s")
            print(f"      • Chunking: {chunk_elapsed:.1f}s")
            print(f"      • Embeddings: {embed_elapsed:.1f}s ({len(embeddings)/embed_elapsed:.1f} emb/sec)")
            print(f"      • Upload: {upload_elapsed:.1f}s")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Error enriching with embeddings: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def update_documents(self, skip_upload: bool = True) -> bool:
        """
        Update documents by running the indexer on blob storage.
        
        Args:
            skip_upload: If True (default), skip uploading local files and use documents already in blob.
                        If False, upload local files before running indexer.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            import time
            overall_start = time.time()
            
            # Step 1: Create index
            print("\n📋 Step 1: Creating/updating search index...")
            index_start = time.time()
            self._create_index()
            index_elapsed = time.time() - index_start
            print(f"   ⏱️  Index setup: {index_elapsed:.1f}s")
            
            if not skip_upload:
                # Collect and upload documents from local directory
                collection_dir = self._get_collection_dir()
                
                print(f"\n📁 Step 2: Scanning directory: {collection_dir}")
                scan_start = time.time()
                documents = self._collect_documents(collection_dir)
                
                if not documents:
                    print(f"⚠️  No supported documents found in {collection_dir}")
                    print(f"   Supported extensions: {', '.join(SUPPORTED_EXTENSIONS)}")
                    return False
                
                scan_elapsed = time.time() - scan_start
                print(f"✅ Found {len(documents)} document(s) ({scan_elapsed:.1f}s)")
                
                # Upload documents to blob storage
                print(f"\n☁️  Step 3: Uploading documents to blob storage '{self.container_name}'...")
                upload_start = time.time()
                container_client = self._ensure_container_exists()
                
                uploaded_count = 0
                failed_count = 0
                
                for idx, doc_path in enumerate(documents, 1):
                    try:
                        print(f"   [{idx}/{len(documents)}] Uploading {doc_path.name}...", end=" ", flush=True)
                        self._upload_to_blob(doc_path, doc_path.name)
                        uploaded_count += 1
                        print("✅")
                    except Exception as e:
                        print(f"❌ Error: {e}")
                        failed_count += 1
                
                upload_elapsed = time.time() - upload_start
                print(f"\n✅ Uploaded {uploaded_count} document(s) to blob storage ({upload_elapsed:.1f}s)")
                if failed_count > 0:
                    print(f"❌ Failed to upload {failed_count} document(s)")
                if uploaded_count > 0:
                    print(f"   ⏱️  Upload speed: {uploaded_count / upload_elapsed:.1f} docs/sec")
                
                step_offset = 3
            else:
                print("\n⏭️  Step 2: Skipping upload (using documents already in blob storage)")
                # Ensure container exists even if we skip upload
                self._ensure_container_exists()
                step_offset = 2
            
            # Create data source
            print(f"\n🔗 Step {step_offset + 1}: Creating/updating data source connection...")
            self._create_datasource()
            
            # Create indexer
            print(f"\n⚙️  Step {step_offset + 2}: Creating/updating indexer...")
            self._create_indexer()
            
            # Run indexer
            print(f"\n▶️  Step {step_offset + 3}: Running indexer...")
            self.indexer_client.run_indexer(self.indexer_name)
            print(f"✅ Indexer '{self.indexer_name}' started")
            
            # Step 6: Wait for indexer to complete
            print("\n⏳ Step 6: Waiting for indexer to complete...")
            indexer_start = time.time()
            max_wait = 300  # 5 minutes
            wait_interval = 5  # 5 seconds
            elapsed = 0
            
            while elapsed < max_wait:
                time.sleep(wait_interval)
                elapsed += wait_interval
                
                status = self.indexer_client.get_indexer_status(self.indexer_name)
                last_result = status.last_result
                
                if last_result:
                    if last_result.status == "success":
                        indexer_elapsed = time.time() - indexer_start
                        print(f"\n✅ Text indexing completed successfully! ({indexer_elapsed:.1f}s)")
                        print(f"   Items processed: {last_result.item_count}")
                        print(f"   Items failed: {last_result.failed_item_count}")
                        if last_result.item_count > 0:
                            print(f"   ⏱️  Indexing speed: {last_result.item_count / indexer_elapsed:.1f} docs/sec")
                        
                        # Step 7: Generate and add vector embeddings
                        success = self._enrich_with_embeddings()
                        if not success:
                            print("⚠️  Warning: Embedding generation failed, but documents are indexed (text search will work)")
                        
                        # Print overall summary
                        overall_elapsed = time.time() - overall_start
                        print(f"\n🎉 All Done! Total time: {overall_elapsed:.1f}s")
                        print(f"   📊 Time breakdown:")
                        print(f"      • Index setup: {index_elapsed:.1f}s")
                        if not skip_upload:
                            print(f"      • Scanning: {scan_elapsed:.1f}s")
                            print(f"      • Upload: {upload_elapsed:.1f}s")
                        print(f"      • Azure Indexer: {indexer_elapsed:.1f}s")
                        
                        return True
                    elif last_result.status == "transientFailure" or last_result.status == "failed":
                        print(f"\n❌ Indexing failed: {last_result.status}")
                        if last_result.error_message:
                            print(f"   Error: {last_result.error_message}")
                        return False
                
                # Still in progress
                print(f"   Still indexing... ({elapsed}s elapsed)", end="\r", flush=True)
            
            print(f"\n⚠️  Indexer is still running after {max_wait}s")
            print("   Documents will be available once indexing completes")
            print("   Run --update again later to add embeddings")
            return True
            
        except Exception as e:
            print(f"❌ Error updating documents: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_indexer_status(self) -> Optional[Dict[str, Any]]:
        """Get the current status of the indexer."""
        try:
            status = self.indexer_client.get_indexer_status(self.indexer_name)
            
            return {
                "status": status.status,
                "last_result": {
                    "status": status.last_result.status if status.last_result else None,
                    "item_count": status.last_result.item_count if status.last_result else 0,
                    "failed_item_count": status.last_result.failed_item_count if status.last_result else 0,
                    "error_message": status.last_result.error_message if status.last_result else None,
                    "start_time": status.last_result.start_time if status.last_result else None,
                    "end_time": status.last_result.end_time if status.last_result else None,
                } if status.last_result else None,
                "execution_history": [
                    {
                        "status": result.status,
                        "item_count": result.item_count,
                        "failed_item_count": result.failed_item_count,
                        "start_time": result.start_time,
                        "end_time": result.end_time,
                    }
                    for result in (status.execution_history or [])[:5]  # Last 5 runs
                ]
            }
        except Exception as e:
            print(f"❌ Error getting indexer status: {e}")
            return None
    
    def run_indexer(self) -> bool:
        """Manually trigger the indexer to run."""
        try:
            self.indexer_client.run_indexer(self.indexer_name)
            print(f"✅ Indexer '{self.indexer_name}' started")
            return True
        except Exception as e:
            print(f"❌ Error running indexer: {e}")
            return False
    
    def query_documents(self, query: str, **kwargs) -> Optional[str]:
        """Query documents using Azure AI Search with hybrid search (text + vector)."""
        try:
            top = kwargs.get('top', 10)
            use_semantic = kwargs.get('use_semantic', False)
            max_content_length = kwargs.get('max_content_length', 8000)
            
            # Generate query embedding for vector search
            query_vector = self._get_embedding(query)
            
            # Create vector query
            vector_query = VectorizedQuery(
                vector=query_vector,
                k_nearest_neighbors=top,
                fields="content_vector"
            )
            
            # Hybrid search: text + vector + optional semantic reranking
            if use_semantic:
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
                    vector_queries=[vector_query],
                    top=top,
                )
            
            # Format results with chunk information
            response_parts = []
            for idx, result in enumerate(results, 1):
                score = result.get('@search.score', 0)
                title = result.get('metadata_storage_name', 'Unknown')
                filepath = result.get('metadata_storage_path', 'Unknown')
                content = result.get('content', '')[:max_content_length]
                chunk_number = result.get('chunk_number')
                total_chunks = result.get('total_chunks')
                
                # Build chunk info string
                chunk_info = ""
                if chunk_number is not None and total_chunks is not None:
                    chunk_info = f" (Chunk {chunk_number + 1}/{total_chunks})"
                
                response_parts.append(
                    f"[Result {idx}] {title}{chunk_info} (Score: {score:.4f})\n"
                    f"File: {filepath}\n"
                    f"Content: {content}\n"
                    f"{'...(truncated)' if len(result.get('content', '')) > max_content_length else ''}\n"
                )
            
            if response_parts:
                return "\n".join(response_parts)
            else:
                return "No results found."
                
        except Exception as e:
            print(f"❌ Error querying documents: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def delete_documents(self) -> bool:
        """Delete the indexer, data source, and index."""
        try:
            # Delete in reverse order
            print(f"🗑️  Deleting indexer '{self.indexer_name}'...")
            try:
                self.indexer_client.delete_indexer(self.indexer_name)
                print("✅ Indexer deleted")
            except:
                print("⚠️  Indexer not found or already deleted")
            
            print(f"🗑️  Deleting data source '{self.datasource_name}'...")
            try:
                self.indexer_client.delete_data_source_connection(self.datasource_name)
                print("✅ Data source deleted")
            except:
                print("⚠️  Data source not found or already deleted")
            
            print(f"🗑️  Deleting index '{self.index_name}'...")
            try:
                self.index_client.delete_index(self.index_name)
                print("✅ Index deleted")
            except:
                print("⚠️  Index not found or already deleted")
            
            # Note: We don't delete the blob container as it may contain other data
            print(f"\nℹ️  Blob container '{self.container_name}' was not deleted")
            print("   Delete it manually if needed via Azure Portal or Azure CLI")
            
            return True
        except Exception as e:
            print(f"❌ Error deleting resources: {e}")
            return False
    
    def get_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the document store."""
        try:
            # Get index info
            index = self.index_client.get_index(self.index_name)
            
            # Count documents
            results = self.search_client.search(
                search_text="*",
                select=["id"],
                top=0,
                include_total_count=True
            )
            
            doc_count = results.get_count() if hasattr(results, 'get_count') else 0
            
            # Get indexer status
            indexer_status = self.get_indexer_status()
            
            return {
                "total_documents": doc_count,
                "index_name": self.index_name,
                "collection": self.collection,
                "container_name": self.container_name,
                "datasource_name": self.datasource_name,
                "indexer_name": self.indexer_name,
                "indexer_status": indexer_status,
                "approach": "indexer",
            }
        except Exception:
            return None
    
    def list_documents(self) -> List[str]:
        """List all documents in the collection."""
        try:
            results = self.search_client.search(
                search_text="*",
                select=["metadata_storage_path"],
                top=1000,
            )
            
            return [result.get('metadata_storage_path', '') for result in results]
        except Exception as e:
            print(f"❌ Error listing documents: {e}")
            return []
    
    @staticmethod
    def list_all_collections() -> List[str]:
        """List all collections with Azure AI Search indexers."""
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


"""
RAG Search Tool for document retrieval (async).

This tool uses the RAG document store to search through document collections
and retrieve relevant information.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from types import SimpleNamespace
from tool_base import Tool
from rag_client import RAGDocumentStore, DEFAULT_STORAGE_DIR
from config import DEFAULT_EMBEDDING_MODEL, DEFAULT_TOP_K

# Qdrant imports for direct SDK usage
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    QDRANT_CLIENT_AVAILABLE = True
except ImportError:
    QDRANT_CLIENT_AVAILABLE = False
    QdrantClient = None

# ChromaDB imports for direct SDK usage
try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    chromadb = None

# Set up logger
logger = logging.getLogger(__name__)


class RAGSearchTool(Tool):
    """Tool for searching documents using RAG with direct Qdrant SDK (faster) or LlamaIndex (ChromaDB)."""
    
    def __init__(
        self,
        collection: str,
        documents_dir: str = "data/documents",
        storage_dir: str = DEFAULT_STORAGE_DIR,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        top_k: int = DEFAULT_TOP_K
    ):
        """
        Initialize the RAG search tool.
        
        Args:
            collection: Collection name to search
            documents_dir: Base directory for documents
            storage_dir: Directory for vector store storage (Qdrant/Chroma)
            embedding_model: OpenAI embedding model to use
            top_k: Number of chunks to retrieve
        """
        self.collection = collection
        self.top_k = top_k
        
        # Initialize RAG document store
        self.rag_store = RAGDocumentStore(
            collection=collection,
            documents_dir=documents_dir,
            storage_dir=storage_dir,
            embedding_model=embedding_model
        )
    
    def name(self) -> str:
        # Include collection name in tool name for multi-collection support
        return f"search_documents_{self.collection}"
    
    def description(self) -> str:
        return f"""Search through documents in the '{self.collection}' collection to find relevant information. 
Use this tool when you need to answer questions based on documents in the '{self.collection}' collection. 
The tool will retrieve the most relevant document excerpts related to your search query."""
    
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to find relevant documents. Be specific and include key terms."
                }
            },
            "required": ["query"]
        }
    
    async def execute(self, query: str, **kwargs) -> str:
        """
        Execute the RAG search (async).
        
        Args:
            query: Search query string
            **kwargs: Additional parameters (ignored)
            
        Returns:
            str: Formatted search results with document excerpts and metadata
        """
        try:
            logger.info(f"🔍 RAG Tool: Starting document search for query: '{query[:100]}...'")
            logger.info(f"   Collection: '{self.collection}', top_k: {self.top_k}")
            
            # Use RAG store's retrieval in thread pool (LlamaIndex is sync)
            chunks = await asyncio.to_thread(self._retrieve_chunks, query)
            
            if not chunks:
                logger.info(f"⚠️  RAG Tool: No relevant documents found for query")
                return f"No relevant documents found for query: '{query}'"
            
            logger.info(f"✅ RAG Tool: Found {len(chunks)} relevant document chunks")
            
            # Log metadata details for debugging
            for i, chunk in enumerate(chunks[:3], 1):  # Show first 3 chunks
                if hasattr(chunk, 'metadata'):
                    page_label = chunk.metadata.get('page_label', 'NO PAGE INFO')
                    file_name = chunk.metadata.get('file_name', 'NO FILE NAME')
                    logger.info(f"   Chunk {i}: {file_name}, page_label={page_label}")
            
            # Format chunks for LLM consumption
            formatted_results = self._format_chunks(chunks)
            
            # Log sources found
            sources = set()
            pages_found = []
            urls_found = set()
            for chunk in chunks:
                if hasattr(chunk, 'metadata'):
                    if 'file_name' in chunk.metadata:
                        sources.add(chunk.metadata['file_name'])
                    if 'page_label' in chunk.metadata:
                        pages_found.append(chunk.metadata['page_label'])
                    if 'url' in chunk.metadata and chunk.metadata['url']:
                        urls_found.add(chunk.metadata['url'])
            
            if sources:
                logger.info(f"   Sources: {', '.join(sources)}")
            if pages_found:
                logger.info(f"   Pages found in chunks: {', '.join(set(pages_found))}")
            else:
                logger.warning(f"⚠️  RAG Tool: NO PAGE_LABEL metadata found in any chunks!")
            if urls_found:
                logger.info(f"   URLs found: {len(urls_found)} unique URL(s)")
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"❌ RAG Tool: Error searching documents: {str(e)}")
            return f"Error searching documents: {str(e)}"
    
    def _retrieve_chunks(self, query: str) -> List[Any]:
        """
        Retrieve relevant chunks using direct SDK calls (Qdrant) or LlamaIndex (ChromaDB).
        
        Args:
            query: Search query
            
        Returns:
            List of retrieved nodes/chunks (after reranking if enabled), or empty list if no index
        """
        # Check if index exists in state OR if vector store collection actually exists
        index_created_in_state = self.rag_store.state.get("index_created", False)
        vector_collection_name = f"{self.collection}_vectors"
        vector_store_exists = False
        
        # Check if vector store collection actually exists (even if state says it doesn't)
        if not index_created_in_state:
            try:
                if self.rag_store.vector_store_type == "qdrant":
                    if hasattr(self.rag_store, 'qdrant_client') and self.rag_store.qdrant_client:
                        collections = self.rag_store.qdrant_client.get_collections().collections
                        vector_store_exists = any(col.name == vector_collection_name for col in collections)
                elif self.rag_store.vector_store_type == "chroma":
                    if hasattr(self.rag_store, 'chroma_client') and self.rag_store.chroma_client:
                        try:
                            self.rag_store.chroma_client.get_collection(name=vector_collection_name)
                            vector_store_exists = True
                        except Exception:
                            vector_store_exists = False
            except Exception as e:
                logger.debug(f"Could not check vector store existence: {e}")
        
        # If neither state nor actual collection exists, return empty
        if not index_created_in_state and not vector_store_exists:
            # Check if there are documents in the folder
            docs = self.rag_store._scan_documents()
            if docs:
                logger.warning(f"⚠️  RAG Tool: Vector store is empty but {len(docs)} document(s) found in {self.rag_store.documents_dir}")
                logger.warning(f"⚠️  Please run 'python document_manager.py --backend rag --collection {self.collection} --update' to index documents")
            else:
                logger.info(f"ℹ️  RAG Tool: No documents found in collection '{self.collection}'")
            return []
        
        # If vector store exists but state says it doesn't, update state (fix for collections indexed before state update)
        if vector_store_exists and not index_created_in_state:
            logger.info(f"   ℹ️  Vector store exists but state not marked. Updating state...")
            try:
                self.rag_store.state["index_created"] = True
                self.rag_store._save_state()
                logger.info(f"   ✅ Updated state to mark index as created")
            except Exception as e:
                logger.warning(f"   ⚠️  Could not update state: {e}")
        
        # Use direct SDK for Qdrant (faster), LlamaIndex for ChromaDB
        if self.rag_store.vector_store_type == "qdrant":
            return self._retrieve_chunks_qdrant_direct(query, vector_collection_name)
        else:
            return self._retrieve_chunks_chroma_llamaindex(query, vector_collection_name)
    
    def _retrieve_chunks_qdrant_direct(self, query: str, collection_name: str) -> List[Any]:
        """Retrieve chunks directly from Qdrant using native SDK (faster)."""
        try:
            # Generate query embedding (sync call to OpenAI API)
            logger.info(f"   Generating query embedding...")
            response = self.rag_store.client.embeddings.create(
                model=self.rag_store.embedding_model,
                input=[query]
            )
            query_embedding = response.data[0].embedding
            
            logger.info(f"   Retrieving top {self.top_k} chunks from Qdrant (direct SDK)...")
            
            # Direct Qdrant search
            search_results = self.rag_store.qdrant_client.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                limit=self.top_k,
                with_payload=True,
                with_vectors=False  # Don't need vectors for retrieval
            )
            
            # Convert Qdrant results to node-like objects for compatibility
            nodes = []
            for result in search_results:
                payload = result.payload or {}
                
                # Handle different payload formats:
                # 1. Direct SDK format: payload["text"] (from company_collection_scraper.py and new rag_client.py)
                # 2. LlamaIndex format: payload["_node_content"] or payload["text"] (legacy)
                text = payload.get("text", "")
                
                # Fallback to LlamaIndex format for backward compatibility
                if not text:
                    text = payload.get("_node_content", "")
                
                # If still empty, try to get from node dict structure (legacy LlamaIndex)
                if not text and isinstance(payload.get("node"), dict):
                    text = payload["node"].get("text", "") or payload["node"].get("content", "")
                
                # Log warning if text is empty (shouldn't happen)
                if not text:
                    logger.warning(f"   ⚠️  Empty text in chunk (score: {result.score}, payload keys: {list(payload.keys())})")
                    continue
                
                # Extract metadata (everything except text-related keys)
                # LlamaIndex stores metadata separately, direct SDK stores it in payload
                metadata = {}
                text_keys = {"text", "_node_content", "node"}
                for k, v in payload.items():
                    if k not in text_keys:
                        metadata[k] = v
                
                # Also extract metadata from node dict if present (LlamaIndex format)
                if isinstance(payload.get("node"), dict):
                    node_dict = payload["node"]
                    for k, v in node_dict.items():
                        if k not in {"text", "content", "id", "embedding"}:
                            metadata[k] = v
                
                # Create node-like object compatible with _format_chunks
                node = SimpleNamespace(
                    text=text,
                    score=result.score,
                    metadata=metadata
                )
                nodes.append(node)
                
                # Log chunk preview and payload structure for debugging (first chunk only)
                if len(nodes) == 1:
                    preview = text[:200].replace('\n', ' ')
                    logger.debug(f"   First chunk preview: {preview}...")
                    logger.debug(f"   Payload keys: {list(payload.keys())}")
                    logger.debug(f"   Text length: {len(text)} chars")
                    # Check if table markers are present
                    if '|' in text:
                        table_rows = [line for line in text.split('\n') if '|' in line and line.strip().startswith('|')]
                        logger.debug(f"   Table rows detected: {len(table_rows)}")
            
            logger.info(f"   ✅ Retrieved {len(nodes)} chunks from Qdrant")
            
            # Merge adjacent chunks from the same page range to preserve table structures
            nodes = self._merge_adjacent_chunks(nodes)
            
            # Apply reranking if available (needs to convert back to LlamaIndex format temporarily)
            if self.rag_store.cohere_reranker:
                logger.info(f"   Applying Cohere reranking...")
                # Cohere reranker expects NodeWithScore objects, not plain TextNode
                from llama_index.core.schema import TextNode, NodeWithScore
                llamaindex_nodes = []
                for node in nodes:
                    # Create TextNode
                    text_node = TextNode(
                        text=node.text,
                        metadata=node.metadata
                    )
                    # Wrap in NodeWithScore (required by reranker)
                    node_with_score = NodeWithScore(
                        node=text_node,
                        score=node.score
                    )
                    llamaindex_nodes.append(node_with_score)
                
                # Rerank
                reranked_nodes = self.rag_store.cohere_reranker.postprocess_nodes(
                    llamaindex_nodes,
                    query_str=query
                )
                
                # Convert back to our simple format
                nodes = []
                for node_with_score in reranked_nodes:
                    # Extract the node and score from NodeWithScore
                    node = node_with_score.node if hasattr(node_with_score, 'node') else node_with_score
                    score = node_with_score.score if hasattr(node_with_score, 'score') else getattr(node, 'score', 0.0)
                    
                    simple_node = SimpleNamespace(
                        text=node.text,
                        score=score,
                        metadata=node.metadata or {}
                    )
                    nodes.append(simple_node)
                
                logger.info(f"   ✅ Reranked to top {len(nodes)} chunks")
            
            return nodes
            
        except Exception as e:
            logger.error(f"❌ Error retrieving chunks from Qdrant: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _retrieve_chunks_chroma_llamaindex(self, query: str, collection_name: str) -> List[Any]:
        """Retrieve chunks from ChromaDB using LlamaIndex (ChromaDB direct SDK is more complex)."""
        # Load index if not already loaded
        if not self.rag_store.index:
            try:
                logger.info(f"   Loading vector index for collection '{self.collection}'...")
                from llama_index.core import VectorStoreIndex
                
                vector_store = self.rag_store._get_vector_store_for_loading(collection_name)
                self.rag_store.index = VectorStoreIndex.from_vector_store(vector_store)
                logger.info(f"   ✅ Index loaded successfully")
            except Exception as e:
                logger.warning(f"⚠️  RAG Tool: Could not load vector index: {e}")
                # Check if there are documents in the folder
                docs = self.rag_store._scan_documents()
                if docs:
                    logger.warning(f"⚠️  {len(docs)} document(s) found in {self.rag_store.documents_dir} but not indexed")
                    logger.warning(f"⚠️  Please run 'python document_manager.py --backend rag --collection {self.collection} --update' to index documents")
                return []
        
        # Create retriever and get chunks
        logger.info(f"   Retrieving top {self.top_k} chunks from ChromaDB...")
        retriever = self.rag_store.index.as_retriever(similarity_top_k=self.top_k)
        nodes = retriever.retrieve(query)
        
        # Apply reranking if available
        if self.rag_store.cohere_reranker:
            logger.info(f"   Applying Cohere reranking...")
            nodes = self.rag_store.cohere_reranker.postprocess_nodes(
                nodes=nodes,
                query_str=query
            )
            logger.info(f"   ✅ Reranked to top {len(nodes)} chunks")
        
        return nodes
    
    def _merge_adjacent_chunks(self, nodes: List[Any]) -> List[Any]:
        """
        Merge adjacent chunks from the same page range to preserve table structures.
        
        Tables can be split across chunks, so we merge chunks that:
        1. Are from the same or adjacent pages
        2. Are from the same document
        3. Might contain parts of the same table (detected by markdown table syntax)
        """
        if not nodes or len(nodes) <= 1:
            return nodes
        
        merged = []
        i = 0
        
        while i < len(nodes):
            current = nodes[i]
            current_page = self._extract_page_number(current.metadata.get('page_label', ''))
            current_file = current.metadata.get('file_name', '')
            
            # Check if this chunk contains table markers (incomplete table)
            has_table_start = '|' in current.text and current.text.count('|') >= 3
            has_table_end = current.text.strip().endswith('|') or '\n|' in current.text
            
            # Try to merge with next chunks if:
            # 1. They're from the same file
            # 2. They're from the same or adjacent page
            # 3. Current chunk might be part of a table (has table markers)
            merged_text = current.text
            merged_score = current.score  # Use the best score (first chunk's score)
            j = i + 1
            
            while j < len(nodes):
                next_node = nodes[j]
                next_page = self._extract_page_number(next_node.metadata.get('page_label', ''))
                next_file = next_node.metadata.get('file_name', '')
                
                # Stop merging if:
                # - Different file
                # - Page gap > 1 (not adjacent)
                # - We've already merged enough (max 3 chunks to avoid too large merged chunks)
                if (next_file != current_file or 
                    abs(next_page - current_page) > 1 or 
                    j - i >= 2):  # Merge at most 3 chunks (i, i+1, i+2)
                    break
                
                # Merge: combine text with a separator
                merged_text += "\n\n" + next_node.text
                # Update score to average (or keep best)
                merged_score = max(merged_score, next_node.score)
                j += 1
            
            # Create merged node
            merged_node = SimpleNamespace(
                text=merged_text,
                score=merged_score,
                metadata=current.metadata.copy()
            )
            merged.append(merged_node)
            
            # Skip the chunks we just merged
            i = j
        
        if len(merged) < len(nodes):
            logger.info(f"   🔗 Merged {len(nodes)} chunks into {len(merged)} chunks to preserve table structures")
        
        return merged
    
    def _extract_page_number(self, page_label: str) -> int:
        """Extract page number from page_label (e.g., '153-154' -> 153, '156' -> 156)."""
        if not page_label:
            return 0
        try:
            # Handle ranges like "153-154" -> take first page
            if '-' in str(page_label):
                return int(str(page_label).split('-')[0])
            return int(str(page_label))
        except (ValueError, AttributeError):
            return 0
    
    def _format_chunks(self, nodes: List[Any]) -> str:
        """
        Format retrieved chunks for LLM consumption.
        
        Args:
            nodes: List of retrieved nodes
            
        Returns:
            str: Formatted string with all chunks and metadata
        """
        parts = [f"Found {len(nodes)} relevant document excerpts:\n"]
        
        # Track sources and their page labels for consolidated display
        source_pages = {}  # {source_name: set of page_labels}
        source_urls = {}   # {source_name: url or None}
        
        for i, node in enumerate(nodes, 1):
            # Get similarity score
            score = node.score if hasattr(node, 'score') else 'N/A'
            
            # Get source document
            source = "Unknown"
            if hasattr(node, 'metadata') and 'file_name' in node.metadata:
                source = node.metadata['file_name']
                if source not in source_pages:
                    source_pages[source] = set()
            
            # Get page/slide number if available
            page_info = ""
            if hasattr(node, 'metadata') and 'page_label' in node.metadata:
                page_label = node.metadata['page_label']
                if source != "Unknown":
                    source_pages[source].add(str(page_label))
                # Use "slide(s)" for PowerPoint files, "page(s)" for others
                if source.lower().endswith(('.ppt', '.pptx')):
                    label_type = "slides" if '-' in str(page_label) else "slide"
                else:
                    label_type = "pages" if '-' in str(page_label) else "page"
                page_info = f" ({label_type} {page_label})"
            
            # Get URL if available (for web sources)
            url_info = ""
            if hasattr(node, 'metadata') and 'url' in node.metadata and node.metadata['url']:
                url = node.metadata['url']
                url_info = f"\nURL: {url}"
                if source != "Unknown":
                    source_urls[source] = url
            
            # Format chunk
            parts.append(f"\n--- Document Excerpt {i} ---")
            parts.append(f"Source: {source}{page_info}{url_info}")
            parts.append(f"Relevance Score: {score}")
            parts.append(f"Content:\n{node.text}\n")
        
        # Format consolidated sources with page ranges
        sources_list = []
        for source in sorted(source_pages.keys()):
            page_labels = source_pages[source]
            if page_labels:
                # Consolidate page ranges
                page_ranges = self._consolidate_page_ranges(page_labels, source)
                if page_ranges:
                    source_str = f"{source} ({page_ranges})"
                else:
                    source_str = source
            else:
                source_str = source
            
            # Add URL if available
            if source in source_urls:
                source_str += f" - {source_urls[source]}"
            
            sources_list.append(source_str)
        
        if sources_list:
            parts.append(f"\n📚 Sources: {', '.join(sources_list)}")
        else:
            parts.append(f"\n📚 Sources: Unknown")
        
        return "\n".join(parts)
    
    def _consolidate_page_ranges(self, page_labels: set, source: str) -> str:
        """
        Consolidate page labels into ranges (e.g., "pages 151-154" instead of "page 151, page 152, page 153, page 154").
        
        Args:
            page_labels: Set of page label strings (e.g., {"151", "152", "153-154", "156"})
            source: Source document name (to determine if it's slides or pages)
            
        Returns:
            Consolidated string like "pages 151-154, 156" or "slides 1-3, 5"
        """
        # Determine if it's slides or pages
        is_slides = source.lower().endswith(('.ppt', '.pptx'))
        label_type = "slides" if is_slides else "pages"
        
        # Parse all page numbers from labels (handle ranges like "151-154")
        all_pages = set()
        for label in page_labels:
            if '-' in label:
                # Range like "151-154"
                try:
                    start, end = map(int, label.split('-'))
                    all_pages.update(range(start, end + 1))
                except ValueError:
                    # Invalid range, just add the label as-is
                    all_pages.add(label)
            else:
                # Single page
                try:
                    all_pages.add(int(label))
                except ValueError:
                    # Not a number, keep as-is
                    all_pages.add(label)
        
        # Convert to sorted list of integers (filter out non-integers for now)
        numeric_pages = sorted([p for p in all_pages if isinstance(p, int)])
        non_numeric = [p for p in all_pages if not isinstance(p, int)]
        
        if not numeric_pages:
            # No numeric pages, just return original labels joined
            return f"{label_type} {', '.join(sorted(page_labels))}"
        
        # Consolidate consecutive pages into ranges
        ranges = []
        if numeric_pages:
            start = numeric_pages[0]
            end = numeric_pages[0]
            
            for page in numeric_pages[1:]:
                if page == end + 1:
                    # Consecutive, extend range
                    end = page
                else:
                    # Gap found, save current range
                    if start == end:
                        ranges.append(str(start))
                    else:
                        ranges.append(f"{start}-{end}")
                    start = page
                    end = page
            
            # Add final range
            if start == end:
                ranges.append(str(start))
            else:
                ranges.append(f"{start}-{end}")
        
        # Add non-numeric labels
        ranges.extend(non_numeric)
        
        # Format result
        if len(ranges) == 1:
            return f"{label_type[:-1]} {ranges[0]}"  # "page 151" or "slide 1"
        else:
            return f"{label_type} {', '.join(ranges)}"  # "pages 151-154, 156"


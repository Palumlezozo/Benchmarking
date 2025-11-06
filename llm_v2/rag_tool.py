"""
RAG Search Tool for document retrieval (async).

This tool uses the RAG document store to search through document collections
and retrieve relevant information.
"""

import asyncio
import logging
from typing import Dict, Any, List
from tool_base import Tool
from rag_client import RAGDocumentStore, DEFAULT_STORAGE_DIR
from config import DEFAULT_EMBEDDING_MODEL, DEFAULT_TOP_K

# Set up logger
logger = logging.getLogger(__name__)


class RAGSearchTool(Tool):
    """Tool for searching documents using RAG (LlamaIndex + Qdrant or Chroma)."""
    
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
            for chunk in chunks:
                if hasattr(chunk, 'metadata'):
                    if 'file_name' in chunk.metadata:
                        sources.add(chunk.metadata['file_name'])
                    if 'page_label' in chunk.metadata:
                        pages_found.append(chunk.metadata['page_label'])
            
            if sources:
                logger.info(f"   Sources: {', '.join(sources)}")
            if pages_found:
                logger.info(f"   Pages found in chunks: {', '.join(set(pages_found))}")
            else:
                logger.warning(f"⚠️  RAG Tool: NO PAGE_LABEL metadata found in any chunks!")
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"❌ RAG Tool: Error searching documents: {str(e)}")
            return f"Error searching documents: {str(e)}"
    
    def _retrieve_chunks(self, query: str) -> List[Any]:
        """
        Retrieve relevant chunks using the RAG store with optional reranking.
        
        Args:
            query: Search query
            
        Returns:
            List of retrieved nodes/chunks (after reranking if enabled), or empty list if no index
        """
        # Check if index exists
        if not self.rag_store.state.get("index_created", False):
            # Check if there are documents in the folder
            docs = self.rag_store._scan_documents()
            if docs:
                logger.warning(f"⚠️  RAG Tool: Vector store is empty but {len(docs)} document(s) found in {self.rag_store.documents_dir}")
                logger.warning(f"⚠️  Please run 'python document_manager.py --backend rag --collection {self.collection} --update' to index documents")
            else:
                logger.info(f"ℹ️  RAG Tool: No documents found in collection '{self.collection}'")
            return []
        
        # Load index if not already loaded
        if not self.rag_store.index:
            try:
                logger.info(f"   Loading vector index for collection '{self.collection}'...")
                from llama_index.core import VectorStoreIndex
                
                collection_name = f"{self.collection}_vectors"
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
        logger.info(f"   Retrieving top {self.top_k} chunks from vector store...")
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
    
    def _format_chunks(self, nodes: List[Any]) -> str:
        """
        Format retrieved chunks for LLM consumption.
        
        Args:
            nodes: List of retrieved nodes
            
        Returns:
            str: Formatted string with all chunks and metadata
        """
        parts = [f"Found {len(nodes)} relevant document excerpts:\n"]
        
        sources = set()
        for i, node in enumerate(nodes, 1):
            # Get similarity score
            score = node.score if hasattr(node, 'score') else 'N/A'
            
            # Get source document
            source = "Unknown"
            if hasattr(node, 'metadata') and 'file_name' in node.metadata:
                source = node.metadata['file_name']
                sources.add(source)
            
            # Get page/slide number if available
            page_info = ""
            if hasattr(node, 'metadata') and 'page_label' in node.metadata:
                page_label = node.metadata['page_label']
                # Use "slide(s)" for PowerPoint files, "page(s)" for others
                if source.lower().endswith(('.ppt', '.pptx')):
                    label_type = "slides" if '-' in str(page_label) else "slide"
                else:
                    label_type = "pages" if '-' in str(page_label) else "page"
                page_info = f" ({label_type} {page_label})"
            
            # Format chunk
            parts.append(f"\n--- Document Excerpt {i} ---")
            parts.append(f"Source: {source}{page_info}")
            parts.append(f"Relevance Score: {score}")
            parts.append(f"Content:\n{node.text}\n")
        
        parts.append(f"\n📚 Sources: {', '.join(sources) if sources else 'Unknown'}")
        
        return "\n".join(parts)


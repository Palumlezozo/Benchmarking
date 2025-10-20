"""
Factory for creating document store instances.

This module provides a factory function to create the appropriate document store
implementation based on the specified backend type.
"""

from typing import Literal
from document_store_base import DocumentStore

# Backend types
OPENAI_BACKEND = "openai"
RAG_BACKEND = "rag"
SUPPORTED_BACKENDS = [OPENAI_BACKEND, RAG_BACKEND]


def create_document_store(
    backend: Literal["openai", "rag"],
    collection: str,
    **kwargs
) -> DocumentStore:
    """
    Factory function to create a document store instance.
    
    Args:
        backend: The backend type to use ('openai' or 'rag')
        collection: The collection name
        **kwargs: Additional parameters passed to the store constructor
        
    Returns:
        DocumentStore: An instance of the appropriate document store
        
    Raises:
        ValueError: If backend type is not supported
        
    Examples:
        # Create OpenAI document store
        store = create_document_store("openai", "my_collection", model="gpt-5-mini")
        
        # Create RAG document store
        store = create_document_store("rag", "my_collection", embedding_model="text-embedding-3-small")
    """
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unsupported backend: {backend}. "
            f"Must be one of: {', '.join(SUPPORTED_BACKENDS)}"
        )
    
    if backend == OPENAI_BACKEND:
        from openai_document_store import OpenAIDocumentStore
        # Filter out RAG-specific parameters for OpenAI backend (it doesn't use them)
        openai_kwargs = {k: v for k, v in kwargs.items() if k not in ['embedding_model', 'store_md', 'use_llama_parse']}
        return OpenAIDocumentStore(collection=collection, **openai_kwargs)
    
    elif backend == RAG_BACKEND:
        from rag_client import RAGDocumentStore
        return RAGDocumentStore(collection=collection, **kwargs)
    
    else:
        raise ValueError(f"Backend {backend} not implemented")


def list_available_backends() -> list:
    """List all available backend types."""
    return SUPPORTED_BACKENDS.copy()


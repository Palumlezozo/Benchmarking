"""
Abstract base class for document store implementations.

This module defines the interface that all document store implementations must follow.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List, Optional


class DocumentStore(ABC):
    """Abstract base class for document storage and retrieval systems."""
    
    def __init__(self, collection: str):
        """
        Initialize the document store.
        
        Args:
            collection: Name of the collection to manage
        """
        self.collection = collection
    
    @abstractmethod
    def update_documents(self) -> bool:
        """
        Update documents in the store (upload new/modified, remove deleted).
        
        Returns:
            bool: True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def query_documents(self, query: str, **kwargs) -> Optional[str]:
        """
        Query the document store.
        
        Args:
            query: The query string
            **kwargs: Additional query parameters
            
        Returns:
            Optional[str]: The response text, or None if error
        """
        pass
    
    @abstractmethod
    def delete_documents(self) -> bool:
        """
        Delete all documents in the collection.
        
        Returns:
            bool: True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the document store.
        
        Returns:
            Optional[Dict[str, Any]]: Store information (doc count, last updated, etc.)
        """
        pass
    
    @abstractmethod
    def list_documents(self) -> List[str]:
        """
        List all documents in the collection.
        
        Returns:
            List[str]: List of document paths
        """
        pass


"""
OpenAI Document Store Implementation.

This module provides a document store implementation using OpenAI's file search
and vector store capabilities.
"""

import json
import os
from datetime import datetime as dt
from pathlib import Path
from typing import Dict, Any, List, Optional

from openai import OpenAI
from dotenv import load_dotenv

from document_store_base import DocumentStore
from config import DEFAULT_MODEL, DEFAULT_REASONING_EFFORT, DEFAULT_TEXT_VERBOSITY

# Load environment variables
load_dotenv()

# Constants
DOCUMENTS_DIR = "data/documents"
RAG_DIR = "data/rag"
STATE_FILE = "data/rag/openai_state.json"  # Separate state file for OpenAI backend
SUPPORTED_EXTENSIONS = {'.txt', '.md', '.docx', '.pdf', '.xlsx', '.ppt'}


class OpenAIDocumentStore(DocumentStore):
    """Document store implementation using OpenAI file search tool."""
    
    def __init__(
        self,
        collection: str,
        model: str = DEFAULT_MODEL,
        reasoning_effort: str = DEFAULT_REASONING_EFFORT,
        text_verbosity: str = DEFAULT_TEXT_VERBOSITY
    ):
        """
        Initialize the OpenAI Document Store.
        
        Args:
            collection: Collection name
            model: OpenAI model to use for queries
            reasoning_effort: Reasoning effort level
            text_verbosity: Text verbosity level
        """
        super().__init__(collection)
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.text_verbosity = text_verbosity
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.state_file = Path(STATE_FILE)
        
        # Ensure RAG directory exists
        self.rag_dir = Path(RAG_DIR)
        self.rag_dir.mkdir(parents=True, exist_ok=True)
        
        # Load global state (manages all collections)
        self.state = self._load_state()
    
    def _load_state(self) -> Dict[str, Any]:
        """Load the persistent state from file."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"⚠️  Warning: Could not load state file: {e}")
                return {"collections": {}, "last_updated": None}
        return {"collections": {}, "last_updated": None}
    
    def _save_state(self) -> None:
        """Save the current state to file."""
        self.state["last_updated"] = dt.now().isoformat()
        try:
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(self.state, f, indent=2, ensure_ascii=False)
        except IOError as e:
            print(f"❌ Error saving state: {e}")
    
    def _get_collection_dir(self) -> Path:
        """Get the documents directory for the collection."""
        return Path(DOCUMENTS_DIR) / self.collection
    
    def _upload_file_to_openai(self, file_path: Path) -> Optional[str]:
        """Upload a file to OpenAI and return the file ID."""
        try:
            with open(file_path, 'rb') as f:
                file_response = self.client.files.create(
                    file=f,
                    purpose='assistants'
                )
            print(f"✅ Uploaded {file_path.name} (ID: {file_response.id})")
            return file_response.id
        except Exception as e:
            print(f"❌ Error uploading {file_path.name}: {e}")
            return None
    
    def _create_vector_store(self) -> Optional[str]:
        """Create a vector store for the collection."""
        try:
            vector_store_response = self.client.vector_stores.create(
                name=f"{self.collection}_documents"
            )
            print(f"✅ Created vector store for {self.collection} (ID: {vector_store_response.id})")
            return vector_store_response.id
        except Exception as e:
            print(f"❌ Error creating vector store for {self.collection}: {e}")
            return None
    
    def _add_file_to_vector_store(self, vector_store_id: str, file_id: str) -> bool:
        """Add a file to a vector store."""
        try:
            self.client.vector_stores.files.create(
                vector_store_id=vector_store_id,
                file_id=file_id
            )
            return True
        except Exception as e:
            print(f"❌ Error adding file to vector store: {e}")
            return False
    
    def _scan_documents(self) -> Dict[str, Dict[str, Any]]:
        """Scan the documents directory for files and their metadata."""
        docs_dir = self._get_collection_dir()
        documents = {}
        
        if not docs_dir.exists():
            print(f"📁 Creating collection directory: {docs_dir}")
            docs_dir.mkdir(parents=True, exist_ok=True)
            return documents
        
        for file_path in docs_dir.rglob("*"):
            # Filter out macOS resource fork files (._*) and only process supported file types
            if (file_path.is_file() and 
                file_path.suffix.lower() in SUPPORTED_EXTENSIONS and
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
    
    def _get_new_documents(self) -> List[Path]:
        """Get list of new or modified documents for the collection."""
        current_docs = self._scan_documents()
        collection_state = self.state["collections"].get(self.collection, {"documents": {}})
        stored_docs = collection_state.get("documents", {})
        
        new_docs = []
        for file_path, metadata in current_docs.items():
            stored_metadata = stored_docs.get(file_path)
            if not stored_metadata or stored_metadata.get("modified") != metadata["modified"]:
                new_docs.append(Path(file_path))
        
        return new_docs
    
    def _get_deleted_documents(self) -> List[Path]:
        """Get list of deleted documents for the collection."""
        current_docs = self._scan_documents()
        collection_state = self.state["collections"].get(self.collection, {"documents": {}})
        stored_docs = collection_state.get("documents", {})
        
        deleted_docs = []
        for file_path in stored_docs.keys():
            if file_path not in current_docs:
                deleted_docs.append(Path(file_path))
        
        return deleted_docs
    
    def _remove_document_from_openai(self, doc_path: Path) -> bool:
        """Remove a document from OpenAI."""
        try:
            collection_state = self.state["collections"].get(self.collection, {})
            file_ids = collection_state.get("file_ids", {})
            file_id = file_ids.get(str(doc_path))
            
            if file_id:
                # Delete file from OpenAI
                self.client.files.delete(file_id)
                print(f"🗑️  Deleted {doc_path.name} from OpenAI")
                return True
            else:
                print(f"ℹ️  {doc_path.name} not found in OpenAI")
                return True
                
        except Exception as e:
            print(f"❌ Error removing document from OpenAI: {e}")
            return False
    
    # DocumentStore interface implementation
    
    def update_documents(self) -> bool:
        """Update documents for the collection."""
        # Initialize collection state if needed
        if self.collection not in self.state["collections"]:
            self.state["collections"][self.collection] = {
                "documents": {},
                "file_ids": {},
                "vector_store_id": None,
                "last_updated": None
            }
        
        collection_state = self.state["collections"][self.collection]
        
        # Ensure structure exists
        if "documents" not in collection_state:
            collection_state["documents"] = {}
        if "file_ids" not in collection_state:
            collection_state["file_ids"] = {}
        if "vector_store_id" not in collection_state:
            collection_state["vector_store_id"] = None
        
        # Get new and deleted documents
        new_docs = self._get_new_documents()
        deleted_docs = self._get_deleted_documents()
        
        if not new_docs and not deleted_docs:
            print(f"✅ No changes for collection '{self.collection}'")
            return True
        
        total_changes = len(new_docs) + len(deleted_docs)
        
        # Process deleted documents first
        for doc_path in deleted_docs:
            print(f"🗑️  Removing {doc_path.name}...")
            if self._remove_document_from_openai(doc_path):
                # Remove from state
                if str(doc_path) in collection_state["documents"]:
                    del collection_state["documents"][str(doc_path)]
                if str(doc_path) in collection_state["file_ids"]:
                    del collection_state["file_ids"][str(doc_path)]
                print(f"✅ Removed {doc_path.name}")
            else:
                print(f"❌ Failed to remove {doc_path.name}")
        
        # Process new documents
        for doc_path in new_docs:
            print(f"📤 Uploading {doc_path.name}...")
            file_id = self._upload_file_to_openai(doc_path)
            if file_id:
                # Get or create vector store
                vector_store_id = collection_state["vector_store_id"]
                if not vector_store_id:
                    vector_store_id = self._create_vector_store()
                    if vector_store_id:
                        collection_state["vector_store_id"] = vector_store_id
                    else:
                        print(f"❌ Failed to create vector store for collection '{self.collection}'")
                        continue
                
                # Add file to vector store
                if self._add_file_to_vector_store(vector_store_id, file_id):
                    # Update document metadata in state
                    collection_state["documents"][str(doc_path)] = {
                        "size": doc_path.stat().st_size,
                        "modified": doc_path.stat().st_mtime,
                        "extension": doc_path.suffix.lower(),
                        "file_id": file_id
                    }
                    collection_state["file_ids"][str(doc_path)] = file_id
                    print(f"✅ Added {doc_path.name} to vector store")
                else:
                    print(f"❌ Failed to add {doc_path.name} to vector store")
            else:
                print(f"❌ Failed to process {doc_path.name}")
        
        # Update last updated timestamp
        collection_state["last_updated"] = dt.now().isoformat()
        
        # Save state
        self._save_state()
        print(f"✅ Updated collection '{self.collection}' ({total_changes} changes)")
        return True
    
    def query_documents(self, query: str, **kwargs) -> Optional[str]:
        """Query documents using OpenAI vector stores."""
        try:
            # Get collection state
            collection_state = self.state["collections"].get(self.collection)
            if not collection_state:
                # Check if there are documents in the folder
                docs = self._scan_documents()
                if docs:
                    print(f"⚠️  Vector store is empty but {len(docs)} document(s) found in {self._get_collection_dir()}")
                    print(f"   Please run document_manager.py --backend openai --collection {self.collection} --update to index documents")
                    return f"No documents indexed yet. Found {len(docs)} document(s) that need to be indexed."
                else:
                    return f"No documents found in collection '{self.collection}'."
            
            # Get vector store ID
            vector_store_id = collection_state.get("vector_store_id")
            if not vector_store_id:
                # Check if there are documents in the folder
                docs = self._scan_documents()
                if docs:
                    print(f"⚠️  Vector store is empty but {len(docs)} document(s) found in {self._get_collection_dir()}")
                    print(f"   Please run document_manager.py --backend openai --collection {self.collection} --update to index documents")
                    return f"No documents indexed yet. Found {len(docs)} document(s) that need to be indexed."
                else:
                    return f"No documents found in collection '{self.collection}'."
            
            # Use responses API with file search tool
            response = self.client.responses.parse(
                model=self.model,
                input=[
                    {
                        "role": "system", 
                        "content": f"""
You are a document analysis assistant for the '{self.collection}' collection.

You have access to relevant documents. When answering questions:
1. Search through the provided documents to find relevant information
2. Base your answers on the provided document content
3. Be specific and cite relevant information from the documents
4. If information is not available in the documents, clearly state this
5. Provide accurate, well-sourced answers
6. Do not include citation markers, reference numbers, or any [cite] tags in your output
"""
                    },
                    {
                        "role": "user",
                        "content": query
                    }
                ],
                tools=[{
                    "type": "file_search",
                    "vector_store_ids": [vector_store_id]
                }],
                tool_choice="auto",
                reasoning={"effort": self.reasoning_effort},
                text={"verbosity": self.text_verbosity}
            )
            
            # Extract the text content from the response
            if hasattr(response, 'output') and response.output:
                # Find the text content in the response
                for item in response.output:
                    if hasattr(item, 'content') and item.content:
                        for content_item in item.content:
                            if hasattr(content_item, 'text'):
                                return content_item.text
            return "No response content found."
                
        except Exception as e:
            print(f"❌ Error querying documents: {e}")
            return None
    
    def delete_documents(self) -> bool:
        """Delete all documents for the collection."""
        try:
            collection_state = self.state["collections"].get(self.collection)
            if not collection_state:
                print(f"ℹ️  No documents found for collection '{self.collection}'")
                return True
            
            # Delete files from OpenAI
            file_ids = collection_state.get("file_ids", {})
            for file_path, file_id in file_ids.items():
                try:
                    self.client.files.delete(file_id)
                    print(f"🗑️  Deleted {Path(file_path).name} from OpenAI")
                except Exception as e:
                    print(f"⚠️  Warning: Could not delete {Path(file_path).name}: {e}")
            
            # Remove from state
            del self.state["collections"][self.collection]
            
            self._save_state()
            print(f"✅ Deleted documents for collection '{self.collection}'")
            return True
            
        except Exception as e:
            print(f"❌ Error deleting documents for collection '{self.collection}': {e}")
            return False
    
    def get_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the collection's document store."""
        collection_state = self.state["collections"].get(self.collection)
        if not collection_state:
            return None
        
        documents = collection_state.get("documents", {})
        
        return {
            "total_documents": len(documents),
            "last_updated": collection_state.get("last_updated"),
            "collection": self.collection,
            "vector_store_id": collection_state.get("vector_store_id")
        }
    
    def list_documents(self) -> List[str]:
        """List all documents in the collection."""
        collection_state = self.state["collections"].get(self.collection, {})
        documents = collection_state.get("documents", {})
        return list(documents.keys())
    
    @classmethod
    def list_all_collections(cls) -> List[str]:
        """List all collections across the state file (class method)."""
        state_path = Path(STATE_FILE)
        if state_path.exists():
            try:
                with open(state_path, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                return list(state.get("collections", {}).keys())
            except Exception:
                return []
        return []


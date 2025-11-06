# LLM-v2: Advanced Document Q&A System

A powerful, flexible document Q&A system that combines multiple document search backends with web search capabilities to provide comprehensive answers to questions about your document collections.

## 🌟 Key Features

### Multi-Backend Support
- **RAG Backend (Default)**: Local vector search using LlamaIndex + Qdrant (default) or ChromaDB with Azure OpenAI or standard OpenAI
- **OpenAI Backend**: Cloud-based using OpenAI's native file_search tool (use `--openai-tools` flag)
- **Hybrid Approach**: Combine document search with real-time web search
- **Flexible API**: Auto-detects Azure OpenAI or falls back to standard OpenAI
- **Vector Store Options**: Qdrant (default, Docker-based) or ChromaDB (local files)

### Advanced Document Processing
- **LlamaParse Integration**: High-quality document parsing with OCR support
- **Multiple File Formats**: PDF, DOCX, PPTX, XLSX, TXT, MD, HTML
- **Page Number Tracking**: Accurate page citations in responses
- **Cohere Reranking**: Improved relevance with optional reranking

### Multi-Collection Support
- **Query Multiple Collections**: Search across 2+ document collections simultaneously
- **Flexible Organization**: Organize documents by topic, year, department, etc.
- **Intelligent Routing**: System automatically searches relevant collections

### Smart Features
- **Structured Responses**: Content, sources, and insights separated
- **Markdown Table Export**: Auto-detect tables and export to Excel
- **Conversation History**: Context-aware follow-up questions
- **Web Search Integration**: Tavily for current information and news
- **Graceful Error Handling**: Clear warnings when collections need indexing

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Azure OpenAI Integration](#azure-openai-integration)
- [Architecture](#architecture)
- [Scripts Overview](#scripts-overview)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Advanced Features](#advanced-features)
- [API Keys Required](#api-keys-required)
- [Troubleshooting](#troubleshooting)

## 🚀 Installation

### Prerequisites
- Python 3.12 or higher
- uv (recommended) or pip
- Docker and Docker Compose (for Qdrant vector store)

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd llm-v2
```

2. **Start Qdrant (Vector Store)**

The default vector store is Qdrant, which runs in Docker:

```bash
# Start Qdrant container
docker-compose up -d

# Verify it's running
docker ps | grep qdrant

# Check Qdrant health
curl http://localhost:6333/health
```

Qdrant will store data in `data/rag/qdrant/` directory.

**Note:** If you prefer to use ChromaDB instead of Qdrant, you can use the `--chroma` flag with `document_manager.py`.

3. **Install dependencies**

Using uv (recommended):
```bash
# Install uv if not already installed
# On macOS (recommended):
brew install uv

# On Linux/Windows or if you prefer the official installer:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Activate the virtual environment
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate  # On Windows
```

Using pip:
```bash
pip install -e .
```

4. **Configure environment variables**

Create a `.env` file in the project root:

```bash
# Copy the example file
cp .env.example .env

# Then edit .env with your actual API keys
# See .env.example for all available options and detailed comments
```

**Quick setup (minimum required):**

```bash
# Azure OpenAI (used for RAG backend - default, recommended)
AZURE_OPENAI_API_KEY=your_azure_api_key_here
AZURE_OPENAI_BASE_URL=https://your-resource.openai.azure.com/openai/v1/

# OR Standard OpenAI (used for --openai-tools backend or as fallback)
OPENAI_API_KEY=your_openai_api_key_here

# Optional but recommended
TAVILY_API_KEY=your_tavily_api_key_here        # For web search
LLAMA_CLOUD_API_KEY=your_llama_cloud_api_key  # For LlamaParse
LLAMA_CLOUD_BASE_URL=https://api.cloud.llamaindex.ai  # Optional: European endpoint
COHERE_API_KEY=your_cohere_api_key            # For reranking
```

**Note:** See `.env.example` for a complete template with all available options and detailed documentation.

5. **Create document directories**
```bash
mkdir -p data/documents/collection
mkdir -p data/rag/vector_stores
```

## ⚡ Quick Start

> **Note:** Scripts can be run in two ways:
> - **Recommended:** `uv run python -m llm_v2.chatbot` (as module)
> - **Alternative:** `uv run python llm_v2/chatbot.py` (direct script)
> 
> Both methods work identically for `chatbot.py`, `document_manager.py`, and `azure_document_manager.py`.  
> The module method (`-m`) is recommended but not required.
> 
> **With activated virtual environment:** If you've activated the venv with `source .venv/bin/activate`, you can run scripts directly:
> - `python -m llm_v2.chatbot` or `python llm_v2/chatbot.py`

### 1. Add Documents

Place your documents in `data/documents/<collection_name>/`:

```bash
# Example structure
data/documents/
├── financials/
│   ├── 2023_report.pdf
│   └── 2024_report.pdf
├── operations/
│   ├── procedures.docx
│   └── guidelines.pdf
└── collection/  # default collection
    └── document.pdf
```

### 2. Index Documents

**Using RAG Backend (default, uses Qdrant and Azure OpenAI if configured):**
```bash
# Direct script (both --collection and --collections work):
python llm_v2/document_manager.py --collection financials --update
python llm_v2/document_manager.py --collections financials --update

# Or as module:
uv run python -m llm_v2.document_manager --collection financials --update

# Or with activated venv:
python -m llm_v2.document_manager --collection financials --update

# Use ChromaDB instead of Qdrant:
python llm_v2/document_manager.py --collection financials --update --chroma
```

**Using OpenAI Backend (with native file_search):**
```bash
python llm_v2/document_manager.py --openai-tools --collection financials --update
```

### 3. Start Chatting

**Interactive mode (RAG backend with Azure OpenAI):**
```bash
# Recommended (as module):
uv run python -m llm_v2.chatbot --collection financials

# Alternative (direct script):
uv run python llm_v2/chatbot.py --collection financials

# Or use --collections alias:
uv run python -m llm_v2.chatbot --collections financials
```

**Query multiple collections:**
```bash
uv run python -m llm_v2.chatbot --collection "financials,operations,collection"
# Or:
uv run python -m llm_v2.chatbot --collections "financials,operations,collection"
```

**Single query:**
```bash
uv run python -m llm_v2.chatbot --collection financials --query "What were the key metrics in 2023?"
```

**Using OpenAI native tools (file_search + web_search):**
```bash
uv run python -m llm_v2.chatbot --openai-tools --collection financials
```

## ☁️ Azure OpenAI Integration

The system is designed to work seamlessly with **Azure OpenAI** for cost-effective and enterprise-ready deployments.

### How It Works

**Without `--openai-tools` flag (RAG Backend - Default):**
- ✅ Uses **Azure OpenAI** for all operations (if configured)
- Handles: embeddings, tool orchestration, final responses
- Falls back to standard OpenAI if Azure not configured
- Status shows: `(Azure OpenAI)` when active

**With `--openai-tools` flag (OpenAI Backend):**
- ✅ Uses **Standard OpenAI** for native file_search and web_search tools
- Required for OpenAI-specific features

### Configuration

Add these to your `.env` file:

```bash
# Azure OpenAI (for RAG backend - default)
AZURE_OPENAI_API_KEY=your_azure_api_key
AZURE_OPENAI_BASE_URL=https://your-resource.openai.azure.com/openai/v1/

# Standard OpenAI (for --openai-tools)
OPENAI_API_KEY=your_openai_api_key
```

### Benefits

1. **Cost Control**: Use Azure's pricing models for bulk operations
2. **Enterprise Ready**: Leverage Azure's compliance and security features
3. **Flexibility**: Switch between Azure and OpenAI with environment variables
4. **Transparent**: System clearly indicates which provider is being used

### Testing Azure Connection

```bash
# Test your Azure OpenAI setup
python tests/azure-test.py
```

This verifies:
- ✅ Connection to Azure endpoint
- ✅ API key validity
- ✅ Model deployment
- ✅ Responses API functionality


## 🏗️ Architecture

### System Components

```
┌────────────────────────────────────────────────────────────────────┐
│                             Chatbot                                 │
│  ┌──────────────┐                    ┌──────────────────┐        │
│  │ OpenAI Tools │                    │    My-Tools      │        │
│  │ - file_search│                    │  - RAG Search    │        │
│  │ - web_search │                    │  - Tavily Web    │        │
│  └──────────────┘                    │  - Tavily News   │        │
│                                       └──────────────────┘        │
└────────────────────────────────────────────────────────────────────┘
           │                                    │
           ▼                                    ▼
┌─────────────────┐                ┌────────────────────┐
│OpenAI Vector    │                │ RAG Document Store │
│Store            │                │ - LlamaIndex       │
│- Cloud-based    │                │ - Qdrant (default) │
│- Managed by     │                │   or ChromaDB      │
│  OpenAI         │                │ - Cohere Rerank    │
│                 │                │ - LlamaParse       │
│                 │                │ - Azure OpenAI     │
└─────────────────┘                │   or OpenAI        │
                                   └────────────────────┘
```

### Backend Comparison

| Feature | RAG Backend (Default) | OpenAI Backend |
|---------|-------------|----------------|
| **API Provider** | Azure OpenAI or Standard OpenAI | Standard OpenAI |
| **Storage** | Local (Qdrant/ChromaDB) | Cloud (OpenAI) |
| **Vector Store** | Qdrant (default) or ChromaDB | N/A (managed by OpenAI) |
| **Privacy** | ✅ Full control (local storage) | ⚠️ Data sent to OpenAI |
| **Cost** | Lower (embeddings only) | Higher (storage + search) |
| **Performance** | Fast (local vector search) | Depends on API |
| **Reranking** | ✅ Cohere supported | ❌ Not available |
| **Multi-collection** | ✅ Separate tools | ✅ Multiple stores |
| **Page tracking** | ✅ Via LlamaParse | ✅ Native support |
| **Web Search** | ✅ Tavily integration | ✅ OpenAI web_search |
| **Scalability** | High (Qdrant) / Limited (ChromaDB) | High (cloud-based) |
| **Large Documents** | ✅ Supported | ⚠️ Limited by API |

## 📚 Scripts Overview

### Main Scripts

#### `chatbot.py` - Interactive Document Q&A
The main interface for querying documents. Supports both interactive and single-query modes.

**Key Features:**
- Multiple collection support (use `--collection` or `--collections`)
- Two backends: RAG (default) or OpenAI native tools (`--openai-tools`)
- Conversation history
- Markdown table detection and Excel export
- Structured responses (content, sources, insights)

**Usage:**
```bash
# Interactive mode (RAG + Azure OpenAI)
uv run python -m llm_v2.chatbot --collection "docs1,docs2"
uv run python -m llm_v2.chatbot --collections "docs1,docs2"  # Both work!

# OpenAI native tools
uv run python -m llm_v2.chatbot --openai-tools --collection docs

# Single query
uv run python -m llm_v2.chatbot --query "What is X?" --collection docs

# With specific model
uv run python -m llm_v2.chatbot --model gpt-5 --reasoning-effort high

# Light mode (fast and cheap)
uv run python -m llm_v2.chatbot --light

# Debug mode
uv run python -m llm_v2.chatbot --info
```

#### `document_manager.py` - Document Indexing & Management
Manages document collections: upload, index, query, delete.

**Key Features:**
- Backend selection: RAG (default) or OpenAI native (`--openai-tools`)
- Vector store selection: Qdrant (default) or ChromaDB (`--chroma`)
- Document indexing with LlamaParse (high-quality parsing)
- Collection management (use `--collection` or `--collections`)
- Direct querying (for testing)
- Markdown storage from LlamaParse outputs (`--store-md`)

**Usage:**
```bash
# Index documents (RAG backend with Qdrant and Azure OpenAI)
python llm_v2/document_manager.py --collection docs --update
python llm_v2/document_manager.py --collections docs --update  # Both work!

# Use ChromaDB instead of Qdrant
python llm_v2/document_manager.py --collection docs --update --chroma

# Index with OpenAI native backend
python llm_v2/document_manager.py --openai-tools --collection docs --update

# Index and store markdown outputs from LlamaParse
python llm_v2/document_manager.py --collection docs --update --store-md

# Index using classical LlamaIndex parsing (bypass LlamaParse)
python llm_v2/document_manager.py --collection docs --update --no-llama-parse

# Query directly
python llm_v2/document_manager.py --collection docs --query "What is X?"

# Show collection info
python llm_v2/document_manager.py --collection docs --info

# Delete collection
python llm_v2/document_manager.py --collection docs --delete

# List all collections
python llm_v2/document_manager.py --list-collections
python llm_v2/document_manager.py --openai-tools --list-collections  # OpenAI backend
```

#### `company_collection_scraper.py` - Website Scraping & Collection Creation
Scrapes company websites and saves content as markdown files to a collection.

**Key Features:**
- Tavily integration for website discovery (company name search)
- Crawl4AI integration for multi-page website crawling
- Interactive website selection from search results
- Automatic markdown conversion and saving
- Collection directory creation
- Configurable crawl depth and page limits

**Usage:**
```bash
# Scrape website with company name (Tavily search)
python -m llm_v2.company_collection_scraper --collection apple --company "Apple Inc"

# Scrape website with direct URL
python -m llm_v2.company_collection_scraper --collection sonaca --website "https://www.sonaca.com"

# Delete collection content
python -m llm_v2.company_collection_scraper --collection sonaca --delete
```

**Configuration:**
Crawl4AI settings can be configured in `llm_v2/config.py`:
- `CRAWL4AI_MAX_DEPTH`: Maximum crawl depth (default: 2)
- `CRAWL4AI_MAX_PAGES`: Maximum pages to crawl (default: 50)
- `CRAWL4AI_WORD_COUNT_THRESHOLD`: Minimum words per page (default: 200)
- `CRAWL4AI_BROWSER_TYPE`: Browser type (default: "chromium")
- `CRAWL4AI_HEADLESS`: Run in headless mode (default: True)


### Supporting Modules

#### `config.py` - Centralized Configuration
All configuration parameters in one place.

**Configurable:**
- Model settings (model, reasoning effort, verbosity)
- LlamaParse options (parse mode, workers, caching)
- RAG settings (chunk size, top-k, reranking)
- Tavily web search options

#### `rag_client.py` - RAG Document Store
LlamaIndex + Qdrant (default) or ChromaDB implementation for local vector search with Azure OpenAI support.

**Features:**
- Qdrant vector store (default, Docker-based) or ChromaDB (local files)
- Azure OpenAI integration (automatic fallback to standard OpenAI)
- Document chunking with semantic splitting
- Page number extraction from LlamaParse markers
- Optional markdown storage from LlamaParse outputs
- Flexible parsing modes (LlamaParse or classical LlamaIndex)
- Cohere reranking (optional)
- Efficient caching and state management
- Optimized Qdrant configuration (HNSW parameters, segment settings)

#### `openai_document_store.py` - OpenAI Document Store
OpenAI's cloud-based vector store implementation.

**Features:**
- File upload and vector store management
- Automatic change detection
- Incremental updates

#### `tool_calling_engine.py` - Async Tool Orchestration
Handles tool execution for the `my-tools` mode.

**Features:**
- Async tool execution
- Automatic tool routing
- Response parsing and error handling

#### `rag_tool.py` - RAG Search Tool
Wraps RAG document store as a tool for multi-tool systems.

**Features:**
- Per-collection tool instances
- Unique naming for multi-collection support
- Graceful degradation when collections unavailable

#### `tavily_tool.py` - Web Search Tools
Tavily integration for web and news search.

**Features:**
- General web search
- News-specific search
- Configurable result count and depth
- Clean HTML content extraction

## ⚙️ Configuration

### Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-...              # OpenAI API key

# Optional - Web Search
TAVILY_API_KEY=tvly-...            # Tavily API key for web search

# Optional - Document Parsing
LLAMA_CLOUD_API_KEY=llx-...        # LlamaParse API key for better parsing
LLAMA_CLOUD_BASE_URL=https://api.cloud.llamaindex.ai  # Optional: for European or custom endpoints

# Optional - Reranking
COHERE_API_KEY=...                 # Cohere API key for result reranking

```

### Config.py Settings

Edit `llm_v2/config.py` for fine-tuning:

**Model Configuration**
OPENAI_MODEL=gpt-5-mini            # Model: gpt-5, gpt-5-mini, gpt-5-nano
REASONING_EFFORT=medium            # Reasoning: low, medium, high
TEXT_VERBOSITY=medium              # Verbosity: low, medium, high
EMBEDDING_MODEL=text-embedding-3-small  # For RAG backend

**RAG Configuration**
CHUNK_SIZE=1024                    # Text chunk size
CHUNK_OVERLAP=100                  # Overlap between chunks
TOP_K=20                           # Initial retrieval count
COHERE_RERANK_TOP_N=6             # Final count after reranking

**LlamaParse Settings:**
```python
USE_LLAMA_PARSE = True
LLAMA_PARSE_PARSE_MODE = "parse_page_with_llm"  # Quality mode (None = use multimodal model)
LLAMA_PARSE_NUM_WORKERS = 12                     # Parallel processing
LLAMA_PARSE_INVALIDATE_CACHE = True              # Force re-parse

# Azure OpenAI Configuration for LlamaParse (optional)
LLAMA_PARSE_USE_VENDOR_MULTIMODAL_MODEL = True   # Use vendor multimodal model
LLAMA_PARSE_VENDOR_MULTIMODAL_MODEL_NAME = "openai-gpt-5-mini"  # Vendor multimodal model name
LLAMA_PARSE_AZURE_OPENAI_DEPLOYMENT_NAME = "gpt-5-mini"  # Azure OpenAI deployment name
LLAMA_PARSE_AZURE_OPENAI_ENDPOINT = "https://..."  # Azure OpenAI endpoint URL
LLAMA_PARSE_AZURE_OPENAI_API_VERSION = "2025-01-01-preview"  # API version
```

**Tavily Settings:**
```python
DEFAULT_TAVILY_MAX_RESULTS = 5
DEFAULT_TAVILY_SEARCH_DEPTH = "basic"  # or "advanced"
DEFAULT_TAVILY_INCLUDE_ANSWER = True
```

**Qdrant Optimization Settings:**
```python
# HNSW Index Parameters
QDRANT_HNSW_M = 16  # Number of bi-directional links (12-16 recommended)
QDRANT_HNSW_EF_CONSTRUCT = 200  # Candidate list size during construction (100-200)
QDRANT_HNSW_FULL_SCAN_THRESHOLD = 10000  # Use full scan if collection smaller than this

# Memory and Storage
QDRANT_ON_DISK = False  # Store vectors on disk (False = faster, True = less RAM)
QDRANT_ON_DISK_PAYLOAD = True  # Store payload on disk (recommended)

# Segment Configuration
QDRANT_DEFAULT_SEGMENT_NUMBER = None  # Auto (set to CPU cores for optimal parallelism)
QDRANT_MAX_SEGMENT_SIZE = None  # Auto

# Optimizer Configuration
QDRANT_DELETED_THRESHOLD = 0.2  # Vacuum trigger threshold (20%)
QDRANT_INDEXING_THRESHOLD = 10000  # Min vectors before creating index
QDRANT_FLUSH_INTERVAL_SEC = 5  # Flush interval
```

**Crawl4AI Settings (for website scraping):**
```python
CRAWL4AI_BROWSER_TYPE = "chromium"  # "chromium", "firefox", "webkit"
CRAWL4AI_HEADLESS = True  # Run in headless mode
CRAWL4AI_MAX_DEPTH = 2  # Maximum crawl depth
CRAWL4AI_MAX_PAGES = 50  # Maximum pages to crawl
CRAWL4AI_WORD_COUNT_THRESHOLD = 200  # Minimum words per page
```

**Reranking Settings:**
```python
USE_COHERE_RERANK = True
COHERE_RERANK_MODEL = "rerank-english-v3.0"
COHERE_RERANK_TOP_N = 6  # Keep top 6 after reranking
```

## 💡 Usage Examples

### Basic Usage

**1. Index a single collection:**
```bash
python llm_v2/document_manager.py --collection reports --update
```

**2. Chat with the collection:**
```bash
uv run python -m llm_v2.chatbot --collection reports
```

**3. Ask a question:**
```
You> What were the revenue figures for Q4?
```

### Multi-Collection Queries

**Query across multiple years:**
```bash
uv run python -m llm_v2.chatbot --collection "2022_reports,2023_reports,2024_reports" \
  --query "Compare revenue growth across all years"
```

**Query different document types:**
```bash
uv run python -m llm_v2.chatbot --collection "financials,operations,compliance"
```

### Advanced Querying

**High-quality mode:**
```bash
uv run python -m llm_v2.chatbot --model gpt-5 --reasoning-effort high --collection reports
```

**Fast/economical mode:**
```bash
uv run python -m llm_v2.chatbot --light --collection reports
```

**With OpenAI native tools:**
```bash
uv run python -m llm_v2.chatbot --openai-tools --collection reports
```

**Debug mode (show API calls):**
```bash
uv run python -m llm_v2.chatbot --info --collection reports --query "What is X?"
```

### Table Extraction

When the chatbot detects markdown tables in responses:

```
📊 2 markdown tables detected. Would you like to save to Excel? [Y/n]:
```

- Each table is saved to a separate Excel file
- Auto-formatted with headers, borders, and column widths
- Filenames: `export_TIMESTAMP_table1.xlsx`, `export_TIMESTAMP_table2.xlsx`

### Conversation History

The chatbot maintains context across questions:

```
You> What are the main products?
Assistant> Our main products include...

You> What about their pricing?  # "their" refers to products from previous answer
Assistant> The pricing for these products is...

# Clear history
You> clear

# Show history
You> history
```

## 🎯 Advanced Features

### Multi-Collection Strategy

**Organize by Time Period:**
```
data/documents/
├── q1_2024/
├── q2_2024/
├── q3_2024/
└── q4_2024/
```

Query: `--collection "q1_2024,q2_2024,q3_2024,q4_2024"`

**Organize by Department:**
```
data/documents/
├── finance/
├── operations/
├── hr/
└── legal/
```

Query: `--collection "finance,operations"`

**Organize by Document Type:**
```
data/documents/
├── annual_reports/
├── quarterly_reports/
├── presentations/
└── press_releases/
```

### Document Processing Options

#### LlamaParse Integration (Recommended)

For best results with PDFs containing tables, charts, or complex layouts:

1. Get API key from https://cloud.llamaindex.ai/
2. Add to `.env`:
   ```bash
   LLAMA_CLOUD_API_KEY=llx-...
   # Optional: Use European endpoint for GDPR compliance
   LLAMA_CLOUD_BASE_URL=https://api.cloud.llamaindex.ai
   
   # Optional: Azure OpenAI for LlamaParse multimodal model
   LLAMA_PARSE_AZURE_OPENAI_KEY=your_azure_key  # If using Azure OpenAI with LlamaParse
   ```
3. LlamaParse automatically extracts:
   - Tables as markdown
   - Page numbers for citations
   - Text from images (OCR)
   - Complex layouts

**Azure OpenAI Integration for LlamaParse:**

You can configure LlamaParse to use Azure OpenAI's multimodal models for better document parsing. Configure in `config.py`:

```python
LLAMA_PARSE_USE_VENDOR_MULTIMODAL_MODEL = True
LLAMA_PARSE_VENDOR_MULTIMODAL_MODEL_NAME = "openai-gpt-5-mini"
LLAMA_PARSE_AZURE_OPENAI_DEPLOYMENT_NAME = "gpt-5-mini"
LLAMA_PARSE_AZURE_OPENAI_ENDPOINT = "https://your-resource.openai.azure.com/..."
LLAMA_PARSE_AZURE_OPENAI_API_VERSION = "2025-01-01-preview"
```

**Note:** 
- If `LLAMA_CLOUD_BASE_URL` is set, the system will use that endpoint (e.g., European endpoint for data residency requirements)
- When using vendor multimodal model, set `LLAMA_PARSE_PARSE_MODE = None` (multimodal mode doesn't work with `parse_page_with_llm` mode)

#### Storing Markdown Outputs (`--store-md`)

Save the markdown outputs from LlamaParse for inspection or reuse:

```bash
# Store markdown files in data/rag/markdowns/<collection>/
python llm_v2/document_manager.py --collection docs --update --store-md
```

**Use cases:**
- Inspect parsed document content
- Debug parsing issues
- Reuse markdown for other purposes
- Archive parsed documents

Markdown files are saved with the same name as the source document (e.g., `report.pdf` → `report.md`)

#### Classical Parsing Mode (`--no-llama-parse`)

Bypass LlamaParse and use standard LlamaIndex document readers:

```bash
# Use pypdf for PDFs instead of LlamaParse
python llm_v2/document_manager.py --collection docs --update --no-llama-parse
```

**When to use:**
- Faster processing without cloud API calls
- Simple text-only documents
- Testing or debugging
- No LlamaParse API key available
- Cost optimization

**Trade-offs:**
- No page number tracking
- Lower quality table extraction
- No OCR support
- Simpler layout handling

### Cohere Reranking

Improves result quality by reranking initial retrieval:

1. Get API key from https://cohere.com/
2. Add to `.env`: `COHERE_API_KEY=...`
3. Process:
   - Initial retrieval: Top 20 chunks
   - Reranking: Reduce to top 6 most relevant
   - Better quality with same token budget

### Custom Tool Combinations

**RAG + Web Search (my-tools mode):**
- Searches local documents first
- Falls back to web search for current info
- Combines results intelligently

**OpenAI Native (openai mode):**
- Uses OpenAI's file_search tool
- Built-in web_search tool
- Simpler but less control

## 🔑 API Keys Required

### Minimum Required

**Choose one of these for LLM operations:**

- **Azure OpenAI** (Recommended for RAG backend)
  - `AZURE_OPENAI_API_KEY`: Your Azure API key
  - `AZURE_OPENAI_BASE_URL`: Your Azure endpoint (e.g., `https://your-resource.openai.azure.com/openai/v1/`)
  - Get at: https://portal.azure.com/
  - Used by default for RAG backend (document processing, embeddings, responses)
  - Also required for Azure AI Search backend (for embeddings)
  
- **Standard OpenAI** (Required for `--openai-tools`)
  - `OPENAI_API_KEY`: Your OpenAI API key
  - Get at: https://platform.openai.com/api-keys
  - Required when using `--openai-tools` flag
  - Fallback for RAG backend if Azure not configured
  - Cost: Pay per use (embeddings + completions)


### Recommended Optional

- **Tavily** (`TAVILY_API_KEY`): For web search capabilities
  - Get at: https://tavily.com/
  - Free tier available
  - Enables real-time information retrieval

- **LlamaParse** (`LLAMA_CLOUD_API_KEY`): For high-quality document parsing
  - Get at: https://cloud.llamaindex.ai/
  - Free tier available (1000 pages/day)
  - Significantly improves PDF/table extraction
  - Optional: Set `LLAMA_CLOUD_BASE_URL` for European endpoint (GDPR compliance)

- **Cohere** (`COHERE_API_KEY`): For result reranking
  - Get at: https://cohere.com/
  - Free tier available
  - Improves relevance of retrieved chunks

## 🐛 Troubleshooting

### Empty Vector Store Warning

**Symptom:**
```
⚠️  RAG vector store is empty but 5 document(s) found in data/documents/collection
   Run 'python document_manager.py --collection collection --update' to index documents
```

**Solution:**
```bash
python llm_v2/document_manager.py --collection collection --update
```

### No Page Numbers in Citations

**Issue:** Citations show filenames but no page numbers

**Causes:**
1. Not using LlamaParse (using pypdf fallback)
2. LlamaParse API key not configured
3. Documents don't support page extraction

**Solution:**
1. Add `LLAMA_CLOUD_API_KEY` to `.env`
2. Re-index documents: `--update`
3. Check if supported format (PDF, DOCX, PPTX)

### Slow Queries

**Optimization steps:**

1. **Use lighter model:**
   ```bash
   python llm_v2/chatbot.py --light
   ```

2. **Reduce chunk retrieval:**
   Edit `config.py`:
   ```python
   DEFAULT_TOP_K = 10  # Reduced from 20 (default)
   ```
   
   **Note:** The RAG tool now correctly uses `DEFAULT_TOP_K` from config (20 chunks by default). Before reranking, this determines how many initial chunks are retrieved from the vector store.

3. **Disable reranking:**
   Edit `config.py`:
   ```python
   USE_COHERE_RERANK = False
   ```

### Qdrant Connection Issues

**Qdrant not starting or connection errors:**

The system now provides clear error messages when Qdrant is configured but not responding:

```
❌ Error: Qdrant is configured but not responding.
   Host: localhost:6333
   Error: Connection refused
   Please ensure Qdrant is running and accessible.
   You can start Qdrant with: docker run -p 6333:6333 qdrant/qdrant
```

**Troubleshooting steps:**

1. **Check Docker is running:**
   ```bash
   docker ps
   ```

2. **Start Qdrant:**
   ```bash
   docker-compose up -d
   # Or manually:
   docker run -p 6333:6333 qdrant/qdrant
   ```

3. **Check Qdrant health:**
   ```bash
   curl http://localhost:6333/health
   ```

4. **View Qdrant logs:**
   ```bash
   docker logs qdrant
   ```

5. **Verify connection:**
   The system automatically tests the connection during initialization. If you see connection errors, ensure:
   - Qdrant container is running (`docker ps | grep qdrant`)
   - Port 6333 is accessible (not blocked by firewall)
   - Host and port in config match your setup

6. **Use ChromaDB as fallback:**
   ```bash
   python llm_v2/document_manager.py --collection docs --update --chroma
   ```

### Import Errors

**Missing dependencies:**
```bash
uv sync
# or
pip install -e .
```

**Module not found:**
```bash
# Make sure you're in the right directory
cd /path/to/llm-v2

# Or add to PYTHONPATH
export PYTHONPATH=/path/to/llm-v2:$PYTHONPATH
```

### Memory Issues

**Large documents causing OOM:**

1. **Reduce chunk size:**
   ```python
   DEFAULT_CHUNK_SIZE = 1024  # Reduced from 2048
   ```

2. **Process fewer documents:**
   Split into multiple collections

3. **Reduce workers:**
   ```python
   LLAMA_PARSE_NUM_WORKERS = 4  # Reduced from 12 (default)
   ```

4. **Optimize Qdrant settings (in config.py):**
   ```python
   QDRANT_ON_DISK = True  # Store vectors on disk instead of RAM
   QDRANT_DEFAULT_SEGMENT_NUMBER = 2  # Reduce parallelism
   ```

## 📁 Project Structure

```
llm-v2/
├── llm_v2/                          # Main package
│   ├── chatbot.py                   # Interactive Q&A interface
│   ├── document_manager.py          # Document indexing CLI (unified interface)
│   ├── config.py                    # Centralized configuration
│   ├── rag_client.py                # RAG implementation (LlamaIndex + ChromaDB)
│   ├── openai_document_store.py     # OpenAI vector store implementation
│   ├── rag_tool.py                  # RAG search tool wrapper
│   ├── tavily_tool.py               # Web search tools (Tavily)
│   ├── tool_calling_engine.py       # Async tool orchestration
│   ├── tool_base.py                 # Base tool interface
│   ├── document_store_base.py       # Document store interface
│   ├── document_store_factory.py    # Factory for creating stores
│   └── utils.py                     # Utility functions
├── archive/                          # Archived scripts (Azure AI Search)
│   ├── azure_document_store.py      # (Archived)
│   └── azure_indexer_document_store.py  # (Archived)
├── data/
│   ├── documents/                   # Document collections (organized by folder)
│   │   ├── collection/              # Default collection
│   │   ├── financials/              # Example: Financial documents
│   │   └── operations/              # Example: Operations documents
│   └── rag/
│       ├── qdrant/                  # Qdrant vector store data (Docker volume)
│       ├── chroma_stores/           # ChromaDB vector stores (legacy/--chroma)
│       ├── vector_stores/           # Collection state files (RAG backend)
│       ├── markdowns/               # Markdown outputs from LlamaParse (with --store-md)
│       ├── openai_state.json        # OpenAI backend state
│       └── rag_state.json           # RAG backend state (per collection)
├── tests/                           # Test files
├── .env                            # Environment variables (create this)
├── .python-version                 # Python version for uv
├── pyproject.toml                  # Python project config (PEP 621)
├── uv.lock                         # Dependency lock file (uv)
├── poetry.lock                     # Legacy lock file (can be removed)
└── README.md                       # This file
```

## 🔄 Workflow

### Typical Usage Flow

```
1. Organize Documents
   └─> Place PDFs/docs in data/documents/<collection_name>/

2. Index Collection (RAG backend with Azure OpenAI or OpenAI)
   └─> python llm_v2/document_manager.py --collection <name> --update

3. Query Documents
   ├─> Interactive: uv run python -m llm_v2.chatbot --collection <name>
   └─> Single query: uv run python -m llm_v2.chatbot --collection <name> --query "..."

4. (Optional) Update Index
   └─> Re-run step 2 when documents change
```

### Multi-Collection Workflow

```
1. Organize by Category
   ├─> data/documents/2023_reports/
   ├─> data/documents/2024_reports/
   └─> data/documents/forecasts/

2. Index Each Collection (RAG backend with Azure OpenAI or OpenAI)
   ├─> python llm_v2/document_manager.py --collection 2023_reports --update
   ├─> python llm_v2/document_manager.py --collection 2024_reports --update
   └─> python llm_v2/document_manager.py --collection forecasts --update

3. Query Multiple Collections
   └─> uv run python -m llm_v2.chatbot --collection "2023_reports,2024_reports,forecasts"
```

## 📊 Performance Tips

### For Best Quality
```bash
# Use GPT-5 with high reasoning
uv run python -m llm_v2.chatbot \
  --model gpt-5 \
  --reasoning-effort high \
  --collection docs

# Enable all features in .env
LLAMA_CLOUD_API_KEY=...  # Better parsing
COHERE_API_KEY=...       # Better ranking
TAVILY_API_KEY=...       # Web search
```

### For Best Speed
```bash
# Use light mode
uv run python -m llm_v2.chatbot --light --collection docs

# Or manually specify
uv run python -m llm_v2.chatbot \
  --model gpt-5-nano \
  --reasoning-effort low \
  --collection docs
```

### For Best Cost
```bash
# Use RAG backend with Azure OpenAI (default)
python llm_v2/document_manager.py --update

# Use nano model
uv run python -m llm_v2.chatbot --model gpt-5-nano

# Disable reranking
# Edit config.py: USE_COHERE_RERANK = False
```

## 🤝 Contributing

This is a private project, but suggestions and improvements are welcome:

1. Document any issues with detailed steps to reproduce
2. Suggest features with use cases
3. Share performance optimization ideas

## 📄 License

Private project. All rights reserved.

## 📞 Support

For questions or issues:
- Check the [Troubleshooting](#troubleshooting) section
- Review the markdown documentation files in the project root
- Contact: luc.machiels@karomia.eu

---

**Version:** 0.3.0
**Last Updated:** November 2025  
**Python:** 3.12+


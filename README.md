# LLM-v2: Advanced Document Q&A System

A powerful, flexible document Q&A system that combines multiple document search backends with web search capabilities to provide comprehensive answers to questions about your document collections.

## 🌟 Key Features

### Multi-Backend Support
- **RAG Backend (Default)**: Local vector search using LlamaIndex + ChromaDB with Azure OpenAI
- **OpenAI Backend**: Cloud-based using OpenAI's native file_search tool (use `--openai-tools` flag)
- **Hybrid Approach**: Combine document search with real-time web search
- **Flexible API**: Auto-detects Azure OpenAI or falls back to standard OpenAI

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
- Poetry (recommended) or pip

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd llm-v2
```

2. **Install dependencies**

Using Poetry (recommended):
```bash
poetry install
poetry shell
```

Using pip:
```bash
pip install -r requirements.txt
```

3. **Configure environment variables**

Create a `.env` file in the project root:

```bash
# Azure OpenAI (used for RAG backend - default)
AZURE_OPENAI_API_KEY=your_azure_api_key_here
AZURE_OPENAI_BASE_URL=https://your-resource.openai.azure.com/openai/v1/

# Standard OpenAI (used for --openai-tools backend)
OPENAI_API_KEY=your_openai_api_key_here

# Optional but recommended
TAVILY_API_KEY=your_tavily_api_key_here        # For web search
LLAMA_CLOUD_API_KEY=your_llama_cloud_api_key  # For LlamaParse
COHERE_API_KEY=your_cohere_api_key            # For reranking

```

4. **Create document directories**
```bash
mkdir -p data/documents/collection
mkdir -p data/rag/chroma_stores
```

## ⚡ Quick Start

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

**Using RAG Backend (default, uses Azure OpenAI if configured):**
```bash
python llm_v2/document_manager.py --collection financials --update
# Or explicitly:
python llm_v2/document_manager.py --collections financials --update  # Both work!
```

**Using OpenAI Backend (with native file_search):**
```bash
python llm_v2/document_manager.py --openai-tools --collection financials --update
```

### 3. Start Chatting

**Interactive mode (RAG backend with Azure OpenAI):**
```bash
python llm_v2/chatbot.py --collection financials
# Or use --collections alias:
python llm_v2/chatbot.py --collections financials
```

**Query multiple collections:**
```bash
python llm_v2/chatbot.py --collection "financials,operations,collection"
# Or:
python llm_v2/chatbot.py --collections "financials,operations,collection"
```

**Single query:**
```bash
python llm_v2/chatbot.py --collection financials --query "What were the key metrics in 2023?"
```

**Using OpenAI native tools (file_search + web_search):**
```bash
python llm_v2/chatbot.py --openai-tools --collection financials
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
┌─────────────────────────────────────────────────────────────┐
│                        Chatbot                               │
│  ┌──────────────────┐           ┌──────────────────┐       │
│  │   OpenAI Tools   │           │    My-Tools      │       │
│  │  - file_search   │           │  - RAG Search    │       │
│  │  - web_search    │           │  - Tavily Web    │       │
│  └──────────────────┘           │  - Tavily News   │       │
│                                  └──────────────────┘       │
└─────────────────────────────────────────────────────────────┘
                    │                           │
                    ▼                           ▼
    ┌───────────────────────┐   ┌──────────────────────────┐
    │  OpenAI Vector Store  │   │   RAG Document Store     │
    │  - Cloud-based        │   │   - LlamaIndex           │
    │  - Managed by OpenAI  │   │   - ChromaDB             │
    └───────────────────────┘   │   - Cohere Rerank        │
                                │   - LlamaParse           │
                                └──────────────────────────┘
```

### Backend Comparison

| Feature | RAG Backend (Default) | OpenAI Backend (`--openai-tools`) |
|---------|-------------|----------------|
| **API Provider** | Azure OpenAI (configurable) | Standard OpenAI |
| **Storage** | Local (ChromaDB) | Cloud (OpenAI) |
| **Privacy** | ✅ Full control | ⚠️ Data sent to OpenAI |
| **Cost** | Lower (embeddings only) | Higher (storage + search) |
| **Performance** | Fast (local) | Depends on API |
| **Reranking** | ✅ Cohere supported | ❌ Not available |
| **Multi-collection** | ✅ Separate tools per collection | ✅ Multiple vector stores |
| **Page tracking** | ✅ Via LlamaParse | ✅ Native support |
| **Web Search** | ✅ Tavily integration | ✅ OpenAI web_search |

## 📚 Scripts Overview

### Main Scripts

#### `chatbot.py` - Interactive Document Q&A
The main interface for querying documents. Supports both interactive and single-query modes.

**Key Features:**
- Multiple collection support (use `--collection` or `--collections`)
- Two backends: RAG (default, uses Azure OpenAI) or OpenAI native tools (`--openai-tools`)
- Conversation history
- Markdown table detection and Excel export
- Structured responses (content, sources, insights)

**Usage:**
```bash
# Interactive mode (RAG + Azure OpenAI)
python llm_v2/chatbot.py --collection "docs1,docs2"
python llm_v2/chatbot.py --collections "docs1,docs2"  # Both work!

# OpenAI native tools
python llm_v2/chatbot.py --openai-tools --collection docs

# Single query
python llm_v2/chatbot.py --query "What is X?" --collection docs

# With specific model
python llm_v2/chatbot.py --model gpt-5 --reasoning-effort high

# Light mode (fast and cheap)
python llm_v2/chatbot.py --light

# Debug mode
python llm_v2/chatbot.py --info
```

#### `document_manager.py` - Document Indexing & Management
Manages document collections: upload, index, query, delete.

**Key Features:**
- Backend selection: RAG (default, uses Azure OpenAI) or OpenAI native (`--openai-tools`)
- Document indexing
- Collection management (use `--collection` or `--collections`)
- Direct querying (for testing)

**Usage:**
```bash
# Index documents (RAG backend with Azure OpenAI)
python llm_v2/document_manager.py --collection docs --update
python llm_v2/document_manager.py --collections docs --update  # Both work!

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
python llm_v2/document_manager.py --openai-tools --list-collections  # For OpenAI backend
```

### Supporting Modules

#### `config.py` - Centralized Configuration
All configuration parameters in one place.

**Configurable:**
- Model settings (model, reasoning effort, verbosity)
- LlamaParse options (parse mode, workers, caching)
- RAG settings (chunk size, top-k, reranking)
- Tavily web search options

#### `rag_client.py` - RAG Document Store
LlamaIndex + ChromaDB implementation for local vector search with Azure OpenAI support.

**Features:**
- Azure OpenAI integration (automatic fallback to standard OpenAI)
- Document chunking with semantic splitting
- Page number extraction from LlamaParse markers
- Optional markdown storage from LlamaParse outputs
- Flexible parsing modes (LlamaParse or classical LlamaIndex)
- Cohere reranking (optional)
- Efficient caching and state management

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
LLAMA_PARSE_PARSE_MODE = "parse_page_with_llm"  # Quality mode
LLAMA_PARSE_NUM_WORKERS = 10                     # Parallel processing
LLAMA_PARSE_INVALIDATE_CACHE = False             # Force re-parse
```

**Tavily Settings:**
```python
DEFAULT_TAVILY_MAX_RESULTS = 5
DEFAULT_TAVILY_SEARCH_DEPTH = "basic"  # or "advanced"
DEFAULT_TAVILY_INCLUDE_ANSWER = True
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
python llm_v2/chatbot.py --collection reports
```

**3. Ask a question:**
```
You> What were the revenue figures for Q4?
```

### Multi-Collection Queries

**Query across multiple years:**
```bash
python llm_v2/chatbot.py --collection "2022_reports,2023_reports,2024_reports" \
  --query "Compare revenue growth across all years"
```

**Query different document types:**
```bash
python llm_v2/chatbot.py --collection "financials,operations,compliance"
```

### Advanced Querying

**High-quality mode:**
```bash
python llm_v2/chatbot.py --model gpt-5 --reasoning-effort high --collection reports
```

**Fast/economical mode:**
```bash
python llm_v2/chatbot.py --light --collection reports
```

**With OpenAI native tools:**
```bash
python llm_v2/chatbot.py --openai-tools --collection reports
```

**Debug mode (show API calls):**
```bash
python llm_v2/chatbot.py --info --collection reports --query "What is X?"
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
2. Add to `.env`: `LLAMA_CLOUD_API_KEY=llx-...`
3. LlamaParse automatically extracts:
   - Tables as markdown
   - Page numbers for citations
   - Text from images (OCR)
   - Complex layouts

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

### Import Errors

**Missing dependencies:**
```bash
poetry install
# or
pip install -r requirements.txt
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
   LLAMA_PARSE_NUM_WORKERS = 4  # Reduced from 10
   ```

## 📁 Project Structure

```
llm-v2/
├── llm_v2/                          # Main package
│   ├── chatbot.py                   # Interactive Q&A interface
│   ├── document_manager.py          # Document indexing CLI
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
├── data/
│   ├── documents/                   # Document collections (organized by folder)
│   │   ├── collection/              # Default collection
│   │   ├── financials/              # Example: Financial documents
│   │   └── operations/              # Example: Operations documents
│   └── rag/
│       ├── chroma_stores/           # ChromaDB vector stores
│       ├── markdowns/               # Markdown outputs from LlamaParse (with --store-md)
│       ├── openai_state.json        # OpenAI backend state
│       └── rag_state.json           # RAG backend state (per collection)
├── tests/                           # Test files
├── .env                            # Environment variables (create this)
├── pyproject.toml                  # Python project config
├── poetry.lock                     # Dependency lock file
└── README.md                       # This file
```

## 🔄 Workflow

### Typical Usage Flow

```
1. Organize Documents
   └─> Place PDFs/docs in data/documents/<collection_name>/

2. Index Collection (RAG backend with Azure OpenAI)
   └─> python document_manager.py --collection <name> --update

3. Query Documents
   ├─> Interactive: python chatbot.py --collection <name>
   └─> Single query: python chatbot.py --collection <name> --query "..."

4. (Optional) Update Index
   └─> Re-run step 2 when documents change
```

### Multi-Collection Workflow

```
1. Organize by Category
   ├─> data/documents/2023_reports/
   ├─> data/documents/2024_reports/
   └─> data/documents/forecasts/

2. Index Each Collection (RAG backend with Azure OpenAI)
   ├─> python document_manager.py --collection 2023_reports --update
   ├─> python document_manager.py --collection 2024_reports --update
   └─> python document_manager.py --collection forecasts --update

3. Query Multiple Collections
   └─> python chatbot.py --collection "2023_reports,2024_reports,forecasts"
```

## 📊 Performance Tips

### For Best Quality
```bash
# Use GPT-5 with high reasoning
python llm_v2/chatbot.py \
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
python llm_v2/chatbot.py --light --collection docs

# Or manually specify
python llm_v2/chatbot.py \
  --model gpt-5-nano \
  --reasoning-effort low \
  --collection docs
```

### For Best Cost
```bash
# Use RAG backend with Azure OpenAI (default)
python llm_v2/document_manager.py --update

# Use nano model
python llm_v2/chatbot.py --model gpt-5-nano

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

**Version:** 0.1.0
**Last Updated:** October 20, 2025  
**Python:** 3.12+
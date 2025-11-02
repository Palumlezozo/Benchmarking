# Cadenza IRO Verification

This folder contains the IRO (Impacts, Risks, Opportunities) verification tool that uses RAG (Retrieval Augmented Generation) and web search to verify factual claims against document collections.

## Overview

The `verify_statements.py` script reads an Excel file containing IRO questions and contexts, extracts the IRO statement and stated facts, verifies the facts against a document collection using RAG (with web search fallback), and outputs a new Excel file with comprehensive verification results.

## Features

- **IRO Extraction**: Automatically extracts the Impact/Risk/Opportunity statement from questions
- **Fact Identification**: Identifies concrete factual claims that support the IRO evaluation (excluding the IRO itself)
- **Multi-Source Verification**: 
  - Primary: Searches document collections using RAG
  - Fallback: Uses Tavily web search for facts not found in documents
- **Parallel Processing**: Verifies multiple IROs concurrently for faster processing (up to 5 by default)
- **Real-time Monitoring**: Shows which row/IRO is being processed in the terminal
- **Structured Verification**: For each IRO, provides:
  1. IRO Statement: Clear one-sentence description of the possibility being evaluated
  2. Stated Facts: List of concrete factual claims from the context (excluding the IRO)
  3. Facts Verified: Whether the stated facts are supported (yes/no/partial)
  4. Evidence Description: What was found in documents or web that supports/refutes the facts
  5. Sources Consulted: All sources used (documents with page numbers and web URLs)
- **Azure OpenAI Integration**: Uses Azure OpenAI with tool calling engine and thinking process
- **Progress Tracking**: Shows real-time progress bar and completion status
- **Formatted Output**: Creates a new Excel file with original data plus verification results

## Prerequisites

1. **Environment Setup**: Make sure your `.env` file contains:
   ```
   AZURE_OPENAI_API_KEY=your-azure-api-key
   AZURE_OPENAI_BASE_URL=https://your-resource.openai.azure.com/openai/v1/
   LLAMA_CLOUD_API_KEY=your-llama-cloud-api-key
   TAVILY_API_KEY=your-tavily-api-key  # Optional: for web search fallback
   ```
   
   **Note**: The `TAVILY_API_KEY` is optional but recommended. If not provided, the tool will only search documents without web fallback.

2. **Document Collection**: Ensure you have indexed your document collection using the main `document_manager.py`:
   ```bash
   python llm_v2/document_manager.py --collection your_collection --update
   ```

## Usage

### Basic Usage

```bash
python cadenza/verify_statements.py --input FILE.xlsx --collection COLLECTION
```

### Parameters

- `--input` (required): Path to the input Excel file containing IROs to verify
- `--collection` (required): Name of the document collection to search against
- `--output` (optional): Path to the output file (default: `results_{input_filename}.xlsx`)
- `--question-column` (optional): Name of the column containing IRO questions (default: `questions`)
- `--context-column` (optional): Name of the column containing contexts with stated facts (default: `contexts`)
- `--model` (optional): OpenAI model to use (default: from config)
- `--concurrent` (optional): Maximum number of concurrent verifications (default: 5)
- `--limit` (optional): Process only the first N IROs (for testing)
- `--no-rerank` (optional): Disable Cohere reranking to avoid rate limits (faster, less accurate)

### Examples

**Verify IROs in a file:**
```bash
python cadenza/verify_statements.py \
  --input cadenza/hannecard_iros.xlsx \
  --collection hannecard
```
This will create `cadenza/results_hannecard_iros.xlsx`

**Specify output file:**
```bash
python cadenza/verify_statements.py \
  --input cadenza/hannecard_iros.xlsx \
  --collection hannecard \
  --output data/verified_results.xlsx
```

**Use custom column names:**
```bash
python cadenza/verify_statements.py \
  --input data/iros.xlsx \
  --collection hannecard \
  --question-column "iro_questions" \
  --context-column "supporting_facts"
```

**Test with first 5 IROs only:**
```bash
python cadenza/verify_statements.py \
  --input cadenza/hannecard_iros.xlsx \
  --collection hannecard \
  --limit 5
```

**Control concurrency level:**
```bash
# Process up to 10 IROs in parallel (faster, more API usage)
python cadenza/verify_statements.py \
  --input cadenza/hannecard_iros.xlsx \
  --collection hannecard \
  --concurrent 10

# Process only 2 at a time (slower, but more conservative)
python cadenza/verify_statements.py \
  --input cadenza/hannecard_iros.xlsx \
  --collection hannecard \
  --concurrent 2
```

## Input File Format

The input Excel file should have two columns:
- `questions` (or specify a different name with `--question-column`): Contains the question that evaluates the IRO
- `contexts` (or specify a different name with `--context-column`): Contains the factual claims supporting the IRO

Example:

| ID | questions | contexts | category |
|----|-----------|----------|----------|
| 1  | Does the company face water scarcity risks? | The company operates in water-stressed regions. Water consumption increased by 20%. | Environmental |
| 2  | Is there an opportunity for renewable energy transition? | Current energy mix is 80% fossil fuels. Renewable energy costs have decreased significantly. | Energy |

## Output File Format

The output Excel file will contain all original columns plus five new columns:

| ID | questions | contexts | category | **IRO Statement** | **Stated Facts** | **Facts Verified (yes/no/partial)** | **Evidence Description** | **Sources Consulted** |
|----|-----------|----------|----------|-------------------|------------------|-------------------------------------|--------------------------|----------------------|
| 1  | Does the company face water scarcity risks? | The company operates in water-stressed regions. Water consumption increased by 20%. | Environmental | The company may face water scarcity risks | - The company operates in water-stressed regions<br>- Water consumption increased by 20% | yes | Documents confirm operations in water-stressed regions (Annual Report, page 12). Water consumption increase verified (Sustainability Report, page 8). | Annual Report 2024 (page 12), Sustainability Report 2024 (page 8) |
| 2  | Is there an opportunity for renewable energy transition? | Current energy mix is 80% fossil fuels. Renewable energy costs have decreased significantly. | Energy | There is an opportunity for renewable energy transition | - Current energy mix is 80% fossil fuels<br>- Renewable energy costs have decreased significantly | partial | Energy mix confirmed in documents (Annual Report, page 15). Renewable cost claim verified via web search. | Annual Report 2024 (page 15), Web: https://example.com/renewable-costs |

## How It Works

1. **Load Input File**: Reads the Excel file and identifies the question and context columns
2. **Parallel Processing**: Processes multiple IROs concurrently (up to 5 by default)
3. **For Each IRO** (in parallel):
   - Displays which row is being processed in the terminal
   - **Extract IRO**: Identifies the Impact/Risk/Opportunity statement from the question
   - **Identify Facts**: Lists the concrete factual claims from the context (excluding the IRO itself)
   - **Verify Facts**: Uses the Tool Calling Engine to verify facts:
     - First searches the document collection using RAG
     - If facts are not found, falls back to web search (if Tavily API key is available)
   - Thinking process is enabled for deeper reasoning
   - LLM analyzes all evidence and generates structured verification results
   - Shows completion status for each row
4. **Save Results**: Writes a new Excel file with all original data plus verification results

## Architecture

The script uses the same RAG infrastructure as the main chatbot, plus web search:

- **Tool Calling Engine** (`llm_v2/tool_calling_engine.py`): Orchestrates tool usage and enables thinking process
- **RAG Tool** (`llm_v2/rag_tool.py`): Searches document collections using LlamaIndex and ChromaDB
- **Tavily Tool** (`llm_v2/tavily_tool.py`): Searches the web for facts not found in documents
- **Azure OpenAI**: Analyzes search results and verifies facts with extended reasoning
- **Async Processing**: Uses asyncio for parallel verification with semaphore-based rate limiting
- **Structured Output**: Uses Pydantic models to ensure consistent verification format

## Key Concepts

### IRO vs Facts Separation

The script distinguishes between:
- **IRO (Possibility)**: The Impact/Risk/Opportunity being evaluated - e.g., "The company may face water scarcity risks"
- **Facts (Reality)**: Concrete factual claims supporting the evaluation - e.g., "The company operates in water-stressed regions"

This separation is crucial because:
- We verify the **facts**, not the IRO possibility itself
- IROs are future-oriented possibilities that may not exist in documents
- Facts are concrete claims about current reality that can be verified
- This prevents getting "partial" results when IRO possibilities aren't directly mentioned

### Multi-Source Verification

1. **Primary Source**: Documents in the collection
   - Searches using semantic similarity
   - Returns relevant chunks with page numbers
2. **Fallback Source**: Web search via Tavily
   - Used only when facts aren't found in documents
   - Returns web pages with URLs
   - Helps verify general facts not in company documents

Sources are clearly marked in the output to distinguish between document-based and web-based evidence.

## Troubleshooting

### "Collection not found"
Make sure you've indexed your documents first:
```bash
python llm_v2/document_manager.py --collection COLLECTION --update
```

### "Azure OpenAI credentials not found"
Check your `.env` file and ensure `AZURE_OPENAI_API_KEY` and `AZURE_OPENAI_BASE_URL` are set.

### "Column 'questions' or 'contexts' not found"
Either:
- Rename your columns to `questions` and `contexts`, OR
- Use `--question-column` and `--context-column` parameters to specify your column names

### Web search not working
If you see "TAVILY_API_KEY not found" warning:
- Add `TAVILY_API_KEY` to your `.env` file to enable web search fallback
- Without it, the tool will only search documents (still functional, just no web fallback)

### Empty or incorrect results
- Verify your document collection has been properly indexed
- Check that the collection name matches exactly
- Ensure documents contain relevant information
- Review the separation of IRO (possibility) vs Facts (concrete claims) in your contexts

## Performance

- **Parallel Processing**: Up to 5 IROs processed concurrently by default (configurable with `--concurrent`)
- Processing time depends on:
  - Number of IROs to verify
  - Size of the document collection
  - Whether web search fallback is needed
  - Model used for analysis
  - Concurrency level (higher = faster, but more API usage)
- Typical processing time: 
  - 8-15 seconds per IRO with documents only
  - 15-25 seconds per IRO if web search fallback is used
  - Effective rate: ~3-5 seconds per IRO with parallel processing (default concurrency of 5)
- Progress is shown in real-time with live updates for each row
- Real-time monitoring shows which rows are being processed

## Notes

- **Concurrency Control**: Uses asyncio.Semaphore to limit parallel requests and respect API rate limits
- **Terminal Monitoring**: Each row being processed is displayed in the terminal with color-coded status
- **Smart Fallback**: Web search is only used when facts aren't found in documents (reduces API calls)
- **Clear Source Attribution**: Output clearly indicates whether evidence came from documents or web
- **IRO Separation**: The IRO possibility is separated from facts to enable accurate verification
- Empty questions or contexts are skipped with a warning
- Errors during verification are captured in the output (marked as "error")
- The output file uses `results_` prefix for easy identification (e.g., `results_hannecard_iros.xlsx`)
- Progress bar shows overall completion with elapsed time


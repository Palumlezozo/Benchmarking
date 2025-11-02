"""
Centralized configuration for all scripts.

This module provides a single source of truth for all common configuration parameters
used across scripts, ensuring consistency and maintainability.
"""

import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field


@dataclass
class Config:
    """Centralized configuration for all DMA v2 scripts."""
    
    # Regular Model Configuration
    DEFAULT_MODEL: str = "gpt-5-mini"
    DEFAULT_REASONING_EFFORT: str = "medium"
    DEFAULT_TEXT_VERBOSITY: str = "medium"
    
    # High-End Model Configuration
    DEFAULT_MODEL_HIGH: str = "gpt-5"
    DEFAULT_REASONING_EFFORT_HIGH: str = "high"
    DEFAULT_TEXT_VERBOSITY_HIGH: str = "high"
    
    # Model Support
    SUPPORTED_MODELS: List[str] = field(default_factory=lambda: ["gpt-5", "gpt-5-mini", "gpt-5-nano"])
    
    # Embedding Model Configuration (for RAG)
    DEFAULT_EMBEDDING_MODEL: str = "text-embedding-3-large"
    SUPPORTED_EMBEDDING_MODELS: List[str] = field(default_factory=lambda: [
        "text-embedding-3-small",
        "text-embedding-3-large",
        "text-embedding-ada-002"
    ])
    
    # Rate Limiting Configuration
    EMBEDDING_BATCH_SIZE: int = 20  # Chunks per embedding API call (reduce if hitting rate limits)
    EMBEDDING_DELAY_SECONDS: float = 1  # Delay between embedding batches (increase if hitting rate limits)
    MAX_NODES_PER_BATCH: int = 40  # Max nodes to process before waiting (helps with rate limits)
    
    # Tavily Web Search Configuration
    DEFAULT_TAVILY_MAX_RESULTS: int = 5
    DEFAULT_TAVILY_SEARCH_DEPTH: str = "basic"  # Options: "basic", "advanced"
    DEFAULT_TAVILY_INCLUDE_ANSWER: bool = True
    DEFAULT_TAVILY_INCLUDE_RAW_CONTENT: bool = False
    DEFAULT_TAVILY_MAX_CONTENT_LENGTH: int = 500  # Max chars per snippet
    
    # LlamaParse Document Parsing Configuration
    USE_LLAMA_PARSE: bool = True  # Use LlamaParse if API key available
    LLAMA_PARSE_RESULT_TYPE: str = "markdown"  # Options: "text", "markdown"
    LLAMA_PARSE_VERBOSE: bool = False
    LLAMA_PARSE_LANGUAGE: str = "en"  # Languages for OCR (English, French, Dutch)
    LLAMA_PARSE_PARSE_MODE: str = "parse_page_with_llm"
    LLAMA_PARSE_INVALIDATE_CACHE: bool = True  # Force re-parsing of cached documents (set to True to refresh)
    LLAMA_PARSE_DO_NOT_CACHE: bool = True  # Don't cache parsing results (TEMPORARY FOR DEBUGGING)
    LLAMA_PARSE_NUM_WORKERS: int = 12  # Number of workers for parallel processing
    # Advanced parsing options -> disabled if without_llm 
    LLAMA_PARSE_SKIP_DIAGONAL_TEXT: bool = False
    LLAMA_PARSE_SPREADSHEET_EXTRACT_SUB_TABLES: bool = False  # Extract sub-tables from spreadsheets
    LLAMA_PARSE_SPREADSHEET_FORCE_FORMULA_COMPUTATION: bool = False  # Force formula computation in spreadsheets
    # Large document partitioning (but then, we can't do batch processing)
    LLAMA_PARSE_PARTITION_PAGES: Optional[int] = None  # Number of pages per partition for large documents (None = no partitioning)
    
    # Processing Configuration
    DEFAULT_MAX_CONCURRENT: int = 15
    DEFAULT_TIMEOUT: int = 200
    
    # Similarity and Deduplication
    DEFAULT_SIMILARITY_THRESHOLD: float = 0.8
    
    # Chunking Configuration
    DEFAULT_CHUNK_SIZE: int = 1024
    DEFAULT_CHUNK_OVERLAP: int = 100
    DEFAULT_TOP_K: int = 20  # Number of chunks to retrieve before reranking
    
    # Cohere Rerank Configuration
    USE_COHERE_RERANK: bool = True  # Use Cohere rerank if API key available
    COHERE_RERANK_MODEL: str = "rerank-english-v3.0"  # Cohere rerank model
    COHERE_RERANK_TOP_N: int = 6  # Number of chunks to keep after reranking
    
    # Tool Calling Configuration
    MAX_TOOL_ITERATIONS: int = 4  # Maximum number of tool-calling iterations in the reasoning loop

# Global configuration instance
config = Config()


def parse_model_config(model_string: str) -> tuple[str, str, str]:
    """
    Parse combined model string: 'model,reasoning_effort,text_verbosity'
    
    Args:
        model_string: String in format 'model,reasoning_effort,text_verbosity'
        
    Returns:
        Tuple of (model, reasoning_effort, text_verbosity)
        
    Raises:
        ValueError: If any part of the configuration is invalid
    """
    parts = model_string.split(',')
    
    model = parts[0].strip() if len(parts) > 0 else config.DEFAULT_MODEL
    reasoning_effort = parts[1].strip() if len(parts) > 1 else config.DEFAULT_REASONING_EFFORT  
    text_verbosity = parts[2].strip() if len(parts) > 2 else config.DEFAULT_TEXT_VERBOSITY
    
    # Validate each part
    if model not in config.SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model: {model}. Supported models: {config.SUPPORTED_MODELS}")
    if reasoning_effort not in ["low", "medium", "high"]:
        raise ValueError(f"Invalid reasoning effort: {reasoning_effort}. Must be one of: low, medium, high")
    if text_verbosity not in ["low", "medium", "high"]:
        raise ValueError(f"Invalid text verbosity: {text_verbosity}. Must be one of: low, medium, high")
    
    return model, reasoning_effort, text_verbosity


def get_model_from_env() -> str:
    """Get model from environment variable or use default."""
    return os.getenv("OPENAI_MODEL", config.DEFAULT_MODEL)


def get_reasoning_effort_from_env() -> str:
    """Get reasoning effort from environment variable or use default."""
    return os.getenv("REASONING_EFFORT", config.DEFAULT_REASONING_EFFORT)


def get_text_verbosity_from_env() -> str:
    """Get text verbosity from environment variable or use default."""
    return os.getenv("TEXT_VERBOSITY", config.DEFAULT_TEXT_VERBOSITY)


def get_model_config_from_env() -> tuple[str, str, str]:
    """Get regular model config from environment variable or use defaults."""
    model_string = os.getenv("OPENAI_MODEL_CONFIG")
    if model_string:
        return parse_model_config(model_string)
    else:
        return (
            get_model_from_env(),
            get_reasoning_effort_from_env(), 
            get_text_verbosity_from_env()
        )


def get_model_high_config_from_env() -> tuple[str, str, str]:
    """Get high-end model config from environment variable or use defaults."""
    model_string = os.getenv("OPENAI_MODEL_HIGH_CONFIG")
    if model_string:
        return parse_model_config(model_string)
    else:
        return (
            os.getenv("OPENAI_MODEL_HIGH", config.DEFAULT_MODEL_HIGH),
            os.getenv("REASONING_EFFORT_HIGH", config.DEFAULT_REASONING_EFFORT_HIGH),
            os.getenv("TEXT_VERBOSITY_HIGH", config.DEFAULT_TEXT_VERBOSITY_HIGH)
        )


def get_max_concurrent_from_env() -> int:
    """Get max concurrent from environment variable or use default."""
    try:
        return int(os.getenv("MAX_CONCURRENT", config.DEFAULT_MAX_CONCURRENT))
    except ValueError:
        return config.DEFAULT_MAX_CONCURRENT


def get_timeout_from_env() -> int:
    """Get timeout from environment variable or use default."""
    try:
        return int(os.getenv("TIMEOUT", config.DEFAULT_TIMEOUT))
    except ValueError:
        return config.DEFAULT_TIMEOUT


def get_similarity_threshold_from_env() -> float:
    """Get similarity threshold from environment variable or use default."""
    try:
        return float(os.getenv("SIMILARITY_THRESHOLD", config.DEFAULT_SIMILARITY_THRESHOLD))
    except ValueError:
        return config.DEFAULT_SIMILARITY_THRESHOLD


def get_embedding_model_from_env() -> str:
    """Get embedding model from environment variable or use default."""
    return os.getenv("EMBEDDING_MODEL", config.DEFAULT_EMBEDDING_MODEL)


def add_common_arguments(parser, include_concurrent: bool = True, include_timeout: bool = True, 
                        include_similarity: bool = True, include_high_model: bool = False,
                        include_embedding_model: bool = False) -> None:
    """
    Add common CLI arguments to an ArgumentParser.
    
    Args:
        parser: ArgumentParser instance to add arguments to
        include_concurrent: Whether to include --max-concurrent argument
        include_timeout: Whether to include --timeout argument  
        include_similarity: Whether to include --similarity-threshold argument
        include_high_model: Whether to include --model-high argument
        include_embedding_model: Whether to include --embedding-model argument (for RAG)
    """
    
    # Regular model argument (combined format)
    parser.add_argument(
        "--model",
        default=get_model_from_env(),
        help=f"Regular model configuration as 'model,reasoning_effort,text_verbosity' (e.g., 'gpt-5-mini,medium,medium'). Default: {config.DEFAULT_MODEL},{config.DEFAULT_REASONING_EFFORT},{config.DEFAULT_TEXT_VERBOSITY}"
    )
    
    # High-end model argument (optional)
    if include_high_model:
        parser.add_argument(
            "--model-high",
            default=os.getenv("OPENAI_MODEL_HIGH", config.DEFAULT_MODEL_HIGH),
            help=f"High-end model configuration as 'model,reasoning_effort,text_verbosity' (e.g., 'gpt-5,high,high'). Default: {config.DEFAULT_MODEL_HIGH},{config.DEFAULT_REASONING_EFFORT_HIGH},{config.DEFAULT_TEXT_VERBOSITY_HIGH}"
        )
    
    # Backward compatibility: individual arguments (optional)
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high"],
        default=get_reasoning_effort_from_env(),
        help=f"Reasoning effort level (overrides --model if specified). Default: {config.DEFAULT_REASONING_EFFORT}"
    )
    
    parser.add_argument(
        "--text-verbosity",
        choices=["low", "medium", "high"],
        default=get_text_verbosity_from_env(),
        help=f"Text verbosity level (overrides --model if specified). Default: {config.DEFAULT_TEXT_VERBOSITY}"
    )
    
    # Optional arguments based on script needs
    if include_concurrent:
        parser.add_argument(
            "--max-concurrent",
            type=int,
            default=get_max_concurrent_from_env(),
            help=f"Maximum number of items to process concurrently (default: {config.DEFAULT_MAX_CONCURRENT})"
        )
    
    if include_timeout:
        parser.add_argument(
            "--timeout",
            type=int,
            default=get_timeout_from_env(),
            help=f"API timeout in seconds (default: {config.DEFAULT_TIMEOUT})"
        )
    
    if include_similarity:
        parser.add_argument(
            "--similarity-threshold",
            type=float,
            default=get_similarity_threshold_from_env(),
            help=f"Similarity threshold for deduplication (default: {config.DEFAULT_SIMILARITY_THRESHOLD})"
        )
    
    if include_embedding_model:
        parser.add_argument(
            "--embedding-model",
            choices=config.SUPPORTED_EMBEDDING_MODELS,
            default=get_embedding_model_from_env(),
            help=f"Embedding model for RAG (default: {config.DEFAULT_EMBEDDING_MODEL})"
        )


def parse_model_from_args(args) -> tuple[str, str, str]:
    """
    Parse model configuration from command line arguments.
    
    Args:
        args: Parsed arguments containing model configuration
        
    Returns:
        Tuple of (model, reasoning_effort, text_verbosity)
    """
    try:
        # Try to parse as combined format first
        return parse_model_config(args.model)
    except ValueError as e:
        # Fall back to individual arguments if combined format fails
        # In fallback mode, args.model should be a single model name, not combined format
        model = args.model if args.model in config.SUPPORTED_MODELS else config.DEFAULT_MODEL
        reasoning_effort = args.reasoning_effort
        text_verbosity = args.text_verbosity
        return model, reasoning_effort, text_verbosity


def parse_model_high_from_args(args) -> tuple[str, str, str]:
    """
    Parse high-end model configuration from command line arguments.
    
    Args:
        args: Parsed arguments containing model configuration
        
    Returns:
        Tuple of (model, reasoning_effort, text_verbosity)
    """
    if not hasattr(args, 'model_high'):
        raise ValueError("--model-high argument not available. Use include_high_model=True in add_common_arguments()")
    
    try:
        # Try to parse as combined format first
        return parse_model_config(args.model_high)
    except ValueError:
        # Fall back to individual arguments if combined format fails
        model = args.model_high if args.model_high in config.SUPPORTED_MODELS else config.DEFAULT_MODEL_HIGH
        reasoning_effort = getattr(args, 'reasoning_effort_high', config.DEFAULT_REASONING_EFFORT_HIGH)
        text_verbosity = getattr(args, 'text_verbosity_high', config.DEFAULT_TEXT_VERBOSITY_HIGH)
        return model, reasoning_effort, text_verbosity


def get_metadata_dict(model: str = None, reasoning_effort: str = None, 
                     text_verbosity: str = None, timeout: int = None,
                     max_concurrent: int = None, similarity_threshold: float = None,
                     **additional_params) -> Dict[str, Any]:
    """
    Create a standardized metadata dictionary for YAML outputs.
    
    Args:
        model: OpenAI model used
        reasoning_effort: Reasoning effort level
        text_verbosity: Text verbosity level
        timeout: API timeout in seconds
        max_concurrent: Max concurrent processing
        similarity_threshold: Similarity threshold for deduplication
        **additional_params: Any additional parameters to include
        
    Returns:
        Dictionary with standardized metadata structure
    """
    from datetime import datetime
    
    metadata = {
        "generated_at": datetime.now().isoformat(),
        "model_used": model or get_model_from_env(),
        "reasoning_effort": reasoning_effort or get_reasoning_effort_from_env(),
        "text_verbosity": text_verbosity or get_text_verbosity_from_env(),
    }
    
    # Add optional parameters if provided
    if timeout is not None:
        metadata["timeout_seconds"] = timeout
    if max_concurrent is not None:
        metadata["max_concurrent"] = max_concurrent
    if similarity_threshold is not None:
        metadata["similarity_threshold"] = similarity_threshold
    
    # Add any additional parameters
    metadata.update(additional_params)
    
    return metadata


# Convenience functions for backward compatibility
DEFAULT_MODEL = config.DEFAULT_MODEL
DEFAULT_REASONING_EFFORT = config.DEFAULT_REASONING_EFFORT
DEFAULT_TEXT_VERBOSITY = config.DEFAULT_TEXT_VERBOSITY
DEFAULT_MODEL_HIGH = config.DEFAULT_MODEL_HIGH
DEFAULT_REASONING_EFFORT_HIGH = config.DEFAULT_REASONING_EFFORT_HIGH
DEFAULT_TEXT_VERBOSITY_HIGH = config.DEFAULT_TEXT_VERBOSITY_HIGH
DEFAULT_EMBEDDING_MODEL = config.DEFAULT_EMBEDDING_MODEL
EMBEDDING_BATCH_SIZE = config.EMBEDDING_BATCH_SIZE
EMBEDDING_DELAY_SECONDS = config.EMBEDDING_DELAY_SECONDS
MAX_NODES_PER_BATCH = config.MAX_NODES_PER_BATCH
DEFAULT_MAX_CONCURRENT = config.DEFAULT_MAX_CONCURRENT
DEFAULT_TIMEOUT = config.DEFAULT_TIMEOUT
DEFAULT_SIMILARITY_THRESHOLD = config.DEFAULT_SIMILARITY_THRESHOLD
DEFAULT_CHUNK_SIZE = config.DEFAULT_CHUNK_SIZE
DEFAULT_CHUNK_OVERLAP = config.DEFAULT_CHUNK_OVERLAP
DEFAULT_TOP_K = config.DEFAULT_TOP_K
USE_COHERE_RERANK = config.USE_COHERE_RERANK
COHERE_RERANK_MODEL = config.COHERE_RERANK_MODEL
COHERE_RERANK_TOP_N = config.COHERE_RERANK_TOP_N
MAX_TOOL_ITERATIONS = config.MAX_TOOL_ITERATIONS
SUPPORTED_MODELS = config.SUPPORTED_MODELS
SUPPORTED_EMBEDDING_MODELS = config.SUPPORTED_EMBEDDING_MODELS

# Tavily configuration exports
DEFAULT_TAVILY_MAX_RESULTS = config.DEFAULT_TAVILY_MAX_RESULTS
DEFAULT_TAVILY_SEARCH_DEPTH = config.DEFAULT_TAVILY_SEARCH_DEPTH
DEFAULT_TAVILY_INCLUDE_ANSWER = config.DEFAULT_TAVILY_INCLUDE_ANSWER
DEFAULT_TAVILY_INCLUDE_RAW_CONTENT = config.DEFAULT_TAVILY_INCLUDE_RAW_CONTENT
DEFAULT_TAVILY_MAX_CONTENT_LENGTH = config.DEFAULT_TAVILY_MAX_CONTENT_LENGTH

# LlamaParse configuration exports
USE_LLAMA_PARSE = config.USE_LLAMA_PARSE
LLAMA_PARSE_RESULT_TYPE = config.LLAMA_PARSE_RESULT_TYPE
LLAMA_PARSE_VERBOSE = config.LLAMA_PARSE_VERBOSE
LLAMA_PARSE_LANGUAGE = config.LLAMA_PARSE_LANGUAGE
LLAMA_PARSE_PARSE_MODE = config.LLAMA_PARSE_PARSE_MODE
LLAMA_PARSE_INVALIDATE_CACHE = config.LLAMA_PARSE_INVALIDATE_CACHE
LLAMA_PARSE_DO_NOT_CACHE = config.LLAMA_PARSE_DO_NOT_CACHE
LLAMA_PARSE_NUM_WORKERS = config.LLAMA_PARSE_NUM_WORKERS
LLAMA_PARSE_SKIP_DIAGONAL_TEXT = config.LLAMA_PARSE_SKIP_DIAGONAL_TEXT
LLAMA_PARSE_SPREADSHEET_EXTRACT_SUB_TABLES = config.LLAMA_PARSE_SPREADSHEET_EXTRACT_SUB_TABLES
LLAMA_PARSE_SPREADSHEET_FORCE_FORMULA_COMPUTATION = config.LLAMA_PARSE_SPREADSHEET_FORCE_FORMULA_COMPUTATION
LLAMA_PARSE_PARTITION_PAGES = config.LLAMA_PARSE_PARTITION_PAGES
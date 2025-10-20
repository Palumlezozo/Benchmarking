#!/usr/bin/env python3
"""
Utility functions for LLM v2 scripts.
"""

import re


def clean_text(text: str) -> str:
    """
    Clean citation artifacts, LLM artifacts, and sanitize text for safe output in various formats (YAML, Excel, etc.).
    
    NOTE: This function replaces newlines with spaces - use clean_markdown_text() for markdown content.
    
    Args:
        text: Input text to clean and sanitize
        
    Returns:
        Cleaned and sanitized text safe for output
    """
    if not text:
        return text
    
    text = str(text)
    
    # Remove LLM citation artifacts using regex patterns
    # Remove specific patterns like "cite turn0search17" with special characters
    text = re.sub(r'[^\w\s.,!?;:()-]*cite[^\w\s.,!?;:()-]*turn\d+search\d*[^\w\s.,!?;:()-]*', '', text)
    
    # Remove patterns with spaces between cite and turn
    text = re.sub(r'\s+cite\s+turn\d+search\d*', '', text)
    
    # Remove turn patterns with any number (turn0search, turn1file, etc.)
    text = re.sub(r'turn\d+search\d*', '', text)
    text = re.sub(r'turn\d+file\d*', '', text)
    
    # Remove citation markers that are clearly artifacts (with special characters or numbers)
    text = re.sub(r'[^\w\s.,!?;:()-]*turn\d+[^\w\s.,!?;:()-]*', '', text)
    text = re.sub(r'[^\w\s.,!?;:()-]*cite\d+[^\w\s.,!?;:()-]*', '', text)
    
    # Remove standalone citation artifacts (but preserve normal use of "cite")
    text = re.sub(r'\bcite\d+\b', '', text)  # Remove "cite1", "cite2", etc.
    text = re.sub(r'[^\w\s.,!?;:()-]*cite[^\w\s.,!?;:()-]*', '', text)  # Remove cite with special chars
    text = re.sub(r'[^\w\s.,!?;:()-]+\d+[^\w\s.,!?;:()-]*', '', text)  # Remove patterns like "1 3 7"
    text = re.sub(r'\s+\d+\s*$', '', text)  # Remove trailing single digits like " 3"
    text = re.sub(r'[^\w\s.,!?;:()-]+$', '', text)  # Remove trailing non-word characters
    text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single space
    text = text.strip()
    text = re.sub(r'[^\w\s.,!?;:()-]+$', '', text)  # Remove trailing punctuation artifacts
    
    # Additional sanitization for safe output in various formats (YAML, Excel, etc.)
    # Replace problematic characters that might cause issues in various output formats
    text = text.replace('\x19', "'")  # Replace problematic apostrophe
    text = text.replace('\n', ' ')    # Replace newlines with spaces
    text = text.replace('\r', ' ')    # Replace carriage returns with spaces
    text = text.replace('\t', ' ')    # Replace tabs with spaces
    
    # Remove any remaining control characters (except common ones)
    text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')
    
    # Normalize multiple spaces to single space
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def clean_markdown_text(text: str) -> str:
    """
    Clean citation artifacts from markdown text while preserving markdown structure (newlines, tables, etc.).
    
    Use this for markdown content that needs to be rendered properly.
    
    Args:
        text: Input markdown text to clean
        
    Returns:
        Cleaned markdown text with structure preserved
    """
    if not text:
        return text
    
    text = str(text)
    
    # Remove LLM citation artifacts using regex patterns
    # Remove specific patterns like "cite turn0search17" with special characters
    text = re.sub(r'[^\w\s.,!?;:()-]*cite[^\w\s.,!?;:()-]*turn\d+search\d*[^\w\s.,!?;:()-]*', '', text)
    
    # Remove patterns with spaces between cite and turn
    text = re.sub(r'\s+cite\s+turn\d+search\d*', '', text)
    
    # Remove turn patterns with any number (turn0search, turn1file, etc.)
    text = re.sub(r'turn\d+search\d*', '', text)
    text = re.sub(r'turn\d+file\d*', '', text)
    
    # Remove citation markers that are clearly artifacts (with special characters or numbers)
    text = re.sub(r'[^\w\s.,!?;:()-]*turn\d+[^\w\s.,!?;:()-]*', '', text)
    text = re.sub(r'[^\w\s.,!?;:()-]*cite\d+[^\w\s.,!?;:()-]*', '', text)
    
    # Remove standalone citation artifacts (but preserve normal use of "cite")
    text = re.sub(r'\bcite\d+\b', '', text)  # Remove "cite1", "cite2", etc.
    text = re.sub(r'[^\w\s.,!?;:()-]*cite[^\w\s.,!?;:()-]*', '', text)  # Remove cite with special chars
    
    # Remove patterns like "1 3 7" but be careful not to break table cells
    # Only remove if they're surrounded by whitespace (not in tables)
    text = re.sub(r'(?<!\|)\s+[^\w\s.,!?;:()|\-]+\d+[^\w\s.,!?;:()|\-]*(?!\|)', ' ', text)
    
    # Remove trailing single digits like " 3" at end of lines (not in tables)
    text = re.sub(r'(?<!\|)\s+\d+\s*$', '', text, flags=re.MULTILINE)
    
    # Replace problematic control characters
    text = text.replace('\x19', "'")  # Replace problematic apostrophe
    text = text.replace('\r\n', '\n')  # Normalize line endings
    text = text.replace('\r', '\n')    # Normalize line endings
    
    # Remove control characters except newlines and tabs (which are needed for markdown)
    text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
    
    # Clean up excessive blank lines (more than 2 consecutive)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Clean up trailing whitespace on each line
    text = '\n'.join(line.rstrip() for line in text.split('\n'))
    
    # Strip leading/trailing whitespace from the entire text
    text = text.strip()
    
    return text


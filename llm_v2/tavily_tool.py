"""
Tavily Web Search Tool

This module provides a web search tool using the Tavily API (async).
"""

import asyncio
import logging
import os
from typing import Dict, Any, Optional
from tavily import TavilyClient

from tool_base import Tool
from config import (
    DEFAULT_TAVILY_MAX_RESULTS,
    DEFAULT_TAVILY_SEARCH_DEPTH,
    DEFAULT_TAVILY_INCLUDE_ANSWER,
    DEFAULT_TAVILY_INCLUDE_RAW_CONTENT,
    DEFAULT_TAVILY_MAX_CONTENT_LENGTH
)

# Set up logger
logger = logging.getLogger(__name__)


class TavilySearchTool(Tool):
    """Tool for searching the web using Tavily API."""
    
    def __init__(
        self, 
        max_results: int = DEFAULT_TAVILY_MAX_RESULTS,
        include_answer: bool = DEFAULT_TAVILY_INCLUDE_ANSWER,
        include_raw_content: bool = DEFAULT_TAVILY_INCLUDE_RAW_CONTENT,
        max_content_length: int = DEFAULT_TAVILY_MAX_CONTENT_LENGTH
    ):
        """
        Initialize the Tavily search tool.
        
        Args:
            max_results: Maximum number of search results to return
            include_answer: Whether to include AI-generated answer
            include_raw_content: Whether to include raw page content
            max_content_length: Maximum length of content snippets
        """
        self.max_results = max_results
        self.include_answer = include_answer
        self.include_raw_content = include_raw_content
        self.max_content_length = max_content_length
        
        api_key = os.getenv("TAVILY_API_KEY")
        
        if not api_key:
            raise ValueError("TAVILY_API_KEY not found in environment variables")
        
        self.client = TavilyClient(api_key=api_key)
        print(f"🔍 TavilySearchTool initialized (max_results={max_results})")
    
    def name(self) -> str:
        """Return the tool's name."""
        return "web_search"
    
    def description(self) -> str:
        """Return a description of what the tool does."""
        return (
            "Search the web for current information, news, facts, and data. "
            "Use this tool when you need up-to-date information, current events, "
            "recent news, or information not available in the document collection. "
            "Returns relevant web pages with titles, URLs, and content snippets."
        )
    
    def parameters_schema(self) -> Dict[str, Any]:
        """Return the JSON schema for the tool's parameters."""
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to look up on the web. Be specific and include relevant keywords."
                }
            },
            "required": ["query"]
        }
    
    async def execute(self, query: str, **kwargs) -> str:
        """
        Execute a web search using Tavily (async).
        
        Args:
            query: The search query
            **kwargs: Additional parameters (ignored)
            
        Returns:
            str: Formatted search results with titles, URLs, and content
        """
        try:
            logger.info(f"🌐 Web Search Tool: Searching for: '{query[:100]}...'")
            logger.info(f"   Max results: {self.max_results}, Include answer: {self.include_answer}")
            
            # Perform search in thread pool (Tavily client is sync)
            response = await asyncio.to_thread(
                self.client.search,
                query=query,
                max_results=self.max_results,
                include_answer=self.include_answer,
                include_raw_content=self.include_raw_content
            )
            
            # Format results
            results = []
            
            # Include Tavily's AI-generated answer if available
            if response.get("answer"):
                results.append(f"**Quick Answer:**\n{response['answer']}\n")
            
            # Format search results
            if response.get("results"):
                results.append("**Search Results:**\n")
                for i, result in enumerate(response["results"], 1):
                    title = result.get("title", "Untitled")
                    url = result.get("url", "")
                    content = result.get("content", "")
                    
                    # Format as markdown with sources
                    result_text = f"{i}. **{title}**\n"
                    result_text += f"   URL: {url}\n"
                    if content:
                        # Truncate content if too long
                        if len(content) > self.max_content_length:
                            content = content[:self.max_content_length] + "..."
                        result_text += f"   Content: {content}\n"
                    
                    results.append(result_text)
            
            if not results:
                logger.info(f"⚠️  Web Search Tool: No results found")
                return f"No results found for query: {query}"
            
            result_count = len(response.get("results", []))
            logger.info(f"✅ Web Search Tool: Found {result_count} web results")
            
            return "\n".join(results)
            
        except Exception as e:
            logger.error(f"❌ Web Search Tool: Error: {str(e)}")
            return f"Error performing web search: {str(e)}"


class TavilyNewsSearchTool(Tool):
    """Tool for searching recent news using Tavily API."""
    
    def __init__(
        self,
        max_results: int = DEFAULT_TAVILY_MAX_RESULTS,
        include_answer: bool = DEFAULT_TAVILY_INCLUDE_ANSWER,
        include_raw_content: bool = DEFAULT_TAVILY_INCLUDE_RAW_CONTENT,
        max_content_length: int = DEFAULT_TAVILY_MAX_CONTENT_LENGTH,
        search_depth: str = DEFAULT_TAVILY_SEARCH_DEPTH
    ):
        """
        Initialize the Tavily news search tool.
        
        Args:
            max_results: Maximum number of news results to return
            include_answer: Whether to include AI-generated summary
            include_raw_content: Whether to include raw page content
            max_content_length: Maximum length of content snippets
            search_depth: Search depth ("basic" or "advanced")
        """
        self.max_results = max_results
        self.include_answer = include_answer
        self.include_raw_content = include_raw_content
        self.max_content_length = max_content_length
        self.search_depth = search_depth
        
        api_key = os.getenv("TAVILY_API_KEY")
        
        if not api_key:
            raise ValueError("TAVILY_API_KEY not found in environment variables")
        
        self.client = TavilyClient(api_key=api_key)
        print(f"📰 TavilyNewsSearchTool initialized (max_results={max_results})")
    
    def name(self) -> str:
        """Return the tool's name."""
        return "news_search"
    
    def description(self) -> str:
        """Return a description of what the tool does."""
        return (
            "Search for recent news articles and current events. "
            "Use this tool specifically when you need the latest news, "
            "recent developments, or breaking news about a topic. "
            "Returns recent news articles with titles, URLs, and summaries."
        )
    
    def parameters_schema(self) -> Dict[str, Any]:
        """Return the JSON schema for the tool's parameters."""
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The news search query. Include topic, company, event, or keywords for news you want to find."
                }
            },
            "required": ["query"]
        }
    
    async def execute(self, query: str, **kwargs) -> str:
        """
        Execute a news search using Tavily (async).
        
        Args:
            query: The search query
            **kwargs: Additional parameters (ignored)
            
        Returns:
            str: Formatted news results with titles, URLs, and content
        """
        try:
            logger.info(f"📰 News Search Tool: Searching for: '{query[:100]}...'")
            logger.info(f"   Max results: {self.max_results}, Search depth: {self.search_depth}")
            
            # Perform news search in thread pool (Tavily client is sync)
            response = await asyncio.to_thread(
                self.client.search,
                query=query,
                max_results=self.max_results,
                search_depth=self.search_depth,
                include_answer=self.include_answer,
                include_raw_content=self.include_raw_content,
                topic="news"  # Focus on news content
            )
            
            # Format results
            results = []
            
            # Include Tavily's AI-generated summary if available
            if response.get("answer"):
                results.append(f"**News Summary:**\n{response['answer']}\n")
            
            # Format news results
            if response.get("results"):
                results.append("**Recent News Articles:**\n")
                for i, result in enumerate(response["results"], 1):
                    title = result.get("title", "Untitled")
                    url = result.get("url", "")
                    content = result.get("content", "")
                    
                    # Format as markdown with sources
                    result_text = f"{i}. **{title}**\n"
                    result_text += f"   URL: {url}\n"
                    if content:
                        # Truncate content if too long
                        if len(content) > self.max_content_length:
                            content = content[:self.max_content_length] + "..."
                        result_text += f"   Content: {content}\n"
                    
                    results.append(result_text)
            
            if not results:
                logger.info(f"⚠️  News Search Tool: No news articles found")
                return f"No news articles found for query: {query}"
            
            result_count = len(response.get("results", []))
            logger.info(f"✅ News Search Tool: Found {result_count} news articles")
            
            return "\n".join(results)
            
        except Exception as e:
            logger.error(f"❌ News Search Tool: Error: {str(e)}")
            return f"Error performing news search: {str(e)}"


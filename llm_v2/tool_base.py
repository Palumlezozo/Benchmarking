"""
Base classes for tool-calling framework.

This module provides the abstract base class for tools that can be used
by the LLM in a tool-calling loop (ReAct pattern).
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class Tool(ABC):
    """Abstract base class for tools that can be called by the LLM."""
    
    @abstractmethod
    def name(self) -> str:
        """Return the tool's name (used by LLM to identify the tool)."""
        pass
    
    @abstractmethod
    def description(self) -> str:
        """Return a description of what the tool does (used by LLM to decide when to use it)."""
        pass
    
    @abstractmethod
    def parameters_schema(self) -> Dict[str, Any]:
        """
        Return the JSON schema for the tool's parameters.
        
        This follows OpenAI's function calling format:
        {
            "type": "object",
            "properties": {
                "param_name": {
                    "type": "string",
                    "description": "Parameter description"
                }
            },
            "required": ["param_name"]
        }
        """
        pass
    
    @abstractmethod
    async def execute(self, **kwargs) -> str:
        """
        Execute the tool with the given parameters (async).
        
        Args:
            **kwargs: Tool-specific parameters
            
        Returns:
            str: The result of executing the tool
        """
        pass
    
    def to_openai_function(self) -> Dict[str, Any]:
        """
        Convert the tool to OpenAI function calling format.
        
        Returns:
            Dict suitable for OpenAI's functions parameter
        """
        return {
            "type": "function",
            "function": {
                "name": self.name(),
                "description": self.description(),
                "parameters": self.parameters_schema()
            }
        }


class ToolResult:
    """Result of a tool execution."""
    
    def __init__(self, tool_name: str, success: bool, result: str, error: Optional[str] = None):
        """
        Initialize a tool result.
        
        Args:
            tool_name: Name of the tool that was executed
            success: Whether the tool executed successfully
            result: The result string from the tool
            error: Optional error message if execution failed
        """
        self.tool_name = tool_name
        self.success = success
        self.result = result
        self.error = error
    
    def __str__(self) -> str:
        if self.success:
            return f"[{self.tool_name}] {self.result}"
        else:
            return f"[{self.tool_name}] ERROR: {self.error}"


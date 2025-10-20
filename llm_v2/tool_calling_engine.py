"""
Tool-Calling Query Engine.

This module implements a query engine that uses OpenAI's responses API
to dynamically select and execute tools, then generates structured responses.

Architecture (100% responses API):
1. User query → responses.parse → LLM outputs ToolDecision (which tool to call)
2. Execute tool(s) and collect results
3. Feed results back → responses.parse → Final ChatResponse
"""

import asyncio
import json
import os
from typing import List, Dict, Any, Optional, Union
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from tool_base import Tool, ToolResult
from config import DEFAULT_MODEL, DEFAULT_REASONING_EFFORT, DEFAULT_TEXT_VERBOSITY


class ToolCall(BaseModel):
    """A single tool call decision."""
    tool_name: str = Field(description="Name of the tool to call")
    parameters: str = Field(description="JSON-encoded string of parameters to pass to the tool")


class ToolDecision(BaseModel):
    """Decision about which tool(s) to call, if any."""
    should_call_tool: bool = Field(description="Whether a tool should be called")
    tool_calls: List[ToolCall] = Field(default_factory=list, description="List of tools to call")
    reasoning: str = Field(description="Brief explanation of why this tool decision was made")


class ToolCallingEngine:
    """Query engine that uses tool calling to answer questions."""
    
    def __init__(
        self,
        tools: List[Tool],
        model: str = DEFAULT_MODEL,
        reasoning_effort: str = DEFAULT_REASONING_EFFORT,
        text_verbosity: str = DEFAULT_TEXT_VERBOSITY,
        max_tool_iterations: int = 3
    ):
        """
        Initialize the tool-calling engine.
        
        Args:
            tools: List of available tools
            model: OpenAI model to use
            reasoning_effort: Reasoning effort level
            text_verbosity: Text verbosity level
            max_tool_iterations: Maximum number of tool-calling iterations
        """
        self.tools = {tool.name(): tool for tool in tools}
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.text_verbosity = text_verbosity
        self.max_tool_iterations = max_tool_iterations
        
        # Use Azure OpenAI if configured, otherwise use standard OpenAI
        azure_base_url = os.getenv("AZURE_OPENAI_BASE_URL")
        azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        
        if azure_base_url and azure_api_key:
            self.client = AsyncOpenAI(
                api_key=azure_api_key,
                base_url=azure_base_url
            )
        else:
            self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    async def query(
        self,
        question: str,
        conversation_history: List[Dict[str, str]] = None,
        response_format: Optional[BaseModel] = None,
        system_prompt: Optional[str] = None
    ) -> Any:
        """
        Execute a query with tool calling (async).
        
        Args:
            question: The user's question
            conversation_history: Previous conversation messages
            response_format: Optional Pydantic model for structured output
            system_prompt: Optional custom system prompt
            
        Returns:
            Parsed response (Pydantic model if response_format provided, else dict)
        """
        # Build messages
        messages = self._build_messages(question, conversation_history, system_prompt)
        
        # Step 1-2: Tool calling loop
        final_messages, tool_results = await self._execute_tool_loop(messages)
        
        # Step 3: Generate final structured response
        if response_format:
            return await self._generate_structured_response(final_messages, response_format)
        else:
            return await self._generate_text_response(final_messages)
    
    def _build_messages(
        self,
        question: str,
        conversation_history: Optional[List[Dict[str, str]]],
        system_prompt: Optional[str]
    ) -> List[Dict[str, str]]:
        """Build the message list for the LLM."""
        messages = []
        
        # Add system prompt
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        else:
            messages.append({
                "role": "system",
                "content": "You are a helpful assistant with access to tools. Use the available tools when needed to answer questions accurately."
            })
        
        # Add conversation history
        if conversation_history:
            messages.extend(conversation_history)
        
        # Add current question
        messages.append({"role": "user", "content": question})
        
        return messages
    
    async def _execute_tool_loop(
        self,
        messages: List[Dict[str, str]]
    ) -> tuple[List[Dict[str, str]], List[ToolResult]]:
        """
        Execute the tool-calling loop using responses API (async).
        
        Args:
            messages: Initial messages
            
        Returns:
            Tuple of (final_messages, tool_results)
        """
        current_messages = messages.copy()
        all_tool_results = []
        iteration = 0
        
        while iteration < self.max_tool_iterations:
            iteration += 1
            
            # Build tool descriptions for the prompt
            tool_descriptions = self._format_tools_for_prompt()
            
            # Create system prompt with tool information
            decision_prompt = self._build_tool_decision_prompt(tool_descriptions)
            
            # Ask LLM to decide which tool(s) to call using responses API
            tool_messages = [
                {"role": "system", "content": decision_prompt},
                *current_messages
            ]
            
            # Use responses API to get structured tool decision (async)
            response = await self.client.responses.parse(
                model=self.model,
                input=tool_messages,
                text_format=ToolDecision,
                reasoning={"effort": self.reasoning_effort},
                text={"verbosity": self.text_verbosity}
            )
            
            # Extract tool decision
            tool_decision = self._extract_tool_decision(response)
            
            if not tool_decision or not tool_decision.should_call_tool or not tool_decision.tool_calls:
                # No tools to call, we're done
                break
            
            # Execute tool calls IN PARALLEL! ⚡
            tool_tasks = []
            for tool_call in tool_decision.tool_calls:
                # Parse parameters from JSON string
                try:
                    params = json.loads(tool_call.parameters)
                except json.JSONDecodeError:
                    params = {}
                
                # Create task for parallel execution
                tool_tasks.append(self._execute_tool(tool_call.tool_name, params))
            
            # Execute all tools concurrently
            tool_results = await asyncio.gather(*tool_tasks)
            
            # Process results
            tool_results_text = []
            for tool_result in tool_results:
                all_tool_results.append(tool_result)
                tool_results_text.append(f"**Tool: {tool_result.tool_name}**\n{tool_result.result}")
            
            # Add tool results as a user message
            combined_results = "\n\n---\n\n".join(tool_results_text)
            current_messages.append({
                "role": "user",
                "content": f"Tool Results:\n\n{combined_results}"
            })
        
        return current_messages, all_tool_results
    
    def _format_tools_for_prompt(self) -> str:
        """Format available tools as a string for the prompt."""
        tool_descriptions = []
        for tool in self.tools.values():
            params_schema = tool.parameters_schema()
            params_desc = []
            if "properties" in params_schema:
                for param_name, param_info in params_schema["properties"].items():
                    required = param_name in params_schema.get("required", [])
                    param_desc = f"  - {param_name} ({param_info.get('type', 'any')}){' [REQUIRED]' if required else ''}: {param_info.get('description', '')}"
                    params_desc.append(param_desc)
            
            tool_desc = f"""
**{tool.name()}**
Description: {tool.description()}
Parameters:
{chr(10).join(params_desc) if params_desc else '  (no parameters)'}
"""
            tool_descriptions.append(tool_desc)
        
        return "\n".join(tool_descriptions)
    
    def _build_tool_decision_prompt(self, tool_descriptions: str) -> str:
        """Build the system prompt for tool decision."""
        return f"""You are a tool-using assistant. You have access to the following tools:

{tool_descriptions}

Your task is to decide whether you need to call any tools to answer the user's question.

Analyze the user's question and:
1. Determine if you need to use a tool
2. If yes, specify which tool(s) to call and with what parameters (as JSON string)
3. If no, set should_call_tool to false

IMPORTANT: Format the parameters field as a JSON string, e.g., '{{"query": "search term"}}'

Provide your decision in the structured format."""
    
    def _extract_tool_decision(self, response) -> Optional[ToolDecision]:
        """Extract ToolDecision from responses API response - get the LAST decision after reasoning."""
        try:
            if hasattr(response, 'output') and response.output:
                # Collect all tool decisions (don't break on first one)
                all_decisions = []
                
                for item in response.output:
                    if hasattr(item, 'type') and item.type == 'message':
                        if hasattr(item, 'content') and item.content:
                            if isinstance(item.content, list):
                                for content_item in item.content:
                                    if hasattr(content_item, 'type') and content_item.type == 'output_text':
                                        if hasattr(content_item, 'text_parsed'):
                                            all_decisions.append(content_item.text_parsed)
                                            break
                                        elif hasattr(content_item, 'parsed'):
                                            all_decisions.append(content_item.parsed)
                                            break
                            elif isinstance(item.content, ToolDecision):
                                all_decisions.append(item.content)
                
                # Return the LAST decision (final one after reasoning)
                if all_decisions:
                    return all_decisions[-1]
        except Exception as e:
            print(f"Error extracting tool decision: {e}")
        
        return None
    
    async def _execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> ToolResult:
        """
        Execute a single tool (async).
        
        Args:
            tool_name: Name of the tool to execute
            tool_args: Arguments for the tool
            
        Returns:
            ToolResult with execution results
        """
        try:
            if tool_name not in self.tools:
                return ToolResult(
                    tool_name=tool_name,
                    success=False,
                    result="",
                    error=f"Tool '{tool_name}' not found"
                )
            
            tool = self.tools[tool_name]
            result = await tool.execute(**tool_args)
            
            return ToolResult(
                tool_name=tool_name,
                success=True,
                result=result,
                error=None
            )
            
        except Exception as e:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                result="",
                error=str(e)
            )
    
    async def _generate_structured_response(
        self,
        messages: List[Dict[str, str]],
        response_format: BaseModel
    ) -> BaseModel:
        """
        Generate final structured response using responses.parse (async).
        
        Args:
            messages: Conversation messages including tool results
            response_format: Pydantic model for response structure
            
        Returns:
            Parsed Pydantic model instance
        """
        # Use responses API for structured output
        # Messages are already in the correct format (no tool_calls or role='tool')
        response = await self.client.responses.parse(
            model=self.model,
            input=messages,
            text_format=response_format,
            reasoning={"effort": self.reasoning_effort},
            text={"verbosity": self.text_verbosity}
        )
        
        # Extract the parsed response - get the LAST response after reasoning
        parsed_response = None
        if hasattr(response, 'output') and response.output:
            # Collect all parsed responses (don't break on first one)
            all_responses = []
            
            for item in response.output:
                if hasattr(item, 'type') and item.type == 'message' and hasattr(item, 'role') and item.role == 'assistant':
                    if hasattr(item, 'content') and item.content:
                        if isinstance(item.content, list):
                            for content_item in item.content:
                                if hasattr(content_item, 'type') and content_item.type == 'output_text':
                                    if hasattr(content_item, 'text_parsed') and content_item.text_parsed:
                                        all_responses.append(content_item.text_parsed)
                                        break
                                    elif hasattr(content_item, 'parsed') and content_item.parsed:
                                        all_responses.append(content_item.parsed)
                                        break
                        elif isinstance(item.content, response_format):
                            all_responses.append(item.content)
            
            # Return the LAST response (final answer after reasoning)
            if all_responses:
                parsed_response = all_responses[-1]
        
        return parsed_response
    
    async def _generate_text_response(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Generate final text response (async).
        
        Args:
            messages: Conversation messages including tool results
            
        Returns:
            Response dictionary
        """
        response = await self.client.responses.parse(
            model=self.model,
            input=messages,
            reasoning={"effort": self.reasoning_effort},
            text={"verbosity": self.text_verbosity}
        )
        
        # Extract text from response
        text_content = ""
        if hasattr(response, 'output') and response.output:
            for item in response.output:
                if hasattr(item, 'type') and item.type == 'message':
                    if hasattr(item, 'content') and item.content:
                        if isinstance(item.content, list):
                            for content_item in item.content:
                                if hasattr(content_item, 'type') and content_item.type == 'output_text':
                                    if hasattr(content_item, 'text'):
                                        text_content = content_item.text
                                        break
        
        return {"content": text_content}


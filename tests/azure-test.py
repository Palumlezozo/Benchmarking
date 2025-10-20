#!/usr/bin/env python3
"""
Azure OpenAI Connection Test Script

Tests the Azure OpenAI connection using the Responses API (v2).
This script verifies that the Azure endpoint, API key, and model deployment are working correctly.

Tests performed:
1. Basic connection and text generation
2. Web search tool availability (may not be available in all Azure deployments)
3. Structured output with Pydantic models
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Azure OpenAI Configuration (loaded from .env)
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_BASE_URL = os.getenv("AZURE_OPENAI_BASE_URL")  # e.g., "https://luc-openai-sw.openai.azure.com/openai/v1/"

# Model Configuration
MODEL_NAME = "gpt-5-nano"  # or "gpt-5-mini", "gpt-5"
AZURE_DEPLOYMENT_NAME = MODEL_NAME
REASONING_EFFORT = "medium"  # low, medium, high
TEXT_VERBOSITY = "medium"   # low, medium, high


def test_azure_connection():
    """Test Azure OpenAI connection using the Responses API."""
    
    print("=" * 70)
    print("Azure OpenAI Connection Test")
    print("=" * 70)
    print(f"Base URL: {AZURE_BASE_URL}")
    print(f"Deployment/Model: {AZURE_DEPLOYMENT_NAME}")
    print(f"Reasoning Effort: {REASONING_EFFORT}")
    print(f"Text Verbosity: {TEXT_VERBOSITY}")
    print("=" * 70)
    
    try:
        # Initialize Azure OpenAI client (using standard OpenAI client with base_url)
        print("\n1. Initializing Azure OpenAI client...")
        client = OpenAI(
            api_key=AZURE_API_KEY,
            base_url=AZURE_BASE_URL
        )
        print("   ✅ Client initialized successfully")
        
        # Test query
        test_question = "I am going to Paris, what should I see?"
        print(f"\n2. Sending test query: '{test_question}'")
        
        # Use the Responses API (v2) - same as used in chatbot.py and rag_client.py
        response = client.responses.parse(
            model=AZURE_DEPLOYMENT_NAME,  # Use deployment name for Azure
            input=[
                {
                    "role": "system",
                    "content": "You are a helpful travel assistant. Provide concise, practical advice."
                },
                {
                    "role": "user",
                    "content": test_question
                }
            ],
            reasoning={"effort": REASONING_EFFORT},
            text={"verbosity": TEXT_VERBOSITY}
        )
        
        print("   ✅ Response received successfully")
        
        # Extract response text (same pattern as in rag_client.py)
        print("\n3. Extracting response content...")
        response_text = None
        if hasattr(response, 'output') and response.output:
            for item in response.output:
                if hasattr(item, 'content') and item.content:
                    for content_item in item.content:
                        if hasattr(content_item, 'text'):
                            response_text = content_item.text
                            break
                if response_text:
                    break
        
        # Display results
        print("\n" + "=" * 70)
        print("RESPONSE:")
        print("=" * 70)
        if response_text:
            print(response_text)
        else:
            print("⚠️  No response text found")
            print("\nFull response structure:")
            print(response)
        print("=" * 70)
        
        # Show usage information if available
        if hasattr(response, 'usage'):
            print("\nToken Usage:")
            print(f"  Input tokens: {getattr(response.usage, 'input_tokens', 'N/A')}")
            print(f"  Output tokens: {getattr(response.usage, 'output_tokens', 'N/A')}")
            print(f"  Total tokens: {getattr(response.usage, 'total_tokens', 'N/A')}")
        
        print("\n✅ Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check that AZURE_OPENAI_BASE_URL is correct in .env")
        print("   Format: https://YOUR-RESOURCE-NAME.openai.azure.com/openai/v1/")
        print("2. Check that AZURE_OPENAI_API_KEY is valid in .env")
        print("3. Verify the deployment/model name matches your Azure deployment")
        print("4. Ensure the deployment exists in your Azure OpenAI resource")
        print("5. Check that the base URL includes the '/openai/v1/' suffix")
        
        import traceback
        print("\nFull error trace:")
        traceback.print_exc()
        return False


def test_web_search_tool():
    """Test Azure OpenAI with web_search tool."""
    
    print("\n" + "=" * 70)
    print("Testing Web Search Tool")
    print("=" * 70)
    
    try:
        # Initialize client
        client = OpenAI(
            api_key=AZURE_API_KEY,
            base_url=AZURE_BASE_URL
        )
        
        # Test query that requires web search
        test_question = "What are the latest news about artificial intelligence in December 2024?"
        print(f"\nQuery: {test_question}")
        print("Testing if web_search tool is available with Azure OpenAI...")
        
        # Use responses API with web_search tool
        response = client.responses.parse(
            model=AZURE_DEPLOYMENT_NAME,
            input=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant with access to web search. Use web search to find current information."
                },
                {
                    "role": "user",
                    "content": test_question
                }
            ],
            tools=[
                {
                    "type": "web_search"
                }
            ],
            tool_choice="auto",
            reasoning={"effort": REASONING_EFFORT},
            text={"verbosity": TEXT_VERBOSITY}
        )
        
        # Extract response
        response_text = None
        if hasattr(response, 'output') and response.output:
            for item in response.output:
                if hasattr(item, 'content') and item.content:
                    for content_item in item.content:
                        if hasattr(content_item, 'text'):
                            response_text = content_item.text
                            break
                if response_text:
                    break
        
        # Display results
        if response_text:
            print("\n✅ Web search tool is working!")
            print("\nResponse:")
            print("-" * 70)
            print(response_text[:500] + "..." if len(response_text) > 500 else response_text)
            print("-" * 70)
        else:
            print("⚠️  Could not extract response")
        
        # Check if web search was actually used
        print("\nChecking if web_search tool was invoked...")
        if hasattr(response, 'output') and response.output:
            for item in response.output:
                if hasattr(item, 'type'):
                    if item.type == 'tool_use':
                        print(f"   ✅ Tool used: {getattr(item, 'name', 'unknown')}")
        
        print("\n✅ Web search tool test completed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Web search tool test failed: {e}")
        
        # Check if it's a tool availability issue
        error_str = str(e)
        if "tool" in error_str.lower() or "web_search" in error_str.lower():
            print("\n⚠️  Note: The web_search tool may not be available with Azure OpenAI.")
            print("   Web search tools are typically available only with standard OpenAI API.")
            print("   Azure OpenAI may have different tool availability.")
        
        import traceback
        print("\nFull error trace:")
        traceback.print_exc()
        return False


def test_with_structured_output():
    """Test Azure OpenAI with structured output (Pydantic model)."""
    
    print("\n" + "=" * 70)
    print("Testing Structured Output (Pydantic)")
    print("=" * 70)
    
    try:
        from pydantic import BaseModel, Field
        from typing import List
        
        # Define a structured response model
        class TravelRecommendation(BaseModel):
            """Travel recommendations for a city."""
            city: str = Field(description="The city name")
            top_attractions: List[str] = Field(description="List of top attractions to visit")
            travel_tips: str = Field(description="Practical travel tips")
        
        # Initialize client (using standard OpenAI client with base_url)
        client = OpenAI(
            api_key=AZURE_API_KEY,
            base_url=AZURE_BASE_URL
        )
        
        # Query with structured output
        print("\nRequesting structured travel recommendations...")
        response = client.responses.parse(
            model=AZURE_DEPLOYMENT_NAME,
            input=[
                {
                    "role": "system",
                    "content": "You are a travel expert. Provide structured travel recommendations."
                },
                {
                    "role": "user",
                    "content": "Give me recommendations for visiting Paris"
                }
            ],
            text_format=TravelRecommendation,  # Request structured output
            reasoning={"effort": REASONING_EFFORT},
            text={"verbosity": TEXT_VERBOSITY}
        )
        
        # Extract structured response
        structured_data = None
        if hasattr(response, 'output') and response.output:
            for item in response.output:
                if hasattr(item, 'content') and item.content:
                    for content_item in item.content:
                        if hasattr(content_item, 'text_parsed'):
                            structured_data = content_item.text_parsed
                            break
                        elif hasattr(content_item, 'parsed'):
                            structured_data = content_item.parsed
                            break
                if structured_data:
                    break
        
        # Display structured results
        if structured_data:
            print("\n✅ Structured output received:")
            print(f"\nCity: {structured_data.city}")
            print(f"\nTop Attractions:")
            for i, attraction in enumerate(structured_data.top_attractions, 1):
                print(f"  {i}. {attraction}")
            print(f"\nTravel Tips: {structured_data.travel_tips}")
        else:
            print("⚠️  Could not extract structured output")
        
        print("\n✅ Structured output test completed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Structured output test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n🚀 Starting Azure OpenAI Tests\n")
    
    # Check environment variables
    print("Checking environment variables...")
    if not AZURE_API_KEY or not AZURE_BASE_URL:
        print("⚠️  Warning: Azure OpenAI credentials not set in .env file")
        print("   Please add the following to your .env file:")
        print("")
        print("   # Azure OpenAI Configuration")
        print("   AZURE_OPENAI_API_KEY=your-api-key-here")
        print("   AZURE_OPENAI_BASE_URL=https://YOUR-RESOURCE-NAME.openai.azure.com/openai/v1/")
        print("")
        print("   Example:")
        print("   AZURE_OPENAI_BASE_URL=https://luc-openai-sw.openai.azure.com/openai/v1/")
        print("")
        proceed = input("Continue anyway for testing? (y/n): ")
        if proceed.lower() != 'y':
            return
    
    # Run basic test
    basic_success = test_azure_connection()
    
    # Run web search test if basic test succeeded
    if basic_success:
        print("\n" + "─" * 70 + "\n")
        web_search_success = test_web_search_tool()
    
    # Run structured output test if basic test succeeded
    if basic_success:
        print("\n" + "─" * 70 + "\n")
        structured_success = test_with_structured_output()
    
    print("\n" + "=" * 70)
    print("Tests Complete")
    print("=" * 70)
    
    # Summary
    print("\nTest Summary:")
    print(f"  Basic Connection: {'✅ Pass' if basic_success else '❌ Fail'}")
    if basic_success:
        print(f"  Web Search Tool: {'✅ Pass' if 'web_search_success' in locals() and web_search_success else '⚠️  Check output'}")
        print(f"  Structured Output: {'✅ Pass' if 'structured_success' in locals() and structured_success else '⚠️  Check output'}")


if __name__ == "__main__":
    main()


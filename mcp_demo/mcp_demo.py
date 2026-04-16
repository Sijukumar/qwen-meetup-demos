#!/usr/bin/env python3
"""
MCP (Model Context Protocol) Real Implementation Demo
Using Filesystem MCP Server with stdio transport

This script demonstrates REAL MCP protocol usage with Qwen models.
The MCP server runs as a subprocess and communicates via stdio.

Prerequisites:
    1. Install Node.js (v18+): https://nodejs.org/
    2. Install MCP Python SDK: pip install mcp

Environment Setup:
    export DASHSCOPE_API_KEY="sk-your-api-key"

Usage:
    python mcp_real_demo.py

What is MCP?
    - Model Context Protocol connects Qwen to external tools
    - Uses real stdio transport to communicate with MCP server
    - Dynamic tool discovery and execution
    - Model decides which tools to use
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from contextlib import AsyncExitStack

# Try to import MCP SDK
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
except ImportError:
    print("Error: MCP SDK not installed.")
    print("Please run: pip install mcp")
    sys.exit(1)

# Load environment variables
load_dotenv()

# Configuration
API_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-b809035c44e14c3ab3a976bd1fbdd77a")
BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
MODEL = "qwen3.5-plus"

# Demo folder for Filesystem MCP
DEMO_FOLDER = os.path.expanduser("~/mcp_demo_files")


class MCPClient:
    """MCP Client that manages connection to MCP server."""
    
    def __init__(self):
        self.session = None
        self.exit_stack = AsyncExitStack()
        self.tools = []
        
    async def connect_to_server(self, demo_folder: str):
        """
        Connect to the Filesystem MCP server.
        
        Args:
            demo_folder: Path to the folder accessible by MCP server
        """
        print(f"\n[Connecting to MCP server...]")
        print(f"  Folder: {demo_folder}")
        
        # Server parameters for stdio transport
        server_params = StdioServerParameters(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", demo_folder]
        )
        
        # Connect to server via stdio
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )
        
        # Initialize session
        await self.session.initialize()
        
        # List available tools
        response = await self.session.list_tools()
        self.tools = response.tools
        
        print(f"  ✓ Connected to MCP server")
        print(f"  ✓ Available tools: {len(self.tools)}")
        for tool in self.tools:
            print(f"    - {tool.name}: {tool.description}")
        
        return self.tools
    
    async def call_tool(self, tool_name: str, arguments: dict):
        """
        Call a tool on the MCP server.
        
        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments
            
        Returns:
            Tool execution result
        """
        print(f"\n[Calling MCP Tool]")
        print(f"  Tool: {tool_name}")
        print(f"  Arguments: {arguments}")
        
        result = await self.session.call_tool(tool_name, arguments=arguments)
        
        print(f"  ✓ Tool execution complete")
        return result
    
    async def cleanup(self):
        """Clean up resources."""
        await self.exit_stack.aclose()


def get_openai_client():
    """Initialize OpenAI client for Alibaba Cloud Model Studio."""
    if not API_KEY:
        print("Error: DASHSCOPE_API_KEY not found.")
        print("Please set it in the .env file or as an environment variable.")
        sys.exit(1)
    
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)


def setup_demo_files():
    """Create demo files for the Filesystem MCP server."""
    demo_folder = Path(DEMO_FOLDER)
    demo_folder.mkdir(exist_ok=True)
    
    files = {
        "project_alpha.txt": """Project Alpha - Internal Documentation

Launch Date: March 25, 2026
Budget: $50,000
Team Size: 5 developers

Key Risks:
1. Supply chain delays in Singapore
2. Integration with legacy systems
3. User adoption challenges

Status: On Track
Last Updated: March 19, 2026
""",
        "meeting_notes.txt": """Team Meeting - March 18, 2026

Attendees: Alice, Bob, Charlie, Diana

Agenda:
1. Review Q1 progress
2. Discuss MCP integration demo
3. Plan for Alibaba Cloud workshop

Action Items:
- Alice: Prepare demo script
- Bob: Test Filesystem MCP server
- Charlie: Create sample data files
- Diana: Schedule rehearsal

Next Meeting: March 20, 2026
""",
        "tech_stack.md": """# Technology Stack

## Backend
- Language: Python 3.13
- Framework: FastAPI
- Database: PostgreSQL
- Cache: Redis

## AI/ML
- Platform: Alibaba Cloud Model Studio
- Models: Qwen3.5-Plus, Qwen3-Omni
- Protocol: MCP (Model Context Protocol)

## Frontend
- Framework: React 18
- UI Library: Ant Design
- State: Zustand

## Infrastructure
- Cloud: Alibaba Cloud
- Region: Singapore
- Container: Docker + Kubernetes
"""
    }
    
    for filename, content in files.items():
        filepath = demo_folder / filename
        if not filepath.exists():
            filepath.write_text(content)
            print(f"  Created: {filepath}")
    
    return str(demo_folder)


def format_tools_for_llm(tools):
    """Format MCP tools for LLM function calling."""
    formatted_tools = []
    for tool in tools:
        formatted_tool = {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema
            }
        }
        formatted_tools.append(formatted_tool)
    return formatted_tools


async def chat_with_mcp(mcp_client: MCPClient, openai_client: OpenAI, user_query: str):
    """
    Chat with Qwen using MCP tools.
    
    Args:
        mcp_client: Connected MCP client
        openai_client: OpenAI client
        user_query: User's question
        
    Returns:
        AI response
    """
    print(f"\n[User Query]: {user_query}")
    
    # Format tools for LLM
    tools = format_tools_for_llm(mcp_client.tools)
    
    # Build system prompt
    system_prompt = """You are an AI assistant with access to a Filesystem MCP server.
You can read files, list directories, and search files to answer user questions.

When you need to access files:
1. Use the appropriate tool (read_file, list_directory, search_files, etc.)
2. Wait for the tool result
3. Provide a helpful answer based on the file contents

Always be helpful and reference specific files when answering."""
    
    # First LLM call - may include tool calls
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ]
    
    print("\n[Step 1] Sending query to LLM with available tools...")
    
    response = openai_client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools if tools else None,
        tool_choice="auto" if tools else None,
        temperature=0.7
    )
    
    message = response.choices[0].message
    
    # Check if model wants to use tools
    if message.tool_calls:
        print(f"  ✓ Model decided to use {len(message.tool_calls)} tool(s)")
        
        # Add assistant message with tool calls
        messages.append({
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                } for tc in message.tool_calls
            ]
        })
        
        # Execute each tool call
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            
            print(f"\n[Step 2] Executing tool: {tool_name}")
            print(f"  Arguments: {tool_args}")
            
            # Call MCP tool
            result = await mcp_client.call_tool(tool_name, tool_args)
            
            # Extract result content
            result_content = ""
            if result.content:
                for item in result.content:
                    if hasattr(item, 'text'):
                        result_content += item.text
            
            print(f"  Result preview: {result_content[:200]}...")
            
            # Add tool result to messages
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result_content
            })
        
        # Second LLM call with tool results
        print("\n[Step 3] Sending tool results back to LLM...")
        
        final_response = openai_client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.7
        )
        
        return final_response.choices[0].message.content
    else:
        # No tools used, return direct response
        print("  ✓ Model answered directly without tools")
        return message.content


async def main():
    """Main function to run the real MCP demo."""
    print("\n" + "=" * 70)
    print("   MCP (Model Context Protocol) - REAL Implementation")
    print("   Using Filesystem MCP Server with stdio transport")
    print("=" * 70)
    
    # Check prerequisites
    print("\n[Checking Prerequisites]")
    
    # Setup demo files
    print("\nSetting up demo files...")
    demo_folder = setup_demo_files()
    print(f"✓ Demo folder ready: {demo_folder}")
    
    # Initialize OpenAI client
    print("\n[Initializing OpenAI client...]")
    openai_client = get_openai_client()
    print("✓ Connected to Model Studio")
    
    # Initialize MCP client
    print("\n[Initializing MCP client...]")
    mcp_client = MCPClient()
    
    try:
        # Connect to MCP server
        tools = await mcp_client.connect_to_server(demo_folder)
        
        print("\n" + "=" * 70)
        print("   MCP Demo Ready!")
        print("=" * 70)
        print("\nCommands:")
        print("  /help      - Show example queries")
        print("  /tools     - List available MCP tools")
        print("  /files     - List files in demo folder")
        print("  quit/exit  - End the session")
        print("\nExample queries:")
        print('  "What is Project Alpha launch date?"')
        print('  "Read the meeting notes"')
        print('  "What technology stack are we using?"')
        print('  "List all files in the demo folder"')
        print("=" * 70)
        
        # Interactive loop
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in ('quit', 'exit', 'q'):
                    print("\nThank you for trying the real MCP demo!")
                    break
                
                if user_input.lower() == '/help':
                    print("\nExample queries:")
                    print('  "What is Project Alpha launch date?"')
                    print('  "Read the meeting notes"')
                    print('  "What technology stack are we using?"')
                    print('  "List all files in the demo folder"')
                    print('  "Search for Alice in the files"')
                    continue
                
                if user_input.lower() == '/tools':
                    print(f"\nAvailable MCP tools:")
                    for tool in tools:
                        print(f"  - {tool.name}: {tool.description}")
                    continue
                
                if user_input.lower() == '/files':
                    print(f"\nFiles in {demo_folder}:")
                    folder = Path(demo_folder)
                    for i, file_path in enumerate(folder.glob("*"), 1):
                        if file_path.is_file():
                            print(f"  {i}. {file_path.name}")
                    continue
                
                if not user_input:
                    continue
                
                # Process query with MCP
                print("\n" + "-" * 70)
                response = await chat_with_mcp(mcp_client, openai_client, user_input)
                print("-" * 70)
                print(f"\nAnswer: {response}")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    finally:
        # Cleanup
        print("\n[Cleaning up...]")
        await mcp_client.cleanup()
        print("✓ Disconnected from MCP server")


if __name__ == "__main__":
    # Run async main
    asyncio.run(main())

# MCP Server Configuration Guide

## Overview

This guide explains how to configure and use the Model Context Protocol (MCP) server with Qwen models.

## What is MCP Server Configuration?

MCP server configuration involves:
1. Starting an MCP server process
2. Connecting via stdio transport
3. Discovering available tools dynamically
4. Executing tools through the MCP protocol

## MCP Server Types

### 1. Official MCP Servers

| Server | Purpose | Install Command |
|--------|---------|-----------------|
| `server-filesystem` | File operations | `npx @modelcontextprotocol/server-filesystem /path` |
| `server-postgres` | Database queries | `npx @modelcontextprotocol/server-postgres <connection_string>` |
| `server-sqlite` | SQLite database | `npx @modelcontextprotocol/server-sqlite /path/to/db` |
| `server-github` | GitHub API | `npx @modelcontextprotocol/server-github` |

### 2. Custom MCP Servers

You can build your own MCP server using:
- **Node.js**: `@modelcontextprotocol/sdk`
- **Python**: `mcp` package

## Configuration Steps

### Step 1: Define Server Parameters

```python
from mcp import StdioServerParameters

# Filesystem MCP Server
filesystem_params = StdioServerParameters(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/path/to/folder"]
)

# Postgres MCP Server
postgres_params = StdioServerParameters(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-postgres", 
          "postgres://user:pass@localhost/db"]
)
```

### Step 2: Connect to Server

```python
from mcp import ClientSession
from mcp.client.stdio import stdio_client
from contextlib import AsyncExitStack

async def connect_to_mcp_server(server_params):
    """Connect to MCP server and return session."""
    
    # Manage resources
    exit_stack = AsyncExitStack()
    
    # Connect via stdio
    stdio_transport = await exit_stack.enter_async_context(
        stdio_client(server_params)
    )
    stdio, write = stdio_transport
    
    # Create session
    session = await exit_stack.enter_async_context(
        ClientSession(stdio, write)
    )
    
    # Initialize (JSON-RPC handshake)
    await session.initialize()
    
    return session, exit_stack
```

### Step 3: Discover Tools

```python
async def discover_tools(session):
    """Get available tools from MCP server."""
    
    response = await session.list_tools()
    
    print(f"Discovered {len(response.tools)} tools:")
    for tool in response.tools:
        print(f"  - {tool.name}: {tool.description}")
        print(f"    Parameters: {tool.inputSchema}")
    
    return response.tools
```

### Step 4: Execute Tools

```python
async def execute_tool(session, tool_name, arguments):
    """Execute a tool on the MCP server."""
    
    result = await session.call_tool(tool_name, arguments=arguments)
    
    # Extract result content
    content = ""
    for item in result.content:
        if hasattr(item, 'text'):
            content += item.text
    
    return content
```

## Complete Configuration Example

```python
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from contextlib import AsyncExitStack

async def mcp_server_demo():
    """Complete MCP server configuration and usage."""
    
    # 1. Configure server parameters
    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", 
              "/Users/username/documents"]
    )
    
    # 2. Connect to server
    exit_stack = AsyncExitStack()
    
    try:
        stdio_transport = await exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        stdio, write = stdio_transport
        
        session = await exit_stack.enter_async_context(
            ClientSession(stdio, write)
        )
        
        await session.initialize()
        print("✓ Connected to MCP server")
        
        # 3. Discover tools
        tools_response = await session.list_tools()
        print(f"✓ Discovered {len(tools_response.tools)} tools")
        
        # 4. Execute a tool
        result = await session.call_tool(
            "list_directory",
            arguments={"path": "/Users/username/documents"}
        )
        
        print("Directory contents:")
        for item in result.content:
            if hasattr(item, 'text'):
                print(item.text)
    
    finally:
        # 5. Cleanup
        await exit_stack.aclose()
        print("✓ Disconnected from MCP server")

# Run
asyncio.run(mcp_server_demo())
```

## Integration with Qwen

### Format Tools for OpenAI API

```python
def format_mcp_tools_for_qwen(mcp_tools):
    """Convert MCP tools to OpenAI function format."""
    
    openai_tools = []
    for tool in mcp_tools:
        openai_tool = {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema
            }
        }
        openai_tools.append(openai_tool)
    
    return openai_tools
```

### Send Tools to Qwen

```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-your-api-key",
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
)

# Format MCP tools
mcp_tools = await session.list_tools()
openai_tools = format_mcp_tools_for_qwen(mcp_tools.tools)

# Send to Qwen with tools
response = client.chat.completions.create(
    model="qwen3.5-plus",
    messages=[
        {"role": "system", "content": "You have access to filesystem tools."},
        {"role": "user", "content": "List files in my documents folder"}
    ],
    tools=openai_tools,
    tool_choice="auto"
)
```

### Handle Tool Calls from Qwen

```python
message = response.choices[0].message

if message.tool_calls:
    for tool_call in message.tool_calls:
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)
        
        # Execute via MCP
        result = await session.call_tool(tool_name, tool_args)
        
        # Process result...
```

## Multiple MCP Servers

```python
async def connect_multiple_servers():
    """Connect to multiple MCP servers simultaneously."""
    
    servers = {
        "filesystem": StdioServerParameters(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/data"]
        ),
        "database": StdioServerParameters(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-postgres", 
                  "postgres://localhost/mydb"]
        )
    }
    
    sessions = {}
    exit_stack = AsyncExitStack()
    
    for name, params in servers.items():
        stdio_transport = await exit_stack.enter_async_context(
            stdio_client(params)
        )
        stdio, write = stdio_transport
        
        session = await exit_stack.enter_async_context(
            ClientSession(stdio, write)
        )
        
        await session.initialize()
        sessions[name] = session
    
    # Use multiple servers...
    
    await exit_stack.aclose()
```

## Configuration Options

### Environment Variables

```bash
# MCP Server specific
export MCP_SERVER_TIMEOUT=30000
export MCP_LOG_LEVEL=debug

# For custom servers
export MY_MCP_SERVER_API_KEY=xxx
```

### Server Arguments

```python
# With environment variables
server_params = StdioServerParameters(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/path"],
    env={"MCP_LOG_LEVEL": "debug"}
)
```

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `command not found: npx` | Install Node.js: `brew install node` |
| Connection timeout | Check server is running; increase timeout |
| Tool not found | Verify tool name; check server capabilities |
| Permission denied | Check folder permissions for filesystem server |

### Debug Mode

```python
import logging

logging.basicConfig(level=logging.DEBUG)

# MCP SDK will log JSON-RPC messages
```

## Best Practices

1. **Always cleanup**: Use `AsyncExitStack` for proper resource management
2. **Handle errors**: Wrap tool calls in try-except blocks
3. **Validate inputs**: Check tool arguments before calling
4. **Timeout**: Set reasonable timeouts for tool execution
5. **Logging**: Enable debug logging during development

## References

- [MCP Specification](https://modelcontextprotocol.io)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [Official MCP Servers](https://github.com/modelcontextprotocol/servers)

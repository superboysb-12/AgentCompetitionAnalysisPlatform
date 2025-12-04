#!/usr/bin/env python3
"""
LangChain ReAct Agent with MCP Tools and Web Scraping
Simple terminal-based chatbot implementation
"""

import time
import json
import yaml
import requests
from typing import Any, Dict
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langchain_community.document_loaders import WebBaseLoader
from langgraph.prebuilt import create_react_agent


# Global MCP config
MCP_CONFIG = {}
MCP_SESSION_ID = None


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def initialize_mcp_session() -> bool:
    """Initialize MCP session and obtain session ID"""
    global MCP_SESSION_ID

    try:
        mcp_url = MCP_CONFIG['url']
        if not mcp_url.endswith('/'):
            mcp_url += '/'

        # Send initialize request
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "langchain-react-agent",
                    "version": "1.0.0"
                }
            }
        }

        response = requests.post(
            mcp_url,
            json=init_request,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream"
            },
            timeout=MCP_CONFIG['timeout']
        )
        response.raise_for_status()

        # Extract session ID from response headers
        session_id = response.headers.get('Mcp-Session-Id')
        if session_id:
            MCP_SESSION_ID = session_id
            return True

        return False

    except Exception as e:
        print(f"Warning: Failed to initialize MCP session: {e}")
        return False


def call_mcp_tool(tool_name: str, **kwargs) -> str:
    """Call MCP tool via Streamable HTTP transport"""
    try:
        # Initialize session if not already done
        global MCP_SESSION_ID
        if MCP_SESSION_ID is None:
            if not initialize_mcp_session():
                return "Error: Failed to initialize MCP session"

        # MCP endpoint URL - ensure trailing slash for Streamable HTTP
        mcp_url = MCP_CONFIG['url']
        if not mcp_url.endswith('/'):
            mcp_url += '/'

        # Build JSON-RPC request
        request_id = int(time.time() * 1000)
        json_rpc_request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": kwargs
            }
        }

        # Build headers with session ID
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream"
        }
        if MCP_SESSION_ID:
            headers["Mcp-Session-Id"] = MCP_SESSION_ID

        # MCP Streamable HTTP: POST directly to the MCP endpoint
        # Spec: https://modelcontextprotocol.io/specification/2025-03-26/basic/transports/
        response = requests.post(
            mcp_url,
            json=json_rpc_request,
            headers=headers,
            timeout=MCP_CONFIG['timeout']
        )
        response.raise_for_status()

        # Parse response based on Content-Type
        content_type = response.headers.get('Content-Type', '')

        # Handle SSE response (text/event-stream)
        if 'text/event-stream' in content_type:
            lines = response.text.strip().split('\n')
            for line in lines:
                if line.startswith('data: '):
                    result = json.loads(line[6:])  # Remove 'data: ' prefix
                    return _extract_mcp_result(result)
            return response.text  # Fallback

        # Handle JSON response (application/json)
        return _extract_mcp_result(response.json())

    except requests.RequestException as e:
        # Capture full error details for debugging
        error_details = [f"Error calling MCP tool '{tool_name}': {str(e)}"]

        if hasattr(e, 'response') and e.response is not None:
            error_details.append(f"\n--- Response Details ---")
            error_details.append(f"Status Code: {e.response.status_code}")
            error_details.append(f"Headers: {dict(e.response.headers)}")
            error_details.append(f"Body: {e.response.text}")
            error_details.append(f"\n--- Request Details ---")
            error_details.append(f"URL: {e.response.request.url}")
            error_details.append(f"Method: {e.response.request.method}")
            error_details.append(f"Headers: {dict(e.response.request.headers)}")
            error_details.append(f"Body: {e.response.request.body}")

        return "\n".join(error_details)
    except Exception as e:
        return f"Unexpected error: {str(e)}"


def _extract_mcp_result(result: dict) -> str:
    """Extract content from MCP JSON-RPC result"""
    # Check for JSON-RPC error
    if "error" in result:
        error = result["error"]
        return f"MCP Error [{error.get('code')}]: {error.get('message')}"

    # Extract content from result
    if "result" in result:
        tool_result = result["result"]
        if "content" in tool_result:
            content_items = tool_result["content"]
            texts = [item.get("text", "") for item in content_items if item.get("type") == "text"]
            return "\n".join(texts)
        return str(tool_result)

    return str(result)


@tool
def neo4j_query(query: str) -> str:
    """Execute Cypher queries on Neo4j graph database. Input should be a valid Cypher query string."""
    return call_mcp_tool("read_neo4j_cypher", query=query)


@tool
def neo4j_schema() -> str:
    """Get the schema of the Neo4j database including node labels, attributes, and relationships."""
    return call_mcp_tool("get_neo4j_schema")


@tool
def web_scraper(url: str) -> str:
    """Scrape and extract text content from a web page. Input should be a valid URL."""
    try:
        loader = WebBaseLoader(url)
        docs = loader.load()
        if docs:
            return docs[0].page_content[:2000]  # Limit content length
        return "No content found"
    except Exception as e:
        return f"Error scraping {url}: {str(e)}"


def create_agent(config: Dict[str, Any]):
    """Create ReAct agent with tools"""
    # Set global MCP config
    global MCP_CONFIG
    MCP_CONFIG = config['mcp']

    # Initialize LLM
    llm = ChatOpenAI(
        base_url=config['llm']['endpoint'],
        api_key=config['llm']['api_key'],
        model=config['llm']['model'],
        temperature=config['llm']['temperature'],
        max_tokens=config['llm']['max_tokens']
    )

    # Define tools
    tools = [neo4j_query, neo4j_schema, web_scraper]

    # Get system prompt from config
    system_prompt = config['agent'].get('system_prompt',
                                        "You are a helpful assistant with access to tools. Use them when needed to answer questions.")

    # Create ReAct agent using LangGraph with custom system prompt
    agent = create_react_agent(llm, tools, prompt=SystemMessage(content=system_prompt))

    return agent


def main():
    """Main terminal chat loop"""
    print("=" * 60)
    print("LangChain ReAct Agent - Terminal Chat")
    print("=" * 60)
    print("Loading configuration...")

    try:
        config = load_config()
        print("Creating agent...")
        agent = create_agent(config)
        print("Agent ready! Type 'quit' or 'exit' to stop.\n")

        while True:
            try:
                user_input = input("\nYou: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\nGoodbye!")
                    break

                # Get agent response with streaming
                if config['agent']['verbose']:
                    print("\n" + "="*60)
                    print("Agent Working...")
                    print("="*60)

                # Stream agent execution step by step
                step_count = 0
                final_message = None
                verbose = config['agent']['verbose']

                for chunk in agent.stream({"messages": [("human", user_input)]}):
                    # Get the node name (e.g., 'agent', 'tools')
                    node_name = next((k for k in chunk.keys() if k not in ('__start__', '__end__')), None)

                    if not (verbose and node_name):
                        continue

                    # Get messages from the current node
                    messages = chunk.get(node_name, {}).get('messages', [])

                    for msg in messages:
                        # Show AI messages
                        if hasattr(msg, 'type') and msg.type == 'ai':
                            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                                # AI is calling a tool
                                step_count += 1
                                print(f"\n[Step {step_count}] 🤖 AI Decision:")
                                for tool_call in msg.tool_calls:
                                    tool_name = tool_call.get('name', 'unknown')
                                    args = tool_call.get('args', {})
                                    print(f"  → Calling tool: {tool_name}")
                                    for key, value in args.items():
                                        value_str = str(value)[:100]
                                        if len(str(value)) > 100:
                                            value_str += "..."
                                        print(f"    {key}: {value_str}")
                            elif hasattr(msg, 'content') and msg.content:
                                # Save final AI response
                                final_message = msg

                        # Show tool messages
                        elif hasattr(msg, 'type') and msg.type == 'tool':
                            step_count += 1
                            tool_name = getattr(msg, 'name', 'unknown')
                            content = msg.content
                            print(f"\n[Step {step_count}] 🔧 Tool Result ({tool_name}):")
                            # Show first 200 chars of result
                            if len(content) > 200:
                                print(f"  {content[:200]}...")
                            else:
                                print(f"  {content}")

                if verbose:
                    print("\n" + "="*60)

                # Display the final answer
                if final_message and hasattr(final_message, 'content'):
                    print(f"\n✅ Final Answer:\n{final_message.content}")
                else:
                    print(f"\nNo final answer generated.")

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {str(e)}")
                if config['agent']['verbose']:
                    import traceback
                    traceback.print_exc()
                continue

    except FileNotFoundError:
        print("Error: config.yaml not found. Please create a configuration file.")
    except Exception as e:
        print(f"Error initializing agent: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

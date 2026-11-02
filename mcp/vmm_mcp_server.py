#!/usr/bin/env python3
"""
llamampere-vmm-swap: Model Context Protocol (MCP) Server
========================================================
Exposes low-level VMM fast-swap and agent orchestration tools to any MCP host
(Google Antigravity IDE, Claude Desktop, Cursor, Continue.dev).

Supported Tools:
  - `vmm_list_models`: Discovers configured models, active VRAM status, and latencies.
  - `vmm_swap_model`: Manually triggers a sub-2.3s VMM RAM<->VRAM hot swap.
  - `vmm_query_agent`: Sends a prompt to a specialized agent role with automatic swap.
"""

import sys
import json
import urllib.request
import urllib.error

ROUTER_STATUS_URL = "http://127.0.0.1:8000/status"
ROUTER_SWAP_URL = "http://127.0.0.1:8000/v1/vmm/swap"
ROUTER_CHAT_URL = "http://127.0.0.1:8000/v1/chat/completions"

TOOLS = [
    {
        "name": "vmm_list_models",
        "description": "Lists all available agent models, their roles, active VRAM residency, and recent swap metrics.",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "vmm_swap_model",
        "description": "Triggers a sub-2.3s CUDA VMM hot-swap to bring a specific model into GPU VRAM.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "target_model": {
                    "type": "string",
                    "enum": ["architect", "coder", "debugger"],
                    "description": "Model role to swap into VRAM"
                }
            },
            "required": ["target_model"]
        }
    },
    {
        "name": "vmm_query_agent",
        "description": "Sends a prompt to an agent role (architect, coder, debugger) with transparent sub-2.3s VMM swap.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "role": {
                    "type": "string",
                    "enum": ["architect", "coder", "debugger"],
                    "description": "Specialized agent role to query"
                },
                "prompt": {
                    "type": "string",
                    "description": "User instruction or code context"
                },
                "system_prompt": {
                    "type": "string",
                    "description": "Optional system prompt instructions"
                }
            },
            "required": ["role", "prompt"]
        }
    }
]

def handle_vmm_list_models() -> dict:
    try:
        req = urllib.request.Request(ROUTER_STATUS_URL)
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return {"content": [{"type": "text", "text": json.dumps(data, indent=2)}]}
    except Exception as e:
        return {"content": [{"type": "text", "text": f"Router offline or error: {e}"}], "isError": True}

def handle_vmm_swap_model(target_model: str) -> dict:
    try:
        payload = json.dumps({"model": target_model}).encode("utf-8")
        req = urllib.request.Request(ROUTER_SWAP_URL, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return {"content": [{"type": "text", "text": f"Swapped to '{target_model}' in {data.get('swap_latency_seconds')} seconds."}]}
    except Exception as e:
        return {"content": [{"type": "text", "text": f"Swap failed: {e}"}], "isError": True}

def handle_vmm_query_agent(role: str, prompt: str, system_prompt: str = "") -> dict:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = json.dumps({
        "model": role,
        "messages": messages,
        "stream": False
    }).encode("utf-8")

    try:
        req = urllib.request.Request(ROUTER_CHAT_URL, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as resp:
            res_json = json.loads(resp.read().decode("utf-8"))
            content = res_json["choices"][0]["message"]["content"]
            return {"content": [{"type": "text", "text": content}]}
    except Exception as e:
        return {"content": [{"type": "text", "text": f"Agent query failed: {e}"}], "isError": True}

def send_response(response_obj: dict):
    line = json.dumps(response_obj)
    sys.stdout.write(f"Content-Length: {len(line.encode('utf-8'))}\r\n\r\n{line}")
    sys.stdout.flush()

def main():
    while True:
        header = sys.stdin.readline()
        if not header:
            break
        header = header.strip()
        if not header.startswith("Content-Length:"):
            continue
        try:
            content_length = int(header.split(":")[1].strip())
            # Skip empty line
            sys.stdin.readline()
            body_bytes = sys.stdin.read(content_length)
            msg = json.loads(body_bytes)
        except Exception:
            continue

        msg_id = msg.get("id")
        method = msg.get("method")
        params = msg.get("params", {})

        if method == "initialize":
            send_response({
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "llamampere-vmm-mcp", "version": "1.0.0"}
                }
            })
        elif method == "tools/list":
            send_response({
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {"tools": TOOLS}
            })
        elif method == "tools/call":
            tool_name = params.get("name")
            args = params.get("arguments", {})
            if tool_name == "vmm_list_models":
                res = handle_vmm_list_models()
            elif tool_name == "vmm_swap_model":
                res = handle_vmm_swap_model(args.get("target_model", "architect"))
            elif tool_name == "vmm_query_agent":
                res = handle_vmm_query_agent(args.get("role", "architect"), args.get("prompt", ""), args.get("system_prompt", ""))
            else:
                res = {"content": [{"type": "text", "text": f"Unknown tool: {tool_name}"}], "isError": True}

            send_response({
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": res
            })

if __name__ == "__main__":
    main()

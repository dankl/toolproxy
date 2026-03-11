"""
Shared fixtures and tool definitions for live integration tests.

Verbindungseinstellungen via Env-Variablen (kein SSH-Tunnel nötig):

    export LITELLM_MASTER_KEY="dein-key"
    export LIVE_UPSTREAM_URL="http://<LITELLM_HOST>:4000/v1"    # Pflicht: LiteLLM-Endpoint
    export LIVE_MODEL="openai/gpt-oss-120b"                     # optional, das ist der Default

Ausführen:
    cd toolproxy && python3 -m pytest -m live -v
"""
import asyncio
import json
import os
import pytest
from unittest.mock import patch

import app.main as main_module
from app.services.vllm_client import VLLMClient
from tests.conftest import _tool, user_msg


# ── Shared tool builder helpers ───────────────────────────────────────────────

SYSTEM_MSG = {"role": "system", "content": "You are a coding assistant. Use tools to complete tasks."}


def post(client, tools, prompt, system=SYSTEM_MSG):
    model = os.environ.get("LIVE_MODEL", "openai/gpt-oss-120b")
    resp = client.post("/v1/chat/completions", json={
        "model": model,
        "messages": [system, user_msg(prompt)],
        "tools": tools,
    })
    assert resp.status_code == 200, resp.text
    return resp.json()


def parse_tool_call(response_json: dict) -> tuple[str, dict]:
    choices = response_json["choices"]
    assert choices, "No choices in response"
    msg = choices[0]["message"]
    tool_calls = msg.get("tool_calls") or []
    assert tool_calls, f"No tool_calls in response. Content: {msg.get('content', '')[:300]}"
    tc = tool_calls[0]
    return tc["function"]["name"], json.loads(tc["function"]["arguments"])


# ── Roo Code tool definitions ─────────────────────────────────────────────────

TOOL_WRITE_TO_FILE = _tool(
    "write_to_file", "Write content to a file",
    {"path": {"type": "string"}, "content": {"type": "string"}},
    ["path", "content"],
)
TOOL_READ_FILE = _tool(
    "read_file", "Read the contents of a file",
    {"path": {"type": "string"}},
    ["path"],
)
TOOL_APPLY_DIFF = _tool(
    "apply_diff", "Apply a unified diff patch to a file",
    {"path": {"type": "string"}, "diff": {"type": "string"}},
    ["path", "diff"],
)
TOOL_LIST_FILES = _tool(
    "list_files", "List files in a directory",
    {"path": {"type": "string"}},
    ["path"],
)
TOOL_DELETE_FILE = _tool(
    "delete_file", "Delete a file from disk",
    {"path": {"type": "string"}},
    ["path"],
)
TOOL_EXECUTE_COMMAND = _tool(
    "execute_command", "Run a shell command",
    {"command": {"type": "string"}},
    ["command"],
)
TOOL_ATTEMPT_COMPLETION = _tool(
    "attempt_completion", "Signal that the task is complete",
    {"result": {"type": "string"}},
    ["result"],
)
TOOL_ASK_FOLLOWUP_QUESTION = _tool(
    "ask_followup_question", "Ask the user a clarifying question",
    {"question": {"type": "string"}},
    ["question"],
)
TOOL_SEARCH_FILES = _tool(
    "search_files", "Search for a pattern in files",
    {"path": {"type": "string"}, "regex": {"type": "string"}},
    ["path", "regex"],
)

ROO_CODE_TOOLS = [
    TOOL_WRITE_TO_FILE,
    TOOL_READ_FILE,
    TOOL_APPLY_DIFF,
    TOOL_LIST_FILES,
    TOOL_DELETE_FILE,
    TOOL_EXECUTE_COMMAND,
    TOOL_ATTEMPT_COMPLETION,
    TOOL_ASK_FOLLOWUP_QUESTION,
    TOOL_SEARCH_FILES,
]

# ── OpenCode tool definitions ─────────────────────────────────────────────────

TOOL_WRITE = _tool(
    "write", "Write content to a file",
    {"filePath": {"type": "string"}, "content": {"type": "string"}},
    ["filePath", "content"],
)
TOOL_READ = _tool(
    "read", "Read a file",
    {"filePath": {"type": "string"}},
    ["filePath"],
)
TOOL_LIST = _tool(
    "list", "List files in a directory",
    {"dirPath": {"type": "string"}},
    ["dirPath"],
)
TOOL_BASH = _tool(
    "bash", "Run a shell command",
    {"command": {"type": "string"}, "description": {"type": "string"}},
    ["command", "description"],
)
TOOL_EDIT = _tool(
    "edit", "Edit a file by replacing a string",
    {"filePath": {"type": "string"}, "oldString": {"type": "string"}, "newString": {"type": "string"}},
    ["filePath", "oldString", "newString"],
)
TOOL_GLOB = _tool(
    "glob", "Find files matching a pattern",
    {"pattern": {"type": "string"}},
    ["pattern"],
)
TOOL_GREP = _tool(
    "grep", "Search file contents with a regex",
    {"pattern": {"type": "string"}, "path": {"type": "string"}},
    ["pattern"],
)

OPEN_CODE_TOOLS = [
    TOOL_WRITE,
    TOOL_READ,
    TOOL_LIST,
    TOOL_BASH,
    TOOL_EDIT,
    TOOL_GLOB,
    TOOL_GREP,
]

# ── Fixture ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def live(client):
    """
    Replace the module-level upstream_client with a real VLLMClient.

    Reads connection settings from environment variables so the same tests
    work on any setup without editing this file:

        LIVE_UPSTREAM_URL   — Pflicht: LiteLLM-Endpoint (kein Default)
        LIVE_MODEL          — default: openai/gpt-oss-120b
        LIVE_API_KEY        — default: $LITELLM_MASTER_KEY (then "dummy-key")
    """
    base_url = os.environ.get("LIVE_UPSTREAM_URL")
    if not base_url:
        pytest.skip("LIVE_UPSTREAM_URL nicht gesetzt — Live-Test übersprungen")
    model = os.environ.get("LIVE_MODEL", "openai/gpt-oss-120b")
    api_key = os.environ.get("LIVE_API_KEY") or os.environ.get("LITELLM_MASTER_KEY", "dummy-key")

    real_uc = VLLMClient(
        base_url=base_url,
        model=model,
        api_key=api_key,
        timeout=120,
        max_retries=1,
    )
    with patch.object(main_module, "upstream_client", real_uc):
        yield client
    asyncio.run(real_uc.close())

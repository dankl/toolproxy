"""
Shared fixtures and helpers for toolproxy tests.
"""
import json
import pytest
from unittest.mock import AsyncMock, patch
from starlette.testclient import TestClient

import app.main as main_module
from app.main import app


# ──────────────────────────────────────────────────────────────────────────────
# Standard Roo-Code-like tool definitions
# ──────────────────────────────────────────────────────────────────────────────

def _tool(name, description, properties, required):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


TOOL_WRITE_TO_FILE = _tool(
    "write_to_file", "Write content to a file",
    {"path": {"type": "string"}, "content": {"type": "string"}},
    ["path", "content"],
)
TOOL_READ_FILE = _tool(
    "read_file", "Read a file",
    {"path": {"type": "string"}},
    ["path"],
)
TOOL_APPLY_DIFF = _tool(
    "apply_diff", "Apply a unified diff to a file",
    {"path": {"type": "string"}, "diff": {"type": "string"}},
    ["path", "diff"],
)
TOOL_LIST_FILES = _tool(
    "list_files", "List files in a directory",
    {"path": {"type": "string", "description": "Directory path"}},
    ["path"],
)
TOOL_APPEND_TO_FILE = _tool(
    "append_to_file", "Append content to an existing file",
    {"path": {"type": "string"}, "content": {"type": "string"}},
    ["path", "content"],
)
TOOL_ATTEMPT_COMPLETION = _tool(
    "attempt_completion", "Signal task completion with a summary",
    {"result": {"type": "string"}},
    ["result"],
)

DEFAULT_TOOLS = [
    TOOL_WRITE_TO_FILE,
    TOOL_APPEND_TO_FILE,
    TOOL_READ_FILE,
    TOOL_APPLY_DIFF,
    TOOL_LIST_FILES,
    TOOL_ATTEMPT_COMPLETION,
]

# Common message helpers
SYSTEM_MSG = {"role": "system", "content": "You are a coding assistant."}


def user_msg(text: str) -> dict:
    return {"role": "user", "content": text}


def llm_response(content: str) -> dict:
    """Build a minimal upstream LLM response dict."""
    return {
        "id": "test-chatcmpl-001",
        "created": 1_700_000_000,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": None,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 120, "completion_tokens": 40, "total_tokens": 160},
    }


def parse_tool_call(response_json: dict) -> tuple[str, dict]:
    """Extract (tool_name, arguments_dict) from a /v1/chat/completions response."""
    tc = response_json["choices"][0]["message"]["tool_calls"][0]
    return tc["function"]["name"], json.loads(tc["function"]["arguments"])


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def client():
    """FastAPI TestClient — starts the app lifespan once for the whole session."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def llm(client):
    """
    Per-test AsyncMock for upstream_client.chat_completion.

    The lifespan already set main_module.upstream_client = VLLMClient(...).
    We patch that module attribute so every call inside the endpoint hits our mock.

    Usage in tests:
        llm.return_value = llm_response("<some_xml>...</some_xml>")
        # or
        async def my_side_effect(messages=None, **kwargs): ...
        llm.side_effect = my_side_effect
    """
    mock = AsyncMock()
    with patch.object(main_module, "upstream_client") as mock_uc:
        mock_uc.chat_completion = mock
        yield mock

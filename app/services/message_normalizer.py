"""
Message normalisation for toolproxy.

Converts the client's raw OpenAI message list into the flat, XML-normalised
form the model expects: content arrays flattened, tool_calls rendered as XML.
"""
import json
import logging
from typing import Dict, List, Optional

from app.services.xml_parser import dict_to_xml_element

logger = logging.getLogger(__name__)


def _tool_call_to_xml(name: str, args_str: str) -> str:
    """Convert an OpenAI tool_call to XML for conversation history."""
    try:
        args = json.loads(args_str) if isinstance(args_str, str) else args_str
    except Exception:
        args = {"content": str(args_str)}
    return dict_to_xml_element(args, name)


def normalize_messages(
    messages: List[Dict],
    request_id: str,
    canonical_map: Optional[Dict[str, str]] = None,
) -> List[Dict]:
    """
    Normalise incoming messages so the model sees consistent XML tool calls in history:

    - Content arrays (OpenAI/Anthropic multi-block format) → flat string
      - text blocks      → plain text
      - tool_use blocks  → XML  (assistant called a tool)
      - tool_result blocks → plain text  (result of a tool execution)
    - tool_calls field on assistant messages → XML in content

    This means the model reads its own past tool calls as XML and learns to
    keep producing XML in subsequent turns.

    canonical_map: optional dict of client-name → canonical-name.  When provided,
    tool call names in the history are translated to canonical names before being
    written as XML.  This keeps the model's XML history consistent with
    the canonical tool names used in the system prompt and priming examples.
    """
    normalized = []
    for msg in messages:
        msg = dict(msg)

        # role:tool (tool results from client) → role:user so the model understands it
        if msg.get("role") == "tool":
            raw = msg.get("content", "")
            if isinstance(raw, list):
                raw = "\n".join(
                    item.get("text", str(item)) if isinstance(item, dict) else str(item)
                    for item in raw
                )
            msg["role"] = "user"
            msg["content"] = f"[Tool Result]\n{raw}"
            msg.pop("tool_call_id", None)
            msg.pop("name", None)
            normalized.append(msg)
            continue

        content = msg.get("content")

        if isinstance(content, list):
            # If an assistant message mixes text and tool_use blocks, drop the text.
            # The proxy rule is "one tool call, nothing else". Text alongside tool_use
            # is always hallucinated content (e.g. "[Awaiting tool result]...") that
            # pollutes history and causes the model to keep confabulating.
            if msg.get("role") == "assistant":
                has_tool_use = any(
                    isinstance(item, dict) and item.get("type") == "tool_use"
                    for item in content
                )
                if has_tool_use:
                    content = [
                        item for item in content
                        if not (isinstance(item, dict) and item.get("type") == "text")
                    ]

            parts = []
            for item in content:
                if not isinstance(item, dict):
                    parts.append(str(item))
                    continue
                t = item.get("type", "")
                if t == "text":
                    parts.append(item.get("text", ""))
                elif t == "tool_use":
                    # Assistant called a tool — represent as XML
                    xml = dict_to_xml_element(item.get("input", {}), item.get("name", "unknown"))
                    parts.append(xml)
                elif t == "tool_result":
                    # Anthropic-format tool result (role:user with content array).
                    # Add [Tool Result] prefix — same as OpenAI role:tool messages —
                    # so loop_detection and any future logic can identify tool results
                    # regardless of which client format (OpenAI vs Anthropic) is used.
                    inner = item.get("content", "")
                    inner_text = ""
                    if isinstance(inner, list):
                        for sub in inner:
                            if isinstance(sub, dict) and sub.get("type") == "text":
                                inner_text += sub.get("text", "")
                    else:
                        inner_text = str(inner)
                    parts.append(f"[Tool Result]\n{inner_text}")
                elif "text" in item:
                    parts.append(item["text"])
            msg["content"] = "\n".join(p for p in parts if p)

        # tool_calls field on assistant → XML appended to content
        tool_calls = msg.get("tool_calls")
        if msg.get("role") == "assistant" and tool_calls:
            xml_parts = []
            for tc in tool_calls:
                name = tc.get("function", {}).get("name", "unknown")
                # Translate client-specific names (e.g. "write") to canonical
                # names (e.g. "write_to_file") so the model always sees the same XML
                # tag names in history as in the system prompt / priming examples.
                if canonical_map:
                    name = canonical_map.get(name, name)
                args = tc.get("function", {}).get("arguments", "{}")
                xml_parts.append(_tool_call_to_xml(name, args))
            msg["content"] = "\n".join(xml_parts)
            # Remove tool_calls so the upstream only sees XML in content —
            # not both XML and native tool_calls, which confuses models without
            # native function-call support.
            msg.pop("tool_calls", None)
            logger.debug(f"[{request_id}] Normalised {len(tool_calls)} tool_calls → XML")

        normalized.append(msg)
    return normalized

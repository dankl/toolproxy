"""
Text-synthesis helpers for toolproxy.

When the model returns plain prose instead of an XML tool call, these functions
infer the correct tool call from context.
"""
import json
import logging
import re
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_FILE_EXTENSIONS = {"md", "py", "ts", "js", "json", "yaml", "yml", "txt", "toml", "sh"}


def _extract_target_file_from_context(messages: List[Dict]) -> Optional[str]:
    """Return the most likely target file path from VSCode Open Tabs in the last user message."""
    candidate_files: List[str] = []
    user_text = ""

    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        items = content if isinstance(content, list) else [{"type": "text", "text": content}]
        for item in items:
            if not isinstance(item, dict) or item.get("type") != "text":
                continue
            text = item.get("text", "")
            user_text += " " + text
            tabs_match = re.search(r"# VSCode Open Tabs\n(.*?)(?:\n\n|\n#)", text, re.DOTALL)
            if tabs_match:
                for tab in tabs_match.group(1).split(","):
                    tab = tab.strip()
                    ext = tab.rsplit(".", 1)[-1].lower() if "." in tab else ""
                    if ext in _FILE_EXTENSIONS and not tab.startswith("../") and "/tmp/" not in tab:
                        candidate_files.append(tab)
        break  # only look at the most recent user message

    if not candidate_files:
        return None
    if len(candidate_files) == 1:
        return candidate_files[0]

    user_lower = user_text.lower()
    for f in candidate_files:
        if f.rsplit("/", 1)[-1].lower() in user_lower:
            return f
    return candidate_files[0]


def _was_recently_written(path: str, messages: List[Dict], lookback: int = 10) -> bool:
    """Return True if write_to_file was already called for this exact path in recent history."""
    recent = messages[-lookback:] if len(messages) > lookback else messages
    _WRITE_TOOLS = {"write_to_file", "append_to_file"}

    for msg in recent:
        if msg.get("role") != "assistant":
            continue
        tool_calls = msg.get("tool_calls") or []
        content = msg.get("content") or ""

        for tc in tool_calls:
            func = tc.get("function", {})
            if func.get("name") not in _WRITE_TOOLS:
                continue
            try:
                args = json.loads(func.get("arguments", "{}"))
                if isinstance(args, dict) and args.get("path") == path:
                    return True
            except (json.JSONDecodeError, AttributeError):
                pass

        if "write_to_file" in content or "append_to_file" in content:
            if f'"path": "{path}"' in content or f'<path>{path}</path>' in content:
                return True

    return False


def synthesize_tool_call_from_text(
    content: str, messages: List[Dict], request_tools: List[Dict], request_id: str
) -> Optional[List[Dict]]:
    """
    Convert a plain-text model response into a tool call so clients like Roo Code
    don't reject it with '[ERROR] You did not use a tool'.

    Strategy:
    1. Long text that looks like file content + determinable target file
       → write_to_file(path, content)
    2. Fallback → attempt_completion(result=content)
    """
    stripped = content.strip()
    if not stripped:
        return None

    tool_names = {t.get("function", {}).get("name", "") for t in request_tools if "function" in t}

    # Never treat XML tool call syntax as file content — it would overwrite files with raw XML
    looks_like_xml_tool_call = stripped.startswith("<") and re.search(r"<\w+>[\s\S]*</\w+>", stripped)

    looks_like_file_content = (
        not looks_like_xml_tool_call
        and len(stripped) > 200
        and any(marker in stripped for marker in ("##", "```", " | ", "---", "\n\n"))
    )

    if looks_like_file_content and "write_to_file" in tool_names:
        target = _extract_target_file_from_context(messages)
        if target and not _was_recently_written(target, messages):
            logger.info(
                f"[{request_id}] Text response looks like file content → "
                f"synthesizing write_to_file({target!r})"
            )
            return [{
                "id": f"call_{request_id}_write_to_file",
                "type": "function",
                "function": {
                    "name": "write_to_file",
                    "arguments": json.dumps({"path": target, "content": stripped}),
                },
            }]

    # Rescue truncated or otherwise unmatched write_to_file XML.
    # Triggers when the response starts with <write_to_file> (or alias) but
    # extract_xml_tool_calls found nothing — typically because:
    #   (a) the response was cut off before </write_to_file>, or
    #   (b) </write_to_file> appeared inside <content> causing lazy regex to stop early.
    if looks_like_xml_tool_call and "write_to_file" in tool_names:
        m = re.search(
            # Matches <write>, <write_file>, <write_to_file> with <path> or <filePath>
            r"<write(?:(?:_to)?_file)?\b[^>]*>\s*<(?:file)?path>(.*?)</(?:file)?path>\s*<content>([\s\S]+?)(?:</content>|$)",
            stripped,
            re.IGNORECASE,
        )
        if m:
            path_val = m.group(1).strip()
            content_val = m.group(2).strip()
            if path_val and content_val:
                logger.info(
                    f"[{request_id}] Partial XML rescue → write_to_file({path_val!r})"
                )
                return [{
                    "id": f"call_{request_id}_write_to_file",
                    "type": "function",
                    "function": {
                        "name": "write_to_file",
                        "arguments": json.dumps({"path": path_val, "content": content_val}),
                    },
                }]

    if "attempt_completion" in tool_names:
        logger.info(f"[{request_id}] Text response → synthesizing attempt_completion fallback")
        return [{
            "id": f"call_{request_id}_attempt_completion",
            "type": "function",
            "function": {
                "name": "attempt_completion",
                "arguments": json.dumps({"result": stripped}),
            },
        }]

    return None

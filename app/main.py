"""
toolproxy — XML Tool Call Proxy

Translates OpenAI native tool_calls API ↔ XML format that the OCI model
(gpt-oss-120b) reliably produces. Sits between any OpenAI-compatible client
and oci-proxy / LiteLLM.

Flow:
  Client (Roo Code, opencode, etc.)
    ↓  OpenAI JSON with tools[] array
  toolproxy (Port 8007)
    ↓  XML System Prompt + XML-normalised history
  oci-proxy (8005) → Oracle OCI GenAI
    ↓  XML tool call in response
  toolproxy — parses XML → OpenAI tool_calls
    ↑  Standard OpenAI response
  Client
"""
import json
import logging
import re
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.config import settings
from app.services.json_repair import safe_parse_json
from app.services.vllm_client import VLLMClient
from app.services.xml_parser import (
    convert_xml_tool_calls_to_openai_format,
    dict_to_xml_element,
    extract_tool_names_from_request,
    extract_xml_tool_calls,
)
from app.services.xml_prompt_builder import build_xml_system_prompt

logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool name / argument recovery helpers (same approach as roocode-proxy)
# ---------------------------------------------------------------------------

_TOOL_NAME_KEYS = ("name", "tool", "action", "function", "command", "method")
_TOOL_ARGS_KEYS = ("arguments", "parameters", "params", "args", "input", "inputs")

_PARAM_ALIASES: Dict[str, str] = {
    "patch": "diff",
    "file_content": "content",
    "text": "content",
    "body": "content",
    "file": "path",
    "filename": "path",
    "filepath": "path",
    "directory": "path",
    "folder": "path",
    "cmd": "command",
    "shell": "command",
}

# Hallucinated tool name → real tool name
_TOOL_NAME_ALIASES: Dict[str, str] = {
    "write_file": "write_to_file",
    "create_file": "write_to_file",
    "open_file": "read_file",
    "view_file": "read_file",
    "read_file_content": "read_file",
    "apply_patch": "apply_diff",
    "patch_file": "apply_diff",
    "list_dir": "list_files",
    "ls": "list_files",
    "list_directory": "list_files",
}


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------

upstream_client: VLLMClient = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global upstream_client
    logger.info(
        f"toolproxy starting | upstream={settings.upstream_url} | model={settings.upstream_model}"
    )
    upstream_client = VLLMClient(
        base_url=settings.upstream_url,
        model=settings.upstream_model,
        api_key=settings.upstream_api_key,
        timeout=settings.request_timeout,
        max_retries=settings.max_retries,
    )
    yield
    await upstream_client.close()


app = FastAPI(
    title="toolproxy",
    description="XML Tool Call Proxy — translates OpenAI tool_calls ↔ XML for LLM backends",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "service": "toolproxy",
        "upstream": settings.upstream_url,
        "model": settings.upstream_model,
    }


# ---------------------------------------------------------------------------
# Argument / tool name recovery
# ---------------------------------------------------------------------------


def _alias_params(args: Dict) -> Dict:
    return {_PARAM_ALIASES.get(k, k): v for k, v in args.items()}


def _score_args(args: Dict, tools: List[Dict]) -> Optional[str]:
    """Score args dict against tool schemas. Returns best-matching tool name or None."""
    best, best_score = None, 0
    for tool in tools:
        fn = tool.get("function", {})
        name = fn.get("name")
        required = fn.get("parameters", {}).get("required", [])
        all_params = set(fn.get("parameters", {}).get("properties", {}).keys())
        arg_keys = set(args.keys())
        req_match = sum(1 for p in required if p in args)
        opt_match = len(arg_keys & all_params)
        extra = len(arg_keys - all_params)
        if req_match == len(required) and len(required) > 0:
            score = 1000 + opt_match * 10 - extra * 5
        else:
            score = req_match * 10 + opt_match - extra * 5
        if score > best_score:
            best_score, best = score, name
    return best if best_score >= 1000 else None


def _try_json_tool_call(
    parsed: Dict, tool_names: List[str], tools: List[Dict], request_id: str
) -> Optional[List[Dict]]:
    """Try to extract a tool call from a parsed JSON dict using any known key combination."""
    for name_key in _TOOL_NAME_KEYS:
        tool_name = parsed.get(name_key)
        if not isinstance(tool_name, str):
            continue
        for args_key in _TOOL_ARGS_KEYS:
            if args_key not in parsed:
                continue
            args = parsed[args_key]

            if tool_name in tool_names:
                args_str = args if isinstance(args, str) else json.dumps(args)
                logger.info(f"[{request_id}] JSON tool call: {tool_name}")
                return [
                    {
                        "id": f"call_{request_id}_{tool_name}",
                        "type": "function",
                        "function": {"name": tool_name, "arguments": args_str},
                    }
                ]

            if isinstance(args, dict):
                # Hallucinated tool name — try to recover via arg scoring
                effective_args = args
                recovered = _score_args(args, tools)
                if not recovered:
                    aliased = _alias_params(args)
                    if aliased != args:
                        recovered = _score_args(aliased, tools)
                        if recovered:
                            effective_args = aliased
                if recovered:
                    logger.info(
                        f"[{request_id}] Hallucinated {tool_name!r} → recovered {recovered!r}"
                    )
                    return [
                        {
                            "id": f"call_{request_id}_{recovered}",
                            "type": "function",
                            "function": {"name": recovered, "arguments": json.dumps(effective_args)},
                        }
                    ]
    return None


# ---------------------------------------------------------------------------
# Message normalisation
# ---------------------------------------------------------------------------


def _tool_call_to_xml(name: str, args_str: str) -> str:
    """Convert an OpenAI tool_call to XML for conversation history."""
    try:
        args = json.loads(args_str) if isinstance(args_str, str) else args_str
    except Exception:
        args = {"content": str(args_str)}
    return dict_to_xml_element(args, name)


def _normalize_messages(messages: List[Dict], request_id: str) -> List[Dict]:
    """
    Normalise incoming messages so the model sees consistent XML tool calls in history:

    - Content arrays (OpenAI/Anthropic multi-block format) → flat string
      - text blocks      → plain text
      - tool_use blocks  → XML  (assistant called a tool)
      - tool_result blocks → plain text  (result of a tool execution)
    - tool_calls field on assistant messages → XML in content

    This means the model reads its own past tool calls as XML and learns to
    keep producing XML in subsequent turns.
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
                    inner = item.get("content", "")
                    if isinstance(inner, list):
                        for sub in inner:
                            if isinstance(sub, dict) and sub.get("type") == "text":
                                parts.append(sub.get("text", ""))
                    else:
                        parts.append(str(inner))
                elif "text" in item:
                    parts.append(item["text"])
            msg["content"] = "\n".join(p for p in parts if p)

        # tool_calls field on assistant → XML appended to content
        tool_calls = msg.get("tool_calls")
        if msg.get("role") == "assistant" and tool_calls:
            xml_parts = []
            for tc in tool_calls:
                name = tc.get("function", {}).get("name", "unknown")
                args = tc.get("function", {}).get("arguments", "{}")
                xml_parts.append(_tool_call_to_xml(name, args))
            msg["content"] = "\n".join(xml_parts)
            logger.debug(f"[{request_id}] Normalised {len(tool_calls)} tool_calls → XML")

        normalized.append(msg)
    return normalized


# ---------------------------------------------------------------------------
# Single-turn priming
# ---------------------------------------------------------------------------


def _make_prime_xml(tool_name: str, params: Dict, required: List[str]) -> str:
    """Build a plausible single-line XML example call for a tool."""
    if "path" in params:
        return f"<{tool_name}>\n<path>README.md</path>\n</{tool_name}>"
    if "content" in params and "path" in params:
        return f"<{tool_name}>\n<path>README.md</path>\n<content>example content</content>\n</{tool_name}>"
    param = required[0] if required else (list(params.keys())[0] if params else "input")
    return f"<{tool_name}>\n<{param}>example</{param}>\n</{tool_name}>"


def _inject_priming(messages: List[Dict], tools: List[Dict]) -> List[Dict]:
    """
    Inject up to 3 synthetic (question, XML-answer) pairs before the first user
    message when the conversation has only 1 user turn.

    Three examples cover all major tool categories the model will encounter:
      1. read_file   — shows a read/query tool call
      2. write_to_file — shows a write/create tool call
      3. attempt_completion — shows how to finish

    More examples = stronger XML-format prior, suppressing the model's own
    chat-template format leak ([assistant to=...] / <assistant to=...>).

    Roo Code is always multi-turn after turn 1, so this mainly helps for
    direct API tests and tools like opencode on first contact.
    """
    user_msgs = [m for m in messages if m.get("role") == "user"]
    if len(user_msgs) != 1 or not tools:
        return messages

    tool_map: Dict[str, Dict] = {}
    for t in tools:
        fn = t.get("function", {})
        n = fn.get("name", "")
        if n:
            tool_map[n] = fn

    # Select up to 3 representative tools in priority order
    candidates = []
    for preferred in ("read_file", "write_to_file", "attempt_completion"):
        if preferred in tool_map:
            candidates.append(preferred)
    # Fill remaining slots from whatever tools are available
    for name in tool_map:
        if name not in candidates:
            candidates.append(name)
        if len(candidates) >= 3:
            break

    prime_qa = [
        ("What files are in this project?", "read_file"),
        ("Please write the implementation.", "write_to_file"),
        ("Are you done?", "attempt_completion"),
    ]

    priming: List[Dict] = []
    for (question, _), tool_name in zip(prime_qa, candidates):
        fn = tool_map[tool_name]
        params = fn.get("parameters", {}).get("properties", {})
        required = fn.get("parameters", {}).get("required", [])
        xml = _make_prime_xml(tool_name, params, required)
        priming.append({"role": "user", "content": question})
        priming.append({"role": "assistant", "content": xml})

    # Insert before the first user message (after system prompt)
    result: List[Dict] = []
    inserted = False
    for msg in messages:
        if msg.get("role") == "user" and not inserted:
            result.extend(priming)
            inserted = True
        result.append(msg)
    return result


# ---------------------------------------------------------------------------
# Text-synthesis helpers
# ---------------------------------------------------------------------------

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
    """Return True if write_to_file was already called for this path in recent history."""
    recent = messages[-lookback:] if len(messages) > lookback else messages
    consecutive_writes = 0

    for msg in recent:
        if msg.get("role") != "assistant":
            continue
        tool_calls = msg.get("tool_calls") or []
        content = msg.get("content") or ""

        if "read_file" in {tc.get("function", {}).get("name", "") for tc in tool_calls}:
            consecutive_writes = 0
            continue

        _WRITE_TOOLS = {"write_to_file", "append_to_file"}
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
            consecutive_writes += 1

        if "write_to_file" in content or "append_to_file" in content:
            if f'"path": "{path}"' in content or f'<path>{path}</path>' in content:
                return True
            consecutive_writes += 1

    return consecutive_writes >= 2


def _synthesize_tool_call_from_text(
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
            r"<write(?:_to)?_file\b[^>]*>\s*<path>(.*?)</path>\s*<content>([\s\S]+?)(?:</content>|$)",
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


# ---------------------------------------------------------------------------
# apply_diff → write_to_file conversion
# ---------------------------------------------------------------------------


def _convert_new_file_diffs(
    tool_calls: List[Dict], tool_names: List[str], request_id: str
) -> List[Dict]:
    """
    Convert apply_diff calls that contain only additions (+lines) into write_to_file.

    When the model uses apply_diff with only + lines on a non-existent file it is
    actually trying to create that file. Roo Code's apply_diff cannot handle this
    case — write_to_file is the correct tool.

    A diff is treated as "new file creation" when every non-empty, non-header line
    starts with '+' (no context lines, no removals).
    """
    if "write_to_file" not in tool_names:
        return tool_calls

    result = []
    for tc in tool_calls:
        func = tc.get("function", {})
        if func.get("name") != "apply_diff":
            result.append(tc)
            continue

        try:
            args = json.loads(func.get("arguments", "{}"))
        except json.JSONDecodeError:
            result.append(tc)
            continue

        diff = args.get("diff", "")
        path = args.get("path", "")
        if not diff or not path:
            result.append(tc)
            continue

        content_lines: List[str] = []
        is_new_file = True
        for line in diff.split("\n"):
            if line.startswith("@@") or line == "":
                continue  # hunk headers and blank lines are fine
            if line.startswith("+"):
                content_lines.append(line[1:])  # strip leading '+'
            else:
                is_new_file = False  # context line or removal → existing file
                break

        if is_new_file and content_lines:
            content = "\n".join(content_lines)
            logger.info(
                f"[{request_id}] apply_diff all-additions on {path!r} → write_to_file"
            )
            result.append({
                "id": tc.get("id", f"call_{request_id}_write_to_file"),
                "type": "function",
                "function": {
                    "name": "write_to_file",
                    "arguments": json.dumps({"path": path, "content": content}),
                },
            })
        else:
            result.append(tc)

    return result


# ---------------------------------------------------------------------------
# Loop detection
# ---------------------------------------------------------------------------

_WRITE_SUCCESS_WORDS = ("successfully", "appended", "created", "written", "modified", "updated")


def detect_success_loop(messages: List[Dict], request_id: str = "") -> Optional[str]:
    """
    Detect when write-type tools keep succeeding but the model keeps calling them again.

    After normalization, Roo Code tool results arrive as role:user messages starting
    with '[Tool Result]'. If we see 2+ such success results in recent history for
    write-type operations WITHOUT an intervening genuine user instruction, the model
    is looping — inject a stop hint.

    The counter resets at each genuine user instruction (non-[Tool Result] message)
    so that successful writes from a prior completed task don't block new edits.
    """
    recent = messages[-12:] if len(messages) > 12 else messages
    success_count = 0

    for msg in recent:
        if msg.get("role") != "user":
            continue
        content = str(msg.get("content", ""))
        if "[Tool Result]" not in content:
            # Genuine user instruction → new task boundary, reset counter
            logger.debug(f"[{request_id}] success_loop: reset at genuine user message")
            success_count = 0
            continue
        lower = content.lower()
        if any(word in lower for word in _WRITE_SUCCESS_WORDS):
            success_count += 1
            logger.debug(f"[{request_id}] success_loop: count={success_count}")

    if success_count < 2:
        return None

    logger.warning(
        f"[{request_id}] SUCCESS LOOP: {success_count} consecutive successful "
        f"write operations — injecting stop hint"
    )
    return (
        f"STOP: The file operation has already succeeded {success_count} times. "
        "Do NOT repeat the tool call — you are creating duplicates. "
        "Call attempt_completion to report that the task is done."
    )


# ---------------------------------------------------------------------------
# XML tool name aliasing
# ---------------------------------------------------------------------------


def _resolve_xml_tool_name(name: str, tool_names: List[str], request_id: str) -> str:
    """Map a hallucinated XML tool name to the correct one if possible."""
    if name in tool_names:
        return name
    alias = _TOOL_NAME_ALIASES.get(name)
    if alias and alias in tool_names:
        logger.info(f"[{request_id}] XML alias: {name!r} → {alias!r}")
        return alias
    return name  # pass through unchanged; caller decides what to do


# ---------------------------------------------------------------------------
# JSON fallback cascade
# ---------------------------------------------------------------------------


def _parse_json_fallback(
    content: str, tools: List[Dict], tool_names: List[str], request_id: str
) -> Optional[List[Dict]]:
    """
    JSON parsing cascade — used when XML parsing finds nothing.
    Handles all the non-standard formats the OCI model occasionally falls back to.
    """
    # [Tool Call: name]\n{args}
    if "[Tool Call:" in content:
        m = re.search(r"\[Tool Call:\s*(\w+)\]\s*\n([\s\S]+)", content)
        if m:
            name, raw = m.group(1).strip(), m.group(2).strip()
            args = safe_parse_json(raw)
            if name in tool_names and args is not None:
                logger.info(f"[{request_id}] JSON fallback: [Tool Call:] → {name}")
                return [
                    {
                        "id": f"call_{request_id}_{name}",
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(args)},
                    }
                ]

    # ### TOOL_CALL marker
    if "### TOOL_CALL" in content:
        json_part = content.split("### TOOL_CALL", 1)[1].strip()
        parsed = safe_parse_json(json_part)
        if parsed and "tool_calls" in parsed:
            logger.info(f"[{request_id}] JSON fallback: ### TOOL_CALL marker")
            return parsed["tool_calls"]

    # Bare JSON object
    stripped = content.strip()
    if stripped.startswith("{"):
        parsed = safe_parse_json(stripped)
        if parsed:
            named = _try_json_tool_call(parsed, tool_names, tools, request_id)
            if named:
                return named
            matched = _score_args(parsed, tools)
            if matched:
                logger.info(f"[{request_id}] JSON fallback: raw JSON → {matched}")
                return [
                    {
                        "id": f"call_{request_id}_{matched}",
                        "type": "function",
                        "function": {"name": matched, "arguments": json.dumps(parsed)},
                    }
                ]

    # JSON in code blocks
    if "```" in content:
        for lang, block in re.findall(r"```(\w*)\s*\n(.*?)\n```", content, re.DOTALL):
            block = block.strip()
            if not block.startswith("{"):
                continue
            if lang in tool_names:
                logger.info(f"[{request_id}] JSON fallback: code block lang tag → {lang}")
                return [
                    {
                        "id": f"call_{request_id}_{lang}",
                        "type": "function",
                        "function": {"name": lang, "arguments": block},
                    }
                ]
            parsed = safe_parse_json(block)
            if not parsed:
                continue
            named = _try_json_tool_call(parsed, tool_names, tools, request_id)
            if named:
                return named
            matched = _score_args(parsed, tools)
            if matched:
                logger.info(f"[{request_id}] JSON fallback: code block scoring → {matched}")
                return [
                    {
                        "id": f"call_{request_id}_{matched}",
                        "type": "function",
                        "function": {"name": matched, "arguments": json.dumps(parsed)},
                    }
                ]

    return None


# ---------------------------------------------------------------------------
# Main endpoint
# ---------------------------------------------------------------------------


@app.post("/v1/chat/completions")
async def chat_completions(request: Dict[str, Any]) -> Dict[str, Any]:
    request_id = str(uuid.uuid4())[:8]
    request_tools = request.get("tools", [])
    tool_names = extract_tool_names_from_request(request_tools)

    logger.info(
        f"[{request_id}] model={request.get('model', 'N/A')} "
        f"messages={len(request.get('messages', []))} tools={len(request_tools)}"
    )
    if settings.log_level == "DEBUG":
        logger.debug(f"[{request_id}] Full request: {json.dumps(request, indent=2, default=str)}")

    messages: List[Dict] = request.get("messages", [])

    # 1. Normalise history: flatten arrays, tool_use/tool_calls → XML
    messages = _normalize_messages(messages, request_id)

    # 2. Build & inject XML system prompt
    existing_system: Optional[str] = None
    if messages and messages[0].get("role") == "system":
        existing_system = messages[0]["content"]
        messages = messages[1:]

    if request_tools:
        xml_system = build_xml_system_prompt(request_tools, existing_system)
        messages = [{"role": "system", "content": xml_system}] + messages
    elif existing_system:
        messages = [{"role": "system", "content": existing_system}] + messages

    # 3. Inject priming pair for single-turn conversations
    if request_tools:
        messages = _inject_priming(messages, request_tools)

    # 3b. Loop detection — append stop hint to last user message if model is looping
    loop_hint = detect_success_loop(messages, request_id)
    if loop_hint:
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                messages[i] = dict(messages[i])
                messages[i]["content"] = str(messages[i].get("content", "")) + f"\n\n[CORRECTION] {loop_hint}"
                break

    # 4. Call upstream LLM (no native tools forwarded — model doesn't support them)
    # OCI default is 2048 tokens — too small for large file writes. Use 8192 as minimum.
    # Model stops naturally when done (~3-10k tokens measured); 8192 covers the natural range.
    # (finish_reason is always "stop" from OCI even when truncated, so we can't detect it.)
    response = await upstream_client.chat_completion(
        messages=messages,
        temperature=request.get("temperature", 0.7),
        max_tokens=request.get("max_tokens") or 8192,
    )

    assistant_message = response["choices"][0]["message"]
    content: str = assistant_message.get("content") or ""
    tool_calls: Optional[List[Dict]] = assistant_message.get("tool_calls")

    # Log raw model output so wrong/unexpected content is visible in logs
    if content:
        preview = content[:300].replace("\n", "\\n")
        logger.info(
            f"[{request_id}] model output ({len(content)} chars): {preview}"
            f"{'…' if len(content) > 300 else ''}"
        )
        logger.debug(f"[{request_id}] model output full:\n{content}")

    # 5. PRIMARY: XML parsing
    if not tool_calls and content and tool_names:
        xml_calls, _ = extract_xml_tool_calls(content, tool_names, request_id)

        # Also try aliased names (write_file → write_to_file, etc.)
        if not xml_calls:
            aliased_names = list(tool_names)
            for hallucinated, real in _TOOL_NAME_ALIASES.items():
                if real in tool_names and hallucinated not in aliased_names:
                    aliased_names.append(hallucinated)
            if len(aliased_names) > len(tool_names):
                xml_calls, _ = extract_xml_tool_calls(content, aliased_names, request_id)

        if xml_calls:
            for xc in xml_calls:
                xc.name = _resolve_xml_tool_name(xc.name, tool_names, request_id)
            tool_calls = convert_xml_tool_calls_to_openai_format(xml_calls)
            assistant_message["tool_calls"] = tool_calls
            assistant_message["content"] = ""
            logger.info(f"[{request_id}] XML parsed {len(tool_calls)} tool call(s)")

    # 6. FALLBACK: JSON cascade
    if not tool_calls and content:
        tool_calls = _parse_json_fallback(content, request_tools, tool_names, request_id)
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls
            assistant_message["content"] = ""

    # 7. TEXT-SYNTHESIS: convert prose response to write_to_file / attempt_completion
    if not tool_calls and content and request_tools:
        tool_calls = _synthesize_tool_call_from_text(content, messages, request_tools, request_id)
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls
            assistant_message["content"] = ""

    # 8. NORMALIZE: apply_diff with only additions → write_to_file (model tries to create new files)
    if tool_calls:
        tool_calls = _convert_new_file_diffs(tool_calls, tool_names, request_id)
        assistant_message["tool_calls"] = tool_calls or None

    # 9. LIMIT: only return the first tool call — model often batches multiple calls but
    #    Roo Code handles one tool at a time reliably
    if tool_calls and len(tool_calls) > 1:
        logger.info(
            f"[{request_id}] {len(tool_calls)} tool calls → keeping only first "
            f"({tool_calls[0]['function']['name']!r})"
        )
        tool_calls = tool_calls[:1]
        assistant_message["tool_calls"] = tool_calls

    if not tool_calls:
        logger.info(f"[{request_id}] No tool calls found — returning text response")

    finish_reason = "tool_calls" if tool_calls else response["choices"][0].get("finish_reason", "stop")

    return {
        "id": response.get("id", f"chatcmpl-{request_id}"),
        "object": "chat.completion",
        "created": response.get("created", 0),
        "model": request.get("model", settings.upstream_model),
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": assistant_message.get("content", ""),
                    "tool_calls": tool_calls,
                },
                "finish_reason": finish_reason,
            }
        ],
        "usage": response.get("usage", {}),
    }

"""
toolproxy — XML Tool Call Proxy

Translates OpenAI native tool_calls API ↔ XML format for models that do not
support native function calls. Sits between any OpenAI-compatible client and
any OpenAI-compatible upstream LLM (e.g. LiteLLM, vLLM, custom endpoints).

Flow:
  Client (Roo Code, Cline, opencode, etc.)
    ↓  OpenAI JSON with tools[] array
  toolproxy (Port 8007)
    ↓  XML System Prompt + XML-normalised history
  Upstream LLM (OpenAI-compatible endpoint)
    ↓  XML tool call in response
  toolproxy — parses XML → OpenAI tool_calls
    ↑  Standard OpenAI response
  Client
"""
import datetime
import json
import logging
import re
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse

from app.config import settings
from app.services.file_write_guard import guard_write_to_file
from app.services.loop_detection import detect_ask_followup_loop, detect_repetitive_tool_loop, detect_success_loop
from app.services.message_normalizer import normalize_messages
from app.services.priming import inject_priming
from app.services.text_synthesis import synthesize_tool_call_from_text
from app.services.tool_call_fixups import (
    convert_move_file_to_execute_command,
    convert_new_file_diffs,
    fix_ask_followup_question_params,
    parse_json_fallback,
    rescue_xml_in_attempt_completion,
    validate_apply_diff_completeness,
)
from app.services.tool_mapping import (
    ClientType,
    _OPEN_CODE_TO_CANONICAL,
    _TOOL_NAME_ALIASES,
    _ROO_CODE_BUILTIN_TOOLS,
    _canonicalize_tools,
    _decanonicalize_tool_calls,
    _remap_args_to_schema,
    _resolve_xml_tool_name,
    detect_client_type,
)
from app.services.vllm_client import VLLMClient
from app.services.xml_parser import (
    convert_xml_tool_calls_to_openai_format,
    extract_tool_names_from_request,
    extract_xml_tool_calls,
)
from app.services.audit_log import extract_file_info, setup_audit_logger, write_audit
from app.services.xml_prompt_builder import build_xml_system_prompt

VERSION = "1.6.13"  # text_synthesis: rescue truncated apply_diff XML (missing closing tag)

# Chat-Template-Leak artifacts produced by gpt-oss models — strip these from preamble text
# but preserve any legitimate reasoning the model outputs before the XML tool call.
_LEAK_PATTERN = re.compile(
    r"\[assistant\s+to=\S+[^\]]*\]"         # [assistant to=write_to_file code<|message|>...]
    r"|<assistant\s+to=\S+[^>]*>"           # <assistant to=write_to_file code>...
    r"|(?:\[Awaiting tool results?\]\s*)+"  # model hallucinating pending tool execution
    ,
    re.IGNORECASE,
)

_TRUNCATION_RE = re.compile(r"\btruncated\b", re.IGNORECASE)
_TRUNCATION_REMINDER = (
    "[REMINDER] The previous file read was truncated. "
    "You MAY call read_file again with offset= to read the next page. "
    "You MAY call write_to_file directly if you know the correct content. "
    "NEVER call apply_diff on a truncated file — you do not have the full content. "
    "NEVER write '[Tool Result]' anywhere in your response — that is SYSTEM output only."
)


def _inject_truncation_reminder(messages: List[Dict], request_id: str) -> List[Dict]:
    """
    If the last user message contains Roo Code's file-truncation notice, append a
    reminder so the model doesn't hallucinate [Tool Result] blocks.
    """
    last_user = next((m for m in reversed(messages) if m.get("role") == "user"), None)
    if last_user is None:
        return messages

    content = last_user.get("content") or ""
    if isinstance(content, list):
        # Scan all string values in content blocks (covers both "text" and "content" keys)
        parts: List[str] = []
        for part in content:
            if isinstance(part, dict):
                for v in part.values():
                    if isinstance(v, str):
                        parts.append(v)
        content = " ".join(parts)

    if not _TRUNCATION_RE.search(content):
        return messages

    logger.info(f"[{request_id}] Truncated file in last user message — injecting reminder")
    result = list(messages)
    for i in range(len(result) - 1, -1, -1):
        if result[i].get("role") == "user":
            result[i] = dict(result[i])
            existing = result[i].get("content") or ""
            if isinstance(existing, list):
                existing = list(existing) + [{"type": "text", "text": _TRUNCATION_REMINDER}]
                result[i]["content"] = existing
            else:
                result[i]["content"] = str(existing) + f"\n\n{_TRUNCATION_REMINDER}"
            break
    return result


logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _write_audit_entry(entry: Dict[str, Any]) -> None:
    """Fire-and-forget wrapper so audit errors never bubble up to the client."""
    try:
        write_audit(entry)
    except Exception:
        pass


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
    setup_audit_logger(settings.audit_log_path)
    if settings.audit_log_path:
        logger.info(f"Audit log: {settings.audit_log_path}")
    upstream_client = VLLMClient(
        base_url=settings.upstream_url,
        model=settings.upstream_model,
        api_key=settings.upstream_api_key,
        timeout=settings.request_timeout,
        max_retries=settings.max_retries,
        retry_on_timeout=settings.retry_on_timeout,
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
        "version": VERSION,
        "upstream": settings.upstream_url,
        "model": settings.upstream_model,
    }


# ---------------------------------------------------------------------------
# SSE streaming helpers
# ---------------------------------------------------------------------------


def _sse_chunk(data: Dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


async def _stream_response(
    response_id: str,
    created: int,
    model: str,
    message: Dict[str, Any],
    finish_reason: str,
    usage: Dict,
) -> AsyncIterator[str]:
    """
    Emit the fully-processed response as SSE chunks.

    We always process the complete response first (XML parsing, synthesis, etc.),
    then fake-stream it — necessary because toolproxy needs the full model output
    to do XML→tool_call conversion before the client sees anything.
    """
    base = {"id": response_id, "object": "chat.completion.chunk", "created": created, "model": model}
    tool_calls = message.get("tool_calls")
    content = message.get("content", "") or ""

    # Chunk 1: role
    yield _sse_chunk({**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": None}, "finish_reason": None}]})

    if tool_calls:
        # Chunk 2: tool call header (name + id, no arguments yet)
        tc = tool_calls[0]
        yield _sse_chunk({**base, "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": tc["id"], "type": "function", "function": {"name": tc["function"]["name"], "arguments": ""}}]}, "finish_reason": None}]})
        # Chunk 3: arguments
        args_str = tc["function"].get("arguments", "")
        if args_str:
            yield _sse_chunk({**base, "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": args_str}}]}, "finish_reason": None}]})
        # Chunk 4: finish
        yield _sse_chunk({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}], "usage": usage})
    else:
        # Text response — send content as one chunk
        if content:
            yield _sse_chunk({**base, "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}]})
        yield _sse_chunk({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}], "usage": usage})

    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Main endpoint
# ---------------------------------------------------------------------------


@app.post("/v1/chat/completions")
async def chat_completions(request: Dict[str, Any]):
    request_id = str(uuid.uuid4())[:8]
    stream = bool(request.get("stream", False))
    request_tools = request.get("tools", [])
    tool_names = extract_tool_names_from_request(request_tools)
    client_type = detect_client_type(tool_names)
    _audit: Dict[str, Any] = {
        "req": request_id,
        "client": client_type.value,
        "tools": len(request_tools),
        "msgs": len(request.get("messages", [])),
    }

    # Translate client-specific tool names to canonical (ROO_CODE-style) names.
    # The model always speaks the canonical XML language; we translate back on the way out.
    canonical_tools = _canonicalize_tools(request_tools, client_type)
    tool_names = [t.get("function", {}).get("name", "") for t in canonical_tools if t.get("function")]

    logger.info(
        f"[{request_id}] model={request.get('model', 'N/A')} "
        f"messages={len(request.get('messages', []))} tools={len(request_tools)} "
        f"client={client_type.value}"
    )
    if settings.log_level == "DEBUG":
        logger.debug(f"[{request_id}] Full request: {json.dumps(request, indent=2, default=str)}")

    messages: List[Dict] = request.get("messages", [])

    # 1. Normalise history: flatten arrays, tool_use/tool_calls → XML
    # Pass the canonical map for OpenCode so client names (write/read) in history
    # are translated to canonical names (write_to_file/read_file) before XML rendering.
    _canon_map = _OPEN_CODE_TO_CANONICAL if client_type == ClientType.OPEN_CODE else None
    messages = normalize_messages(messages, request_id, canonical_map=_canon_map)

    # 2. Build & inject XML system prompt
    existing_system: Optional[str] = None
    if messages and messages[0].get("role") == "system":
        existing_system = messages[0]["content"]
        messages = messages[1:]

    if canonical_tools:
        xml_system = build_xml_system_prompt(canonical_tools, existing_system, client_type)
        messages = [{"role": "system", "content": xml_system}] + messages
    elif existing_system:
        messages = [{"role": "system", "content": existing_system}] + messages

    # 3. Inject priming pair for single-turn conversations
    if canonical_tools:
        messages = inject_priming(messages, canonical_tools, client_type)

    # 3b. Loop detection — append stop hint to last user message if model is looping.
    # detect_repetitive_tool_loop runs FIRST: it fires at threshold=3 and gives a more
    # accurate hint ("not making progress, try a different approach") for cases like
    # apply_diff called on the same file 3× in a row.  detect_success_loop fires at
    # threshold=2 with a broader hint; it acts as a fallback for write-type loops that
    # don't yet reach the repetitive threshold.
    # detect_ask_followup_loop uses a frequency window (not consecutive) because Roo Code
    # sends a genuine user message after each answer, which would reset a consecutive counter.
    loop_hint = (
        detect_repetitive_tool_loop(messages, request_id, client_type)
        or detect_success_loop(messages, request_id, client_type)
        or detect_ask_followup_loop(messages, request_id, client_type)
    )
    if loop_hint:
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                messages[i] = dict(messages[i])
                messages[i]["content"] = str(messages[i].get("content", "")) + f"\n\n[CORRECTION] {loop_hint}"
                break

    # 3c. Truncation reminder — when last user message contains Roo Code's truncation notice,
    #     remind the model it may use read_file offset= or write_to_file but must NOT hallucinate [Tool Result]
    messages = _inject_truncation_reminder(messages, request_id)

    # 4. Call upstream LLM (no native tools forwarded — model doesn't support them)
    # Default to 8192 tokens if not specified — many models have low defaults (e.g. 2048)
    # which is too small for large file writes. Model stops naturally when done.
    try:
        _t0 = time.monotonic()
        response = await upstream_client.chat_completion(
            messages=messages,
            temperature=request.get("temperature", 0.7),
            max_tokens=request.get("max_tokens") or 8192,
        )
        _audit["upstream_ms"] = int((time.monotonic() - _t0) * 1000)
    except Exception as e:
        _audit["upstream_ms"] = int((time.monotonic() - _t0) * 1000)
        _audit["mechanism"] = "upstream_error"
        _audit["error"] = type(e).__name__
        _write_audit_entry(_audit)

        # httpx.HTTPStatusError: str(e) is just "Client error '400 Bad Request' for url '...'"
        # — the response body (which contains the OCI error detail) is in e.response.text.
        err_str = str(e)
        response_body = ""
        if hasattr(e, "response"):
            try:
                response_body = e.response.text or ""
            except Exception:
                pass
        combined = (err_str + " " + response_body).lower()
        is_content_filter = "inappropriate content" in combined
        is_empty_response = "oci model returned an empty response" in combined

        if is_empty_response:
            logger.error(
                f"[{request_id}] OCI silent rate-limiting / empty response (500). "
                f"OCI returned HTTP 200 with empty body — transient, retry the request."
            )
            return JSONResponse(
                status_code=503,
                content={"error": {
                    "message": "OCI model returned an empty response (silent rate-limiting). "
                               "This is transient — please retry the request.",
                    "type": "rate_limit_error",
                }},
            )

        if is_content_filter:
            # Log the last user message to help diagnose what triggered the filter
            last_user = next(
                (m for m in reversed(messages) if m.get("role") == "user"), None
            )
            if last_user:
                content = str(last_user.get("content", ""))
                logger.info(
                    f"[{request_id}] Content-filter trigger — last user message (first 500 chars): "
                    f"{content[:500]!r}"
                )
            logger.error(
                f"[{request_id}] OCI content filter blocked request (400). "
                f"See INFO log above for triggering content."
            )
            return JSONResponse(
                status_code=400,
                content={"error": {
                    "message": "Request blocked by OCI content filter: Inappropriate content detected. "
                               "The conversation may contain content that triggers Oracle's safety filter "
                               "(e.g. security-related code, certain keywords). "
                               "Try rephrasing or splitting the request.",
                    "type": "content_filter_error",
                }},
            )

        logger.error(f"[{request_id}] Upstream error: {e}")
        return JSONResponse(
            status_code=502,
            content={"error": {"message": f"Upstream LLM error: {e}", "type": "upstream_error"}},
        )

    assistant_message = response["choices"][0]["message"]
    content: str = assistant_message.get("content") or ""
    tool_calls: Optional[List[Dict]] = assistant_message.get("tool_calls")
    _audit["response_chars"] = len(content)

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
        xml_calls, remaining = extract_xml_tool_calls(content, tool_names, request_id)

        # Also try aliased names (write_file → write_to_file, etc.)
        if not xml_calls:
            aliased_names = list(tool_names)
            for hallucinated, real in _TOOL_NAME_ALIASES.items():
                if (real in tool_names or real in _ROO_CODE_BUILTIN_TOOLS) and hallucinated not in aliased_names:
                    aliased_names.append(hallucinated)
            if len(aliased_names) > len(tool_names):
                xml_calls, remaining = extract_xml_tool_calls(content, aliased_names, request_id)

        # Also try Roo Code built-in tools not always present in the tools[] array
        # (e.g. delete_file, rename_file — valid Roo tools the model may call)
        if not xml_calls and client_type in (ClientType.ROO_CODE, ClientType.CLINE):
            extra = [n for n in _ROO_CODE_BUILTIN_TOOLS if n not in tool_names]
            if extra:
                xml_calls, remaining = extract_xml_tool_calls(content, extra, request_id)

        if xml_calls:
            for xc in xml_calls:
                xc.name = _resolve_xml_tool_name(xc.name, tool_names, request_id)
            tool_calls = convert_xml_tool_calls_to_openai_format(xml_calls)
            assistant_message["tool_calls"] = tool_calls
            # Keep any legitimate preamble text; only strip Chat-Template-Leak artifacts.
            # `remaining` is the original content with the XML block(s) already removed.
            preamble = _LEAK_PATTERN.sub("", remaining).strip()
            if preamble:
                logger.debug(f"[{request_id}] Preserving preamble text ({len(preamble)} chars)")
            assistant_message["content"] = preamble
            _audit["mechanism"] = "xml_parsed"
            _audit["preamble_len"] = len(preamble)
            names = ", ".join(tc["function"]["name"] for tc in tool_calls)
            logger.info(f"[{request_id}] XML parsed {len(tool_calls)} tool call(s): {names}")

    # 5b. FIXUP: move_file / rename_file → execute_command(mv ...)
    #     These are Roo builtins not in tools[] — Roo Code silently drops them.
    if tool_calls:
        tool_calls = convert_move_file_to_execute_command(tool_calls, tool_names, request_id)
        assistant_message["tool_calls"] = tool_calls

    # 6. FALLBACK: JSON cascade
    if not tool_calls and content:
        tool_calls = parse_json_fallback(content, canonical_tools, tool_names, request_id)
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls
            assistant_message["content"] = ""
            _audit["mechanism"] = "json_fallback"

    # 7. TEXT-SYNTHESIS: convert prose response to write_to_file / attempt_completion
    if not tool_calls and content and canonical_tools:
        tool_calls = synthesize_tool_call_from_text(content, messages, canonical_tools, request_id)
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls
            assistant_message["content"] = ""
            _audit["mechanism"] = "text_synthesis"

    # 7b. EMERGENCY FALLBACK: upstream returned empty content and no tool calls.
    #     This happens when the upstream LLM (e.g. OCI) silently returns null/empty
    #     for a specific prompt instead of a valid response.  Without this fallback
    #     the proxy would return an empty assistant message and clients like Roo Code
    #     throw "The language model did not provide any assistant messages".
    if not tool_calls and not content and canonical_tools:
        tool_names_set = {t.get("function", {}).get("name", "") for t in canonical_tools if "function" in t}
        if "attempt_completion" in tool_names_set:
            logger.warning(f"[{request_id}] Empty model response — synthesizing attempt_completion fallback")
            tool_calls = [{
                "id": f"call_{request_id}_fallback",
                "type": "function",
                "function": {
                    "name": "attempt_completion",
                    "arguments": json.dumps({"result": "Task completed."}),
                },
            }]
            assistant_message["tool_calls"] = tool_calls
            assistant_message["content"] = ""
            _audit["mechanism"] = "empty_fallback"

    # 8. SCHEMA REMAP: translate canonical priming param names to actual schema names
    #    (e.g. model outputs <path> but OpenCode's write tool expects filePath)
    if tool_calls:
        tool_calls = _remap_args_to_schema(tool_calls, canonical_tools, request_id)
        assistant_message["tool_calls"] = tool_calls

    # 9. NORMALIZE: apply_diff with only additions → write_to_file (model tries to create new files)
    if tool_calls:
        tool_calls = convert_new_file_diffs(tool_calls, tool_names, request_id)
        assistant_message["tool_calls"] = tool_calls or None

    # 9b. VALIDATE: drop apply_diff / replace_in_file calls with incomplete diffs
    #     (missing >>>>>>> REPLACE marker = truncated or corrupt model output)
    if tool_calls:
        tool_calls = validate_apply_diff_completeness(tool_calls, request_id)
        assistant_message["tool_calls"] = tool_calls or None

    # 10. RESCUE: if attempt_completion.result contains XML tool calls, extract them
    if tool_calls:
        tool_calls = rescue_xml_in_attempt_completion(tool_calls, tool_names, request_id)
        assistant_message["tool_calls"] = tool_calls

    # 10b. WRITE GUARD: intercept markdown docs written into config files (application.yml etc.)
    #      Replaces bad write with ask_followup_question if available; logs warning otherwise.
    if tool_calls:
        _before_guard = tool_calls[0]["function"]["name"] if tool_calls else None
        tool_calls = guard_write_to_file(tool_calls, request_id, canonical_tools)
        assistant_message["tool_calls"] = tool_calls
        if tool_calls and tool_calls[0]["function"]["name"] != _before_guard:
            _audit["mechanism"] = "write_guard"

    # 11. NORMALIZE: ask_followup_question follow_up string → array
    if tool_calls:
        tool_calls = fix_ask_followup_question_params(tool_calls, request_id)
        assistant_message["tool_calls"] = tool_calls

    # 12. LIMIT: only return the first tool call — model often batches multiple calls but
    #     Roo Code handles one tool at a time reliably
    if tool_calls and len(tool_calls) > 1:
        logger.info(
            f"[{request_id}] {len(tool_calls)} tool calls → keeping only first "
            f"({tool_calls[0]['function']['name']!r})"
        )
        tool_calls = tool_calls[:1]
        assistant_message["tool_calls"] = tool_calls

    # 12. DECANONICALIZE: translate canonical tool names back to client-specific names
    if tool_calls:
        tool_calls = _decanonicalize_tool_calls(tool_calls, client_type, request_id)
        assistant_message["tool_calls"] = tool_calls

    if not tool_calls:
        logger.info(f"[{request_id}] No tool calls found — returning text response")
        _audit.setdefault("mechanism", "no_tool_call")

    # Write audit entry with final tool call info (anonymized)
    _audit["ts"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if tool_calls:
        _tc = tool_calls[0]["function"]
        _audit["tool"] = _tc.get("name", "")
        _audit.update(extract_file_info(_tc.get("name", ""), _tc.get("arguments", "")))
    _write_audit_entry(_audit)

    finish_reason = "tool_calls" if tool_calls else response["choices"][0].get("finish_reason", "stop")

    # Omit tool_calls key entirely when empty — some clients (e.g. OpenCode) break on null
    message: Dict[str, Any] = {
        "role": "assistant",
        "content": assistant_message.get("content", ""),
    }
    if tool_calls:
        message["tool_calls"] = tool_calls

    response_id = response.get("id", f"chatcmpl-{request_id}")
    created = response.get("created", 0) or int(time.time())
    model = request.get("model", settings.upstream_model)
    usage = response.get("usage", {})

    if stream:
        return StreamingResponse(
            _stream_response(response_id, created, model, message, finish_reason, usage),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    return {
        "id": response_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": usage,
    }

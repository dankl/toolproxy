"""
toolproxy — XML Tool Call Proxy

Translates OpenAI native tool_calls API ↔ XML format for models that do not
support native function calls. Sits between any OpenAI-compatible client and
any OpenAI-compatible upstream LLM (e.g. LiteLLM, vLLM, custom endpoints).

Flow:
  Client (Roo Code, opencode, etc.)
    ↓  OpenAI JSON with tools[] array
  toolproxy (Port 8007)
    ↓  XML System Prompt + XML-normalised history
  Upstream LLM (OpenAI-compatible endpoint)
    ↓  XML tool call in response
  toolproxy — parses XML → OpenAI tool_calls
    ↑  Standard OpenAI response
  Client
"""
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse

from app.config import settings
from app.services.loop_detection import detect_success_loop
from app.services.message_normalizer import normalize_messages
from app.services.priming import inject_priming
from app.services.text_synthesis import synthesize_tool_call_from_text
from app.services.tool_call_fixups import (
    convert_move_file_to_execute_command,
    convert_new_file_diffs,
    parse_json_fallback,
    rescue_xml_in_attempt_completion,
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
from app.services.xml_prompt_builder import build_xml_system_prompt

VERSION = "1.1.4"  # move_file/rename_file → execute_command(mv); priming uses execute_command

logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


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

    # 3b. Loop detection — append stop hint to last user message if model is looping
    loop_hint = detect_success_loop(messages, request_id, client_type)
    if loop_hint:
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                messages[i] = dict(messages[i])
                messages[i]["content"] = str(messages[i].get("content", "")) + f"\n\n[CORRECTION] {loop_hint}"
                break

    # 4. Call upstream LLM (no native tools forwarded — model doesn't support them)
    # Default to 8192 tokens if not specified — many models have low defaults (e.g. 2048)
    # which is too small for large file writes. Model stops naturally when done.
    try:
        response = await upstream_client.chat_completion(
            messages=messages,
            temperature=request.get("temperature", 0.7),
            max_tokens=request.get("max_tokens") or 8192,
        )
    except Exception as e:
        logger.error(f"[{request_id}] Upstream error after retries: {e}")
        return JSONResponse(
            status_code=502,
            content={"error": {"message": f"Upstream LLM error: {e}", "type": "upstream_error"}},
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
                if (real in tool_names or real in _ROO_CODE_BUILTIN_TOOLS) and hallucinated not in aliased_names:
                    aliased_names.append(hallucinated)
            if len(aliased_names) > len(tool_names):
                xml_calls, _ = extract_xml_tool_calls(content, aliased_names, request_id)

        # Also try Roo Code built-in tools not always present in the tools[] array
        # (e.g. delete_file, rename_file — valid Roo tools the model may call)
        if not xml_calls and client_type == ClientType.ROO_CODE:
            extra = [n for n in _ROO_CODE_BUILTIN_TOOLS if n not in tool_names]
            if extra:
                xml_calls, _ = extract_xml_tool_calls(content, extra, request_id)

        if xml_calls:
            for xc in xml_calls:
                xc.name = _resolve_xml_tool_name(xc.name, tool_names, request_id)
            tool_calls = convert_xml_tool_calls_to_openai_format(xml_calls)
            assistant_message["tool_calls"] = tool_calls
            assistant_message["content"] = ""
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

    # 7. TEXT-SYNTHESIS: convert prose response to write_to_file / attempt_completion
    if not tool_calls and content and canonical_tools:
        tool_calls = synthesize_tool_call_from_text(content, messages, canonical_tools, request_id)
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls
            assistant_message["content"] = ""

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

    # 8. SCHEMA REMAP: translate canonical priming param names to actual schema names
    #    (e.g. model outputs <path> but OpenCode's write tool expects filePath)
    if tool_calls:
        tool_calls = _remap_args_to_schema(tool_calls, canonical_tools, request_id)
        assistant_message["tool_calls"] = tool_calls

    # 9. NORMALIZE: apply_diff with only additions → write_to_file (model tries to create new files)
    if tool_calls:
        tool_calls = convert_new_file_diffs(tool_calls, tool_names, request_id)
        assistant_message["tool_calls"] = tool_calls or None

    # 10. RESCUE: if attempt_completion.result contains XML tool calls, extract them
    if tool_calls:
        tool_calls = rescue_xml_in_attempt_completion(tool_calls, tool_names, request_id)
        assistant_message["tool_calls"] = tool_calls

    # 11. LIMIT: only return the first tool call — model often batches multiple calls but
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

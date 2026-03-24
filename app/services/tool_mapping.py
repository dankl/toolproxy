"""
Client-type detection and tool name mapping.

The model always speaks one canonical XML language (ROO_CODE-style).
Client-specific tool names are translated in/out so the model never
has to deal with per-client naming differences.
"""
import json
import logging
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class ClientType(Enum):
    ROO_CODE = "roo_code"
    OPEN_CODE = "open_code"
    CLINE = "cline"
    GENERIC = "generic"


_ROO_CODE_SIGNALS = frozenset({"attempt_completion", "write_to_file", "read_file", "apply_diff"})
_OPEN_CODE_SIGNALS = frozenset({"bash", "edit", "glob", "grep"})
# Cline uses replace_in_file (not apply_diff) as its edit tool — that's the distinguishing signal
_CLINE_SIGNALS = frozenset({"replace_in_file"})

# Canonical tool name mapping.
# The model always sees and outputs ROO_CODE-style tool names (canonical).
# For OpenCode clients we translate incoming names → canonical before sending to the model,
# then translate canonical names → client names in the response.
# Only tools with equivalent semantics are mapped; edit/bash/glob/grep etc. stay as-is.
_OPEN_CODE_TO_CANONICAL: Dict[str, str] = {
    "write": "write_to_file",
    "read": "read_file",
    "list": "list_files",
    "bash": "execute_command",
}
_CANONICAL_TO_OPEN_CODE: Dict[str, str] = {v: k for k, v in _OPEN_CODE_TO_CANONICAL.items()}

# Schema-aware parameter aliases: model output key → list of candidate schema keys.
# Applied post-parsing when the model uses the canonical priming name (e.g. "path")
# but the actual tool schema uses a different name. The first candidate that exists
# in the schema wins. Only fires when the output key is NOT in the schema.
# Known real-world cases:
# - OpenCode's "write" tool uses "file_path" (current) or "filePath" (older versions)
_SCHEMA_PARAM_ALIASES: Dict[str, List[str]] = {
    "path":      ["file_path", "filePath"],   # model outputs <path>, schema may differ
    "file_path": ["path"],
    "filePath":  ["path"],
}

# Hallucinated tool name → real tool name.
# Also includes OpenCode client names (write/read) so that if the model echoes
# them back from history they are recognized as canonical equivalents.
_TOOL_NAME_ALIASES: Dict[str, str] = {
    "write": "write_to_file",       # OpenCode client name (defensive alias)
    "read": "read_file",            # OpenCode client name (defensive alias)
    "write_file": "write_to_file",
    "create_file": "write_to_file",
    "open_file": "read_file",
    "view_file": "read_file",
    "read_file_content": "read_file",
    "apply_patch": "apply_diff",
    "patch_file": "apply_diff",
    "patch": "apply_diff",              # OpenCode patch tool → apply_diff
    "list_dir": "list_files",
    "ls": "list_files",
    "list_directory": "list_files",
    "search": "search_files",
    "find": "search_files",
    "search_code": "search_files",
    "search_codebase": "codebase_search",   # real Roo Code tool (semantic search)
    "semantic_search": "codebase_search",
    "rename": "move_file",          # hallucinated rename → move_file (Roo Code builtin)
    "move": "move_file",            # hallucinated move → move_file
    "bash": "execute_command",      # OpenCode-style name / hallucination
    "edit_file": "edit",            # hallucinated name → OpenCode edit tool
    "run_command": "execute_command",
    "execute": "execute_command",
    "run": "execute_command",
    "shell": "execute_command",
    "ask_followup": "ask_followup_question",
    "ask_question": "ask_followup_question",
    "ask_user": "ask_followup_question",
    "followup_question": "ask_followup_question",
    "create_task": "new_task",
    "task": "new_task",
    "subtask": "new_task",
    "read_output": "read_command_output",
    "get_output": "read_command_output",
    "command_output": "read_command_output",
    "use_skill": "skill",
    "run_skill": "skill",
    "change_mode": "switch_mode",
    "set_mode": "switch_mode",
    "mode": "switch_mode",
    "update_todos": "update_todo_list",
    "update_todo": "update_todo_list",
    "todo": "update_todo_list",
    "set_todos": "update_todo_list",
    "search_and_replace": "edit",           # Roo Code backward-compat alias for edit
    "search_replace": "edit",
    "replace_in_file": "edit",
    "find_replace": "edit",
}

# Static parameter remapping for builtin tools that are not in the tools[] schema.
# The model may output wrong param names (e.g. old_path/new_path for move_file);
# remap them to the actual Roo Code parameter names.
_BUILTIN_PARAM_REMAP: Dict[str, Dict[str, str]] = {
    "move_file": {"old_path": "source", "new_path": "destination"},
}


# Roo Code built-in tools that may not always appear in the tools[] array sent by
# the client (e.g. delete_file, rename_file) but which the model legitimately calls.
# Used as a fallback in XML parsing so we don't drop these tool calls.
_ROO_CODE_BUILTIN_TOOLS: frozenset = frozenset({
    "delete_file",
    "delete_folder",
    "rename_file",
    "move_file",
    "create_directory",
    "search_and_replace",   # Roo Code backward-compat alias for edit
})


def detect_client_type(tool_names: List[str]) -> ClientType:
    """Identify which agent framework is making the request based on its tool set."""
    names = set(tool_names)
    if names & _OPEN_CODE_SIGNALS:
        return ClientType.OPEN_CODE
    # Cline: has replace_in_file but NOT apply_diff (Roo Code uses apply_diff instead)
    if (names & _CLINE_SIGNALS) and "apply_diff" not in names:
        return ClientType.CLINE
    if names & _ROO_CODE_SIGNALS:
        return ClientType.ROO_CODE
    return ClientType.GENERIC


def _canonicalize_tools(tools: List[Dict], client_type: ClientType) -> List[Dict]:
    """Rename client-specific tool names to canonical (ROO_CODE-style) names.

    The model always speaks the canonical XML language. Client-specific names
    are translated here before the request is sent to the model.
    """
    if client_type != ClientType.OPEN_CODE:
        return tools
    result = []
    for t in tools:
        func = t.get("function", {})
        name = func.get("name", "")
        canonical = _OPEN_CODE_TO_CANONICAL.get(name)
        if canonical:
            t = {**t, "function": {**func, "name": canonical}}
        result.append(t)
    return result


def _decanonicalize_tool_calls(
    tool_calls: List[Dict], client_type: ClientType, request_id: str
) -> List[Dict]:
    """Translate canonical tool names back to client-specific names before returning."""
    if client_type != ClientType.OPEN_CODE:
        return tool_calls
    result = []
    for tc in tool_calls:
        func = tc.get("function", {})
        name = func.get("name", "")
        client_name = _CANONICAL_TO_OPEN_CODE.get(name)
        if client_name:
            logger.debug(f"[{request_id}] Decanonicalize: {name!r} → {client_name!r}")
            func = {**func, "name": client_name}
            tc = {**tc, "function": func}
        # OpenCode's bash tool requires a "description" field — add it if missing
        if func.get("name") == "bash":
            try:
                args = json.loads(func.get("arguments", "{}"))
            except (json.JSONDecodeError, TypeError):
                args = {}
            if "description" not in args:
                cmd = args.get("command", "")
                args["description"] = cmd[:120] if cmd else "shell command"
                logger.debug(f"[{request_id}] Added bash description: {args['description']!r}")
                tc = {**tc, "function": {**func, "arguments": json.dumps(args)}}
        result.append(tc)
    return result


def _remap_args_to_schema(
    tool_calls: List[Dict], tools: List[Dict], request_id: str
) -> List[Dict]:
    """
    Rename tool call arguments to match the actual tool schema.

    When the model outputs canonical priming params (e.g. <path>) but the client's
    tool schema uses a different name (e.g. filePath for OpenCode's write tool),
    this function detects the mismatch and renames via _SCHEMA_PARAM_ALIASES.
    """
    tool_map = {
        t.get("function", {}).get("name", ""): t.get("function", {})
        for t in tools if t.get("function", {}).get("name")
    }
    result = []
    for tc in tool_calls:
        func = tc.get("function", {})
        name = func.get("name", "")
        tool_def = tool_map.get(name)
        if not tool_def:
            # Builtin tool (not in tools[] schema) — apply static param remap if available
            static_remap = _BUILTIN_PARAM_REMAP.get(name)
            if static_remap:
                try:
                    args = json.loads(func.get("arguments", "{}"))
                except (json.JSONDecodeError, TypeError):
                    result.append(tc)
                    continue
                remapped = {}
                changed = False
                for key, value in args.items():
                    target = static_remap.get(key)
                    if target:
                        logger.info(f"[{request_id}] Builtin param remap {name}: {key!r} → {target!r}")
                        remapped[target] = value
                        changed = True
                    else:
                        remapped[key] = value
                if changed:
                    tc = {**tc, "function": {**func, "arguments": json.dumps(remapped, ensure_ascii=False)}}
            result.append(tc)
            continue
        schema_params = set(tool_def.get("parameters", {}).get("properties", {}).keys())
        try:
            args = json.loads(func.get("arguments", "{}"))
        except (json.JSONDecodeError, TypeError):
            result.append(tc)
            continue
        remapped = {}
        changed = False
        for key, value in args.items():
            if key not in schema_params:
                candidates = _SCHEMA_PARAM_ALIASES.get(key, [])
                target = next((c for c in candidates if c in schema_params), None)
                if target:
                    logger.info(f"[{request_id}] Schema remap {name}: {key!r} → {target!r}")
                    remapped[target] = value
                    changed = True
                    continue
            remapped[key] = value
        if changed:
            tc = {**tc, "function": {**func, "arguments": json.dumps(remapped, ensure_ascii=False)}}
        result.append(tc)
    return result


def _resolve_xml_tool_name(name: str, tool_names: List[str], request_id: str) -> str:
    """Map a hallucinated XML tool name to the correct one if possible."""
    if name in tool_names:
        return name
    alias = _TOOL_NAME_ALIASES.get(name)
    if alias and (alias in tool_names or alias in _ROO_CODE_BUILTIN_TOOLS):
        logger.info(f"[{request_id}] XML alias: {name!r} → {alias!r}")
        return alias
    return name  # pass through unchanged; caller decides what to do

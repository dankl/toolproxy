"""
Post-parse tool call correction pipeline for toolproxy.

Handles JSON fallback parsing, argument aliasing, hallucinated name recovery,
apply_diff→write_to_file conversion, and attempt_completion XML rescue.
"""
import json
import logging
import re
from typing import Dict, List, Optional

from app.services.json_repair import safe_parse_json
from app.services.tool_mapping import _TOOL_NAME_ALIASES, _resolve_xml_tool_name
from app.services.xml_parser import (
    convert_xml_tool_calls_to_openai_format,
    extract_xml_tool_calls,
)

logger = logging.getLogger(__name__)

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


def parse_json_fallback(
    content: str, tools: List[Dict], tool_names: List[str], request_id: str
) -> Optional[List[Dict]]:
    """
    JSON parsing cascade — used when XML parsing finds nothing.
    Handles all the non-standard formats the model occasionally falls back to.
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


def convert_new_file_diffs(
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


_DIFF_TOOLS = {"apply_diff", "replace_in_file"}

_HUNK_START = "<<<<<<< SEARCH\n"
_HUNK_SEP = "\n=======\n"
_HUNK_END = "\n>>>>>>> REPLACE"


def _deduplicate_diff_hunks(diff: str, request_id: str) -> tuple:
    """Remove duplicate and truncated SEARCH/REPLACE hunks.

    Returns (cleaned_diff, num_dropped).
    Duplicate = identical SEARCH block seen before.
    Truncated = hunk missing >>>>>>> REPLACE (e.g. model hit token limit mid-diff).
    """
    parts = diff.split(_HUNK_START)
    seen: set = set()
    kept: list = []
    dropped = 0

    for part in parts[1:]:
        sep = part.find(_HUNK_SEP)
        end = part.find(_HUNK_END)
        if sep == -1 or end == -1 or sep >= end:
            dropped += 1
            continue
        search_content = part[:sep]
        if search_content in seen:
            dropped += 1
            continue
        seen.add(search_content)
        kept.append(_HUNK_START + part[: end + len(_HUNK_END)] + "\n")

    return "".join(kept), dropped


# Matches a unified diff hunk header.
# Handles both the full form "@@ -1,3 +1,4 @@" and the minimal "@@ " or bare "@@"
# that some models emit as a shorthand.
_UNIFIED_HUNK_RE = re.compile(r"^@@", re.MULTILINE)
# Matches the file header lines produced by git diff / GNU diff
_UNIFIED_HEADER_RE = re.compile(r"^(?:---|\+\+\+|diff --git)\s", re.MULTILINE)


def _convert_unified_diff_to_search_replace(diff: str) -> Optional[str]:
    """
    Convert a unified diff to Roo Code / Cline SEARCH/REPLACE format.

    Supports standard git-style unified diffs:
        @@ -1,3 +1,4 @@
        -old line
        +new line
         context line

    Each hunk becomes one SEARCH/REPLACE block.  Context lines appear in
    both halves.  The file-header lines (---, +++, diff --git) are ignored.

    Returns the converted diff string, or None if the input does not look
    like a unified diff (no @@ hunk header found).
    """
    if not _UNIFIED_HUNK_RE.search(diff):
        return None

    lines = diff.splitlines()
    blocks: List[str] = []
    search_lines: List[str] = []
    replace_lines: List[str] = []
    in_hunk = False

    def _flush():
        if search_lines or replace_lines:
            block = (
                "<<<<<<< SEARCH\n"
                + "\n".join(search_lines)
                + ("\n" if search_lines else "")
                + "=======\n"
                + "\n".join(replace_lines)
                + ("\n" if replace_lines else "")
                + ">>>>>>> REPLACE"
            )
            blocks.append(block)

    for line in lines:
        # Skip file-header lines (---, +++, diff --git ...)
        if _UNIFIED_HEADER_RE.match(line):
            continue
        # Hunk header — start a new block
        if _UNIFIED_HUNK_RE.match(line):
            if in_hunk:
                _flush()
                search_lines = []
                replace_lines = []
            in_hunk = True
            continue
        if not in_hunk:
            continue
        if line.startswith("-"):
            search_lines.append(line[1:])
        elif line.startswith("+"):
            replace_lines.append(line[1:])
        else:
            # Context line — goes into both halves (strip leading space if present)
            ctx = line[1:] if line.startswith(" ") else line
            search_lines.append(ctx)
            replace_lines.append(ctx)

    if in_hunk:
        _flush()

    if not blocks:
        return None

    return "\n".join(blocks)


def validate_apply_diff_completeness(
    tool_calls: List[Dict], request_id: str
) -> List[Dict]:
    """
    Fix or drop apply_diff / replace_in_file calls whose diff is not in the
    expected SEARCH/REPLACE format.

    Strategy (in order):
    1. If the diff already contains '>>>>>>> REPLACE' → pass through unchanged.
    2. If the diff looks like a unified diff (has @@ hunk headers) → convert to
       SEARCH/REPLACE format and pass through with a log line.
    3. Otherwise → drop the call and let the fallback chain handle it.
    """
    result = []
    for tc in tool_calls:
        name = tc.get("function", {}).get("name", "")
        if name not in _DIFF_TOOLS:
            result.append(tc)
            continue

        try:
            args = json.loads(tc["function"].get("arguments", "{}"))
        except (json.JSONDecodeError, KeyError):
            result.append(tc)
            continue

        diff = args.get("diff", "")

        # Already correct format — deduplicate hunks then pass through
        if not diff or ">>>>>>> REPLACE" in diff:
            if diff and _HUNK_START in diff:
                cleaned, dropped = _deduplicate_diff_hunks(diff, request_id)
                if dropped:
                    logger.warning(
                        f"[{request_id}] apply_diff: dropped {dropped} duplicate/truncated "
                        f"hunk(s) for {args.get('path', '?')!r} "
                        f"({len(diff)} → {len(cleaned)} chars)"
                    )
                    args = dict(args)
                    args["diff"] = cleaned
                    tc = dict(tc)
                    tc["function"] = dict(tc["function"])
                    tc["function"]["arguments"] = json.dumps(args)
            result.append(tc)
            continue

        # Try unified diff conversion
        converted = _convert_unified_diff_to_search_replace(diff)
        if converted:
            logger.info(
                f"[{request_id}] Converted unified diff → SEARCH/REPLACE "
                f"for {name} ({len(converted)} chars)"
            )
            args["diff"] = converted
            tc = dict(tc)
            tc["function"] = dict(tc["function"])
            tc["function"]["arguments"] = json.dumps(args)
            result.append(tc)
            continue

        # Not unified diff and no REPLACE marker — drop
        logger.warning(
            f"[{request_id}] Dropping corrupt {name}: "
            "diff is missing '>>>>>>> REPLACE' closing marker"
        )
        # do not append — drop

    return result


def convert_move_file_to_execute_command(
    tool_calls: List[Dict], tool_names: List[str], request_id: str
) -> List[Dict]:
    """
    Convert move_file / rename_file → execute_command(mv source destination).

    move_file and rename_file are Roo Code builtins that are NOT in the tools[]
    array sent to the proxy. Roo Code silently drops tool_calls for unknown tools,
    so we convert them to execute_command which IS in tools[].
    Works for both files and folders since mv handles both.
    """
    if "execute_command" not in tool_names:
        return tool_calls

    result = []
    for tc in tool_calls:
        name = tc.get("function", {}).get("name", "")
        if name in ("move_file", "rename_file"):
            try:
                args = json.loads(tc["function"].get("arguments", "{}"))
            except (json.JSONDecodeError, TypeError):
                result.append(tc)
                continue
            source = args.get("source") or args.get("path") or args.get("old_path") or ""
            dest = args.get("destination") or args.get("new_name") or args.get("new_path") or ""
            if source and dest:
                cmd = f"mv '{source}' '{dest}'"
                logger.info(f"[{request_id}] Convert {name} → execute_command: {cmd!r}")
                tc = {
                    **tc,
                    "function": {
                        "name": "execute_command",
                        "arguments": json.dumps({"command": cmd}),
                    },
                }
            else:
                logger.warning(
                    f"[{request_id}] {name}: could not extract source/destination from {args}"
                )
        result.append(tc)
    return result


def rescue_xml_in_attempt_completion(
    tool_calls: List[Dict],
    tool_names: List[str],
    request_id: str,
) -> List[Dict]:
    """
    Rescue XML tool calls that the model accidentally placed inside
    attempt_completion.result instead of outputting them directly.

    The model sometimes produces:
        <attempt_completion>
          <result>
            <delete_file><path>foo.md</path></delete_file>
            <delete_folder><path>foo</path></delete_folder>
          </result>
        </attempt_completion>

    The XML inside result is shown as plain text to the user — it is never
    executed.  This function detects that pattern and replaces the
    attempt_completion with the first real tool call extracted from result.
    """
    if not tool_calls:
        return tool_calls

    # Only act on a single attempt_completion
    if len(tool_calls) != 1:
        return tool_calls
    tc = tool_calls[0]
    if tc.get("function", {}).get("name") != "attempt_completion":
        return tool_calls

    try:
        args = json.loads(tc["function"].get("arguments", "{}"))
    except json.JSONDecodeError:
        return tool_calls

    result_content = args.get("result", "")
    if not result_content:
        return tool_calls

    # Build alias lookup (hallucinated names → real names)
    aliased_names = list(tool_names)
    for hallucinated, real in _TOOL_NAME_ALIASES.items():
        if real in tool_names and hallucinated not in aliased_names:
            aliased_names.append(hallucinated)

    # Case A: result is a dict — ET parsed nested XML as child elements.
    # e.g. <result><delete_file><path>foo</path></delete_file></result>
    #      → args["result"] = {"delete_file": {"path": "foo"}}
    # Any key here was a valid XML child element → treat all as tool calls (no filter).
    if isinstance(result_content, dict):
        rescued: List[Dict] = []
        for child_name, child_args in result_content.items():
            real_name = _resolve_xml_tool_name(child_name, tool_names, request_id)
            # Multiple same-tag children → list; single child → dict
            items = child_args if isinstance(child_args, list) else [child_args]
            for item in items:
                arg_dict = item if isinstance(item, dict) else {}
                rescued.append({
                    "id": f"call_{request_id}_{real_name}",
                    "type": "function",
                    "function": {
                        "name": real_name,
                        "arguments": json.dumps(arg_dict),
                    },
                })
        if rescued:
            logger.warning(
                f"[{request_id}] attempt_completion.result contained nested tool calls "
                f"({[r['function']['name'] for r in rescued]}) — rescuing"
            )
            return rescued

    # Case B: result is a string containing raw XML tags
    elif isinstance(result_content, str) and "<" in result_content:
        xml_calls, _ = extract_xml_tool_calls(result_content, aliased_names, request_id)
        if xml_calls:
            logger.warning(
                f"[{request_id}] attempt_completion.result contained XML tool calls "
                f"({[xc.name for xc in xml_calls]}) — rescuing"
            )
            for xc in xml_calls:
                xc.name = _resolve_xml_tool_name(xc.name, tool_names, request_id)
            return convert_xml_tool_calls_to_openai_format(xml_calls)

    return tool_calls


def fix_ask_followup_question_params(
    tool_calls: List[Dict], request_id: str
) -> List[Dict]:
    """
    Normalize ask_followup_question arguments.

    The model sometimes outputs `follow_up` as a newline-separated string
    instead of a JSON array.  Roo Code requires an array and rejects the
    call with "Missing value for required parameter 'follow_up'" otherwise.

    Example bad input:
        "follow_up": "Show output\\nRestart server\\nCheck config"
    Fixed output:
        "follow_up": ["Show output", "Restart server", "Check config"]
    """
    result = []
    for tc in tool_calls:
        if tc.get("function", {}).get("name") != "ask_followup_question":
            result.append(tc)
            continue

        try:
            args = json.loads(tc["function"].get("arguments", "{}"))
        except json.JSONDecodeError:
            result.append(tc)
            continue

        follow_up = args.get("follow_up")
        if isinstance(follow_up, str):
            items = [s.strip() for s in follow_up.splitlines() if s.strip()]
            if items:
                args["follow_up"] = items
                logger.info(
                    f"[{request_id}] ask_followup_question: follow_up string → array "
                    f"({len(items)} items)"
                )
                tc = {
                    **tc,
                    "function": {
                        "name": "ask_followup_question",
                        "arguments": json.dumps(args),
                    },
                }

        result.append(tc)
    return result

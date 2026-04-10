"""
Loop detection for toolproxy.

Two detectors:
1. detect_success_loop  — write-type tools keep succeeding but model calls them again.
2. detect_repetitive_tool_loop — model calls the same tool with the same key argument
   repeatedly, regardless of success/failure (e.g. read_file on the same path 3× in a row).
"""
import logging
import re
from typing import Dict, List, Optional

from app.services.tool_mapping import ClientType

logger = logging.getLogger(__name__)

_WRITE_SUCCESS_WORDS = ("successfully", "appended", "created", "written", "modified", "updated")

# Markers that indicate a new user instruction is embedded inside a tool result.
# Roo Code embeds the user's follow-up message inside the attempt_completion
# tool_result as <user_message>...</user_message>.  Other clients may use
# different markers — extend this tuple as needed.
_NEW_TASK_MARKERS = ("<user_message>",)


def detect_success_loop(
    messages: List[Dict],
    request_id: str = "",
    client_type: ClientType = ClientType.ROO_CODE,
) -> Optional[str]:
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
        # New task boundary: a user instruction is embedded inside the tool result.
        # Roo Code does this when the user replies after attempt_completion — the new
        # instruction arrives as <user_message> inside the attempt_completion result.
        # Any client that follows this pattern (extend _NEW_TASK_MARKERS for others).
        if any(marker in content for marker in _NEW_TASK_MARKERS):
            logger.debug(f"[{request_id}] success_loop: reset at new task boundary in tool result")
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
    if client_type == ClientType.OPEN_CODE:
        return (
            f"STOP: The file operation reported success {success_count} times in a row "
            "but the task may not be complete. "
            "Use read_file to verify the current file state, "
            "or use write_to_file with the COMPLETE corrected content. "
            "Do NOT repeat the same partial operation again."
        )
    return (
        f"STOP: The file operation reported success {success_count} times in a row "
        "but the task may not be complete. "
        "Use read_file to verify the current file state, "
        "or use write_to_file with the COMPLETE corrected content. "
        "If the task is truly done, call attempt_completion."
    )


# ---------------------------------------------------------------------------
# Repetitive tool-call loop detection
# ---------------------------------------------------------------------------

# Regex to extract the outermost XML tool tag name and its <path> child (if any).
# Matches: <tool_name ...>...<path>value</path>...</tool_name>
_XML_TOOL_RE = re.compile(r"<([a-z_]+)[\s>]", re.IGNORECASE)
_XML_PATH_RE = re.compile(r"<(?:path|filePath)>([^<]{1,300})</(?:path|filePath)>", re.IGNORECASE)

# How many consecutive identical calls before we fire the hint.
_REPETITIVE_THRESHOLD = 3


def _extract_tool_call_key(content: str) -> Optional[str]:
    """
    Extract a deduplication key from an assistant message that contains an XML
    tool call.  Key format: "<tool_name>|<path>" or "<tool_name>" if no path.
    """
    tool_match = _XML_TOOL_RE.search(content)
    if not tool_match:
        return None
    tool_name = tool_match.group(1).lower()
    path_match = _XML_PATH_RE.search(content)
    if path_match:
        return f"{tool_name}|{path_match.group(1).strip()}"
    return tool_name


def detect_repetitive_tool_loop(
    messages: List[Dict],
    request_id: str = "",
    client_type: ClientType = ClientType.ROO_CODE,
) -> Optional[str]:
    """
    Detect when the model calls the exact same tool with the same path/args
    N consecutive times (e.g. read_file on the same file over and over).

    Scans the last 12 messages, looking at assistant turns only.
    Resets the counter when a genuine user message (non-tool-result) appears.
    """
    recent = messages[-12:] if len(messages) > 12 else messages

    consecutive: int = 0
    last_key: Optional[str] = None

    for msg in recent:
        role = msg.get("role", "")
        content = str(msg.get("content", ""))

        if role == "user":
            if "[Tool Result]" not in content:
                # Genuine user instruction — reset
                consecutive = 0
                last_key = None
            continue

        if role != "assistant":
            continue

        key = _extract_tool_call_key(content)
        if key is None:
            # No tool call in this assistant turn — reset streak
            consecutive = 0
            last_key = None
            continue

        if key == last_key:
            consecutive += 1
            logger.debug(f"[{request_id}] repetitive_loop: key={key!r} count={consecutive}")
        else:
            consecutive = 1
            last_key = key

    if consecutive < _REPETITIVE_THRESHOLD:
        return None

    tool_name = (last_key or "").split("|")[0]
    logger.warning(
        f"[{request_id}] REPETITIVE LOOP: tool={tool_name!r} called {consecutive}× "
        f"in a row — injecting correction hint"
    )
    if client_type == ClientType.OPEN_CODE:
        return (
            f"STOP: You have called '{tool_name}' {consecutive} times in a row "
            "without making progress. Use write_to_file with the COMPLETE corrected "
            "content, or read_file to verify the current state first. "
            "Do not repeat the same partial operation again."
        )
    return (
        f"STOP: You have called '{tool_name}' {consecutive} times in a row "
        "without making progress. Use write_to_file with the COMPLETE corrected "
        "content, or read_file to verify the current state first. "
        "Do not repeat the same partial operation again."
    )


# ---------------------------------------------------------------------------
# ask_followup_question loop detection
# ---------------------------------------------------------------------------

# Window size and threshold for ask_followup_question frequency detection.
# Each ask_followup_question cycle adds 3 messages (tool call + tool result +
# genuine user reply), so a window of 24 covers 8 cycles.
_AFQ_WINDOW = 24
_AFQ_THRESHOLD = 3

_AFQ_RE = re.compile(r"<ask_followup_question[\s>]", re.IGNORECASE)


def detect_ask_followup_loop(
    messages: List[Dict],
    request_id: str = "",
    client_type: ClientType = ClientType.ROO_CODE,
) -> Optional[str]:
    """
    Detect when the model repeatedly calls ask_followup_question without making
    progress.

    Unlike detect_repetitive_tool_loop, this detector uses a frequency window
    rather than consecutive counting — because Roo Code sends a genuine user
    message after each answer, which would reset a consecutive counter.

    Fires when ask_followup_question appears >= _AFQ_THRESHOLD times in the
    last _AFQ_WINDOW messages.
    """
    recent = messages[-_AFQ_WINDOW:] if len(messages) > _AFQ_WINDOW else messages
    count = sum(
        1
        for msg in recent
        if msg.get("role") == "assistant" and _AFQ_RE.search(str(msg.get("content", "")))
    )

    if count < _AFQ_THRESHOLD:
        return None

    logger.warning(
        f"[{request_id}] ASK FOLLOWUP LOOP: ask_followup_question called {count}× "
        f"in last {len(recent)} messages — injecting correction hint"
    )
    return (
        f"STOP: You have asked ask_followup_question {count} times without making progress. "
        "The user has already provided their input. "
        "Stop asking clarifying questions and take direct action based on the information you have. "
        "If something is unclear, make a reasonable assumption and proceed."
    )

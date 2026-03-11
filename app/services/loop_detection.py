"""
Success-loop detection for toolproxy.

Detects when write-type tools keep succeeding but the model keeps calling them
again, and returns a stop-hint string to inject into the conversation.
"""
import logging
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
            f"STOP: The file operation has already succeeded {success_count} times. "
            "Do NOT repeat the tool call — you are creating duplicates. "
            "The task is done — respond with a brief plain-text summary."
        )
    return (
        f"STOP: The file operation has already succeeded {success_count} times. "
        "Do NOT repeat the tool call — you are creating duplicates. "
        "Call attempt_completion to report that the task is done."
    )

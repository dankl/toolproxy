"""
Anonymized audit logger for toolproxy.

Writes one JSON-Lines entry per request to a rotating log file.
No file contents, no paths, no user messages — only structural metadata
that is useful for pattern analysis without revealing sensitive information.

Fields logged:
  ts              ISO-8601 timestamp
  req             Request ID (8 chars)
  client          Detected client type (roo_code / cline / open_code / generic)
  tools           Number of tools in the request
  msgs            Number of messages in the request
  upstream_ms     Latency to upstream LLM in milliseconds
  response_chars  Length of raw model output in characters (no content)
  preamble_len    Characters of text before the XML tool call (0 = clean output)
  mechanism       How the tool call was resolved:
                    xml_parsed / json_fallback / text_synthesis /
                    write_guard / empty_fallback / no_tool_call
  fallback_pat    For json_fallback: which pattern matched (e.g. "[Tool Call:]")
  tool            Tool name called (e.g. write_to_file)
  path_ext        File extension from path argument (e.g. .java) — no path
  path_hash       First 6 chars of MD5(path) — same path = same hash, unreadable
  content_lines   Line count of content argument (write_to_file / apply_diff)
  content_chars   Char count of content argument
  loop_type       If loop detected: success / repetitive
  loop_count      If repetitive loop: number of consecutive calls
"""
import hashlib
import json
import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, Optional

_audit_logger: Optional[logging.Logger] = None


def setup_audit_logger(log_path: str) -> None:
    """Initialize the audit logger with a rotating file handler."""
    global _audit_logger
    if not log_path:
        return

    try:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
    except OSError as e:
        logging.getLogger(__name__).warning(f"Audit log disabled — cannot create directory: {e}")
        return

    _audit_logger = logging.getLogger("toolproxy.audit")
    _audit_logger.setLevel(logging.INFO)
    _audit_logger.propagate = False  # don't leak into the main log

    handler = RotatingFileHandler(
        log_path,
        maxBytes=5 * 1024 * 1024,  # 5 MB per file
        backupCount=5,              # keep 5 rotated files = max 25 MB
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    _audit_logger.addHandler(handler)


def write_audit(entry: Dict[str, Any]) -> None:
    """Append one JSON-Lines entry to the audit log."""
    if _audit_logger is None:
        return
    _audit_logger.info(json.dumps(entry, ensure_ascii=False))


def hash_path(path: str) -> str:
    """Return first 6 hex chars of MD5(path) — identifies a path without revealing it."""
    return hashlib.md5(path.encode()).hexdigest()[:6]


def extract_file_info(tool_name: str, arguments_json: str) -> Dict[str, Any]:
    """
    Extract anonymized file metadata from a tool call's arguments JSON.

    Returns a dict with zero or more of:
      path_ext, path_hash, content_lines, content_chars
    """
    try:
        args = json.loads(arguments_json or "{}")
    except Exception:
        return {}

    info: Dict[str, Any] = {}

    path = args.get("path") or args.get("filePath") or ""
    if path:
        _, ext = os.path.splitext(path)
        info["path_ext"] = ext.lower() if ext else "(no ext)"
        info["path_hash"] = hash_path(path)

    # Content metrics for write_to_file / apply_diff / replace_in_file
    content = args.get("content") or args.get("diff") or ""
    if content and isinstance(content, str) and tool_name in ("write_to_file", "apply_diff", "replace_in_file"):
        info["content_chars"] = len(content)
        info["content_lines"] = content.count("\n") + 1

    return info

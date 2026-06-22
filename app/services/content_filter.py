"""
Pre-request content filter for OCI safety filter prevention.

OCI's content safety gateway blocks HTTP 400 with "inappropriate content"
for certain patterns in either the prompt or the model's response. This
module scans outgoing messages for known trigger patterns BEFORE they reach
OCI, giving the proxy a chance to sanitize the content or return a clear
error instead of the opaque OCI filter response.

Configuration via env var CONTENT_FILTER_PATTERNS (JSON array of objects):
  [
    {"pattern": "hack", "replacement": "[redacted]", "reason": "Security slang"},
    {"pattern": "(?i)malware", "replacement": "[software]", "reason": "Malware reference"}
  ]

If a pattern has no "replacement", the match is only logged + warned.
If a pattern has a "replacement", the content is sanitized automatically.
"""
import json
import logging
import re
from typing import Dict, List, Optional, Tuple

from app.config import settings

logger = logging.getLogger(__name__)

# Default blocked patterns known to trigger OCI content filter in code-gen contexts.
# These are conservative — they flag things that OFTEN trigger the filter without
# being overly broad.  Users can override via CONTENT_FILTER_PATTERNS env var.
_DEFAULT_PATTERNS: List[Dict[str, str]] = [
    # Terms that OCI's safety filter frequently blocks in code context.
    # These are NOT censored in the normal sense — they're patterns that
    # cause the model to return HTTP 400 before any code is generated.
    {"pattern": r"(?i)\bexploit\b",           "replacement": "", "reason": "OCI filter trigger: 'exploit'"},
    {"pattern": r"(?i)\bmalware\b",           "replacement": "", "reason": "OCI filter trigger: 'malware'"},
    {"pattern": r"(?i)\bransomware\b",        "replacement": "", "reason": "OCI filter trigger: 'ransomware'"},
    {"pattern": r"(?i)\bbackdoor\b",          "replacement": "", "reason": "OCI filter trigger: 'backdoor'"},
    {"pattern": r"(?i)\bkeylogger\b",         "replacement": "", "reason": "OCI filter trigger: 'keylogger'"},
    {"pattern": r"(?i)\btrojan\b",            "replacement": "", "reason": "OCI filter trigger: 'trojan'"},
    {"pattern": r"(?i)\bcrack\b",             "replacement": "", "reason": "OCI filter trigger: 'crack'"},
    {"pattern": r"(?i)\bkeygen\b",            "replacement": "", "reason": "OCI filter trigger: 'keygen'"},
    {"pattern": r"(?i)\bphishing\b",          "replacement": "", "reason": "OCI filter trigger: 'phishing'"},
    {"pattern": r"(?i)\bbotnet\b",            "replacement": "", "reason": "OCI filter trigger: 'botnet'"},
    {"pattern": r"(?i)\bdarknet\b",           "replacement": "", "reason": "OCI filter trigger: 'darknet'"},
    {"pattern": r"(?i)\bdoxing\b",            "replacement": "", "reason": "OCI filter trigger: 'doxing'"},
    {"pattern": r"(?i)\bddos\b",              "replacement": "", "reason": "OCI filter trigger: 'ddos'"},
    {"pattern": r"(?i)\bsql\s*inject",         "replacement": "", "reason": "OCI filter trigger: SQL injection"},
    {"pattern": r"(?i)\bxss\b",               "replacement": "", "reason": "OCI filter trigger: XSS"},
    {"pattern": r"(?i)\bbuffer\s*overflow",    "replacement": "", "reason": "OCI filter trigger: buffer overflow"},
    {"pattern": r"(?i)\brootkit\b",           "replacement": "", "reason": "OCI filter trigger: 'rootkit'"},
    {"pattern": r"(?i)\bspyware\b",           "replacement": "", "reason": "OCI filter trigger: 'spyware'"},
    {"pattern": r"(?i)\bsextortion\b",        "replacement": "", "reason": "OCI filter trigger: 'sextortion'"},
    # Validation error cascades: Client sends back schema validation errors as
    # [Tool Result], which then triggers OCI's content filter on the NEXT turn.
    {"pattern": r"(?i)validation failed for tool",  "replacement": "", "reason": "OCI filter trigger: validation error"},
    {"pattern": r"(?i)must have required properties", "replacement": "", "reason": "OCI filter trigger: schema validation"},
    {"pattern": r"(?i)must not have additional properties", "replacement": "", "reason": "OCI filter trigger: schema validation"},
]

# Pre-compiled cache
_patterns_cache: Optional[List[Tuple[re.Pattern, str, str, bool]]] = None


def _load_patterns() -> List[Tuple[re.Pattern, str, str, bool]]:
    """Load and compile filter patterns from settings or defaults."""
    global _patterns_cache
    if _patterns_cache is not None:
        return _patterns_cache

    raw = settings.content_filter_patterns
    if raw:
        try:
            config = json.loads(raw)
            if not isinstance(config, list):
                raise ValueError("CONTENT_FILTER_PATTERNS must be a JSON array")
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Invalid CONTENT_FILTER_PATTERNS: {e} — using defaults")
            config = _DEFAULT_PATTERNS
    else:
        config = _DEFAULT_PATTERNS

    compiled = []
    for entry in config:
        pat_str = entry.get("pattern", "")
        replacement = entry.get("replacement", "")
        reason = entry.get("reason", "Unknown trigger")
        if not pat_str:
            continue
        try:
            pat = re.compile(pat_str)
            has_replacement = "replacement" in entry
            compiled.append((pat, replacement, reason, has_replacement))
        except re.error as e:
            logger.warning(f"Invalid regex in CONTENT_FILTER_PATTERNS: {pat_str!r} — {e}")
            continue

    _patterns_cache = compiled
    logger.info(
        f"Content filter loaded: {len(compiled)} pattern(s) "
        f"({'auto-sanitize enabled' if any(p[3] for p in compiled) else 'detect-only'})"
    )
    return compiled


def _get_text_content(msg: Dict) -> str:
    """Extract plain text from a message dict, handling content arrays."""
    content = msg.get("content") or ""
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                for v in part.values():
                    if isinstance(v, str):
                        parts.append(v)
        return " ".join(parts)
    return str(content)


def scan_messages(messages: List[Dict], request_id: str = "") -> List[Dict]:
    """
    Scan all messages for patterns that might trigger OCI's content filter.

    Returns a list of match dicts, one per unique pattern triggered:
      [{"pattern": "...", "reason": "...", "location": "user/assistant/system #N", "sanitized": bool}]
    """
    patterns = _load_patterns()
    if not patterns:
        return []

    hits = []
    seen = set()

    for i, msg in enumerate(messages):
        role = msg.get("role", "unknown")
        text = _get_text_content(msg)
        if not text:
            continue

        for pat, replacement, reason, has_replacement in patterns:
            if reason in seen:
                continue
            match = pat.search(text)
            if match:
                seen.add(reason)
                hits.append({
                    "pattern": pat.pattern,
                    "reason": reason,
                    "location": f"{role} #{i}",
                    "sanitized": has_replacement,
                    "matched_text": match.group()[:60],
                })

    return hits


def sanitize_messages(messages: List[Dict], request_id: str = "") -> List[Dict]:
    """
    Apply replacements to messages for patterns that have a replacement defined.

    Returns (sanitized_messages, was_modified).
    """
    patterns = _load_patterns()
    if not patterns:
        return messages, False

    modified = False
    for i, msg in enumerate(messages):
        text = _get_text_content(msg)
        if not text:
            continue

        new_text = text
        for pat, replacement, reason, has_replacement in patterns:
            if has_replacement:
                new_text = pat.sub(replacement, new_text)

        if new_text != text:
            modified = True
            msg = dict(msg)
            msg["content"] = new_text
            messages[i] = msg
            logger.info(
                f"[{request_id}] Sanitized {msg.get('role', '?')} message "
                f"(pattern matched, content modified)"
            )

    return messages, modified


def reset_cache() -> None:
    """Reset pattern cache (useful for tests)."""
    global _patterns_cache
    _patterns_cache = None

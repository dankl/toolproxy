"""
JSON repair service with fallback for malformed LLM outputs.
"""
import json
import logging
from typing import Optional, Dict, List, Any

try:
    from json_repair import repair_json
    JSON_REPAIR_AVAILABLE = True
except ImportError:
    JSON_REPAIR_AVAILABLE = False
    logging.warning("json-repair library not available, using fallback")


logger = logging.getLogger(__name__)


def safe_parse_json(json_string: str) -> Optional[Dict]:
    """
    Parse JSON with repair fallback.

    Args:
        json_string: JSON string to parse

    Returns:
        Parsed dict or None if parsing fails

    """
    if not json_string or not json_string.strip():
        return None

    try:
        # First attempt: direct JSON parse
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        logger.warning(f"Initial JSON parse failed: {e}")

        if JSON_REPAIR_AVAILABLE:
            try:
                # Second attempt: repair and parse
                repaired = repair_json(json_string)
                logger.info("JSON repair successful")
                return json.loads(repaired)
            except Exception as repair_error:
                logger.error(f"JSON repair failed: {repair_error}")
                return None
        else:
            logger.error("JSON repair library not available")
            return None


def safe_parse_tool_calls(response_content: str) -> Optional[List[Dict]]:
    """
    Parse tool calls from LLM response with JSON repair fallback.

    Args:
        response_content: Response content that may contain tool_calls

    Returns:
        List of tool call dicts if successful, None otherwise
    """
    parsed = safe_parse_json(response_content)
    if parsed:
        return parsed.get("tool_calls")
    return None


def safe_parse_arguments(args_string: str) -> Dict[str, Any]:
    """
    Parse tool call arguments with repair fallback.

    Args:
        args_string: JSON string of arguments

    Returns:
        Parsed arguments dict

    Raises:
        ValueError: If parsing fails even after repair
    """
    parsed = safe_parse_json(args_string)
    if parsed is None:
        raise ValueError(f"Failed to parse arguments: {args_string}")
    return parsed

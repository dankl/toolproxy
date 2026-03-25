"""
XML Tool Call Parser for RooCode/Cline compatibility.

This module extracts XML-formatted tool calls from LLM responses and converts
them to OpenAI-compatible tool_calls format. This is necessary because:
1. RooCode/Cline uses XML format for tool definitions
2. GPT-OSS-20B may output tool calls in XML format
3. The OpenAI API expects tool_calls in JSON format

Based on patterns from: https://github.com/irreg/native_tool_call_adapter
"""
import hashlib
import json
import logging
import re
import secrets
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class XMLToolCall:
    """Represents a parsed XML tool call."""
    name: str
    arguments: Dict[str, Any]
    id: str
    raw_xml: str
    reasoning_content: Optional[str] = None


def extract_xml_tool_calls(
    content: str,
    tool_names: List[str],
    request_id: str = ""
) -> Tuple[List[XMLToolCall], str]:
    """
    Extract XML-formatted tool calls from content.

    Args:
        content: The response content that may contain XML tool calls
        tool_names: List of valid tool names to look for
        request_id: Request ID for logging

    Returns:
        Tuple of (list of XMLToolCall objects, remaining content with XML removed)
    """
    log_prefix = f"[{request_id}] " if request_id else ""

    if not content or not tool_names:
        return [], content

    tool_calls = []
    remaining_content = content

    # Build regex pattern for all tool names
    tool_pattern = "|".join(re.escape(name) for name in tool_names)
    pattern = re.compile(
        rf"<({tool_pattern})\b[^>]*>([\s\S]*?)</\1>",
        re.IGNORECASE
    )

    for match in pattern.finditer(content):
        tool_name = match.group(1)
        xml_content = match.group(0)

        try:
            # Parse XML to extract arguments
            arguments, tool_id, reasoning = parse_xml_to_arguments(xml_content, tool_name)

            tool_call = XMLToolCall(
                name=tool_name,
                arguments=arguments,
                id=tool_id or generate_tool_id(xml_content),
                raw_xml=xml_content,
                reasoning_content=reasoning
            )
            tool_calls.append(tool_call)

            # Remove the XML from remaining content
            remaining_content = remaining_content.replace(xml_content, "", 1)

            logger.debug(f"{log_prefix}Extracted XML tool call: {tool_name}")
            logger.debug(f"{log_prefix}Arguments: {arguments}")

        except Exception as e:
            logger.warning(f"{log_prefix}Failed to parse XML tool call: {e}")
            continue

    return tool_calls, remaining_content.strip()


def parse_xml_to_arguments(
    xml_string: str,
    expected_tool_name: str
) -> Tuple[Dict[str, Any], Optional[str], Optional[str]]:
    """
    Parse XML string to extract tool arguments.

    Args:
        xml_string: The XML string to parse
        expected_tool_name: The expected root tag name

    Returns:
        Tuple of (arguments dict, tool_id if present, reasoning_content if present)
    """
    # Always run fix_xml_string first to escape raw-text tags (content, diff, result)
    # before ET parsing — this prevents JSX/HTML inside <content> from being
    # misinterpreted as XML child elements even when the raw XML is technically valid.
    fixed_xml = fix_xml_string(xml_string)
    root = ET.fromstring(fixed_xml)

    # Verify root tag matches expected tool name
    if root.tag.lower() != expected_tool_name.lower():
        raise ValueError(f"Root tag '{root.tag}' doesn't match expected '{expected_tool_name}'")

    # Extract special elements
    tool_id = None
    reasoning_content = None

    for child in list(root):
        if child.tag == "id":
            tool_id = child.text
            root.remove(child)
        elif child.tag == "think":
            reasoning_content = child.text
            root.remove(child)

    # Convert remaining XML to dict
    arguments = xml_element_to_dict(root)

    return arguments, tool_id, reasoning_content


def xml_element_to_dict(elem: ET.Element) -> Dict[str, Any]:
    """
    Convert XML element to dictionary.

    Handles:
    - Simple text content
    - Nested elements
    - Repeated elements (converted to lists)
    - Attributes
    """
    result = {}

    # Group children by tag
    children_by_tag: Dict[str, List[ET.Element]] = defaultdict(list)
    for child in elem:
        children_by_tag[child.tag].append(child)

    if not children_by_tag:
        # Leaf node - return text content
        text = (elem.text or "").strip()
        if elem.attrib:
            # Has attributes - return as object with value
            return {"value": text, **elem.attrib}
        return text

    # Process children
    for tag, children in children_by_tag.items():
        if len(children) == 1:
            # Single child - recurse
            result[tag] = xml_element_to_dict(children[0])
        else:
            # Multiple children with same tag - create list
            result[tag] = [xml_element_to_dict(child) for child in children]

    return result


def fix_xml_string(xml_string: str) -> str:
    """
    Attempt to fix common XML issues.
    """
    # Pre-escape content in "raw text" tags that may contain XML-special characters
    # (e.g. <content> in write_to_file may contain <repo-url>, XML snippets, etc.)
    _RAW_TEXT_TAGS = ("content", "diff", "result", "output")

    def escape_raw_content(m: re.Match) -> str:
        tag, inner = m.group(1), m.group(2)
        if '<' in inner or '>' in inner:
            # Escape bare & first (skip already-valid entities)
            inner = re.sub(r'&(?!amp;|lt;|gt;|quot;|apos;)', '&amp;', inner)
            inner = inner.replace('<', '&lt;').replace('>', '&gt;')
        return f'<{tag}>{inner}</{tag}>'

    # Insert missing </diff> BEFORE escaping so the escape regex can match the full <diff> block
    fixed = xml_string
    if re.search(r'<diff\b', fixed, re.IGNORECASE) and not re.search(r'</diff>', fixed, re.IGNORECASE):
        fixed = re.sub(
            r'</(apply_diff|replace_in_file)>',
            r'</diff>\n</\1>',
            fixed,
            flags=re.IGNORECASE,
        )

    tag_pat = '|'.join(_RAW_TEXT_TAGS)
    fixed = re.sub(
        rf'<({tag_pat})>([\s\S]*?)</\1>',
        escape_raw_content,
        fixed,
        flags=re.IGNORECASE,
    )

    # Replace remaining unescaped ampersands
    fixed = re.sub(r'&(?!amp;|lt;|gt;|quot;|apos;)', '&amp;', fixed)

    # Fix pseudo-tags in parentheses (e.g., "(</tag>)" -> "(tag)")
    def replace_pseudo_tags(match):
        content = match.group(1)
        converted = re.sub(r"</?([\w]*)\s*/?>", r"`\1`", content)
        return f"({converted})"

    fixed = re.sub(r"\(([^)]*)\)", replace_pseudo_tags, fixed)

    return fixed


def generate_tool_id(xml_content: str) -> str:
    """
    Generate a tool ID from XML content. The MD5 prefix is stable for
    debugging, but a random suffix ensures uniqueness even when the model
    repeats the exact same tool call (which would cause duplicate IDs and
    confuse clients like Roo Code).
    """
    base = hashlib.md5(xml_content.encode()).hexdigest()[:12]
    suffix = secrets.token_hex(2)
    return f"call_{base}_{suffix}"


def convert_xml_tool_calls_to_openai_format(
    xml_tool_calls: List[XMLToolCall]
) -> List[Dict[str, Any]]:
    """
    Convert XMLToolCall objects to OpenAI tool_calls format.

    Args:
        xml_tool_calls: List of XMLToolCall objects

    Returns:
        List of tool_calls in OpenAI format
    """
    return [
        {
            "id": tc.id,
            "type": "function",
            "function": {
                "name": tc.name,
                "arguments": json.dumps(tc.arguments, ensure_ascii=False)
            }
        }
        for tc in xml_tool_calls
    ]


def dict_to_xml_element(
    data: Dict[str, Any],
    root_name: str,
    tool_id: Optional[str] = None,
    reasoning_content: Optional[str] = None
) -> str:
    """
    Convert dictionary back to XML string.

    This is useful for converting OpenAI tool_calls back to XML format
    that RooCode expects.

    Args:
        data: The arguments dictionary
        root_name: The tool name (root tag)
        tool_id: Optional tool call ID
        reasoning_content: Optional reasoning/thinking content

    Returns:
        XML string
    """
    root = ET.Element(root_name)

    def build_element(parent: ET.Element, obj: Any) -> None:
        if isinstance(obj, dict):
            if "value" in obj:
                # Handle value + attributes pattern
                parent.text = str(obj["value"]) if obj["value"] is not None else ""
                for k, v in obj.items():
                    if k != "value":
                        parent.set(k, str(v))
            else:
                for key, value in obj.items():
                    if isinstance(value, list):
                        for item in value:
                            child = ET.SubElement(parent, key)
                            build_element(child, item)
                    else:
                        child = ET.SubElement(parent, key)
                        build_element(child, value)
        else:
            parent.text = str(obj) if obj is not None else ""

    build_element(root, data)

    # Add ID element
    if tool_id:
        id_elem = ET.SubElement(root, "id")
        id_elem.text = tool_id

    # Add reasoning content
    if reasoning_content:
        think_elem = ET.SubElement(root, "think")
        think_elem.text = reasoning_content

    return ET.tostring(root, encoding="unicode")


def extract_tool_names_from_request(request_tools: List[Dict]) -> List[str]:
    """
    Extract tool names from request tools list.

    Args:
        request_tools: List of tool definitions from the request

    Returns:
        List of tool names
    """
    return [
        tool.get("function", {}).get("name", "")
        for tool in request_tools
        if tool.get("function", {}).get("name")
    ]

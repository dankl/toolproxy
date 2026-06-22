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

    # Remove nested/contained duplicate tool calls (e.g. inner <edit> inside <edits><edit>)
    # Remove if tc is contained in other AND tc is smaller (inner block)
    deduped: List[XMLToolCall] = []
    for tc in tool_calls:
        if any(
            other is not tc and other.name == tc.name 
            and tc.raw_xml in other.raw_xml 
            and len(tc.raw_xml) < len(other.raw_xml)
            for other in tool_calls
        ):
            logger.debug(f"{log_prefix}Removing nested {tc.name} (contained in larger {tc.name})")
            continue
        deduped.append(tc)

    # Remove exact duplicate tool calls (same name and arguments)
    seen = set()
    final: List[XMLToolCall] = []
    for tc in deduped:
        key = (tc.name, json.dumps(tc.arguments, sort_keys=True))
        if key in seen:
            logger.info(f"{log_prefix}Removing duplicate {tc.name} call (identical arguments)")
            continue
        seen.add(key)
        final.append(tc)

    return final, remaining_content.strip()


def _close_unclosed_root(xml_string: str) -> str:
    """If the outermost tag is unclosed, append its closing tag.

    Handles:   <write_to_file><path>foo</path>   (missing </write_to_file>)
    Returns:   <write_to_file><path>foo</path></write_to_file>
    """
    # Find the root tag name (first <tagname ...> at start, after whitespace)
    m = re.match(r'\s*<\s*(\w+)(?:\s[^>]*)?>', xml_string)
    if not m:
        return xml_string
    root_name = m.group(1)
    # Check if closing tag already exists
    if re.search(rf'</\s*{re.escape(root_name)}\s*>', xml_string, re.IGNORECASE):
        return xml_string  # already closed
    return xml_string + f'</{root_name}>'


def _strip_whitespace_in_tags(xml_string: str) -> str:
    """Strip whitespace between < and tag name: < write_to_file > → <write_to_file>

    Uses negative lookbehind to avoid affecting non-XML patterns
    like <<<<<<< SEARCH (multiple consecutive < characters).

    The lookahead ``(?=[\\s/>])`` requires the matched word to be followed by
    whitespace, ``/`` or ``>`` — i.e. it must look like a real tag. This prevents
    corrupting shell redirects inside raw-text tags such as
    ``<command>sort < in.txt</command>`` (where ``< in`` would otherwise be
    collapsed to ``<in``).
    """
    # Fix opening tags: < tagname → <tagname (only when it looks like a tag)
    s = re.sub(r'(?<!<)<\s+(\w+)(?=[\s/>])', r'<\1', xml_string)
    # Fix closing tags: </ tagname > → </tagname>
    s = re.sub(r'</\s+(\w+)\s*>', r'</\1>', s)
    return s


def _parse_xml_with_fallback(xml_string: str) -> Optional[ET.Element]:
    """Try multiple strategies to parse XML, falling back through alternatives.

    Strategy order:
    1. Direct ET.fromstring (already fixed)
    2. Close unclosed root tag and retry
    3. Wrap in synthetic root if it looks like multiple fragments
    """
    strategies = [
        ("direct", lambda s: ET.fromstring(s)),
        ("close_root", lambda s: ET.fromstring(_close_unclosed_root(s))),
        ("wrapped_root", lambda s: ET.fromstring(f"<root>{_close_unclosed_root(s)}</root>") if not s.strip().startswith("<root") else None),
    ]
    errors = []
    for name, fn in strategies:
        try:
            result = fn(xml_string)
            if result is not None:
                return result
        except ET.ParseError as e:
            errors.append(f"{name}: {e}")
            continue
    logger.debug(f"XML parse fallback exhausted. Errors: {'; '.join(errors)}")
    return None


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
    root = _parse_xml_with_fallback(fixed_xml)
    if root is None:
        raise ValueError(f"Failed to parse XML after all fallback strategies for '{expected_tool_name}'")

    # If we used wrapped_root strategy, unwrap to find the actual tool element
    if root.tag == "root":
        for child in root:
            if child.tag.lower() == expected_tool_name.lower():
                root = child
                break
        else:
            raise ValueError(f"Root tag not found after wrapped-root fallback for '{expected_tool_name}'")

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
    - Tag names are preserved as-is (case-sensitive). The schema remap step
      (_remap_args_to_schema in tool_mapping.py) handles case normalization
      via _SCHEMA_PARAM_ALIASES.
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


def _convert_self_closing_tags(xml_string: str) -> str:
    """Convert self-closing tags to open/close pairs: <param/> → <param></param>

    The model sometimes outputs empty params as <path/> instead of <path></path>.
    ET.fromstring handles self-closing tags fine, but our regex-based extraction
    and raw-text escaping works better with explicit open/close pairs.
    """
    return re.sub(
        r'<(\w+)([^>]*?)\s*/>',
        r'<\1\2></\1>',
        xml_string,
        flags=re.IGNORECASE,
    )


def _convert_attributes_to_elements(xml_string: str) -> str:
    """Convert attribute-style syntax to element style: <tool param="val"> → <tool><param>val</param>

    Some models output tool calls as attributes instead of child elements:
      <write_to_file path="README.md" content="# Hello" />
    instead of:
      <write_to_file><path>README.md</path><content># Hello</content></write_to_file>

    This regex converts the attribute form to element form before ET parsing.
    Only applies to known tool root tags (not generic XML).
    """
    _KNOWN_TOOL_TAGS = (
        "write_to_file", "read_file", "apply_diff", "replace_in_file",
        "attempt_completion", "execute_command", "ask_followup_question",
        "use_mcp_tool", "move_file", "rename_file", "delete_file",
        "delete_folder", "create_directory", "list_files", "search_files",
        "bash", "edit", "glob", "grep", "question", "read", "write",
        "list", "new_task", "browser_action",
    )

    def _replace_attr(am: re.Match) -> str:
        k, v = am.group(1), am.group(2)
        return f"<{k}>{v}</{k}>"

    def _convert_attrs_in_match(tag: str, attrs_part: str, inner: str = "") -> str:
        attrs_converted = re.sub(r'\s+(\w+)\s*=\s*"([^"]*)"', _replace_attr, attrs_part)
        attrs_converted = re.sub(r"\s+(\w+)\s*=\s*'([^']*)'", _replace_attr, attrs_converted)
        return f"<{tag}>{attrs_converted}{inner}</{tag}>"

    tag_pat = "|".join(_KNOWN_TOOL_TAGS)

    # Step 1: self-closing with attributes → non-self-closing
    # <tool a="b" /> → <tool a="b"></tool>
    result = re.sub(
        rf'<({tag_pat})([^>]*)\s*/>',
        r'<\1\2></\1>',
        xml_string,
        flags=re.IGNORECASE,
    )

    # Step 2: open/close with attributes → element format
    # <tool a="b">inner</tool> → <tool><a>b</a>inner</tool>
    result = re.sub(
        rf'<({tag_pat})([^>]*\s+\w+\s*=\s*["\'][^"\']*["\'][^>]*)>([\s\S]*?)</\1>',
        lambda m: _convert_attrs_in_match(m.group(1), m.group(2), m.group(3) or ""),
        result,
        flags=re.IGNORECASE,
    )
    return result


def _strip_cdata(xml_string: str) -> str:
    """Strip CDATA sections and return plain text: <![CDATA[content]]> → content"""
    return re.sub(r'<!\[CDATA\[([\s\S]*?)\]\]>', r'\1', xml_string)


def fix_xml_string(xml_string: str) -> str:
    """
    Attempt to fix common XML issues.

    Pipeline (applied in order):
    1. Strip CDATA wrappers
    2. Convert self-closing tags to open/close pairs
    3. Convert attribute format to element format
    4. Fix mismatched closing tags for simple param tags
    5. Insert missing </diff> before escape step
    6. Escape raw-text tags (content, diff, result, output)
    7. Fix unescaped ampersands
    8. Fix pseudo-tags in parentheses
    """
    fixed = xml_string

    # 1. Strip CDATA — handle <![CDATA[...]]> wrapped content
    fixed = _strip_cdata(fixed)

    # 2. Self-closing tags: <param/> → <param></param>
    fixed = _convert_self_closing_tags(fixed)

    # 3. Attribute format: <tool param="val"> → <tool><param>val</param>
    fixed = _convert_attributes_to_elements(fixed)

    # 4. Strip whitespace in tags: < write_to_file > → <write_to_file>
    fixed = _strip_whitespace_in_tags(fixed)

    # Pre-escape content in "raw text" tags that may contain XML-special characters
    # (e.g. <content> in write_to_file may contain <repo-url>, XML snippets, etc.)
    # `command` is included because shell commands frequently contain unescaped
    # angle brackets — most notably heredocs (python3 - <<'PY' ... PY) and
    # redirects (sort < in.txt > out.txt) — which would break ET.fromstring.
    _RAW_TEXT_TAGS = (
        "content", "diff", "result", "output",
        "oldString", "newString", "old_string", "new_string",
        "command",
    )

    def escape_raw_content(m: re.Match) -> str:
        tag, inner = m.group(1), m.group(2)
        if '<' in inner or '>' in inner:
            # Escape bare & first (skip already-valid entities)
            inner = re.sub(r'&(?!amp;|lt;|gt;|quot;|apos;)', '&amp;', inner)
            inner = inner.replace('<', '&lt;').replace('>', '&gt;')
        return f'<{tag}>{inner}</{tag}>'

    # 6. Fix mismatched closing tags for simple parameter tags (no nested XML in their content).
    # The model sometimes closes <path> with </diff> (anticipating the next sibling tag name).
    # e.g. <path>src/Foo.java</diff> → <path>src/Foo.java</path>
    _SIMPLE_PARAM_TAGS = (
        "path", "filePath", "source", "destination",
        "command", "description", "question", "regex",
        "oldString", "newString", "pattern",
    )
    for _tag in _SIMPLE_PARAM_TAGS:
        fixed = re.sub(
            rf'<{re.escape(_tag)}>([^<]*?)</(?!{re.escape(_tag)}\s*>)[a-zA-Z_][a-zA-Z0-9_]*\s*>',
            rf'<{_tag}>\1</{_tag}>',
            fixed,
            flags=re.IGNORECASE,
        )

    # 7. Insert missing </diff> BEFORE escaping so the escape regex can match the full <diff> block
    if re.search(r'<diff\b', fixed, re.IGNORECASE) and not re.search(r'</diff>', fixed, re.IGNORECASE):
        fixed = re.sub(
            r'</(apply_diff|replace_in_file)>',
            r'</diff>\n</\1>',
            fixed,
            flags=re.IGNORECASE,
        )

    # 7b. Insert missing </edits> when <edits> is present but not closed.
    #      The regex may match outer <edit> opener with inner </edit> closer, leaving
    #      <edits> unclosed. Keep the inner </edit>, insert </edits> after it, then
    #      ensure the outer <edit> is closed.
    if re.search(r'<edits\b', fixed, re.IGNORECASE) and not re.search(r'</edits>', fixed, re.IGNORECASE):
        fixed = re.sub(
            r'</edit>',
            r'</edit>\n</edits>\n</edit>',
            fixed,
            count=1,
            flags=re.IGNORECASE,
        )
        # After inserting </edits>, the outer <edit> may still be unclosed.
        # If XML starts with <edit ...> but doesn't end with </edit>, close it.
        m = re.match(r'\s*<\s*(\w+)', fixed)
        if m and not re.search(rf'</\s*{re.escape(m.group(1))}\s*>\s*$', fixed, re.IGNORECASE):
            fixed = fixed + f'</{m.group(1)}>'

    # 8. Escape raw-text tags
    tag_pat = '|'.join(_RAW_TEXT_TAGS)
    fixed = re.sub(
        rf'<({tag_pat})>([\s\S]*?)</\1>',
        escape_raw_content,
        fixed,
        flags=re.IGNORECASE,
    )

    # 9. Replace remaining unescaped ampersands
    fixed = re.sub(r'&(?!amp;|lt;|gt;|quot;|apos;)', '&amp;', fixed)

    # 10. Fix pseudo-tags in parentheses (e.g., "(</tag>)" -> "(tag)")
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


def extract_mcp_tool_calls(
    content: str,
    request_id: str = ""
) -> Tuple[List[XMLToolCall], str]:
    """
    Extract <use_mcp_tool> blocks from Roo Code XML-mode responses.

    Roo Code uses this format when operating in XML mode (tools=0, system-prompt
    based tool definitions) and MCP servers are connected:

        <use_mcp_tool>
          <server_name>grafana-mcp</server_name>
          <tool_name>list_datasources</tool_name>
          <arguments>{"type": "loki"}</arguments>
        </use_mcp_tool>

    The resulting tool call name follows Roo Code's native-mode convention:
    "{server_name}__{tool_name}" (e.g. "grafana-mcp__list_datasources").
    """
    log_prefix = f"[{request_id}] " if request_id else ""

    if not content or "<use_mcp_tool" not in content.lower():
        return [], content

    pattern = re.compile(
        r"<use_mcp_tool\b[^>]*>([\s\S]*?)</use_mcp_tool>",
        re.IGNORECASE,
    )

    tool_calls: List[XMLToolCall] = []
    remaining_content = content

    for match in pattern.finditer(content):
        xml_content = match.group(0)
        inner = match.group(1)

        try:
            server_match = re.search(
                r"<server_name[^>]*>([\s\S]*?)</server_name>", inner, re.IGNORECASE
            )
            server_name = server_match.group(1).strip() if server_match else ""

            tool_name_match = re.search(
                r"<tool_name[^>]*>([\s\S]*?)</tool_name>", inner, re.IGNORECASE
            )
            if not tool_name_match:
                logger.warning(f"{log_prefix}use_mcp_tool block missing <tool_name> — skipping")
                continue
            tool_name = tool_name_match.group(1).strip()

            args_match = re.search(
                r"<arguments[^>]*>([\s\S]*?)</arguments>", inner, re.IGNORECASE
            )
            arguments_raw = args_match.group(1).strip() if args_match else "{}"

            try:
                arguments = json.loads(arguments_raw) if arguments_raw else {}
            except json.JSONDecodeError:
                logger.warning(
                    f"{log_prefix}use_mcp_tool arguments not valid JSON, using raw string: "
                    f"{arguments_raw[:80]}"
                )
                arguments = {"_raw": arguments_raw}

            full_name = f"{server_name}__{tool_name}" if server_name else tool_name

            tool_call = XMLToolCall(
                name=full_name,
                arguments=arguments,
                id=generate_tool_id(xml_content),
                raw_xml=xml_content,
            )
            tool_calls.append(tool_call)
            remaining_content = remaining_content.replace(xml_content, "", 1)
            logger.debug(f"{log_prefix}Extracted MCP tool call: {full_name}")

        except Exception as e:
            logger.warning(f"{log_prefix}Failed to parse use_mcp_tool block: {e}")
            continue

    return tool_calls, remaining_content.strip()


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

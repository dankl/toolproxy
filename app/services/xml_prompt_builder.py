"""
XML system prompt builder for toolproxy.

Converts OpenAI tool definitions → XML examples for the system prompt,
enabling the OCI model (gpt-oss-120b) to output XML tool calls reliably.
"""
from typing import Dict, List, Optional


def json_schema_to_xml_example(tool_def: Dict) -> str:
    """Convert an OpenAI tool definition to an XML usage example."""
    func = tool_def.get("function", {})
    name = func.get("name", "unknown_tool")
    params = func.get("parameters", {}).get("properties", {})
    required = set(func.get("parameters", {}).get("required", []))

    lines = [f"<{name}>"]
    for param_name, param_info in params.items():
        desc = param_info.get("description", "")
        req_note = " (required)" if param_name in required else " (optional)"
        lines.append(f"  <{param_name}>{desc}{req_note}</{param_name}>")
    lines.append(f"</{name}>")
    return "\n".join(lines)


def build_xml_system_prompt(tools: List[Dict], existing_system: Optional[str] = None) -> str:
    """
    Build a system prompt that instructs the model to use XML tool calls.

    The XML format section is placed BEFORE the existing system prompt so the
    model encounters the format instruction before any long context (file trees,
    workspace info, etc.) that could dilute it.
    """
    tool_names = [t.get("function", {}).get("name", "") for t in tools if t.get("function")]
    tool_list = ", ".join(f"`{n}`" for n in tool_names if n)
    tool_examples = "\n\n".join(json_schema_to_xml_example(t) for t in tools)

    xml_section = f"""## TOOL CALLING FORMAT

You have access to these tools: {tool_list}

To call a tool, respond with ONLY the XML — nothing before or after:

<tool_name>
  <parameter_name>value</parameter_name>
</tool_name>

Example:
<read_file>
  <path>README.md</path>
</read_file>

## TOOL DEFINITIONS

{tool_examples}

## RULES

1. Your ENTIRE response must be exactly one XML tool call — nothing else
2. Use ONLY tool names from the list above
3. Name corrections: `write_file` → `write_to_file` | `open_file` → `read_file` | `apply_patch` → `apply_diff`
4. To CREATE a new file: use `write_to_file`. To MODIFY an existing file: use `apply_diff`
5. NEVER output file content as plain text — always call the tool
6. When the task is complete, call `attempt_completion`
   → A task is ONLY complete when ALL required files exist on disk.
     Write every file with write_to_file first, THEN call attempt_completion.
     Do NOT describe files in text — write them with the tool.
7. In `attempt_completion`, the `result` field must contain ONLY a short plain-text summary — NO code, NO XML, NO file contents

## FORBIDDEN FORMATS

NEVER use any of these formats — they will be ignored:
- `[assistant to=write_to_file code<|message|>{{...}}` — WRONG
- `<assistant to=write_to_file code>{{...}}` — WRONG
- JSON objects like `{{"tool": "name", "parameters": {{...}}}}` — WRONG
- Plain text descriptions of what you would do — WRONG

====

"""

    if existing_system:
        # XML format instruction goes FIRST so the model sees it before long context
        return xml_section + existing_system
    return xml_section + "You are an AI coding assistant that uses tools to complete tasks."

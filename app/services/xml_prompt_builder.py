"""
XML system prompt builder for toolproxy.

Converts OpenAI tool definitions → XML examples for the system prompt,
enabling models without native function-call support to output XML tool calls reliably.
"""
from enum import Enum
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


def _build_rules_roo_code() -> str:
    return """## RULES

1. Your ENTIRE response must be exactly one XML tool call — nothing else
2. Use ONLY tool names from the list above — NEVER invent tool names
3. To CREATE a new file: use `write_to_file` with the COMPLETE file content.
   To MODIFY an existing file:
   - First call `read_file` to get the current content
   - Then use `apply_diff` to change specific sections, OR `write_to_file` with the COMPLETE updated content
   - NEVER use `write_to_file` with only the new/changed portion — this destroys existing content
4. NEVER output file content as plain text — always call the tool
5. When the task is complete, call `attempt_completion`
   → A task is ONLY complete when ALL required files exist on disk and all required actions are done.
   → `result` must be a SHORT plain-text summary (1-2 sentences) of what was done — NOTHING ELSE.
   → NEVER put questions, options, XML, code, or file contents inside `result`.
   → Do NOT ask the user to choose — just complete the task using the most straightforward approach.
6. Name corrections: `write_file` → `write_to_file` | `open_file` → `read_file` | `apply_patch` → `apply_diff`

## CRITICAL RULE — renaming or moving files and folders

To rename or move any file or folder: use `move_file` with `<source>` and `<destination>`.
- NEVER use `rename`, `move`, or any other name — ALWAYS `move_file`
- NEVER call `read_file` before `move_file` — `move_file` works on paths, not content
- NEVER do copy+delete (write_to_file the content, then delete_file the original) — use `move_file`

WRONG (never do this):
[read_file test/Witze.md]    ← unnecessary
[write_to_file witzeordner/Witze.md ...]    ← manual copy
[delete_file test/Witze.md]    ← manual delete
[delete_folder test]    ← manual delete

CORRECT:
```
<move_file>
<source>test</source>
<destination>witzeordner</destination>
</move_file>
```

## CRITICAL EXAMPLE — attempt_completion with pending actions

WRONG (never do this — XML inside result is displayed as text, NOT executed):
```
<attempt_completion>
<result>
<delete_file><path>ai-guide/README.md</path></delete_file>
<delete_folder><path>ai-guide</path></delete_folder>
</result>
</attempt_completion>
```

CORRECT — perform each action first, then summarize in plain text:
```
<delete_file>
<path>ai-guide/README.md</path>
</delete_file>
```
[after file is deleted, next turn:]
```
<delete_folder>
<path>ai-guide</path>
</delete_folder>
```
[after folder is deleted, next turn:]
```
<attempt_completion>
<result>Deleted ai-guide/README.md and the ai-guide folder.</result>
</attempt_completion>
```

## FORBIDDEN FORMATS

NEVER use any of these formats — they will be ignored:
- `[assistant to=write_to_file code<|message|>{...}` — WRONG
- `<assistant to=write_to_file code>{...}` — WRONG
- JSON objects like `{"tool": "name", "parameters": {...}}` — WRONG
- YAML-style like `[write_to_file]\npath: foo\ncontent: |` — WRONG
- Plain text descriptions of what you would do — WRONG
- Describing updated file content in a code block — WRONG (see example below)

## CRITICAL EXAMPLE — updating a file

WRONG (never do this):
```
Hier ist der aktualisierte guides/ai-guide.md:
```markdown
# AI Guide
...
```
```

CORRECT (always do this):
```
<write_to_file>
<path>guides/ai-guide.md</path>
<content># AI Guide
...
</content>
</write_to_file>
```

No matter what language you think in — your ENTIRE response must be the XML tool call. No preamble, no explanation, no code fences around the XML.

## CRITICAL EXAMPLE — adding content to an existing file

WRONG (never do this — overwrites the entire file with only the new part):
```
<write_to_file>
<path>guides/ai-guide.md</path>
<content>## RagFlow

Only the new section...
</content>
</write_to_file>
```

CORRECT option 1 — read the file first, then write the COMPLETE updated content:
```
<read_file>
<path>guides/ai-guide.md</path>
</read_file>
```
[after seeing the current content, call write_to_file with the FULL file including the new section]

CORRECT option 2 — use apply_diff to append a section:
```
<apply_diff>
<path>guides/ai-guide.md</path>
<diff>
@@ ... @@
+## RagFlow
+
+New section content here...
</diff>
</apply_diff>
```

====

"""


def _build_rules_open_code() -> str:
    return """## RULES

1. Your ENTIRE response must be exactly one XML tool call — nothing else
2. Use ONLY tool names from the list above
3. File operations: `read_file` to read | `write_to_file` to create/overwrite | `edit` to modify sections
4. When modifying an existing file: use `edit` with oldString/newString — do NOT rewrite the entire file with `write_to_file` unless creating it fresh
5. NEVER output file content as plain text — always call the tool
6. To run shell commands: use `bash`
7. To ask a clarifying question: use `question`
8. When done with ALL steps (every required file written, every command run): respond with a brief plain-text summary (no XML needed)
   → Only use this AFTER all files have been written with the `write_to_file` tool. Never skip writing a file and describe its content in text instead.

## CRITICAL EXAMPLE — creating a file

WRONG (never do this):
```
Hier ist der Inhalt für die Datei witze.md:
# 5 Witze
1. ...
Du kannst diesen Text in witze.md kopieren.
```

CORRECT (always do this):
```
<write_to_file>
<path>witze.md</path>
<content># 5 Witze
1. ...
</content>
</write_to_file>
```

Rule 5 overrides Rule 8: if a file still needs to be written, ALWAYS use the `write_to_file` tool — never describe the content in text.

## FORBIDDEN FORMATS

NEVER use any of these formats — they will be ignored:
- JSON objects like `{"tool": "name", "parameters": {...}}` — WRONG
- YAML-style — WRONG
- Plain text descriptions of what you would do — WRONG
- Code blocks containing file content — WRONG (use `write` or `edit` tool instead)

====

"""


def _build_rules_generic() -> str:
    return """## RULES

1. Your ENTIRE response must be exactly one XML tool call — nothing else
2. Use ONLY tool names from the list above
3. NEVER output content as plain text — always call the tool

====

"""


def build_xml_system_prompt(tools: List[Dict], existing_system: Optional[str] = None, client_type=None) -> str:
    """
    Build a system prompt that instructs the model to use XML tool calls.

    The XML format section is placed BEFORE the existing system prompt so the
    model encounters the format instruction before any long context (file trees,
    workspace info, etc.) that could dilute it.
    """
    # Avoid circular import — ClientType is defined in main.py but we receive it as a value
    from enum import Enum

    class _CT(Enum):
        ROO_CODE = "roo_code"
        OPEN_CODE = "open_code"
        GENERIC = "generic"

    if client_type is None or getattr(client_type, "value", None) == "roo_code":
        rules = _build_rules_roo_code()
    elif getattr(client_type, "value", None) == "open_code":
        rules = _build_rules_open_code()
    else:
        rules = _build_rules_generic()

    tool_names = [t.get("function", {}).get("name", "") for t in tools if t.get("function")]
    tool_list = ", ".join(f"`{n}`" for n in tool_names if n)
    tool_examples = "\n\n".join(json_schema_to_xml_example(t) for t in tools)

    # Pick a representative read-tool for the example
    read_tool = next((n for n in tool_names if n in ("read_file", "read")), tool_names[0] if tool_names else "read_file")
    read_param = "filePath" if read_tool == "read" else "path"

    xml_section = f"""## TOOL CALLING FORMAT

You have access to these tools: {tool_list}

To call a tool, respond with ONLY the XML — nothing before or after:

<tool_name>
  <parameter_name>value</parameter_name>
</tool_name>

Example:
<{read_tool}>
  <{read_param}>README.md</{read_param}>
</{read_tool}>

## TOOL DEFINITIONS

{tool_examples}

{rules}"""

    if existing_system:
        # XML format instruction goes FIRST so the model sees it before long context
        return xml_section + existing_system
    return xml_section + "You are an AI coding assistant that uses tools to complete tasks."

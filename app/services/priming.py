"""
Single-turn priming for toolproxy.

Injects synthetic (question, XML-answer) pairs on the first turn so the model
learns the expected XML format before it sees the real task.
"""
import logging
from typing import Dict, List

from app.services.tool_mapping import ClientType

logger = logging.getLogger(__name__)

_PRIMING_PREFERRED: Dict[str, List[str]] = {
    ClientType.ROO_CODE.value: ["read_file", "write_to_file", "attempt_completion"],
    ClientType.OPEN_CODE.value: ["read_file", "write_to_file", "bash"],
    ClientType.CLINE.value: ["read_file", "write_to_file", "replace_in_file"],
    ClientType.GENERIC.value: [],
}

_PRIMING_QUESTIONS: Dict[str, List[str]] = {
    ClientType.ROO_CODE.value: [
        "Read the file README.md and show me what's in it.",
        "Create a new file called hello.py with a Hello World function.",
        "The task is complete.",
    ],
    ClientType.OPEN_CODE.value: [
        "Read the file README.md and show me what's in it.",
        "Create a new file called hello.py.",
        "Run the tests.",
    ],
    ClientType.CLINE.value: [
        "Read the file README.md and show me what's in it.",
        "Create a new file called hello.py with a Hello World function.",
        "Fix the typo 'recieve' in src/utils.py.",
    ],
}

# Static priming sequences injected unconditionally before the dynamic tool pairs.
# Each sequence is a list of {"role", "content"} messages and may span multiple
# turns to teach the model multi-step patterns (e.g. read → write).
# This ensures the model reliably outputs XML even when the system prompt is
# dropped by upstream adapters.
_STATIC_PRIMING_SEQUENCES: Dict[str, List[List[Dict]]] = {
    ClientType.OPEN_CODE.value: [
        # glob — find files by pattern
        [
            {"role": "user", "content": "Find all Python files in the project."},
            {"role": "assistant", "content": "<glob>\n<pattern>**/*.py</pattern>\n</glob>"},
        ],
        # grep — search file contents by regex
        [
            {"role": "user", "content": "Search for TODO comments in the codebase."},
            {"role": "assistant", "content": "<grep>\n<pattern>TODO</pattern>\n</grep>"},
        ],
        # edit — replace a string inside a file
        [
            {"role": "user", "content": "In main.py fix the typo: replace 'recieve' with 'receive'."},
            {"role": "assistant", "content": "<edit>\n<filePath>main.py</filePath>\n<oldString>recieve</oldString>\n<newString>receive</newString>\n</edit>"},
        ],
        # bash — run shell commands
        [
            {"role": "user", "content": "Run the test suite."},
            {"role": "assistant", "content": "<bash>\n<command>pytest -q</command>\n<description>run test suite</description>\n</bash>"},
        ],
    ],
    ClientType.ROO_CODE.value: [
        # Single-turn: rename / move — use execute_command with mv (move_file is a Roo builtin
        # not in tools[] so Roo Code silently drops it; execute_command IS in tools[])
        [
            {"role": "user", "content": "Rename the folder 'old_folder' to 'new_folder'."},
            {"role": "assistant", "content": "<execute_command>\n<command>mv old_folder new_folder</command>\n</execute_command>"},
        ],
        # Two-turn: append to file — read first, then write full updated content
        [
            {"role": "user", "content": "Add 3 new items to the list in notes.md."},
            {"role": "assistant", "content": "<read_file>\n<path>notes.md</path>\n</read_file>"},
            {"role": "user", "content": "[Tool Result]\n1. Buy milk\n2. Buy eggs"},
            {"role": "assistant", "content": "<write_to_file>\n<path>notes.md</path>\n<content>1. Buy milk\n2. Buy eggs\n3. Call dentist\n4. Water plants\n5. Fix bike</content>\n</write_to_file>"},
        ],
    ],
    ClientType.CLINE.value: [
        # replace_in_file with SEARCH/REPLACE block format
        [
            {"role": "user", "content": "Fix the typo 'recieve' in src/utils.py."},
            {"role": "assistant", "content": "<replace_in_file>\n<path>src/utils.py</path>\n<diff>\n<<<<<<< SEARCH\ndef recieve_data(payload):\n=======\ndef receive_data(payload):\n>>>>>>> REPLACE\n</diff>\n</replace_in_file>"},
        ],
        # Two-turn: append to file — read first, then write full updated content
        [
            {"role": "user", "content": "Add a section to README.md."},
            {"role": "assistant", "content": "<read_file>\n<path>README.md</path>\n</read_file>"},
            {"role": "user", "content": "[Tool Result]\n# Project\n\nDescription here."},
            {"role": "assistant", "content": "<write_to_file>\n<path>README.md</path>\n<content># Project\n\nDescription here.\n\n## Installation\n\nRun `npm install`.\n</content>\n</write_to_file>"},
        ],
    ],
}


def _make_prime_xml(tool_name: str, params: Dict, required: List[str]) -> str:
    """Build a plausible XML example call for a tool.

    Always uses <path> as the canonical path parameter — even for tools that use
    'filePath' in their schema. The schema-aware remapping (_remap_args_to_schema)
    translates back to the correct name after parsing, so the model learns one
    consistent XML convention.
    """
    has_path = "path" in params or "filePath" in params
    if has_path and "content" in params:
        return f"<{tool_name}>\n<path>example.md</path>\n<content>example content</content>\n</{tool_name}>"
    if has_path:
        return f"<{tool_name}>\n<path>README.md</path>\n</{tool_name}>"
    param = required[0] if required else (list(params.keys())[0] if params else "input")
    return f"<{tool_name}>\n<{param}>example</{param}>\n</{tool_name}>"


def inject_priming(
    messages: List[Dict],
    tools: List[Dict],
    client_type: ClientType = ClientType.ROO_CODE,
) -> List[Dict]:
    """
    Inject up to 3 synthetic (question, XML-answer) pairs before the first user
    message when the conversation has only 1 user turn.

    Three examples cover all major tool categories the model will encounter.
    More examples = stronger XML-format prior, suppressing the model's own
    chat-template format leak ([assistant to=...] / <assistant to=...>).
    """
    if not tools:
        return messages

    tool_map: Dict[str, Dict] = {}
    for t in tools:
        fn = t.get("function", {})
        n = fn.get("name", "")
        if n:
            tool_map[n] = fn

    preferred = _PRIMING_PREFERRED.get(client_type.value, [])
    questions = _PRIMING_QUESTIONS.get(client_type.value, _PRIMING_QUESTIONS[ClientType.ROO_CODE.value])

    # Select up to 3 representative tools in priority order
    candidates = []
    for preferred_name in preferred:
        if preferred_name in tool_map:
            candidates.append(preferred_name)
    # Fill remaining slots from whatever tools are available
    for name in tool_map:
        if name not in candidates:
            candidates.append(name)
        if len(candidates) >= 3:
            break

    prime_qa = list(zip(questions, preferred or candidates))

    priming: List[Dict] = []

    # Prepend static sequences (rename example, read→write example, etc.)
    for sequence in _STATIC_PRIMING_SEQUENCES.get(client_type.value, []):
        priming.extend(sequence)

    for (question, _), tool_name in zip(prime_qa, candidates):
        fn = tool_map[tool_name]
        params = fn.get("parameters", {}).get("properties", {})
        required = fn.get("parameters", {}).get("required", [])
        xml = _make_prime_xml(tool_name, params, required)
        priming.append({"role": "user", "content": question})
        priming.append({"role": "assistant", "content": xml})

    # Insert before the first user message (after system prompt)
    result: List[Dict] = []
    inserted = False
    for msg in messages:
        if msg.get("role") == "user" and not inserted:
            result.extend(priming)
            inserted = True
        result.append(msg)
    return result

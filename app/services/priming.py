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

# Static priming sequences injected before the dynamic tool pairs.
# Each entry is a dict with:
#   "requires": list of tool names that must ALL be present in the session's
#               tool_map for this sequence to be injected.  Empty list = always.
#   "messages": list of {"role", "content"} messages for the sequence.
#
# Gating on "requires" means:
#   - The model only sees examples for tools it actually has access to.
#   - Sessions with a minimal tool set don't pay the token cost of irrelevant
#     sequences (e.g. no apply_diff priming when apply_diff isn't available).
_STATIC_PRIMING_SEQUENCES: Dict[str, List[Dict]] = {
    ClientType.OPEN_CODE.value: [
        # glob — find files by pattern
        {
            "requires": ["glob"],
            "messages": [
                {"role": "user", "content": "Find all Python files in the project."},
                {"role": "assistant", "content": "<glob>\n<pattern>**/*.py</pattern>\n</glob>"},
            ],
        },
        # grep — search file contents by regex
        {
            "requires": ["grep"],
            "messages": [
                {"role": "user", "content": "Search for TODO comments in the codebase."},
                {"role": "assistant", "content": "<grep>\n<pattern>TODO</pattern>\n</grep>"},
            ],
        },
        # edit — replace a string inside a file
        {
            "requires": ["edit"],
            "messages": [
                {"role": "user", "content": "In main.py fix the typo: replace 'recieve' with 'receive'."},
                {"role": "assistant", "content": "<edit>\n<filePath>main.py</filePath>\n<oldString>recieve</oldString>\n<newString>receive</newString>\n</edit>"},
            ],
        },
        # bash — run shell commands
        {
            "requires": ["bash"],
            "messages": [
                {"role": "user", "content": "Run the test suite."},
                {"role": "assistant", "content": "<bash>\n<command>pytest -q</command>\n<description>run test suite</description>\n</bash>"},
            ],
        },
    ],
    ClientType.ROO_CODE.value: [
        # Single-turn: rename / move — use execute_command with mv (move_file is a Roo builtin
        # not in tools[] so Roo Code silently drops it; execute_command IS in tools[])
        {
            "requires": ["execute_command"],
            "messages": [
                {"role": "user", "content": "Rename the folder 'old_folder' to 'new_folder'."},
                {"role": "assistant", "content": "<execute_command>\n<command>mv old_folder new_folder</command>\n</execute_command>"},
            ],
        },
        # Two-turn: append to file — read first, then write full updated content
        {
            "requires": ["read_file", "write_to_file"],
            "messages": [
                {"role": "user", "content": "Add 3 new items to the list in notes.md."},
                {"role": "assistant", "content": "<read_file>\n<path>notes.md</path>\n</read_file>"},
                {"role": "user", "content": "[Tool Result]\n1. Buy milk\n2. Buy eggs"},
                {"role": "assistant", "content": "<write_to_file>\n<path>notes.md</path>\n<content>1. Buy milk\n2. Buy eggs\n3. Call dentist\n4. Water plants\n5. Fix bike</content>\n</write_to_file>"},
            ],
        },
        # Three-turn: write succeeds → attempt_completion (NEVER simulate [Tool Result]).
        # Teaches: [Tool Result] always comes from the user turn, never from the assistant.
        # After a successful write, call attempt_completion — do NOT keep rewriting.
        {
            "requires": ["write_to_file", "attempt_completion"],
            "messages": [
                {"role": "user", "content": "Fix the Windows batch script to use the native .bat launcher."},
                {"role": "assistant", "content": "<write_to_file>\n<path>Tools/start.bat</path>\n<content>@echo off\nset TOOL=%~dp0bin\\tool.bat\ncall \"%TOOL%\" %*\n</content>\n</write_to_file>"},
                {"role": "user", "content": "[Tool Result]\nThe content was successfully saved to Tools/start.bat."},
                {"role": "assistant", "content": "<attempt_completion>\n<result>Fixed Tools/start.bat to call the native tool.bat launcher instead of the shell script.</result>\n</attempt_completion>"},
            ],
        },
        # Single-turn: apply_diff — Roo Code SEARCH/REPLACE format (NOT unified diff).
        # Teaches: diff content uses <<<<<<< SEARCH / ======= / >>>>>>> REPLACE markers,
        # not the --- a/file / +++ b/file unified diff format the model defaults to.
        {
            "requires": ["apply_diff"],
            "messages": [
                {"role": "user", "content": "Fix the typo 'pritn' in app.py using apply_diff."},
                {"role": "assistant", "content": "<apply_diff>\n<path>app.py</path>\n<diff>\n<<<<<<< SEARCH\npritn('hello')\n=======\nprint('hello')\n>>>>>>> REPLACE\n</diff>\n</apply_diff>"},
            ],
        },
        # Two-turn: truncated file — write directly, NEVER simulate [Tool Result]
        # Uses the exact Roo Code truncation format so the model recognises it.
        {
            "requires": ["read_file", "write_to_file"],
            "messages": [
                {"role": "user", "content": "Fix the pom.xml to include the missing test dependency."},
                {"role": "assistant", "content": "<read_file>\n<path>taskmanager/backend/pom.xml</path>\n</read_file>"},
                {
                    "role": "user",
                    "content": (
                        "[Tool Result]\n"
                        "File: taskmanager/backend/pom.xml\n"
                        "IMPORTANT: File content truncated.\n"
                        "Status: Showing lines 1-40 of 83 total lines.\n"
                        "To read more: Use the read_file tool with offset=41 and limit=30.\n"
                        "\n"
                        " 1 | <?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
                        " 2 | <project xmlns=\"http://maven.apache.org/POM/4.0.0\">\n"
                        " 3 |   <modelVersion>4.0.0</modelVersion>\n"
                        " 4 |   <groupId>com.example</groupId>\n"
                        " 5 |   <artifactId>taskmanager</artifactId>\n"
                        " 6 |   <dependencies>"
                    ),
                },
                {
                    "role": "assistant",
                    "content": (
                        "<write_to_file>\n"
                        "<path>taskmanager/backend/pom.xml</path>\n"
                        "<content><?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
                        "<project xmlns=\"http://maven.apache.org/POM/4.0.0\">\n"
                        "  <modelVersion>4.0.0</modelVersion>\n"
                        "  <groupId>com.example</groupId>\n"
                        "  <artifactId>taskmanager</artifactId>\n"
                        "  <version>0.0.1-SNAPSHOT</version>\n"
                        "  <dependencies>\n"
                        "    <dependency>\n"
                        "      <groupId>org.springframework.boot</groupId>\n"
                        "      <artifactId>spring-boot-starter-web</artifactId>\n"
                        "    </dependency>\n"
                        "    <dependency>\n"
                        "      <groupId>org.springframework.boot</groupId>\n"
                        "      <artifactId>spring-boot-starter-test</artifactId>\n"
                        "      <scope>test</scope>\n"
                        "    </dependency>\n"
                        "  </dependencies>\n"
                        "</project>\n"
                        "</content>\n"
                        "</write_to_file>"
                    ),
                },
            ],
        },
    ],
    ClientType.CLINE.value: [
        # replace_in_file with SEARCH/REPLACE block format
        {
            "requires": ["replace_in_file"],
            "messages": [
                {"role": "user", "content": "Fix the typo 'recieve' in src/utils.py."},
                {"role": "assistant", "content": "<replace_in_file>\n<path>src/utils.py</path>\n<diff>\n<<<<<<< SEARCH\ndef recieve_data(payload):\n=======\ndef receive_data(payload):\n>>>>>>> REPLACE\n</diff>\n</replace_in_file>"},
            ],
        },
        # Two-turn: append to file — read first, then write full updated content
        {
            "requires": ["read_file", "write_to_file"],
            "messages": [
                {"role": "user", "content": "Add a section to README.md."},
                {"role": "assistant", "content": "<read_file>\n<path>README.md</path>\n</read_file>"},
                {"role": "user", "content": "[Tool Result]\n# Project\n\nDescription here."},
                {"role": "assistant", "content": "<write_to_file>\n<path>README.md</path>\n<content># Project\n\nDescription here.\n\n## Installation\n\nRun `npm install`.\n</content>\n</write_to_file>"},
            ],
        },
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

    Static sequences are gated on their "requires" list — a sequence is only
    injected when all its required tools are present in the session's tool_map.
    This avoids teaching formats for tools the model cannot use and keeps the
    token cost proportional to the actual tool set.
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

    # Prepend static sequences — only inject when all required tools are present.
    for entry in _STATIC_PRIMING_SEQUENCES.get(client_type.value, []):
        required_tools = entry.get("requires", [])
        if all(t in tool_map for t in required_tools):
            priming.extend(entry["messages"])
            logger.debug(
                f"priming: injecting sequence {entry['messages'][0]['content'][:40]!r} "
                f"(requires={required_tools})"
            )
        else:
            missing = [t for t in required_tools if t not in tool_map]
            logger.debug(f"priming: skipping sequence (missing tools: {missing})")

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

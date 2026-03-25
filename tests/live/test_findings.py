"""
Live regression tests for findings from the stability analysis.
See docs/stabilitaets-analyse.md for full issue descriptions.

Run:
    python3 -m pytest -m live -v tests/live/test_findings.py
"""
import shlex
import pytest
from tests.live.conftest import (
    ROO_CODE_TOOLS,
    TOOL_WRITE_TO_FILE, TOOL_ATTEMPT_COMPLETION, TOOL_EXECUTE_COMMAND,
    post, parse_tool_call,
)

pytestmark = pytest.mark.live


def post_multi_turn(client, tools, messages):
    """Post a full pre-built message list rather than a single user prompt."""
    import os
    model = os.environ.get("LIVE_MODEL", "openai/gpt-oss-120b")
    resp = client.post("/v1/chat/completions", json={
        "model": model,
        "messages": messages,
        "tools": tools,
    })
    assert resp.status_code == 200, resp.text
    return resp.json()


class TestLoopDetectionFalsePositive:
    """
    Finding: Loop detection fires on error messages containing success words.

    Scenario: two consecutive tool results where the first contains "created"
    inside an error message. The proxy should NOT inject a stop hint — the
    model must be allowed to continue writing files.

    If the bug is present: proxy injects STOP hint → model calls attempt_completion
    If the bug is fixed:   proxy stays silent   → model calls write_to_file
    """

    def test_error_message_with_success_word_does_not_trigger_loop(self, live):
        messages = [
            {"role": "system", "content": "You are a coding assistant. Use tools to complete tasks."},
            {"role": "user", "content": "Write hello.py and then goodbye.py."},
            # Simulated assistant turn: model called write_to_file for hello.py
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_001",
                "type": "function",
                "function": {"name": "write_to_file", "arguments": '{"path": "hello.py", "content": "print(\'hello\')"}'},
            }]},
            # Tool result 1: error message containing the word "created" → should NOT count as success
            {"role": "tool", "tool_call_id": "call_001", "content": "Error: file could not be created — permission denied"},
            # Simulated retry: model calls write_to_file again
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_002",
                "type": "function",
                "function": {"name": "write_to_file", "arguments": '{"path": "hello.py", "content": "print(\'hello\')"}'},
            }]},
            # Tool result 2: genuine success
            {"role": "tool", "tool_call_id": "call_002", "content": "File written successfully"},
            # Now ask for the next file — proxy should NOT have injected STOP hint
            {"role": "user", "content": "Good. Now write goodbye.py with print('goodbye')."},
        ]

        resp = post_multi_turn(live, ROO_CODE_TOOLS, messages)
        name, args = parse_tool_call(resp)

        # If loop detection false-positived, proxy injected a STOP hint and the
        # model likely called attempt_completion instead of writing the file.
        assert name == "write_to_file", (
            f"Expected write_to_file but got {name!r} — "
            "loop detection likely fired on the error message containing 'created'"
        )
        assert args.get("path") == "goodbye.py"


class TestShellInjection:
    """
    Finding: move_file → execute_command conversion uses single-quoted paths.
    A path containing an apostrophe breaks the shell command.

    The proxy converts move_file(source="user's notes.txt", destination="notes.txt")
    to execute_command(command="mv 'user's notes.txt' 'notes.txt'") — broken shell syntax.

    After fix (shlex.quote): mv 'user'"'"'s notes.txt' notes.txt — valid shell.
    """

    def test_apostrophe_in_path_produces_valid_shell_command(self, live):
        # Only offer execute_command so the model is forced to use move_file or
        # execute_command directly. The proxy converts move_file → execute_command.
        tools = [TOOL_EXECUTE_COMMAND, TOOL_ATTEMPT_COMPLETION]
        resp = post(live, tools,
                    "Rename the file \"user's notes.txt\" to \"users_notes.txt\". "
                    "Use move_file or execute_command.")
        name, args = parse_tool_call(resp)

        assert name == "execute_command", f"Expected execute_command, got {name!r}"
        cmd = args.get("command", "")
        assert cmd, "execute_command has no 'command' argument"

        # The command must be parseable by the shell — shlex.split() raises
        # ValueError if the quoting is broken (e.g. unmatched single quote).
        try:
            parts = shlex.split(cmd)
        except ValueError as e:
            pytest.fail(
                f"Shell command has broken quoting (apostrophe injection bug): {cmd!r}\n"
                f"shlex error: {e}"
            )

        # The parsed command must contain the filename with the apostrophe intact.
        full_cmd = " ".join(parts)
        assert "user" in full_cmd and "notes" in full_cmd, (
            f"Filename not found in parsed command: {parts}"
        )


class TestApplyDiffSearchReplaceMarkers:
    """
    Bug: apply_diff with <<<<<<< SEARCH markers failed to parse because the diff
    content is invalid XML. fix_xml_string must escape the markers so ET can parse them.

    If the bug is present: proxy returns attempt_completion fallback instead of apply_diff
    If the bug is fixed:   proxy returns apply_diff with diff argument as a string
    """

    def test_apply_diff_search_replace_parsed_as_string(self, live):
        messages = [
            {"role": "system", "content": "You are a coding assistant. Use tools to complete tasks."},
            {"role": "user", "content": "Change the line 'include: [\"src\"]' to 'include: [\"src/**/*\"]' in tsconfig.json using apply_diff."},
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_001",
                "type": "function",
                "function": {"name": "read_file", "arguments": '{"path": "tsconfig.json"}'},
            }]},
            {"role": "tool", "tool_call_id": "call_001", "content": '{\n  "compilerOptions": {},\n  "include": ["src"]\n}\n'},
            {"role": "user", "content": "Now apply the change with apply_diff."},
        ]
        resp = post_multi_turn(live, ROO_CODE_TOOLS, messages)
        name, args = parse_tool_call(resp)

        assert name == "apply_diff", f"Expected apply_diff, got {name!r}"
        diff = args.get("diff", "")
        assert isinstance(diff, str), f"diff must be a string, got {type(diff)}"
        assert diff.strip(), "diff must not be empty"


class TestWriteToFileJsxContent:
    """
    Bug: write_to_file with JSX content (e.g. <div>, <h1>) is accidentally valid XML.
    ET parsed the content tag as having child elements → xml_element_to_dict returned
    a dict instead of a string → audit_log crashed with AttributeError.

    If the bug is present: proxy returns 500 or content argument is a dict
    If the bug is fixed:   proxy returns write_to_file with content as a plain string
    """

    def test_jsx_content_argument_is_string(self, live):
        resp = post(
            live,
            ROO_CODE_TOOLS,
            "Write a minimal React component to src/App.tsx. "
            "It should render a <div> with an <h1> saying 'Hello'.",
        )
        name, args = parse_tool_call(resp)

        assert name == "write_to_file", f"Expected write_to_file, got {name!r}"
        content = args.get("content", "")
        assert isinstance(content, str), (
            f"content must be a string, got {type(content)} — "
            "JSX inside <content> was likely parsed as XML child elements"
        )
        assert content.strip(), "content must not be empty"


class TestMultiTurnNormalization:
    """
    Finding: tool_calls key not removed from assistant messages after normalization.
    Upstream receives both content (XML) and tool_calls — potential schema violation.

    This test verifies that multi-turn conversations with prior tool_calls in history
    work correctly end-to-end (the upstream doesn't reject the request).
    """

    def test_multi_turn_with_prior_tool_call_in_history(self, live):
        messages = [
            {"role": "system", "content": "You are a coding assistant. Use tools to complete tasks."},
            {"role": "user", "content": "Read README.md"},
            # Prior assistant turn with tool_calls in history — proxy must normalize this
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_001",
                "type": "function",
                "function": {"name": "read_file", "arguments": '{"path": "README.md"}'},
            }]},
            {"role": "tool", "tool_call_id": "call_001", "content": "# My Project\nThis is the README."},
            {"role": "user", "content": "Now write a file called summary.md with a one-line summary of what you read."},
        ]

        resp = post_multi_turn(live, ROO_CODE_TOOLS, messages)
        name, args = parse_tool_call(resp)

        assert name == "write_to_file", f"Expected write_to_file, got {name!r}"
        assert args.get("path") == "summary.md"
        assert "content" in args

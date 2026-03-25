"""
Failing tests that PROVE the bugs found in static analysis exist.

Each test is written to FAIL against the current code and PASS after the fix
is applied. Run before fixing to see red, run after fixing to see green.

See docs/stabilitaets-analyse.md for full issue descriptions.
See docs/fix-plan.md for the corresponding fixes.
"""
import shlex
import json
import pytest

from app.services.loop_detection import detect_success_loop
from app.services.message_normalizer import normalize_messages
from app.services.tool_call_fixups import (
    convert_move_file_to_execute_command,
    fix_ask_followup_question_params,
)


# ─────────────────────────────────────────────────────────────────────────────
# BUG 1: Loop detection false positive
# File: app/services/loop_detection.py:60
#
# Word-matching fires even when the tool result is an ERROR.
# "Error: file could not be created" contains "created" → increments counter.
#
# Fix: also check that the content does NOT contain error-indicating words.
# ─────────────────────────────────────────────────────────────────────────────

class TestLoopDetectionFalsePositive:

    def test_error_message_with_success_word_does_not_trigger_loop(self):
        """
        Two tool results: first is an ERROR containing 'created', second is a
        real success. The loop hint must NOT fire — only 1 real success exists.

        FAILS today:  'created' in error message increments counter to 1,
                      'successfully' increments to 2 → stop hint injected (wrong)
        PASSES after fix: error indicator suppresses the first increment → count=1 → no hint
        """
        messages = [
            {"role": "user", "content": "Write hello.py"},
            {"role": "user", "content": "[Tool Result]\nError: file could not be created — permission denied"},
            {"role": "user", "content": "[Tool Result]\nFile written successfully"},
        ]
        result = detect_success_loop(messages)
        assert result is None, (
            f"BUG: Loop detection fired on an error message.\n"
            f"Hint injected: {result!r}\n"
            f"Cause: 'created' in error message counted as a success."
        )

    def test_two_real_successes_still_trigger_loop(self):
        """
        Sanity check: two genuine successes must still trigger the loop hint.
        This must keep passing before AND after the fix.
        """
        messages = [
            {"role": "user", "content": "Write some files"},
            {"role": "user", "content": "[Tool Result]\nFile written successfully"},
            {"role": "user", "content": "[Tool Result]\nFile created successfully"},
        ]
        result = detect_success_loop(messages)
        assert result is not None, "Two real successes must still trigger the loop hint"

    def test_error_only_never_triggers_loop(self):
        """
        Two error messages — even if both contain success words — must not trigger.

        FAILS today: both contain "created" → count=2 → hint fires (wrong)
        PASSES after fix: error indicator suppresses both → count=0 → no hint
        """
        messages = [
            {"role": "user", "content": "Write some files"},
            {"role": "user", "content": "[Tool Result]\nError: directory could not be created"},
            {"role": "user", "content": "[Tool Result]\nError: file was not written — disk full"},
        ]
        result = detect_success_loop(messages)
        assert result is None, (
            f"BUG: Loop detection fired on two error messages.\n"
            f"Hint injected: {result!r}\n"
            f"Cause: success words in error messages counted as successes."
        )


# ─────────────────────────────────────────────────────────────────────────────
# BUG 2: Shell injection via apostrophe in file path
# File: app/services/tool_call_fixups.py:288
#
# cmd = f"mv '{source}' '{dest}'"
# A path like "user's notes.txt" produces: mv 'user's notes.txt' 'dest'
# → broken shell syntax (unmatched single quote)
#
# Fix: use shlex.quote(source) and shlex.quote(dest)
# ─────────────────────────────────────────────────────────────────────────────

class TestShellInjectionApostrophe:

    def _get_mv_command(self, source: str, dest: str) -> str:
        """Helper: run the move_file fixup and return the mv command string."""
        tool_calls = [{
            "id": "call_001",
            "type": "function",
            "function": {
                "name": "move_file",
                "arguments": json.dumps({"source": source, "destination": dest}),
            },
        }]
        result = convert_move_file_to_execute_command(
            tool_calls, ["execute_command"], request_id="test"
        )
        assert result, "Expected a tool call result"
        assert result[0]["function"]["name"] == "execute_command"
        return json.loads(result[0]["function"]["arguments"])["command"]

    def test_apostrophe_in_source_path_produces_valid_shell_command(self):
        """
        source = "user's notes.txt" → mv command must be valid shell syntax.

        FAILS today:  mv 'user's notes.txt' 'dest.txt'  ← broken quoting
        PASSES after fix: shlex.quote produces valid escaping
        """
        cmd = self._get_mv_command("user's notes.txt", "dest.txt")
        try:
            parts = shlex.split(cmd)
        except ValueError as e:
            pytest.fail(
                f"BUG: Shell command has broken quoting (apostrophe injection).\n"
                f"Command: {cmd!r}\n"
                f"shlex error: {e}\n"
                f"Fix: use shlex.quote() instead of f\"mv '{{source}}' '{{dest}}'\""
            )
        # After correct parsing, the original filename must be intact
        assert "user's notes.txt" in parts, (
            f"Filename not preserved after quoting. Parsed parts: {parts}"
        )

    def test_apostrophe_in_dest_path_produces_valid_shell_command(self):
        """Apostrophe in destination path must also be handled."""
        cmd = self._get_mv_command("source.txt", "laura's folder/dest.txt")
        try:
            parts = shlex.split(cmd)
        except ValueError as e:
            pytest.fail(
                f"BUG: Apostrophe in destination path breaks shell command.\n"
                f"Command: {cmd!r}\n"
                f"shlex error: {e}"
            )
        assert "laura's folder/dest.txt" in parts

    def test_normal_path_without_apostrophe_still_works(self):
        """Sanity check: normal paths must keep working before and after the fix."""
        cmd = self._get_mv_command("old_folder", "new_folder")
        parts = shlex.split(cmd)
        assert parts == ["mv", "old_folder", "new_folder"]


# ─────────────────────────────────────────────────────────────────────────────
# BUG 3: tool_calls key not removed after normalization
# File: app/services/message_normalizer.py:112
#
# After converting tool_calls to XML in content, the tool_calls key is not
# popped from the message dict. The upstream receives both keys.
#
# Fix: add msg.pop("tool_calls", None) after writing XML to content.
# ─────────────────────────────────────────────────────────────────────────────

class TestToolCallsKeyNotRemovedAfterNormalization:

    def test_normalized_assistant_message_has_no_tool_calls_key(self):
        """
        An assistant message with tool_calls must NOT have the tool_calls key
        after normalization — it must be replaced by XML in content.

        FAILS today:  tool_calls key is still present after normalization
        PASSES after fix: msg.pop("tool_calls", None) removes it
        """
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_001",
                    "type": "function",
                    "function": {
                        "name": "write_to_file",
                        "arguments": json.dumps({"path": "foo.py", "content": "print('hi')"}),
                    },
                }],
            }
        ]
        normalized = normalize_messages(messages, request_id="test")
        msg = normalized[0]

        assert "tool_calls" not in msg, (
            f"BUG: tool_calls key still present after normalization.\n"
            f"Upstream receives both content (XML) and tool_calls (OpenAI format).\n"
            f"Keys present: {list(msg.keys())}\n"
            f"Fix: add msg.pop('tool_calls', None) in message_normalizer.py:112"
        )
        assert msg["content"], "content must have XML after normalization"
        assert "write_to_file" in msg["content"], "XML must contain the tool name"

    def test_assistant_message_without_tool_calls_is_unchanged(self):
        """Plain assistant text messages must pass through without modification."""
        messages = [{"role": "assistant", "content": "Here is my plan."}]
        normalized = normalize_messages(messages, request_id="test")
        assert normalized[0]["content"] == "Here is my plan."
        assert "tool_calls" not in normalized[0]


# ─────────────────────────────────────────────────────────────────────────────
# PATH CONSISTENCY rule in system prompt
# File: app/services/xml_prompt_builder.py
#
# When a project has subfolders (e.g. backend/, frontend/), the model must
# use consistent path prefixes for all files in that component.
# Without this rule the model mixes roots:
#   backend/pom.xml  +  src/main/java/...   ← broken Maven structure
#
# Fix: add an explicit PATH CONSISTENCY note to Rule 3 in the system prompt.
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# BUG 4: Model hallucinating "[Awaiting tool result]" text in responses
# Files: app/main.py (LEAK_PATTERN), app/services/message_normalizer.py
#
# The model sometimes outputs "[Awaiting tool result]..." hallucinated content
# alongside a real tool_use block in the same response.  Two problems:
#
# a) _LEAK_PATTERN in main.py does not strip "[Awaiting tool result]", so it
#    survives as preamble text returned to Roo Code.
#
# b) normalize_messages does not drop text blocks from assistant messages that
#    also contain tool_use blocks.  The hallucinated text gets concatenated with
#    the XML in history and is fed back to the upstream, which makes the model
#    produce more hallucinated content.
#
# Fix a): extend _LEAK_PATTERN to also match "[Awaiting tool result(s)]".
# Fix b): drop text blocks from assistant content arrays that contain tool_use.
# ─────────────────────────────────────────────────────────────────────────────

class TestAwaitingToolResultHallucination:

    def test_text_block_dropped_when_assistant_has_tool_use(self):
        """
        An assistant message with both text ("[Awaiting tool result]...") and
        tool_use blocks must have the text stripped after normalization.

        FAILS today:  text content is concatenated with the XML → poisoned history
        PASSES after fix: text blocks are dropped when tool_use is present
        """
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "[Awaiting tool result][Awaiting tool result][Tool Result]\nsome fake content",
                    },
                    {
                        "type": "tool_use",
                        "id": "call_001",
                        "name": "read_file",
                        "input": {"path": "backend/src/main/java/Foo.java"},
                    },
                ],
            }
        ]
        normalized = normalize_messages(messages, request_id="test")
        msg = normalized[0]

        assert "[Awaiting tool result]" not in msg["content"], (
            f"BUG: Hallucinated '[Awaiting tool result]' text survived normalization.\n"
            f"Content: {msg['content']!r}\n"
            f"Fix: drop text blocks from assistant messages that contain tool_use blocks."
        )
        assert "read_file" in msg["content"], "XML tool call must still be present"

    def test_pure_text_assistant_message_unchanged(self):
        """Assistant messages with ONLY text (no tool_use) must pass through unchanged."""
        messages = [{"role": "assistant", "content": [{"type": "text", "text": "Sure, I can help."}]}]
        normalized = normalize_messages(messages, request_id="test")
        assert "Sure, I can help." in normalized[0]["content"]

    def test_leak_pattern_strips_awaiting_tool_result(self):
        """
        _LEAK_PATTERN must match and strip "[Awaiting tool result]" variants.

        FAILS today:  pattern only covers [assistant to=...] artifacts
        PASSES after fix: pattern also covers [Awaiting tool result(s)]
        """
        import re
        from app.main import _LEAK_PATTERN

        cases = [
            "[Awaiting tool result]",
            "[Awaiting tool result][Awaiting tool result]",
            "[Awaiting tool results]",
            "[awaiting tool result]",  # lowercase
        ]
        for text in cases:
            cleaned = _LEAK_PATTERN.sub("", text).strip()
            assert cleaned == "", (
                f"BUG: _LEAK_PATTERN did not strip {text!r}.\n"
                f"Result: {cleaned!r}\n"
                f"Fix: add '|(?:\\[Awaiting tool results?\\]\\s*)+' to _LEAK_PATTERN"
            )

    def test_leak_pattern_preserves_legitimate_preamble(self):
        """Legitimate reasoning text before a tool call must not be stripped."""
        from app.main import _LEAK_PATTERN

        text = "I need to check the file first before modifying it."
        cleaned = _LEAK_PATTERN.sub("", text).strip()
        assert cleaned == text, f"Legitimate preamble was incorrectly stripped: {cleaned!r}"


class TestPathConsistencyRule:

    def _get_system_prompt(self, client_type_value: str = "roo_code") -> str:
        from app.services.xml_prompt_builder import build_xml_system_prompt
        from app.services.tool_mapping import ClientType
        from tests.conftest import DEFAULT_TOOLS

        ct = ClientType(client_type_value)
        return build_xml_system_prompt(DEFAULT_TOOLS, existing_system=None, client_type=ct)

    def test_roo_code_prompt_contains_path_consistency_rule(self):
        """
        The Roo Code system prompt must explain that files in a subfolder project
        must use that subfolder as a prefix consistently.
        """
        prompt = self._get_system_prompt("roo_code")
        assert "PATH CONSISTENCY" in prompt, (
            "System prompt is missing the PATH CONSISTENCY rule.\n"
            "Without it the model mixes roots (backend/pom.xml + src/main/java/...)\n"
            "causing ENOENT when it later reads files it thinks it wrote."
        )

    def test_path_consistency_rule_contains_wrong_correct_example(self):
        """The rule must show a concrete WRONG/CORRECT example so the model learns it."""
        prompt = self._get_system_prompt("roo_code")
        assert "WRONG" in prompt and "CORRECT" in prompt, (
            "PATH CONSISTENCY rule must include a WRONG/CORRECT example."
        )
        # The broken pattern must be shown explicitly
        assert "backend/pom.xml" in prompt, (
            "Example must reference the typical broken pattern (backend/pom.xml + src/...)"
        )

    def test_cline_prompt_does_not_break(self):
        """Changing roo_code rules must not break cline prompt generation."""
        prompt = self._get_system_prompt("cline")
        assert "write_to_file" in prompt
        assert "replace_in_file" in prompt


# ─────────────────────────────────────────────────────────────────────────────
# BUG: ask_followup_question follow_up string instead of array
# File: app/services/tool_call_fixups.py (fix_ask_followup_question_params)
#
# The model outputs follow_up as a newline-separated string instead of an
# array. Roo Code rejects the call with "Missing value for required parameter
# 'follow_up'" causing a hard Roo Code error.
# ─────────────────────────────────────────────────────────────────────────────

class TestAskFollowupQuestionFixup:

    def _make_tc(self, follow_up):
        return [{
            "id": "call_test_ask",
            "type": "function",
            "function": {
                "name": "ask_followup_question",
                "arguments": json.dumps({
                    "question": "What should I do?",
                    "follow_up": follow_up,
                }),
            },
        }]

    def test_string_follow_up_converted_to_array(self):
        """follow_up as newline-separated string must become an array."""
        tc = self._make_tc("Show output\nRestart server\nCheck config")
        result = fix_ask_followup_question_params(tc, "req1")
        args = json.loads(result[0]["function"]["arguments"])
        assert isinstance(args["follow_up"], list), "follow_up must be a list"
        assert args["follow_up"] == ["Show output", "Restart server", "Check config"]

    def test_array_follow_up_unchanged(self):
        """follow_up already as array must pass through unchanged."""
        tc = self._make_tc(["Option A", "Option B"])
        result = fix_ask_followup_question_params(tc, "req2")
        args = json.loads(result[0]["function"]["arguments"])
        assert args["follow_up"] == ["Option A", "Option B"]

    def test_other_tools_untouched(self):
        """Non-ask_followup_question tool calls must not be modified."""
        tc = [{
            "id": "call_other",
            "type": "function",
            "function": {
                "name": "execute_command",
                "arguments": json.dumps({"command": "ls"}),
            },
        }]
        result = fix_ask_followup_question_params(tc, "req3")
        assert result == tc


# ─────────────────────────────────────────────────────────────────────────────
# BUG: Model hallucinating [Tool Result] in its own content
# Files: xml_prompt_builder.py, priming.py
#
# When a file read returns truncated content ("..."), the model misdiagnoses
# it as "file corrupted" and writes fake [Tool Result] blocks in its response
# instead of calling write_to_file. This produces garbage output and Roo Code
# aborts the conversation.
#
# Fix 1 (system prompt): FORBIDDEN FORMATS now explicitly bans [Tool Result]
# Fix 2 (priming): a two-turn truncated-file example teaches the correct pattern
# ─────────────────────────────────────────────────────────────────────────────

class TestToolResultHallucinationPrevention:

    DEFAULT_TOOLS = [
        {"type": "function", "function": {"name": "read_file", "parameters": {"properties": {"path": {}}, "required": ["path"]}}},
        {"type": "function", "function": {"name": "write_to_file", "parameters": {"properties": {"path": {}, "content": {}}, "required": ["path", "content"]}}},
        {"type": "function", "function": {"name": "attempt_completion", "parameters": {"properties": {"result": {}}, "required": ["result"]}}},
    ]

    def _get_system_prompt(self):
        from app.services.xml_prompt_builder import build_xml_system_prompt
        from app.services.tool_mapping import ClientType
        ct = ClientType("roo_code")
        return build_xml_system_prompt(self.DEFAULT_TOOLS, existing_system=None, client_type=ct)

    def _get_priming_messages(self):
        from app.services.priming import inject_priming
        from app.services.tool_mapping import ClientType
        base = [{"role": "user", "content": "Do something."}]
        return inject_priming(base, self.DEFAULT_TOOLS, client_type=ClientType("roo_code"))

    def test_system_prompt_forbids_tool_result_in_response(self):
        """System prompt must explicitly forbid writing [Tool Result] in the response."""
        prompt = self._get_system_prompt()
        assert "[Tool Result]" in prompt, (
            "System prompt must mention [Tool Result] in FORBIDDEN FORMATS.\n"
            "Without this the model hallucinates tool results in its own content."
        )
        assert "NEVER write it yourself" in prompt or "Never write it yourself" in prompt, (
            "The [Tool Result] rule must include 'NEVER write it yourself'."
        )

    def test_priming_contains_truncated_file_sequence(self):
        """Priming must include a two-turn example showing truncated file → write_to_file."""
        messages = self._get_priming_messages()
        # Find an assistant message that calls write_to_file after a truncated [Tool Result]
        found_truncated_user = False
        found_write_after_truncation = False
        for i, msg in enumerate(messages):
            if msg.get("role") == "user" and "truncated" in msg.get("content", ""):
                found_truncated_user = True
                # Next assistant message must be a write_to_file
                if i + 1 < len(messages):
                    next_msg = messages[i + 1]
                    if next_msg.get("role") == "assistant" and "write_to_file" in next_msg.get("content", ""):
                        found_write_after_truncation = True
        assert found_truncated_user, (
            "Priming must contain a user message with truncated file content.\n"
            "Without it the model has no example of the correct truncated-file behaviour."
        )
        assert found_write_after_truncation, (
            "After a truncated [Tool Result] the priming must show write_to_file as response.\n"
            "Without this the model may hallucinate [Tool Result] blocks instead."
        )

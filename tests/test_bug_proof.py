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

from app.services.loop_detection import detect_ask_followup_loop, detect_success_loop
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
            if msg.get("role") == "user" and "truncated" in msg.get("content", "").lower():
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

    def test_priming_uses_exact_roo_code_truncation_format(self):
        """
        Priming truncated-file example must use the exact Roo Code format:
          'IMPORTANT: File content truncated.'
          'To read more: Use the read_file tool with offset=...'

        FAILS if priming still uses old '[content truncated — N lines]' placeholder.
        PASSES after fix A (priming.py): example updated to real Roo Code format.
        """
        messages = self._get_priming_messages()
        roo_format_found = any(
            "IMPORTANT: File content truncated" in msg.get("content", "")
            for msg in messages
            if msg.get("role") == "user"
        )
        assert roo_format_found, (
            "Priming must use the exact Roo Code truncation format "
            "('IMPORTANT: File content truncated.') — not a placeholder like "
            "'[content truncated — N lines]'. "
            "Without the exact format the model does not recognize the pattern."
        )

    def test_forbidden_formats_mentions_offset_option(self):
        """
        FORBIDDEN FORMATS must clarify that read_file with offset= IS allowed —
        only [Tool Result] hallucination is forbidden.

        FAILS if the rule only says 'write_to_file directly' without mentioning offset.
        PASSES after fix B (xml_prompt_builder.py).
        """
        prompt = self._get_system_prompt()
        assert "offset" in prompt, (
            "FORBIDDEN FORMATS must mention that read_file with offset= is allowed "
            "when a file is truncated. Without this the model avoids paging through "
            "large files and may hallucinate instead."
        )


# ─────────────────────────────────────────────────────────────────────────────
# BUG: _inject_truncation_reminder (Fix C — dynamic reminder injection)
# File: app/main.py
#
# When the last user message contains Roo Code's truncation notice
# ("IMPORTANT: File content truncated"), toolproxy must append a reminder
# so the model does not hallucinate [Tool Result] blocks.
# ─────────────────────────────────────────────────────────────────────────────

class TestInjectTruncationReminder:

    _ROO_TRUNCATION = (
        "[Tool Result]\n"
        "File: taskmanager/backend/pom.xml\n"
        "IMPORTANT: File content truncated.\n"
        "Status: Showing lines 1-40 of 83 total lines.\n"
        "To read more: Use the read_file tool with offset=41 and limit=30.\n"
        "\n"
        " 1 | <?xml version=\"1.0\" encoding=\"UTF-8\"?>"
    )

    def _call(self, messages):
        from app.main import _inject_truncation_reminder
        return _inject_truncation_reminder(messages, request_id="test")

    def test_reminder_injected_when_last_user_message_is_truncated(self):
        """
        Reminder must be appended to the last user message when it contains
        the Roo Code truncation marker.

        FAILS before fix C: function does not exist yet.
        PASSES after fix C: reminder appended.
        """
        messages = [
            {"role": "system", "content": "You are a coding assistant."},
            {"role": "user", "content": self._ROO_TRUNCATION},
        ]
        result = self._call(messages)
        last_user = next(m for m in reversed(result) if m.get("role") == "user")
        assert "REMINDER" in last_user["content"], (
            "_inject_truncation_reminder must append a [REMINDER] to the last user "
            "message when it contains 'IMPORTANT: File content truncated'."
        )
        assert "offset" in last_user["content"], (
            "The reminder must mention that read_file with offset= is allowed."
        )
        assert "[Tool Result]" in last_user["content"] or "Tool Result" in last_user["content"], (
            "The reminder must warn about [Tool Result] hallucination."
        )

    def test_reminder_not_injected_for_normal_content(self):
        """
        Reminder must NOT fire when the last user message is normal content.
        """
        messages = [
            {"role": "system", "content": "You are a coding assistant."},
            {"role": "user", "content": "[Tool Result]\nFile: hello.py\n1 | print('hello')"},
        ]
        result = self._call(messages)
        last_user = next(m for m in reversed(result) if m.get("role") == "user")
        # Content must be unchanged (no reminder injected)
        assert last_user["content"] == messages[-1]["content"], (
            "Reminder must not be injected when the tool result is not truncated."
        )

    def test_reminder_not_injected_when_truncation_is_not_in_last_message(self):
        """
        Truncation in an EARLIER message must not trigger the reminder.
        Only the last user message matters.
        """
        messages = [
            {"role": "system", "content": "You are a coding assistant."},
            {"role": "user", "content": self._ROO_TRUNCATION},          # earlier — truncated
            {"role": "assistant", "content": "<read_file><path>pom.xml</path></read_file>"},
            {"role": "user", "content": "[Tool Result]\n 1 | <?xml ..."},  # last — NOT truncated
        ]
        result = self._call(messages)
        last_user = next(m for m in reversed(result) if m.get("role") == "user")
        assert "REMINDER" not in last_user["content"], (
            "Reminder must only fire when the LAST user message is truncated, "
            "not when a previous message was truncated."
        )

    def test_messages_list_not_mutated_in_place(self):
        """_inject_truncation_reminder must return a new list, not mutate the original."""
        original_content = self._ROO_TRUNCATION
        messages = [{"role": "user", "content": original_content}]
        result = self._call(messages)
        # Original list must be unchanged
        assert messages[0]["content"] == original_content, (
            "_inject_truncation_reminder mutated the original messages list."
        )
        # Result must be a different list
        assert result is not messages, "Must return a new list, not the original."

    def test_list_content_type_handled(self):
        """
        Roo Code sometimes sends content as a list of content blocks (not a plain string).
        The reminder must still be appended correctly.
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "content": self._ROO_TRUNCATION},
                    {"type": "text", "text": "environment_details here"},
                ],
            }
        ]
        result = self._call(messages)
        last_user = next(m for m in reversed(result) if m.get("role") == "user")
        # When content is a list, reminder is appended as a new text block
        content = last_user["content"]
        assert isinstance(content, list), "List content must stay a list"
        texts = " ".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
        assert "REMINDER" in texts, (
            "Reminder must be appended as a text block when content is a list."
        )


# ─────────────────────────────────────────────────────────────────────────────
# BUG: apply_diff loop — wrong hint because detect_success_loop fires before
# detect_repetitive_tool_loop (v1.6.5)
# File: app/main.py, app/services/loop_detection.py
#
# When the model calls apply_diff on the same file 3+ times in a row, the
# detect_success_loop fires first (threshold=2, because "modified" appears in
# every apply_diff result) with the hint "already succeeded, call
# attempt_completion". The model CORRECTLY ignores this hint because the
# compilation is still broken — the task is NOT done. detect_repetitive_tool_loop
# would give the right hint ("not making progress, try a different approach")
# but is never reached due to short-circuit evaluation.
#
# Fix 1 (main.py): run detect_repetitive_tool_loop BEFORE detect_success_loop.
# Fix 2 (loop_detection.py): update success_loop hint to not say
#   "call attempt_completion" but "verify with read_file or write_to_file".
# ─────────────────────────────────────────────────────────────────────────────

class TestApplyDiffLoopHint:
    """
    Verify that an apply_diff loop on the same file gets the correct
    'not making progress' hint, not the misleading 'already succeeded' hint.
    """

    _APPLY_DIFF_RESULT = (
        '{"path":"taskmanager/backend/src/main/java/com/example/taskmanager/controller/TaskController.java",'
        '"operation":"modified","notice":"Proceed with the task."}'
    )

    def _build_apply_diff_history(self, repeats: int) -> list:
        """Build a normalized message history with N consecutive apply_diff calls."""
        file_path = "taskmanager/backend/src/main/java/com/example/taskmanager/controller/TaskController.java"
        diff_content = "<<<<<<< SEARCH\n:start_line:66\n-------\n    }\n=======\n>>>>>>> REPLACE"
        msgs = [
            {"role": "system", "content": "You are a coding assistant."},
            {"role": "user", "content": "Fix the extra closing brace in TaskController.java."},
        ]
        for i in range(repeats):
            msgs.append({
                "role": "assistant",
                "content": (
                    f"<apply_diff>\n"
                    f"<path>{file_path}</path>\n"
                    f"<diff>{diff_content}</diff>\n"
                    f"</apply_diff>"
                ),
            })
            msgs.append({
                "role": "user",
                "content": f"[Tool Result]\n{self._APPLY_DIFF_RESULT}",
            })
        return msgs

    def test_repetitive_hint_fires_before_success_hint_for_apply_diff(self):
        """
        After 3 consecutive apply_diff calls on the same file, the hint must come
        from detect_repetitive_tool_loop ('not making progress') NOT from
        detect_success_loop ('already succeeded, call attempt_completion').

        FAILS before fix: detect_success_loop fires first (threshold=2) with the
          wrong hint; detect_repetitive_tool_loop is never reached.
        PASSES after fix: detect_repetitive_tool_loop runs first → correct hint.
        """
        from app.services.loop_detection import detect_repetitive_tool_loop, detect_success_loop
        from app.services.tool_mapping import ClientType

        msgs = self._build_apply_diff_history(repeats=3)
        repetitive_hint = detect_repetitive_tool_loop(msgs, "test", ClientType.ROO_CODE)
        success_hint = detect_success_loop(msgs, "test", ClientType.ROO_CODE)

        # Both detectors should fire
        assert repetitive_hint is not None, (
            "detect_repetitive_tool_loop must fire after 3 consecutive apply_diff calls."
        )
        assert success_hint is not None, (
            "detect_success_loop fires too (because 'modified' is in results) — "
            "but it must NOT be the one used."
        )

        # The hint actually used by main.py must be the repetitive one (fires first now)
        loop_hint = repetitive_hint or success_hint
        assert loop_hint is repetitive_hint, (
            "With the new order (repetitive first), the repetitive hint must win.\n"
            f"Got: {loop_hint!r}\n"
            f"Expected: {repetitive_hint!r}"
        )
        assert "attempt_completion" not in loop_hint, (
            "The hint injected for an apply_diff loop must NOT say 'call attempt_completion'.\n"
            "The task is not done — compilation still fails. The model correctly ignores\n"
            f"that hint. Got: {loop_hint!r}"
        )
        assert "different approach" in loop_hint or "write_to_file" in loop_hint.lower(), (
            f"The hint must suggest a different approach (write_to_file). Got: {loop_hint!r}"
        )

    def test_success_loop_hint_no_longer_says_call_attempt_completion(self):
        """
        The success_loop hint must NOT say 'call attempt_completion' any more.
        It should instead suggest read_file / write_to_file for verification.

        FAILS before fix: hint ends with 'Call attempt_completion...'
        PASSES after fix: hint says 'use read_file to verify or write_to_file'
        """
        from app.services.loop_detection import detect_success_loop
        from app.services.tool_mapping import ClientType

        msgs = self._build_apply_diff_history(repeats=2)
        hint = detect_success_loop(msgs, "test", ClientType.ROO_CODE)
        assert hint is not None, "detect_success_loop must fire after 2 'modified' results."
        assert "read_file" in hint, (
            f"success_loop hint must mention read_file for verification. Got: {hint!r}"
        )
        assert "write_to_file" in hint, (
            f"success_loop hint must mention write_to_file as alternative. Got: {hint!r}"
        )

    def test_repetitive_loop_does_not_fire_after_only_two_apply_diffs(self):
        """
        Repetitive loop threshold is 3. After only 2 apply_diffs, it must not fire.
        (success_loop still fires at 2 — that is acceptable fallback behaviour.)
        """
        from app.services.loop_detection import detect_repetitive_tool_loop
        from app.services.tool_mapping import ClientType

        msgs = self._build_apply_diff_history(repeats=2)
        hint = detect_repetitive_tool_loop(msgs, "test", ClientType.ROO_CODE)
        assert hint is None, (
            f"detect_repetitive_tool_loop must NOT fire after only 2 apply_diffs. "
            f"Got: {hint!r}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# FIX v1.6.6: Generic truncation detection
# File: app/main.py (_TRUNCATION_RE)
#
# Previously _TRUNCATION_MARKER only matched the exact string
# "IMPORTANT: File content truncated". Roo Code also uses other formats:
#   - "(Truncated)" at the end of a file read result
#   - "file was truncated" in error messages
#
# Fix: replace string check with re.compile(r"\btruncated\b", re.IGNORECASE).
# ─────────────────────────────────────────────────────────────────────────────

class TestGenericTruncationDetection:

    def _call(self, messages):
        from app.main import _inject_truncation_reminder
        return _inject_truncation_reminder(messages, request_id="test")

    def test_reminder_fires_for_classic_important_format(self):
        """Backward compat: original 'IMPORTANT: File content truncated' still fires."""
        messages = [{"role": "user", "content": (
            "[Tool Result]\nFile: foo.java\n"
            "IMPORTANT: File content truncated.\n"
            "1 | package com.example;"
        )}]
        result = self._call(messages)
        last_user = next(m for m in reversed(result) if m["role"] == "user")
        assert "REMINDER" in last_user["content"]

    def test_reminder_fires_for_parenthesised_truncated(self):
        """
        Roo Code sometimes ends a file result with '(Truncated)' instead of the
        IMPORTANT header. The regex must catch this format too.
        """
        messages = [{"role": "user", "content": (
            "[Tool Result]\n"
            "File: taskmanager/frontend/src/components/TaskList.tsx\n"
            "1 | import React from 'react';\n"
            "34 |     ... ... ... ... ... ... ... ... ... ...\n"
            "(Truncated)"
        )}]
        result = self._call(messages)
        last_user = next(m for m in reversed(result) if m["role"] == "user")
        assert "REMINDER" in last_user["content"], (
            "_inject_truncation_reminder must fire for '(Truncated)' format. "
            "Regex \\btruncated\\b (IGNORECASE) should catch this."
        )

    def test_reminder_fires_case_insensitive(self):
        """'truncated' in any casing must trigger the reminder."""
        for variant in ["Truncated", "TRUNCATED", "truncated"]:
            messages = [{"role": "user", "content": f"File content {variant}."}]
            result = self._call(messages)
            last_user = next(m for m in reversed(result) if m["role"] == "user")
            assert "REMINDER" in last_user["content"], (
                f"Reminder must fire for casing variant {variant!r}"
            )

    def test_reminder_not_fire_for_unrelated_word(self):
        """'truncate' without word boundary (e.g. 'untruncated') must not match."""
        messages = [{"role": "user", "content": "The file was not untruncated."}]
        result = self._call(messages)
        last_user = next(m for m in reversed(result) if m["role"] == "user")
        assert "REMINDER" not in last_user["content"], (
            "\\btruncated\\b must not match inside 'untruncated' — word boundary required."
        )

    def test_reminder_mentions_apply_diff_ban(self):
        """
        The reminder injected for a truncated file must explicitly say
        'apply_diff' is not allowed. Without this, the model may try to patch
        a file it has only partially read.
        """
        messages = [{"role": "user", "content": "(Truncated) file content here"}]
        result = self._call(messages)
        last_user = next(m for m in reversed(result) if m["role"] == "user")
        assert "apply_diff" in last_user["content"], (
            "Truncation reminder must explicitly ban apply_diff. "
            "The model must not attempt a partial patch on a truncated file."
        )


# ─────────────────────────────────────────────────────────────────────────────
# FIX v1.6.7: validate_apply_diff_completeness
# File: app/services/tool_call_fixups.py
#
# A corrupt or truncated apply_diff is missing the '>>>>>>> REPLACE' closing
# marker. Passing it to Roo Code causes an apply failure. Dropping it is
# safer than letting Roo Code receive garbage.
# ─────────────────────────────────────────────────────────────────────────────

class TestValidateApplyDiffCompleteness:

    def _make_tc(self, name: str, diff: str, path: str = "src/Foo.java") -> list:
        return [{
            "id": "call_test",
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps({"path": path, "diff": diff}),
            },
        }]

    _VALID_DIFF = (
        "<<<<<<< SEARCH\n"
        "    public void old() {}\n"
        "=======\n"
        "    public void new() {}\n"
        ">>>>>>> REPLACE"
    )

    _CORRUPT_DIFF = (
        "<<<<<<< SEARCH\n"
        "    public void old() {}\n"
        "=======\n"
        "    public void new() {\n"
        "        const newId?????????"  # truncated, no >>>>>>> REPLACE
    )

    def test_valid_apply_diff_passes_through(self):
        """A complete apply_diff with >>>>>>> REPLACE must be returned unchanged."""
        from app.services.tool_call_fixups import validate_apply_diff_completeness
        tc = self._make_tc("apply_diff", self._VALID_DIFF)
        result = validate_apply_diff_completeness(tc, "test")
        assert len(result) == 1, "Valid apply_diff must not be dropped"
        assert result[0]["function"]["name"] == "apply_diff"

    def test_corrupt_apply_diff_is_dropped(self):
        """
        An apply_diff missing '>>>>>>> REPLACE' must be dropped entirely.

        FAILS if validate_apply_diff_completeness does not exist or lets it through.
        PASSES after fix: corrupt diffs are silently dropped.
        """
        from app.services.tool_call_fixups import validate_apply_diff_completeness
        tc = self._make_tc("apply_diff", self._CORRUPT_DIFF)
        result = validate_apply_diff_completeness(tc, "test")
        assert len(result) == 0, (
            "Corrupt apply_diff (missing >>>>>>> REPLACE) must be dropped. "
            f"Got: {result}"
        )

    def test_corrupt_replace_in_file_is_dropped(self):
        """replace_in_file (Cline) with corrupt diff must also be dropped."""
        from app.services.tool_call_fixups import validate_apply_diff_completeness
        tc = self._make_tc("replace_in_file", self._CORRUPT_DIFF)
        result = validate_apply_diff_completeness(tc, "test")
        assert len(result) == 0, (
            "Corrupt replace_in_file (missing >>>>>>> REPLACE) must be dropped."
        )

    def test_valid_replace_in_file_passes_through(self):
        """Complete replace_in_file must pass through unchanged."""
        from app.services.tool_call_fixups import validate_apply_diff_completeness
        tc = self._make_tc("replace_in_file", self._VALID_DIFF)
        result = validate_apply_diff_completeness(tc, "test")
        assert len(result) == 1

    def test_other_tools_always_pass_through(self):
        """Non-diff tools (write_to_file etc.) must never be filtered."""
        from app.services.tool_call_fixups import validate_apply_diff_completeness
        tc = [{
            "id": "call_w",
            "type": "function",
            "function": {
                "name": "write_to_file",
                "arguments": json.dumps({"path": "foo.py", "content": "print('hi')"}),
            },
        }]
        result = validate_apply_diff_completeness(tc, "test")
        assert result == tc

    def test_empty_diff_is_not_dropped(self):
        """
        An apply_diff with no diff argument at all (empty string / missing key)
        must pass through — it may be a degenerate case handled elsewhere.
        Only non-empty diffs missing >>>>>>> REPLACE are dropped.
        """
        from app.services.tool_call_fixups import validate_apply_diff_completeness
        tc = self._make_tc("apply_diff", "")
        result = validate_apply_diff_completeness(tc, "test")
        assert len(result) == 1, "Empty diff must not be dropped (handled elsewhere)"


# ─────────────────────────────────────────────────────────────────────────────
# ask_followup_question loop detection (v1.6.11)
#
# Roo Code sends a genuine user message after each answer, resetting the
# consecutive counter in detect_repetitive_tool_loop. The new frequency-window
# detector catches this pattern instead.
# ─────────────────────────────────────────────────────────────────────────────

def _afq_msg(question: str = "How to proceed?") -> dict:
    """Build a normalized assistant message with an ask_followup_question call."""
    return {
        "role": "assistant",
        "content": f"<ask_followup_question><question>{question}</question>"
                   "<follow_up><option>Option A</option></follow_up></ask_followup_question>",
    }


def _afq_result(answer: str = "Option A") -> list:
    """Simulate one Roo Code cycle: tool result + genuine user reply."""
    return [
        {"role": "user", "content": f"[Tool Result]\n{answer}"},
        {"role": "user", "content": "Please continue."},
    ]


class TestAskFollowupLoop:

    def test_three_ask_followup_calls_trigger_hint(self):
        """
        3 ask_followup_question calls in recent history → hint fires.
        Reproduces the real pattern where Roo Code sends a genuine user message
        after each answer, which would reset the consecutive counter.
        """
        messages = [{"role": "user", "content": "Fix the schema issue."}]
        for _ in range(3):
            messages.append(_afq_msg())
            messages.extend(_afq_result())

        result = detect_ask_followup_loop(messages)
        assert result is not None, (
            "Expected loop hint after 3 ask_followup_question calls, got None. "
            "Frequency-window detector should fire regardless of genuine user messages in between."
        )
        assert "ask_followup_question" in result.lower()

    def test_two_ask_followup_calls_no_hint(self):
        """Below threshold (2 calls) — no hint."""
        messages = [{"role": "user", "content": "Fix the schema issue."}]
        for _ in range(2):
            messages.append(_afq_msg())
            messages.extend(_afq_result())

        result = detect_ask_followup_loop(messages)
        assert result is None, (
            f"Expected no hint for 2 ask_followup_question calls, got: {result!r}"
        )

    def test_hint_fires_even_with_genuine_user_messages_between(self):
        """
        The core of the fix: genuine user messages between calls must NOT prevent detection.
        This is exactly what broke detect_repetitive_tool_loop for this pattern.
        """
        messages = [
            {"role": "user", "content": "Task description"},
            _afq_msg("What approach?"),
            {"role": "user", "content": "[Tool Result]\nAdd placeholder"},
            {"role": "user", "content": "Please proceed."},   # ← genuine user msg resets consecutive
            _afq_msg("How to proceed?"),
            {"role": "user", "content": "[Tool Result]\nOption A"},
            {"role": "user", "content": "Continue please."},  # ← another reset
            _afq_msg("Are you sure?"),
            {"role": "user", "content": "[Tool Result]\nYes"},
            {"role": "user", "content": "Go ahead."},
        ]
        result = detect_ask_followup_loop(messages)
        assert result is not None, (
            "Frequency-window detector must fire even with genuine user messages between calls."
        )

    def test_old_ask_followup_outside_window_not_counted(self):
        """
        ask_followup_question calls outside the window (_AFQ_WINDOW=24) must not count.
        Long conversations should not be penalized for questions asked much earlier.
        """
        old_messages = []
        for _ in range(5):  # 5 old cycles, outside the window
            old_messages.append(_afq_msg())
            old_messages.extend(_afq_result())

        # Pad with unrelated turns to push old messages out of the window
        padding = [{"role": "user", "content": f"[Tool Result]\nStep {i}"} for i in range(20)]
        recent = [
            _afq_msg(),          # only 1 recent call
            {"role": "user", "content": "[Tool Result]\nOption A"},
        ]
        messages = old_messages + padding + recent
        result = detect_ask_followup_loop(messages)
        assert result is None, (
            "Old ask_followup_question calls outside the window must not trigger the hint."
        )

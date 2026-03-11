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
from app.services.tool_call_fixups import convert_move_file_to_execute_command


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

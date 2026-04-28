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
from tests.conftest import _tool, DEFAULT_TOOLS
from app.services.tool_call_fixups import (
    convert_move_file_to_execute_command,
    fix_ask_followup_question_params,
    _deduplicate_diff_hunks,
    validate_apply_diff_completeness,
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


# ─────────────────────────────────────────────────────────────────────────────
# BUG 5: Exploding apply_diff — context corruption causes repeated hunks
#
# When the model's context is corrupted it regenerates the same SEARCH/REPLACE
# hunks dozens of times in a single apply_diff call.  The 21 000+ char diff
# is passed through unchanged (it contains >>>>>>> REPLACE so the existing
# completeness check treats it as valid).
#
# Real example: App.tsx patch with ~10 unique changes → diff contained each
# hunk ~10–15 times + a truncated final hunk (no >>>>>>> REPLACE closing).
#
# Fix: _deduplicate_diff_hunks() deduplicates by SEARCH content and drops
#      truncated trailing hunks; called from validate_apply_diff_completeness.
# ─────────────────────────────────────────────────────────────────────────────

def _hunk(search: str, replace: str) -> str:
    return f"<<<<<<< SEARCH\n{search}\n=======\n{replace}\n>>>>>>> REPLACE\n"


class TestDiffHunkDeduplication:

    def test_clean_diff_passes_through_unchanged(self):
        """A diff with all-unique hunks must not be modified."""
        diff = _hunk("foo(prev =>", "foo((prev: T[]) =>") + _hunk("bar(x =>", "bar((x: number) =>")
        cleaned, dropped = _deduplicate_diff_hunks(diff, "test")
        assert dropped == 0
        assert "foo(prev =>" in cleaned
        assert "bar(x =>" in cleaned

    def test_duplicate_hunks_are_removed(self):
        """The same SEARCH/REPLACE block appearing N times must be kept exactly once."""
        hunk = _hunk("setTasks(prev =>", "setTasks((prev: Task[]) =>")
        diff = hunk * 10
        cleaned, dropped = _deduplicate_diff_hunks(diff, "test")
        assert dropped == 9
        assert cleaned.count("<<<<<<< SEARCH") == 1
        assert "setTasks(prev =>" in cleaned

    def test_truncated_last_hunk_is_dropped(self):
        """A hunk missing >>>>>>> REPLACE (model hit token limit) must be silently dropped."""
        good = _hunk("setTasks(prev =>", "setTasks((prev: Task[]) =>")
        truncated = "<<<<<<< SEARCH\nsetEditingId(prev =>\n=======\nsetEditingId((prev: number | null) =>\n"
        diff = good + truncated
        cleaned, dropped = _deduplicate_diff_hunks(diff, "test")
        assert dropped == 1
        assert "setTasks" in cleaned
        assert "setEditingId" not in cleaned

    def test_mixed_duplicates_and_unique_hunks(self):
        """Duplicates are dropped; unique hunks are preserved in order."""
        h1 = _hunk("alpha =>", "(alpha: A) =>")
        h2 = _hunk("beta =>", "(beta: B) =>")
        diff = h1 + h2 + h1 + h2 + h1   # h1 ×3, h2 ×2
        cleaned, dropped = _deduplicate_diff_hunks(diff, "test")
        assert dropped == 3
        assert cleaned.count("<<<<<<< SEARCH") == 2
        assert "alpha =>" in cleaned
        assert "beta =>" in cleaned

    def test_real_world_exploding_diff(self):
        """
        Replica of the App.tsx incident: ~10 unique hunks each repeated 10×
        plus a truncated last hunk.  After cleanup only unique hunks remain.
        """
        unique_hunks = [
            _hunk("setTasks(prev =>", "setTasks((prev: Task[]) =>"),
            _hunk("setEditingId(prev =>", "setEditingId((prev: number | null) =>"),
            _hunk("setNewTask(prev =>", "setNewTask((prev: Partial<Task>) =>"),
            _hunk("handleChange = (", "handleChange = (\n  e: ChangeEvent<HTMLInputElement>"),
            _hunk("tasks.find(t =>", "tasks.find((t: Task) =>"),
        ]
        truncated = "<<<<<<< SEARCH\nsetTasks(prev =>\n=======\nsetTasks((prev: Task[]) =>\n"
        diff = "".join(unique_hunks * 10) + truncated
        cleaned, dropped = _deduplicate_diff_hunks(diff, "test")
        # 10 repetitions - 1 kept = 9 dropped per unique hunk (5 × 9 = 45) + 1 truncated
        assert dropped == 46
        assert cleaned.count("<<<<<<< SEARCH") == len(unique_hunks)

    def test_validate_apply_diff_completeness_deduplicates(self):
        """validate_apply_diff_completeness must call deduplication for correct-format diffs."""
        hunk = _hunk("old_code", "new_code")
        bloated_diff = hunk * 5
        tool_calls = [{
            "id": "call_001",
            "type": "function",
            "function": {
                "name": "apply_diff",
                "arguments": json.dumps({"path": "src/App.tsx", "diff": bloated_diff}),
            },
        }]
        result = validate_apply_diff_completeness(tool_calls, "test")
        assert result, "Tool call must not be dropped"
        args = json.loads(result[0]["function"]["arguments"])
        assert args["diff"].count("<<<<<<< SEARCH") == 1, (
            "Duplicate hunks must be removed by validate_apply_diff_completeness"
        )

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
# FIX v1.6.14: Search-loop hint is tool-aware + ONE CALL enforcement
# File: app/services/loop_detection.py
#
# Previously detect_repetitive_tool_loop gave the same hint for ALL tools:
#   "Use write_to_file with the COMPLETE corrected content"
#
# For search tools (search_files, list_files, read_file, grep, glob) this
# is nonsensical — the model interprets it as "I need to try harder" and
# batches 14 calls in one response (observed in prod log 2026-04-10).
#
# Fix 1: tool-aware hint — search tools get "item likely doesn't exist, try a
#         completely different approach or call attempt_completion".
# Fix 2: all hints now include "CRITICAL: Always send exactly ONE tool call
#         per response — never batch multiple calls."
# ─────────────────────────────────────────────────────────────────────────────

class TestSearchLoopHint:
    """
    Verify that search tool loops produce a search-appropriate hint (not a
    write-tool hint), and that all hints enforce the ONE CALL rule.
    """

    def _build_search_loop_history(self, tool: str, path: str, repeats: int) -> list:
        """Build a normalized history with N consecutive identical search calls."""
        msgs = [
            {"role": "system", "content": "You are a coding assistant."},
            {"role": "user", "content": f"Find where helm upgrade is called in the project."},
        ]
        for i in range(repeats):
            msgs.append({
                "role": "assistant",
                "content": f"<{tool}>\n<path>{path}</path>\n<regex>helm upgrade</regex>\n</{tool}>",
            })
            msgs.append({
                "role": "user",
                "content": "[Tool Result]\nNo matches found.",
            })
        return msgs

    def test_search_loop_hint_does_not_say_write_to_file(self):
        """
        After 3× search_files with no results, the hint must NOT say
        'Use write_to_file' — that makes no sense for a search operation
        and causes the model to batch multiple calls instead.

        FAILS before fix: hint says 'Use write_to_file with the COMPLETE corrected content'
        PASSES after fix: hint says 'try a completely different approach'
        """
        from app.services.loop_detection import detect_repetitive_tool_loop
        from app.services.tool_mapping import ClientType

        msgs = self._build_search_loop_history("search_files", ".", repeats=3)
        hint = detect_repetitive_tool_loop(msgs, "test", ClientType.ROO_CODE)

        assert hint is not None, (
            "detect_repetitive_tool_loop must fire after 3 consecutive search_files calls."
        )
        assert "write_to_file" not in hint, (
            f"Search-loop hint must NOT mention write_to_file — this is a search operation.\n"
            f"Got: {hint!r}\n"
            "Saying 'use write_to_file' causes the model to batch multiple search calls."
        )

    def test_search_loop_hint_suggests_different_approach(self):
        """
        The search-loop hint must suggest stopping or trying a different approach,
        not repeating the same search operation.
        """
        from app.services.loop_detection import detect_repetitive_tool_loop
        from app.services.tool_mapping import ClientType

        msgs = self._build_search_loop_history("search_files", ".", repeats=3)
        hint = detect_repetitive_tool_loop(msgs, "test", ClientType.ROO_CODE)

        assert hint is not None
        assert "different" in hint.lower() or "attempt_completion" in hint, (
            f"Search-loop hint must suggest a different approach or attempt_completion.\n"
            f"Got: {hint!r}"
        )

    def test_search_loop_hint_contains_one_call_rule(self):
        """
        The hint for a search loop must explicitly say to send ONE tool call —
        not batch multiple calls. This prevents the '14 calls at once' behaviour.

        FAILS before fix: hint has no mention of batching prohibition.
        PASSES after fix: hint includes 'ONE tool call' / 'never batch'.
        """
        from app.services.loop_detection import detect_repetitive_tool_loop
        from app.services.tool_mapping import ClientType

        msgs = self._build_search_loop_history("search_files", ".", repeats=3)
        hint = detect_repetitive_tool_loop(msgs, "test", ClientType.ROO_CODE)

        assert hint is not None
        assert "ONE" in hint or "one" in hint.lower(), (
            f"Search-loop hint must enforce the ONE tool call rule.\n"
            f"Got: {hint!r}"
        )

    def test_write_loop_hint_also_contains_one_call_rule(self):
        """
        The write-type loop hint (apply_diff / write_to_file) must also include
        the ONE CALL enforcement after fix 2.
        """
        from app.services.loop_detection import detect_repetitive_tool_loop
        from app.services.tool_mapping import ClientType

        file_path = "src/main/java/Foo.java"
        msgs = [
            {"role": "system", "content": "You are a coding assistant."},
            {"role": "user", "content": "Fix Foo.java"},
        ]
        for i in range(3):
            msgs.append({
                "role": "assistant",
                "content": f"<apply_diff>\n<path>{file_path}</path>\n<diff>some diff</diff>\n</apply_diff>",
            })
            msgs.append({
                "role": "user",
                "content": '[Tool Result]\n{"operation":"modified"}',
            })

        hint = detect_repetitive_tool_loop(msgs, "test", ClientType.ROO_CODE)
        assert hint is not None
        assert "ONE" in hint or "one" in hint.lower(), (
            f"Write-loop hint must also enforce the ONE tool call rule.\n"
            f"Got: {hint!r}"
        )

    def test_list_files_loop_treated_as_search_tool(self):
        """list_files is in _SEARCH_TOOLS — must get the search-appropriate hint."""
        from app.services.loop_detection import detect_repetitive_tool_loop
        from app.services.tool_mapping import ClientType

        msgs = [
            {"role": "system", "content": "You are a coding assistant."},
            {"role": "user", "content": "Find the helm scripts."},
        ]
        for i in range(3):
            msgs.append({
                "role": "assistant",
                "content": "<list_files>\n<path>Tools/scripts</path>\n</list_files>",
            })
            msgs.append({
                "role": "user",
                "content": "[Tool Result]\nbuild.sh\nstart.sh",
            })

        hint = detect_repetitive_tool_loop(msgs, "test", ClientType.ROO_CODE)
        assert hint is not None
        assert "write_to_file" not in hint, (
            f"list_files loop must get the search hint, not the write hint.\nGot: {hint!r}"
        )

    def test_search_loop_below_threshold_no_hint(self):
        """2 identical search_files calls must not trigger the hint (threshold=3)."""
        from app.services.loop_detection import detect_repetitive_tool_loop
        from app.services.tool_mapping import ClientType

        msgs = self._build_search_loop_history("search_files", ".", repeats=2)
        hint = detect_repetitive_tool_loop(msgs, "test", ClientType.ROO_CODE)
        assert hint is None, (
            f"Hint must not fire below threshold (2 calls < 3). Got: {hint!r}"
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
# FIX v1.6.22: Unified diff → SEARCH/REPLACE conversion
# File: app/services/tool_call_fixups.py
#
# The model consistently outputs unified diff format (--- a/file / @@ -1 +1 @@)
# inside apply_diff when the user message contains a unified diff.  Instead of
# dropping these, the proxy converts them to Roo Code SEARCH/REPLACE format.
# ─────────────────────────────────────────────────────────────────────────────

class TestUnifiedDiffConversion:

    def _make_tc(self, diff: str, name: str = "apply_diff") -> list:
        return [{
            "id": "call_test",
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps({"path": "app.py", "diff": diff}),
            },
        }]

    def test_simple_unified_diff_converted(self):
        """
        A minimal unified diff is converted to SEARCH/REPLACE and NOT dropped.

        FAILS before fix: diff missing >>>>>>> REPLACE → dropped.
        PASSES after fix: unified diff detected and converted.
        """
        from app.services.tool_call_fixups import validate_apply_diff_completeness

        diff = "@@ -1 +1 @@\n-pritn('hello')\n+print('hello')\n"
        result = validate_apply_diff_completeness(self._make_tc(diff), "test")

        assert len(result) == 1, "Unified diff must be converted, not dropped"
        args = json.loads(result[0]["function"]["arguments"])
        converted = args["diff"]
        assert "<<<<<<< SEARCH" in converted
        assert "=======" in converted
        assert ">>>>>>> REPLACE" in converted
        assert "pritn('hello')" in converted
        assert "print('hello')" in converted

    def test_git_diff_headers_stripped(self):
        """The --- a/ and +++ b/ header lines must not appear in the output."""
        from app.services.tool_call_fixups import validate_apply_diff_completeness

        diff = (
            "--- a/app.py\n"
            "+++ b/app.py\n"
            "@@ -1 +1 @@\n"
            "-old\n"
            "+new\n"
        )
        result = validate_apply_diff_completeness(self._make_tc(diff), "test")
        assert len(result) == 1
        args = json.loads(result[0]["function"]["arguments"])
        converted = args["diff"]
        assert "--- a/" not in converted
        assert "+++ b/" not in converted
        assert "old" in converted
        assert "new" in converted

    def test_context_lines_in_both_halves(self):
        """Context lines (no +/- prefix) must appear in both SEARCH and REPLACE."""
        from app.services.tool_call_fixups import validate_apply_diff_completeness

        diff = (
            "@@ -1,3 +1,3 @@\n"
            " def foo():\n"
            "-    return 1\n"
            "+    return 2\n"
            " # end\n"
        )
        result = validate_apply_diff_completeness(self._make_tc(diff), "test")
        assert len(result) == 1
        args = json.loads(result[0]["function"]["arguments"])
        converted = args["diff"]
        # Both halves must contain context lines
        search_part = converted.split("=======")[0]
        replace_part = converted.split("=======")[1]
        assert "def foo():" in search_part
        assert "def foo():" in replace_part
        assert "# end" in search_part
        assert "# end" in replace_part

    def test_multiple_hunks_produce_multiple_blocks(self):
        """Multiple @@ hunks become multiple SEARCH/REPLACE blocks."""
        from app.services.tool_call_fixups import validate_apply_diff_completeness

        diff = (
            "@@ -1 +1 @@\n"
            "-old_a\n"
            "+new_a\n"
            "@@ -10 +10 @@\n"
            "-old_b\n"
            "+new_b\n"
        )
        result = validate_apply_diff_completeness(self._make_tc(diff), "test")
        assert len(result) == 1
        args = json.loads(result[0]["function"]["arguments"])
        converted = args["diff"]
        assert converted.count("<<<<<<< SEARCH") == 2
        assert converted.count(">>>>>>> REPLACE") == 2

    def test_non_unified_corrupt_diff_still_dropped(self):
        """
        A diff that has neither >>>>>>> REPLACE nor @@ headers is still dropped.
        (Truly corrupt/truncated output with no recoverable structure.)
        """
        from app.services.tool_call_fixups import validate_apply_diff_completeness

        diff = "<<<<<<< SEARCH\nold line\n=======\nnew line\n"  # missing REPLACE
        result = validate_apply_diff_completeness(self._make_tc(diff), "test")
        assert len(result) == 0, "Corrupt diff without @@ headers must still be dropped"

    def test_already_correct_diff_unchanged(self):
        """A diff already in SEARCH/REPLACE format must pass through unchanged."""
        from app.services.tool_call_fixups import validate_apply_diff_completeness

        diff = "<<<<<<< SEARCH\nold\n=======\nnew\n>>>>>>> REPLACE"
        result = validate_apply_diff_completeness(self._make_tc(diff), "test")
        assert len(result) == 1
        args = json.loads(result[0]["function"]["arguments"])
        assert args["diff"] == diff  # unchanged

    def test_bare_at_at_hunk_header_converted(self):
        """
        Some models emit a bare '@@' line (without -line,count +line,count coordinates).
        This minimal form must also trigger the unified diff converter.

        Reproduces: model output '<diff>\\n@@\\n-old\\n+new\\n</diff>'
        """
        from app.services.tool_call_fixups import validate_apply_diff_completeness

        diff = '@@\n-  "include": ["src"]\n+  "include": ["src/**/*"]\n'
        result = validate_apply_diff_completeness(self._make_tc(diff), "test")
        assert len(result) == 1, "Bare @@ hunk header must be converted, not dropped"
        args = json.loads(result[0]["function"]["arguments"])
        converted = args["diff"]
        assert "<<<<<<< SEARCH" in converted
        assert ">>>>>>> REPLACE" in converted
        assert '"include": ["src"]' in converted
        assert '"include": ["src/**/*"]' in converted


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


# ─────────────────────────────────────────────────────────────────────────────
# FIX v1.6.18: _remap_args_to_schema AttributeError for non-dict arguments
# File: app/services/tool_mapping.py
#
# When a tool call has no arguments (e.g. <list_namespaces></list_namespaces>),
# the XML parser sets arguments to '""'. json.loads('""') returns "" (a str),
# not {} (a dict). _remap_args_to_schema then crashes:
#   AttributeError: 'str' object has no attribute 'items'
# ─────────────────────────────────────────────────────────────────────────────

class TestRemapArgsNonDict:

    def _make_tc(self, name: str, arguments: str) -> list:
        return [{
            "id": "call_test",
            "type": "function",
            "function": {"name": name, "arguments": arguments},
        }]

    def _list_namespaces_tool(self):
        return _tool(
            "list_namespaces", "List Kubernetes namespaces",
            {"prefix": {"type": "string"}}, [],
        )

    def test_empty_string_args_does_not_crash(self):
        """
        Tool call with arguments='""' (JSON-encoded empty string, not object)
        must not raise AttributeError in _remap_args_to_schema.

        Reproduces the 500 error seen in prod logs (2026-04-16 09:18:32):
          <list_namespaces></list_namespaces> → arguments='""' → str.items() crash

        FAILS before fix: json.loads('""') = "" → "".items() → AttributeError
        PASSES after fix: isinstance(args, dict) check skips remapping for non-dict
        """
        from app.services.tool_mapping import _remap_args_to_schema
        tools = [self._list_namespaces_tool()]
        tc = self._make_tc("list_namespaces", '""')
        result = _remap_args_to_schema(tc, tools, "test")
        assert result[0]["function"]["name"] == "list_namespaces", (
            "Tool call must be returned unchanged when args is not a dict."
        )

    def test_null_args_does_not_crash(self):
        """arguments='null' → json.loads → None (not dict) → must skip remapping."""
        from app.services.tool_mapping import _remap_args_to_schema
        tools = [self._list_namespaces_tool()]
        tc = self._make_tc("list_namespaces", "null")
        result = _remap_args_to_schema(tc, tools, "test")
        assert result[0]["function"]["name"] == "list_namespaces"

    def test_list_args_does_not_crash(self):
        """arguments='[]' → json.loads → [] (list, not dict) → must skip remapping."""
        from app.services.tool_mapping import _remap_args_to_schema
        tools = [self._list_namespaces_tool()]
        tc = self._make_tc("list_namespaces", "[]")
        result = _remap_args_to_schema(tc, tools, "test")
        assert result[0]["function"]["name"] == "list_namespaces"

    def test_normal_dict_args_still_remapped(self):
        """Normal dict args must still be remapped correctly after the fix."""
        from app.services.tool_mapping import _remap_args_to_schema
        tools = [_tool(
            "write_to_file", "Write file",
            {"path": {"type": "string"}, "content": {"type": "string"}},
            ["path", "content"],
        )]
        tc = self._make_tc("write_to_file", json.dumps({"path": "foo.py", "content": "hi"}))
        result = _remap_args_to_schema(tc, tools, "test")
        args = json.loads(result[0]["function"]["arguments"])
        assert "path" in args and args["path"] == "foo.py"


# ─────────────────────────────────────────────────────────────────────────────
# FIX v1.6.19: Loop detection reads OpenAI-format tool_calls in history
# File: app/services/loop_detection.py
#
# When the model repeatedly calls write_to_file on the same path, the
# conversation history contains assistant messages with OpenAI-format
# tool_calls (as returned by toolproxy) — NOT XML in content.
# Previously _extract_tool_call_key only checked 'content' → loop invisible.
# ─────────────────────────────────────────────────────────────────────────────

class TestLoopDetectionOpenAIFormat:

    def _build_write_loop_history(self, path: str, repeats: int, content_varies: bool = True) -> list:
        """
        Build a normalized history where write_to_file was called N times on the
        same path using OpenAI-format tool_calls (as toolproxy returns them).
        """
        msgs = [
            {"role": "system", "content": "You are a coding assistant."},
            {"role": "user", "content": "Fix the Windows batch script."},
        ]
        for i in range(repeats):
            call_id = f"call_write_{i:02d}"
            content_val = f"@echo off\n:: version {i}\ncall tool.bat\n" if content_varies else "@echo off\ncall tool.bat\n"
            msgs.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "write_to_file",
                        "arguments": json.dumps({"path": path, "content": content_val}),
                    },
                }],
            })
            msgs.append({
                "role": "user",
                "content": f"[Tool Result]\nThe content was successfully saved to {path}.",
            })
        return msgs

    def test_openai_format_write_loop_detected(self):
        """
        write_to_file called 3× on the same path via OpenAI-format tool_calls
        must trigger detect_repetitive_tool_loop.

        Reproduces the create-topics.bat write loop (2026-04-16 11:10–11:17) where
        the model alternated kafka-topics.bat/.sh content but always wrote the same path.

        FAILS before fix: _extract_tool_call_key only checked 'content' field →
                          OpenAI-format tool_calls invisible → loop not detected
        PASSES after fix: _extract_key_from_message also checks 'tool_calls' field
        """
        from app.services.loop_detection import detect_repetitive_tool_loop
        from app.services.tool_mapping import ClientType

        path = "Tools/kafka/windows/create-topics.bat"
        msgs = self._build_write_loop_history(path, repeats=3)
        hint = detect_repetitive_tool_loop(msgs, "test", ClientType.ROO_CODE)

        assert hint is not None, (
            "detect_repetitive_tool_loop must fire after 3 consecutive write_to_file "
            "calls using OpenAI-format tool_calls in history.\n"
            "FAIL: loop detection was only checking 'content' field, missing tool_calls."
        )

    def test_below_threshold_no_detection(self):
        """2 consecutive OpenAI-format calls on same path must not trigger (threshold=3)."""
        from app.services.loop_detection import detect_repetitive_tool_loop
        from app.services.tool_mapping import ClientType

        msgs = self._build_write_loop_history("Tools/kafka/windows/create-topics.bat", repeats=2)
        hint = detect_repetitive_tool_loop(msgs, "test", ClientType.ROO_CODE)
        assert hint is None, f"Must not fire below threshold. Got: {hint!r}"

    def test_different_paths_do_not_trigger(self):
        """Alternating write_to_file on different paths must not trigger."""
        from app.services.loop_detection import detect_repetitive_tool_loop
        from app.services.tool_mapping import ClientType

        msgs = [
            {"role": "system", "content": "You are a coding assistant."},
            {"role": "user", "content": "Fix the scripts."},
        ]
        for i, path in enumerate(["script_a.bat", "script_b.bat", "script_a.bat"]):
            call_id = f"call_{i:02d}"
            msgs.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "write_to_file",
                        "arguments": json.dumps({"path": path, "content": "content"}),
                    },
                }],
            })
            msgs.append({"role": "user", "content": f"[Tool Result]\nSaved {path}."})

        hint = detect_repetitive_tool_loop(msgs, "test", ClientType.ROO_CODE)
        assert hint is None, f"Different paths must not trigger loop. Got: {hint!r}"

    def test_mixed_xml_and_openai_format_detected(self):
        """
        Mix of XML-format and OpenAI-format tool_calls on same path must also
        trigger once threshold is reached.
        """
        from app.services.loop_detection import detect_repetitive_tool_loop
        from app.services.tool_mapping import ClientType

        path = "Tools/kafka/windows/create-topics.bat"
        msgs = [
            {"role": "system", "content": "You are a coding assistant."},
            {"role": "user", "content": "Fix the script."},
            # First call: XML format in content
            {"role": "assistant", "content": f"<write_to_file>\n<path>{path}</path>\n<content>v1</content>\n</write_to_file>"},
            {"role": "user", "content": "[Tool Result]\nSaved."},
            # Second call: OpenAI format
            {"role": "assistant", "content": None, "tool_calls": [{"id": "c2", "type": "function", "function": {"name": "write_to_file", "arguments": json.dumps({"path": path, "content": "v2"})}}]},
            {"role": "user", "content": "[Tool Result]\nSaved."},
            # Third call: OpenAI format again
            {"role": "assistant", "content": None, "tool_calls": [{"id": "c3", "type": "function", "function": {"name": "write_to_file", "arguments": json.dumps({"path": path, "content": "v3"})}}]},
            {"role": "user", "content": "[Tool Result]\nSaved."},
        ]
        hint = detect_repetitive_tool_loop(msgs, "test", ClientType.ROO_CODE)
        assert hint is not None, (
            "Mixed XML/OpenAI format write_to_file calls on same path must trigger loop detection."
        )

    def test_trailing_genuine_user_message_does_not_reset_streak(self):
        """
        When the history ends with a genuine user message (e.g. "still broken,
        please fix it"), the repetitive-loop streak built from prior assistant
        calls must NOT be reset.

        Reproduces the production scenario: 3× write_to_file → [success] →
        genuine user follow-up → toolproxy call #4.  The trailing message is
        the current prompt (unresponded); excluding it from the scan preserves
        the streak so the correction hint fires.

        FAILS before fix (v1.6.20): genuine user message at end resets consecutive=0.
        PASSES after fix: trailing genuine user message is excluded from scan.
        """
        from app.services.loop_detection import detect_repetitive_tool_loop
        from app.services.tool_mapping import ClientType

        path = "Tools/kafka/windows/create-topics.bat"
        msgs = self._build_write_loop_history(path, repeats=3)
        # Append the current user prompt — exactly as toolproxy receives it
        msgs.append({"role": "user", "content": "The script still doesn't work. Please fix it."})

        hint = detect_repetitive_tool_loop(msgs, "test", ClientType.ROO_CODE)
        assert hint is not None, (
            "detect_repetitive_tool_loop must fire even when history ends with a genuine "
            "user message (the current prompt). The trailing message must be excluded from "
            "the scan so a 'please try again' doesn't erase the 3-write streak.\n"
            "FAIL: consecutive counter was reset by the trailing user message."
        )

    def test_new_genuine_user_turn_in_middle_resets_streak(self):
        """
        A genuine user message in the MIDDLE of the history (not trailing) must
        still reset the consecutive counter — it represents a real task boundary.
        """
        from app.services.loop_detection import detect_repetitive_tool_loop
        from app.services.tool_mapping import ClientType

        path = "script.bat"
        msgs = [
            {"role": "system", "content": "You are a coding assistant."},
            {"role": "user", "content": "Fix the script."},
            # 2 writes before the task boundary
            {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "write_to_file", "arguments": json.dumps({"path": path, "content": "v1"})}}]},
            {"role": "user", "content": "[Tool Result]\nSaved."},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "c2", "type": "function", "function": {"name": "write_to_file", "arguments": json.dumps({"path": path, "content": "v2"})}}]},
            {"role": "user", "content": "[Tool Result]\nSaved."},
            # Genuine user message mid-history — resets streak
            {"role": "user", "content": "Actually, write a different file instead."},
            # 2 writes after the task boundary (below threshold)
            {"role": "assistant", "content": None, "tool_calls": [{"id": "c3", "type": "function", "function": {"name": "write_to_file", "arguments": json.dumps({"path": path, "content": "v3"})}}]},
            {"role": "user", "content": "[Tool Result]\nSaved."},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "c4", "type": "function", "function": {"name": "write_to_file", "arguments": json.dumps({"path": path, "content": "v4"})}}]},
            {"role": "user", "content": "[Tool Result]\nSaved."},
        ]
        hint = detect_repetitive_tool_loop(msgs, "test", ClientType.ROO_CODE)
        assert hint is None, (
            "A genuine user message in the MIDDLE of history must reset the streak. "
            "Only 2 writes after the reset — below threshold=3. Must not fire."
        )


# ─────────────────────────────────────────────────────────────────────────────
# FIX v1.6.19: Priming teaches write_to_file → attempt_completion pattern
# File: app/services/priming.py
#
# A new static priming sequence shows: after a successful write_to_file,
# call attempt_completion. Prevents the model from rewriting the same file
# in a loop (observed: create-topics.bat written 6+ times, 2026-04-16).
# ─────────────────────────────────────────────────────────────────────────────

class TestPrimingWriteCompletionSequence:

    def _get_priming_messages(self):
        from app.services.priming import inject_priming
        from app.services.tool_mapping import ClientType
        base = [{"role": "user", "content": "Do something."}]
        return inject_priming(base, DEFAULT_TOOLS, client_type=ClientType("roo_code"))

    def test_priming_contains_write_then_attempt_completion(self):
        """
        Priming must contain a sequence:
          assistant: write_to_file
          user:      [Tool Result] successfully saved
          assistant: attempt_completion

        Teaches the model: after a successful write, stop — do not keep rewriting.

        FAILS before fix: no such sequence existed in static priming
        PASSES after fix: new three-turn sequence added to _STATIC_PRIMING_SEQUENCES
        """
        messages = self._get_priming_messages()

        found = False
        for i, msg in enumerate(messages):
            if (msg.get("role") == "assistant" and
                    "write_to_file" in msg.get("content", "") and
                    i + 2 < len(messages)):
                next_user = messages[i + 1]
                next_asst = messages[i + 2]
                if (next_user.get("role") == "user" and
                        "successfully" in next_user.get("content", "").lower() and
                        next_asst.get("role") == "assistant" and
                        "attempt_completion" in next_asst.get("content", "")):
                    found = True
                    break

        assert found, (
            "Priming must contain: write_to_file → [Tool Result] success → attempt_completion.\n"
            "This sequence teaches the model to stop after a successful write.\n"
            "Without it the model rewrites the same file in a loop."
        )

    def test_priming_write_completion_sequence_uses_bat_file(self):
        """
        The new priming sequence must use a .bat file example — specifically relevant
        to the Windows script loop bug (create-topics.bat, 2026-04-16).
        """
        messages = self._get_priming_messages()
        bat_in_priming = any(
            ".bat" in msg.get("content", "")
            for msg in messages
            if msg.get("role") == "assistant"
        )
        assert bat_in_priming, (
            "Priming must include a .bat file example for the write→completion sequence. "
            "This directly addresses the Windows script loop pattern."
        )


# ─────────────────────────────────────────────────────────────────────────────
# FIX v1.6.21: Priming requires-gating — only inject sequences for tools that
# are present in the session.  Saves tokens and avoids teaching the model
# formats for tools it cannot use.  Also adds apply_diff SEARCH/REPLACE example.
# File: app/services/priming.py
# ─────────────────────────────────────────────────────────────────────────────

class TestPrimingRequiresGating:

    def _make_tools(self, *names):
        return [_tool(n, f"desc {n}", {"path": {"type": "string"}}, ["path"]) for n in names]

    def test_apply_diff_sequence_injected_when_tool_present(self):
        """
        apply_diff priming sequence must be injected when apply_diff is in tools.
        """
        from app.services.priming import inject_priming
        from app.services.tool_mapping import ClientType

        tools = self._make_tools("apply_diff", "write_to_file", "attempt_completion")
        msgs = inject_priming([{"role": "user", "content": "Fix it."}], tools, ClientType.ROO_CODE)

        has_search_replace = any(
            "<<<<<<< SEARCH" in msg.get("content", "")
            for msg in msgs if msg.get("role") == "assistant"
        )
        assert has_search_replace, (
            "apply_diff priming must inject a SEARCH/REPLACE example when apply_diff is in tools."
        )

    def test_apply_diff_sequence_skipped_when_tool_absent(self):
        """
        apply_diff priming sequence must NOT be injected when apply_diff is NOT in tools.
        Token cost is avoided for sessions that don't use apply_diff.
        """
        from app.services.priming import inject_priming
        from app.services.tool_mapping import ClientType

        tools = self._make_tools("read_file", "write_to_file", "attempt_completion")
        msgs = inject_priming([{"role": "user", "content": "Fix it."}], tools, ClientType.ROO_CODE)

        has_search_replace = any(
            "<<<<<<< SEARCH" in msg.get("content", "")
            for msg in msgs if msg.get("role") == "assistant"
        )
        assert not has_search_replace, (
            "apply_diff priming must NOT be injected when apply_diff is absent from tools. "
            "Sessions without apply_diff should not pay the token cost."
        )

    def test_execute_command_sequence_skipped_when_absent(self):
        """
        execute_command (rename/mv) priming must be skipped when execute_command is not in tools.
        """
        from app.services.priming import inject_priming
        from app.services.tool_mapping import ClientType

        tools = self._make_tools("read_file", "write_to_file")
        msgs = inject_priming([{"role": "user", "content": "Do something."}], tools, ClientType.ROO_CODE)

        has_mv = any(
            "mv old_folder" in msg.get("content", "")
            for msg in msgs if msg.get("role") == "assistant"
        )
        assert not has_mv, (
            "execute_command priming must be skipped when execute_command is not in tools."
        )

    def test_token_savings_minimal_toolset(self):
        """
        A session with only read_file + attempt_completion must receive fewer
        priming messages than a session with the full Roo Code tool set.
        """
        from app.services.priming import inject_priming
        from app.services.tool_mapping import ClientType

        base = [{"role": "user", "content": "Check the file."}]
        minimal_tools = self._make_tools("read_file", "attempt_completion")
        full_tools = self._make_tools(
            "read_file", "write_to_file", "apply_diff",
            "execute_command", "attempt_completion",
        )

        minimal_msgs = inject_priming(base[:], minimal_tools, ClientType.ROO_CODE)
        full_msgs = inject_priming(base[:], full_tools, ClientType.ROO_CODE)

        assert len(minimal_msgs) < len(full_msgs), (
            f"Minimal tool set should produce fewer priming messages than full set. "
            f"Got minimal={len(minimal_msgs)} vs full={len(full_msgs)}."
        )


# ─────────────────────────────────────────────────────────────────────────────
# FIX v1.6.23: remove use_mcp_tool hint injection for raw XML with tools=0
# File: app/main.py
#
# The hint polluted the model's context when the client (e.g. Roo Code in
# XML-passthrough mode) was already handling the raw XML natively.
# The "[toolproxy] No tool definitions registered..." text was fed back into
# the model context on every turn, causing drift and confusing the model.
# Fix: remove the hint injection entirely — clients that handle XML-in-text
# do not need guidance; clients using use_mcp_tool already work via MCP path.
# ─────────────────────────────────────────────────────────────────────────────

class TestMcpHintInjection:

    def test_raw_xml_with_no_tools_passes_through_unchanged(self, client):
        """
        Model outputs raw <list_datasources>...</list_datasources> with tools=[].
        Response content must be passed through as-is — no hint appended.

        v1.6.23: hint injection removed because it polluted model context for
        clients that handle XML-in-text natively (Roo Code XML passthrough mode).
        """
        from unittest.mock import AsyncMock, patch
        import app.main as main_module
        from tests.conftest import llm_response

        raw_xml = "<list_datasources>\n<type>prometheus</type>\n</list_datasources>"
        with patch.object(main_module, "upstream_client") as mock_uc:
            mock_uc.chat_completion = AsyncMock(return_value=llm_response(raw_xml))
            resp = client.post("/v1/chat/completions", json={
                "model": "oracle-llm",
                "messages": [{"role": "user", "content": "List Prometheus datasources."}],
                "tools": [],
            })

        assert resp.status_code == 200, resp.text
        msg = resp.json()["choices"][0]["message"]
        content = msg.get("content") or ""

        assert "use_mcp_tool" not in content, (
            "Hint must NOT be injected — hint injection was removed in v1.6.23.\n"
            f"Content: {content[:300]!r}"
        )
        assert "[toolproxy]" not in content, (
            "toolproxy must not append any internal hint text to client responses."
        )
        assert not msg.get("tool_calls"), "No tool_calls for raw XML with tools=0"

    def test_raw_xml_with_tools_defined_no_hint(self, client):
        """When tools ARE defined, XML is parsed normally — no hint injected."""
        from unittest.mock import AsyncMock, patch
        import app.main as main_module
        from tests.conftest import llm_response, TOOL_READ_FILE, TOOL_ATTEMPT_COMPLETION

        with patch.object(main_module, "upstream_client") as mock_uc:
            mock_uc.chat_completion = AsyncMock(
                return_value=llm_response("<read_file>\n<path>README.md</path>\n</read_file>")
            )
            resp = client.post("/v1/chat/completions", json={
                "model": "oracle-llm",
                "messages": [{"role": "user", "content": "Read the README."}],
                "tools": [TOOL_READ_FILE, TOOL_ATTEMPT_COMPLETION],
            })

        assert resp.status_code == 200, resp.text
        msg = resp.json()["choices"][0]["message"]
        tool_calls = msg.get("tool_calls") or []
        content = msg.get("content") or ""

        assert tool_calls, "XML with tools defined must be parsed as a tool call"
        assert "use_mcp_tool" not in content, "Hint must not appear when XML was parsed correctly"

    def test_plain_text_with_no_tools_no_hint(self, client):
        """Plain text response (no XML) with tools=[] must not trigger hint."""
        from unittest.mock import AsyncMock, patch
        import app.main as main_module
        from tests.conftest import llm_response

        with patch.object(main_module, "upstream_client") as mock_uc:
            mock_uc.chat_completion = AsyncMock(
                return_value=llm_response("The Prometheus datasource UID is abc123.")
            )
            resp = client.post("/v1/chat/completions", json={
                "model": "oracle-llm",
                "messages": [{"role": "user", "content": "What is the Prometheus UID?"}],
                "tools": [],
            })

        assert resp.status_code == 200, resp.text
        content = resp.json()["choices"][0]["message"].get("content") or ""
        assert "use_mcp_tool" not in content, "Hint must not appear for plain text responses"

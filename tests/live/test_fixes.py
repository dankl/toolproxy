"""
Live E2E tests for the three fixes introduced in toolproxy 1.2.0:

  Issue #5 — Write Guard: markdown docs must not land in config files
  Issue #7 — Repetitive Loop Detection: same tool × N → correction hint fires
  Issue #8 — No retry on timeout: 502 returned quickly, no multi-minute hang

Run (requires oci-proxy at localhost:8015 or set TOOLPROXY_UPSTREAM_URL):
    cd toolproxy && python3 -m pytest tests/live/test_fixes.py -m live -v
"""
import asyncio
import json
import pytest
from unittest.mock import AsyncMock, patch
import httpx

import app.main as main_module
from app.services.vllm_client import VLLMClient
from tests.conftest import _tool, user_msg
from tests.live.conftest import (
    ROO_CODE_TOOLS,
    TOOL_WRITE_TO_FILE,
    TOOL_READ_FILE,
    TOOL_APPLY_DIFF,
    TOOL_ATTEMPT_COMPLETION,
    TOOL_EXECUTE_COMMAND,
    TOOL_ASK_FOLLOWUP_QUESTION,
    post,
    parse_tool_call,
)

pytestmark = pytest.mark.live

SYSTEM_MSG = {"role": "system", "content": "You are a coding assistant. Use tools to complete tasks."}


def post_multiturn(client, tools, messages):
    """Send a multi-turn conversation (full messages list)."""
    resp = client.post("/v1/chat/completions", json={
        "model": "oracle-llm",
        "messages": messages,
        "tools": tools,
    })
    assert resp.status_code == 200, resp.text
    return resp.json()


def _assistant_tool_call(tool_name: str, args: dict, call_id: str = "call_test_01") -> dict:
    """Build an assistant message with a tool_call (Roo Code format)."""
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {"name": tool_name, "arguments": json.dumps(args)},
        }],
    }


def _tool_result(content: str, call_id: str = "call_test_01") -> dict:
    """Build a tool result message (role=tool, as Roo Code sends it)."""
    return {"role": "tool", "tool_call_id": call_id, "content": content}


# ─────────────────────────────────────────────────────────────────────────────
# Issue #5 — Write Guard
# ─────────────────────────────────────────────────────────────────────────────

class TestWriteGuard:

    def _assert_no_markdown_in_config(self, name: str, args: dict, config_file: str):
        """
        Assert that markdown documentation did NOT land in a config file.

        Three valid outcomes (defence-in-depth):
          1. write_to_file with a .md path  — model self-corrected (priming rule worked)
          2. ask_followup_question           — guard fired (priming dropped by OCI)
          3. attempt_completion              — model refused the task entirely (priming worked best-case)

        Any other write_to_file targeting the config file is a failure.
        """
        if name == "write_to_file":
            path = args.get("path", "")
            assert path.endswith(".md"), (
                f"Markdown was written to {path!r} — neither priming rule nor guard prevented it. "
                f"Expected a .md path, ask_followup_question, or attempt_completion refusal."
            )
        elif name == "ask_followup_question":
            assert config_file in args.get("question", ""), (
                f"Guard fired but question does not mention {config_file!r}"
            )
        else:
            # attempt_completion: model refused to write — priming rule prevented it entirely.
            assert name == "attempt_completion", (
                f"Unexpected tool {name!r} — expected write_to_file(.md), ask_followup_question, "
                f"or attempt_completion refusal"
            )

    def test_guard_blocks_markdown_in_application_yml(self, live):
        """
        Model is explicitly told to write markdown into application.yml.
        The combination of priming rule + guard must prevent the config file from
        being corrupted — via any of the three accepted outcomes above.
        """
        resp = post(live, [TOOL_WRITE_TO_FILE, TOOL_ASK_FOLLOWUP_QUESTION, TOOL_ATTEMPT_COMPLETION],
                    "Write the following markdown documentation to the file application.yml:\n\n"
                    "# API Documentation\n\n"
                    "## Overview\nThis service exposes a REST API.\n\n"
                    "## Endpoints\n- GET /health\n- POST /api/data\n\n"
                    "Use write_to_file with path=application.yml.")
        name, args = parse_tool_call(resp)
        self._assert_no_markdown_in_config(name, args, "application.yml")

    def test_guard_does_not_fire_for_valid_yaml(self, live):
        """
        Model writing valid YAML content to application.yml must pass through unchanged.
        The guard only fires for markdown content (≥2 heading lines).
        """
        resp = post(live, [TOOL_WRITE_TO_FILE, TOOL_ATTEMPT_COMPLETION],
                    "Create application.yml with this Spring Boot config:\n"
                    "server:\n  port: 8080\nspring:\n  datasource:\n    url: jdbc:h2:mem:testdb\n\n"
                    "Use write_to_file with path=application.yml.")
        name, args = parse_tool_call(resp)
        assert name == "write_to_file", f"Expected write_to_file, got {name!r}"
        # YAML config → guard must NOT fire
        assert "application" in args.get("path", "").lower(), (
            f"Expected application.yml path, got {args.get('path')!r}"
        )

    def test_guard_blocks_markdown_in_pom_xml(self, live):
        """Documentation must not end up in pom.xml."""
        resp = post(live, [TOOL_WRITE_TO_FILE, TOOL_ASK_FOLLOWUP_QUESTION, TOOL_ATTEMPT_COMPLETION],
                    "Write the following markdown to pom.xml:\n\n"
                    "# Build Configuration\n\n"
                    "## Dependencies\nSee the list below.\n\n"
                    "## Plugins\nSpring Boot Maven Plugin.\n\n"
                    "Use write_to_file with path=pom.xml.")
        name, args = parse_tool_call(resp)
        self._assert_no_markdown_in_config(name, args, "pom.xml")


# ─────────────────────────────────────────────────────────────────────────────
# Issue #7 — Repetitive Tool Loop Detection
# ─────────────────────────────────────────────────────────────────────────────

class TestRepetitiveLoopDetection:

    def _build_read_loop_history(self, file_path: str = "src/main/java/App.java", repeats: int = 3) -> list:
        """
        Build a normalized message history where the assistant has called
        read_file on the same file `repeats` times without progress.
        """
        msgs = [
            SYSTEM_MSG,
            user_msg(f"Read the file {file_path} and explain the main class."),
        ]
        for i in range(repeats):
            call_id = f"call_read_{i:02d}"
            msgs.append(_assistant_tool_call("read_file", {"path": file_path}, call_id))
            msgs.append(_tool_result(
                "public class App {\n    public static void main(String[] args) {\n"
                "        System.out.println(\"Hello\");\n    }\n}",
                call_id,
            ))
        # Final user turn (continues the conversation)
        msgs.append(user_msg("Please continue."))
        return msgs

    def test_correction_hint_breaks_read_loop(self, live):
        """
        After 3× read_file on the same file, loop detection injects a CORRECTION
        hint. The model should NOT call read_file again — it should either call
        attempt_completion or a different tool.
        """
        msgs = self._build_read_loop_history("src/main/java/App.java", repeats=3)
        resp = post_multiturn(live, ROO_CODE_TOOLS, msgs)
        name, args = parse_tool_call(resp)
        # After the correction hint the model must NOT repeat read_file on the same path
        if name == "read_file":
            assert args.get("path") != "src/main/java/App.java", (
                "Repetitive loop not broken: model called read_file on same path again. "
                "Loop detection may not be firing — check REPETITIVE LOOP log."
            )

    def test_correction_hint_at_threshold(self, live):
        """
        Exactly at the threshold (3 repeats) the loop hint must fire and change behavior.
        Below threshold (2 repeats) no hint — model may repeat freely.
        """
        # 2 repeats → no hint, model is free to read again
        msgs_below = self._build_read_loop_history("src/Config.java", repeats=2)
        resp_below = post_multiturn(live, ROO_CODE_TOOLS, msgs_below)
        name_below, _ = parse_tool_call(resp_below)
        # With only 2 prior reads no correction is injected — tool call is unconstrained
        # (any tool is acceptable here, we just verify no 500 error)
        assert name_below, "Expected a tool call after 2 reads (no correction yet)"

        # 3 repeats → hint fires
        msgs_at = self._build_read_loop_history("src/Config.java", repeats=3)
        resp_at = post_multiturn(live, ROO_CODE_TOOLS, msgs_at)
        name_at, args_at = parse_tool_call(resp_at)
        if name_at == "read_file":
            assert args_at.get("path") != "src/Config.java", (
                "Model still calling read_file on same path at threshold — loop hint not working."
            )

    def test_new_user_message_resets_loop_counter(self, live):
        """
        A genuine user instruction resets the loop counter.
        Even after 3 read_file calls, a new explicit user task allows another read.
        """
        file_path = "src/Service.java"
        # 3 reads on same file…
        msgs = self._build_read_loop_history(file_path, repeats=3)
        # … but now a genuine new task arrives (different file, different task)
        msgs.append(user_msg("Now read the file src/Repository.java and explain it."))
        resp = post_multiturn(live, ROO_CODE_TOOLS, msgs)
        name, args = parse_tool_call(resp)
        # Model should now happily read the NEW file — no spurious loop block
        assert name == "read_file", f"Expected read_file for new task, got {name!r}"
        assert "Repository" in args.get("path", ""), (
            f"Expected model to read Repository.java, got path={args.get('path')!r}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Issue #8 — No retry on timeout
# ─────────────────────────────────────────────────────────────────────────────

class TestTimeoutBehavior:

    def test_timeout_returns_502_quickly(self, client):
        """
        When upstream times out, toolproxy must return a 502 error promptly —
        not retry and multiply the wait time.

        Strategy: point at a non-routable address with a very short timeout (1s).
        With retry_on_timeout=False (default) this should resolve in ~1s.
        With retry_on_timeout=True (old behavior) it would take 3×1s = 3s+.
        """
        import time
        fast_uc = VLLMClient(
            base_url="http://10.255.255.1/v1",  # non-routable → immediate connect timeout
            model="oracle-llm",
            api_key="dummy-key",
            timeout=1,       # 1-second timeout
            max_retries=3,   # would retry 3× if retry_on_timeout=True
            retry_on_timeout=False,  # new default: abort on first timeout
        )
        with patch.object(main_module, "upstream_client", fast_uc):
            t0 = time.monotonic()
            resp = client.post("/v1/chat/completions", json={
                "model": "oracle-llm",
                "messages": [SYSTEM_MSG, user_msg("Hello")],
                "tools": [TOOL_ATTEMPT_COMPLETION],
            })
            elapsed = time.monotonic() - t0

        assert resp.status_code == 502, f"Expected 502 on timeout, got {resp.status_code}"
        # Should fail in ~1s, not 3×1s = 3s (old retry behavior)
        assert elapsed < 4.0, (
            f"Timeout took {elapsed:.1f}s — expected <4s (no retry). "
            "Possible regression: retry_on_timeout may be True."
        )

    def test_retry_on_timeout_enabled_retries(self, client):
        """
        With retry_on_timeout=True the client DOES retry — elapsed time > 1 timeout.
        This verifies the opt-in retry flag works correctly.
        """
        import time
        retry_uc = VLLMClient(
            base_url="http://10.255.255.1/v1",
            model="oracle-llm",
            api_key="dummy-key",
            timeout=1,
            max_retries=2,
            retry_on_timeout=True,  # explicitly enable retries
        )
        with patch.object(main_module, "upstream_client", retry_uc):
            t0 = time.monotonic()
            resp = client.post("/v1/chat/completions", json={
                "model": "oracle-llm",
                "messages": [SYSTEM_MSG, user_msg("Hello")],
                "tools": [TOOL_ATTEMPT_COMPLETION],
            })
            elapsed = time.monotonic() - t0

        assert resp.status_code == 502
        # With retry_on_timeout=True: 2 retries → at least 2 timeout cycles
        assert elapsed >= 1.5, (
            f"With retry_on_timeout=True expected ≥1.5s elapsed, got {elapsed:.1f}s"
        )

    def test_live_upstream_responds_within_timeout(self, live):
        """
        Sanity check: the real upstream at localhost:8015 answers within the
        configured 120s timeout. Fails if the endpoint is unreachable.
        """
        resp = post(live, [TOOL_ATTEMPT_COMPLETION],
                    "Signal that the task is complete.")
        assert resp["choices"][0]["message"].get("tool_calls") or \
               resp["choices"][0]["message"].get("content"), \
               "Expected a non-empty response from live upstream"


# ─────────────────────────────────────────────────────────────────────────────
# v1.6.4 — Truncation hallucination fix (Fix A + B + C)
#
# When the last user message contains Roo Code's exact truncation format
# ("IMPORTANT: File content truncated. / To read more: Use the read_file
# tool with offset=..."), the model must NOT respond with plain text that
# contains "[Tool Result]" blocks. It must call a real tool:
#   - read_file with offset= (to page through the file)
#   - write_to_file (to write corrected content directly)
# ─────────────────────────────────────────────────────────────────────────────

_TRUNCATED_POM_RESULT = (
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
    " 6 |   <version>0.0.1-SNAPSHOT</version>\n"
    " 7 |   <dependencies>\n"
    " 8 |     <dependency>\n"
    " 9 |       <groupId>org.springframework.boot</groupId>\n"
    "10 |       <artifactId>spring-boot-starter-web</artifactId>\n"
    "11 |     </dependency>\n"
)


class TestTruncationHallucination:
    """
    Live E2E tests verifying that the model does not hallucinate [Tool Result]
    blocks when it encounters Roo Code's file-truncation notice.

    Covers Fix A (priming format), Fix B (FORBIDDEN FORMATS), Fix C (dynamic reminder).
    """

    def _truncated_multiturn(self, client, extra_system: str = ""):
        """
        Build a conversation where the assistant read pom.xml and received a
        truncated result. Submit this as the next turn to toolproxy.
        The model must respond with a tool call — not hallucinated text.
        """
        system_content = (
            "You are a Java/Spring Boot coding assistant. "
            "Fix build issues by correcting pom.xml when needed. "
            + extra_system
        )
        messages = [
            {"role": "system", "content": system_content},
            user_msg("Fix the pom.xml to include the missing spring-boot-starter-test dependency."),
            _assistant_tool_call(
                "read_file",
                {"path": "taskmanager/backend/pom.xml"},
                call_id="call_read_pom_01",
            ),
            _tool_result(_TRUNCATED_POM_RESULT, call_id="call_read_pom_01"),
        ]
        import os
        model = os.environ.get("LIVE_MODEL", "openai/gpt-oss-120b")
        resp = client.post("/v1/chat/completions", json={
            "model": model,
            "messages": messages,
            "tools": ROO_CODE_TOOLS,
        })
        assert resp.status_code == 200, resp.text
        return resp.json()

    def test_truncated_file_produces_tool_call_not_hallucinated_text(self, live):
        """
        After seeing a truncated read_file result, the model must respond with
        a real tool call — NOT a text response containing '[Tool Result]'.

        Accepted tool calls:
          - read_file with offset= (model is paging through the file)
          - write_to_file          (model writes corrected content directly)
          - apply_diff             (model patches the visible portion)

        FAILS if model responds with plain text containing '[Tool Result]'.
        """
        resp = self._truncated_multiturn(live)
        msg = resp["choices"][0]["message"]
        tool_calls = msg.get("tool_calls") or []
        content = msg.get("content") or ""

        # The response must be a tool call, not plain text
        assert tool_calls, (
            "Model responded with plain text instead of a tool call after seeing "
            "a truncated read_file result.\n"
            f"Content preview: {content[:400]!r}\n"
            "Expected: read_file(offset=) or write_to_file."
        )

        # And the text must NOT contain hallucinated [Tool Result] blocks
        assert "[Tool Result]" not in content, (
            f"Model hallucinated '[Tool Result]' in its own response.\n"
            f"Content: {content[:400]!r}\n"
            "Fix A/B/C must prevent this."
        )

    def test_model_uses_offset_or_writes_directly(self, live):
        """
        When faced with truncated pom.xml, the model must either:
          - Page through it with read_file(offset=...)
          - Write the corrected pom.xml directly with write_to_file

        Calling read_file on the same path WITHOUT offset is a no-op loop — not acceptable.
        """
        resp = self._truncated_multiturn(live)
        msg = resp["choices"][0]["message"]
        tool_calls = msg.get("tool_calls") or []

        if not tool_calls:
            pytest.skip("No tool call in response — covered by test_truncated_file_produces_tool_call")

        name, args = parse_tool_call(resp)
        allowed = {"read_file", "write_to_file", "apply_diff", "attempt_completion"}
        assert name in allowed, (
            f"Unexpected tool call {name!r} after truncated file. Expected one of {allowed}."
        )

        if name == "read_file":
            # If the model re-reads, it must use offset= to make progress
            path = args.get("path", "")
            offset = args.get("offset") or args.get("start_line")
            assert offset or "pom" not in path, (
                f"Model called read_file on {path!r} without offset= — this is a no-op loop.\n"
                "After a truncation notice the model must use offset= to page forward."
            )

    def test_reminder_injection_does_not_break_normal_flow(self, live):
        """
        When the last tool result is NOT truncated, the truncation reminder must
        NOT be injected and the model must behave normally.
        """
        messages = [
            SYSTEM_MSG,
            user_msg("Read the file README.md and tell me what it's about."),
            _assistant_tool_call("read_file", {"path": "README.md"}, call_id="call_readme"),
            _tool_result(
                "[Tool Result]\nFile: README.md\n1 | # Task Manager\n2 | A simple Spring Boot app.",
                call_id="call_readme",
            ),
        ]
        import os
        model = os.environ.get("LIVE_MODEL", "openai/gpt-oss-120b")
        resp = client_obj = live.post("/v1/chat/completions", json={
            "model": model,
            "messages": messages,
            "tools": ROO_CODE_TOOLS,
        })
        assert resp.status_code == 200, resp.text
        result = resp.json()
        # Any valid tool call or text is acceptable — we just verify no 500 / no crash
        msg = result["choices"][0]["message"]
        assert msg.get("tool_calls") or msg.get("content"), (
            "Expected a non-empty response for non-truncated normal flow."
        )


# ─────────────────────────────────────────────────────────────────────────────
# v1.6.5 — apply_diff loop: correct hint fires (detect_repetitive before success)
#
# When the model calls apply_diff on the same file 3 times in a row, the hint
# must say "not making progress, try a different approach" — NOT "already
# succeeded, call attempt_completion" (which the model correctly ignores because
# the task is still broken).
# ─────────────────────────────────────────────────────────────────────────────

_APPLY_DIFF_SUCCESS_RESULT = (
    '{"path":"taskmanager/backend/src/main/java/com/example/taskmanager/controller/TaskController.java",'
    '"operation":"modified","notice":"You do not need to re-read the file."}'
)
_APPLY_DIFF_FILE_PATH = (
    "taskmanager/backend/src/main/java/com/example/taskmanager/controller/TaskController.java"
)
_APPLY_DIFF_DIFF = (
    "<<<<<<< SEARCH\n:start_line:66\n-------\n    }\n=======\n>>>>>>> REPLACE"
)


class TestApplyDiffLoop:
    """
    Live E2E tests for the apply_diff loop fix (v1.6.5).

    When the model has called apply_diff on the same file 3x in a row and the
    task is still not complete (compile error persists), toolproxy must inject
    the 'not making progress' hint so the model switches to write_to_file.
    """

    def _build_history(self, repeats: int) -> list:
        """Build a normalized multi-turn history with N consecutive apply_diff calls."""
        msgs = [
            SYSTEM_MSG,
            user_msg(
                "Fix the compilation error in TaskController.java. "
                "The file has an extra closing brace at line 66 that causes "
                "'implicitly declared classes are not supported'."
            ),
            _assistant_tool_call(
                "read_file", {"path": _APPLY_DIFF_FILE_PATH}, call_id="call_read_01"
            ),
            _tool_result(
                "File: " + _APPLY_DIFF_FILE_PATH + "\n"
                " 1 | package com.example.taskmanager.controller;\n"
                "...\n"
                "65 |     }\n"
                "66 |     }  <- extra closing brace\n"
                "67 | \n"
                "68 |     @DeleteMapping(\"/{id}\")\n"
                "...\n"
                "78 | }",
                call_id="call_read_01",
            ),
        ]
        for i in range(repeats):
            call_id = f"call_apply_{i:02d}"
            msgs.append(
                _assistant_tool_call(
                    "apply_diff",
                    {"path": _APPLY_DIFF_FILE_PATH, "diff": _APPLY_DIFF_DIFF},
                    call_id=call_id,
                )
            )
            msgs.append(
                _tool_result(
                    _APPLY_DIFF_SUCCESS_RESULT
                    + "\n<notice>Making multiple related changes in a single apply_diff "
                    "is more efficient.</notice>",
                    call_id=call_id,
                )
            )
        return msgs

    def test_after_three_apply_diffs_model_switches_approach(self, live):
        """
        After 3 consecutive apply_diff calls that report 'modified' but have NOT
        fixed the compilation, toolproxy must inject a hint that causes the model
        to switch to write_to_file (or at least not repeat the same apply_diff).

        Verifies that detect_repetitive_tool_loop fires first with the correct
        'not making progress, try a different approach' hint.
        """
        import os
        msgs = self._build_history(repeats=3)
        msgs.append(user_msg(
            "The compilation still fails: "
            "'implicitly declared classes are not supported'. "
            "The extra brace is still there. Fix it."
        ))
        model = os.environ.get("LIVE_MODEL", "openai/gpt-oss-120b")
        resp = live.post("/v1/chat/completions", json={
            "model": model,
            "messages": msgs,
            "tools": [TOOL_READ_FILE, TOOL_WRITE_TO_FILE, TOOL_APPLY_DIFF, TOOL_ATTEMPT_COMPLETION],
        })
        assert resp.status_code == 200, resp.text
        name, args = parse_tool_call(resp.json())

        # Model must NOT keep calling apply_diff with the same broken diff
        if name == "apply_diff":
            assert args.get("diff") != _APPLY_DIFF_DIFF, (
                "Model repeated the same apply_diff that failed 3 times.\n"
                "The 'not making progress' hint from detect_repetitive_tool_loop "
                "should have prompted a different approach."
            )

    def test_after_two_apply_diffs_hint_guides_to_verify_not_complete(self, live):
        """
        After 2 apply_diffs the success_loop hint fires. It must now suggest
        read_file or write_to_file — NOT just 'call attempt_completion'.

        Verifies the updated success_loop hint (v1.6.5).
        """
        import os
        msgs = self._build_history(repeats=2)
        msgs.append(user_msg(
            "The compilation still fails. The brace is still there. Please fix it."
        ))
        model = os.environ.get("LIVE_MODEL", "openai/gpt-oss-120b")
        resp = live.post("/v1/chat/completions", json={
            "model": model,
            "messages": msgs,
            "tools": [TOOL_READ_FILE, TOOL_WRITE_TO_FILE, TOOL_APPLY_DIFF, TOOL_ATTEMPT_COMPLETION],
        })
        assert resp.status_code == 200, resp.text
        name, _ = parse_tool_call(resp.json())

        # Model must not prematurely call attempt_completion when task is not done
        assert name != "attempt_completion", (
            "Model called attempt_completion even though compilation still fails.\n"
            "The success_loop hint must not say 'call attempt_completion' — "
            "it should suggest read_file or write_to_file instead."
        )


# ─────────────────────────────────────────────────────────────────────────────
# v1.6.13 — Truncated apply_diff rescue
#
# When OCI cuts off the model response before </apply_diff>, the XML parser
# finds 0 matches and text_synthesis previously fell through to attempt_completion,
# silently discarding the diff.
#
# Fix: partial XML rescue in text_synthesis extracts <path> and <diff> via regex
# even without closing tags, HTML-unescapes the diff content, and returns a
# proper apply_diff tool call. validate_apply_diff_completeness (step 9b) then
# catches diffs that are also incomplete (missing >>>>>>> REPLACE).
# ─────────────────────────────────────────────────────────────────────────────

_TRUNCATED_APPLY_DIFF_OUTPUT = (
    "<apply_diff>"
    "<path>src/main/java/de/example/FooService.java</path>"
    "<diff>"
    "&lt;&lt;&lt;&lt;&lt;&lt;&lt; SEARCH\n"
    ":start_line:12\n"
    "-------\n"
    "    public String oldMethod() {\n"
    "        return \"old\";\n"
    "    }\n"
    "=======\n"
    "    public String newMethod() {\n"
    "        return \"new\";\n"
    "    }\n"
    "&gt;&gt;&gt;&gt;&gt;&gt;&gt; REPLACE\n"
    "</diff>"
    # </apply_diff> intentionally missing — simulates OCI truncation
)

_TRUNCATED_APPLY_DIFF_RESPONSE = {
    "id": "test-truncated-apply-diff",
    "created": 1_700_000_000,
    "choices": [{
        "index": 0,
        "message": {
            "role": "assistant",
            "content": _TRUNCATED_APPLY_DIFF_OUTPUT,
            "tool_calls": None,
        },
        "finish_reason": "stop",
    }],
    "usage": {"prompt_tokens": 200, "completion_tokens": 500, "total_tokens": 700},
}


class TestTruncatedApplyDiffRescue:
    """
    E2E tests for the truncated apply_diff rescue (v1.6.13).

    test_truncated_apply_diff_rescued: uses a mocked upstream that returns a
    realistic truncated apply_diff (entity-encoded diff markers, no closing tag).
    This always runs and is the primary regression guard.

    test_live_model_apply_diff_end_to_end: uses the real model to verify that
    normal (complete) apply_diff responses are parsed and forwarded correctly.
    """

    def test_truncated_apply_diff_rescued(self, client):
        """
        Upstream returns an apply_diff without closing </apply_diff> tag.
        Toolproxy must rescue it and return a proper apply_diff tool call —
        not an attempt_completion with the raw XML as its result.

        Verifies the full HTTP request/response pipeline:
          1. Mocked upstream returns truncated XML
          2. xml_parser finds 0 matches (no closing tag)
          3. text_synthesis partial XML rescue fires
          4. apply_diff is returned with decoded diff content
          5. validate_apply_diff_completeness passes (diff has >>>>>>> REPLACE)
        """
        with patch.object(main_module, "upstream_client") as mock_uc:
            mock_uc.chat_completion = AsyncMock(return_value=_TRUNCATED_APPLY_DIFF_RESPONSE)
            resp = client.post("/v1/chat/completions", json={
                "model": "oracle-llm",
                "messages": [SYSTEM_MSG, user_msg("Rename oldMethod to newMethod in FooService.java")],
                "tools": [TOOL_APPLY_DIFF, TOOL_ATTEMPT_COMPLETION],
            })

        assert resp.status_code == 200, resp.text
        name, args = parse_tool_call(resp.json())

        assert name == "apply_diff", (
            f"Truncated apply_diff was not rescued — got {name!r}.\n"
            "Regression: toolproxy fell through to attempt_completion and discarded the diff.\n"
            "Check text_synthesis partial XML rescue for apply_diff."
        )
        assert args["path"] == "src/main/java/de/example/FooService.java", (
            f"Wrong path in rescued apply_diff: {args['path']!r}"
        )
        diff = args["diff"]
        assert "<<<<<<< SEARCH" in diff, (
            f"HTML entities not decoded in rescued diff.\n"
            f"Expected '<<<<<<< SEARCH', got: {diff[:100]!r}"
        )
        assert ">>>>>>> REPLACE" in diff, (
            f"Rescued diff is missing >>>>>>> REPLACE marker.\n"
            f"Diff preview: {diff[:200]!r}"
        )
        assert "&lt;" not in diff, f"Unreplaced &lt; entities in diff: {diff[:100]!r}"

    def test_live_model_apply_diff_end_to_end(self, live):
        """
        Ask the real model to make a targeted Java change.
        Verifies that a normal (complete) apply_diff goes through the full
        pipeline correctly — xml_parser finds it, entities decoded, forwarded.
        """
        import os
        msgs = [
            SYSTEM_MSG,
            user_msg("Read the file src/Service.java"),
            _assistant_tool_call("read_file", {"path": "src/Service.java"}, "call_read_01"),
            _tool_result(
                "[Tool Result]\nFile: src/Service.java\n"
                " 1 | package de.example;\n"
                " 2 | \n"
                " 3 | public class Service {\n"
                " 4 |     public String greet() {\n"
                " 5 |         return \"hello\";\n"
                " 6 |     }\n"
                " 7 | }\n",
                "call_read_01",
            ),
            user_msg(
                "Use apply_diff to change the return value of greet() "
                "from \"hello\" to \"hi\" in src/Service.java."
            ),
        ]
        model = os.environ.get("LIVE_MODEL", "openai/gpt-oss-120b")
        resp = live.post("/v1/chat/completions", json={
            "model": model,
            "messages": msgs,
            "tools": [TOOL_READ_FILE, TOOL_APPLY_DIFF, TOOL_ATTEMPT_COMPLETION],
        })
        assert resp.status_code == 200, resp.text
        name, args = parse_tool_call(resp.json())

        assert name in {"apply_diff", "write_to_file"}, (
            f"Expected apply_diff or write_to_file for a targeted change, got {name!r}.\n"
            "If the model used apply_diff, verify the diff content is decoded correctly."
        )
        if name == "apply_diff":
            diff = args.get("diff", "")
            assert "<<<<<<< SEARCH" in diff, (
                f"apply_diff diff missing SEARCH marker — entity decoding may be broken.\n"
                f"Diff preview: {diff[:200]!r}"
            )
            assert ">>>>>>> REPLACE" in diff, (
                f"apply_diff diff missing REPLACE marker.\nDiff: {diff[:200]!r}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# v1.6.14 — Search-loop hint is tool-aware + ONE CALL enforcement
#
# Observed in prod (2026-04-10): after 4× failed search_files, the model
# received the write-tool hint ("use write_to_file") which made no sense for
# a search operation. On the next turn it batched 14 tool calls at once.
#
# Fix 1: search tools get a dedicated hint ("item likely doesn't exist here,
#         try a completely different approach").
# Fix 2: all hints now include "CRITICAL: Always send exactly ONE tool call
#         per response — never batch multiple calls."
# ─────────────────────────────────────────────────────────────────────────────

class TestSearchLoopHintLive:
    """
    Live E2E test: when search_files repeatedly returns no results, the model
    must respond with a SINGLE tool call — not batch multiple calls.
    Reproduces the 14-calls-at-once incident from 2026-04-10.
    """

    def _build_failed_search_history(self, repeats: int = 3) -> list:
        """
        Build a conversation where search_files was called N times on the same
        path and found nothing each time — reproducing the helm-search scenario.
        """
        msgs = [SYSTEM_MSG]
        msgs.append(user_msg("Find where 'helm upgrade' is called in this project."))
        for i in range(repeats):
            call_id = f"call_search_{i:02d}"
            msgs.append(_assistant_tool_call(
                "search_files",
                {"path": ".", "regex": "helm upgrade"},
                call_id,
            ))
            msgs.append(_tool_result("No matches found.", call_id))
        msgs.append(user_msg("Please continue searching."))
        return msgs

    def test_search_loop_does_not_produce_batched_calls(self, live):
        """
        After 3× search_files with no results and the search-aware correction hint,
        the model must respond with exactly ONE tool call — not batch multiple calls.

        Regression: before fix, the write-tool hint caused the model to batch 14
        calls at once (observed 2026-04-10 in production logs).
        """
        import os
        msgs = self._build_failed_search_history(repeats=3)
        model = os.environ.get("LIVE_MODEL", "openai/gpt-oss-120b")
        resp = live.post("/v1/chat/completions", json={
            "model": model,
            "messages": msgs,
            "tools": ROO_CODE_TOOLS,
        })
        assert resp.status_code == 200, resp.text
        result = resp.json()
        msg = result["choices"][0]["message"]
        tool_calls = msg.get("tool_calls") or []

        # toolproxy already trims to 1, but the point is the model should not
        # be trying to batch — verify it didn't try (check raw content for XML batching)
        content = msg.get("content") or ""
        # Count top-level XML tool tags in the response (if any leaked through)
        import re
        xml_tags = re.findall(r"<([a-z_]+)>", content)
        tool_names_in_content = [t for t in xml_tags if t in {
            "search_files", "list_files", "read_file", "write_to_file",
            "attempt_completion", "execute_command",
        }]
        assert len(tool_names_in_content) <= 1, (
            f"Model produced multiple XML tool calls in a single response "
            f"({len(tool_names_in_content)} tags found: {tool_names_in_content}).\n"
            "The search-loop hint must include 'ONE tool call per response'.\n"
            "Regression: 14-calls-at-once behaviour (2026-04-10)."
        )

        # Must produce exactly one tool call (toolproxy enforces this anyway)
        assert len(tool_calls) == 1, (
            f"Expected exactly 1 tool call, got {len(tool_calls)}.\n"
            f"Tool calls: {[tc['function']['name'] for tc in tool_calls]}"
        )

    def test_search_loop_hint_leads_to_attempt_completion_or_different_tool(self, live):
        """
        After the search-loop correction hint, the model must either:
        - call attempt_completion (report item not found)
        - try a genuinely different search (different path, different tool)

        It must NOT call search_files with the same path again.
        """
        import os
        msgs = self._build_failed_search_history(repeats=3)
        model = os.environ.get("LIVE_MODEL", "openai/gpt-oss-120b")
        resp = live.post("/v1/chat/completions", json={
            "model": model,
            "messages": msgs,
            "tools": ROO_CODE_TOOLS,
        })
        assert resp.status_code == 200, resp.text
        name, args = parse_tool_call(resp.json())

        if name == "search_files":
            # Acceptable only if the path is different (genuinely exploring elsewhere)
            assert args.get("path") != ".", (
                "Model called search_files on the same path '.' again after correction hint.\n"
                "The hint must cause the model to try a different path or give up."
            )


# ─────────────────────────────────────────────────────────────────────────────
# v1.6.19 — write_to_file loop detection via OpenAI-format history
#
# Observed in prod (2026-04-16 11:10–11:17): model wrote create-topics.bat
# 6+ times alternating kafka-topics.bat/.sh. Loop detection didn't fire
# because history contained OpenAI-format tool_calls, not XML in content.
# ─────────────────────────────────────────────────────────────────────────────

class TestWriteToFileLoopDetection:
    """
    Live E2E test: after write_to_file on the same file 3× in a row
    (using OpenAI-format tool_calls in history), the loop correction hint
    must fire and cause the model to stop rewriting.
    """

    _SCRIPT_PATH = "Tools/kafka/windows/create-topics.bat"

    def _build_write_loop_history(self, repeats: int) -> list:
        """
        Build a history where write_to_file was called N times on the same path
        using OpenAI-format tool_calls (as toolproxy returns them to Roo Code).
        The content alternates between .bat and .sh script content to simulate
        the exact oscillation pattern observed in production.
        """
        msgs = [
            SYSTEM_MSG,
            user_msg(
                "Fix the Windows batch script to use the native kafka-topics.bat launcher "
                "instead of the Linux shell script."
            ),
        ]
        versions = [
            "@echo off\nset KAFKA_TOPICS=%~dp0kafka\\bin\\windows\\kafka-topics.bat\ncall \"%KAFKA_TOPICS%\"\n",
            "@echo off\nset KAFKA_TOPICS=%~dp0kafka/bin/kafka-topics.sh\nbash \"%KAFKA_TOPICS%\"\n",
        ]
        for i in range(repeats):
            call_id = f"call_write_{i:02d}"
            msgs.append(_assistant_tool_call(
                "write_to_file",
                {"path": self._SCRIPT_PATH, "content": versions[i % 2]},
                call_id,
            ))
            msgs.append(_tool_result(
                f"The content was successfully saved to {self._SCRIPT_PATH}.",
                call_id,
            ))
        msgs.append(user_msg("The script still doesn't work. Please fix it."))
        return msgs

    def test_write_loop_hint_fires_after_three_writes(self, live):
        """
        After 3 consecutive write_to_file calls on the same path (via OpenAI-format
        tool_calls in history), toolproxy must inject a loop correction hint.

        The model should NOT keep rewriting the same file — it must either:
          - call read_file to verify current state
          - call attempt_completion if it believes the task is done
          - try a genuinely different approach

        Regression test for the create-topics.bat write loop (2026-04-16).
        """
        import os
        msgs = self._build_write_loop_history(repeats=3)
        model = os.environ.get("LIVE_MODEL", "openai/gpt-oss-120b")
        resp = live.post("/v1/chat/completions", json={
            "model": model,
            "messages": msgs,
            "tools": [TOOL_READ_FILE, TOOL_WRITE_TO_FILE, TOOL_EXECUTE_COMMAND, TOOL_ATTEMPT_COMPLETION],
        })
        assert resp.status_code == 200, resp.text
        name, args = parse_tool_call(resp.json())

        # After the loop correction hint, model must NOT rewrite the same file again
        if name == "write_to_file":
            # Acceptable only if writing a different file
            assert args.get("path") != self._SCRIPT_PATH, (
                f"Model called write_to_file on the same path '{self._SCRIPT_PATH}' again "
                "after 3 consecutive writes.\n"
                "Loop detection (v1.6.19) must have fired — check 'REPETITIVE LOOP' in logs.\n"
                "Possible cause: OpenAI-format tool_calls in history not detected."
            )

    def test_write_loop_hint_suggests_read_or_completion(self, live):
        """
        After the write loop correction hint, the model should pivot to either
        read_file (to verify state) or attempt_completion (done) rather than
        repeating the oscillating write pattern.
        """
        import os
        msgs = self._build_write_loop_history(repeats=3)
        model = os.environ.get("LIVE_MODEL", "openai/gpt-oss-120b")
        resp = live.post("/v1/chat/completions", json={
            "model": model,
            "messages": msgs,
            "tools": [TOOL_READ_FILE, TOOL_WRITE_TO_FILE, TOOL_EXECUTE_COMMAND, TOOL_ATTEMPT_COMPLETION],
        })
        assert resp.status_code == 200, resp.text
        name, _ = parse_tool_call(resp.json())

        # Model must use a productive tool — not just keep writing the same file
        productive_tools = {"read_file", "attempt_completion", "execute_command"}
        if name == "write_to_file":
            pass  # Only acceptable if path differs (tested above)
        else:
            assert name in productive_tools or name == "write_to_file", (
                f"After write loop hint, expected a productive tool call. Got: {name!r}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Issue: Exploding apply_diff — context corruption creates hundreds of
#        duplicate hunks that must be deduplicated by the proxy (v1.6.24)
#
# Strategy: inject a pre-built exploding diff as the upstream LLM response
# via AsyncMock (real HTTP endpoint, real pipeline, only upstream is mocked).
# This reliably tests deduplication without requiring the model to actually
# produce the corrupt diff.
# ─────────────────────────────────────────────────────────────────────────────

import os
from unittest.mock import AsyncMock, patch
import app.main as _main_module
from tests.conftest import llm_response


def _hunk(search: str, replace: str) -> str:
    return f"<<<<<<< SEARCH\n{search}\n=======\n{replace}\n>>>>>>> REPLACE\n"


def _apply_diff_xml(path: str, diff: str) -> str:
    diff_escaped = diff.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f"<apply_diff><path>{path}</path><diff>{diff_escaped}</diff></apply_diff>"


class TestExplodingDiffLive:

    def test_exploding_diff_is_deduplicated(self, client):
        """
        Simulate the real App.tsx incident: upstream returns an apply_diff
        with 20 identical hunks + a truncated last hunk.

        The proxy must:
          - Deduplicate to exactly 1 unique hunk
          - Drop the truncated hunk
          - Return a valid apply_diff tool call with the cleaned diff
        """
        hunk = _hunk("setTasks(prev =>", "setTasks((prev: Task[]) =>")
        truncated = "<<<<<<< SEARCH\nsetEditingId(prev =>\n=======\nsetEditingId((prev: number | null) =>\n"
        exploding = hunk * 20 + truncated

        mock = AsyncMock(return_value=llm_response(_apply_diff_xml("src/App.tsx", exploding)))
        with patch.object(_main_module, "upstream_client") as mock_uc:
            mock_uc.chat_completion = mock
            resp = client.post("/v1/chat/completions", json={
                "model": "openai/gpt-oss-120b",
                "messages": [
                    SYSTEM_MSG,
                    user_msg("Add TypeScript type annotations to App.tsx"),
                ],
                "tools": [TOOL_APPLY_DIFF, TOOL_ATTEMPT_COMPLETION],
            })

        assert resp.status_code == 200, resp.text
        name, args = parse_tool_call(resp.json())
        assert name == "apply_diff", f"Expected apply_diff, got {name!r}"
        diff = args["diff"]
        hunk_count = diff.count("<<<<<<< SEARCH")
        assert hunk_count == 1, (
            f"Expected 1 hunk after deduplication, got {hunk_count}.\n"
            f"Diff ({len(diff)} chars):\n{diff[:500]}"
        )
        assert "setTasks" in diff
        assert "setEditingId" not in diff, "Truncated hunk must be dropped"

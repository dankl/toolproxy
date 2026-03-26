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
from unittest.mock import patch
import httpx

import app.main as main_module
from app.services.vllm_client import VLLMClient
from tests.conftest import _tool, user_msg
from tests.live.conftest import (
    UPSTREAM_URL,
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

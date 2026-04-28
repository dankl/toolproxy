"""
Integration and roundtrip tests for the /v1/chat/completions endpoint.

Each test mocks the upstream LLM (upstream_client.chat_completion) and sends
real HTTP requests through the FastAPI app.  The "roundtrip" tests simulate
what Roo Code does across multiple turns:

  Turn N request  →  toolproxy  →  mock LLM returns XML
                  →  toolproxy parses XML  →  OpenAI tool_call response
  Turn N+1 request (with tool result in messages)  →  ...  repeat

Assertions cover:
  - Correct tool_call name and arguments returned
  - finish_reason == "tool_calls" when a tool call was found
  - History normalisation (role:tool → role:user, tool_calls → XML)
  - Priming injection on first turn
  - Aliased tool names (write_file → write_to_file)
  - Text synthesis (plain prose → attempt_completion or write_to_file)
  - Success-loop detection injects a CORRECTION stop-hint
"""
import json

import pytest

from tests.conftest import (
    DEFAULT_TOOLS,
    SYSTEM_MSG,
    llm_response,
    parse_tool_call,
    user_msg,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _post(client, messages, llm_content, tools=None):
    """Fire a single request with the given mock LLM content."""
    return client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": messages,
            "tools": tools if tools is not None else DEFAULT_TOOLS,
        },
    )


def _tool_result_msg(tool_call_id: str, content: str) -> dict:
    """Build the role:tool message Roo Code sends back after executing a tool."""
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}


def _assistant_tool_call_msg(tool_calls: list) -> dict:
    """Build the assistant message with tool_calls that Roo Code adds to history."""
    return {"role": "assistant", "content": None, "tool_calls": tool_calls}


# ──────────────────────────────────────────────────────────────────────────────
# Single-turn: basic XML parsing
# ──────────────────────────────────────────────────────────────────────────────

class TestSingleTurn:
    def test_simple_write_to_file(self, client, llm):
        llm.return_value = llm_response(
            "<write_to_file><path>foo.py</path><content>print('hi')</content></write_to_file>"
        )
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [SYSTEM_MSG, user_msg("Write foo.py")],
            "tools": DEFAULT_TOOLS,
        })
        assert resp.status_code == 200
        name, args = parse_tool_call(resp.json())
        assert name == "write_to_file"
        assert args["path"] == "foo.py"
        assert "print" in args["content"]
        assert resp.json()["choices"][0]["finish_reason"] == "tool_calls"
        # content must be empty when tool_calls are present (Roo Code contract)
        assert resp.json()["choices"][0]["message"]["content"] == ""

    def test_write_to_file_with_xml_in_content(self, client, llm):
        """
        Regression: model writes a README with <repo-url> in the shell snippet.
        The angle brackets inside <content> must not break XML parsing.
        Before the fix this silently fell through to attempt_completion.
        """
        xml = (
            "<write_to_file>"
            "<path>README.md</path>"
            "<content>"
            "# Project\n\n"
            "git clone <repo-url>\n"
            "curl <host>/api/hello\n\n"
            "```xml\n<dependency>\n  <groupId>org.springframework</groupId>\n</dependency>\n```"
            "</content>"
            "</write_to_file>"
        )
        llm.return_value = llm_response(xml)
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [SYSTEM_MSG, user_msg("Write README.md")],
            "tools": DEFAULT_TOOLS,
        })
        assert resp.status_code == 200
        name, args = parse_tool_call(resp.json())
        assert name == "write_to_file"
        assert args["path"] == "README.md"
        # Special chars must survive the round-trip intact
        assert "<repo-url>" in args["content"]
        assert "<host>" in args["content"]
        assert "<dependency>" in args["content"]

    def test_read_file(self, client, llm):
        llm.return_value = llm_response(
            "<read_file><path>src/main.py</path></read_file>"
        )
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [SYSTEM_MSG, user_msg("Show me main.py")],
            "tools": DEFAULT_TOOLS,
        })
        name, args = parse_tool_call(resp.json())
        assert name == "read_file"
        assert args["path"] == "src/main.py"

    def test_attempt_completion(self, client, llm):
        llm.return_value = llm_response(
            "<attempt_completion><result>Task finished.</result></attempt_completion>"
        )
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [SYSTEM_MSG, user_msg("Are you done?")],
            "tools": DEFAULT_TOOLS,
        })
        name, args = parse_tool_call(resp.json())
        assert name == "attempt_completion"
        assert "Task finished" in args["result"]

    def test_aliased_write_file_recovered(self, client, llm):
        """Hallucinated 'write_file' → aliased to 'write_to_file'."""
        llm.return_value = llm_response(
            "<write_file><path>x.py</path><content>pass</content></write_file>"
        )
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [SYSTEM_MSG, user_msg("Write x.py")],
            "tools": DEFAULT_TOOLS,
        })
        name, args = parse_tool_call(resp.json())
        assert name == "write_to_file"
        assert args["path"] == "x.py"

    def test_aliased_open_file_recovered(self, client, llm):
        """Hallucinated 'open_file' → aliased to 'read_file'."""
        llm.return_value = llm_response(
            "<open_file><path>app.py</path></open_file>"
        )
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [SYSTEM_MSG, user_msg("Open app.py")],
            "tools": DEFAULT_TOOLS,
        })
        name, args = parse_tool_call(resp.json())
        assert name == "read_file"
        assert args["path"] == "app.py"

    def test_preamble_text_before_xml(self, client, llm):
        """Model adds prose before the XML — tool call must still be extracted."""
        llm.return_value = llm_response(
            "Sure, I'll write that file now.\n\n"
            "<write_to_file><path>y.py</path><content>x = 1</content></write_to_file>"
        )
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [SYSTEM_MSG, user_msg("Write y.py")],
            "tools": DEFAULT_TOOLS,
        })
        name, args = parse_tool_call(resp.json())
        assert name == "write_to_file"


# ──────────────────────────────────────────────────────────────────────────────
# Text synthesis
# ──────────────────────────────────────────────────────────────────────────────

class TestTextSynthesis:
    def test_short_text_synthesized_as_attempt_completion(self, client, llm):
        """Short text with no XML → attempt_completion fallback."""
        llm.return_value = llm_response("Task done.")
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [SYSTEM_MSG, user_msg("What did you do?")],
            "tools": DEFAULT_TOOLS,
        })
        name, args = parse_tool_call(resp.json())
        assert name == "attempt_completion"
        assert "Task done" in args["result"]

    def test_long_markdown_with_open_tab_synthesized_as_write(self, client, llm):
        """
        Long markdown text + VSCode Open Tabs hint → write_to_file synthesis.
        The target file is extracted from the '# VSCode Open Tabs' section.
        """
        # Must be > 200 chars to trigger looks_like_file_content in synthesis.
        # Use CHANGELOG.md — priming uses README.md as its example, which would
        # falsely trigger _was_recently_written and fall through to attempt_completion.
        long_md = (
            "# Changelog\n\n"
            "All notable changes to this project will be documented here.\n\n"
            "## [Unreleased]\n\n"
            "### Added\n\n"
            "- Initial project setup\n"
            "- REST API with Spring Boot\n"
            "- H2 in-memory database\n\n"
            "## [1.0.0] - 2026-01-01\n\n"
            "### Changed\n\n"
            "- First stable release\n\n"
            "## License\n\nMIT"
        )
        llm.return_value = llm_response(long_md)
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [
                SYSTEM_MSG,
                # Roo Code appends a blank line after the tabs list — required
                # for the regex in _extract_target_file_from_context to match.
                user_msg("Write CHANGELOG.md\n\n# VSCode Open Tabs\nCHANGELOG.md\n\n"),
            ],
            "tools": DEFAULT_TOOLS,
        })
        name, args = parse_tool_call(resp.json())
        assert name == "write_to_file"
        assert args["path"] == "CHANGELOG.md"
        assert "Changelog" in args["content"]

    def test_xml_like_text_not_written_as_file(self, client, llm):
        """
        If the model response looks like an XML tool call (starts with <), it must
        NOT be synthesized as file content — that would write raw XML to disk.
        """
        llm.return_value = llm_response(
            "<unknown_tool><path>x</path></unknown_tool>"
        )
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [
                SYSTEM_MSG,
                user_msg("Do something\n\n# VSCode Open Tabs\nx.py"),
            ],
            "tools": DEFAULT_TOOLS,
        })
        name, _ = parse_tool_call(resp.json())
        # Must not write the raw XML as file content — should fall back to
        # attempt_completion (the XML itself becomes the result summary)
        assert name == "attempt_completion"


# ──────────────────────────────────────────────────────────────────────────────
# Priming injection
# ──────────────────────────────────────────────────────────────────────────────

class TestPriming:
    def test_priming_injected_on_first_turn(self, client, llm):
        """
        A single-user-message request must receive priming (synthetic Q&A pairs)
        before the real user message.  We capture what the LLM actually receives.
        """
        captured = []

        async def capture_and_respond(messages=None, **kwargs):
            captured.extend(messages or [])
            return llm_response(
                "<attempt_completion><result>done</result></attempt_completion>"
            )

        llm.side_effect = capture_and_respond

        client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [SYSTEM_MSG, user_msg("Are you done?")],
            "tools": DEFAULT_TOOLS,
        })

        non_system = [m for m in captured if m["role"] != "system"]
        # At least 2 priming messages (1 pair) + 1 real user message
        assert len(non_system) >= 3

        # Priming assistant messages must contain XML examples
        prime_assistants = [m for m in non_system[:-1] if m["role"] == "assistant"]
        assert len(prime_assistants) >= 1
        assert any("<" in (m.get("content") or "") for m in prime_assistants)

    def test_priming_injected_on_multi_turn(self, client, llm):
        """
        Priming is injected on ALL turns (not just the first) so the model
        reliably outputs XML even in Turn 2+ without a system prompt.
        """
        captured = []

        async def capture_and_respond(messages=None, **kwargs):
            captured.extend(messages or [])
            return llm_response(
                "<attempt_completion><result>done</result></attempt_completion>"
            )

        llm.side_effect = capture_and_respond

        client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [
                SYSTEM_MSG,
                user_msg("First turn"),
                {"role": "assistant", "content": "OK"},
                user_msg("Second turn"),
            ],
            "tools": DEFAULT_TOOLS,
        })

        non_system = [m for m in captured if m["role"] != "system"]
        user_messages = [m for m in non_system if m["role"] == "user"]
        # Priming adds user messages before the 2 original user messages
        assert len(user_messages) > 2


# ──────────────────────────────────────────────────────────────────────────────
# History normalisation (role:tool, tool_calls → XML)
# ──────────────────────────────────────────────────────────────────────────────

class TestHistoryNormalisation:
    def _capture_messages(self, llm, response_xml: str) -> list:
        captured = []

        async def side_effect(messages=None, **kwargs):
            captured.extend(messages or [])
            return llm_response(response_xml)

        llm.side_effect = side_effect
        return captured

    def test_role_tool_becomes_role_user_with_tag(self, client, llm):
        """
        A role:tool message in the request must arrive at the LLM as
        role:user with a '[Tool Result]' prefix.
        """
        captured = self._capture_messages(
            llm,
            "<attempt_completion><result>done</result></attempt_completion>",
        )

        client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [
                SYSTEM_MSG,
                user_msg("Write foo.py"),
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "write_to_file",
                            "arguments": '{"path":"foo.py","content":"pass"}',
                        },
                    }],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": "File written successfully: foo.py",
                },
            ],
            "tools": DEFAULT_TOOLS,
        })

        tool_result_msgs = [
            m for m in captured
            if m.get("role") == "user" and "[Tool Result]" in str(m.get("content", ""))
        ]
        # At least 1 tool result (priming may add additional [Tool Result] messages)
        assert len(tool_result_msgs) >= 1
        # The actual tool result must be present (last one is from the real conversation)
        assert any("File written successfully" in m["content"] for m in tool_result_msgs)

        # No raw role:tool messages must reach the LLM
        assert not any(m.get("role") == "tool" for m in captured)

    def test_tool_calls_in_assistant_message_become_xml(self, client, llm):
        """
        An assistant message with tool_calls must have those calls converted
        to XML inside the content field when sent to the LLM.
        """
        captured = self._capture_messages(
            llm,
            "<attempt_completion><result>done</result></attempt_completion>",
        )

        client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [
                SYSTEM_MSG,
                user_msg("Write foo.py"),
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "write_to_file",
                            "arguments": '{"path":"foo.py","content":"pass"}',
                        },
                    }],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "OK"},
            ],
            "tools": DEFAULT_TOOLS,
        })

        assistant_msgs = [m for m in captured if m.get("role") == "assistant"]
        xml_in_history = " ".join(m.get("content") or "" for m in assistant_msgs)
        assert "<write_to_file>" in xml_in_history

    def test_anthropic_content_array_flattened(self, client, llm):
        """
        Anthropic-style content arrays (tool_use / tool_result blocks inside
        the content list) must be flattened to a plain string.
        """
        captured = self._capture_messages(
            llm,
            "<attempt_completion><result>done</result></attempt_completion>",
        )

        client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [
                SYSTEM_MSG,
                user_msg("Go"),
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Let me read that."},
                        {"type": "tool_use", "id": "tu_1", "name": "read_file",
                         "input": {"path": "app.py"}},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "tu_1",
                         "content": [{"type": "text", "text": "def main(): pass"}]},
                    ],
                },
            ],
            "tools": DEFAULT_TOOLS,
        })

        # The assistant message must be a plain string containing the XML call
        assistant_msgs = [m for m in captured if m.get("role") == "assistant"]
        for m in assistant_msgs:
            assert isinstance(m.get("content"), (str, type(None)))

        # The tool_result must always appear with [Tool Result] prefix — regardless
        # of whether the client sent role:tool (OpenAI) or content:[tool_result] (Anthropic).
        user_content = " ".join(
            str(m.get("content", "")) for m in captured if m.get("role") == "user"
        )
        assert "[Tool Result]" in user_content


# ──────────────────────────────────────────────────────────────────────────────
# Full roundtrips (multi-turn)
# ──────────────────────────────────────────────────────────────────────────────

class TestRoundtrip:
    def test_write_then_complete_two_turns(self, client, llm):
        """
        Turn 1: model writes a file (README with XML in content — the bug).
        Turn 2: Roo Code sends the tool result; model calls attempt_completion.

        Key assertions:
          - Turn 1 returns correct write_to_file call with preserved content
          - Turn 2 upstream messages show the write_to_file as XML in history
          - Turn 2 returns correct attempt_completion call
        """
        # ── TURN 1 ────────────────────────────────────────────────────────
        write_xml = (
            "<write_to_file>"
            "<path>README.md</path>"
            "<content>"
            "# My Project\n\n"
            "git clone <repo-url>\n"
            "curl <localhost:8080>/api\n"
            "</content>"
            "</write_to_file>"
        )
        llm.return_value = llm_response(write_xml)

        t1_messages = [SYSTEM_MSG, user_msg("Write a README.md\n\n# VSCode Open Tabs\nREADME.md")]
        r1 = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": t1_messages,
            "tools": DEFAULT_TOOLS,
        })
        assert r1.status_code == 200
        d1 = r1.json()

        name1, args1 = parse_tool_call(d1)
        assert name1 == "write_to_file"
        assert args1["path"] == "README.md"
        assert "<repo-url>" in args1["content"]     # content preserved
        assert "<localhost:8080>" in args1["content"]

        tc1 = d1["choices"][0]["message"]["tool_calls"][0]

        # ── TURN 2 ────────────────────────────────────────────────────────
        t2_captured = []

        async def t2_side_effect(messages=None, **kwargs):
            t2_captured.extend(messages or [])
            return llm_response(
                "<attempt_completion>"
                "<result>README.md created successfully.</result>"
                "</attempt_completion>"
            )

        llm.side_effect = t2_side_effect

        t2_messages = [
            *t1_messages,
            _assistant_tool_call_msg(d1["choices"][0]["message"]["tool_calls"]),
            _tool_result_msg(tc1["id"], "File written successfully: README.md"),
        ]
        r2 = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": t2_messages,
            "tools": DEFAULT_TOOLS,
        })
        assert r2.status_code == 200
        name2, args2 = parse_tool_call(r2.json())
        assert name2 == "attempt_completion"
        assert "README.md" in args2["result"]

        # Turn 2 history: assistant tool_call must appear as XML
        assistant_msgs = [m for m in t2_captured if m.get("role") == "assistant"]
        xml_in_history = " ".join(m.get("content") or "" for m in assistant_msgs)
        assert "<write_to_file>" in xml_in_history

        # Turn 2 history: role:tool must be converted to role:user [Tool Result]
        tool_results = [
            m for m in t2_captured
            if m.get("role") == "user" and "[Tool Result]" in str(m.get("content", ""))
        ]
        assert len(tool_results) >= 1
        assert "File written successfully" in tool_results[-1]["content"]

        # No raw role:tool in what was sent to the LLM
        assert not any(m.get("role") == "tool" for m in t2_captured)

    def test_three_turn_read_write_complete(self, client, llm):
        """
        Turn 1: model reads a file.
        Turn 2: Roo Code returns file content; model writes an improved version.
        Turn 3: Roo Code confirms write; model calls attempt_completion.

        Verifies that all three turns normalise history correctly and that
        both tool calls appear as XML in Turn 3's upstream messages.
        """
        # ── TURN 1: read ──────────────────────────────────────────────────
        llm.return_value = llm_response(
            "<read_file><path>src/main.py</path></read_file>"
        )
        t1_msgs = [SYSTEM_MSG, user_msg("Improve main.py and save it.")]
        r1 = client.post("/v1/chat/completions", json={
            "model": "test-model", "messages": t1_msgs, "tools": DEFAULT_TOOLS,
        })
        tc1 = r1.json()["choices"][0]["message"]["tool_calls"][0]
        assert tc1["function"]["name"] == "read_file"

        # ── TURN 2: write ──────────────────────────────────────────────────
        t2_captured = []

        async def t2_side_effect(messages=None, **kwargs):
            t2_captured.extend(messages or [])
            return llm_response(
                "<write_to_file>"
                "<path>src/main.py</path>"
                "<content>def main():\n    print('improved')\n</content>"
                "</write_to_file>"
            )

        llm.side_effect = t2_side_effect

        t2_msgs = [
            *t1_msgs,
            _assistant_tool_call_msg(r1.json()["choices"][0]["message"]["tool_calls"]),
            _tool_result_msg(tc1["id"], "def main():\n    pass\n"),
        ]
        r2 = client.post("/v1/chat/completions", json={
            "model": "test-model", "messages": t2_msgs, "tools": DEFAULT_TOOLS,
        })
        tc2 = r2.json()["choices"][0]["message"]["tool_calls"][0]
        assert tc2["function"]["name"] == "write_to_file"

        # Turn 2 history must show the read_file call as XML
        t2_xml_history = " ".join(
            m.get("content") or "" for m in t2_captured if m.get("role") == "assistant"
        )
        assert "<read_file>" in t2_xml_history

        # ── TURN 3: complete ───────────────────────────────────────────────
        t3_captured = []

        async def t3_side_effect(messages=None, **kwargs):
            t3_captured.extend(messages or [])
            return llm_response(
                "<attempt_completion>"
                "<result>main.py has been improved.</result>"
                "</attempt_completion>"
            )

        llm.side_effect = t3_side_effect

        t3_msgs = [
            *t2_msgs,
            _assistant_tool_call_msg(r2.json()["choices"][0]["message"]["tool_calls"]),
            _tool_result_msg(tc2["id"], "File written successfully: src/main.py"),
        ]
        r3 = client.post("/v1/chat/completions", json={
            "model": "test-model", "messages": t3_msgs, "tools": DEFAULT_TOOLS,
        })
        name3, args3 = parse_tool_call(r3.json())
        assert name3 == "attempt_completion"
        assert "main.py" in args3["result"]

        # Turn 3 history must contain BOTH prior tool calls as XML
        t3_xml_history = " ".join(
            m.get("content") or "" for m in t3_captured if m.get("role") == "assistant"
        )
        assert "<read_file>" in t3_xml_history
        assert "<write_to_file>" in t3_xml_history


# ──────────────────────────────────────────────────────────────────────────────
# Success-loop detection
# ──────────────────────────────────────────────────────────────────────────────

class TestSuccessLoopDetection:
    def test_stop_hint_injected_after_two_successful_writes(self, client, llm):
        """
        Model is looping: 2 successful write tool results with NO genuine user
        instruction in between. The stop hint must be injected.

        The history ends with tool results — no new user message — which is exactly
        how Roo Code sends requests when the model loops without user intervention.
        """
        captured = []

        async def side_effect(messages=None, **kwargs):
            captured.extend(messages or [])
            return llm_response(
                "<attempt_completion><result>Done.</result></attempt_completion>"
            )

        llm.side_effect = side_effect

        # History: user starts task, model writes 2 files, history ends with
        # the second tool result (no follow-up user message = looping model)
        messages = [
            SYSTEM_MSG,
            user_msg("Write three files."),
            _assistant_tool_call_msg([{
                "id": "call_1", "type": "function",
                "function": {"name": "write_to_file",
                             "arguments": '{"path":"a.py","content":"x"}'},
            }]),
            _tool_result_msg("call_1", "File written successfully: a.py"),
            _assistant_tool_call_msg([{
                "id": "call_2", "type": "function",
                "function": {"name": "write_to_file",
                             "arguments": '{"path":"b.py","content":"y"}'},
            }]),
            _tool_result_msg("call_2", "File written successfully: b.py"),
            # No genuine user message here — the model is trying to call a tool again
        ]

        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": messages,
            "tools": DEFAULT_TOOLS,
        })
        assert resp.status_code == 200

        # The last user message sent to the LLM must contain the stop hint
        last_user = next(
            (m for m in reversed(captured) if m.get("role") == "user"), None
        )
        assert last_user is not None
        content = last_user["content"]
        assert "[CORRECTION]" in content or "STOP" in content

    def test_stop_hint_reset_by_new_user_instruction(self, client, llm):
        """
        Regression: a genuine user instruction after two successful writes must
        reset the loop counter. The stop hint must NOT fire, so the model can
        make the requested edit.

        This was the original bug: the user asked "Add the section" after a
        completed task and the proxy injected a stop hint, causing the model to
        report success without actually editing the file.
        """
        captured = []

        async def side_effect(messages=None, **kwargs):
            captured.extend(messages or [])
            return llm_response(
                "<write_to_file><path>guide.md</path><content>## New section\n\nContent here.</content></write_to_file>"
            )

        llm.side_effect = side_effect

        # History: task A finished with 2 writes, then the user requests a NEW edit
        messages = [
            SYSTEM_MSG,
            user_msg("Write two files."),
            _assistant_tool_call_msg([{
                "id": "call_1", "type": "function",
                "function": {"name": "write_to_file",
                             "arguments": '{"path":"a.py","content":"x"}'},
            }]),
            _tool_result_msg("call_1", "File written successfully: a.py"),
            _assistant_tool_call_msg([{
                "id": "call_2", "type": "function",
                "function": {"name": "write_to_file",
                             "arguments": '{"path":"b.py","content":"y"}'},
            }]),
            _tool_result_msg("call_2", "File written successfully: b.py"),
            # Genuine new user instruction — resets the loop counter
            user_msg("The file has no changes. Add the section."),
        ]

        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": messages,
            "tools": DEFAULT_TOOLS,
        })
        assert resp.status_code == 200

        last_user = next(
            (m for m in reversed(captured) if m.get("role") == "user"), None
        )
        assert last_user is not None
        # Stop hint must NOT be present — user gave a new instruction
        content = last_user["content"]
        assert "[CORRECTION]" not in content
        assert "STOP" not in content

        # Model should have been allowed to write
        data = resp.json()
        tool_calls = data["choices"][0]["message"]["tool_calls"]
        assert tool_calls and tool_calls[0]["function"]["name"] == "write_to_file"

    def test_no_stop_hint_with_only_one_successful_write(self, client, llm):

        """One successful write is normal — no loop, no stop hint."""
        captured = []

        async def side_effect(messages=None, **kwargs):
            captured.extend(messages or [])
            return llm_response(
                "<write_to_file><path>c.py</path><content>z</content></write_to_file>"
            )

        llm.side_effect = side_effect

        messages = [
            SYSTEM_MSG,
            user_msg("Write two files."),
            _assistant_tool_call_msg([{
                "id": "call_1", "type": "function",
                "function": {"name": "write_to_file",
                             "arguments": '{"path":"a.py","content":"x"}'},
            }]),
            _tool_result_msg("call_1", "File written successfully: a.py"),
            user_msg("Now write c.py."),
        ]

        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": messages,
            "tools": DEFAULT_TOOLS,
        })
        assert resp.status_code == 200

        last_user = next(
            (m for m in reversed(captured) if m.get("role") == "user"), None
        )
        # No stop hint expected
        assert "[CORRECTION]" not in (last_user["content"] or "")
        assert "STOP" not in (last_user["content"] or "")


# ──────────────────────────────────────────────────────────────────────────────
# Partial XML rescue (truncated response)
# ──────────────────────────────────────────────────────────────────────────────

class TestPartialXmlRescue:
    """
    When extract_xml_tool_calls finds no match (truncated response or lazy-regex
    stop caused by </write_to_file> inside <content>), the synthesis step must
    rescue the partial XML and return write_to_file instead of attempt_completion.
    """

    def test_truncated_write_to_file_rescued(self, client, llm):
        """Response cut off before </write_to_file> — must be rescued."""
        # Simulate a truncated response: no </content> and no </write_to_file>
        truncated = (
            "<write_to_file><path>Plan.md</path>"
            "<content># Plan\n\nStep 1: do this\nStep 2: do that\n"
            "npm install && pip install flask"
            # ← no </content> or </write_to_file>
        )
        llm.return_value = llm_response(truncated)
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [SYSTEM_MSG, user_msg("Write Plan.md")],
            "tools": DEFAULT_TOOLS,
        })
        assert resp.status_code == 200
        name, args = parse_tool_call(resp.json())
        assert name == "write_to_file", f"Expected write_to_file, got '{name}'"
        assert args["path"] == "Plan.md"
        assert "Step 1" in args["content"]

    def test_write_to_file_alias_in_partial_xml(self, client, llm):
        """Alias write_file (not write_to_file) also rescued."""
        truncated = (
            "<write_file><path>Notes.md</path>"
            "<content>Some notes here without closing tags"
        )
        llm.return_value = llm_response(truncated)
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [SYSTEM_MSG, user_msg("Write Notes.md")],
            "tools": DEFAULT_TOOLS,
        })
        assert resp.status_code == 200
        name, args = parse_tool_call(resp.json())
        assert name == "write_to_file"
        assert args["path"] == "Notes.md"


# ──────────────────────────────────────────────────────────────────────────────
# Roo Code built-in tools not always in tools[] (delete_file, rename_file, …)
# ──────────────────────────────────────────────────────────────────────────────

class TestRooCodeBuiltinTools:
    """Built-in Roo tools that may be absent from tools[] but should still be parsed."""

    _TOOLS_WITHOUT_DELETE = [t for t in DEFAULT_TOOLS if t["function"]["name"] != "delete_file"]

    def test_delete_file_not_in_tools_array_still_parsed(self, client, llm):
        """Model outputs delete_file even though it's not in the request tools[] — must pass through."""
        llm.return_value = llm_response(
            "<delete_file><path>test/witze.md</path></delete_file>"
        )
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [SYSTEM_MSG, user_msg("rename folder test to witzeordner")],
            "tools": self._TOOLS_WITHOUT_DELETE,
        })
        assert resp.status_code == 200
        name, args = parse_tool_call(resp.json())
        assert name == "delete_file", f"Expected delete_file, got '{name}'"
        assert args["path"] == "test/witze.md"

    def test_rename_file_not_in_tools_array_still_parsed(self, client, llm):
        """rename_file is a Roo built-in — should be parsed even when not in tools[]."""
        llm.return_value = llm_response(
            "<rename_file><path>old.md</path><newPath>new.md</newPath></rename_file>"
        )
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [SYSTEM_MSG, user_msg("rename old.md to new.md")],
            "tools": DEFAULT_TOOLS,  # rename_file not in DEFAULT_TOOLS
        })
        assert resp.status_code == 200
        name, args = parse_tool_call(resp.json())
        assert name == "rename_file"
        assert args["path"] == "old.md"


# ──────────────────────────────────────────────────────────────────────────────
# Rescue: XML tool calls inside attempt_completion.result
# ──────────────────────────────────────────────────────────────────────────────

class TestAttemptCompletionXmlRescue:
    """Model placed XML tool calls inside attempt_completion.result instead of
    outputting them directly — proxy must extract and return the first real call."""

    def test_delete_file_rescued_from_result(self, client, llm):
        """Single delete_file wrapped in attempt_completion.result → rescued."""
        llm.return_value = llm_response(
            "<attempt_completion>"
            "<result>"
            "<delete_file><path>ai-guide/README.md</path></delete_file>"
            "</result>"
            "</attempt_completion>"
        )
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [SYSTEM_MSG, user_msg("Delete ai-guide/README.md")],
            "tools": DEFAULT_TOOLS,
        })
        assert resp.status_code == 200
        name, args = parse_tool_call(resp.json())
        assert name == "delete_file", f"Expected delete_file, got '{name}'"
        assert args["path"] == "ai-guide/README.md"

    def test_multiple_xml_in_result_returns_first(self, client, llm):
        """Multiple XML calls in result → only first is returned (LIMIT applies after rescue)."""
        llm.return_value = llm_response(
            "<attempt_completion>"
            "<result>"
            "<delete_file><path>ai-guide/README.md</path></delete_file>"
            "<delete_file><path>ai-guide/n8n-workflows/.gitkeep</path></delete_file>"
            "</result>"
            "</attempt_completion>"
        )
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [SYSTEM_MSG, user_msg("Clean up ai-guide")],
            "tools": DEFAULT_TOOLS,
        })
        assert resp.status_code == 200
        name, args = parse_tool_call(resp.json())
        assert name == "delete_file"
        assert args["path"] == "ai-guide/README.md"

    def test_plain_text_result_not_rescued(self, client, llm):
        """Normal attempt_completion with plain text result must pass through unchanged."""
        llm.return_value = llm_response(
            "<attempt_completion>"
            "<result>All files have been written successfully.</result>"
            "</attempt_completion>"
        )
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [SYSTEM_MSG, user_msg("Are you done?")],
            "tools": DEFAULT_TOOLS,
        })
        assert resp.status_code == 200
        name, args = parse_tool_call(resp.json())
        assert name == "attempt_completion"
        assert "successfully" in args["result"]

    def test_write_to_file_rescued_from_result(self, client, llm):
        """write_to_file inside attempt_completion.result → rescued."""
        llm.return_value = llm_response(
            "<attempt_completion>"
            "<result>"
            "<write_to_file><path>output.md</path><content># Done</content></write_to_file>"
            "</result>"
            "</attempt_completion>"
        )
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [SYSTEM_MSG, user_msg("Write output.md")],
            "tools": DEFAULT_TOOLS,
        })
        assert resp.status_code == 200
        name, args = parse_tool_call(resp.json())
        assert name == "write_to_file"
        assert args["path"] == "output.md"
        assert "Done" in args["content"]


# ──────────────────────────────────────────────────────────────────────────────
# Regression: Todo.md Joke API plan (& and => in content)
# ──────────────────────────────────────────────────────────────────────────────

# Exact content from the real Roo Code session where the model wrote a Joke API
# plan.  The critical characters are:
#   && from shell: "npm init && pip install flask"
#   &  standalone: "Linter & Formatter"
#   >  from arrow function: "(req, res) => { … }"
# Without fix_xml_string these cause ET.ParseError → fallthrough to
# attempt_completion → the raw XML appeared in the Roo Code "Task Completed"
# banner instead of creating the file.
_JOKE_API_PLAN = (
    "# Todo.md\n\n"
    "Projekt: Simple Joke API\n"
    "Ziel: Eine kleine Web‑API, die über GET /getWitz einen Witz zurückliefert.\n\n"
    "1. Projekt‑Setup\n"
    "npm init -y (Node) / python -m venv venv && pip install flask (Python)\n"
    "Linter & Formatter einrichten (ESLint / Prettier oder flake8/black)\n\n"
    "2. Statische Witze‑Quelle\n"
    "Datei jokes.json anlegen:\n"
    '["Warum können Geister so schlecht lügen? Weil man durch sie hindurchsehen kann!"]\n\n'
    "3. API‑Implementierung\n"
    "Route definieren:\n"
    "  Node: app.get('/getWitz', (req, res) => { res.json({ witz: joke }) })\n"
    "  Python: @app.route('/getWitz') def get_witz(): return jsonify({'witz': joke})\n\n"
    "4. Tests\n"
    "Unit‑Tests für die Zufallsauswahl (pytest / Jest)\n\n"
    "Beispiel‑Projektstruktur:\n"
    "/joke-api\n"
    "│\n"
    "├─ jokes.json\n"
    "├─ index.js\n"
    "└─ README.md\n"
)


class TestJokeApiScenario:
    """
    Reproduces the exact 2-turn Roo Code session that triggered the bug:

      Turn 1: "Schreibe einen Witz in Todo.md"
              → model outputs write_to_file with simple content → OK (file created)

      Turn 2: "Erstelle mir einen Plan für eine Witze-API in Todo.md"
              → model outputs write_to_file whose <content> contains:
                  - && (shell command chaining)
                  - & (standalone, e.g. "Linter & Formatter")
                  - > (from JavaScript arrow function "=>")
              → WITHOUT fix: ET.ParseError → fallthrough to attempt_completion
                             → raw XML appeared in "Task Completed" banner
              → WITH fix:    fix_xml_string escapes the special chars → file created
    """

    def test_turn1_joke_written_to_todo_md(self, client, llm):
        """Turn 1: simple write_to_file with clean content must succeed."""
        llm.return_value = llm_response(
            "<write_to_file><path>Todo.md</path>"
            "<content># Todo.md\n\n**Witz des Tages**\n\n"
            "> Warum können Geister so schlecht lügen?  \n"
            "> Weil man durch sie hindurchsehen kann! 😄</content>"
            "</write_to_file>"
        )
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [SYSTEM_MSG, user_msg("Schreibe einen Witz in Todo.md")],
            "tools": DEFAULT_TOOLS,
        })
        assert resp.status_code == 200
        name, args = parse_tool_call(resp.json())
        assert name == "write_to_file", f"Expected write_to_file, got '{name}'"
        assert args["path"] == "Todo.md"
        assert "Geister" in args["content"]

    def test_turn2_plan_with_special_chars(self, client, llm):
        """
        Turn 2: write_to_file whose content has && and => must NOT fall through
        to attempt_completion.
        """
        plan_xml = (
            f"<write_to_file><path>Todo.md</path>"
            f"<content>{_JOKE_API_PLAN}</content>"
            f"</write_to_file>"
        )
        llm.return_value = llm_response(plan_xml)
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [
                SYSTEM_MSG,
                user_msg(
                    "Erstelle mir in @/Todo.md einen Plan für eine Witze-API.\n\n"
                    "# VSCode Open Tabs\nTodo.md\n\n"
                ),
            ],
            "tools": DEFAULT_TOOLS,
        })
        assert resp.status_code == 200
        name, args = parse_tool_call(resp.json())
        assert name == "write_to_file", (
            f"Expected write_to_file but got '{name}' — "
            "XML with & and => likely caused ET.ParseError → attempt_completion fallthrough"
        )
        assert args["path"] == "Todo.md"
        assert "getWitz" in args["content"]
        assert "&&" in args["content"]   # shell && must survive round-trip
        assert "=>" in args["content"]   # arrow function must survive round-trip

    def test_full_two_turn_joke_then_plan(self, client, llm):
        """
        Full 2-turn reproduction:
          Turn 1 → write joke to Todo.md (clean content)
          Turn 2 → write API plan to Todo.md (content with && and =>)

        Verifies both turns complete as write_to_file, not attempt_completion.
        """
        # ── TURN 1: write joke ────────────────────────────────────────────
        joke_xml = (
            "<write_to_file><path>Todo.md</path>"
            "<content># Todo.md\n\n"
            "Warum können Geister so schlecht lügen?\n"
            "Weil man durch sie hindurchsehen kann!</content>"
            "</write_to_file>"
        )
        llm.return_value = llm_response(joke_xml)
        t1_msgs = [SYSTEM_MSG, user_msg("Schreibe einen Witz in Todo.md")]
        r1 = client.post("/v1/chat/completions", json={
            "model": "test-model", "messages": t1_msgs, "tools": DEFAULT_TOOLS,
        })
        assert r1.status_code == 200
        name1, _ = parse_tool_call(r1.json())
        assert name1 == "write_to_file", f"Turn 1: expected write_to_file, got '{name1}'"
        tc1 = r1.json()["choices"][0]["message"]["tool_calls"][0]

        # ── TURN 2: write plan (the bug) ──────────────────────────────────
        plan_xml = (
            f"<write_to_file><path>Todo.md</path>"
            f"<content>{_JOKE_API_PLAN}</content>"
            f"</write_to_file>"
        )
        llm.return_value = llm_response(plan_xml)
        t2_msgs = [
            *t1_msgs,
            _assistant_tool_call_msg(r1.json()["choices"][0]["message"]["tool_calls"]),
            _tool_result_msg(tc1["id"], "File written successfully: Todo.md"),
            user_msg(
                "Ich möchte eigentlich eine App haben, die mir Witze via einer einfachen API "
                "anbietet. Erstelle mir in @/Todo.md einen Plan, wie die App zu bauen ist.\n\n"
                "# VSCode Open Tabs\nTodo.md\n\n"
            ),
        ]
        r2 = client.post("/v1/chat/completions", json={
            "model": "test-model", "messages": t2_msgs, "tools": DEFAULT_TOOLS,
        })
        assert r2.status_code == 200
        name2, args2 = parse_tool_call(r2.json())
        assert name2 == "write_to_file", (
            f"Turn 2: expected write_to_file but got '{name2}' — "
            "content with && and => caused XML parse failure"
        )
        assert args2["path"] == "Todo.md"
        assert "getWitz" in args2["content"]
        assert "&&" in args2["content"]
        assert "=>" in args2["content"]


# ──────────────────────────────────────────────────────────────────────────────
# Regression: prose + code-block response (model skips XML and describes file)
# ──────────────────────────────────────────────────────────────────────────────

# Reproduces the exact second-turn failure from [67551cb0]:
#   User asks to add a section to guides/ai-guide.md.
#   Model answers with German prose + a ```markdown code block instead of XML.
#   Without VSCode Open Tabs the proxy cannot determine the target file and
#   falls back to attempt_completion — the file is never updated.
#
# These tests document the current proxy-level behaviour so we catch regressions
# if the synthesis logic changes.  The REAL fix is the system-prompt improvement
# that makes the model less likely to produce this format in the first place.

_PROSE_CODEBLOCK_RESPONSE = (
    "Hier ist der aktualisierte **guides/ai-guide.md** mit einem kurzen "
    "Abschnitt zu RagFlow am Ende (maximal 5 Sätze):\n\n"
    "```markdown\n"
    "# AI Guide — AgentGarage\n\n"
    "Eine kurze Übersicht der verfügbaren KI-Dienste und wie du sie nutzt.\n\n"
    "---\n\n"
    "## Das Modell\n\n"
    "AgentGarage betreibt ein selbst-gehostetes Large Language Model.\n\n"
    "## RagFlow\n\n"
    "RagFlow ist eine Open-Source RAG-Engine für Dokumentensuche.\n"
    "```"
)


class TestProseCodeBlockResponse:
    """
    Model ignores XML format and instead describes the file content in prose
    with a markdown code block — e.g. 'Hier ist der aktualisierte file.md: ```...```'.
    """

    def test_no_tabs_falls_back_to_attempt_completion(self, client, llm):
        """
        Without a VSCode Open Tabs hint the proxy cannot determine the target
        file → falls back to attempt_completion instead of write_to_file.

        This is the exact failure from log [67551cb0].  The test documents the
        current limitation: system-prompt improvements reduce the frequency but
        the proxy itself has no code-level fix for this case yet.
        """
        llm.return_value = llm_response(_PROSE_CODEBLOCK_RESPONSE)
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [
                SYSTEM_MSG,
                # No "# VSCode Open Tabs" section → _extract_target_file returns None
                user_msg("Füge einen kurzen Abschnitt zu RagFlow in guides/ai-guide.md ein."),
            ],
            "tools": DEFAULT_TOOLS,
        })
        assert resp.status_code == 200
        name, _ = parse_tool_call(resp.json())
        # Current behaviour: falls through to attempt_completion because
        # target file cannot be determined from context alone.
        assert name == "attempt_completion", (
            f"Expected attempt_completion (known limitation), got '{name}'. "
            "If this is now write_to_file, update the test — the proxy was improved!"
        )

    def test_with_tabs_synthesizes_write_to_file(self, client, llm):
        """
        With a VSCode Open Tabs hint pointing at guides/ai-guide.md the proxy
        CAN determine the target file and synthesizes write_to_file.

        Note: the synthesized content includes the prose preamble — not ideal,
        but better than attempt_completion (file is at least written).
        """
        llm.return_value = llm_response(_PROSE_CODEBLOCK_RESPONSE)
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [
                SYSTEM_MSG,
                user_msg(
                    "Füge einen kurzen Abschnitt zu RagFlow in guides/ai-guide.md ein.\n\n"
                    "# VSCode Open Tabs\nguides/ai-guide.md\n\n"
                ),
            ],
            "tools": DEFAULT_TOOLS,
        })
        assert resp.status_code == 200
        name, args = parse_tool_call(resp.json())
        assert name == "write_to_file", (
            f"Expected write_to_file when Open Tabs contains the target, got '{name}'"
        )
        assert args["path"] == "guides/ai-guide.md"
        assert "RagFlow" in args["content"]

    def test_multi_turn_prose_codeblock_in_second_turn(self, client, llm):
        """
        Full two-turn reproduction of [d30af879] + [67551cb0]:

          Turn 1: model correctly outputs <write_to_file> XML → file created.
          Turn 2: model ignores XML format, returns prose + code block.
                  No VSCode tabs in the second user message → attempt_completion.

        This is the exact session that triggered the system-prompt improvement.
        """
        # ── TURN 1: correct XML ────────────────────────────────────────────
        llm.return_value = llm_response(
            "<write_to_file>"
            "<path>guides/ai-guide.md</path>"
            "<content># AI Guide — AgentGarage\n\nEine kurze Übersicht.</content>"
            "</write_to_file>"
        )
        t1_msgs = [SYSTEM_MSG, user_msg("Erstelle guides/ai-guide.md")]
        r1 = client.post("/v1/chat/completions", json={
            "model": "test-model", "messages": t1_msgs, "tools": DEFAULT_TOOLS,
        })
        assert r1.status_code == 200
        name1, _ = parse_tool_call(r1.json())
        assert name1 == "write_to_file"
        tc1 = r1.json()["choices"][0]["message"]["tool_calls"][0]

        # ── TURN 2: model returns prose + code block instead of XML ────────
        llm.return_value = llm_response(_PROSE_CODEBLOCK_RESPONSE)
        t2_msgs = [
            *t1_msgs,
            _assistant_tool_call_msg(r1.json()["choices"][0]["message"]["tool_calls"]),
            _tool_result_msg(tc1["id"], "File written successfully: guides/ai-guide.md"),
            # Note: no VSCode Open Tabs hint → proxy cannot find target file
            user_msg("Füge einen kurzen RagFlow-Abschnitt am Ende hinzu."),
        ]
        r2 = client.post("/v1/chat/completions", json={
            "model": "test-model", "messages": t2_msgs, "tools": DEFAULT_TOOLS,
        })
        assert r2.status_code == 200
        name2, _ = parse_tool_call(r2.json())
        # Proxy falls back to attempt_completion — file is NOT updated.
        # The system-prompt improvement makes this scenario less likely to occur.
        assert name2 == "attempt_completion", (
            f"Turn 2: expected attempt_completion (known limitation), got '{name2}'"
        )


# ──────────────────────────────────────────────────────────────────────────────
# SSE streaming (stream=True)
# ──────────────────────────────────────────────────────────────────────────────


class TestStreaming:
    """When the client sends stream=True, toolproxy must return SSE chunks."""

    def _parse_sse(self, text: str) -> list:
        """Parse raw SSE text into a list of parsed data dicts."""
        events = []
        for line in text.splitlines():
            if line.startswith("data: ") and line != "data: [DONE]":
                events.append(json.loads(line[6:]))
        return events

    def test_stream_tool_call(self, client, llm):
        """stream=True with XML tool call → SSE chunks with tool_calls delta."""
        llm.return_value = llm_response(
            "<read_file><path>README.md</path></read_file>"
        )
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [SYSTEM_MSG, user_msg("Read README.md")],
            "tools": DEFAULT_TOOLS,
            "stream": True,
        })
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

        events = self._parse_sse(resp.text)
        assert len(events) >= 3

        # First chunk: role
        assert events[0]["choices"][0]["delta"].get("role") == "assistant"

        # Find the chunk with tool call name
        name_chunk = next(
            (e for e in events if e["choices"][0]["delta"].get("tool_calls")),
            None,
        )
        assert name_chunk is not None
        tc_delta = name_chunk["choices"][0]["delta"]["tool_calls"][0]
        assert tc_delta["function"]["name"] == "read_file"

        # Last data chunk: finish_reason == tool_calls
        finish_chunk = next(
            (e for e in reversed(events) if e["choices"][0].get("finish_reason")),
            None,
        )
        assert finish_chunk is not None
        assert finish_chunk["choices"][0]["finish_reason"] == "tool_calls"

        # Response must end with [DONE]
        assert resp.text.strip().endswith("data: [DONE]")

    def test_stream_text_response(self, client, llm):
        """stream=True with plain text response → SSE chunks with content delta."""
        llm.return_value = llm_response("Hello, world!")
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [user_msg("Say hi")],
            "tools": [],
            "stream": True,
        })
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

        events = self._parse_sse(resp.text)
        content = "".join(
            e["choices"][0]["delta"].get("content", "")
            for e in events
            if e["choices"][0]["delta"].get("content")
        )
        assert "Hello" in content

        finish_chunk = next(
            (e for e in reversed(events) if e["choices"][0].get("finish_reason")),
            None,
        )
        assert finish_chunk["choices"][0]["finish_reason"] == "stop"
        assert resp.text.strip().endswith("data: [DONE]")

    def test_non_stream_still_works(self, client, llm):
        """stream=False (or omitted) must still return a regular JSON response."""
        llm.return_value = llm_response(
            "<read_file><path>README.md</path></read_file>"
        )
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [SYSTEM_MSG, user_msg("Read README.md")],
            "tools": DEFAULT_TOOLS,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"
        name, args = parse_tool_call(data)
        assert name == "read_file"


# ──────────────────────────────────────────────────────────────────────────────
# Helpers for Anthropic-format messages (Roo Code sends this format)
# ──────────────────────────────────────────────────────────────────────────────

def _roo_tool_result_msg(tool_use_id: str, content: str) -> dict:
    """
    Build the role:user message that Roo Code sends after tool execution.

    Roo Code uses the Anthropic API format internally: tool results arrive as
    content:[{type:"tool_result"}] inside a role:user message, NOT as role:tool.
    """
    return {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": tool_use_id, "content": content},
        ],
    }


def _roo_assistant_tool_use_msg(tool_use_id: str, name: str, input_: dict) -> dict:
    """Build the role:assistant message with a tool_use block (Anthropic format)."""
    return {
        "role": "assistant",
        "content": [
            {"type": "tool_use", "id": tool_use_id, "name": name, "input": input_},
        ],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Emergency fallback: empty upstream response
# ──────────────────────────────────────────────────────────────────────────────

class TestEmptyModelResponseFallback:
    """
    When the upstream LLM returns empty content (e.g. OCI silently returns null),
    toolproxy must synthesize an attempt_completion rather than returning an empty
    assistant message that clients like Roo Code reject as "no assistant messages".
    """

    def test_empty_content_synthesizes_attempt_completion(self, client, llm):
        """Empty upstream response → attempt_completion fallback."""
        llm.return_value = llm_response("")  # upstream returns empty string

        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [SYSTEM_MSG, user_msg("Do something.")],
            "tools": DEFAULT_TOOLS,
        })
        assert resp.status_code == 200
        data = resp.json()
        name, args = parse_tool_call(data)
        assert name == "attempt_completion"
        assert "result" in args

    def test_empty_content_finish_reason_is_tool_calls(self, client, llm):
        """finish_reason must be tool_calls when fallback fires."""
        llm.return_value = llm_response("")

        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [SYSTEM_MSG, user_msg("Do something.")],
            "tools": DEFAULT_TOOLS,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["finish_reason"] == "tool_calls"

    def test_empty_content_no_attempt_completion_tool_returns_text(self, client, llm):
        """If attempt_completion is not in the tools list, return empty text (no crash)."""
        tools_without_completion = [t for t in DEFAULT_TOOLS if t["function"]["name"] != "attempt_completion"]
        llm.return_value = llm_response("")

        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [SYSTEM_MSG, user_msg("Do something.")],
            "tools": tools_without_completion,
        })
        assert resp.status_code == 200
        # No crash — just returns whatever (empty text response is acceptable here)
        data = resp.json()
        assert "choices" in data


# ──────────────────────────────────────────────────────────────────────────────
# Loop detection with Roo Code Anthropic format
# ──────────────────────────────────────────────────────────────────────────────

class TestLoopDetectionRooCodeFormat:
    """
    Loop detection must work when Roo Code sends tool results in Anthropic format
    (role:user + content:[{type:"tool_result"}]) instead of OpenAI role:tool.

    Before the fix: the normalizer did not add [Tool Result] prefix to Anthropic
    tool_result blocks, so loop_detection treated every tool result as a genuine
    user instruction and always reset the counter → detection never fired.
    """

    def test_loop_detected_with_anthropic_format_tool_results(self, client, llm):
        """
        Two consecutive Roo Code-format tool results containing 'created' must
        trigger the loop stop-hint, just like role:tool results do.
        """
        captured = []

        async def side_effect(messages=None, **kwargs):
            captured.extend(messages or [])
            return llm_response(
                "<attempt_completion><result>Done.</result></attempt_completion>"
            )

        llm.side_effect = side_effect

        messages = [
            SYSTEM_MSG,
            user_msg("Create two files."),
            _roo_assistant_tool_use_msg("tu_1", "write_to_file",
                                        {"path": "a.py", "content": "x"}),
            _roo_tool_result_msg("tu_1", '{"path":"a.py","operation":"created"}'),
            _roo_assistant_tool_use_msg("tu_2", "write_to_file",
                                        {"path": "b.py", "content": "y"}),
            _roo_tool_result_msg("tu_2", '{"path":"b.py","operation":"created"}'),
        ]

        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": messages,
            "tools": DEFAULT_TOOLS,
        })
        assert resp.status_code == 200

        last_user = next(
            (m for m in reversed(captured) if m.get("role") == "user"), None
        )
        assert last_user is not None
        content = last_user["content"]
        assert "[CORRECTION]" in content or "STOP" in content

    def test_loop_resets_on_user_message_in_attempt_completion_result(self, client, llm):
        """
        Reproduces the exact bug from the diagnostics:
          1. write_to_file(test/Witze.md) → created
          2. attempt_completion → user replies 'rename test to witzeordner'
          3. write_to_file(witzeordner/Witze.md) → created

        After step 3 the model needs to respond (attempt_completion).
        Loop detection must NOT fire (false positive) because the two writes
        belong to two different user instructions separated by a new task boundary.
        """
        captured = []

        async def side_effect(messages=None, **kwargs):
            captured.extend(messages or [])
            return llm_response(
                "<attempt_completion><result>Renamed.</result></attempt_completion>"
            )

        llm.side_effect = side_effect

        messages = [
            SYSTEM_MSG,
            # First instruction: create test/Witze.md
            user_msg("Create test/Witze.md."),
            _roo_assistant_tool_use_msg("tu_1", "write_to_file",
                                        {"path": "test/Witze.md", "content": "# Jokes"}),
            _roo_tool_result_msg("tu_1", '{"path":"test/Witze.md","operation":"created"}'),
            _roo_assistant_tool_use_msg("tu_ac", "attempt_completion",
                                        {"result": "Done."}),
            # attempt_completion result contains the new user instruction
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu_ac",
                        "content": [{"type": "text",
                                     "text": "<user_message>Rename test to witzeordner</user_message>"}],
                    },
                ],
            },
            # Second instruction: write to new location
            _roo_assistant_tool_use_msg("tu_2", "write_to_file",
                                        {"path": "witzeordner/Witze.md", "content": "# Jokes"}),
            _roo_tool_result_msg("tu_2", '{"path":"witzeordner/Witze.md","operation":"created"}'),
        ]

        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": messages,
            "tools": DEFAULT_TOOLS,
        })
        assert resp.status_code == 200

        last_user = next(
            (m for m in reversed(captured) if m.get("role") == "user"), None
        )
        assert last_user is not None
        content = last_user["content"]
        # Stop hint must NOT fire — these are two legitimate writes for two instructions
        assert "[CORRECTION]" not in content
        assert "STOP" not in content

    def test_anthropic_tool_result_gets_tool_result_prefix(self, client, llm):
        """
        After normalization, Anthropic-format tool_result blocks must carry the
        [Tool Result] prefix in the upstream messages, identical to role:tool messages.
        """
        captured = []

        async def side_effect(messages=None, **kwargs):
            captured.extend(messages or [])
            return llm_response(
                "<attempt_completion><result>Done.</result></attempt_completion>"
            )

        llm.side_effect = side_effect

        messages = [
            SYSTEM_MSG,
            user_msg("Do something."),
            _roo_assistant_tool_use_msg("tu_1", "read_file", {"path": "app.py"}),
            _roo_tool_result_msg("tu_1", "def main(): pass"),
        ]

        client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": messages,
            "tools": DEFAULT_TOOLS,
        })

        user_msgs = [m for m in captured if m.get("role") == "user"]
        # Find the normalized tool result message
        tool_result_content = " ".join(str(m.get("content", "")) for m in user_msgs)
        assert "[Tool Result]" in tool_result_content


# ──────────────────────────────────────────────────────────────────────────────
# rename hallucination → move_file
# ──────────────────────────────────────────────────────────────────────────────

class TestRenameHallucination:
    """Model outputs <rename> — must be resolved to move_file with correct params."""

    def test_rename_xml_resolved_to_move_file(self, client, llm):
        """<rename><old_path>...</old_path><new_path>...</new_path></rename> → move_file(source, destination)"""
        llm.return_value = llm_response(
            "<rename><old_path>test</old_path><new_path>witzeordner</new_path></rename>"
        )
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [SYSTEM_MSG, user_msg("Rename test to witzeordner")],
            "tools": DEFAULT_TOOLS,
        })
        assert resp.status_code == 200
        name, args = parse_tool_call(resp.json())
        assert name == "move_file"
        assert args["source"] == "test"
        assert args["destination"] == "witzeordner"

    def test_rename_without_old_new_path_passthrough(self, client, llm):
        """If model already uses correct param names they pass through unchanged."""
        llm.return_value = llm_response(
            "<move_file><source>foo</source><destination>bar</destination></move_file>"
        )
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [SYSTEM_MSG, user_msg("Move foo to bar")],
            "tools": DEFAULT_TOOLS,
        })
        assert resp.status_code == 200
        name, args = parse_tool_call(resp.json())
        assert name == "move_file"
        assert args == {"source": "foo", "destination": "bar"}

    def test_rename_finish_reason_tool_calls(self, client, llm):
        """finish_reason must be tool_calls when rename is resolved."""
        llm.return_value = llm_response(
            "<rename><old_path>a</old_path><new_path>b</new_path></rename>"
        )
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [SYSTEM_MSG, user_msg("Rename a to b")],
            "tools": DEFAULT_TOOLS,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["finish_reason"] == "tool_calls"


# ──────────────────────────────────────────────────────────────────────────────
# History normalisation: tool_calls stripped before reaching upstream
# ──────────────────────────────────────────────────────────────────────────────

class TestHistoryNormalizationUpstream:
    """
    Regression for the bug where assistant messages in history still had the
    tool_calls key after normalization, sending both XML (content) and native
    tool_calls to the upstream in the same message.
    """

    def test_tool_calls_not_in_upstream_messages(self, client, llm):
        """
        Turn 2: history includes a Turn-1 assistant message with tool_calls.
        The upstream must NOT see tool_calls in that message — only XML in content.
        """
        captured = {}

        async def capture_and_respond(messages=None, **kwargs):
            captured["messages"] = messages
            return llm_response(
                "<attempt_completion><result>Done.</result></attempt_completion>"
            )

        llm.side_effect = capture_and_respond

        # Simulate a Turn-2 request: Roo Code sends the Turn-1 assistant tool_call
        # in history, followed by the tool result, and a new user message.
        turn1_tc_id = "call_abc123"
        turn1_assistant = _assistant_tool_call_msg([{
            "id": turn1_tc_id,
            "type": "function",
            "function": {
                "name": "write_to_file",
                "arguments": json.dumps({"path": "foo.py", "content": "x = 1"}),
            },
        }])
        turn1_result = _tool_result_msg(turn1_tc_id, '{"path":"foo.py","operation":"created"}')

        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [
                SYSTEM_MSG,
                user_msg("Write foo.py"),
                turn1_assistant,
                turn1_result,
                user_msg("Now finish."),
            ],
            "tools": DEFAULT_TOOLS,
        })

        assert resp.status_code == 200
        assert captured.get("messages"), "Upstream was never called"

        for msg in captured["messages"]:
            assert "tool_calls" not in msg, (
                f"BUG: tool_calls key found in upstream message (role={msg.get('role')!r}).\n"
                f"Upstream receives conflicting formats (XML in content + OpenAI tool_calls).\n"
                f"Message keys: {list(msg.keys())}"
            )


# ──────────────────────────────────────────────────────────────────────────────
# Exploding apply_diff deduplication (v1.6.24)
# ──────────────────────────────────────────────────────────────────────────────

def _hunk(search: str, replace: str) -> str:
    return f"<<<<<<< SEARCH\n{search}\n=======\n{replace}\n>>>>>>> REPLACE\n"


class TestExplodingDiffDeduplication:

    def _apply_diff_xml(self, path: str, diff: str) -> str:
        diff_escaped = diff.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return (
            f"<apply_diff><path>{path}</path>"
            f"<diff>{diff_escaped}</diff></apply_diff>"
        )

    def test_duplicate_hunks_are_deduplicated_in_pipeline(self, client, llm):
        """
        Upstream returns an apply_diff with 10 identical hunks.
        The proxy must deduplicate to exactly 1 hunk before returning the tool call.
        """
        hunk = _hunk("setTasks(prev =>", "setTasks((prev: Task[]) =>")
        exploding_diff = hunk * 10
        llm.return_value = llm_response(
            self._apply_diff_xml("src/App.tsx", exploding_diff)
        )
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [SYSTEM_MSG, user_msg("Add types to App.tsx")],
            "tools": DEFAULT_TOOLS,
        })
        assert resp.status_code == 200
        name, args = parse_tool_call(resp.json())
        assert name == "apply_diff"
        assert args["diff"].count("<<<<<<< SEARCH") == 1, (
            f"Expected 1 hunk after deduplication, got:\n{args['diff']}"
        )

    def test_truncated_last_hunk_dropped_in_pipeline(self, client, llm):
        """Upstream returns a diff with a good hunk + truncated last hunk. Truncated must be dropped."""
        good = _hunk("setTasks(prev =>", "setTasks((prev: Task[]) =>")
        truncated = "<<<<<<< SEARCH\nsetEditingId(prev =>\n=======\nsetEditingId((prev: number | null) =>\n"
        llm.return_value = llm_response(
            self._apply_diff_xml("src/App.tsx", good + truncated)
        )
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [SYSTEM_MSG, user_msg("Fix App.tsx")],
            "tools": DEFAULT_TOOLS,
        })
        assert resp.status_code == 200
        name, args = parse_tool_call(resp.json())
        assert name == "apply_diff"
        assert args["diff"].count("<<<<<<< SEARCH") == 1
        assert "setTasks" in args["diff"]
        assert "setEditingId" not in args["diff"]

    def test_clean_diff_passes_through_in_pipeline(self, client, llm):
        """A diff with 3 unique hunks must reach the client with all 3 intact."""
        diff = (
            _hunk("alpha =>", "(alpha: A) =>")
            + _hunk("beta =>", "(beta: B) =>")
            + _hunk("gamma =>", "(gamma: C) =>")
        )
        llm.return_value = llm_response(
            self._apply_diff_xml("src/util.ts", diff)
        )
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [SYSTEM_MSG, user_msg("Fix util.ts")],
            "tools": DEFAULT_TOOLS,
        })
        assert resp.status_code == 200
        name, args = parse_tool_call(resp.json())
        assert name == "apply_diff"
        assert args["diff"].count("<<<<<<< SEARCH") == 3


# ──────────────────────────────────────────────────────────────────────────────
# YAML-like tool call format
# ──────────────────────────────────────────────────────────────────────────────


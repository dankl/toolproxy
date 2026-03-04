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

    def test_priming_not_injected_on_multi_turn(self, client, llm):
        """
        Multi-turn conversations (2+ user messages) must NOT get priming —
        the model already has context.
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
        # Only the 2 original user messages — no priming added
        assert len(user_messages) == 2


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
        assert len(tool_result_msgs) == 1
        assert "File written successfully" in tool_result_msgs[0]["content"]

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

        # The tool_result should appear as [Tool Result] in a user message
        user_content = " ".join(
            str(m.get("content", "")) for m in captured if m.get("role") == "user"
        )
        assert "[Tool Result]" in user_content or "def main" in user_content


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
        If the conversation history contains 2+ successful [Tool Result] messages
        for write operations, a CORRECTION / STOP hint must be appended to the
        last user message before calling the LLM.
        """
        captured = []

        async def side_effect(messages=None, **kwargs):
            captured.extend(messages or [])
            return llm_response(
                "<attempt_completion><result>Done.</result></attempt_completion>"
            )

        llm.side_effect = side_effect

        # Build a history with two successful write results already in place
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
            user_msg("Continue please."),
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

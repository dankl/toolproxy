"""
Tests for text_synthesis — specifically the write_to_file synthesis heuristic
and the apply_diff partial XML rescue.

Bug fixed in v1.6.12:
  _extract_target_file_from_context fell back to candidate_files[0] (the first
  VSCode Open Tab) when no filename was mentioned in the user message. This caused
  a large markdown explanation to be synthesized as write_to_file('.gitlab-ci.yml')
  because that file happened to be the first open tab — even though the response
  had nothing to do with CI/CD.

Fix: require the filename to appear in the user message text. If no candidate
file is explicitly mentioned, return None and let the fallback use attempt_completion.

Bug fixed in v1.6.13:
  When the model output a large <apply_diff> block without a closing </apply_diff>
  tag (OCI/vLLM response truncated before the closing tag), extract_xml_tool_calls
  found 0 matches and text_synthesis converted the entire diff to attempt_completion.
  The actual apply_diff content was lost.

Fix: partial XML rescue extracts <path> and <diff> via regex even without closing
tags, HTML-unescapes the diff content (model encodes <<<<<<< as &lt;&lt;&lt;...), and
returns a proper apply_diff tool call. validate_apply_diff_completeness in main.py
step 9b then drops it cleanly if the diff itself is also truncated.
"""
import json
import pytest

from app.services.text_synthesis import (
    _extract_target_file_from_context,
    synthesize_tool_call_from_text,
)
from tests.conftest import (
    DEFAULT_TOOLS,
    TOOL_APPLY_DIFF,
    TOOL_ATTEMPT_COMPLETION,
    TOOL_WRITE_TO_FILE,
    llm_response,
    parse_tool_call,
    user_msg,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _user_with_tabs(task: str, tabs: list[str]) -> dict:
    """User message with a VSCode Open Tabs section."""
    tabs_section = "# VSCode Open Tabs\n" + ", ".join(tabs)
    return {"role": "user", "content": f"{task}\n\n{tabs_section}\n\n# Current File\nsrc/Main.java"}


LONG_MARKDOWN_GUIDE = """\
**Creating an API call to save data in the TOS (FOS) via the generated OpenAPI client**

Below is a concise guide you can follow in your Java project.

## Prerequisites

- The OpenAPI generator has already produced Java stubs under `target/generated-sources/openapi`
- You have a working Spring Boot application

## Step 1: Inject the API client

```java
@Autowired
private OutboundInitiationsAnfrageApi api;
```

## Step 2: Build the request object

```java
OutboundInitiationsAnfrage body = new OutboundInitiationsAnfrage();
body.setType("TOS");
body.setPayload(payload);
```

## Step 3: Call the API

```java
api.createInitiationsAnfrage(body);
```

| Field   | Type   | Required |
|---------|--------|----------|
| type    | String | yes      |
| payload | Object | yes      |
"""


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests: _extract_target_file_from_context
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractTargetFile:

    def test_returns_none_when_no_open_tabs(self):
        messages = [user_msg("Explain how to use the API")]
        assert _extract_target_file_from_context(messages) is None

    def test_returns_none_when_tab_not_mentioned_in_message(self):
        """
        Regression: .gitlab-ci.yml is open but has nothing to do with the user's task.
        Must return None — not the first open tab.
        """
        messages = [_user_with_tabs(
            "Show me how to call the TOS API in Java",
            [".gitlab-ci.yml", "pom.xml"],
        )]
        assert _extract_target_file_from_context(messages) is None

    def test_returns_none_for_single_tab_not_mentioned(self):
        """
        Previously returned candidate_files[0] immediately when only one tab existed.
        Now must return None if not mentioned.
        """
        messages = [_user_with_tabs(
            "Explain the build process",
            [".gitlab-ci.yml"],
        )]
        assert _extract_target_file_from_context(messages) is None

    def test_returns_file_when_mentioned_in_user_text(self):
        messages = [_user_with_tabs(
            "Update the README.md with the new API instructions",
            ["README.md", ".gitlab-ci.yml"],
        )]
        assert _extract_target_file_from_context(messages) == "README.md"

    def test_returns_correct_file_among_multiple_candidates(self):
        """Second candidate is the one mentioned — must not return first."""
        messages = [_user_with_tabs(
            "Write the setup steps to docs/setup.md",
            ["README.md", "docs/setup.md", ".gitlab-ci.yml"],
        )]
        assert _extract_target_file_from_context(messages) == "docs/setup.md"

    def test_match_is_case_insensitive(self):
        messages = [_user_with_tabs(
            "Update readme.md with the changelog",
            ["README.md"],
        )]
        assert _extract_target_file_from_context(messages) == "README.md"

    def test_returns_none_when_no_user_message(self):
        messages = [{"role": "assistant", "content": "Hello"}]
        assert _extract_target_file_from_context(messages) is None

    def test_only_looks_at_most_recent_user_message(self):
        """
        Older user messages mention setup.md, but the latest one doesn't.
        Must return None — only the last user message is considered.
        """
        messages = [
            _user_with_tabs("Write to setup.md please", ["setup.md"]),
            {"role": "assistant", "content": "Done"},
            _user_with_tabs("Now explain the API", [".gitlab-ci.yml"]),
        ]
        assert _extract_target_file_from_context(messages) is None


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests: synthesize_tool_call_from_text
# ─────────────────────────────────────────────────────────────────────────────

class TestSynthesizeToolCall:

    TOOLS = [TOOL_WRITE_TO_FILE, TOOL_ATTEMPT_COMPLETION]

    def test_explanatory_text_falls_through_to_attempt_completion(self):
        """
        Regression: large markdown guide must NOT be synthesized as write_to_file
        when the target file is not mentioned in the user message.
        """
        messages = [_user_with_tabs(
            "Show me how to call the TOS API in Java",
            [".gitlab-ci.yml"],
        )]
        result = synthesize_tool_call_from_text(
            LONG_MARKDOWN_GUIDE, messages, self.TOOLS, "test-req"
        )
        assert result is not None, "Expected attempt_completion fallback"
        assert result[0]["function"]["name"] == "attempt_completion", (
            f"Expected attempt_completion but got: {result[0]['function']['name']!r}\n"
            "Regression: large markdown guide was synthesized as write_to_file for "
            "an unrelated open tab."
        )

    def test_explicit_file_mention_triggers_write_to_file(self):
        """When the user explicitly names the file, write_to_file synthesis must fire."""
        messages = [_user_with_tabs(
            "Write that guide to README.md",
            ["README.md", ".gitlab-ci.yml"],
        )]
        result = synthesize_tool_call_from_text(
            LONG_MARKDOWN_GUIDE, messages, self.TOOLS, "test-req"
        )
        assert result is not None
        assert result[0]["function"]["name"] == "write_to_file"
        args = json.loads(result[0]["function"]["arguments"])
        assert args["path"] == "README.md"

    def test_short_text_not_synthesized_as_file(self):
        """Responses under 200 chars must not trigger write_to_file."""
        messages = [_user_with_tabs("Update README.md", ["README.md"])]
        result = synthesize_tool_call_from_text(
            "Done! I updated the file.", messages, self.TOOLS, "test-req"
        )
        # Short text → no write_to_file
        if result:
            assert result[0]["function"]["name"] != "write_to_file"

    def test_no_tools_available_returns_none(self):
        messages = [_user_with_tabs("Update README.md", ["README.md"])]
        result = synthesize_tool_call_from_text(
            LONG_MARKDOWN_GUIDE, messages, [], "test-req"
        )
        assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# apply_diff partial XML rescue (v1.6.13)
# ─────────────────────────────────────────────────────────────────────────────

# Realistic truncated apply_diff as the model would produce it:
# <<<<<<< SEARCH is entity-encoded (&lt;&lt;&lt;...), </apply_diff> is missing.
_APPLY_DIFF_TRUNCATED = (
    "<apply_diff>"
    "<path>src/main/java/de/example/FooService.java</path>"
    "<diff>"
    "&lt;&lt;&lt;&lt;&lt;&lt;&lt; SEARCH\n"
    ":start_line:44\n"
    "-------\n"
    "  public void oldMethod() {}\n"
    "=======\n"
    "  public void newMethod() {}\n"
    "&gt;&gt;&gt;&gt;&gt;&gt;&gt; REPLACE\n"
    "</diff>"
    # </apply_diff> intentionally missing — simulates OCI truncation
)

_APPLY_DIFF_COMPLETE = (
    "<apply_diff>"
    "<path>src/main/java/de/example/FooService.java</path>"
    "<diff>"
    "&lt;&lt;&lt;&lt;&lt;&lt;&lt; SEARCH\n"
    ":start_line:44\n"
    "-------\n"
    "  public void oldMethod() {}\n"
    "=======\n"
    "  public void newMethod() {}\n"
    "&gt;&gt;&gt;&gt;&gt;&gt;&gt; REPLACE\n"
    "</diff>"
    "</apply_diff>"
)

_APPLY_DIFF_TRUNCATED_DIFF = (
    "<apply_diff>"
    "<path>src/main/java/de/example/FooService.java</path>"
    "<diff>"
    "&lt;&lt;&lt;&lt;&lt;&lt;&lt; SEARCH\n"
    ":start_line:44\n"
    "-------\n"
    "  public void oldMethod() {}\n"
    "=======\n"
    "  public void newMethod()\n"
    # >>>>>>> REPLACE intentionally missing — diff itself is truncated
)

TOOLS_WITH_APPLY_DIFF = [TOOL_APPLY_DIFF, TOOL_ATTEMPT_COMPLETION]


class TestApplyDiffRescue:

    def _messages(self):
        return [{"role": "user", "content": "Fix the method signature"}]

    def test_rescue_fires_when_closing_tag_missing(self):
        """
        Regression: apply_diff output truncated before </apply_diff> must be
        rescued instead of falling through to attempt_completion.
        """
        result = synthesize_tool_call_from_text(
            _APPLY_DIFF_TRUNCATED, self._messages(), TOOLS_WITH_APPLY_DIFF, "test-req"
        )
        assert result is not None
        assert result[0]["function"]["name"] == "apply_diff", (
            "Regression: truncated apply_diff was not rescued.\n"
            f"Got: {result[0]['function']['name']!r}"
        )

    def test_rescue_extracts_correct_path(self):
        result = synthesize_tool_call_from_text(
            _APPLY_DIFF_TRUNCATED, self._messages(), TOOLS_WITH_APPLY_DIFF, "test-req"
        )
        args = json.loads(result[0]["function"]["arguments"])
        assert args["path"] == "src/main/java/de/example/FooService.java"

    def test_rescue_decodes_html_entities_in_diff(self):
        """
        The model entity-encodes <<<<<<< SEARCH as &lt;&lt;&lt;...
        The rescued diff must contain literal < and > characters so that
        validate_apply_diff_completeness and Roo Code can process it.
        """
        result = synthesize_tool_call_from_text(
            _APPLY_DIFF_TRUNCATED, self._messages(), TOOLS_WITH_APPLY_DIFF, "test-req"
        )
        args = json.loads(result[0]["function"]["arguments"])
        diff = args["diff"]
        assert "<<<<<<< SEARCH" in diff, f"Entity decoding failed, diff={diff[:100]!r}"
        assert ">>>>>>> REPLACE" in diff, f"Entity decoding failed, diff={diff[:100]!r}"
        assert "&lt;" not in diff, "HTML entities not fully decoded"
        assert "&gt;" not in diff, "HTML entities not fully decoded"

    def test_rescue_also_works_when_apply_diff_is_complete(self):
        """
        A complete apply_diff (closing tag present) should normally be handled
        by xml_parser, not text_synthesis. But if xml_parser fails for any reason,
        text_synthesis rescue must still work correctly.
        """
        result = synthesize_tool_call_from_text(
            _APPLY_DIFF_COMPLETE, self._messages(), TOOLS_WITH_APPLY_DIFF, "test-req"
        )
        assert result is not None
        assert result[0]["function"]["name"] == "apply_diff"
        args = json.loads(result[0]["function"]["arguments"])
        assert "<<<<<<< SEARCH" in args["diff"]

    def test_rescue_fires_even_when_diff_is_also_truncated(self):
        """
        When both </apply_diff> AND >>>>>>> REPLACE are missing, rescue still
        extracts and returns the apply_diff. validate_apply_diff_completeness
        in main.py step 9b is responsible for dropping it cleanly.
        """
        result = synthesize_tool_call_from_text(
            _APPLY_DIFF_TRUNCATED_DIFF, self._messages(), TOOLS_WITH_APPLY_DIFF, "test-req"
        )
        assert result is not None
        assert result[0]["function"]["name"] == "apply_diff"
        # The diff is incomplete — validate_apply_diff_completeness will handle it

    def test_no_rescue_without_apply_diff_tool(self):
        """If apply_diff is not in the tool list, rescue must not fire."""
        result = synthesize_tool_call_from_text(
            _APPLY_DIFF_TRUNCATED, self._messages(), [TOOL_ATTEMPT_COMPLETION], "test-req"
        )
        # Should fall through to attempt_completion, not apply_diff
        if result:
            assert result[0]["function"]["name"] != "apply_diff"

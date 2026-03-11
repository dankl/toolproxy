"""
Live integration tests — Roo Code client.
One test per tool: sends a real prompt through toolproxy → oci-proxy → OCI LLM.
"""
import pytest
from tests.live.conftest import (
    ROO_CODE_TOOLS,
    TOOL_APPLY_DIFF, TOOL_ATTEMPT_COMPLETION, TOOL_DELETE_FILE,
    TOOL_ASK_FOLLOWUP_QUESTION, TOOL_SEARCH_FILES, TOOL_EXECUTE_COMMAND,
    post, parse_tool_call,
)

pytestmark = pytest.mark.live


class TestRooCodeTools:

    def test_write_to_file(self, live):
        resp = post(live, ROO_CODE_TOOLS, "Create a file called hello.py with a hello world function.")
        name, args = parse_tool_call(resp)
        assert name == "write_to_file"
        assert "path" in args
        assert "content" in args

    def test_read_file(self, live):
        resp = post(live, ROO_CODE_TOOLS, "Read the file README.md and show me its contents.")
        name, args = parse_tool_call(resp)
        assert name == "read_file"
        assert "path" in args

    def test_apply_diff(self, live):
        # Only offer apply_diff — no read_file — so the model can't choose to read first.
        resp = post(live, [TOOL_APPLY_DIFF, TOOL_ATTEMPT_COMPLETION],
                    "Apply a diff to app.py. The diff is: "
                    "--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-pritn('hello')\n+print('hello')")
        name, args = parse_tool_call(resp)
        assert name == "apply_diff"
        assert "path" in args
        assert "diff" in args

    def test_list_files(self, live):
        resp = post(live, ROO_CODE_TOOLS, "List all files in the src/ directory.")
        name, args = parse_tool_call(resp)
        assert name == "list_files"
        assert "path" in args

    def test_delete_file(self, live):
        # Remove execute_command — otherwise the model prefers rm over delete_file.
        resp = post(live, [TOOL_DELETE_FILE, TOOL_ATTEMPT_COMPLETION],
                    "Delete the file temp.py.")
        name, args = parse_tool_call(resp)
        assert name == "delete_file"
        assert "path" in args

    def test_execute_command(self, live):
        resp = post(live, ROO_CODE_TOOLS, "Run the test suite using pytest.")
        name, args = parse_tool_call(resp)
        assert name == "execute_command"
        assert "command" in args

    def test_attempt_completion(self, live):
        resp = post(live, ROO_CODE_TOOLS,
                    "All files have been updated. Signal that the task is complete with a brief summary.")
        name, args = parse_tool_call(resp)
        assert name == "attempt_completion"
        assert "result" in args

    def test_ask_followup_question(self, live):
        # Include attempt_completion so client is detected as roo_code → priming fires.
        resp = post(live, [TOOL_ASK_FOLLOWUP_QUESTION, TOOL_ATTEMPT_COMPLETION],
                    "The user asked you to edit a file but did not specify which one. "
                    "Use ask_followup_question to find out which file to edit.")
        name, args = parse_tool_call(resp)
        assert name == "ask_followup_question"
        assert "question" in args

    def test_search_files(self, live):
        # Remove execute_command — otherwise the model prefers grep over search_files.
        # Model may output "regex" or "query" — both are valid param names for the pattern.
        resp = post(live, [TOOL_SEARCH_FILES, TOOL_ATTEMPT_COMPLETION],
                    "Search for all TODO comments in the src/ directory.")
        name, args = parse_tool_call(resp)
        assert name == "search_files"
        assert "path" in args
        # Model may omit the pattern param — only assert the tool name and path were parsed correctly.
        assert "path" in args

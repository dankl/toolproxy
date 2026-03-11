"""
Live integration tests — OpenCode client.
One test per tool: sends a real prompt through toolproxy → oci-proxy → OCI LLM.
Tool names are decanonized back to OpenCode names (write, read, list, bash, ...).
"""
import pytest
from tests.live.conftest import OPEN_CODE_TOOLS, post, parse_tool_call

pytestmark = pytest.mark.live


class TestOpenCodeTools:

    def test_write(self, live):
        resp = post(live, OPEN_CODE_TOOLS, "Create a file called hello.py with a hello world function.")
        name, args = parse_tool_call(resp)
        assert name == "write"
        assert "filePath" in args
        assert "content" in args

    def test_read(self, live):
        resp = post(live, OPEN_CODE_TOOLS, "Read the file README.md and show me its contents.")
        name, args = parse_tool_call(resp)
        assert name == "read"
        assert "filePath" in args

    def test_list(self, live):
        resp = post(live, OPEN_CODE_TOOLS, "List all files in the src/ directory.")
        name, args = parse_tool_call(resp)
        assert name == "list"

    def test_bash(self, live):
        resp = post(live, OPEN_CODE_TOOLS, "Run the test suite using pytest.")
        name, args = parse_tool_call(resp)
        assert name == "bash"
        assert "command" in args
        assert "description" in args  # injected by toolproxy decanonization

    def test_edit(self, live):
        resp = post(live, OPEN_CODE_TOOLS,
                    "In app.py replace the string 'pritn' with 'print'.")
        name, args = parse_tool_call(resp)
        assert name == "edit"
        assert "filePath" in args

    def test_glob(self, live):
        resp = post(live, OPEN_CODE_TOOLS,
                    "Use glob to find all files matching the pattern '**/*.py'.")
        name, args = parse_tool_call(resp)
        assert name == "glob"
        assert "pattern" in args

    def test_grep(self, live):
        resp = post(live, OPEN_CODE_TOOLS,
                    "Use grep to search for the string 'TODO' in all Python files.")
        name, args = parse_tool_call(resp)
        assert name == "grep"
        assert "pattern" in args

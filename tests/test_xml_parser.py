"""
Unit tests for xml_parser.py — no app startup needed.

Tests cover:
  - fix_xml_string: angle-bracket escaping in content/diff tags
  - extract_xml_tool_calls: XML extraction, special chars, multi-line content
  - xml_element_to_dict: nested elements, leaf nodes
"""
import json
import xml.etree.ElementTree as ET

import pytest

from app.services.xml_parser import (
    convert_xml_tool_calls_to_openai_format,
    extract_mcp_tool_calls,
    extract_xml_tool_calls,
    fix_xml_string,
    parse_xml_to_arguments,
    xml_element_to_dict,
    _convert_self_closing_tags,
    _convert_attributes_to_elements,
    _strip_cdata,
)

TOOL_NAMES = [
    "write_to_file",
    "read_file",
    "apply_diff",
    "list_files",
    "attempt_completion",
]


# ──────────────────────────────────────────────────────────────────────────────
# fix_xml_string
# ──────────────────────────────────────────────────────────────────────────────

class TestFixXmlString:
    def test_clean_content_unchanged(self):
        xml = (
            "<write_to_file>"
            "<path>hello.py</path>"
            "<content>print('hello')</content>"
            "</write_to_file>"
        )
        fixed = fix_xml_string(xml)
        root = ET.fromstring(fixed)
        assert root.find("content").text == "print('hello')"

    def test_escapes_angle_brackets_in_content(self):
        """THE README BUG: <repo-url> inside <content> must survive XML parsing."""
        xml = (
            "<write_to_file>"
            "<path>README.md</path>"
            "<content>git clone <repo-url>\ncurl <host>/api</content>"
            "</write_to_file>"
        )
        fixed = fix_xml_string(xml)
        root = ET.fromstring(fixed)
        text = root.find("content").text
        assert "<repo-url>" in text
        assert "<host>" in text

    def test_escapes_angle_brackets_in_diff(self):
        xml = (
            "<apply_diff>"
            "<path>x.py</path>"
            "<diff>- old <tag>\n+ new <tag/></diff>"
            "</apply_diff>"
        )
        fixed = fix_xml_string(xml)
        root = ET.fromstring(fixed)
        assert "<tag>" in root.find("diff").text

    def test_diff_with_search_replace_markers(self):
        """Regression: <<<<<<< SEARCH markers inside <diff> must survive round-trip."""
        xml = (
            "<apply_diff>"
            "<path>tsconfig.json</path>"
            '<diff>\n<<<<<<< SEARCH\n:start_line:7\n-------\n  "include": ["src"]\n=======\n  "include": ["src/**/*"]\n>>>>>>> REPLACE\n</diff>'
            "</apply_diff>"
        )
        fixed = fix_xml_string(xml)
        root = ET.fromstring(fixed)
        assert "<<<<<<< SEARCH" in root.find("diff").text
        assert ">>>>>>> REPLACE" in root.find("diff").text

    def test_missing_closing_diff_tag(self):
        """Regression: model sometimes omits </diff> — should be auto-inserted."""
        xml = (
            "<apply_diff>"
            "<path>tsconfig.json</path>"
            '<diff>\n<<<<<<< SEARCH\n:start_line:7\n-------\n  "include": ["src"]\n=======\n  "include": ["src/**/*"]\n>>>>>>> REPLACE\n'
            "</apply_diff>"
        )
        fixed = fix_xml_string(xml)
        root = ET.fromstring(fixed)
        assert "<<<<<<< SEARCH" in root.find("diff").text

    def test_escapes_angle_brackets_in_result(self):
        xml = (
            "<attempt_completion>"
            "<result>See <docs> for more info.</result>"
            "</attempt_completion>"
        )
        fixed = fix_xml_string(xml)
        root = ET.fromstring(fixed)
        assert "<docs>" in root.find("result").text

    def test_escapes_heredoc_in_command(self):
        """Shell heredocs (python3 - <<'PY' ... PY) contain bare < that break ET.

        <command> is a raw-text tag so its content must be escaped before parsing,
        then restored verbatim on read. Regression for the OpenClaw bug where
        <exec><command>python3 - <<'PY' ...</command></exec> was dropped and the
        raw XML leaked to the client.
        """
        xml = (
            "<execute_command>"
            "<command>python3 - <<'PY'\nimport chromadb\nprint('ok')\nPY\n</command>"
            "</execute_command>"
        )
        fixed = fix_xml_string(xml)
        root = ET.fromstring(fixed)  # must not raise
        cmd = root.find("command").text
        assert "<<'PY'" in cmd
        assert "import chromadb" in cmd
        assert cmd.rstrip().endswith("PY")

    def test_escapes_redirect_in_command(self):
        """Shell redirects (sort < in.txt > out.txt) must survive parsing."""
        xml = (
            "<execute_command>"
            "<command>sort < in.txt > out.txt</command>"
            "</execute_command>"
        )
        fixed = fix_xml_string(xml)
        root = ET.fromstring(fixed)
        assert root.find("command").text == "sort < in.txt > out.txt"

    def test_no_double_escaping_of_entities(self):
        """Content that already uses &lt; must not be double-escaped."""
        xml = (
            "<write_to_file>"
            "<path>x.html</path>"
            "<content>&lt;div&gt;test&lt;/div&gt;</content>"
            "</write_to_file>"
        )
        fixed = fix_xml_string(xml)
        root = ET.fromstring(fixed)
        # ET decodes entities on read → we get back the original chars
        assert "<div>" in root.find("content").text

    def test_escapes_bare_ampersand_in_content(self):
        xml = (
            "<write_to_file>"
            "<path>x.sh</path>"
            "<content>A & B && C</content>"
            "</write_to_file>"
        )
        fixed = fix_xml_string(xml)
        ET.fromstring(fixed)  # must not raise

    def test_path_tag_not_modified(self):
        """The <path> tag content is plain text — fix_xml_string must not touch it."""
        xml = (
            "<write_to_file>"
            "<path>src/main/java/App.java</path>"
            "<content>public class App {}</content>"
            "</write_to_file>"
        )
        fixed = fix_xml_string(xml)
        root = ET.fromstring(fixed)
        assert root.find("path").text == "src/main/java/App.java"

    def test_mismatched_path_tag_fixed(self):
        """
        Model closes <path> with </diff> (anticipates the next sibling tag name).
        fix_xml_string must correct the closing tag so ET can parse the XML.

        Real-world example from logs (v1.6.10):
          <apply_diff>
            <path>src/.../StringLatinPatternUpdater.java</diff>  ← wrong
            <diff>...</diff>
          </apply_diff>
        """
        xml = (
            "<apply_diff>"
            "<path>src/main/java/Foo.java</diff>"
            "<diff><<<<<<< SEARCH\nold\n=======\nnew\n>>>>>>> REPLACE</diff>"
            "</apply_diff>"
        )
        fixed = fix_xml_string(xml)
        root = ET.fromstring(fixed)  # must not raise ParseError
        assert root.find("path").text == "src/main/java/Foo.java"

    def test_mismatched_path_tag_various_wrong_closers(self):
        """Fix works regardless of which wrong closing tag the model uses."""
        for wrong in ("diff", "content", "result", "apply_diff"):
            xml = f"<write_to_file><path>src/App.java</{wrong}><content>x</content></write_to_file>"
            fixed = fix_xml_string(xml)
            try:
                root = ET.fromstring(fixed)
                assert root.find("path").text == "src/App.java", f"path wrong for </{wrong}>"
            except ET.ParseError as e:
                raise AssertionError(f"ParseError for </{wrong}>: {e}\nFixed: {fixed}")

    def test_correct_tags_not_modified_by_mismatch_fix(self):
        """Correctly matched tags must be untouched."""
        xml = (
            "<apply_diff>"
            "<path>src/main/java/Foo.java</path>"
            "<diff>content</diff>"
            "</apply_diff>"
        )
        fixed = fix_xml_string(xml)
        root = ET.fromstring(fixed)
        assert root.find("path").text == "src/main/java/Foo.java"
        assert root.find("diff").text == "content"

    def test_multiline_content_with_xml_snippets(self):
        """README-style content with multiple XML-like elements."""
        inner = (
            "# Spring App\n\n"
            "```xml\n<dependency>\n  <groupId>org.springframework</groupId>\n</dependency>\n```\n\n"
            "Run: `curl <localhost:8080>/hello`"
        )
        xml = f"<write_to_file><path>README.md</path><content>{inner}</content></write_to_file>"
        fixed = fix_xml_string(xml)
        root = ET.fromstring(fixed)
        content = root.find("content").text
        assert "<dependency>" in content
        assert "<localhost:8080>" in content

    def test_escapes_ampersand_only_content(self):
        """
        Regression (Todo.md plan bug): content with && but NO < or >.
        The first pass skips it (no angle brackets present), the second pass
        must still catch the bare & characters.
        """
        xml = (
            "<write_to_file>"
            "<path>install.sh</path>"
            "<content>npm install && pip install flask\nLinter & Formatter</content>"
            "</write_to_file>"
        )
        fixed = fix_xml_string(xml)
        root = ET.fromstring(fixed)  # must not raise ParseError
        content = root.find("content").text
        assert "&&" in content
        assert "& " in content

    def test_escapes_ampersand_and_arrow_function(self):
        """
        Regression (Todo.md plan bug): content with BOTH bare & AND > (from =>).
        The > triggers the first pass, which must also escape all & before replacing <>.
        """
        xml = (
            "<write_to_file>"
            "<path>app.js</path>"
            "<content>"
            "app.get('/getWitz', (req, res) => { res.json({}) })\n"
            "Linter & Formatter\n"
            "npm install && pip install flask"
            "</content>"
            "</write_to_file>"
        )
        fixed = fix_xml_string(xml)
        root = ET.fromstring(fixed)  # must not raise ParseError
        content = root.find("content").text
        assert "=>" in content    # arrow function survived
        assert "&&" in content    # shell && survived
        assert "& " in content    # standalone & survived


# ──────────────────────────────────────────────────────────────────────────────
# extract_xml_tool_calls
# ──────────────────────────────────────────────────────────────────────────────

class TestExtractXmlToolCalls:
    def test_simple_write_to_file(self):
        content = (
            "<write_to_file>"
            "<path>foo.py</path>"
            "<content>print('hi')</content>"
            "</write_to_file>"
        )
        calls, remaining = extract_xml_tool_calls(content, TOOL_NAMES)
        assert len(calls) == 1
        assert calls[0].name == "write_to_file"
        assert calls[0].arguments["path"] == "foo.py"
        assert "print" in calls[0].arguments["content"]
        assert remaining.strip() == ""

    def test_xml_special_chars_in_content_regression(self):
        """
        Regression test for the README.md bug:
        Content with <url> tags must not break extraction.
        """
        content = (
            "<write_to_file>"
            "<path>README.md</path>"
            "<content># Project\n\ngit clone <repo-url>\ncurl <localhost:8080>/api</content>"
            "</write_to_file>"
        )
        calls, _ = extract_xml_tool_calls(content, TOOL_NAMES)
        assert len(calls) == 1
        assert calls[0].name == "write_to_file"
        assert calls[0].arguments["path"] == "README.md"
        assert "<repo-url>" in calls[0].arguments["content"]
        assert "<localhost:8080>" in calls[0].arguments["content"]

    def test_preamble_text_before_xml_ignored(self):
        """Model outputs explanatory text before the XML — call is still found."""
        content = (
            "Sure, I'll write that file now.\n\n"
            "<write_to_file><path>x.py</path><content>pass</content></write_to_file>"
        )
        calls, _ = extract_xml_tool_calls(content, TOOL_NAMES)
        assert len(calls) == 1
        assert calls[0].name == "write_to_file"

    def test_read_file(self):
        content = "<read_file><path>src/main.py</path></read_file>"
        calls, _ = extract_xml_tool_calls(content, TOOL_NAMES)
        assert len(calls) == 1
        assert calls[0].name == "read_file"
        assert calls[0].arguments["path"] == "src/main.py"

    def test_attempt_completion(self):
        content = "<attempt_completion><result>All done.</result></attempt_completion>"
        calls, _ = extract_xml_tool_calls(content, TOOL_NAMES)
        assert len(calls) == 1
        assert calls[0].name == "attempt_completion"
        assert calls[0].arguments["result"] == "All done."

    def test_attempt_completion_result_must_not_contain_xml(self):
        """
        Regression: model puts <write_to_file> XML inside <result> of attempt_completion.
        The result text must come through as plain text, not as a nested call.
        """
        inner = "<write_to_file><path>x</path><content>y</content></write_to_file>"
        content = f"<attempt_completion><result>{inner}</result></attempt_completion>"
        # fix_xml_string should escape the inner XML
        calls, _ = extract_xml_tool_calls(content, TOOL_NAMES)
        # Only the outer attempt_completion should be extracted
        assert len(calls) == 1
        assert calls[0].name == "attempt_completion"

    def test_jsx_content_returned_as_string(self):
        """Regression: JSX inside <content> is valid XML — must come back as string, not dict."""
        content = (
            "<write_to_file>"
            "<path>src/App.tsx</path>"
            "<content>const App = () => {\n  return (\n    <div><h1>Hello</h1></div>\n  );\n};\nexport default App;\n</content>"
            "</write_to_file>"
        )
        calls, _ = extract_xml_tool_calls(content, TOOL_NAMES)
        assert len(calls) == 1
        result = calls[0].arguments["content"]
        assert isinstance(result, str), f"content should be str, got {type(result)}"
        assert "<div>" in result

    def test_unknown_tool_not_extracted(self):
        content = "<update_todo_list>[{...}]</update_todo_list>"
        calls, _ = extract_xml_tool_calls(content, TOOL_NAMES)
        assert calls == []

    def test_multiline_content(self):
        content = (
            "<write_to_file><path>app.py</path>"
            "<content>\nimport os\n\ndef main():\n    pass\n</content>"
            "</write_to_file>"
        )
        calls, _ = extract_xml_tool_calls(content, TOOL_NAMES)
        assert len(calls) == 1
        assert "def main" in calls[0].arguments["content"]

    def test_multiple_calls_extracted(self):
        content = (
            "<read_file><path>a.py</path></read_file>\n"
            "<write_to_file><path>b.py</path><content>x</content></write_to_file>"
        )
        calls, _ = extract_xml_tool_calls(content, TOOL_NAMES)
        assert len(calls) == 2
        assert calls[0].name == "read_file"
        assert calls[1].name == "write_to_file"

    def test_exec_alias_with_heredoc_command(self):
        """Regression: <exec> alias with a heredoc-bearing <command>.

        The model output `<exec><command>python3 - <<'PY' ... PY</command></exec>`
        followed by a `NO_REPLY` sentinel. Previously the bare `<<` broke
        ET.fromstring (command was not a raw-text tag), so the tool call was
        dropped and the raw XML + NO_REPLY leaked to the client.
        """
        content = (
            "<exec><command>python3 - <<'PY'\n"
            "import chromadb\n"
            "client = chromadb.HttpClient(host=\"localhost\", port=\"8000\")\n"
            "print(client.get_collection(name=\"docs\").count())\n"
            "PY\n"
            "</command></exec>NO_REPLY "
        )
        calls, remaining = extract_xml_tool_calls(content, ["execute_command", "exec"])
        assert len(calls) == 1
        assert calls[0].name == "exec"
        cmd = calls[0].arguments["command"]
        assert "<<'PY'" in cmd
        assert "import chromadb" in cmd
        assert cmd.rstrip().endswith("PY")
        # XML block fully removed; only the trailing sentinel remains
        assert "<exec>" not in remaining
        assert "<command>" not in remaining
        assert "NO_REPLY" in remaining

    def test_empty_content_returns_nothing(self):
        calls, rem = extract_xml_tool_calls("", TOOL_NAMES)
        assert calls == []

    def test_no_tool_names_returns_nothing(self):
        content = "<write_to_file><path>x</path><content>y</content></write_to_file>"
        calls, _ = extract_xml_tool_calls(content, [])
        assert calls == []

    def test_convert_to_openai_format(self):
        """Full pipeline: XML string → OpenAI tool_calls format."""
        content = "<write_to_file><path>x.py</path><content>pass</content></write_to_file>"
        calls, _ = extract_xml_tool_calls(content, TOOL_NAMES)
        openai_calls = convert_xml_tool_calls_to_openai_format(calls)

        assert len(openai_calls) == 1
        tc = openai_calls[0]
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "write_to_file"
        # arguments must be a valid JSON string
        args = json.loads(tc["function"]["arguments"])
        assert args["path"] == "x.py"
        assert args["content"] == "pass"


# ──────────────────────────────────────────────────────────────────────────────
# xml_element_to_dict
# ──────────────────────────────────────────────────────────────────────────────

class TestXmlElementToDict:
    def _elem(self, xml_string: str) -> ET.Element:
        return ET.fromstring(xml_string)

    def test_leaf_plain_text(self):
        elem = self._elem("<path>src/main.py</path>")
        result = xml_element_to_dict(elem)
        assert result == "src/main.py"

    def test_leaf_multiline_text(self):
        elem = self._elem("<content>line1\nline2\nline3</content>")
        result = xml_element_to_dict(elem)
        assert "line1" in result
        assert "line3" in result

    def test_nested_two_children(self):
        elem = self._elem(
            "<write_to_file>"
            "<path>foo.py</path>"
            "<content>pass</content>"
            "</write_to_file>"
        )
        result = xml_element_to_dict(elem)
        assert isinstance(result, dict)
        assert result["path"] == "foo.py"
        assert result["content"] == "pass"

    def test_deeply_nested(self):
        elem = self._elem(
            "<outer>"
            "<middle>"
            "<inner>value</inner>"
            "</middle>"
            "</outer>"
        )
        result = xml_element_to_dict(elem)
        assert result["middle"]["inner"] == "value"

    def test_repeated_children_become_list(self):
        elem = self._elem(
            "<root>"
            "<item>a</item>"
            "<item>b</item>"
            "<item>c</item>"
            "</root>"
        )
        result = xml_element_to_dict(elem)
        assert isinstance(result["item"], list)
        assert result["item"] == ["a", "b", "c"]

    def test_element_with_attributes(self):
        elem = self._elem('<file name="foo.py">content</file>')
        result = xml_element_to_dict(elem)
        assert isinstance(result, dict)
        assert result["value"] == "content"
        assert result["name"] == "foo.py"


# ──────────────────────────────────────────────────────────────────────────────
# extract_mcp_tool_calls
# ──────────────────────────────────────────────────────────────────────────────

class TestExtractMcpToolCalls:
    """Tests for Roo Code XML-mode <use_mcp_tool> parsing."""

    _BASIC = (
        "<use_mcp_tool>\n"
        "<server_name>grafana-mcp</server_name>\n"
        "<tool_name>list_datasources</tool_name>\n"
        '<arguments>\n{"type": "loki"}\n</arguments>\n'
        "</use_mcp_tool>"
    )

    def test_basic_parse(self):
        calls, remaining = extract_mcp_tool_calls(self._BASIC)
        assert len(calls) == 1
        assert calls[0].name == "grafana-mcp__list_datasources"
        assert calls[0].arguments == {"type": "loki"}
        assert remaining == ""

    def test_tool_id_generated(self):
        calls, _ = extract_mcp_tool_calls(self._BASIC)
        assert calls[0].id.startswith("call_")

    def test_no_server_name_uses_tool_name_only(self):
        xml = (
            "<use_mcp_tool>\n"
            "<tool_name>query_logs</tool_name>\n"
            "<arguments>{}</arguments>\n"
            "</use_mcp_tool>"
        )
        calls, _ = extract_mcp_tool_calls(xml)
        assert len(calls) == 1
        assert calls[0].name == "query_logs"

    def test_empty_arguments_defaults_to_empty_dict(self):
        xml = (
            "<use_mcp_tool>\n"
            "<server_name>my-mcp</server_name>\n"
            "<tool_name>ping</tool_name>\n"
            "<arguments></arguments>\n"
            "</use_mcp_tool>"
        )
        calls, _ = extract_mcp_tool_calls(xml)
        assert calls[0].arguments == {}

    def test_preamble_text_preserved_in_remaining(self):
        content = "Sure, let me check.\n" + self._BASIC
        calls, remaining = extract_mcp_tool_calls(content)
        assert len(calls) == 1
        assert "Sure, let me check." in remaining

    def test_xml_block_removed_from_remaining(self):
        calls, remaining = extract_mcp_tool_calls(self._BASIC)
        assert "<use_mcp_tool>" not in remaining

    def test_no_mcp_tag_returns_unchanged(self):
        content = "Just a normal text response."
        calls, remaining = extract_mcp_tool_calls(content)
        assert calls == []
        assert remaining == content

    def test_empty_content_returns_empty(self):
        calls, remaining = extract_mcp_tool_calls("")
        assert calls == []
        assert remaining == ""

    def test_missing_tool_name_skipped(self):
        xml = (
            "<use_mcp_tool>\n"
            "<server_name>grafana-mcp</server_name>\n"
            "<arguments>{}</arguments>\n"
            "</use_mcp_tool>"
        )
        calls, _ = extract_mcp_tool_calls(xml)
        assert calls == []

    def test_complex_json_arguments(self):
        xml = (
            "<use_mcp_tool>\n"
            "<server_name>grafana-mcp</server_name>\n"
            "<tool_name>query_loki_logs</tool_name>\n"
            '<arguments>{"datasourceUid": "abc123", "logql": "{pod=\\"my-pod\\"}", "limit": 50}</arguments>\n'
            "</use_mcp_tool>"
        )
        calls, _ = extract_mcp_tool_calls(xml)
        assert len(calls) == 1
        assert calls[0].name == "grafana-mcp__query_loki_logs"
        assert calls[0].arguments["limit"] == 50
        assert calls[0].arguments["datasourceUid"] == "abc123"

    def test_convert_to_openai_format(self):
        calls, _ = extract_mcp_tool_calls(self._BASIC)
        openai_calls = convert_xml_tool_calls_to_openai_format(calls)
        assert len(openai_calls) == 1
        tc = openai_calls[0]
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "grafana-mcp__list_datasources"
        args = json.loads(tc["function"]["arguments"])
        assert args == {"type": "loki"}


# ──────────────────────────────────────────────────────────────────────────────
# CDATA handling
# ──────────────────────────────────────────────────────────────────────────────

class TestCdataHandling:
    def test_strip_cdata_simple(self):
        assert _strip_cdata("<![CDATA[hello]]>") == "hello"

    def test_strip_cdata_with_xml_inside(self):
        raw = "<![CDATA[<path>foo</path>]]>"
        assert _strip_cdata(raw) == "<path>foo</path>"

    def test_strip_cdata_mixed_content(self):
        raw = "before<![CDATA[inner]]>after"
        assert _strip_cdata(raw) == "beforeinnerafter"

    def test_cdata_in_content_tag_survives_parsing(self):
        xml = (
            "<write_to_file>"
            "<path>foo.html</path>"
            "<content><![CDATA[<div>Hello</div>]]></content>"
            "</write_to_file>"
        )
        args, tid, reason = parse_xml_to_arguments(xml, "write_to_file")
        assert "<div>Hello</div>" in str(args.get("content", ""))

    def test_cdata_inside_write_to_file_content(self):
        xml = (
            "<write_to_file>"
            "<path>foo.html</path>"
            "<content><![CDATA[<repo-url>https://example.com</repo-url>]]></content>"
            "</write_to_file>"
        )
        args, tid, reason = parse_xml_to_arguments(xml, "write_to_file")
        content = args.get("content", "")
        assert "<repo-url>" in content


# ──────────────────────────────────────────────────────────────────────────────
# Self-closing tag handling
# ──────────────────────────────────────────────────────────────────────────────

class TestSelfClosingTags:
    def test_self_closing_param_converted(self):
        result = _convert_self_closing_tags("<path/>")
        assert result == "<path></path>"

    def test_self_closing_with_whitespace(self):
        result = _convert_self_closing_tags("<path />")
        assert result == "<path></path>"

    def test_self_closing_in_tool_call(self):
        xml = "<read_file><path /></read_file>"
        result = _convert_self_closing_tags(xml)
        assert "<path></path>" in result

    def test_self_closing_parsed_successfully(self):
        xml = "<read_file><path /></read_file>"
        fixed = fix_xml_string(xml)
        args, tid, reason = parse_xml_to_arguments(fixed, "read_file")
        assert args.get("path") == ""


# ──────────────────────────────────────────────────────────────────────────────
# Attribute format handling
# ──────────────────────────────────────────────────────────────────────────────

class TestAttributeFormat:
    def test_attributes_converted_to_elements(self):
        xml = '<write_to_file path="README.md" content="# Hello" />'
        result = _convert_attributes_to_elements(xml)
        assert "<path>README.md</path>" in result
        assert "<content># Hello</content>" in result

    def test_attribute_format_parsed_successfully(self):
        xml = '<write_to_file path="README.md" content="# Hello" />'
        fixed = fix_xml_string(xml)
        args, tid, reason = parse_xml_to_arguments(fixed, "write_to_file")
        assert args.get("path") == "README.md"
        assert args.get("content") == "# Hello"

    def test_mixed_attribute_and_element_still_works(self):
        """Some models mix attributes and child elements."""
        xml = '<apply_diff path="app.py"><diff>old\nnew</diff></apply_diff>'
        fixed = fix_xml_string(xml)
        args, tid, reason = parse_xml_to_arguments(fixed, "apply_diff")
        assert args.get("path") == "app.py"
        assert "old" in args.get("diff", "")


# ──────────────────────────────────────────────────────────────────────────────
# Robust fallback parsing
# ──────────────────────────────────────────────────────────────────────────────

class TestRobustFallbackParsing:
    def test_missing_root_close_still_parsed(self):
        """Truncated XML without closing root tag should still parse via wrapped-root fallback."""
        xml = "<write_to_file><path>foo.py</path><content>print('hello')</content>"
        # No </write_to_file> — should still parse
        args, tid, reason = parse_xml_to_arguments(xml, "write_to_file")
        assert args.get("path") == "foo.py"
        assert "hello" in str(args.get("content", ""))

    def test_extra_whitespace_in_tags(self):
        xml = "< write_to_file ><path>foo</path></ write_to_file >"
        # ET is lenient with whitespace in tags — should still work after fix
        fixed = fix_xml_string(xml)
        args, tid, reason = parse_xml_to_arguments(fixed, "write_to_file")
        assert args.get("path") == "foo"

    def test_case_insensitive_tag_names(self):
        """Model sometimes uses uppercase in tag names. Keys preserved as-is,
        case-insensitive matching is handled by _remap_args_to_schema downstream."""
        xml = "<WRITE_TO_FILE><PATH>foo.py</PATH><CONTENT>print('hi')</CONTENT></WRITE_TO_FILE>"
        args, tid, reason = parse_xml_to_arguments(xml, "write_to_file")
        # Keys are uppercase because xml_element_to_dict preserves tag case
        assert args.get("PATH") == "foo.py"
        assert args.get("CONTENT") == "print('hi')"


# ──────────────────────────────────────────────────────────────────────────────
# Escaping edge cases
# ──────────────────────────────────────────────────────────────────────────────

class TestEscapingEdgeCases:
    def test_xml_in_string_values_old_and_new(self):
        """OpenCode edit tool: oldString/newString may contain XML-like content.
        Keys preserve XML tag case; schema remap handles normalization downstream."""
        xml = (
            "<edit>"
            "<filePath>main.py</filePath>"
            "<oldString>if x &lt; 5:</oldString>"
            "<newString>if x &lt;= 5:</newString>"
            "</edit>"
        )
        fixed = fix_xml_string(xml)
        args, tid, reason = parse_xml_to_arguments(fixed, "edit")
        assert "x < 5" in str(args.get("oldString", ""))
        assert "x <= 5" in str(args.get("newString", ""))

    def test_arrow_function_in_content(self):
        """JavaScript arrow functions use => which can confuse XML heuristics."""
        xml = (
            "<write_to_file>"
            "<path>app.js</path>"
            "<content>const fn = (req, res) => { return res.json({ ok: true }); }</content>"
            "</write_to_file>"
        )
        fixed = fix_xml_string(xml)
        args, tid, reason = parse_xml_to_arguments(fixed, "write_to_file")
        content = args.get("content", "")
        assert "req, res" in content
        assert "=>" in content

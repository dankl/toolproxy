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
    extract_xml_tool_calls,
    fix_xml_string,
    xml_element_to_dict,
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

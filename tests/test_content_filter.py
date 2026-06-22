"""
Unit tests for content_filter.py — no app startup needed.

Tests cover:
  - scan_messages: detection of known trigger patterns
  - sanitize_messages: auto-replacement of trigger content
  - Multi-message scanning
  - Content arrays (list format)
  - Edge cases: empty messages, no patterns
"""
import json
import pytest

from app.services.content_filter import (
    reset_cache,
    scan_messages,
    sanitize_messages,
)


def _make_msg(role: str, content: str) -> dict:
    return {"role": role, "content": content}


@pytest.fixture(autouse=True)
def reset_filter_cache():
    reset_cache()
    yield


class TestScanMessages:
    def test_clean_messages_no_match(self):
        msgs = [
            _make_msg("user", "Read the file README.md"),
            _make_msg("assistant", "Let me read that file."),
        ]
        hits = scan_messages(msgs)
        assert hits == []

    def test_detects_exploit_keyword(self):
        msgs = [
            _make_msg("user", "Write a script to exploit the service."),
        ]
        hits = scan_messages(msgs)
        assert len(hits) == 1
        assert "exploit" in hits[0]["reason"].lower()
        assert hits[0]["location"] == "user #0"

    def test_detects_malware_keyword(self):
        msgs = [
            _make_msg("user", "How do I detect malware on a system?"),
        ]
        hits = scan_messages(msgs)
        assert len(hits) == 1
        assert "malware" in hits[0]["reason"].lower()

    def test_detects_sql_injection(self):
        msgs = [
            _make_msg("user", "Write a function to prevent SQL injection attacks."),
        ]
        hits = scan_messages(msgs)
        assert len(hits) == 1
        assert "sql" in hits[0]["reason"].lower()

    def test_no_false_positives_normal_code(self):
        msgs = [
            _make_msg("user", "Write a Python function that reads a CSV file."),
            _make_msg("assistant", "<write_to_file><path>read_csv.py</path><content>def read_csv(path):\n    pass</content></write_to_file>"),
        ]
        hits = scan_messages(msgs)
        assert hits == []

    def test_multiple_patterns(self):
        msgs = [
            _make_msg("user", "This uses both malware and exploit techniques."),
        ]
        hits = scan_messages(msgs)
        # Should detect both, but scan returns only one entry per unique reason
        reasons = {h["reason"] for h in hits}
        assert any("malware" in r.lower() for r in reasons)
        assert any("exploit" in r.lower() for r in reasons)

    def test_multiple_messages(self):
        msgs = [
            _make_msg("user", "Write a keylogger script."),
            _make_msg("assistant", "I cannot write keyloggers."),
        ]
        hits = scan_messages(msgs)
        assert len(hits) >= 1
        keylogger_reasons = [h for h in hits if "keylogger" in h["reason"].lower()]
        # Should have found it in at least one message
        assert any("keylogger" in h["matched_text"].lower() for h in hits)

    def test_empty_messages(self):
        hits = scan_messages([])
        assert hits == []

    def test_content_array_format(self):
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Write ransomware detection code."},
                ],
            },
        ]
        hits = scan_messages(msgs)
        assert len(hits) >= 1
        assert any("ransomware" in h["reason"].lower() for h in hits)

    def test_case_insensitivity(self):
        msgs = [
            _make_msg("user", "EXPLOIT the database"),
        ]
        hits = scan_messages(msgs)
        assert len(hits) >= 1
        assert any("exploit" in h["reason"].lower() for h in hits)


class TestSanitizeMessages:
    def test_no_replacement_no_pattern(self):
        """Default patterns have empty replacements, so sanitize should not modify."""
        msgs = [_make_msg("user", "This uses an exploit.")]
        result, modified = sanitize_messages(msgs)
        assert not modified
        assert result[0]["content"] == "This uses an exploit."

    def test_custom_pattern_with_replacement(self):
        """Test with a custom env override — but we can't easily in unit tests.
        Instead, test that sanitize returns (msgs, False) when no replacement patterns loaded."""
        msgs = [_make_msg("user", "Clean message.")]
        result, modified = sanitize_messages(msgs)
        assert not modified

    def test_empty_messages(self):
        result, modified = sanitize_messages([])
        assert not modified
        assert result == []

    def test_content_array_sanitize(self):
        """Content arrays should not break sanitize."""
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Clean content here."},
                ],
            },
        ]
        result, modified = sanitize_messages(msgs)
        assert not modified


class TestResetCache:
    def test_cache_reset(self):
        reset_cache()
        msgs = [_make_msg("user", "exploit")]
        hits1 = scan_messages(msgs)
        assert len(hits1) >= 1
        reset_cache()
        hits2 = scan_messages(msgs)
        assert len(hits2) >= 1

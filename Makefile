.PHONY: test test-quick test-all test-xml test-content test-lint test-validate test-smoke test-watch test-roundtrip test-bugs

PYTHON ?= python3

# ─────────────────────────────────────────────────────────────────────
# toolproxy — test & validation targets
# ─────────────────────────────────────────────────────────────────────

test: test-quick
	@echo "All primary tests passed ✓"

# Quick smoke test (primary modules, 30s)
test-quick:
	$(PYTHON) -m pytest tests/test_xml_parser.py tests/test_content_filter.py tests/test_text_synthesis.py -v --tb=short

# Full suite (excluding live E2E tests, ~1min)
test-all:
	$(PYTHON) -m pytest tests/ --ignore=tests/live -v --tb=short

# XML parser only
test-xml:
	$(PYTHON) -m pytest tests/test_xml_parser.py -v --tb=short

# Content filter only
test-content:
	$(PYTHON) -m pytest tests/test_content_filter.py -v --tb=short

# Roundtrip integration tests
test-roundtrip:
	$(PYTHON) -m pytest tests/test_roundtrip.py -v --tb=short

# Bug-proof regression tests
test-bugs:
	$(PYTHON) -m pytest tests/test_bug_proof.py -v --tb=short

# Validate: run ALL non-live tests (strict mode — fail on first error)
test-validate:
	$(PYTHON) -m pytest tests/ --ignore=tests/live -x --tb=long -v

# Quick: parser + content filter + roundtrip (common path)
test-smoke:
	$(PYTHON) -m pytest tests/test_xml_parser.py tests/test_content_filter.py tests/test_roundtrip.py tests/test_text_synthesis.py -v --tb=short

# Lint (if ruff available)
test-lint:
	@which ruff >/dev/null 2>&1 && ruff check app/ tests/ || echo "ruff not installed — skipping"

# Watch mode — requires pytest-watch (pip install pytest-watch)
test-watch:
	@which ptw >/dev/null 2>&1 && ptw tests/test_xml_parser.py tests/test_content_filter.py || echo "Install pytest-watch: pip install pytest-watch"

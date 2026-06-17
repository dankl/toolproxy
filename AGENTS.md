# Toolproxy - Development Log

## Overview
Toolproxy ist ein FastAPI-basierter Proxy zwischen OpenCode/OpenClaw und gpt-oss-120b (via OCI). Er übersetzt zwischen OpenAI tool_calls API und XML-Format für Modelle, die keine nativen Function Calls unterstützen.

## Architecture
```
Client (OpenCode/OpenClaw)
  ↓ OpenAI JSON with tools[]
toolproxy (Port 8007)
  ↓ XML System Prompt + XML-normalized history
Upstream LLM (OCI/vLLM)
  ↓ XML tool call in response
toolproxy - parses XML → OpenAI tool_calls
  ↓ Standard OpenAI response
Client
```

## Major Changes & Fixes

### 1. XML Parser Improvements (`app/services/xml_parser.py`)

#### CDATA & Self-Closing Tags
- Added `_strip_cdata()` to handle `<![CDATA[...]]>` wrappers
- Added `_convert_self_closing_tags()` for `<param/>` → `<param></param>`
- Added `_convert_attributes_to_elements()` for attribute-style tool calls

#### Fallback Parsing
- Implemented `_parse_xml_with_fallback()` with 3 strategies:
  1. Direct `ET.fromstring()`
  2. Close unclosed root tag
  3. Wrap in synthetic root

#### Raw Text Tag Escaping
- Extended `_RAW_TEXT_TAGS` to include `oldString`, `newString`, `old_string`, `new_string`
- Prevents XML parser errors when these tags contain `<`, `>`, or other special characters

#### Nested Edit Handling
- Fixed regex matching for nested `<edit><edits><edit>...</edit></edits></edit>` structures
- Added automatic `</edits>` insertion when missing
- Implemented nested deduplication: removes inner `<edit>` blocks that are contained in outer blocks
- Added exact duplicate detection: removes identical tool calls (same name + arguments)

### 2. Edit Tool Format Conversion (`app/services/tool_call_fixups.py`)

#### flatten_edit_nesting()
Converts various edit formats to OpenClaw's expected format:

**Input formats:**
- Flat: `{"path": "...", "oldString": "...", "newString": "..."}`
- Nested XML: `{"path": "...", "edits": {"edit": {"oldString": "...", "newString": "..."}}}`

**Output format (OpenClaw):**
```json
{
  "path": "...",
  "edits": [
    {
      "oldText": "...",
      "newText": "..."
    }
  ]
}
```

**Key changes:**
- Renamed output fields from `oldString`/`newString` to `oldText`/`newText`
- Added support for multiple input field name variants (oldString, old_string, oldstring, oldText)
- Handles both flat and nested XML input formats

### 3. Content Filter (`app/services/content_filter.py`)

#### OCI Safety Filter Prevention
- Pre-request scanning for patterns that trigger OCI's content filter
- 22 default patterns for security terms (exploit, malware, ransomware, etc.)
- Validation error patterns to detect schema validation messages

#### Modes
- `detect` (default): Log warnings, let request through
- `sanitize`: Auto-replace trigger patterns
- `block`: Return 400 before upstream call
- `off`: Disable entirely

**Current setting:** `CONTENT_FILTER_MODE=off` (validation errors in history were causing noise)

### 4. Shell Injection Fix (`app/services/tool_call_fixups.py`)

#### convert_move_file_to_execute_command()
- Fixed shell injection vulnerability in `move_file`/`rename_file` conversion
- Changed from manual string quoting to `shlex.quote()`
- Prevents command injection via malicious file paths

### 5. Tool Call Deduplication (`app/services/xml_parser.py`)

#### Nested Deduplication
```python
# Removes inner <edit> blocks contained in outer blocks
if any(
    other is not tc and other.name == tc.name 
    and tc.raw_xml in other.raw_xml 
    and len(tc.raw_xml) < len(other.raw_xml)
    for other in tool_calls
):
    continue
```

#### Exact Deduplication
```python
# Removes identical tool calls (same name + arguments)
key = (tc.name, json.dumps(tc.arguments, sort_keys=True))
if key in seen:
    continue
```

### 6. Tool Name Aliases (`app/services/tool_mapping.py`)

#### Added Aliases
- `exec` → `execute_command`
- `write_file` → `write_to_file`
- `read` → `read_file`
- `bash` → `execute_command`

#### Case-Insensitive Matching
- `_remap_args_to_schema()` now uses case-insensitive fallback for parameter names

### 7. Configuration (`app/config.py`)

#### New Settings
- `content_filter_patterns`: JSON array of custom patterns
- `content_filter_mode`: detect/sanitize/block/off

#### Environment Variables
```bash
UPSTREAM_URL=http://10.3.0.120:4000/v1/
UPSTREAM_API_KEY=sk-...
UPSTREAM_MODEL=openai/gpt-oss-120b
LOG_LEVEL=INFO
CONTENT_FILTER_MODE=off
```

## Testing

### Test Coverage
- 243 tests passing
- 2 pre-existing failures (unrelated loop detection bugs)
- 38 skipped (live integration tests)

### Key Test Files
- `tests/test_xml_parser.py`: XML parsing, CDATA, self-closing tags, fallback strategies
- `tests/test_content_filter.py`: Pattern matching, sanitization, caching
- `tests/test_roundtrip.py`: End-to-end XML ↔ JSON conversion
- `tests/test_text_synthesis.py`: Text-based tool call synthesis

### Running Tests
```bash
cd /root/toolproxy
python3 -m pytest tests/ --tb=short -q
```

## Known Issues

### 1. Loop Detection False Positives
- **Location:** `app/services/loop_detection.py`
- **Problem:** Success words in error messages trigger loop detection
- **Status:** Known bug, not yet fixed
- **Tests:** `test_bug_proof.py::TestLoopDetectionFalsePositive`

### 2. Model Context Staleness
- **Problem:** Model sometimes tries to edit files with outdated content
- **Cause:** Model doesn't re-read files before editing
- **Solution:** Prompt model to read file first, or restart task

## Deployment

### Docker Setup
```bash
cd /root/toolproxy
docker compose up --build -d
```

### Health Check
```bash
curl http://localhost:8007/health
```

### Logs
```bash
docker compose logs -f toolproxy
```

## File Structure

```
/root/toolproxy/
├── app/
│   ├── main.py                    # FastAPI app, request pipeline
│   ├── config.py                  # Pydantic settings
│   └── services/
│       ├── xml_parser.py          # XML ↔ JSON conversion
│       ├── tool_call_fixups.py    # Post-parse corrections
│       ├── tool_mapping.py        # Tool name aliases, schema remapping
│       ├── content_filter.py      # OCI safety filter prevention
│       ├── vllm_client.py         # Upstream LLM client
│       ├── loop_detection.py      # Repetitive tool loop detection
│       ├── message_normalizer.py  # Message format normalization
│       ├── priming.py             # Few-shot example injection
│       └── text_synthesis.py      # Text-based tool call synthesis
├── tests/
│   ├── test_xml_parser.py
│   ├── test_content_filter.py
│   ├── test_roundtrip.py
│   ├── test_text_synthesis.py
│   └── test_bug_proof.py
├── .env                           # Environment configuration
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

## Request Pipeline (main.py)

1. Client detection (OpenCode/OpenClaw/RooCode/Cline)
2. Tool canonicalization
3. System prompt injection (XML format instructions)
4. Message normalization
5. Priming (few-shot examples)
6. Loop detection
7. Content filter (pre-request)
8. Upstream LLM call
9. XML parsing
10. JSON fallback parsing
11. Tool call fixups (edit nesting, move_file conversion, etc.)
12. Schema remapping
13. Multi-call limit (keep only first tool call)
14. Response to client

## Recent Fixes (2026-06-17)

1. **Nested edit deduplication**: Fixed logic to only remove inner blocks smaller than outer blocks
2. **Edit field names**: Changed output from `oldString`/`newString` to `oldText`/`newText` for OpenClaw compatibility
3. **Content filter mode**: Set to `off` to reduce noise from validation errors in history
4. **XML escaping**: Extended raw text tags to include edit-related fields
5. **Duplicate detection**: Added exact duplicate removal for identical tool calls
6. **Heredoc/shell-redirect in `<command>`**: Added `command` to `_RAW_TEXT_TAGS` so
   shell commands containing bare `<`/`>` (e.g. `python3 - <<'PY' ... PY`, `sort < in.txt`)
   are escaped before XML parsing. Previously these broke `ET.fromstring`, the tool call
   was dropped, and the raw XML leaked to the client (OpenClaw saw `<exec>...</exec>NO_REPLY`).
7. **`_strip_whitespace_in_tags` lookahead**: Added `(?=[\s/>])` lookahead so the tag-name
   whitespace fixer only matches real tags — no longer corrupts shell redirects like
   `sort < in.txt` → `sort <in.txt`.
8. **`NO_REPLY` sentinel**: Added `NO_REPLY\b` to `_LEAK_PATTERN` to strip the trailing
   sentinel some models append after a tool call.

## Performance

- Average request latency: ~2-5 seconds (depends on upstream LLM)
- XML parsing: <10ms
- Content filter: <5ms
- Memory usage: ~200-300MB

## Security

- Shell injection protection via `shlex.quote()`
- No secrets in logs
- Content filter prevents OCI safety violations
- Input validation on all tool calls

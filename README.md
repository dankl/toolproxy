# toolproxy

Generischer Tool-Call-Proxy für das OCI-Modell (`gpt-oss-120b`). Übersetzt OpenAI-native `tool_calls` in XML-Format und zurück — für jeden OpenAI-kompatiblen Client (Roo Code, opencode, etc.).

## Warum?

Das OCI-Modell (`gpt-oss-120b`) unterstützt native OpenAI `tool_calls` nicht zuverlässig. Es produziert XML-Tool-Calls jedoch stabil, wenn die Conversation-History XML enthält. Dieser Proxy erledigt die Übersetzung transparent — der Client sieht immer Standard-OpenAI-Format, das Modell immer XML.

## Flow

```
Client (Roo Code / opencode / curl)
  ↓  OpenAI JSON: messages + tools[]
toolproxy :8007
  ├─ System-Prompt: XML-Tool-Definitionen ergänzen
  ├─ History normalisieren: tool_calls + role:tool → XML
  ├─ Priming injizieren (nur 1. Turn)
  ↓  Plain text request ohne tools[]
oci-proxy :8005 → Oracle OCI GenAI
  ↓  XML tool call in response
toolproxy
  ├─ XML parsen → OpenAI tool_calls (primär)
  ├─ Partial XML rescue (abgeschnittene Responses)
  └─ JSON-Fallback-Kaskade (wenn kein XML gefunden)
  ↑  Standard OpenAI response mit tool_calls[]
Client
```

## Kernmechanismen

### 1. XML System-Prompt
Die `tools[]`-Definitionen aus dem Request werden in XML-Beispiele umgewandelt und an den bestehenden System-Prompt angehängt:

```xml
<read_file>
  <path>Pfad zur Datei (required)</path>
</read_file>
```

### 2. Priming (erster Turn)
Beim ersten Turn (nur 1 User-Message) werden bis zu 3 synthetische Beispiel-Paare vorangestellt. Das Modell imitiert Formate, die es in der History sieht — mehr Beispiele = stärkerer Format-Prior, unterdrückt den Chat-Template-Leak des Modells (`[assistant to=...]`).

```
[User]      "What files are in this project?"
[Assistant] <read_file><path>README.md</path></read_file>
[User]      "Please write the implementation."
[Assistant] <write_to_file><path>README.md</path></write_to_file>
[User]      "Are you done?"
[Assistant] <attempt_completion><result>example</result></attempt_completion>
[User]      <echter Request>
```

Tools werden in Priorität `read_file` → `write_to_file` → `attempt_completion` gewählt; fehlende werden durch andere verfügbare Tools ersetzt.

### 3. History-Normalisierung
Damit das Modell in jedem Turn konsistentes XML sieht, werden eingehende Nachrichten normalisiert:

| Eingehend | Wird zu |
|---|---|
| `role: "tool"` (Tool-Ergebnis) | `role: "user"` + `[Tool Result]\n{content}` |
| `role: "assistant"` + `tool_calls[]` | `role: "assistant"` + content als XML |
| Content-Array (tool_use-Blöcke) | Flach-Text mit XML-Darstellung |

### 4. XML-Parsing + Fallbacks
Reihenfolge beim Parsen der Modell-Antwort:
1. **XML** (primär) — regulärer Ausdruck über bekannte Tool-Namen
2. **Partial XML rescue** — `<path>`/`<content>` extrahieren auch ohne schließenden Tag
3. **Tool-Name-Aliasing** — halluzinierte Namen korrigieren (`write_file` → `write_to_file`)
4. **JSON-Kaskade** (Fallback) — `[Tool Call:]`, `### TOOL_CALL`, bare JSON, Code-Blöcke
5. **Text-Synthesis** — Prosa-Antwort → `write_to_file` oder `attempt_completion`

## Konfiguration

| Env-Var | Default | Beschreibung |
|---|---|---|
| `UPSTREAM_URL` | `http://oci-proxy:8005/v1` | Upstream API-URL |
| `UPSTREAM_MODEL` | `openai/gpt-oss-120b` | Modell-Name für Upstream |
| `UPSTREAM_API_KEY` | `dummy-key` | API-Key für Upstream |
| `REQUEST_TIMEOUT` | `180` | Timeout in Sekunden |
| `MAX_RETRIES` | `2` | Wiederholungen bei Timeout |
| `LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` |

## Tool-Name-Aliasing

Das Modell halluziniert gelegentlich Tool-Namen. Bekannte Mappings:

| Modell-Output | Echter Tool-Name |
|---|---|
| `write_file` | `write_to_file` |
| `create_file` | `write_to_file` |
| `open_file` | `read_file` |
| `view_file` | `read_file` |
| `read_file_content` | `read_file` |
| `apply_patch` | `apply_diff` |
| `patch_file` | `apply_diff` |
| `list_dir` / `ls` / `list_directory` | `list_files` |

## Integration

### Roo Code
Provider-Settings → Base URL: `http://localhost:8007`

Wichtig: Roo Code muss auf **native tool_calls** konfiguriert sein (Standard ab v3.36). Der Proxy übersetzt diese intern zu XML.

**Empfohlener Mode: Code**

Das Modell ignoriert Roo-Code-Mode-Instruktionen (Architect, Ask, etc.) und folgt ausschließlich dem XML-System-Prompt des Proxys. Architect-Mode hat dieselben Tools verfügbar wie Code-Mode — Roo Code trennt die Modi nur über System-Prompt-Instruktionen ("denke als Architekt"), nicht über eingeschränkte Tool-Sets. Da das Modell diese kontextuellen Instruktionen nicht zuverlässig befolgt, verhält es sich in jedem Mode wie Code-Mode und versucht direkt Dateien zu schreiben.

→ Immer **Code-Mode** verwenden. Architect/Ask bringen bei diesem Modell keine andere Wirkung.

### opencode
```json
{
  "model": "toolproxy/gpt-oss-120b",
  "baseURL": "http://localhost:8007/v1"
}
```

### Direkt via LiteLLM (empfohlen)
```yaml
# litellm/config.yaml
- model_name: openai/gpt-oss-120b-tools
  litellm_params:
    model: openai/gpt-oss-120b
    api_base: http://toolproxy:8007/v1
    api_key: dummy-key
```

## Starten

```bash
# Container bauen und starten
docker compose up --build toolproxy

# Health-Check
curl http://localhost:8007/health

# Test
curl -X POST http://localhost:8007/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss-120b",
    "messages": [{"role": "user", "content": "Read the file README.md"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "read_file",
        "parameters": {
          "type": "object",
          "properties": {"path": {"type": "string", "description": "File path"}},
          "required": ["path"]
        }
      }
    }]
  }'
```

## Tests

Unit- und Integrationstests liegen in `tests/`. Sie mocken den Upstream komplett — kein laufender Container nötig.

```bash
# Abhängigkeiten installieren (einmalig)
pip install -r requirements-test.txt

# Alle Tests ausführen
python3 -m pytest -v

# Nur eine Test-Klasse
python3 -m pytest -v tests/test_roundtrip.py::TestXmlToolCalls
```

Aktuelle Test-Coverage: ~52 Tests in `test_roundtrip.py` und `test_xml_parser.py`.

## Bekannte Modell-Eigenheiten & Proxy-Fixes

### apply_diff für neue Dateien → write_to_file (implementiert)
Das Modell verwendet `apply_diff` mit ausschließlich `+`-Zeilen um neue Dateien zu erstellen — Roo Code kann das nicht verarbeiten. Der Proxy erkennt "all-additions diffs" automatisch und konvertiert sie zu `write_to_file`.

Im Log sichtbar als:
```
INFO  [abc12345] apply_diff all-additions on 'src/Main.java' → write_to_file
```

### Mehrere Tool Calls pro Turn → Limit 1 (implementiert)
Das Modell gibt oft 3–5 Tool Calls in einem einzigen Response zurück. Roo Code verarbeitet parallele Calls nicht zuverlässig. Der Proxy gibt immer nur den **ersten** Call zurück; die restlichen werden verworfen. Das erzwingt sequenzielle Ausführung: ein Schritt → Ergebnis → nächster Schritt.

Im Log sichtbar als:
```
INFO  [abc12345] 3 tool calls → keeping only first ('write_to_file')
```

### Chat-Template-Leak ([assistant to=...] Format)
Das OCI-Modell gibt gelegentlich seinen internen Chat-Template-Header aus statt XML:
- `[assistant to=write_file code<|message|>{...}` — FALSCH
- `<assistant to=write_file code>{...}` — FALSCH

Unterdrückt durch: XML-Instruktion VOR dem System-Prompt + 3 Priming-Turns + explizite FORBIDDEN FORMATS Sektion.

### Modell plant nicht mehrstufig
Das Modell neigt dazu, nach 1–2 Datei-Writes sofort `attempt_completion` aufzurufen statt alle Dateien zu schreiben. Adressiert durch Prompt-Regel:
> *"A task is ONLY complete when ALL required files exist on disk. Write every file with write_to_file first, THEN call attempt_completion."*

### Sonderzeichen in Datei-Inhalten (implementiert)
Wenn das Modell Dateien mit Code-Inhalten schreibt, enthält `<content>` oft XML-ungültige Zeichen:

| Zeichen | Kontext | Beispiel |
|---|---|---|
| `&&` | Shell-Befehle | `npm install && pip install` |
| `&` | Standalone-Operator | `Linter & Formatter` |
| `=>` | JavaScript Arrow Functions | `(req, res) => { ... }` |

Der XML-Parser schlägt fehl, wenn solche Zeichen unescaped im `<content>`-Tag auftreten. `fix_xml_string` (in `services/xml_parser.py`) pre-escaped diese Zeichen automatisch vor dem Parsen. Betrifft die Tags `content`, `diff`, `result` und `output`.

### Abgeschnittene XML-Responses — Partial XML rescue (implementiert)
Das Modell gibt gelegentlich `<write_to_file>`-Responses zurück, die keinen schließenden `</write_to_file>`-Tag haben (Response wird vom Upstream abgeschnitten). Der reguläre Ausdruck findet dann keine Übereinstimmung, und es gibt keine `WARNING`-Logzeile — der Response landet ohne Fix in `attempt_completion`.

Der Partial XML rescue extrahiert `<path>` und `<content>` unabhängig vom schließenden Tag:

```python
re.search(
    r"<write(?:_to)?_file\b[^>]*>\s*<path>(.*?)</path>\s*<content>([\s\S]+?)(?:</content>|$)",
    stripped,
    re.IGNORECASE,
)
```

Greift auch wenn der Tag als `<write_file>` (Alias) beginnt.

Im Log sichtbar als:
```
INFO  [abc12345] Partial XML rescue → write_to_file('Plan.md')
```

**Diagnose-Tipp**: Wenn das Modell eine Datei nicht erstellt und im Log kein `WARNING` vor der `Text response → synthesizing`-Zeile steht, war der Response wahrscheinlich abgeschnitten.

## Bekannte Einschränkungen & Potenzielle Verbesserungen

### Context-Truncation (nicht implementiert)
`roocode-proxy` kürzt bei sehr langen Conversations den Context automatisch (System-Prompt + letzte N Messages behalten, Mitte durch Summary ersetzen). Beim 20B-Modell war das mit 64k Kontext nötig.

Ob das 120B-Modell das gleiche Problem hat, ist noch unbekannt. Falls Sessions mit vielen Tool-Calls abbrechen oder das Modell den Faden verliert → Context-Truncation aus `roocode-proxy/app/main.py:truncate_context()` übernehmen.

### Text-Synthesis (implementiert)
Wenn das Modell statt einem Tool-Call reine Prosa antwortet, würde Roo Code mit `[ERROR] You did not use a tool` abbrechen. toolproxy fängt das ab:

1. Langer Text der wie File-Content aussieht (`##`, ` ``` `, `---`) + Zieldatei aus VSCode Open Tabs erkennbar → `write_to_file(path, content)`
2. Fallback → `attempt_completion(result=content)`

Im Log sichtbar als:
```
INFO  [abc12345] Text response looks like file content → synthesizing write_to_file('src/main.py')
INFO  [abc12345] Text response → synthesizing attempt_completion fallback
```

### Success-Loop-Detection (implementiert)
Nach 2+ erfolgreichen Write-Operationen in Folge injiziert der Proxy einen Stop-Hint in die letzte User-Message, bevor das Modell antwortet. Verhindert dass das Modell dasselbe Tool nochmal aufruft und Duplikate erzeugt.

Im Log sichtbar als:
```
WARNING [abc12345] SUCCESS LOOP: 2 consecutive successful write operations — injecting stop hint
```

`apply_diff`-Similarity-Loops (gleicher Fehler 2x) noch nicht implementiert — erst Live-Erfahrung abwarten.

---

## Logs verstehen

```
INFO  [abc12345] model=gpt-oss-120b messages=3 tools=12
INFO  [abc12345] XML parsed 1 tool call(s)                               ← Normalfall
INFO  [abc12345] XML alias: 'write_file' → 'write_to_file'               ← Tool-Name-Aliasing
INFO  [abc12345] Partial XML rescue → write_to_file('Plan.md')           ← Abgeschnittener Response
INFO  [abc12345] JSON fallback: [Tool Call:] → read_file                 ← JSON-Fallback
INFO  [abc12345] apply_diff all-additions on 'Main.java' → write_to_file ← Diff-Konvertierung
INFO  [abc12345] 3 tool calls → keeping only first ('write_to_file')     ← Multi-Call Limit
INFO  [abc12345] Text response looks like file content → synthesizing write_to_file(...)  ← Text-Synthesis
INFO  [abc12345] Text response → synthesizing attempt_completion fallback ← Synthesis-Fallback
INFO  [abc12345] No tool calls found — returning text response           ← Reine Textantwort
WARNING [abc12345] SUCCESS LOOP: 2 consecutive successful write operations — injecting stop hint
```

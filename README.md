# toolproxy

Tool-Call-Proxy für OpenAI-kompatible Sprachmodelle ohne nativen Function-Call-Support. Übersetzt OpenAI-native `tool_calls` in XML-Format und zurück — für jeden OpenAI-kompatiblen Client (Roo Code, Cline, opencode, etc.).

## Getestet mit

| Modell | Anmerkung |
|---|---|
| `gpt-oss-120b` | Primäres Entwicklungsmodell; Chat-Template-Leak-Verhalten bekannt und abgefangen |

Das XML-Format und alle Fallbacks wurden auf Basis dieses Modells entwickelt. Andere Modelle ohne native Function-Call-Unterstützung sollten funktionieren, können aber andere Eigenheiten zeigen.

## Warum?

Viele Modelle unterstützen native OpenAI `tool_calls` nicht zuverlässig — sei es wegen fehlender Trainings-Daten oder Chat-Template-Eigenheiten. Solche Modelle produzieren XML-Tool-Calls jedoch stabil, wenn die Conversation-History XML enthält. Dieser Proxy erledigt die Übersetzung transparent — der Client sieht immer Standard-OpenAI-Format, das Modell immer XML.

## Flow

```
Client (Roo Code / Cline / opencode / curl)
  ↓  OpenAI JSON: messages + tools[]
toolproxy :8007
  ├─ Client erkennen (Roo Code / Cline / OpenCode / Generic)
  ├─ Tool-Namen kanonisieren (OpenCode: write→write_to_file, read→read_file)
  ├─ System-Prompt: XML-Tool-Definitionen ergänzen
  ├─ History normalisieren: tool_calls + role:tool → XML
  ├─ Priming injizieren (nur 1. Turn)
  ├─ Loop-Detection: Repetitive-Loop → dann Success-Loop (CORRECTION-Hint)
  ├─ Truncation-Reminder injizieren (wenn letztes Tool-Result truncated)
  ↓  Plain text request ohne tools[]
Upstream LLM (OpenAI-kompatibler Endpoint)
  ↓  XML tool call in response (immer kanonische Namen)
toolproxy
  ├─ XML parsen → OpenAI tool_calls (primär)
  ├─ Partial XML rescue (abgeschnittene Responses)
  ├─ JSON-Fallback-Kaskade (wenn kein XML gefunden)
  ├─ Text-Synthesis (Prosa → write_to_file / attempt_completion)
  ├─ Write Guard: Markdown in Config-Datei → ask_followup_question
  ├─ Schema-Remap: <path> → filePath (OpenCode)
  └─ Tool-Namen dekanonisieren (write_to_file→write, read_file→read für OpenCode)
  ↑  Standard OpenAI response mit tool_calls[]
Client
```

## Tool-Mapping: Client ↔ LLM (kanonisch)

Das Modell spricht immer **kanonische Tool-Namen** (ROO-Code-Stil). Client-spezifische Namen werden beim Eingang übersetzt und beim Ausgang zurückübersetzt — transparent für beide Seiten.

| OpenCode (Client) | LLM (kanonisch) | Roo Code (Client) | Cline (Client) | Mapping |
|---|---|---|---|---|
| `read` | `read_file` | `read_file` | `read_file` | ✅ OpenCode übersetzt |
| `write` | `write_to_file` | `write_to_file` | `write_to_file` | ✅ OpenCode übersetzt |
| `list` | `list_files` | `list_files` | `list_files` | ✅ OpenCode übersetzt |
| `bash` | `bash` | *(kein Äquivalent)* | *(kein Äquivalent)* | — passthrough |
| `edit` | `edit` | *(kein Äquivalent)* | *(kein Äquivalent)* | — passthrough |
| `glob` | `glob` | *(kein Äquivalent)* | *(kein Äquivalent)* | — passthrough |
| `grep` | `grep` | *(kein Äquivalent)* | *(kein Äquivalent)* | — passthrough |
| *(kein Äquivalent)* | `apply_diff` | `apply_diff` | *(kein Äquivalent)* | — nur Roo Code |
| *(kein Äquivalent)* | `replace_in_file` | *(kein Äquivalent)* | `replace_in_file` | — nur Cline |
| *(kein Äquivalent)* | `attempt_completion` | `attempt_completion` | `attempt_completion` | — passthrough |
| *(kein Äquivalent)* | `execute_command` | `execute_command` | `execute_command` | — passthrough |
| *(kein Äquivalent)* | `search_files` | `search_files` | `search_files` | — passthrough |
| *(kein Äquivalent)* | `ask_followup_question` | `ask_followup_question` | `ask_followup_question` | — passthrough |
| *(kein Äquivalent)* | `browser_action` | *(kein Äquivalent)* | `browser_action` | — nur Cline |
| *(kein Äquivalent)* | `use_mcp_tool` | *(kein Äquivalent)* | `use_mcp_tool` | — nur Cline |
| *(kein Äquivalent)* | `new_task` | `new_task` | `new_task` | — passthrough |

**Cline-Erkennung:** `replace_in_file` ∈ Tools AND `apply_diff` ∉ Tools → CLINE. Cline-Tool-Namen sind identisch mit den kanonischen Namen — kein De-/Kanonisierungs-Schritt nötig.

**Parameter-Mapping** (zusätzlich, via Schema-Remap):

| LLM-Output (kanonisch) | OpenCode-Schema | Roo-Schema |
|---|---|---|
| `<path>` in `write_to_file` | → `file_path` oder `filePath` (erste Übereinstimmung) | → `path` |

Im Log sichtbar als:
```
INFO  [abc12345] Decanonicalize: 'write_to_file' → 'write'
INFO  [abc12345] Schema remap write_to_file: 'path' → 'file_path'
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

**requires-Gating**: Jede Static Sequence ist mit den Tools annotiert, die sie voraussetzt. Eine Sequence wird nur injiziert wenn alle required Tools im `tools[]`-Array der Session vorhanden sind. Dadurch zahlen Minimal-Setups (z.B. nur `read_file` + `attempt_completion`) nicht den Token-Preis von `apply_diff`- oder `execute_command`-Beispielen. Das Priming skaliert proportional zum tatsächlich verfügbaren Tool-Set.

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
| `UPSTREAM_URL` | `http://localhost:8000/v1` | Upstream API-URL (OpenAI-kompatibel) |
| `UPSTREAM_MODEL` | `your-model-name` | Modell-Name für Upstream |
| `UPSTREAM_API_KEY` | `dummy-key` | API-Key für Upstream |
| `REQUEST_TIMEOUT` | `180` | Timeout in Sekunden |
| `MAX_RETRIES` | `2` | Wiederholungen bei 5xx-Fehlern |
| `RETRY_ON_TIMEOUT` | `false` | Timeout-Retries (Standard: aus — OCI-Timeouts sind kein transientes Problem, Retries multiplizieren nur die Wartezeit) |
| `LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` |

## Tool-Name-Aliasing

Das Modell halluziniert gelegentlich Tool-Namen. Bekannte Mappings:

| Modell-Output | Echter Tool-Name |
|---|---|
| `write_file`, `create_file` | `write_to_file` |
| `open_file`, `view_file`, `read_file_content` | `read_file` |
| `apply_patch`, `patch_file`, `patch` | `apply_diff` |
| `list_dir`, `ls`, `list_directory` | `list_files` |
| `search`, `find`, `search_code` | `search_files` |
| `search_codebase`, `semantic_search` | `codebase_search` |
| `search_and_replace`, `search_replace`, `find_replace` | `edit` |
| `replace_in_file` | `replace_in_file` (Cline) oder `edit` (Roo Code Fallback) |
| `rename`, `move` | `move_file` → via Fixup zu `execute_command(mv)` |
| `bash`, `run_command`, `execute`, `run`, `shell` | `execute_command` |
| `ask_followup`, `ask_question`, `ask_user`, `followup_question` | `ask_followup_question` |
| `create_task`, `task`, `subtask` | `new_task` |
| `read_output`, `get_output`, `command_output` | `read_command_output` |
| `use_skill`, `run_skill` | `skill` |
| `change_mode`, `set_mode`, `mode` | `switch_mode` |
| `update_todos`, `update_todo`, `todo`, `set_todos` | `update_todo_list` |

## Integration

### Roo Code
Provider-Settings → Base URL: `http://localhost:8007`

Wichtig: Roo Code muss auf **native tool_calls** konfiguriert sein (Standard ab v3.36). Der Proxy übersetzt diese intern zu XML.

**Empfohlener Mode: Code**

Das Modell ignoriert Roo-Code-Mode-Instruktionen (Architect, Ask, etc.) und folgt ausschließlich dem XML-System-Prompt des Proxys. Architect-Mode hat dieselben Tools verfügbar wie Code-Mode — Roo Code trennt die Modi nur über System-Prompt-Instruktionen ("denke als Architekt"), nicht über eingeschränkte Tool-Sets. Da das Modell diese kontextuellen Instruktionen nicht zuverlässig befolgt, verhält es sich in jedem Mode wie Code-Mode und versucht direkt Dateien zu schreiben.

→ Immer **Code-Mode** verwenden. Architect/Ask bringen bei diesem Modell keine andere Wirkung.

### Cline

Provider-Settings → Base URL: `http://localhost:8007`
API Key: beliebig (z.B. `dummy-key`)

Cline verwendet `replace_in_file` mit SEARCH/REPLACE-Blöcken (statt `apply_diff` wie Roo Code). Der Proxy erkennt Cline automatisch und instruiert das Modell entsprechend:

```xml
<replace_in_file>
  <path>src/foo.py</path>
  <diff>
<<<<<<< SEARCH
def old_function():
=======
def new_function():
>>>>>>> REPLACE
  </diff>
</replace_in_file>
```

### opencode
```json
{
  "model": "toolproxy/your-model-name",
  "baseURL": "http://localhost:8007/v1"
}
```

### Direkt via LiteLLM (empfohlen)
```yaml
# litellm/config.yaml
- model_name: openai/your-model-tools
  litellm_params:
    model: openai/your-model-name
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
    "model": "your-model-name",
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

Aktuelle Test-Coverage: 192 Unit-Tests gesamt (ohne Live-Tests).

**Live E2E-Tests** (laufen gegen echtes Modell, kein Mock):
```bash
# oci-proxy muss auf localhost:8015 erreichbar sein (oder per SSH-Tunnel)
python3 -m pytest -m live -v

# Anderen Endpoint:
TOOLPROXY_UPSTREAM_URL=http://localhost:8005/v1 python3 -m pytest -m live -v
```

Live-Tests decken ab: Write Guard, Repetitive-Loop-Detection, Timeout-Verhalten, Truncation-Halluzination, apply_diff-Loop, alle Roo-Code- und OpenCode-Tools.

## Bekannte Modell-Eigenheiten & Proxy-Fixes

### move_file / rename_file → execute_command(mv) (implementiert)

`move_file` und `rename_file` sind Roo Code Builtins, die **nicht** im `tools[]`-Array erscheinen das Roo Code an die API schickt. Roo Code 3.51.1 dropped tool_calls für unbekannte Tools lautlos — kein Approval-Dialog, kein Fehler, nur "no assistant messages" nach Timeout.

Der Proxy fängt das in Step 5c ab: `move_file(source=X, destination=Y)` → `execute_command(command="mv 'X' 'Y'")`. `execute_command` ist immer in `tools[]` → Roo Code zeigt den normalen Approval-Dialog. Funktioniert für Dateien und Ordner gleichermaßen.

Das Priming lehrt das Modell direkt `execute_command` für Rename-Aufgaben zu nutzen; der Fallback greift wenn das Modell trotzdem `move_file` ausgibt.

Im Log sichtbar als:
```
INFO  [abc12345] Convert move_file → execute_command: "mv 'old_folder' 'new_folder'"
```

### apply_diff für neue Dateien → write_to_file (implementiert)
Das Modell verwendet `apply_diff` mit ausschließlich `+`-Zeilen um neue Dateien zu erstellen — Roo Code kann das nicht verarbeiten. Der Proxy erkennt "all-additions diffs" automatisch und konvertiert sie zu `write_to_file`.

Im Log sichtbar als:
```
INFO  [abc12345] apply_diff all-additions on 'src/Main.java' → write_to_file
```

### apply_diff: Unified-Diff → SEARCH/REPLACE Konvertierung (implementiert)

Das Modell gibt in `apply_diff` manchmal Unified-Diff-Format aus statt des von Roo Code erwarteten SEARCH/REPLACE-Formats — besonders wenn der User-Prompt selbst einen `--- a/file / +++ b/file / @@ ... @@`-Diff enthält. Der Proxy erkennt das und konvertiert automatisch:

```
@@ -1 +1 @@                    <<<<<<< SEARCH
-pritn('hello')        →        pritn('hello')
+print('hello')                 =======
                                print('hello')
                                >>>>>>> REPLACE
```

Regeln:
- `---`/`+++`-Headerzeilen werden übersprungen
- Je `@@`-Hunk → ein SEARCH/REPLACE-Block (auch bare `@@` ohne Koordinaten)
- `-`-Zeilen → SEARCH, `+`-Zeilen → REPLACE
- Kontextzeilen (Leerzeichen-Prefix) erscheinen in beiden Hälften
- Mehrere Hunks → mehrere SEARCH/REPLACE-Blöcke

Wenn das Diff weder `>>>>>>> REPLACE` noch `@@`-Marker enthält (wirklich korrupt/abgeschnitten) → wird wie bisher gedroppt.

Im Log sichtbar als:
```
INFO  [abc12345] Converted unified diff → SEARCH/REPLACE for apply_diff (142 chars)
```

### Mehrere Tool Calls pro Turn → Limit 1 (implementiert)
Das Modell gibt oft 3–5 Tool Calls in einem einzigen Response zurück. Roo Code verarbeitet parallele Calls nicht zuverlässig. Der Proxy gibt immer nur den **ersten** Call zurück; die restlichen werden verworfen. Das erzwingt sequenzielle Ausführung: ein Schritt → Ergebnis → nächster Schritt.

Adressiert durch:
- **Rule 1** im System-Prompt: *"NEVER output multiple tool calls in one response. Always wait for the tool result before calling the next tool."*
- **WRONG/CORRECT-Beispiel** im System-Prompt zeigt explizit, dass Batching verboten ist
- **Limit-Step** im Proxy (Step 11 in `main.py`): überschüssige Calls werden still verworfen

Im Log sichtbar als:
```
INFO  [abc12345] 3 tool calls → keeping only first ('write_to_file')
```

### Chat-Template-Leak ([assistant to=...] Format)
Das Modell gibt gelegentlich seinen internen Chat-Template-Header aus statt XML:
- `[assistant to=write_file code<|message|>{...}` — FALSCH
- `<assistant to=write_file code>{...}` — FALSCH

Unterdrückt durch: XML-Instruktion VOR dem System-Prompt + 3 Priming-Turns + explizite FORBIDDEN FORMATS Sektion.

### Modell plant nicht mehrstufig
Das Modell neigt dazu, nach 1–2 Datei-Writes sofort `attempt_completion` aufzurufen statt alle Dateien zu schreiben. Adressiert durch Prompt-Regel:
> *"A task is ONLY complete when ALL required files exist on disk. Write every file with write_to_file first, THEN call attempt_completion."*

### Sonderzeichen in Datei-Inhalten (implementiert)
Wenn das Modell Dateien mit Code-Inhalten schreibt, enthält `<content>` oft XML-ungültige oder -problematische Zeichen:

| Zeichen / Muster | Kontext | Beispiel |
|---|---|---|
| `&&` | Shell-Befehle | `npm install && pip install` |
| `&` | Standalone-Operator | `Linter & Formatter` |
| `=>` | JavaScript Arrow Functions | `(req, res) => { ... }` |
| `<div>`, `<h1>`, ... | JSX / HTML in React-Dateien | `return (<div>...</div>)` |
| `<<<<<<< SEARCH` | apply_diff SEARCH/REPLACE-Marker | Roo-Code/Cline-Diffs |

`fix_xml_string` (in `services/xml_parser.py`) escaped die Inhalte der Tags `content`, `diff`, `result` und `output` **vor** dem ET-Parsing — nicht erst bei einem `ParseError`. Das verhindert, dass zufällig valides XML (z.B. JSX-Struktur `<div><h1>...</h1></div>`) fälschlicherweise als XML-Kind-Elemente geparst wird und `content` als dict statt als String landet.

### apply_diff ohne schließenden `</diff>`-Tag (implementiert)
Das Modell öffnet `<diff>` korrekt, schließt es aber gelegentlich nicht — es fehlt `</diff>` vor `</apply_diff>`. Dadurch findet `fix_xml_string` keinen Match für den `<diff>`-Block, die `<<<<<<< SEARCH`-Marker bleiben unescaped, und ET bricht mit `not well-formed (invalid token)` ab.

`fix_xml_string` erkennt diesen Fall und fügt das fehlende `</diff>` automatisch ein, **bevor** das Escaping läuft — so greift der Escape-Regex danach korrekt.

Im Log sichtbar als (nach Fix: kein Warning mehr):
```
WARNING [abc12345] Failed to parse XML tool call: not well-formed (invalid token): line 4, column 1
```

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

### Loop-Detection (implementiert)

Zwei Detektoren, die einen CORRECTION-Hint in die letzte User-Message injizieren. **Reihenfolge:** Repetitive-Loop wird zuerst geprüft, Success-Loop als Fallback — so bekommt ein `apply_diff`-Loop die präzisere Hint ("not making progress, use write_to_file") statt der irreführenden ("already succeeded, call attempt_completion").

**Repetitive-Loop** (Threshold 3): Dasselbe Tool mit demselben Pfad 3× in Folge → das Modell dreht sich im Kreis ohne Fortschritt. Hint: *"use write_to_file with complete content or read_file to verify"*.

**Success-Loop** (Threshold 2, Fallback): 2+ aufeinanderfolgende erfolgreiche Write-Operationen ohne echten User-Turn → das Modell schreibt Duplikate. Hint: *"operation reported success N times but task may not be complete — verify with read_file or write_to_file"*.

Ein echter User-Turn (keine `[Tool Result]`-Nachricht) setzt beide Zähler zurück — **außer** der letzte User-Turn ist der aktuelle Prompt (die Nachricht die gerade beantwortet wird). Der aktuelle Prompt wird aus dem Scan ausgeschlossen, damit ein „bitte nochmal versuchen" die aufgebaute Streak nicht zurücksetzt.

**OpenAI-Format in History**: Roo Code schickt vergangene Runden als OpenAI-`tool_calls`-Objekte (nicht als XML in `content`). Die Loop-Detection erkennt beide Formate: XML im `content`-Feld (neue Turns) **und** `tool_calls`-Strukturen in der Conversation-History (vergangene Turns). Ohne diesen Fix wurden Roo-Code-Loops aus der History unsichtbar, und der Zähler blieb immer bei 0.

Im Log sichtbar als:
```
WARNING [abc12345] REPETITIVE LOOP: tool='apply_diff' called 3× in a row — injecting correction hint
WARNING [abc12345] SUCCESS LOOP: 2 consecutive successful write operations — injecting stop hint
```

### Doppelte Tool-Call-IDs (implementiert)

Das Modell wiederholt identische Tool-Calls (gleicher Name, gleiche Argumente) wenn es in einer Schleife steckt — z.B. mehrfaches `pkill` nach bereits gestoppten Prozessen. Die ursprüngliche Implementierung verwendete einen MD5-Hash über den XML-Content als ID, was bei identischem Content dieselbe ID erzeugte. Roo Code bricht mit "failure in the model's thought process" ab wenn es eine ID zum zweiten Mal sieht.

Fix: Jede Tool-Call-ID enthält jetzt einen kurzen Zufalls-Suffix (`secrets.token_hex(2)`) — IDs sind immer eindeutig, auch bei identischem Content.

Im Log sichtbar als:
```
INFO  [abc12345] XML parsed 1 tool call(s): execute_command   ← call_abc12345_execute_command_3f7a
INFO  [abc12345] XML parsed 1 tool call(s): execute_command   ← call_abc12345_execute_command_8b2c (different!)
```

### ask_followup_question follow_up: String → Array (implementiert)

Das Modell gibt `follow_up` gelegentlich als Newline-getrennten String statt als JSON-Array aus. Roo Code erwartet ein Array und bricht mit `"Missing value for required parameter 'follow_up'"` ab — der Nutzer sieht einen harten Roo-Code-Fehler.

Fix: `fix_ask_followup_question_params` (Step 11) splittet den String an Zeilenumbrüchen zu einem Array.

Beispiel:
```
Model output:  "follow_up": "Show output\nRestart server\nCheck config"
Proxy fixes:   "follow_up": ["Show output", "Restart server", "Check config"]
```

### Halluzinierter [Tool Result] nach write_to_file (implementiert)

Nach einem erfolgreichen `write_to_file` schreibt das Modell gelegentlich einen gefakten `[Tool Result]`-Block in seinen eigenen Response, anstatt auf das echte Tool-Ergebnis von Roo Code zu warten — und ruft danach `write_to_file` erneut auf. Das erzeugt eine Endlos-Schleife mit wechselndem Dateiinhalt.

Fix (Priming): Static Sequence im ROO_CODE-Priming zeigt das korrekte Muster:
1. `write_to_file` aufrufen
2. `[Tool Result]` kommt **immer aus der User-Turn** (nie selbst produzieren)
3. Nach erfolgreichem Write → `attempt_completion` aufrufen (nicht nochmals schreiben)

### [Tool Result]-Halluzination bei gekürzten Dateien (implementiert)

Roo Code kürzt große Dateien bei `read_file` mit einem spezifischen Format:
```
IMPORTANT: File content truncated.
Status: Showing lines 1-40 of 83 total lines.
To read more: Use the read_file tool with offset=41 and limit=30.
```

Das Modell kann dieses Format falsch interpretieren und `[Tool Result]`-Blöcke in seinen Response-Text schreiben statt ein echtes Tool aufzurufen. Roo Code bricht dann ab.

Fix (dreistufig, defense-in-depth):
1. **Priming** (Fix A): Static Sequence zeigt das exakte Roo Code-Truncation-Format und das korrekte Verhalten: truncated result → direkt `write_to_file` aufrufen
2. **System-Prompt FORBIDDEN FORMATS** (Fix B): `[Tool Result]` explizit verboten; klärt außerdem dass `read_file` mit `offset=` erlaubt ist um weitere Seiten zu lesen
3. **Dynamischer Truncation-Reminder** (Fix C, `_inject_truncation_reminder` in `main.py`): Wenn das letzte User-Message `"IMPORTANT: File content truncated"` enthält, wird ein `[REMINDER]` an die Nachricht angehängt bevor sie ans Modell geht — wirkt auch nach langen Konversationen wenn der Priming-Effekt nachlässt

### Write Guard (implementiert)

Wenn das Modell Markdown-Dokumentation in eine Config-Datei schreiben will (`application.yml`, `pom.xml`, `build.gradle`, etc.), fängt der Guard es ab — defence-in-depth:

1. **Priming-Regel #7** im System-Prompt verhindert es in der Regel direkt — das Modell wählt selbst eine `.md`-Datei
2. **Fallback** (z.B. wenn System-Prompt durch OCI gedroppt wird): Guard ersetzt den `write_to_file`-Call durch `ask_followup_question` — der User wird gefragt wo die Doku hin soll
3. Wenn `ask_followup_question` nicht im Tool-Set: WARNING im Log, Write passiert durch

Im Log sichtbar als:
```
WARNING [abc12345] WRITE GUARD: model tried to write markdown docs into 'application.yml' → replacing with ask_followup_question
```

### Leere XML Tool-Calls (implementiert)

Tool-Calls ohne Argumente (z.B. `<list_namespaces></list_namespaces>`) erzeugen nach dem XML-Parsing leere Strings statt leerer Dicts. `json.loads('""')` gibt `""` zurück — kein `dict`. Der Schema-Remap-Schritt rief dann `.items()` auf einem String auf und warf einen `AttributeError`.

Fix: `_remap_args_to_schema` prüft jetzt nach dem JSON-Parse ob das Ergebnis wirklich ein `dict` ist; andernfalls wird der Tool-Call unverändert durchgereicht.

### MCP-Tool-Hint bei fehlenden Tool-Definitionen (implementiert)

Generic-Clients (kein `tools[]` im Request, z.B. direkte API-Zugriffe) erhalten keinen Tool-Context. Das Modell gibt dann manchmal rohes XML aus — ohne `<use_mcp_tool>`-Wrapper — und bekommt keine Rückmeldung warum nichts passiert.

Fix: Wenn der Response rohes XML enthält **und** kein `tools[]` im Request war, hängt der Proxy automatisch einen Hinweis an:
```
[toolproxy] No tool definitions registered for this session.
To call MCP tools, use the <use_mcp_tool> wrapper:
<use_mcp_tool><server_name>SERVER</server_name><tool_name>TOOL</tool_name><arguments>{...}</arguments></use_mcp_tool>
```

Im Log sichtbar als:
```
INFO  [abc12345] Raw XML detected with tools=0 — injecting use_mcp_tool hint
```

### Kein Retry bei Timeout (implementiert)

Standard: `RETRY_ON_TIMEOUT=false`. Bei einem OCI-Timeout wird der Request nicht wiederholt — das Modell ist in dem Fall in der Regel hängengeblieben, kein transientes Netzwerkproblem. Mit dem alten Default (retry) wartete Roo Code bis zu 3 × 180s = 9 Minuten, bevor eine 502-Fehlermeldung kam. Jetzt kommt die 502 nach dem ersten Timeout.

Opt-in Retry für Endpoints mit bekannten transiente Timeouts: `RETRY_ON_TIMEOUT=true`.

---

## Logs verstehen

Ein normaler Request erzeugt genau 3 INFO-Zeilen:
```
INFO  [abc12345] model=your-model-name messages=3 tools=12 client=roo_code
INFO  [abc12345] model output (542 chars): <write_to_file>\n<path>foo.py</path>...
INFO  [abc12345] XML parsed 1 tool call(s): write_to_file
```

Anomalien und Fixes erscheinen als zusätzliche Zeilen:
```
INFO  [abc12345] XML alias: 'write_file' → 'write_to_file'               ← Tool-Name-Aliasing
INFO  [abc12345] Partial XML rescue → write_to_file('Plan.md')           ← Abgeschnittener Response
INFO  [abc12345] JSON fallback: [Tool Call:] → read_file                 ← JSON-Fallback
INFO  [abc12345] Convert move_file → execute_command: "mv 'src' 'dst'"    ← Rename/Move-Fix
INFO  [abc12345] apply_diff all-additions on 'Main.java' → write_to_file ← All-Additions-Konvertierung
INFO  [abc12345] Converted unified diff → SEARCH/REPLACE for apply_diff (142 chars) ← Unified-Diff-Fix
INFO  [abc12345] 3 tool calls → keeping only first ('write_to_file')     ← Multi-Call Limit
INFO  [abc12345] Text response looks like file content → synthesizing write_to_file(...)  ← Text-Synthesis
INFO  [abc12345] Text response → synthesizing attempt_completion fallback ← Synthesis-Fallback
INFO  [abc12345] Schema remap write_to_file: 'path' → 'filePath'         ← Parameter-Remap (OpenCode)
INFO  [abc12345] ask_followup_question: follow_up string → array (3 items) ← follow_up Fixup
INFO  [abc12345] Raw XML detected with tools=0 — injecting use_mcp_tool hint ← MCP-Hint
INFO  [abc12345] No tool calls found — returning text response           ← Reine Textantwort
WARNING [abc12345] SUCCESS LOOP: 2 consecutive successful write operations — injecting stop hint
WARNING [abc12345] REPETITIVE LOOP: tool='read_file' called 3× in a row — injecting correction hint
WARNING [abc12345] WRITE GUARD: model tried to write markdown docs into 'application.yml' → replacing with ask_followup_question
```

Nur bei `LOG_LEVEL=DEBUG` zusätzlich sichtbar:
```
DEBUG [abc12345] Extracted XML tool call: write_to_file
DEBUG [abc12345] Decanonicalize: 'write_to_file' → 'write'
DEBUG vllm_client — Payload: {...}
```

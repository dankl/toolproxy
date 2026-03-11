# Stabilitätsanalyse

Analysedatum: 2026-03-11
Autor: Claude (claude-sonnet-4-6)

---

## Architekturübersicht

```
Roo Code / OpenCode
      ↓  OpenAI-Format (tool_calls + tools[])
  toolproxy :8007
      ├─ Client-Typ erkennen (Roo Code / OpenCode / Generic)
      ├─ Tool-Namen kanonisieren
      ├─ Nachrichtenverlauf normalisieren  (tool_calls → XML)
      ├─ XML-System-Prompt einbauen
      ├─ Priming-Beispiele einbauen
      ├─ Loop-Erkennung
      ├─ Upstream-vLLM aufrufen
      └─ XML-Antwort parsen → tool_calls (mit mehreren Fallback-Ebenen)
      ↓  OpenAI-Format (tool_calls)
  Upstream-GPT-Modell
```

Fallback-Kette bei der Modellantwort:
1. XML-Parsing (primär)
2. Alias-Auflösung (write_file → write_to_file usw.)
3. Roo-Code-Builtin-Scan
4. JSON-Fallback
5. Text-Synthese (write_to_file oder attempt_completion)
6. Notfall-Fallback (attempt_completion mit "Task completed.")

---

## Gefundene Probleme

### MITTEL — Loop-Erkennung: Falsch-Positive
**Datei:** `app/services/loop_detection.py:60`

Die Worterkennung schlägt an, auch wenn das Tool-Ergebnis ein Fehler ist. Beispiele:
- `"Error: file could not be created"` → enthält `"created"` → erhöht den Erfolgs-Zähler
- `"File read successfully"` von einem `read_file` → erhöht den Schreib-Erfolgs-Zähler

**Auswirkung:** Falscher Stop-Hinweis wird mitten in einer gültigen Aufgabe injiziert.

---

### MITTEL — Text-Synthese schreibt Prosa auf die Festplatte
**Datei:** `app/services/text_synthesis.py:103`

Die `looks_like_file_content`-Heuristik ist zu weit gefasst:
```python
len(stripped) > 200 and any(marker in stripped for marker in ("##", "```", " | ", "---", "\n\n"))
```
Jeder erklärende Text über 200 Zeichen mit einem doppelten Zeilenumbruch (extrem häufig)
kann als Dateiinhalt interpretiert und in den aktiven VSCode-Tab geschrieben werden.
Das ist ein **Datenverlust-Risiko**.

---

### MITTEL — Lazy-Regex kürzt XML beim falschen schließenden Tag
**Datei:** `app/services/xml_parser.py:55`

Das Muster `<(tool_name)\b[^>]*>([\s\S]*?)</\1>` stoppt beim ersten schließenden Tag.
Enthält `<content>` innerhalb eines Tool-Aufrufs denselben Tag-Namen wie das äußere Tool,
wird der Treffer abgeschnitten und falsche Argumente werden stillschweigend erzeugt.
`fix_xml_string` mildert den häufigen Fall, deckt aber nicht alle Tags oder Verschachtelungen ab.

---

### MITTEL — Shell-Injection über Apostrophe in Dateipfaden
**Datei:** `app/services/tool_call_fixups.py:288`

```python
cmd = f"mv '{source}' '{dest}'"
```
Pfade mit `'` (z.B. `users notes.txt`) brechen die Shell-Quotierung.
Je nach Ausführung durch Roo Code kann das zu Fehlern oder unerwartetem Verhalten führen.

---

### MITTEL — Kein Startschutz für `upstream_client`
**Datei:** `app/main.py:72`

`upstream_client` ist beim Modulstart `None` und wird erst in `lifespan()` gesetzt.
Eine Anfrage vor Abschluss des Lifespans löst einen `AttributeError` aus, der als
undurchsichtiger 500-Fehler zurückgegeben wird.

---

### MITTEL — Keine Validierung des Request-Bodys
**Datei:** `app/main.py:167`

Der Endpoint akzeptiert `Dict[str, Any]`. Fehlerhafte Eingaben (falsche Typen, fehlende
Schlüssel) breiten sich durch 6+ Services aus, bevor sie als undurchsichtiger 500-Fehler
abstürzen — statt einer sauberen 422-Antwort.

---

### NIEDRIG-MITTEL — `_score_args` JSON-Fallback: Falsch-Positive
**Datei:** `app/services/tool_call_fixups.py:43`

Jedes JSON-Dict mit einem `path`-Schlüssel erzielt ≥ 1000 Punkte und matched `write_to_file`.
Die Strafpunkte für unbekannte Schlüssel (-5 je Schlüssel) senken den Score nie unter 1000.
Kann einen fehlerhaften Datei-Schreibvorgang durch ein nicht verwandtes JSON-Objekt auslösen.

---

### NIEDRIG-MITTEL — `rescue_xml_in_attempt_completion` ohne Tool-Filter
**Datei:** `app/services/tool_call_fixups.py:355`

Jeder Schlüssel im `result_content`-Dict wird ohne Validierung gegen bekannte Tool-Namen
zu einem Tool-Aufruf. Strukturierter Text in `<result>` (z.B. `<steps>`, `<note>`) wird
zu einem gefälschten Tool-Aufruf, den der Client ablehnen wird.

---

### NIEDRIG-MITTEL — `fix_xml_string` kann verschachtelte gleich-namige Tags nicht verarbeiten
**Datei:** `app/services/xml_parser.py:176`

Enthält `<content>` ein weiteres `<content>`-Tag (z.B. eine Datei, die toolproxy selbst
zeigt), stoppt die Lazy-Regex beim inneren schließenden Tag und lässt den äußeren Inhalt
unescaped.

---

### NIEDRIG — `tool_calls`-Schlüssel wird nach Normalisierung nicht entfernt
**Datei:** `app/services/message_normalizer.py:112`

Nach der Umwandlung von `tool_calls` in XML im `content`-Feld wird der `tool_calls`-Schlüssel
nicht entfernt. Der Upstream erhält sowohl `content` (XML) als auch `tool_calls` (OpenAI-Format)
im Verlauf. Manche OpenAI-kompatiblen APIs behandeln das als Schema-Verletzung.

---

### NIEDRIG — Lifespan-Teardown nicht ausnahme-sicher
**Datei:** `app/main.py:88`

`await upstream_client.close()` ist nicht in `try/finally` eingebettet.
Eine Ausnahme beim Herunterfahren erzeugt einen verwirrenden Traceback.

---

### NIEDRIG — Ungültiger `LOG_LEVEL` führt zu kryptischem `AttributeError`
**Datei:** `app/main.py:62`

`getattr(logging, settings.log_level)` wirft beim Start einen `AttributeError` für Werte
wie `"VERBOSE"` oder `"trace"`. Die Fehlermeldung gibt keinen Hinweis auf die Lösung.

---

### NIEDRIG — Priming-Docstring führt zukünftige Entwickler in die Irre
**Datei:** `app/services/priming.py:96`

Der Docstring sagt "when the conversation has only 1 user turn", aber Priming wird bei
jedem Turn ausgeführt. Ein Entwickler könnte das Verhalten aufgrund des falschen Docstrings
"korrigieren" und dabei absichtliches Verhalten brechen. Außerdem: 4 statische Sequenzen
× 2 Nachrichten = 8 Extranachrichten pro Anfrage für OpenCode — signifikanter Kontext-Overhead
bei langen Gesprächen.

---

### NIEDRIG — Diff-Parser lehnt Standard `---`/`+++`-Header ab
**Datei:** `app/services/tool_call_fixups.py:232`

`---`- und `+++`-Zeilen setzen `is_new_file = False` und verhindern die Umwandlung gültiger
Unified-Diffs, die Standard-Datei-Header enthalten.

---

### NIEDRIG — `asyncio.run()` im Test-Teardown inkompatibel mit async Event-Loops
**Datei:** `tests/live/conftest.py:170`

`asyncio.run(real_uc.close())` schlägt mit `RuntimeError: This event loop is already running`
fehl, wenn pytest-asyncio später hinzugefügt wird. Kein Produktionsproblem, aber bricht
den Test-Teardown.

---

## Lücken in der Testabdeckung

Die bestehende Testsuite (74 Tests) ist für den Happy Path solide. Bemerkenswerte Lücken:

- **Streaming-Pfad** (`stream: true`) — null Tests
- **Loop-Erkennung** schlägt auf Fehlermeldungen an, die Erfolgswörter enthalten
- **Fehlerhafte/leere `messages`**-Arrays im Request-Body
- **`convert_new_file_diffs`** mit Standard `---`/`+++`-Headern
- **OpenCode** Kanonisierung/Dekanonisierung als Roundtrip
- **`fix_xml_string`** mit verschachtelten gleich-namigen Tags
- **Upstream gibt `None`** oder leeren Response-Body zurück

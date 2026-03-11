# Testergebnisse & Zusammenfassung

Datum: 2026-03-11
Modell: `gpt-oss-120b` via LiteLLM

---

## Test-Setup (macOS)

Die Live-Tests laufen direkt gegen die LiteLLM-Instanz — kein SSH-Tunnel, kein Container nötig.

### Voraussetzungen

```bash
# API-Key dauerhaft in der Shell speichern
echo 'export LITELLM_MASTER_KEY="dein-key-hier"' >> ~/.zshrc
source ~/.zshrc

# Abhängigkeiten installieren
pip install -r requirements.txt -r requirements-test.txt
```

### `.env` für lokalen Proxy-Start (optional)

Vorlage: [`.env.example`](../.env.example) im Root kopieren und befüllen:

```bash
cp .env.example .env
# UPSTREAM_URL, UPSTREAM_API_KEY eintragen
```

### Tests ausführen

```bash
# Unit-Tests (kein Modell nötig, ~0.1s)
python3 -m pytest tests/ --ignore=tests/live -v

# Bug-Beweis-Tests (kein Modell nötig — schlagen aktuell fehl, absichtlich)
python3 -m pytest tests/test_bug_proof.py -v

# Live-Tests gegen echtes Modell (~9 Minuten)
python3 -m pytest -m live -v

# Nur Findings-Tests live
python3 -m pytest -m live -v tests/live/test_findings.py
```

### Hinweis: `conftest.py`-Änderung

`tests/live/conftest.py` war ursprünglich auf `localhost:8005` hardcodiert (SSH-Tunnel nötig).
Wurde auf Env-Variablen umgestellt — funktioniert jetzt ohne Tunnel direkt gegen LiteLLM:

| Env-Variable | Pflicht? | Beschreibung |
|---|---|---|
| `LIVE_UPSTREAM_URL` | **Ja** | LiteLLM-Endpoint (`http://<host>:4000/v1`) |
| `LIVE_MODEL` | Nein | Modell-Name (Default: `openai/gpt-oss-120b`) |
| `LIVE_API_KEY` | Nein | API-Key (fallback: `$LITELLM_MASTER_KEY`) |

---

## Gefundene Bugs (Statische Code-Analyse)

| # | Schwere | Datei | Problem |
|---|---|---|---|
| 1 | **Mittel** | `loop_detection.py:60` | Fehlermeldungen mit Wörtern wie `"created"` lösen den Erfolgs-Loop-Detektor aus — Modell bekommt falschen STOP-Hinweis mitten in einer Aufgabe |
| 2 | **Mittel** | `text_synthesis.py:103` | Prosa-Antworten >200 Zeichen mit `\n\n` können stillschweigend in eine offene Datei geschrieben werden — Datenverlust-Risiko |
| 3 | **Mittel** | `xml_parser.py:55` | Lazy-Regex kürzt XML, wenn `<content>` denselben Tag-Namen enthält wie das äußere Tool |
| 4 | **Mittel** | `tool_call_fixups.py:288` | Dateipfade mit Apostrophen (z.B. `user's notes.txt`) erzeugen ungültige Shell-Befehle — `mv 'user's notes.txt'` ist kein valides Syntax |
| 5 | **Mittel** | `main.py:72` | Kein Startschutz — Anfragen vor Abschluss des Lifespans crashen mit undurchsichtigem 500-Fehler |
| 6 | **Mittel** | `main.py:167` | Keine Request-Body-Validierung — fehlerhafte Eingaben crashen mit 500 statt 422 |
| 7 | **Niedrig** | `message_normalizer.py:112` | `tool_calls`-Schlüssel wird nach XML-Konvertierung nicht entfernt — Upstream erhält beide Formate |

---

## Testergebnisse gegen echtes Modell

### Standard-Live-Tests — 16/16 bestanden

```
test_write_to_file          ✅
test_read_file              ✅
test_apply_diff             ✅
test_list_files             ✅
test_delete_file            ✅
test_execute_command        ✅
test_attempt_completion     ✅  (transientes Timeout beim ersten Lauf, beim Retry ok — Upstream war ausgelastet)
test_ask_followup_question  ✅
test_search_files           ✅
test_write (OpenCode)       ✅
test_read (OpenCode)        ✅
test_list (OpenCode)        ✅
test_bash (OpenCode)        ✅
test_edit (OpenCode)        ✅
test_glob (OpenCode)        ✅
test_grep (OpenCode)        ✅
```

**Fazit:** Der Happy Path ist stabil. Die gefundenen Bugs liegen in Edge-Case/Fallback-Schichten und werden nur durch spezifische Modell-Ausgaben ausgelöst.

---

### Bug-Beweis-Unit-Tests — 5 schlagen fehl (Bugs bestätigt)

Diese Tests brauchen kein Modell — sie testen die Proxy-Logik direkt.
Datei: `tests/test_bug_proof.py`

```
test_error_message_with_success_word_does_not_trigger_loop  ❌ BUG BESTÄTIGT
test_error_only_never_triggers_loop                         ❌ BUG BESTÄTIGT
test_apostrophe_in_source_path_produces_valid_shell_command ❌ BUG BESTÄTIGT
test_apostrophe_in_dest_path_produces_valid_shell_command   ❌ BUG BESTÄTIGT
test_normalized_assistant_message_has_no_tool_calls_key     ❌ BUG BESTÄTIGT
```

Fehlermeldungen (Auszug):

```
BUG: Loop detection fired on an error message.
Hint injected: 'STOP: The file operation has already succeeded 2 times...'
Cause: 'created' in error message counted as a success.

BUG: Shell command has broken quoting (apostrophe injection).
Command: "mv 'user's notes.txt' 'dest.txt'"
shlex error: No closing quotation

BUG: tool_calls key still present after normalization.
Keys present: ['role', 'content', 'tool_calls']
```

---

## Vorgeschlagene Fixes

Alle Priorität-1-Fixes sind voneinander unabhängig und können parallel umgesetzt werden.
Konkrete Code-Snippets und vollständiger Fix-Plan: [fix-plan.md](fix-plan.md)

---

## Nächste Schritte

1. **Sofort:** Bug 1, 4 und 7 fixen — jeweils 1-3 Zeilen, unabhängig voneinander
2. **Validierung:** `python3 -m pytest tests/test_bug_proof.py -v` muss danach grün sein
3. **Regression:** `python3 -m pytest tests/ --ignore=tests/live -v` — alle 74 bestehenden Tests müssen weiterhin bestehen
4. **Strategisch:** Constrained-Decoding-Spike in vLLM — siehe [architektur-und-roadmap.md](architektur-und-roadmap.md)

# Fix-Plan

Zuletzt aktualisiert: 2026-03-11

Vollständige Problembeschreibungen siehe [stabilitaets-analyse.md](stabilitaets-analyse.md).

---

## Priorität 1 — Hohe Wirkung, geringes Risiko

Diese Fixes sind in sich geschlossen, einfach zu testen und adressieren direkt die
gemeldeten Stabilitätsprobleme. Können parallel auf separaten Branches umgesetzt werden.

### [ ] Loop-Erkennung: Falsch-Positive beheben
**Datei:** `app/services/loop_detection.py:60`
**Änderung:** Vor dem Erhöhen von `success_count` prüfen, ob der Inhalt KEINE
fehlerweisenden Wörter enthält (`"error"`, `"failed"`, `"could not"`, `"unable to"`).
```python
_ERROR_INDICATORS = ("error", "failed", "could not", "unable to", "exception", "traceback")
has_success = any(word in lower for word in _WRITE_SUCCESS_WORDS)
has_error = any(word in lower for word in _ERROR_INDICATORS)
if has_success and not has_error:
    success_count += 1
```
**Test hinzufügen:** Tool-Ergebnis `"Error: file could not be created"` darf KEINEN
Loop-Hinweis auslösen.

---

### [ ] Text-Synthese-Heuristik einschränken
**Datei:** `app/services/text_synthesis.py:103`
**Änderung:** Lose Marker-Prüfung durch Code-spezifische Marker ersetzen und Antworten
ausschließen, die mit typischen Prosa-Einstiegen beginnen.
```python
_CODE_MARKERS = ("```", "def ", "class ", "import ", "function ", "const ", "export ", "#!/")

looks_like_file_content = (
    not looks_like_xml_tool_call
    and len(stripped) > 200
    and any(marker in stripped for marker in _CODE_MARKERS)
    and not stripped.lower().startswith(("i ", "the ", "here ", "this ", "let me", "to ", "sorry"))
)
```
**Test hinzufügen:** Langer erklärender Prosa-Text (>200 Zeichen, enthält `\n\n`) darf
KEINEN `write_to_file`-Aufruf erzeugen. Echter Code-Inhalt muss es weiterhin auslösen.

---

### [ ] Shell-Injection in move_file-Konvertierung beheben
**Datei:** `app/services/tool_call_fixups.py:288`
**Änderung:** `shlex.quote()` statt manueller Einfach-Quotierung verwenden.
```python
import shlex
cmd = f"mv {shlex.quote(source)} {shlex.quote(dest)}"
```
**Test hinzufügen:** Quellpfad mit `'` (Apostroph) muss einen gültigen Shell-Befehl erzeugen.

---

### [ ] `tool_calls`-Schlüssel nach Normalisierung entfernen
**Datei:** `app/services/message_normalizer.py:112`
**Änderung:** Eine Zeile nach dem Schreiben von XML in `content` hinzufügen.
```python
msg["content"] = "\n".join(xml_parts)
msg.pop("tool_calls", None)   # ← hinzufügen
```
**Test hinzufügen:** Normalisierte Assistenten-Nachricht mit tool_calls darf keinen
`tool_calls`-Schlüssel mehr enthalten.

---

## Priorität 2 — Korrektheit & Zuverlässigkeit

### [ ] Startschutz für `upstream_client = None` hinzufügen
**Datei:** `app/main.py` — Anfang von `chat_completions()`
**Änderung:** 503 zurückgeben, wenn `upstream_client` noch `None` ist.
```python
if upstream_client is None:
    return JSONResponse(
        status_code=503,
        content={"error": {"message": "Service startet, bitte kurz warten", "type": "service_unavailable"}},
    )
```

---

### [ ] Lifespan-Teardown in try/finally einwickeln
**Datei:** `app/main.py:88`
```python
yield
try:
    await upstream_client.close()
except Exception as e:
    logger.warning(f"Fehler beim Schließen des Upstream-Clients: {e}")
```

---

### [ ] `LOG_LEVEL` beim Start validieren
**Datei:** `app/main.py:62`
```python
log_level = getattr(logging, settings.log_level, None)
if not isinstance(log_level, int):
    raise ValueError(
        f"Ungültiger LOG_LEVEL={settings.log_level!r}. Erlaubt: DEBUG, INFO, WARNING, ERROR, CRITICAL"
    )
logging.basicConfig(level=log_level, ...)
```

---

### [ ] Diff-Parser: Standard `---`/`+++`-Header nicht als Kontext-Zeilen behandeln
**Datei:** `app/services/tool_call_fixups.py:232`
**Änderung:** `---`- und `+++`-Zeilen überspringen (wie `@@` und Leerzeilen).
```python
if line.startswith("@@") or line.startswith("---") or line.startswith("+++") or line == "":
    continue
```
**Test hinzufügen:** Neue-Datei-Diff mit `--- /dev/null` / `+++ b/datei`-Headern muss
weiterhin konvertiert werden.

---

## Priorität 3 — Größere Refactorings (vor Start abstimmen)

### [ ] Request-Body-Validierung mit Pydantic
**Datei:** `app/main.py:167`
**Änderung:** `Dict[str, Any]` durch ein Pydantic-Modell ersetzen. FastAPI gibt dann
automatisch 422 für fehlerhafte Anfragen zurück.
```python
class ChatRequest(BaseModel):
    model: str = ""
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    tools: List[Dict[str, Any]] = Field(default_factory=list)
    stream: bool = False
    temperature: float = 0.7
    max_tokens: Optional[int] = None
```
**Hinweis:** Downstream-Code nutzt `request.get(...)` — muss auf `request.messages` usw.
umgestellt werden. Berührt viel Code; **vor dem Start koordinieren**.

---

### [ ] `rescue_xml_in_attempt_completion` mit Tool-Name-Filter versehen
**Datei:** `app/services/tool_call_fixups.py:355`
**Änderung:** Nur Schlüssel akzeptieren, die zu einem bekannten Tool-Namen auflösen.
Unbekannte Schlüssel loggen und überspringen statt als gefälschte Tool-Aufrufe zu emittieren.

---

### [ ] `_score_args` JSON-Fallback: Über-Matching reduzieren
**Datei:** `app/services/tool_call_fixups.py:43`
**Änderung:** Score > 1000 verlangen (d.h. mindestens ein optionaler Parameter muss matchen)
ODER keine unbekannten Schlüssel erlauben, um Falsch-Positive auf `{"path": ...}`-Blobs
zu reduzieren.

---

## Fehlende Tests

Unabhängig von den konkreten Fixes zu ergänzen:

- [ ] Streaming-Pfad (`stream: true`) — aktuell null Tests
- [ ] Loop-Erkennung: Fehlermeldungen mit Erfolgswörtern lösen keinen Loop-Hinweis aus
- [ ] Fehlerhafte Anfrage-Bodies (fehlendes `messages`, falsche Typen) → 4xx statt 500
- [ ] `convert_new_file_diffs` mit `---`/`+++`-Headern
- [ ] OpenCode Kanonisierung/Dekanonisierung als vollständiger Roundtrip
- [ ] `fix_xml_string` mit verschachtelten gleich-namigen Tags (bekannte Einschränkung, zumindest dokumentieren)
- [ ] Upstream gibt leeren/null Response-Body zurück

---

## Hinweise zur Zusammenarbeit

- Alle Priorität-1-Fixes sind **voneinander unabhängig** — können parallel auf separaten Branches bearbeitet werden.
- Priorität-2-Fixes sind ebenfalls unabhängig voneinander.
- Die Pydantic-Validierung (Priorität 3) berührt `main.py` großflächig — **vor dem Start** über das Modell einigen, um Merge-Konflikte zu vermeiden.
- Das verschachtelte-Tag-Problem in `fix_xml_string` ist ohne Ersatz des Regex durch einen echten XML-Parser (z.B. `lxml` mit Recovery-Modus) schwer zu lösen. Separater Spike empfohlen.

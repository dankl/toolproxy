# Architektur & Roadmap

Zuletzt aktualisiert: 2026-03-11

---

## Warum dieser Proxy existiert

Roo Code und OpenCode sprechen das OpenAI-Tool-Calling-Protokoll: Sie senden ein `tools[]`-Array
und erwarten `tool_calls` in der Antwort. Unser selbst gehostetes GPT-Modell (via vLLM) wurde
ohne native Function-Calling-Unterstützung trainiert — es ignoriert `tools[]` und gibt Klartext zurück.

Der Proxy überbrückt diese Lücke, indem er das Protokoll in beide Richtungen übersetzt:

```
Roo Code / OpenCode          toolproxy                     GPT-Modell (vLLM)
───────────────────          ─────────────────────────     ─────────────────
tools: [write_to_file, ...]  → XML-System-Prompt       →   "Nutze <write_to_file>
tool_calls: [{...}]          ← XML-Antwort parsen      ←    <path>...</path>"
```

Ohne Proxy: Das Modell gibt Prosa zurück, Roo Code wirft "You did not use a tool",
die Session bricht ab.

---

## Aktuelle Architektur

### Request-Pipeline (pro Turn)

```
1.  Client-Typ erkennen          (Roo Code / OpenCode / Generic)
2.  Tool-Namen kanonisieren      (OpenCode "write" → kanonisch "write_to_file")
3.  Nachrichtenverlauf normalisieren  (tool_calls → XML)
4.  XML-System-Prompt bauen      (alle Tools im XML-Format beschreiben)
5.  Priming-Beispiele einbauen   (bis zu 14 synthetische Nachrichten mit korrektem XML)
6.  Loop-Erkennung               (Stop-Hinweis einbauen, wenn Modell sich wiederholt)
7.  Upstream-vLLM aufrufen       (keine nativen Tools weitergeleitet)
8.  Antwort parsen               (Fallback-Kette — siehe unten)
9.  Schema-Remap                 (kanonische Parameternamen → client-spezifische Namen)
10. Dekanonisieren               (kanonische Tool-Namen → client-spezifische Namen)
11. Auf 1 Tool-Aufruf begrenzen  (Clients verarbeiten jeweils einen Tool-Aufruf)
12. OpenAI-Antwort zurückgeben
```

### Fallback-Kette bei der Modellantwort

Das Modell folgt nicht immer dem XML-Format. Der Proxy erholt sich in mehreren Schichten:

| Schicht | Auslöser | Aktion |
|---------|----------|--------|
| XML-Parsing | Modell gibt XML-Tool-Aufruf zurück | Direkt parsen |
| Alias-Auflösung | Modell nutzt falschen Tag-Namen (write_file statt write_to_file) | Alias umschreiben |
| Roo-Builtin-Scan | Modell ruft ein Builtin-Tool auf, das nicht in tools[] steht | Akzeptieren |
| JSON-Fallback | Antwort enthält JSON-Blob | Argumente gegen Schemas bewerten |
| Text-Synthese | Antwort ist Prosa | Heuristik: write_to_file oder attempt_completion |
| Notfall-Fallback | Leere Antwort | attempt_completion("Task completed.") |

---

## Grundursache der Instabilität

Alle Stabilitätsprobleme haben eine gemeinsame Ursache:

> Wir bitten ein Modell, das nicht für strukturierte Ausgabe trainiert wurde, zur Laufzeit
> zuverlässig strukturiertes XML zu produzieren.

Die 6-schichtige Fallback-Kette, XML-Fixups, Loop-Erkennung und Text-Synthese-Heuristiken
sind allesamt Laufzeit-Patches für diese Unzuverlässigkeit. Je mehr Schichten hinzugefügt
werden, desto komplexer und anfälliger wird der Proxy. Die konkreten Auswirkungen sind in
[stabilitaets-analyse.md](stabilitaets-analyse.md) beschrieben.

---

## Roadmap

### Option A — Constrained Decoding in vLLM (empfohlener Spike)

vLLM unterstützt grammatik-basierte / JSON-Schema-gesteuerte Generierung. Das Modell wird
auf Token-Ebene gezwungen, valides JSON auszugeben — fehlerhafte Ausgaben sind konstruktionsbedingt
unmöglich.

```python
# Statt XML-System-Prompt + Priming senden wir:
{
    "messages": [...],
    "guided_json": {
        "type": "object",
        "properties": {
            "tool": {"enum": ["write_to_file", "read_file", ...]},
            "arguments": {"type": "object"}
        },
        "required": ["tool", "arguments"]
    }
}
```

**Wenn das funktioniert, vereinfacht sich der Proxy drastisch:**

| Heute | Mit Constrained Decoding |
|-------|--------------------------|
| Großer XML-System-Prompt | Minimaler Prompt |
| 8–14 Priming-Nachrichten pro Turn | Keine |
| 6-schichtige Fallback-Kette | `json.loads()` — fertig |
| Loop-Erkennung | Nicht nötig (Modell gibt immer validen Tool-Aufruf aus) |
| XML-Fixups | Nicht nötig |
| Text-Synthese-Heuristiken | Nicht nötig |

**Spike-Umfang:** 1–2 Tage. Testen, ob vLLM `guided_json` mit unserem Modell zuverlässige
Tool-Aufrufe produziert. Bei Erfolg: XML-Pipeline schrittweise ersetzen. Bei Misserfolg: Option B.

**vLLM-Dokumentation:** https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#guided-decoding

---

### Option B — XML-Ansatz verbessern (inkrementell)

Falls Constrained Decoding nicht praktikabel ist (Modell kooperiert nicht mit der Grammatik
oder vLLM-Versionseinschränkungen), kann der aktuelle XML-Ansatz gehärtet werden:

1. **Besserer System-Prompt & Priming** — hochwertigere Few-Shot-Beispiele reduzieren direkt,
   wie oft Fallbacks ausgelöst werden. Der größte Hebel innerhalb der aktuellen Architektur.

2. **Observability** — aktuell gibt es keine Sicht darauf, welche Fallback-Schicht wie oft
   in der Produktion ausgelöst wird. Ein Zähler pro Schicht würde sofort zeigen, wo zu fokussieren ist:
   ```
   toolproxy_parse_path{method="xml"}          = 847
   toolproxy_parse_path{method="alias"}        = 203
   toolproxy_parse_path{method="json"}         =  41
   toolproxy_parse_path{method="synthesis"}    =  18
   toolproxy_parse_path{method="emergency"}    =   3
   ```

3. **Stabilitäts-Fixes** aus [fix-plan.md](fix-plan.md) — die schlimmsten Symptome beheben,
   während die Architekturfrage geklärt wird.

---

### Option C — Modell Fine-Tunen

Das Modell auf Tool-Calling-Beispielen in einem gewählten Format (XML oder JSON) trainieren.
Beseitigt die Kern-Komplexität des Proxys dauerhaft.

**Abwägungen:** Teuer (Rechenleistung + Zeit), erfordert Trainingsdaten, Modell-Updates
benötigen Neutraining. Nur sinnvoll, wenn das Modell langfristig gepflegt wird und Option A scheitert.

---

## Entscheidungspunkt

Die erste zu beantwortende Frage vor jeder größeren Proxy-Investition:

> **Funktioniert vLLM Constrained/Guided Decoding zuverlässig mit unserem Modell?**

Diese Antwort bestimmt, ob wir in einen besseren XML-Proxy investieren (Option B) oder
ihn schrittweise ersetzen (Option A). Empfehlung: Spike vor der Umsetzung von Priorität-3-Fixes
aus dem Fix-Plan durchführen.

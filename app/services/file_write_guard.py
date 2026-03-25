"""
File-write guard for toolproxy.

Detects when the model tries to write documentation/markdown content into
config files (application.yml, pom.xml, etc.) and asks the user where the
documentation should actually go — instead of silently corrupting the config file.

Primary prevention: the system prompt (xml_prompt_builder.py rule #7) tells the
model not to do this. This guard is a safety net for cases where the system
prompt is dropped by the upstream adapter (known OCI behaviour).
"""
import fnmatch
import json
import logging
import os
import re
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Config/source files whose content should never be markdown documentation.
# Exact basenames:
_PROTECTED_CONFIG_NAMES = {
    # Java / Spring
    "application.yml", "application.yaml", "application.properties",
    "pom.xml", "build.gradle", "build.gradle.kts",
    "settings.gradle", "settings.gradle.kts", "settings.xml",
    "web.xml", "beans.xml", "context.xml",
    "logback.xml", "logback-spring.xml",
    "log4j2.xml", "log4j2.properties",
    "persistence.xml", "hibernate.cfg.xml",
    "checkstyle.xml", "sonar-project.properties",
    # React / Angular / Node
    "package.json", "package-lock.json",
    "tsconfig.json", "angular.json", "nx.json",
    "jest.config.js", "jest.config.ts",
    "karma.conf.js",
    "vite.config.js", "vite.config.ts",
    "webpack.config.js", "webpack.config.ts",
    "babel.config.js", "babel.config.json",
    "postcss.config.js", "postcss.config.ts",
    "tailwind.config.js", "tailwind.config.ts",
    ".eslintrc", ".eslintrc.json", ".eslintrc.yml", ".eslintrc.yaml", ".eslintrc.js",
    ".prettierrc", ".prettierrc.json",
    # Helm
    "Chart.yaml", "Chart.yml",
    "values.yaml", "values.yml",
    # CI/CD & containers
    "Dockerfile", "docker-compose.yml", "docker-compose.yaml",
    ".gitlab-ci.yml",
    # Scripting / build
    "Makefile",
}

# Glob patterns matched against the basename (fnmatch):
_PROTECTED_CONFIG_PATTERNS = [
    # Spring profiles: application-dev.yml, application-prod.properties, …
    "application-*.yml", "application-*.yaml", "application-*.properties",
    # tsconfig variants: tsconfig.app.json, tsconfig.spec.json, …
    "tsconfig.*.json",
    # Helm environment values: values-dev.yaml, values-prod.yml, …
    "values-*.yaml", "values-*.yml",
    # GitHub Actions workflows
    "*.github-workflow.yml",
    # Generic Java XML configs
    "*-context.xml", "*-beans.xml",
]


def _is_protected_config(filename: str) -> bool:
    if filename in _PROTECTED_CONFIG_NAMES:
        return True
    return any(fnmatch.fnmatch(filename, pat) for pat in _PROTECTED_CONFIG_PATTERNS)

# Minimum number of markdown heading lines in the first 30 lines to flag content as docs.
_MD_HEADING_THRESHOLD = 2


def _is_markdown_content(content: str) -> bool:
    """Return True if content looks like markdown documentation, not config/code."""
    lines = [l for l in content.strip().splitlines() if l.strip()][:30]
    heading_count = sum(
        1 for l in lines
        if re.match(r"^#{1,6} ", l) or l.startswith("===") or l.startswith("---")
    )
    return heading_count >= _MD_HEADING_THRESHOLD


def guard_write_to_file(
    tool_calls: List[Dict[str, Any]],
    request_id: str,
    canonical_tools: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Inspect write_to_file calls for markdown content being written to config files.

    If detected and ask_followup_question is available: replace the bad write
    with a question to the user asking where the documentation should go.

    If ask_followup_question is not available: log a warning and pass through
    unchanged — the priming rule should have prevented this; if it didn't, at
    least the problem is visible in the logs.
    """
    available_tools = {
        t.get("function", {}).get("name", "")
        for t in (canonical_tools or [])
        if t.get("function")
    }
    has_ask = "ask_followup_question" in available_tools

    result = []
    for tc in tool_calls:
        fn = tc.get("function", {})
        if fn.get("name") != "write_to_file":
            result.append(tc)
            continue

        try:
            args = json.loads(fn.get("arguments", "{}"))
        except Exception:
            result.append(tc)
            continue

        path: str = args.get("path", "") or args.get("filePath", "")
        content: str = args.get("content", "") or ""
        filename = os.path.basename(path)

        if not _is_protected_config(filename) or not _is_markdown_content(content):
            result.append(tc)
            continue

        # Guard fires
        if has_ask:
            logger.warning(
                f"[{request_id}] WRITE GUARD: model tried to write markdown docs "
                f"into {path!r} → replacing with ask_followup_question"
            )
            question_tc = {
                "id": f"call_{request_id}_guard",
                "type": "function",
                "function": {
                    "name": "ask_followup_question",
                    "arguments": json.dumps({
                        "question": (
                            f"I was about to write documentation content into `{path}`, "
                            f"but that is a config file and should not contain markdown. "
                            f"Where should the documentation go instead? "
                            f"(e.g. README.md, docs/setup.md, CONTRIBUTING.md)"
                        )
                    }),
                },
            }
            result.append(question_tc)
        else:
            logger.warning(
                f"[{request_id}] WRITE GUARD: model tried to write markdown docs "
                f"into {path!r} — ask_followup_question not available, passing through. "
                f"Check system prompt rule #7."
            )
            result.append(tc)

    return result

"""Secret redaction for repository content leaving the backend.

Imported repositories are arbitrary third-party code and may contain committed
credentials. Anything served to the browser or placed in an LLM prompt passes
through here first.

This is a safety net, not a secret scanner: it is deliberately biased towards
catching the common shapes (assignments to key-like names, recognisable token
prefixes) and will miss creative ones. It never rejects content — it masks and
reports, so a file with a suspicious line still renders.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

MASK = "[REDACTED]"

#: Assignment of a *quoted literal* to a suspiciously named variable:
#: ``API_KEY = "..."``, ``password: '...'``.
#:
#: The quotes are load-bearing. An earlier version accepted a bare
#: right-hand side and rewrote FastAPI's
#: ``token = _effective_route_context_var.set(ctx)`` into
#: ``token = [REDACTED])`` — destroying valid code and leaving an unbalanced
#: paren. Committed secrets are string literals; an unquoted RHS is an
#: expression, so only quoted values are considered.
_ASSIGNMENT = re.compile(
    r"""(?ix)
    \b(
        [A-Za-z0-9_\-.]*
        (?: secret | passwd | password | api[_\-]?key | apikey
          | access[_\-]?key | private[_\-]?key | client[_\-]?secret
          | auth[_\-]?token | access[_\-]?token | bearer | credential )
        [A-Za-z0-9_\-.]*
    )
    (\s*[:=]\s*)
    (['"])
    ([^'"\n]{8,})
    \3
    """
)

#: Recognisable credential formats, matched anywhere in a line.
_PATTERNS: tuple[tuple[str, re.Pattern], ...] = (
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{16,}\b")),
    ("openai_key", re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}\b")),
    ("anthropic_key", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}\b")),
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("slack_token", re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}\b")),
    ("google_key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b")),
    ("private_key_block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("connection_string", re.compile(r"\b\w+://[^\s:@/]+:[^\s:@/]+@[^\s/]+")),
)

#: Placeholders that are conventions, not leaks. Redacting these adds noise.
_PLACEHOLDERS = frozenset(
    {
        "none", "null", "true", "false", "changeme", "example", "your_key",
        "your-key", "xxx", "xxxx", "todo", "fixme", "placeholder", "secret",
        "password", "dummy", "test", "sample", "redacted", "os.environ",
        "process.env", "undefined", "...",
    }
)


@dataclass
class RedactionResult:
    text: str
    findings: list[str]

    @property
    def redacted(self) -> bool:
        return bool(self.findings)


def _looks_like_placeholder(value: str) -> bool:
    lowered = value.strip().strip("\"'").lower()
    if lowered in _PLACEHOLDERS:
        return True
    # Environment lookups and template markers are references, not secrets.
    return lowered.startswith(("os.", "process.", "env.", "${", "{{", "<", "$("))


def redact(text: str) -> RedactionResult:
    """Mask likely credentials in ``text``."""
    if not text:
        return RedactionResult(text="", findings=[])

    findings: list[str] = []
    result = text

    for label, pattern in _PATTERNS:
        if pattern.search(result):
            findings.append(label)
            result = pattern.sub(MASK, result)

    def _mask_assignment(match: re.Match) -> str:
        name, separator, quote, value = match.groups()
        if _looks_like_placeholder(value):
            return match.group(0)
        findings.append(f"assignment:{name}")
        return f"{name}{separator}{quote}{MASK}{quote}"

    result = _ASSIGNMENT.sub(_mask_assignment, result)

    return RedactionResult(text=result, findings=sorted(set(findings)))


def redact_text(text: str) -> str:
    """Convenience wrapper when only the masked text is needed."""
    return redact(text).text

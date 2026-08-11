"""Intent classification and reference resolution.

Deliberately heuristic, not an LLM call. Generation on the target hardware
costs ~150s; spending another one just to label a question would double the
latency of every turn to decide something a scored keyword match gets right.
The classifier reports its confidence, so an LLM fallback can be added later
for the low-confidence tail without changing the interface.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

INTENT_SEARCH = "repository_search"
INTENT_CODE_EXPLANATION = "code_explanation"
INTENT_ARCHITECTURE = "architecture_analysis"
INTENT_STRUCTURE = "structure_analysis"
INTENT_DEPENDENCY = "dependency_analysis"
INTENT_FILE = "file_analysis"
INTENT_HISTORY = "git_history_analysis"
INTENT_CODE_GENERATION = "code_generation"

INTENTS: tuple[str, ...] = (
    INTENT_SEARCH,
    INTENT_CODE_EXPLANATION,
    INTENT_ARCHITECTURE,
    INTENT_STRUCTURE,
    INTENT_DEPENDENCY,
    INTENT_FILE,
    INTENT_HISTORY,
    INTENT_CODE_GENERATION,
)

#: (intent, weight, pattern). Weights let a decisive phrase outrank an
#: incidental keyword — "what depends on X" must not lose to the word "code".
_RULES: tuple[tuple[str, int, re.Pattern], ...] = (
    # dependency
    (INTENT_DEPENDENCY, 6, re.compile(r"\b(depends?\s+on|depend(s|ing)?\b.*\bon)\b", re.I)),
    (INTENT_DEPENDENCY, 6, re.compile(r"\b(what|who|which)\b.*\b(uses|imports|calls|references)\b", re.I)),
    (INTENT_DEPENDENCY, 5, re.compile(r"\b(dependents?|dependenc(y|ies)|reverse dependenc)", re.I)),
    (INTENT_DEPENDENCY, 4, re.compile(r"\b(impact|blast radius|affected by)\b", re.I)),
    # history
    (INTENT_HISTORY, 6, re.compile(r"\b(recently|lately|last (week|month|year)|over time)\b", re.I)),
    (INTENT_HISTORY, 5, re.compile(r"\b(git|commit|commits|changelog|history|evolved|evolution)\b", re.I)),
    (INTENT_HISTORY, 5, re.compile(r"\bwhat\s+changed\b", re.I)),
    (INTENT_HISTORY, 4, re.compile(r"\b(who\s+(wrote|changed|added)|when\s+was)\b", re.I)),
    # code generation
    (INTENT_CODE_GENERATION, 7, re.compile(r"\b(write|create|generate|add|implement)\b.*\b(test|tests|function|class|method|endpoint|docstring)\b", re.I)),
    (INTENT_CODE_GENERATION, 6, re.compile(r"\b(refactor|rewrite|convert|port)\b", re.I)),
    (INTENT_CODE_GENERATION, 5, re.compile(r"\bunit tests?\b", re.I)),
    # structure
    (INTENT_STRUCTURE, 6, re.compile(r"\b(repo(sitory)?|project|folder|directory)\s+(structure|layout|tree|organis|organiz)", re.I)),
    (INTENT_STRUCTURE, 6, re.compile(r"\b(show|display|list)\b.*\b(structure|tree|layout|files|folders|modules)\b", re.I)),
    (INTENT_STRUCTURE, 4, re.compile(r"\bwhat('s| is)\s+in\s+(this|the)\s+(repo|project)", re.I)),
    # architecture
    (INTENT_ARCHITECTURE, 7, re.compile(r"\barchitect(ure|ural)\b", re.I)),
    (INTENT_ARCHITECTURE, 5, re.compile(r"\b(overall|high[- ]level|big picture)\b", re.I)),
    (INTENT_ARCHITECTURE, 5, re.compile(r"\bhow\b.*\bmodules?\b.*\b(relate|connect|interact|depend)", re.I)),
    (INTENT_ARCHITECTURE, 4, re.compile(r"\bwhat is this (repo|repository|project|codebase) about\b", re.I)),
    (INTENT_ARCHITECTURE, 4, re.compile(r"\b(design|layered|components?)\b.*\b(overview|overall)\b", re.I)),
    # file analysis
    (INTENT_FILE, 6, re.compile(r"\b(this|that|the)\s+file\b", re.I)),
    (INTENT_FILE, 5, re.compile(r"\b\w+\.(py|js|ts|tsx|jsx|java|go|rs|cpp|c|cs|kt|sql|html|css)\b")),
    (INTENT_FILE, 4, re.compile(r"\b(tell me about|what does)\b.*\b(do|contain)\b", re.I)),
    # search
    (INTENT_SEARCH, 6, re.compile(r"\bwhere\s+(is|are|does|do|can i find)\b", re.I)),
    (INTENT_SEARCH, 5, re.compile(r"\b(find|locate|search for)\b", re.I)),
    (INTENT_SEARCH, 4, re.compile(r"\bwhich file\b", re.I)),
    # explanation (the default, so weights stay low)
    (INTENT_CODE_EXPLANATION, 3, re.compile(r"\bhow\s+does\b", re.I)),
    (INTENT_CODE_EXPLANATION, 3, re.compile(r"\b(explain|describe|walk me through)\b", re.I)),
    (INTENT_CODE_EXPLANATION, 2, re.compile(r"\bwhy\b", re.I)),
    (INTENT_CODE_EXPLANATION, 2, re.compile(r"\bhow\s+(is|are|do|can)\b", re.I)),
)

#: Words that stand in for the previously discussed thing.
_PRONOUNS = re.compile(r"\b(it|its|this|that|these|those|there|here|the same)\b", re.I)

#: A file path or a CamelCase / dotted identifier mentioned in the question.
_FILE_MENTION = re.compile(
    r"\b([\w./-]+\.(?:py|js|ts|tsx|jsx|java|go|rs|cpp|c|cs|kt|sql|html|css))\b"
)
_SYMBOL_MENTION = re.compile(r"\b([A-Z][A-Za-z0-9]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)\b")
_SNAKE_MENTION = re.compile(r"\b([a-z_][a-z0-9_]{3,}(?:\.[a-z_][a-z0-9_]*)*)\s*\(")


@dataclass
class Classification:
    intent: str
    confidence: float
    scores: dict[str, int] = field(default_factory=dict)
    #: File paths and symbol names named in the question.
    mentions: list[str] = field(default_factory=list)
    #: True when the question leans on a pronoun and needs prior context.
    needs_context: bool = False


def extract_mentions(question: str) -> list[str]:
    """Concrete file or symbol names referenced in a question."""
    mentions: list[str] = []
    seen: set[str] = set()

    for pattern in (_FILE_MENTION, _SYMBOL_MENTION, _SNAKE_MENTION):
        for match in pattern.finditer(question):
            value = match.group(1)
            # Sentence-initial capitals are not symbol names.
            if pattern is _SYMBOL_MENTION and (
                len(value) < 3 or value.lower() in _COMMON_WORDS
            ):
                continue
            if value not in seen:
                seen.add(value)
                mentions.append(value)
    return mentions


#: Sentence-initial words that capitalisation makes look like symbols.
#: Without the imperative verbs, "Create tests for this function" resolves
#: focus onto a non-existent symbol named `Create`.
_COMMON_WORDS = frozenset(
    {
        # interrogatives and articles
        "the", "this", "that", "what", "where", "how", "why", "who", "when",
        "which", "does", "did", "and", "but", "for", "it", "is", "are", "was",
        "can", "could", "should", "would", "please",
        # imperative verbs that open a request
        "explain", "show", "give", "tell", "create", "write", "generate",
        "add", "implement", "refactor", "rewrite", "convert", "list",
        "display", "find", "locate", "search", "describe", "walk", "make",
        "build", "help", "summarise", "summarize", "analyse", "analyze",
        # bare domain words that are never a specific symbol
        "api", "ai", "repo", "repository", "project", "codebase", "code",
    }
)


def classify(question: str) -> Classification:
    """Score a question against every intent and pick the best."""
    text = (question or "").strip()
    if not text:
        return Classification(intent=INTENT_CODE_EXPLANATION, confidence=0.0)

    scores: dict[str, int] = {}
    for intent, weight, pattern in _RULES:
        if pattern.search(text):
            scores[intent] = scores.get(intent, 0) + weight

    if not scores:
        # Nothing matched: explanation is the safe default, and the low
        # confidence marks it as a guess.
        return Classification(
            intent=INTENT_CODE_EXPLANATION,
            confidence=0.2,
            scores={},
            mentions=extract_mentions(text),
            needs_context=bool(_PRONOUNS.search(text)),
        )

    best = max(scores, key=scores.get)
    total = sum(scores.values())
    confidence = scores[best] / total if total else 0.0

    return Classification(
        intent=best,
        confidence=round(confidence, 3),
        scores=scores,
        mentions=extract_mentions(text),
        needs_context=bool(_PRONOUNS.search(text)),
    )

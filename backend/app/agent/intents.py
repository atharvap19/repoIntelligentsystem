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

#: Part 10 — "how does a request flow through this repository?". Distinct from
#: explanation because the answer is a *path* through the import graph, not a
#: description of one unit.
INTENT_FLOW = "flow_analysis"

#: Part 8 — "show me the database layer", "everything related to
#: authentication". The subject is a theme rather than a named symbol, so the
#: answer is assembled from several units that share it.
INTENT_CONCEPT = "concept_analysis"

#: Part 13 — "which modules are most depended on?". Answered from computed
#: graph and history metrics, never estimated.
INTENT_HOTSPOT = "hotspot_analysis"

#: Part 16 — a request to write, refactor or modify code. Repoint is
#: read-only, so this intent exists to be declined clearly and to redirect to
#: an explanation of what *would* change, not to be fulfilled.
INTENT_READONLY = "readonly_request"

#: Retained so Phase 3 imports keep resolving. The behaviour behind it changed
#: in Phase 4 — see ``INTENT_READONLY``.
INTENT_CODE_GENERATION = INTENT_READONLY

INTENTS: tuple[str, ...] = (
    INTENT_SEARCH,
    INTENT_CODE_EXPLANATION,
    INTENT_ARCHITECTURE,
    INTENT_STRUCTURE,
    INTENT_DEPENDENCY,
    INTENT_FILE,
    INTENT_HISTORY,
    INTENT_FLOW,
    INTENT_CONCEPT,
    INTENT_HOTSPOT,
    INTENT_READONLY,
)

#: (intent, weight, pattern). Weights let a decisive phrase outrank an
#: incidental keyword — "what depends on X" must not lose to the word "code".
_RULES: tuple[tuple[str, int, re.Pattern], ...] = (
    # dependency
    (INTENT_DEPENDENCY, 6, re.compile(r"\b(depends?\s+on|depend(s|ing)?\b.*\bon)\b", re.I)),
    (INTENT_DEPENDENCY, 6, re.compile(r"\b(what|who|which)\b.*\b(uses?|imports?|calls?|references?)\b", re.I)),
    (INTENT_DEPENDENCY, 5, re.compile(r"\b(what|which)\b.*\b(connected to|related to this|linked to)\b", re.I)),
    (INTENT_DEPENDENCY, 5, re.compile(r"\b(dependents?|dependenc(y|ies)|reverse dependenc)", re.I)),
    (INTENT_DEPENDENCY, 4, re.compile(r"\b(impact|blast radius|affected by)\b", re.I)),
    # history
    (INTENT_HISTORY, 6, re.compile(r"\b(recently|lately|last (week|month|year)|over time)\b", re.I)),
    (INTENT_HISTORY, 5, re.compile(r"\b(git|commit|commits|changelog|history|evolved|evolution)\b", re.I)),
    (INTENT_HISTORY, 5, re.compile(r"\bwhat\s+changed\b", re.I)),
    (INTENT_HISTORY, 4, re.compile(r"\b(who\s+(wrote|changed|added)|when\s+was)\b", re.I)),
    # read-only refusal (Part 16) — a request to change the repository
    (INTENT_READONLY, 7, re.compile(r"\b(write|create|generate|add|implement)\b.*\b(test|tests|function|class|method|endpoint|docstring)\b", re.I)),
    (INTENT_READONLY, 6, re.compile(r"\b(refactor|rewrite|convert|port|migrate)\b", re.I)),
    (INTENT_READONLY, 6, re.compile(r"\b(fix|patch|update|modify|change|delete|remove|rename)\b\s+(this|that|the|my)?\s*\b[\w./-]*\.(py|js|ts|tsx|jsx|java|go|rs)\b", re.I)),
    (INTENT_READONLY, 5, re.compile(r"\bunit tests?\b", re.I)),
    (INTENT_READONLY, 5, re.compile(r"\b(commit|push|apply|make)\s+(this|these|the)?\s*(change|changes|fix|edit)", re.I)),
    # flow / control flow (Part 10)
    (INTENT_FLOW, 7, re.compile(r"\bflows?\s+(through|into|from|to)\b", re.I)),
    (INTENT_FLOW, 7, re.compile(r"\b(request|data|control|execution|call)\s+flow\b", re.I)),
    (INTENT_FLOW, 6, re.compile(r"\bhow\s+does\b.*\b(flow|propagate|travel|move)\b", re.I)),
    (INTENT_FLOW, 6, re.compile(r"\b(trace|walk)\b.*\b(request|path|flow|lifecycle|pipeline)\b", re.I)),
    (INTENT_FLOW, 5, re.compile(r"\b(end[- ]to[- ]end|step[- ]by[- ]step|lifecycle|pipeline)\b", re.I)),
    (INTENT_FLOW, 5, re.compile(r"\bwhat happens (when|to)\b", re.I)),
    # concept / theme (Part 8)
    (INTENT_CONCEPT, 7, re.compile(r"\b(show|find|give)\s+me\s+everything\b", re.I)),
    (INTENT_CONCEPT, 6, re.compile(r"\beverything\s+(related\s+to|about|involving|touching)\b", re.I)),
    (INTENT_CONCEPT, 6, re.compile(r"\b(database|data|auth(entication|orisation|orization)?|logging|caching|api|service|persistence|security|config(uration)?)\s+layer\b", re.I)),
    (INTENT_CONCEPT, 5, re.compile(r"\ball\s+(the\s+)?(files|code|modules|classes)\s+(that|which|related|involved)\b", re.I)),
    # hotspots (Part 13)
    (INTENT_HOTSPOT, 7, re.compile(r"\bhot[\s-]?spots?\b", re.I)),
    (INTENT_HOTSPOT, 6, re.compile(r"\bmost\s+(connected|depended|imported|used|changed|modified|complex)\b", re.I)),
    (INTENT_HOTSPOT, 6, re.compile(r"\b(bottlenecks?|churn|most active)\b", re.I)),
    (INTENT_HOTSPOT, 5, re.compile(r"\bwhich\s+(files?|modules?)\s+(changed|change)\s+the\s+most\b", re.I)),
    (INTENT_HOTSPOT, 5, re.compile(r"\b(riskiest|most fragile|most critical)\b", re.I)),
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

#: A file named without its extension — "explain the indexing file", "the auth
#: module". Part 2 requires this: users do not type extensions, and without it
#: "Explain the indexing file" resolves to nothing and falls back to a blind
#: vector search. The captured stem is looked up in the graph like any other
#: mention, so a name that matches nothing simply drops out.
_BARE_FILE_MENTION = re.compile(
    r"\b(?:the|my|our|a|an)\s+([a-z_][a-z0-9_-]{2,})\s+(?:file|module|script|service|class|function|package)\b",
    re.I,
)

#: The same shape with the words reversed — "class AuthService", "module
#: app.rag.parser", "function called build_context".
#:
#: Two alternatives rather than one, because the loose form matches ordinary
#: English: "this function doing", "this file depend on", "this module
#: evolved" all put a verb where a name would go. So a bare word only counts
#: after an explicit "called"/"named"; otherwise the token must *look* like an
#: identifier — dotted, underscored, or capitalised.
_QUALIFIED_MENTION = re.compile(
    r"\b(?:file|module|class|function|symbol|package)\s+"
    r"(?:(?:called|named)\s+([A-Za-z_][\w.-]{2,})"
    r"|([A-Za-z_][\w-]*(?:[._][\w-]+)+|[A-Z][A-Za-z0-9]{2,}))\b"
)

#: A bare snake_case identifier — "What does create_access_token do?".
#:
#: ``_SNAKE_MENTION`` only fires when the name is followed by ``(``, which is
#: how it appears in code but not how anyone types it in a question. An
#: underscore inside a word is the signal: ordinary English does not contain
#: one, so this cannot match prose the way the bare-word patterns above can.
_IDENTIFIER_MENTION = re.compile(r"\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\b", re.I)

#: Themes rather than names: "the database layer", "everything related to
#: authentication". Feeds concept questions (Part 8), which search by topic
#: instead of resolving one node.
_CONCEPT_MENTION = re.compile(
    r"\b(?:related\s+to|about|involving|touching|everything\s+in)\s+(?:the\s+)?([a-z][\w-]{2,})",
    re.I,
)
_LAYER_MENTION = re.compile(r"\b([a-z][\w-]{2,})\s+layer\b", re.I)


@dataclass
class Classification:
    intent: str
    confidence: float
    scores: dict[str, int] = field(default_factory=dict)
    #: File paths and symbol names named in the question.
    mentions: list[str] = field(default_factory=list)
    #: True when the question leans on a pronoun and needs prior context.
    needs_context: bool = False
    #: Themes named in the question, for concept questions.
    concepts: list[str] = field(default_factory=list)


def extract_mentions(question: str) -> list[str]:
    """Concrete file or symbol names referenced in a question.

    Ordered by how precise the pattern is, because the first resolvable
    mention becomes the turn's focus: an explicit ``routing.py`` must win over
    a bare "the routing module" appearing later in the same sentence.
    """
    mentions: list[str] = []
    seen: set[str] = set()

    patterns = (
        _FILE_MENTION,
        _SYMBOL_MENTION,
        _SNAKE_MENTION,
        _IDENTIFIER_MENTION,
        _QUALIFIED_MENTION,
        _BARE_FILE_MENTION,
    )
    for pattern in patterns:
        for match in pattern.finditer(question):
            # _QUALIFIED_MENTION alternates between two capture groups; the
            # rest have one. Take whichever matched.
            value = next((g for g in match.groups() if g), "")
            if not value:
                continue
            # Sentence-initial capitals are not symbol names.
            if pattern is _SYMBOL_MENTION and (
                len(value) < 3 or value.lower() in _COMMON_WORDS
            ):
                continue
            # The bare forms match ordinary English ("the same file"), so they
            # are filtered harder than an explicit path needs to be.
            if pattern in (_BARE_FILE_MENTION, _QUALIFIED_MENTION) and (
                value.lower() in _COMMON_WORDS or value.lower() in _BARE_STOPWORDS
            ):
                continue
            if value not in seen:
                seen.add(value)
                mentions.append(value)
    return mentions


def extract_concepts(question: str) -> list[str]:
    """Themes a question is about, when it names no specific symbol."""
    concepts: list[str] = []
    seen: set[str] = set()
    for pattern in (_LAYER_MENTION, _CONCEPT_MENTION):
        for match in pattern.finditer(question):
            value = match.group(1).lower()
            if value in _COMMON_WORDS or value in _BARE_STOPWORDS:
                continue
            if value not in seen:
                seen.add(value)
                concepts.append(value)
    return concepts


#: Words that fill the "the ___ file" slot without naming anything.
_BARE_STOPWORDS = frozenset(
    {
        "same", "other", "another", "first", "last", "next", "previous",
        "whole", "entire", "main", "current", "above", "below", "following",
        "each", "every", "any", "some", "these", "those", "such", "right",
        "wrong", "correct", "biggest", "largest", "smallest", "only", "one",
        "two", "three", "new", "old", "big", "small", "top", "bottom",
        # Structural nouns: "about the class Foo" names Foo, not "class".
        "file", "files", "module", "modules", "class", "classes", "function",
        "functions", "method", "methods", "symbol", "symbols", "package",
        "packages", "folder", "directory", "script",
    }
)


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
        # Language and ecosystem names. "How many Python files are there?"
        # otherwise resolves focus onto whatever directory happens to be
        # called `python_types`, and the answer is then offered as a place to
        # explore — the exact false positive Part 5 warns against.
        "python", "javascript", "typescript", "java", "golang", "rust",
        "ruby", "php", "kotlin", "swift", "scala", "html", "css", "sql",
        "json", "yaml", "node", "react", "vue", "django", "flask",
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
            concepts=extract_concepts(text),
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
        concepts=extract_concepts(text),
    )

"""Deterministic, offline answer checker.

It mirrors the signature of Latent's `runlatent.check(prompt, answer,
context=...)` (read from the runlatent 0.3.9 wheel on PyPI, not from a
live call): prompt, answer, and source context in, a verdict plus one
risk score per sentence out. The SDK itself renders context as a
`Knowledge:` line in the messages and posts messages, output, and
context to `/score`; this checker does not post anything.

What it checks, sentence by sentence, against the supplied context:
  * unsupported number: a number in the answer that appears nowhere in
    the context (amounts, percents, counts).
  * wrong date: a date in the answer that appears nowhere in the context.
  * invented name: a capitalized name in the answer that appears nowhere
    in the context.
  * unsupported claim: a factual sentence that shares no content words,
    number, date or name with any context sentence.

It is a checker, not a reader model. It cannot catch a hallucination
that reuses only numbers, dates and names already in the context but
combines them wrongly. That limit is stated in the README, not hidden.
"""

from __future__ import annotations

import re
import time
import uuid

THRESHOLD = 0.70

RISK = {
    "unsupported_number": 0.90,
    "wrong_date": 0.85,
    "invented_name": 0.85,
    "unsupported_claim": 0.60,
}

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "is", "are", "was", "were", "it", "its", "this", "that", "as", "at",
    "by", "from", "per", "will", "has", "have", "had", "be", "been",
    "our", "we", "you", "your", "their", "they", "he", "she", "his",
    "her", "not", "no", "yes", "if", "then", "than", "also", "about",
    "into", "over", "under", "after", "before", "between", "during",
    "which", "who", "what", "when", "where", "how", "all", "each",
    "total", "approximately", "around", "about", "please", "note",
}

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
NUMBER_RE = re.compile(r"\$?\d[\d,]*(\.\d+)?%?")
DATE_RE = re.compile(
    r"\b(?:\d{4}-\d{2}-\d{2}"
    r"|(?:January|February|March|April|May|June|July|August|September"
    r"|October|November|December)\s+\d{1,2},?\s+\d{4}"
    r"|\d{1,2}\s+(?:January|February|March|April|May|June|July|August"
    r"|September|October|November|December)\s+\d{4}"
    r"|Q[1-4]\s+\d{4})\b"
)
NAME_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")
WORD_RE = re.compile(r"[a-z0-9]+")


def _norm_number(raw: str) -> str:
    return raw.replace("$", "").replace(",", "").rstrip("%")


def _numbers(text: str) -> set:
    return {_norm_number(m.group(0)) for m in NUMBER_RE.finditer(text)}


def _dates(text: str) -> set:
    found = set()
    for m in DATE_RE.finditer(text):
        found.add(re.sub(r"\s+", " ", m.group(0)).replace(",", ""))
    return found


def _names(text: str) -> set:
    found = set()
    for m in NAME_RE.finditer(text):
        name = m.group(0)
        first = name.split()[0]
        if first in {"The", "This", "That", "These", "Those", "Our", "Your"}:
            name = " ".join(name.split()[1:])
            if " " not in name:
                continue
        found.add(name)
    return found


def _content_words(text: str) -> set:
    return {w for w in WORD_RE.findall(text.lower()) if w not in STOPWORDS}


def _split_sentences(answer: str) -> list:
    parts = [s.strip() for s in SENTENCE_SPLIT.split(answer.strip()) if s.strip()]
    return parts or ([answer.strip()] if answer.strip() else [])


def score_answer(prompt: str, answer: str, context: str, threshold: float = THRESHOLD) -> dict:
    """Score one answer against its source context. Pure function, no I/O."""
    started = time.perf_counter()
    ctx_numbers = _numbers(context)
    ctx_dates = _dates(context)
    ctx_names = _names(context)
    ctx_sentences = _split_sentences(context)
    ctx_word_sets = [_content_words(s) for s in ctx_sentences]

    sentences = []
    for sent in _split_sentences(answer):
        reasons = []
        for num in _numbers(sent):
            if num not in ctx_numbers:
                reasons.append({"kind": "unsupported_number", "detail": num})
        for date in _dates(sent):
            if date not in ctx_dates:
                reasons.append({"kind": "wrong_date", "detail": date})
        for name in _names(sent):
            if name not in ctx_names:
                reasons.append({"kind": "invented_name", "detail": name})
        if not reasons:
            words = _content_words(sent)
            shared = any(words & ctx_words for ctx_words in ctx_word_sets)
            has_anchor = bool(_numbers(sent) or _dates(sent) or _names(sent))
            if words and not shared and not has_anchor:
                reasons.append({"kind": "unsupported_claim", "detail": sent[:60]})
        risk = max((RISK[r["kind"]] for r in reasons), default=0.0)
        sentences.append({"text": sent, "risk": risk, "reasons": reasons})

    risk = max((s["risk"] for s in sentences), default=0.0)
    verdict = "escalate" if risk >= threshold else "pass"
    latency_ms = (time.perf_counter() - started) * 1000.0
    return {
        "request_id": str(uuid.uuid4()),
        "verdict": verdict,
        "flagged": verdict == "escalate",
        "risk": risk,
        "sentences": sentences,
        "latency_ms": round(latency_ms, 3),
    }

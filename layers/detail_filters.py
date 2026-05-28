"""
Shared quality filters for detail/keyword extraction layers.
Keeps output short, technical, and non-generic.
"""

import re
from typing import List

# Hard caps (per chunk / per file)
MAX_KEYWORDS_PER_ITEM = 5
MAX_DETAILS_PER_ITEM = 4
MAX_KEYWORD_CHARS = 40
MAX_DETAIL_CHARS = 110
MAX_KEYWORD_WORDS = 3
MAX_DETAIL_WORDS = 16

# Global caps after merge
MAX_MERGED_KEYWORDS = 40
MAX_MERGED_DETAILS = 60

_GENERIC_KEYWORD_BLOCKLIST = {
    "project", "system", "user", "assistant", "conversation", "memory",
    "layer", "pipeline", "code", "chat", "file", "data", "work", "goal",
    "problem", "question", "answer", "topic", "context", "summary",
}

_DETAIL_SIGNAL_RE = re.compile(
    r"(\d|=\s*[\d\"']|\.py\b|/|max_|min_|model|config|api|token|retry|"
    r"timeout|limit|epoch|lr|batch|ema|groq|llama|qwen|gpt|error|failed|"
    r"decided|fixed|uses |import )",
    re.IGNORECASE,
)

# Obvious code-snippet markers (language-agnostic, not format-specific)
_CODE_SNIPPET_MARKERS = re.compile(
    r"(\(|\)|self\.|\.get\(|\.json\(|\.encode\(|requests\.|logging\.|"
    r"FileHandler|StreamHandler|match\.group|import )",
    re.IGNORECASE,
)


def filter_keywords(items: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for raw in items:
        s = str(raw).strip()
        if not s or len(s) > MAX_KEYWORD_CHARS:
            continue
        words = s.split()
        if len(words) > MAX_KEYWORD_WORDS:
            continue
        key = s.lower()
        if key in _GENERIC_KEYWORD_BLOCKLIST:
            continue
        if key in seen:
            continue
        # Keyword must look technical (not a plain English phrase)
        if not _looks_technical_keyword(s):
            continue
        seen.add(key)
        out.append(s)
        if len(out) >= MAX_KEYWORDS_PER_ITEM:
            break
    return out


def filter_details(items: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for raw in items:
        s = str(raw).strip()
        if not s or len(s) > MAX_DETAIL_CHARS:
            continue
        if len(s.split()) > MAX_DETAIL_WORDS:
            continue
        if not _DETAIL_SIGNAL_RE.search(s):
            continue
        if _is_generic_detail(s):
            continue
        key = re.sub(r"\s+", " ", s.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
        if len(out) >= MAX_DETAILS_PER_ITEM:
            break
    return out


def cap_merged_lists(keywords: List[str], details: List[str]) -> tuple[List[str], List[str]]:
    return keywords[:MAX_MERGED_KEYWORDS], details[:MAX_MERGED_DETAILS]


def sanitize_code_detail_output(
    keywords: List[str], details: List[str]
) -> tuple[List[str], List[str]]:
    """
    Light post-filter after LLM selection.
    Does NOT parse code structure — only drops obvious garbage and enforces caps.
    """
    kw_out: List[str] = []
    dt_out: List[str] = []
    seen_kw, seen_dt = set(), set()

    for raw in keywords:
        s = str(raw).strip()
        if not s or len(s) > MAX_KEYWORD_CHARS or len(s.split()) > MAX_KEYWORD_WORDS:
            continue
        if _CODE_SNIPPET_MARKERS.search(s):
            continue
        key = s.lower()
        if key in seen_kw:
            continue
        seen_kw.add(key)
        kw_out.append(s)
        if len(kw_out) >= MAX_KEYWORDS_PER_ITEM:
            break

    for raw in details:
        s = str(raw).strip()
        if not s or len(s) > MAX_DETAIL_CHARS or len(s.split()) > MAX_DETAIL_WORDS:
            continue
        if _CODE_SNIPPET_MARKERS.search(s):
            continue
        if "=" not in s:
            continue
        key = re.sub(r"\s+", " ", s.lower())
        if key in seen_dt:
            continue
        seen_dt.add(key)
        dt_out.append(s)
        if len(dt_out) >= MAX_DETAILS_PER_ITEM:
            break

    return kw_out, dt_out


def _looks_technical_keyword(s: str) -> bool:
    if re.search(r"[\d_./\\-]", s):
        return True
    if re.search(r"[A-Z]{2,}", s):
        return True
    if any(x in s.lower() for x in (
        "llm", "api", "json", "chunk", "merge", "groq", "pytorch", "cuda",
        "config", "layer", "parser", "tokenizer", "embedding",
    )):
        return True
    return False


def _is_generic_detail(s: str) -> bool:
    lower = s.lower()
    generic_starts = (
        "the user", "user is", "this chunk", "this conversation",
        "they are", "we are", "it is about", "discussion about",
    )
    if any(lower.startswith(p) for p in generic_starts):
        return True
    if lower.count(" ") > 12 and not re.search(r"\d", s):
        return True
    return False
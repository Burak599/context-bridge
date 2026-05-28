import json
import re
from typing import Dict, List

from layers.llm_client import LLMClient
from layers.detail_filters import (
    filter_details,
    filter_keywords,
    cap_merged_lists,
)
import config


DETAIL_EXTRACTOR_SYSTEM_PROMPT = """You are a strict technical fact extractor for ONE conversation chunk.
You are not a summarizer and not an interpreter. You only copy/extract facts that are explicitly present in the chunk.

Return ONLY JSON:
{
  "keywords": ["..."],
  "details": ["..."]
}

HARD LIMITS (never exceed):
- keywords: MAX 5 items, each MAX 3 words, MAX 40 characters
- details: MAX 4 items, each MAX 16 words, MAX 110 characters

What to include:
- keywords: exact technical terms only (file names, model names, APIs, metrics, env keys)
- details: atomic facts with concrete values (numbers, versions, paths, configs, explicit decisions)

Anti-hallucination rules:
- Do NOT infer missing facts.
- Do NOT generalize from similar projects or common software patterns.
- Do NOT invent file names, paths, models, numbers, decisions, bugs, TODOs, or user goals.
- Do NOT include a detail unless the exact fact is directly supported by words in the chunk.
- If a fact is ambiguous, omit it.
- If you are unsure whether something is explicit, omit it.

What to EXCLUDE (critical):
- generic narration, emotions, greetings, summaries of the whole chunk
- vague phrases without numbers/names (e.g. "user wants better system")
- repeating the same idea in different words

Each detail MUST contain at least one concrete anchor:
a number, version, path, model name, parameter, command, or explicit decision/failure.

If nothing concrete exists in this chunk, return:
{"keywords": [], "details": []}

Return ONLY JSON. No markdown. No thinking tags."""


def _debug(msg: str) -> None:
    print(f"[Detail Layer | DEBUG] {msg}")


class DetailKeywordExtractorLayer:
    """
    Extracts high-signal keywords and atomic details from each chunk.
    """

    def __init__(self, llm_client: LLMClient, debug: bool = True):
        self.llm = llm_client
        self.debug = debug

    def extract_all(self, chunk_texts: List[str]) -> List[Dict]:
        if self.debug:
            _debug(f"Model: {config.DETAIL_MODEL}")
            _debug(f"Input chunk count: {len(chunk_texts)}")

        if not chunk_texts:
            _debug("ABORT: chunk_texts is empty — nothing to extract.")
            return []

        results: List[Dict] = []
        empty_count = 0
        for i, chunk_text in enumerate(chunk_texts, start=1):
            if self.debug:
                preview = chunk_text[:80].replace("\n", " ") + ("..." if len(chunk_text) > 80 else "")
                _debug(f"Chunk {i} input: {len(chunk_text)} chars | preview: {preview!r}")

            if not chunk_text or not chunk_text.strip():
                _debug(f"Chunk {i} SKIP: empty text")
                results.append({"chunk_number": i, "keywords": [], "details": []})
                empty_count += 1
                continue

            parsed = self._extract_single(chunk_number=i, chunk_text=chunk_text)
            results.append(parsed)
            if not parsed.get("keywords") and not parsed.get("details"):
                empty_count += 1

        if self.debug:
            _debug(
                f"Extraction done: {len(results)} chunks, "
                f"{empty_count} empty after filter"
            )
        return results

    def _extract_single(self, chunk_number: int, chunk_text: str) -> Dict:
        user_message = f"Extract keywords and details:\n\n{chunk_text}"
        if self.debug:
            _debug(f"Chunk {chunk_number} LLM request: {len(user_message)} chars")

        try:
            response = self.llm.chat(
                config.DETAIL_MODEL,
                DETAIL_EXTRACTOR_SYSTEM_PROMPT,
                user_message,
            )
            if self.debug:
                _debug(f"Chunk {chunk_number} LLM response: {len(response)} chars")
            parsed = self._parse_response(response, chunk_number)
        except Exception as e:
            _debug(f"Chunk {chunk_number} LLM EXCEPTION: {type(e).__name__}: {e}")
            parsed = {"chunk_number": chunk_number, "keywords": [], "details": []}

        before_kw, before_dt = len(parsed["keywords"]), len(parsed["details"])
        parsed["keywords"] = filter_keywords(parsed["keywords"])
        parsed["details"] = filter_details(parsed["details"])
        if self.debug and (before_kw != len(parsed["keywords"]) or before_dt != len(parsed["details"])):
            _debug(
                f"Chunk {chunk_number} filter: kw {before_kw}->{len(parsed['keywords'])}, "
                f"dt {before_dt}->{len(parsed['details'])}"
            )
        return parsed

    def _parse_response(self, response: str, chunk_number: int) -> Dict:
        empty = {"chunk_number": chunk_number, "keywords": [], "details": []}
        if not response or not response.strip():
            _debug(f"Chunk {chunk_number} PARSE FAIL: empty LLM response")
            return empty

        # Strip <think>...</think> chain-of-thought blocks if present
        cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
        # If an unclosed <think> tag remains, discard everything from that point on
        if "<think>" in cleaned:
            cleaned = cleaned[: cleaned.index("<think>")].strip()

        if not cleaned:
            _debug(f"Chunk {chunk_number} PARSE FAIL: only thinking content")
            return empty

        # Remove markdown code fences if the model wrapped the JSON
        cleaned = re.sub(r"```(?:json)?", "", cleaned).replace("```", "").strip()
        # Extract the outermost JSON object
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start == -1 or end <= start:
            _debug(f"Chunk {chunk_number} PARSE FAIL: no JSON. Head: {cleaned[:300]!r}")
            return empty

        json_slice = cleaned[start:end]
        try:
            data = json.loads(json_slice)
            keywords = data.get("keywords", [])
            details = data.get("details", [])
            if not isinstance(keywords, list):
                keywords = []
            if not isinstance(details, list):
                details = []

            keywords = [str(x).strip() for x in keywords if str(x).strip()]
            details = [str(x).strip() for x in details if str(x).strip()]

            if self.debug:
                _debug(
                    f"Chunk {chunk_number} raw parse: {len(keywords)} kw, {len(details)} dt"
                )
            return {
                "chunk_number": chunk_number,
                "keywords": keywords,
                "details": details,
            }
        except json.JSONDecodeError as e:
            _debug(f"Chunk {chunk_number} JSON ERROR: {e} | {json_slice[:400]!r}")
            return empty


class DetailKeywordMergeLayer:
    """Merges chunk-level keywords/details without LLM."""

    def merge(self, extracted_chunks: List[Dict], debug: bool = True) -> Dict:
        if debug:
            _debug(f"Merge input: {len(extracted_chunks)} chunks")

        keyword_map: Dict[str, str] = {}
        detail_map: Dict[str, str] = {}

        for chunk in extracted_chunks:
            for kw in chunk.get("keywords", []):
                key = self._norm(kw)
                if key and key not in keyword_map:
                    keyword_map[key] = kw
            for dt in chunk.get("details", []):
                key = self._norm(dt)
                if key and key not in detail_map:
                    detail_map[key] = dt

        keywords = sorted(keyword_map.values(), key=lambda x: x.lower())
        details = list(detail_map.values())
        keywords, details = cap_merged_lists(keywords, details)

        merged = {
            "keywords": keywords,
            "details": details,
            "memory_text": self._to_memory_text(keywords, details),
        }

        if debug:
            _debug(
                f"Merge output: {len(keywords)} keywords, {len(details)} details"
            )
        return merged

    def _norm(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    def _to_memory_text(self, keywords: List[str], details: List[str]) -> str:
        lines = [
            "DETAIL KEYWORD MEMORY",
            "",
            "KEYWORDS:",
            ", ".join(keywords) if keywords else "",
            "",
            "DETAILS:",
        ]
        lines.extend(f"- {d}" for d in details)
        return "\n".join(lines).strip()
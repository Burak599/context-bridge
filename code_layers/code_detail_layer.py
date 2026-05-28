import json
import re
from typing import Dict, List

from layers.llm_client import LLMClient
import config


MAX_VARIABLES_PER_FILE = 5

# System prompt for the code detail extractor LLM.
# Instructs the model to pick only the most important variables/parameters per file.
CODE_DETAIL_EXTRACTOR_SYSTEM_PROMPT = """You are an expert code analyst. You read ONE source file and pick ONLY the most important variables/parameters and their values.
You only extract values that are explicitly visible in the provided file. Do not infer or invent missing values.

Return ONLY JSON:
{
  "variables": [
    {"name": "identifier", "value": "literal_value", "kind": "variable|parameter"}
  ]
}

HARD LIMIT: max 5 variables/parameters per file.

Your job is to JUDGE importance — not dump code. Ask yourself:
"If a new developer joins, which exact values must they know to understand how this file works?"

INCLUDE only high-signal items:
- API/model names: LAYER_3_MODEL="qwen/qwen3-32b"
- Limits and sizes: MAX_FILE_CHARS=30000, top_k=5, batch_size=64
- Timeouts, retries, thresholds: timeout=3, MAX_RETRIES=6
- Env keys and paths: GROQ_API_KEY, index_path="./data/faiss"
- Default parameters that control behavior: temperature=0, overlap_messages=3

EXCLUDE:
- Temporary/local runtime variables like response, result, data, text, content, i, item, score
- Logging boilerplate, UI text, prose, comments, explanations
- Method calls or expressions that are not meaningful configuration
- Generic identifiers with no project/config meaning
- Any variable, parameter, value, file path, model name, limit, or config that is not directly present in the file

Format rules:
- Output ONLY fields name, value, kind.
- name must be the exact variable/parameter identifier.
- value must be the exact literal value as text when visible.
- kind must be either "variable" or "parameter".
- If the value is not visible in the file, omit the item instead of guessing.
- If this file has no important variable/parameter, return {"variables": []}.
- Return ONLY JSON. No markdown. No thinking tags."""


def _debug(msg: str) -> None:
    print(f"[Code Detail Layer | DEBUG] {msg}")


class CodeDetailExtractorLayer:
    """LLM-driven extraction of max 5 important variables/parameters per file."""

    def __init__(self, llm_client: LLMClient, debug: bool = True):
        self.llm = llm_client
        self.debug = debug

    def extract_all(self, files: List[Dict]) -> List[Dict]:
        if self.debug:
            _debug(f"Model: {config.CODE_DETAIL_MODEL}")
            _debug(f"Input file count: {len(files)}")

        if not files:
            _debug("ABORT: no files to process")
            return []

        results: List[Dict] = []
        empty_count = 0
        for i, file_info in enumerate(files, start=1):
            path = file_info.get("path", "?")
            if self.debug:
                content = file_info.get("content", "") or ""
                _debug(
                    f"File {i} ({path}): {len(content)} chars, "
                    f"truncated={file_info.get('truncated', False)}"
                )
            parsed = self._extract_single(i, file_info)
            results.append(parsed)
            if not parsed.get("variables"):
                empty_count += 1

        if self.debug:
            _debug(f"Done: {len(results)} files, {empty_count} empty")
        return results

    def _extract_single(self, index: int, file_info: Dict) -> Dict:
        path = file_info["path"]
        content = file_info.get("content", "") or ""

        # Skip files with no meaningful content
        if not content.strip():
            _debug(f"{path} SKIP: empty file content")
            return {"index": index, "file": path, "variables": []}

        user_message = (
            f"File: {path}\n\n"
            f"Extract max {MAX_VARIABLES_PER_FILE} important variables/parameters from this file:\n\n"
            f"```{file_info['extension'].lstrip('.')}\n"
            f"{content}\n"
            f"```"
        )

        try:
            if self.debug:
                _debug(f"{path} LLM request: {len(user_message)} chars")
            response = self.llm.chat(
                config.CODE_DETAIL_MODEL,
                CODE_DETAIL_EXTRACTOR_SYSTEM_PROMPT,
                user_message,
            )
            if self.debug:
                _debug(f"{path} LLM response: {len(response)} chars")
            parsed = self._parse_response(response, index, path)
        except Exception as e:
            _debug(f"{path} LLM EXCEPTION: {type(e).__name__}: {e}")
            parsed = {"index": index, "file": path, "variables": []}

        # Sanitize: enforce strict variable format and apply per-file cap
        before_vars = len(parsed["variables"])
        parsed["variables"] = _enforce_variables_only(parsed["variables"])

        if self.debug:
            _debug(f"{path} sanitize: vars {before_vars}->{len(parsed['variables'])}")
            if parsed["variables"]:
                _debug(f"{path} kept vars={parsed['variables']}")
            elif before_vars:
                _debug(f"{path} ALL REJECTED by sanitize")

        return parsed

    def _parse_response(self, response: str, index: int, file_path: str) -> Dict:
        empty = {"index": index, "file": file_path, "variables": []}

        if not response or not response.strip():
            _debug(f"{file_path} PARSE FAIL: empty response")
            return empty

        # Remove <think>...</think> chain-of-thought blocks if present
        cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
        # Also truncate at any unclosed <think> tag
        if "<think>" in cleaned:
            cleaned = cleaned[: cleaned.index("<think>")].strip()
        if not cleaned:
            _debug(f"{file_path} PARSE FAIL: only thinking tags. Head: {response[:200]!r}")
            return empty

        # Strip markdown code fences if the model wrapped the JSON
        cleaned = re.sub(r"```(?:json)?", "", cleaned).replace("```", "").strip()

        # Extract the outermost JSON object
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start == -1 or end <= start:
            _debug(f"{file_path} PARSE FAIL: no JSON. Head: {cleaned[:300]!r}")
            return empty

        json_slice = cleaned[start:end]
        try:
            data = json.loads(json_slice)
            variables = data.get("variables", [])
            if not isinstance(variables, list):
                variables = []
            if self.debug:
                _debug(f"{file_path} raw LLM: {len(variables)} vars")
            return {
                "index": index,
                "file": file_path,
                "variables": variables,
            }
        except json.JSONDecodeError as e:
            _debug(f"{file_path} JSON ERROR: {e} | {json_slice[:400]!r}")
            return empty


class CodeDetailMergeLayer:
    """Deterministic merge for per-file code details."""

    def merge(self, extracted_files: List[Dict], debug: bool = True) -> Dict:
        if debug:
            _debug(f"Merge input: {len(extracted_files)} files")

        file_blocks: List[Dict] = []
        flat_variables: List[Dict] = []

        for item in extracted_files:
            file_path = item.get("file", "")
            file_variables: List[Dict] = []
            seen = set()

            for var in item.get("variables", []):
                name = str(var.get("name", "")).strip()
                value = str(var.get("value", "")).strip()
                kind = str(var.get("kind", "")).strip() or "variable"

                # Skip entries with missing fields or invalid kind
                if not name or not value or kind not in {"variable", "parameter"}:
                    continue

                # Deduplicate by normalized name+value+kind key
                key = _norm(f"{name}={value}:{kind}")
                if key in seen:
                    continue
                seen.add(key)

                normalized = {
                    "file": file_path,
                    "name": name,
                    "value": value,
                    "kind": kind,
                }
                file_variables.append(normalized)
                flat_variables.append(normalized)

                # Respect per-file variable cap
                if len(file_variables) >= MAX_VARIABLES_PER_FILE:
                    break

            file_blocks.append({"file": file_path, "variables": file_variables})

        merged = {
            "files": file_blocks,
            "variables": flat_variables,
            "memory_text": _to_memory_text(file_blocks),
        }

        if debug:
            _debug(f"Merge output: {len(flat_variables)} variables")

        return merged


def _norm(text: str) -> str:
    """Normalize whitespace and lowercase a string for deduplication."""
    return re.sub(r"\s+", " ", text.strip().lower())


def _to_memory_text(file_blocks: List[Dict]) -> str:
    """Serialize merged file blocks into a human-readable memory text block."""
    lines = [
        "CODE VARIABLE PARAMETER MEMORY",
        "",
        "VARIABLES:",
    ]

    for block in file_blocks:
        file_path = block.get("file", "")
        variables = block.get("variables", [])
        lines.extend(["", f"{file_path} script"])
        if not variables:
            lines.append("(no important variables/parameters found)")
            continue

        for v in variables[:MAX_VARIABLES_PER_FILE]:
            label = "parameter" if v.get("kind") == "parameter" else "variable"
            lines.append(f"{label} {v.get('name', '')}={v.get('value', '')}")

    return "\n".join(lines).strip()


# Regex to validate that a variable name is a proper Python identifier
_STRICT_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _enforce_variables_only(variables: List[Dict]) -> List[Dict]:
    """
    Sanitizes the raw LLM variable list:
    - Drops entries with missing/invalid fields
    - Enforces strict identifier format for names
    - Deduplicates by normalized key
    - Caps output at MAX_VARIABLES_PER_FILE
    """
    out: List[Dict] = []
    seen = set()
    for item in variables:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        value = str(item.get("value", "")).strip()
        kind = str(item.get("kind", "")).strip().lower()
        if not name or not value:
            continue
        if not _STRICT_NAME_RE.match(name):
            continue
        if kind not in {"variable", "parameter"}:
            continue
        key = _norm(f"{name}={value}:{kind}")
        if key in seen:
            continue
        seen.add(key)
        out.append({"name": name, "value": value, "kind": kind})
        if len(out) >= MAX_VARIABLES_PER_FILE:
            break
    return out
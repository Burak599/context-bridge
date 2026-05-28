# code_layers/code_merge_layer.py

import json
import re
from typing import List, Dict
from layers.llm_client import LLMClient
import config

# System prompt for merging file summaries and a relation map into one unified project summary.
CODE_MERGE_SYSTEM_PROMPT = """You are a data consolidation expert. You will receive:
1. A list of individual file summaries (from a code analyzer)
2. A relationship map between those files (from a relation analyzer)

Merge them into a single unified JSON object with exactly these fields:

{
  "project_name": "inferred project name from entry points or file names",
  "purpose": "one sentence describing what this project does",
  "architecture": "one paragraph describing the overall structure and how the layers/modules interact",
  "entry_points": ["main entry point files"],
  "hubs": ["files that are central dependencies, imported by many others"],
  "core_modules": ["most critical files for the project to function"],
  "files": [
    {
      "file": "relative file path",
      "purpose": "what this file does",
      "classes": ["class names"],
      "functions": ["function names"],
      "dependencies": ["internal dependencies"]
    }
  ],
  "relations": ["fileA → fileB: reason"]
}

Rules:
- Do not duplicate information. If architecture is already in the relation map, refine it, don't repeat.
- "files" should include all files from the analyzer output.
- Be concise. No fluff.
- Return ONLY the JSON object. No markdown, no explanation, no extra text."""

# System prompt for merging multiple partial summaries into a single final summary.
CODE_MERGE_PARTIAL_PROMPT = """You are a data consolidation expert. You will receive multiple partial project summaries.
Merge them into a single unified JSON object with exactly these fields:

{
  "project_name": "inferred project name",
  "purpose": "one sentence describing what this project does",
  "architecture": "one paragraph describing the overall structure",
  "entry_points": ["main entry point files"],
  "hubs": ["central dependency files"],
  "core_modules": ["most critical files"],
  "files": [{"file": "path", "purpose": "what it does", "classes": [], "functions": [], "dependencies": []}],
  "relations": ["fileA → fileB: reason"]
}

Rules:
- Combine all files from all summaries, no duplicates.
- Merge architecture descriptions into one coherent paragraph.
- Return ONLY the JSON object. No markdown, no explanation, no extra text."""

GROUP_SIZE = 8


class CodeMergeLayer:
    """
    Hierarchically merges file summaries and the relation map.
    If the file count exceeds GROUP_SIZE, splits them into groups,
    merges each group separately, then sends the intermediate results to a final merge.
    """

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def merge(self, analyses: List[Dict], relation_map: Dict) -> Dict:
        if not analyses:
            return {}

        print(f"[Layer 4] Merging {len(analyses)} file summaries + relation map...")

        # If file count is small enough, merge directly
        if len(analyses) <= GROUP_SIZE:
            return self._merge_with_relation(analyses, relation_map)

        # Split into groups — the first group is sent together with the relation map
        groups = [analyses[i:i+GROUP_SIZE] for i in range(0, len(analyses), GROUP_SIZE)]
        print(f"[Layer 4] {len(groups)} groups created (size {GROUP_SIZE})")

        intermediate = []
        for i, group in enumerate(groups):
            print(f"[Layer 4] Merging group {i+1}/{len(groups)} ({len(group)} files)...")
            # Only attach the relation map to the first group
            if i == 0:
                result = self._merge_with_relation(group, relation_map)
            else:
                result = self._merge_files_only(group)
            if result:
                intermediate.append(result)

        # Send all intermediate results to the final merge
        print(f"[Layer 4] Sending {len(intermediate)} intermediate results to final merge...")
        return self._merge_partials(intermediate)

    # ------------------------------------------------------------------
    # Private merge methods
    # ------------------------------------------------------------------

    def _merge_with_relation(self, analyses: List[Dict], relation_map: Dict) -> Dict:
        payload = {
            "file_summaries": analyses,
            "relation_map":   relation_map,
        }
        user_message = (
            f"Merge these file summaries and relation map into one unified project summary:\n\n"
            f"{json.dumps(payload, indent=2)}"
        )
        print(f"[Layer 4] First merge: sending {len(user_message)} characters...")
        try:
            response = self.llm.chat(config.CODE_MERGE_MODEL, CODE_MERGE_SYSTEM_PROMPT, user_message)
            return self._parse_response(response)
        except Exception as e:
            print(f"[Layer 4] Error: {e}")
            return {}

    def _merge_files_only(self, analyses: List[Dict]) -> Dict:
        user_message = (
            f"Merge these file summaries into one unified project summary:\n\n"
            f"{json.dumps(analyses, indent=2)}"
        )
        print(f"[Layer 4] Group merge: sending {len(user_message)} characters...")
        try:
            response = self.llm.chat(config.CODE_MERGE_MODEL, CODE_MERGE_SYSTEM_PROMPT, user_message)
            return self._parse_response(response)
        except Exception as e:
            print(f"[Layer 4] Error: {e}")
            return {}

    def _merge_partials(self, partials: List[Dict]) -> Dict:
        if len(partials) == 1:
            return partials[0]
        user_message = (
            f"Merge these partial project summaries into one final summary:\n\n"
            f"{json.dumps(partials, indent=2)}"
        )
        print(f"[Layer 4] Final merge: sending {len(user_message)} characters...")
        try:
            response = self.llm.chat(config.CODE_MERGE_MODEL, CODE_MERGE_PARTIAL_PROMPT, user_message)
            return self._parse_response(response)
        except Exception as e:
            print(f"[Layer 4] Error: {e}")
            return {}

    def _parse_response(self, response: str) -> Dict:
        # Strip <think>...</think> chain-of-thought blocks if present
        cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
        # Remove markdown code fences if the model wrapped the JSON
        cleaned = re.sub(r"```(?:json)?", "", cleaned).replace("```", "").strip()

        # Extract the outermost JSON object
        start = cleaned.find("{")
        end   = cleaned.rfind("}") + 1
        if start != -1 and end > start:
            cleaned = cleaned[start:end]

        try:
            data = json.loads(cleaned)
            return {
                "project_name": data.get("project_name", ""),
                "purpose":      data.get("purpose", ""),
                "architecture": data.get("architecture", ""),
                "entry_points": data.get("entry_points", []),
                "hubs":         data.get("hubs", []),
                "core_modules": data.get("core_modules", []),
                "files":        data.get("files", []),
                "relations":    data.get("relations", []),
            }
        except json.JSONDecodeError as e:
            print(f"[Layer 4] JSON parse error: {e}")
            print(f"[Layer 4] Response tail: {response[-200:]}")
            return {}
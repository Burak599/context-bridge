# layers/merge_layer.py
import json
import re
from typing import List, Dict
from layers.llm_client import LLMClient
import config

MERGE_SYSTEM_PROMPT = """You are a data consolidation expert. You will receive multiple JSON analysis objects from different parts of a conversation.
Merge them into a single unified JSON object with exactly these fields:
{
  "topic": "one sentence describing the overall conversation",
  "decisions": [
    {
      "what": "the decision or solution that was reached",
      "why": "the reason or problem that led to this decision, empty string if not explicitly stated",
      "how": "the implementation or method chosen, empty string if not explicitly stated"
    }
  ],
  "open_questions": ["all unresolved questions that still need answers"],
  "progress": "one paragraph summarizing everything accomplished in the conversation",
  "context": "all important background information about the user and their project"
}
Rules:
- Remove duplicate decisions and questions.
- If two decisions contradict each other, keep the more recent one.
- Combine context from all chunks into one coherent description.
- Return ONLY the JSON object. No markdown, no explanation, no extra text."""

GROUP_SIZE = 8


class MergeLayer:
    """
    Splits chunk analyses into groups of GROUP_SIZE,
    merges each group into a small JSON,
    returns the list of group JSONs to FinalMemoryLayer.
    """

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def merge(self, analyses: List[Dict]) -> List[Dict]:
        """
        Args:
            analyses: ChunkAnalyzePipeline output
        Returns:
            List of merged JSON dicts, one per group.
            FinalMemoryLayer will summarize each and combine.
        """
        if not analyses:
            return []

        if len(analyses) <= GROUP_SIZE:
            print(f"[Layer 4] {len(analyses)} chunks → single group merge...")
            result = self._merge_group(analyses, label="Group 1")
            return [result] if result else []

        groups = [analyses[i:i+GROUP_SIZE] for i in range(0, len(analyses), GROUP_SIZE)]
        print(f"[Layer 4] {len(analyses)} chunks → {len(groups)} groups (size {GROUP_SIZE})")

        results = []
        for i, group in enumerate(groups):
            print(f"[Layer 4] Merging group {i+1}/{len(groups)} ({len(group)} chunks)...")
            result = self._merge_group(group, label=f"Group {i+1}")
            if result:
                results.append(result)

        print(f"[Layer 4] ✓ {len(results)} group JSON(s) created.")
        return results

    def _merge_group(self, analyses: List[Dict], label: str = "") -> Dict:
        if not analyses:
            return {}
        if len(analyses) == 1:
            return analyses[0]

        user_message = f"Merge these conversation analyses into one:\n\n{json.dumps(analyses, indent=2)}"
        print(f"[Layer 4] {label}: sending {len(user_message)} characters...")
        try:
            response = self.llm.chat(
                config.LAYER_4_MODEL,
                MERGE_SYSTEM_PROMPT,
                user_message,
            )
            return self._parse_response(response)
        except Exception as e:
            print(f"[Layer 4] {label} error: {e}")
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
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            print(f"[Layer 4] JSON parse error: {e}")
            print(f"[Layer 4] Response tail: {response[-200:]}")
            return {}
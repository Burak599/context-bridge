# layers/final_memory.py

import re
import json
from typing import Dict, List, Union
from layers.llm_client import LLMClient
import config

PARTIAL_MEMORY_SYSTEM_PROMPT = """You are a memory writer. You will receive a structured JSON summary of part of a conversation.
Write a concise plain text memory paragraph that captures everything important from this part. Write it as if you are briefing a new AI assistant.
Cover:
- What was being worked on and the goals
- What was decided, why each decision was made, and how it was implemented
- What was tried but failed, and why
- What problems or questions are still open
- Any important context
Rules:
- Do not leave out any decision, failed attempt, open question, or keyword.
- Write in plain text only. No markdown, no bullet points, no headers.
- Write in third person about the user.
- Preserve all numbers, metrics, and technical parameters exactly.
- Return only the memory text. Nothing else."""

COMBINE_MEMORY_SYSTEM_PROMPT = """You are a memory writer. You will receive multiple memory paragraphs from different parts of a conversation.
Combine them into a single unified plain text memory document.
Cover:
- What the user is working on and their goals
- What was decided and what solutions were found, including why each decision was made and how it was implemented
- What was tried but failed, and why it failed
- What problems or questions are still open
- Any important context about the user or their project
Rules:
- Do not leave out any decision, failed attempt, open question, or keyword from any paragraph.
- Remove duplicates but preserve all unique information.
- Write in plain text only. No markdown, no bullet points, no headers.
- Write in third person about the user.
- Preserve all numbers, metrics, and technical parameters exactly.
- Return only the memory text. Nothing else."""


class FinalMemoryLayer:
    """
    Accepts either:
    - A single merged Dict (legacy, from old MergeLayer)
    - A list of merged Dicts (new, from updated MergeLayer)

    For a list: generates a partial memory for each Dict,
    then combines all partial memories into one final memory.

    For a single Dict: generates memory directly (legacy behavior).
    """

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def generate(self, merged: Union[Dict, List[Dict]]) -> str:
        if not merged:
            return ""

        # Legacy: single dict
        if isinstance(merged, dict):
            return self._generate_single(merged)

        # New: list of dicts
        if isinstance(merged, list):
            if len(merged) == 0:
                return ""
            if len(merged) == 1:
                return self._generate_single(merged[0])
            return self._generate_from_list(merged)

        return ""

    def _generate_from_list(self, merged_list: List[Dict]) -> str:
        print(f"[Layer 5] Generating partial memories for {len(merged_list)} groups...")

        partial_memories = []
        for i, merged in enumerate(merged_list):
            print(f"[Layer 5] Generating summary for group {i+1}/{len(merged_list)}...")
            partial = self._generate_partial(merged, i + 1)
            if partial:
                partial_memories.append(partial)

        if not partial_memories:
            return ""

        if len(partial_memories) == 1:
            return partial_memories[0]

        print(f"[Layer 5] Combining {len(partial_memories)} partial memories...")
        return self._combine_memories(partial_memories)

    def _generate_partial(self, merged: Dict, group_num: int) -> str:
        user_message = f"Convert this conversation summary into a memory paragraph:\n\n{json.dumps(merged, indent=2)}"
        print(f"[Layer 5] Group {group_num}: sending {len(user_message)} characters...")
        try:
            response = self.llm.chat(
                config.LAYER_5_MODEL,
                PARTIAL_MEMORY_SYSTEM_PROMPT,
                user_message,
            )
            cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
            return cleaned
        except Exception as e:
            print(f"[Layer 5] Group {group_num} error: {e}")
            return ""

    def _combine_memories(self, memories: List[str]) -> str:
        combined_text = "\n\n---\n\n".join(memories)
        user_message  = f"Combine these memory paragraphs into one unified memory document:\n\n{combined_text}"
        print(f"[Layer 5] Combining: sending {len(user_message)} characters...")
        try:
            response = self.llm.chat(
                config.LAYER_5_MODEL,
                COMBINE_MEMORY_SYSTEM_PROMPT,
                user_message,
            )
            cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
            return cleaned
        except Exception as e:
            print(f"[Layer 5] Combine error: {e}")
            return "\n\n".join(memories)

    def _generate_single(self, merged: Dict) -> str:
        """Legacy single dict behavior."""
        print("[Layer 5] Generating final memory...")
        user_message = f"Convert this conversation summary into a memory document:\n\n{json.dumps(merged, indent=2)}"
        try:
            response = self.llm.chat(
                config.LAYER_5_MODEL,
                PARTIAL_MEMORY_SYSTEM_PROMPT,
                user_message,
            )
            cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
            return cleaned
        except Exception as e:
            print(f"[Layer 5] Error: {e}")
            return ""
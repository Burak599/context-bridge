# code_layers/code_final_memory.py
import re
from typing import Dict
from layers.llm_client import LLMClient
import config

# System prompt for the final memory generation LLM.
# Instructs the model to produce a structured plain-text project briefing document.
CODE_MEMORY_SYSTEM_PROMPT = """You are a senior software architect writing a project briefing document.
You will receive a unified JSON summary of a software project — its files, architecture, dependencies, and relationships.
Write a clear, concise plain text memory document that an AI assistant can read to immediately understand this codebase.
Structure it exactly like this (use these exact headers):
PROJECT: <project name>
PURPOSE:
One paragraph explaining what this project does and why it exists.
ARCHITECTURE:
One paragraph explaining how the project is structured. Mention the main layers or modules and how they interact.
ENTRY POINTS:
List the main entry point files and what each one does.
KEY FILES:
List the most important files (hubs and core modules), one line each: filename — what it does.
RELATIONSHIPS:
List the key dependencies between files, one line each.
Rules:
- Write in plain text only. No markdown, no bullet points, no JSON.
- Be concise but complete. A developer should understand the project in 30 seconds.
- Do NOT include debug scripts or test files as entry points.
- Do NOT repeat the same information in multiple sections.
- Return only the memory document. Nothing else."""


class CodeFinalMemoryLayer:
    """
    Converts the unified JSON project summary into a plain-text memory document.
    Input : Output of CodeMergeLayer.merge()
    Output: Plain-text project summary (context to be fed to the AI)
    """

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def generate(self, merged: Dict) -> str:
        """
        Args:
            merged: Output of CodeMergeLayer.merge()
        Returns:
            Plain-text project memory document
        """
        if not merged:
            return ""

        print("[Layer 5] Generating final code memory...")

        import json
        user_message = (
            f"Convert this project summary into a memory document:\n\n"
            f"{json.dumps(merged, indent=2)}"
        )

        try:
            response = self.llm.chat(
                config.CODE_MEMORY_MODEL,
                CODE_MEMORY_SYSTEM_PROMPT,
                user_message,
            )
            # Strip all <think>...</think> blocks — there may be multiple or nested ones
            cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
            # If an unclosed <think> tag remains, discard everything from that point on
            if "<think>" in cleaned:
                cleaned = cleaned[:cleaned.index("<think>")]
            cleaned = cleaned.strip()
            return cleaned
        except Exception as e:
            print(f"[Layer 5] Error: {e}")
            return ""
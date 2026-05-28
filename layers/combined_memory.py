# layers/combined_memory.py

import re
from layers.llm_client import LLMClient
import config

COMBINED_MEMORY_SYSTEM_PROMPT = """You are an expert AI context engineer and lossless prompt assembler.
You will receive four separate memory sources:

1. CHAT SUMMARY MEMORY:
   A summary of the developer's conversation history: decisions, solved problems, open questions, project goals, failed attempts.

2. CODE SUMMARY MEMORY:
   A technical summary of the codebase: architecture, entry points, key files, relationships between modules.

3. CHAT DETAIL KEYWORD MEMORY:
   A compact high-signal extraction of exact keywords and atomic details from raw conversation chunks.

4. CODE VARIABLE PARAMETER MEMORY:
   A compact high-signal extraction of important code-level variables, parameters, values, limits, paths, model names, config values.

Your job is to merge ALL FOUR sources into a single, unified context prompt that will be given to a new AI assistant.
This is a lossless assembly task: do not delete, weaken, invent, or hide even one piece of information from the provided sources.

The output must follow this exact structure:

---
You are an AI assistant continuing work on an ongoing software project. Here is everything you need to know:

PROJECT OVERVIEW:
[A clean synthesis of what this project is and what it aims to achieve. Use only the provided sources.]

CODEBASE:
[A clean synthesis of architecture, entry points, key files, module relationships, and important implementation/config values. Use only the provided sources.]

DEVELOPMENT HISTORY:
[A clean synthesis of what has been built, what decisions were made and why, what was tried and failed. Use only the provided sources.]

OPEN QUESTIONS:
[All unresolved problems and questions from the provided sources. If none are present, write that none were explicitly provided.]

COMPLETE SOURCE MEMORY:
[Write exactly this placeholder line: SOURCE MEMORY WILL BE APPENDED VERBATIM BELOW.]

SYSTEM PROMPT:
[Write a natural system prompt for the next AI assistant. The prompt must tell the AI that its first purpose is to understand this full project memory and code/context structure before answering or implementing. Also tell it to rely only on this memory and the visible code, avoid hallucinating missing facts, and ask/check when information is uncertain. Write this section yourself in clear wording; do not use a rigid rule-list style.]

---

Rules:
- Use all four sources. Do not ignore any source because another source seems more general.
- The narrative sections may organize and synthesize, but must not invent facts or contradict the sources.
- COMPLETE SOURCE MEMORY must contain only this placeholder line: SOURCE MEMORY WILL BE APPENDED VERBATIM BELOW.
- If a source block is [EMPTY], include it as [EMPTY].
- Do NOT repeat the same information twice.
- Be concise in synthesis sections.
- Write in second person, addressing the new AI directly.
- The final output must include SYSTEM PROMPT exactly once.
- Return only the final prompt. No explanation, no metadata, nothing else."""


class CombinedMemoryLayer:
    """
    Merges chat memory and code memory into a single AI context prompt.
    """

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def generate(
        self,
        chat_memory: str,
        code_memory: str,
        detail_memory: str = "",
        code_detail_memory: str = "",
    ) -> str:
        """
        Args:
            chat_memory        : Output of FinalMemoryLayer.generate()
            code_memory        : Output of CodeFinalMemoryLayer.generate()
            detail_memory      : Output text of DetailKeywordMergeLayer
            code_detail_memory : Output text of CodeDetailMergeLayer

        Returns:
            Combined AI context prompt
        """
        if not chat_memory and not code_memory and not detail_memory and not code_detail_memory:
            return ""

        print("[Combiner] Merging chat, code, detail, and parameter memories...")

        source_archive = self._build_source_archive(
            chat_memory,
            code_memory,
            detail_memory,
            code_detail_memory,
        )

        user_message = (
            f"BEGIN CHAT SUMMARY MEMORY\n"
            f"{chat_memory or '[EMPTY]'}\n"
            f"END CHAT SUMMARY MEMORY\n\n"
            f"{'=' * 50}\n\n"
            f"BEGIN CODE SUMMARY MEMORY\n"
            f"{code_memory or '[EMPTY]'}\n"
            f"END CODE SUMMARY MEMORY\n\n"
            f"{'=' * 50}\n\n"
            f"BEGIN CHAT DETAIL KEYWORD MEMORY\n"
            f"{detail_memory or '[EMPTY]'}\n"
            f"END CHAT DETAIL KEYWORD MEMORY\n\n"
            f"{'=' * 50}\n\n"
            f"BEGIN CODE VARIABLE PARAMETER MEMORY\n"
            f"{code_detail_memory or '[EMPTY]'}\n"
            f"END CODE VARIABLE PARAMETER MEMORY"
        )

        try:
            response = self.llm.chat(
                config.COMBINED_MODEL,
                COMBINED_MEMORY_SYSTEM_PROMPT,
                user_message,
            )
            # Strip <think>...</think> chain-of-thought blocks if present
            cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
            # If an unclosed <think> tag remains, discard everything from that point on
            if "<think>" in cleaned:
                cleaned = cleaned[:cleaned.index("<think>")]
            return self._append_source_archive(cleaned.strip(), source_archive)

        except Exception as e:
            print(f"[Combiner] Error: {e}")
            return ""

    def _build_source_archive(
        self,
        chat_memory: str,
        code_memory: str,
        detail_memory: str,
        code_detail_memory: str,
    ) -> str:
        return (
            "COMPLETE SOURCE MEMORY:\n"
            "BEGIN CHAT SUMMARY MEMORY\n"
            f"{chat_memory or '[EMPTY]'}\n"
            "END CHAT SUMMARY MEMORY\n\n"
            "BEGIN CODE SUMMARY MEMORY\n"
            f"{code_memory or '[EMPTY]'}\n"
            "END CODE SUMMARY MEMORY\n\n"
            "BEGIN CHAT DETAIL KEYWORD MEMORY\n"
            f"{detail_memory or '[EMPTY]'}\n"
            "END CHAT DETAIL KEYWORD MEMORY\n\n"
            "BEGIN CODE VARIABLE PARAMETER MEMORY\n"
            f"{code_detail_memory or '[EMPTY]'}\n"
            "END CODE VARIABLE PARAMETER MEMORY"
        )

    def _append_source_archive(self, generated_prompt: str, source_archive: str) -> str:
        source_heading = "COMPLETE SOURCE MEMORY:"
        system_heading = "SYSTEM PROMPT:"
        source_idx = generated_prompt.find(source_heading)
        system_idx = generated_prompt.find(system_heading)
        if source_idx != -1 and system_idx != -1 and source_idx < system_idx:
            before = generated_prompt[:source_idx].rstrip()
            after = generated_prompt[system_idx:].lstrip()
            return f"{before}\n\n{source_archive}\n\n{after}".strip()

        placeholder = "SOURCE MEMORY WILL BE APPENDED VERBATIM BELOW."
        full_placeholder = f"COMPLETE SOURCE MEMORY:\n{placeholder}"
        if full_placeholder in generated_prompt:
            generated_prompt = generated_prompt.replace(full_placeholder, source_archive)
            return generated_prompt.strip()

        if placeholder in generated_prompt:
            generated_prompt = generated_prompt.replace(placeholder, source_archive)
            return generated_prompt.strip()

        return f"{generated_prompt.rstrip()}\n\n{source_archive}".strip()
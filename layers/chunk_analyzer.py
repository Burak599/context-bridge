# layers/chunk_analyzer.py

import json
import re
import asyncio
from typing import List, Dict
from groq import AsyncGroq
import config


ANALYZER_SYSTEM_PROMPT = """You are a conversation analyst. You will receive a conversation chunk and extract the most important information from it.

Return ONLY a JSON object with exactly these fields:

{
  "topic": "one sentence describing what this conversation is about",
  "decisions": [
    {
      "what": "the decision or solution that was reached",
      "why": "the reason or problem that led to this decision, empty string if not explicitly stated",
      "how": "the implementation or method chosen, empty string if not explicitly stated",
    }
  ],
  "open_questions": ["list of questions or problems that were raised but not fully resolved"],
  "status": "completed | in_progress | blocked | unknown",
  "keywords": ["important terms, names, numbers, concepts that are central to understanding this conversation"],
  "progress": "one sentence describing what was accomplished or learned in this chunk",
  "context": "any important background information about the user or their project"
}

Rules:
- Be concise. No fluff.
- Preserve all specific numbers, metrics, and technical values exactly as mentioned.
- If a field has nothing relevant, use an empty list [] or empty string "".
- Return ONLY the JSON object. No markdown, no explanation, no extra text.
- Total per decision object must not exceed 250 characters.
- Every decision MUST fit in 250 characters total (what+why+how combined). Be maximally concise. This is a hard limit with zero exceptions.

## HARD LIMITS — NEVER EXCEED:
- Every decision object: 250 characters total (what + why + how combined). Zero exceptions.
- what: 100 chars max. why: 75 chars max. how: 75 chars max.

Primary project detection:
- The conversation may contain multiple topics, side discussions, or examples the user brought up to illustrate a point.
- Identify the PRIMARY project: the one the user is actively building or working on throughout the conversation.
- A topic that appears briefly or is used as an example/test case is NOT the primary project, even if it contains many technical details.
- Decisions and open questions should reflect the primary project. Mention side topics only in "context" and only if they are directly relevant.
- If you are unsure which is the primary project, look for: what the user asks the most questions about, what they are building themselves, what they return to repeatedly."""

MAX_CONCURRENT = 5   # max number of concurrent requests
MAX_RETRIES    = 6   # rate limit retry count
BASE_WAIT      = 2   # exponential backoff starting value


class ChunkAnalyzer:
    """
    Sends each chunk to the LLM in parallel and extracts structured information.
    Uses asyncio.gather + Semaphore to keep concurrent requests at MAX_CONCURRENT.
    """

    def __init__(self, llm_client=None):
        # llm_client is kept for compatibility but not used
        self._async_client = AsyncGroq(api_key=config.get_groq_key())

    def analyze_all(self, chunk_texts: List[str]) -> List[Dict]:
        """
        Analyzes all chunks in parallel.

        Args:
            chunk_texts: Output of chunker.get_chunk_texts()

        Returns:
            List of analysis result dicts per chunk (order preserved)
        """
        return asyncio.run(self._analyze_all_async(chunk_texts))

    # ------------------------------------------------------------------
    # Async core
    # ------------------------------------------------------------------

    async def _analyze_all_async(self, chunk_texts: List[str]) -> List[Dict]:
        semaphore = asyncio.Semaphore(MAX_CONCURRENT)
        tasks = [
            self._analyze_chunk_async(text, i + 1, len(chunk_texts), semaphore)
            for i, text in enumerate(chunk_texts)
        ]
        results = await asyncio.gather(*tasks)
        return list(results)

    async def _analyze_chunk_async(
        self,
        chunk_text: str,
        chunk_number: int,
        total: int,
        semaphore: asyncio.Semaphore,
    ) -> Dict:
        async with semaphore:
            print(f"[Layer 3]   Analyzing chunk {chunk_number}/{total}...")
            user_message = f"Analyze this conversation chunk:\n\n{chunk_text}"

            for attempt in range(MAX_RETRIES):
                try:
                    response = await self._async_client.chat.completions.create(
                        model=config.LAYER_3_MODEL,
                        messages=[
                            {"role": "system", "content": ANALYZER_SYSTEM_PROMPT},
                            {"role": "user",   "content": user_message},
                        ],
                        temperature=0,
                    )
                    content = response.choices[0].message.content.strip()
                    return self._parse_response(content, chunk_number)

                except Exception as e:
                    err = str(e)
                    if "rate_limit" in err or "429" in err:
                        wait = BASE_WAIT ** attempt
                        print(f"[Layer 3]   Chunk {chunk_number} rate limit — waiting {wait}s...")
                        await asyncio.sleep(wait)
                    else:
                        print(f"[Layer 3] Error (chunk {chunk_number}): {e}")
                        return self._empty_result(chunk_number)

            print(f"[Layer 3] Chunk {chunk_number} max retries exceeded.")
            return self._empty_result(chunk_number)

    # ------------------------------------------------------------------
    # Parse & helpers
    # ------------------------------------------------------------------

    def _parse_response(self, response: str, chunk_number: int) -> Dict:
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
                "chunk_number":   chunk_number,
                "topic":          data.get("topic", ""),
                "decisions":      data.get("decisions", []),
                "open_questions": data.get("open_questions", []),
                "progress":       data.get("progress", ""),
                "context":        data.get("context", ""),
            }
        except json.JSONDecodeError:
            print(f"[Layer 3] JSON parse error (chunk {chunk_number})")
            return self._empty_result(chunk_number)

    def _empty_result(self, chunk_number: int) -> Dict:
        return {
            "chunk_number":   chunk_number,
            "topic":          "",
            "decisions":      [],
            "open_questions": [],
            "progress":       "",
            "context":        "",
        }
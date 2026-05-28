# layers/chunking_layer.py

import json
import re
import asyncio
from typing import List, Dict, Tuple
import config
from groq import AsyncGroq

CHUNKING_SYSTEM_PROMPT = """You are a topic-change detector for conversations.

You will receive a numbered list of conversation exchanges.
Your job is to find where the topic significantly changes.

Rules:
- A topic change is when the conversation shifts to a clearly different subject.
- Small topic drifts within the same subject do NOT count as a topic change.
- Return ONLY a JSON array of message numbers where a new topic begins.
- Message 1 is always the start, do NOT include it.
- If there are no topic changes, return an empty array: []

Example response: [6, 12]
Another example with no changes: []

Return ONLY the JSON array. No explanation. No markdown. No extra text."""

MAX_CONCURRENT = 5
MAX_RETRIES    = 6
BASE_WAIT      = 2


class ChunkingLayer:
    def __init__(
        self,
        llm_client=None,
        max_tokens_per_block: int = 2000,
        overlap_messages: int = 3,
        min_chunk_size: int = 2,
    ):
        self.max_tok  = max_tokens_per_block
        self.overlap  = overlap_messages
        self.min_size = min_chunk_size
        self._async_client = AsyncGroq(api_key=config.get_groq_key())

    def chunk(self, messages: List[Dict[str, str]]) -> List[List[Dict[str, str]]]:
        if not messages:
            return []
        if len(messages) <= self.min_size:
            return [messages]

        print(f"[Layer 2] Processing {len(messages)} messages...")
        blocks = self._build_blocks(messages)
        print(f"[Layer 2] {len(blocks)} blocks created, sending in parallel...")

        # Parallel block analysis
        results = asyncio.run(self._detect_all_breaks(blocks))

        all_break_points = set()
        for local_breaks, offset in results:
            global_breaks = {offset + bp for bp in local_breaks}
            all_break_points.update(global_breaks)

        chunks = self._split_by_breaks(messages, sorted(all_break_points))
        chunks = self._merge_small_chunks(chunks)

        print(f"[Layer 2] ✓ {len(messages)} messages → {len(chunks)} chunks")
        return chunks

    def get_chunk_texts(self, chunks: List[List[Dict[str, str]]]) -> List[str]:
        result = []
        for chunk in chunks:
            lines = []
            for msg in chunk:
                role = "User" if msg["role"] == "user" else "AI"
                lines.append(f"{role}: {msg['text']}")
            result.append("\n".join(lines))
        return result

    # ------------------------------------------------------------------
    # Async core
    # ------------------------------------------------------------------

    async def _detect_all_breaks(self, blocks):
        semaphore = asyncio.Semaphore(MAX_CONCURRENT)
        tasks = [
            self._detect_breaks_async(block_msgs, offset, i + 1, len(blocks), semaphore)
            for i, (block_msgs, offset) in enumerate(blocks)
        ]
        return await asyncio.gather(*tasks)

    async def _detect_breaks_async(self, block_msgs, offset, idx, total, semaphore):
        async with semaphore:
            print(f"[Layer 2]   Analyzing block {idx}/{total} ({len(block_msgs)} messages)...")
            formatted = self._format_block_for_llm(block_msgs)
            user_message = f"Find topic changes in this conversation:\n\n{formatted}"

            for attempt in range(MAX_RETRIES):
                try:
                    response = await self._async_client.chat.completions.create(
                        model=config.LAYER_2_MODEL,
                        messages=[
                            {"role": "system", "content": CHUNKING_SYSTEM_PROMPT},
                            {"role": "user",   "content": user_message},
                        ],
                        temperature=0,
                    )
                    content = response.choices[0].message.content.strip()
                    exchange_breaks = self._parse_llm_response(content)
                    msg_breaks = self._exchanges_to_msg_indices(block_msgs, exchange_breaks)
                    return (msg_breaks, offset)

                except Exception as e:
                    err = str(e)
                    if "rate_limit" in err or "429" in err:
                        wait = BASE_WAIT ** attempt
                        print(f"[Layer 2]   Block {idx} rate limit — waiting {wait}s...")
                        await asyncio.sleep(wait)
                    else:
                        print(f"[Layer 2] Block {idx} error: {e}")
                        return ([], offset)

            return ([], offset)

    # ------------------------------------------------------------------
    # Block builder
    # ------------------------------------------------------------------

    def _estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def _build_blocks(self, messages):
        blocks = []
        i = 0
        while i < len(messages):
            block = []
            token_count = 0
            block_start = i

            if blocks and self.overlap > 0:
                prev_block_msgs, _ = blocks[-1]
                overlap_msgs = prev_block_msgs[-self.overlap:]
                for m in overlap_msgs:
                    token_count += self._estimate_tokens(m["text"])
                block.extend(overlap_msgs)

            while i < len(messages):
                msg_tokens = self._estimate_tokens(messages[i]["text"])
                new_msgs_in_block = i - block_start
                if token_count + msg_tokens > self.max_tok and new_msgs_in_block > 0:
                    break
                block.append(messages[i])
                token_count += msg_tokens
                i += 1

            blocks.append((block, block_start))
        return blocks

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _format_block_for_llm(self, block_msgs):
        lines = []
        exchange_num = 1
        i = 0
        while i < len(block_msgs):
            msg = block_msgs[i]
            role = "User" if msg["role"] == "user" else "AI"
            if (msg["role"] == "user"
                    and i + 1 < len(block_msgs)
                    and block_msgs[i+1]["role"] == "assistant"):
                ai_msg = block_msgs[i+1]
                lines.append(f"[{exchange_num}] User: {msg['text']}\n     AI: {ai_msg['text']}")
                i += 2
            else:
                lines.append(f"[{exchange_num}] {role}: {msg['text']}")
                i += 1
            exchange_num += 1
        return "\n\n".join(lines)

    def _parse_llm_response(self, response: str) -> List[int]:
        # Strip <think>...</think> chain-of-thought blocks if present
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
        # Remove markdown code fences if the model wrapped the JSON
        cleaned  = re.sub(r"```(?:json)?", "", response).replace("```", "").strip()
        match    = re.search(r"\[.*?\]", cleaned, re.DOTALL)
        if not match:
            return []
        try:
            result = json.loads(match.group())
            return [int(x) for x in result if isinstance(x, (int, float))]
        except (json.JSONDecodeError, ValueError):
            return []

    def _exchanges_to_msg_indices(self, block_msgs, exchange_breaks):
        msg_breaks = []
        exchange_num = 1
        msg_idx = 0
        i = 0
        while i < len(block_msgs):
            if exchange_num in exchange_breaks:
                msg_breaks.append(msg_idx)
            if (block_msgs[i]["role"] == "user"
                    and i + 1 < len(block_msgs)
                    and block_msgs[i+1]["role"] == "assistant"):
                msg_idx += 2
                i += 2
            else:
                msg_idx += 1
                i += 1
            exchange_num += 1
        return msg_breaks

    def _split_by_breaks(self, messages, break_points):
        if not break_points:
            return [messages]
        chunks, prev = [], 0
        for bp in break_points:
            if bp > prev:
                chunks.append(messages[prev:bp])
            prev = bp
        chunks.append(messages[prev:])
        return [c for c in chunks if c]

    def _merge_small_chunks(self, chunks):
        if len(chunks) <= 1:
            return chunks
        merged = [chunks[0]]
        for chunk in chunks[1:]:
            if len(chunk) < self.min_size:
                merged[-1] = merged[-1] + chunk
            else:
                merged.append(chunk)
        if len(merged) > 1 and len(merged[0]) < self.min_size:
            merged[1] = merged[0] + merged[1]
            merged = merged[1:]
        return merged
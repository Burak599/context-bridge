# main.py

import sys
import os
from layers.input_layer import InputParser
from layers.llm_client import LLMClient
from layers.pipeline import ChunkAnalyzePipeline
from layers.merge_layer import MergeLayer
from layers.final_memory import FinalMemoryLayer
from layers.prompt_generator import PromptGeneratorLayer


def main():
    chat_file_path = sys.argv[1] if len(sys.argv) > 1 else "chat.txt"

    print("=" * 55)
    print("        CHAT MEMORY SYSTEM — MVP")
    print("=" * 55)
    print(f"\n[Input] File: {chat_file_path}")

    if not os.path.exists(chat_file_path):
        print(f"\n[ERROR] '{chat_file_path}' not found!")
        print("Usage: python main.py <file.txt>")
        sys.exit(1)

    with open(chat_file_path, "r", encoding="utf-8") as f:
        raw_chat_history = f.read()

    print(f"[Input] ✓ Read {len(raw_chat_history)} characters.")

    # ----------------------------------------------------------------
    # LAYER 1: Input Layer
    # ----------------------------------------------------------------
    print("\n[Layer 1] Parsing input...")
    parser       = InputParser()
    cleaned_chat = parser.parse_raw_text(raw_chat_history)

    if not cleaned_chat:
        print("\n[ERROR] No messages could be parsed!")
        sys.exit(1)

    print(f"[Layer 1] ✓ Parsed {len(cleaned_chat)} messages.")
    for i, msg in enumerate(cleaned_chat):
        role    = "User" if msg["role"] == "user" else "AI  "
        preview = msg["text"][:60] + "..." if len(msg["text"]) > 60 else msg["text"]
        print(f"  [{i:02d}] {role} | {preview}")

    # ----------------------------------------------------------------
    # LAYERS 2 + 3: Chunking + Analyzing (parallel pipeline)
    # ----------------------------------------------------------------
    print("\n[Layers 2+3] Starting chunking and analyzing in parallel...")
    pipeline = ChunkAnalyzePipeline(
        max_tokens_per_block=2000,
        overlap_messages=3,
        min_chunk_size=2,
    )
    analyses = pipeline.run(cleaned_chat)

    print(f"\n[Layers 2+3] ✓ Analyzed {len(analyses)} chunks.")
    for a in analyses:
        print(f"\n  Chunk {a['chunk_number']}:")
        print(f"    Topic          : {a['topic']}")
        print(f"    Decisions      : {a['decisions']}")
        print(f"    Open Questions : {a['open_questions']}")
        print(f"    Progress       : {a['progress']}")
        print(f"    Context        : {a['context']}")

    # ----------------------------------------------------------------
    # LAYER 4: Merge Layer
    # ----------------------------------------------------------------
    print("\n[Layer 4] Merging chunk analyses...")
    llm    = LLMClient()
    merger = MergeLayer(llm_client=llm)
    merged = merger.merge(analyses)

    print("\n[Layer 4] ✓ Merged.")
    print(f"  Topic          : {merged.get('topic', '')}")
    print(f"  Decisions      : {merged.get('decisions', [])}")
    print(f"  Open Questions : {merged.get('open_questions', [])}")
    print(f"  Progress       : {merged.get('progress', '')}")
    print(f"  Context        : {merged.get('context', '')}")

    # ----------------------------------------------------------------
    # LAYER 5: Final Memory
    # ----------------------------------------------------------------
    print("\n[Layer 5] Generating final memory...")
    final_memory_layer = FinalMemoryLayer(llm_client=llm)
    memory = final_memory_layer.generate(merged)

    print("\n[Layer 5] ✓ Memory generated.")
    print(f"\n{memory}")

    # ----------------------------------------------------------------
    # LAYER 6: Prompt Generator
    # ----------------------------------------------------------------
    print("\n[Layer 6] Generating context prompt...")
    prompt_generator = PromptGeneratorLayer(llm_client=llm)
    final_prompt     = prompt_generator.generate(memory)

    print("\n" + "=" * 55)
    print("             FINAL PROMPT")
    print("=" * 55)
    print(f"\n{final_prompt}")
    print("\n" + "=" * 55)
    print("Pipeline completed — all layers active")
    print("=" * 55)


if __name__ == "__main__":
    main()

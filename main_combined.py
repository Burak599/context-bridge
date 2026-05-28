# main_combined.py
# Usage: python main_combined.py chat.txt /path/to/project

import sys
import os

# Chat pipeline
from layers.input_layer import InputParser
from layers.llm_client import LLMClient
from layers.chunking_layer import ChunkingLayer
from layers.chunk_analyzer import ChunkAnalyzer
from layers.merge_layer import MergeLayer
from layers.final_memory import FinalMemoryLayer
from layers.detail_keyword_layer import DetailKeywordExtractorLayer, DetailKeywordMergeLayer

# Code pipeline
from code_layers.code_input_layer import CodeInputLayer
from code_layers.code_analyzer import CodeAnalyzerLayer
from code_layers.code_relation_layer import CodeRelationLayer
from code_layers.code_merge_layer import CodeMergeLayer
from code_layers.code_final_memory import CodeFinalMemoryLayer
from code_layers.code_detail_layer import CodeDetailExtractorLayer, CodeDetailMergeLayer

# Combiner
from layers.combined_memory import CombinedMemoryLayer


def run_chat_pipeline(chat_file_path: str, llm: LLMClient) -> tuple[str, str]:
    print("\n" + "=" * 55)
    print("  [CHAT PIPELINE] Starting...")
    print("=" * 55)

    if not os.path.exists(chat_file_path):
        print(f"[ERROR] '{chat_file_path}' not found, skipping chat pipeline.")
        return "", ""

    with open(chat_file_path, "r", encoding="utf-8") as f:
        raw_chat = f.read()
    print(f"[Chat | Layer 1] ✓ Read {len(raw_chat)} characters.")

    # Layer 1: Input Parser
    parser = InputParser()
    cleaned_chat = parser.parse_raw_text(raw_chat)
    if not cleaned_chat:
        print("[Chat | Layer 1] ERROR: No messages could be parsed.")
        return "", ""
    print(f"[Chat | Layer 1] ✓ Parsed {len(cleaned_chat)} messages.")

    # Shared chunking path: chunk once, reuse everywhere
    print("[Chat | Shared Chunking] Preparing chunks once...")
    detail_chunker = ChunkingLayer(
        max_tokens_per_block=2000,
        overlap_messages=3,
        min_chunk_size=2,
    )
    detail_chunks = detail_chunker.chunk(cleaned_chat)
    detail_chunk_texts = detail_chunker.get_chunk_texts(detail_chunks)
    print(f"[Chat | Shared Chunking] ✓ Generated {len(detail_chunk_texts)} chunks.")
    if not detail_chunk_texts:
        print("[Chat | DEBUG] WARNING: 0 chunks — chunking returned empty, detail path cannot run.")
    else:
        total_chunk_chars = sum(len(t) for t in detail_chunk_texts)
        print(
            f"[Chat | DEBUG] Chunk sizes: "
            f"min={min(len(t) for t in detail_chunk_texts)}, "
            f"max={max(len(t) for t in detail_chunk_texts)}, "
            f"total={total_chunk_chars} chars"
        )

    detail_extractor = DetailKeywordExtractorLayer(llm_client=llm, debug=True)
    detail_merge = DetailKeywordMergeLayer()
    extracted_details = detail_extractor.extract_all(detail_chunk_texts)
    merged_detail = detail_merge.merge(extracted_details, debug=True)
    detail_memory = merged_detail.get("memory_text", "")
    print(
        f"[Chat | Detail Path] ✓ Processed {len(extracted_details)} chunks, "
        f"{len(merged_detail.get('keywords', []))} keyword, "
        f"{len(merged_detail.get('details', []))} details collected."
    )

    # Layers 2+3: Reuse shared chunks for analyzer (avoid duplicate chunking)
    analyzer = ChunkAnalyzer(llm_client=llm)
    analyses = analyzer.analyze_all(detail_chunk_texts)
    print(f"[Chat | Layers 2+3] ✓ Analyzed {len(analyses)} chunks.")

    # Layer 4: Merge
    merger = MergeLayer(llm_client=llm)
    merged = merger.merge(analyses)
    print("[Chat | Layer 4] ✓ Chunk analyses merged.")

    # Layer 5: Final Memory
    final_memory_layer = FinalMemoryLayer(llm_client=llm)
    memory = final_memory_layer.generate(merged)
    print(f"[DEBUG] chat memory length: {len(memory)}")
    print("[Chat | Layer 5] ✓ Chat memory generated.")

    return memory, detail_memory


def run_code_pipeline(project_path: str, llm: LLMClient) -> tuple[str, str]:
    print("\n" + "=" * 55)
    print("  [CODE PIPELINE] Starting...")
    print("=" * 55)

    if not os.path.isdir(project_path):
        print(f"[ERROR] '{project_path}' is not a directory, skipping code pipeline.")
        return "", ""

    # Layer 1: Code Input
    scanner = CodeInputLayer()
    try:
        files = scanner.scan(project_path)
    except ValueError as e:
        print(f"[Code | Layer 1] ERROR: {e}")
        return "", ""

    if not files:
        print("[Code | Layer 1] ERROR: No code files found.")
        return "", ""
    print(f"[Code | Layer 1] ✓ Found {len(files)} files.")

    # Parallel code detail path (file-level chunks -> details/keywords)
    code_detail_extractor = CodeDetailExtractorLayer(llm_client=llm, debug=True)
    code_detail_merge = CodeDetailMergeLayer()
    code_extracted_details = code_detail_extractor.extract_all(files)
    code_merged_detail = code_detail_merge.merge(code_extracted_details, debug=True)
    code_detail_memory = code_merged_detail.get("memory_text", "")
    print(
        f"[Code | Detail Path] ✓ Processed {len(code_extracted_details)} files, "
        f"{len(code_merged_detail.get('variables', []))} variables/parameters collected."
    )

    # Layer 2: Code Analyzer
    analyzer = CodeAnalyzerLayer(llm_client=llm)
    analyses = analyzer.analyze_all(files)
    print(f"[Code | Layer 2] ✓ Analyzed {len(analyses)} files.")

    # Layer 3: Relation Layer
    relation_layer = CodeRelationLayer(llm_client=llm)
    relation_map   = relation_layer.map(analyses)
    print("[Code | Layer 3] ✓ Relationship map created.")

    # Layer 4: Code Merge
    merge_layer = CodeMergeLayer(llm_client=llm)
    merged      = merge_layer.merge(analyses, relation_map)
    print("[Code | Layer 4] ✓ Code analyses merged.")

    # Layer 5: Code Final Memory
    final_memory_layer = CodeFinalMemoryLayer(llm_client=llm)
    memory = final_memory_layer.generate(merged)
    print("[Code | Layer 5] ✓ Code memory generated.")

    return memory, code_detail_memory


def main():
    if len(sys.argv) < 3:
        print("\nUsage  : python main_combined.py <chat.txt> <project_folder>")
        print("Example: python main_combined.py chat.txt /home/burak/Masaüstü/AgentSummarize")
        sys.exit(1)

    chat_file_path = sys.argv[1]
    project_path   = sys.argv[2]

    print("=" * 55)
    print("     COMBINED MEMORY SYSTEM — MVP")
    print("=" * 55)
    print(f"\n[Input] Chat file     : {chat_file_path}")
    print(f"[Input] Project folder: {os.path.abspath(project_path)}")

    llm = LLMClient()

    chat_memory, detail_memory = run_chat_pipeline(chat_file_path, llm)
    code_memory, code_detail_memory = run_code_pipeline(project_path, llm)

    if not chat_memory and not code_memory and not detail_memory and not code_detail_memory:
        print("\n[ERROR] Both pipelines failed.")
        sys.exit(1)

    print("\n" + "=" * 55)
    print("  [COMBINER] Merging memories...")
    print("=" * 55)

    combiner     = CombinedMemoryLayer(llm_client=llm)
    final_prompt = combiner.generate(chat_memory, code_memory, detail_memory, code_detail_memory)

    print("\n" + "=" * 55)
    print("         CHAT MEMORY (Raw)")
    print("=" * 55)
    print(f"\n{chat_memory}")

    print("\n" + "=" * 55)
    print("         CODE MEMORY (Raw)")
    print("=" * 55)
    print(f"\n{code_memory}")

    print("\n" + "=" * 55)
    print("         DETAIL + KEYWORD MEMORY (Raw)")
    print("=" * 55)
    print(f"\n{detail_memory}")

    print("\n" + "=" * 55)
    print("         CODE VARIABLE/PARAMETER MEMORY (Raw)")
    print("=" * 55)
    print(f"\n{code_detail_memory}")

    print("\n" + "=" * 55)
    print("         FINAL UNIFIED PROMPT")
    print("=" * 55)
    print(f"\n{final_prompt}")
    print("\n" + "=" * 55)
    print("Pipeline completed — all layers active")
    print("=" * 55)


if __name__ == "__main__":
    main()

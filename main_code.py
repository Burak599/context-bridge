# main_code.py

import sys
import os
from code_layers.code_input_layer import CodeInputLayer
from code_layers.code_analyzer import CodeAnalyzerLayer
from code_layers.code_relation_layer import CodeRelationLayer
from code_layers.code_merge_layer import CodeMergeLayer
from code_layers.code_final_memory import CodeFinalMemoryLayer
from layers.llm_client import LLMClient


def main():
    project_path = sys.argv[1] if len(sys.argv) > 1 else "."

    print("=" * 55)
    print("        CODE MEMORY SYSTEM — MVP")
    print("=" * 55)
    print(f"\n[Input] Project folder: {os.path.abspath(project_path)}")

    if not os.path.isdir(project_path):
        print(f"\n[ERROR] '{project_path}' is not a directory!")
        sys.exit(1)

    # ----------------------------------------------------------------
    # LAYER 1: Code Input Layer
    # ----------------------------------------------------------------
    print("\n[Layer 1] Scanning project...")
    scanner = CodeInputLayer()

    try:
        files = scanner.scan(project_path)
    except ValueError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)

    if not files:
        print("\n[ERROR] No code files found!")
        sys.exit(1)

    print(f"[Layer 1] ✓ Found {len(files)} files.")
    print("\n--- Found Files ---")
    for f in files:
        truncated_flag = " [TRUNCATED]" if f["truncated"] else ""
        print(f"  [{f['index']:02d}] {f['path']:<45} {f['size_chars']:>6} characters{truncated_flag}")

    # ----------------------------------------------------------------
    # LAYER 2: Code Analyzer
    # ----------------------------------------------------------------
    print("\n[Layer 2] Analyzing files...")
    llm      = LLMClient()
    analyzer = CodeAnalyzerLayer(llm_client=llm)
    analyses = analyzer.analyze_all(files)

    print(f"\n[Layer 2] ✓ Analyzed {len(analyses)} files.")
    print("\n--- File Analyses ---")
    for a in analyses:
        print(f"\n  {a['file']}")
        print(f"    Purpose      : {a['purpose']}")
        print(f"    Classes      : {a['classes']}")
        print(f"    Functions    : {a['functions']}")
        print(f"    Dependencies : {a['dependencies']}")
        if a["notes"]:
            print(f"    Notes        : {a['notes']}")

    # ----------------------------------------------------------------
    # LAYER 3: Relation Layer
    # ----------------------------------------------------------------
    print("\n[Layer 3] Analyzing relationships between files...")
    relation_layer = CodeRelationLayer(llm_client=llm)
    relation_map   = relation_layer.map(analyses)

    print("\n[Layer 3] ✓ Relationship map created.")
    print("\n--- Relationship Map ---")
    print(f"\n  Architecture : {relation_map.get('architecture', '')}")
    print(f"  Hubs         : {relation_map.get('hubs', [])}")
    print(f"  Entry Points : {relation_map.get('entry_points', [])}")
    print(f"  Core Mod. : {relation_map.get('core_modules', [])}")
    print(f"\n  Relations :")
    for r in relation_map.get("relations", []):
        print(f"    • {r}")

    # ----------------------------------------------------------------
    # LAYER 4: Code Merge Layer
    # ----------------------------------------------------------------
    print("\n[Layer 4] Merging summaries...")
    merge_layer = CodeMergeLayer(llm_client=llm)
    merged      = merge_layer.merge(analyses, relation_map)

    print("\n[Layer 4] ✓ Merged.")
    print("\n--- Unified Project Summary ---")
    print(f"\n  Project Name : {merged.get('project_name', '')}")
    print(f"  Purpose      : {merged.get('purpose', '')}")
    print(f"  Hubs         : {merged.get('hubs', [])}")
    print(f"  Core Mod.    : {merged.get('core_modules', [])}")
    print(f"  Entry Points : {merged.get('entry_points', [])}")

    # ----------------------------------------------------------------
    # LAYER 5: Code Final Memory
    # ----------------------------------------------------------------
    print("\n[Layer 5] Generating final memory...")
    final_memory_layer = CodeFinalMemoryLayer(llm_client=llm)
    memory = final_memory_layer.generate(merged)

    print("\n[Layer 5] ✓ Memory generated.")
    print("\n" + "=" * 55)
    print("           FINAL CODE MEMORY")
    print("=" * 55)
    print(f"\n{memory}")
    print("\n" + "=" * 55)
    print("Pipeline completed — all layers active")
    print("=" * 55)


if __name__ == "__main__":
    main()

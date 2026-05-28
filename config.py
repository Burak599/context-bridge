# config.py

import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

# Each layer uses its own model
LAYER_2_MODEL = "llama-3.1-8b-instant"    # Chunking — fast, simple task
LAYER_3_MODEL = "qwen/qwen3-32b"          # Chunk analysis — highest quality
LAYER_4_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"    # Merge — fast, simple task
LAYER_5_MODEL = "openai/gpt-oss-120b"     # Final memory — strongest model
LAYER_6_MODEL = "llama-3.3-70b-versatile" # Prompt generation — balanced
CODE_ANALYZER_MODEL  = "meta-llama/llama-4-scout-17b-16e-instruct"
CODE_RELATION_MODEL  = "llama-3.3-70b-versatile"
CODE_MERGE_MODEL     = "meta-llama/llama-4-scout-17b-16e-instruct"
CODE_MEMORY_MODEL    = "qwen/qwen3-32b"
COMBINED_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# Detail extractors (chat_detail + code_detail) should use a fast/stable model.
DETAIL_MODEL = LAYER_4_MODEL
CODE_DETAIL_MODEL = LAYER_4_MODEL

def get_groq_key() -> str:
    if not GROQ_API_KEY:
        raise EnvironmentError(
            "\n[ERROR] GROQ_API_KEY not found!\n"
            "Steps:\n"
            "  1. cp .env.example .env\n"
            "  2. Open .env and add your GROQ_API_KEY\n"
            "  3. Run again\n"
        )
    return GROQ_API_KEY

# ContextBridge

AI Project Memory is an MVP system that analyzes a project's codebase and AI chat history, then turns them into one reusable context prompt for future AI assistants.

The goal is simple: when you work in the same AI chat for too long, the assistant can lose focus, miss context, or burn through input tokens quickly. Moving to a new chat often helps, but manually rewriting the full project context is slow and error-prone. This project automates that handoff by turning your codebase and previous chat history into a clean prompt for the next AI assistant. ✨

This project is an MVP and is open to contributions. The architecture, prompts, extraction quality, UI, tests, and provider support can all be improved.

## What It Does 🧠

- Parses AI chat history from a `chat.txt` file.
- Splits long conversations into topic-based chunks.
- Summarizes decisions, progress, context, and open questions.
- Scans and summarizes source code files.
- Builds a code architecture memory from files, classes, functions, and dependencies.
- Extracts important keywords, details, variables, parameters, and config values.
- Combines chat memory, code memory, detail memory, and parameter memory into one final AI context prompt.

## Project Structure 📁

```text
ai-project-memory/
├── main.py                     # Runs only the chat memory pipeline
├── main_code.py                # Runs only the code memory pipeline
├── main_combined.py            # Runs chat + code + detail pipelines together
├── config.py                   # Loads environment variables and model names
├── claude-chat-exporter.js     # Exports Claude web chat history into a text file
├── requirements.txt            # Python dependencies
├── layers/                     # Chat memory pipeline
│   ├── input_layer.py          # Parses raw chat text
│   ├── chunking_layer.py       # Splits chat into topic chunks
│   ├── chunk_analyzer.py       # Extracts decisions, context, progress, questions
│   ├── merge_layer.py          # Merges chunk summaries
│   ├── final_memory.py         # Creates final chat memory
│   ├── detail_keyword_layer.py # Extracts compact chat keywords/details
│   ├── combined_memory.py      # Combines all memories into final prompt
│   └── llm_client.py           # Groq API client wrapper
└── code_layers/                # Code memory pipeline
    ├── code_input_layer.py     # Scans project files
    ├── code_analyzer.py        # Summarizes each code file
    ├── code_relation_layer.py  # Maps relationships between files
    ├── code_merge_layer.py     # Merges code summaries
    ├── code_final_memory.py    # Creates final code memory
    └── code_detail_layer.py    # Extracts important variables/parameters
```

## Setup ⚙️

Use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_real_groq_api_key_here
```

`.env` is ignored by Git because it contains private API keys. The public `.env.example` file only shows which variables are needed:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Never commit your real `.env` file.

## Export Claude Chat History 💬

Open the Claude conversation in your browser.

Press `F12`, open the `Console` tab, paste the contents of `claude-chat-exporter.js`, and press Enter.

The browser will download a text file containing the conversation. Rename it to:

```text
chat.txt
```

Put `chat.txt` in the project root.

## Run 🚀

Run the combined pipeline with:

```bash
python main_combined.py chat.txt /path/to/your/project
```

Example:

```bash
python main_combined.py chat.txt /home/burak/Masaüstü/AgentSummarize
```

The script will generate:

- chat memory
- code memory
- detail/keyword memory
- code variable/parameter memory
- final unified AI context prompt

## Notes 🤝

This is still an MVP. Outputs depend on prompt quality, model behavior, and source data quality. Contributions are welcome for better parsing, safer extraction, tests, documentation, model support, and cleaner final prompt generation.

If this project is useful to you, a star would be appreciated. ⭐

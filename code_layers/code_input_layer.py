# code_layers/code_input_layer.py

import os
from typing import List, Dict

# Directories to exclude from scanning
EXCLUDED_DIRS = {
    "__pycache__", ".git", ".venv", "venv", "env", ".env",
    "node_modules", ".idea", ".vscode", "dist", "build",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", "eggs",
    ".eggs", "*.egg-info",
}

# File extensions to include in scanning
INCLUDED_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx",
    ".java", ".cpp", ".c", ".h", ".cs",
    ".go", ".rs", ".rb", ".php", ".swift",
    ".kt", ".scala", ".r", ".m",
}

# Files to skip entirely
EXCLUDED_FILES = {
    "__init__.py",         # usually empty
    "setup.py",
    "conftest.py",
}

# Max characters per file (truncate files that exceed this limit)
MAX_FILE_CHARS = 30000


class CodeInputLayer:
    """
    Scans the given project directory, reads source files, and
    returns them as an ordered list.

    Sorting logic:
    1. Main entry points first (main.py, app.py, index.py, etc.)
    2. Then by directory depth (shallower first)
    3. Then alphabetically
    """

    ENTRY_POINTS = {
        "main.py", "app.py", "index.py", "run.py",
        "server.py", "cli.py", "manage.py", "start.py",
    }

    def __init__(
        self,
        max_file_chars: int = MAX_FILE_CHARS,
        included_extensions: set = None,
    ):
        self.max_file_chars = max_file_chars
        self.included_extensions = included_extensions or INCLUDED_EXTENSIONS

    def scan(self, project_path: str) -> List[Dict]:
        """
        Scans the directory and returns a sorted list of file dicts.

        Args:
            project_path: Path to the project root directory

        Returns:
            [
                {
                    "index":     1,               # sequential number
                    "path":      "layers/foo.py", # relative to project root
                    "abs_path":  "/full/path...", # absolute path
                    "extension": ".py",
                    "size_chars": 1234,
                    "content":   "...",           # file content (may be truncated)
                    "truncated": False,           # was the content truncated?
                },
                ...
            ]
        """
        if not os.path.isdir(project_path):
            raise ValueError(f"[CodeInputLayer] Invalid directory: {project_path}")

        raw_files = self._collect_files(project_path)
        sorted_files = self._sort_files(raw_files)

        results = []
        for i, file_info in enumerate(sorted_files, start=1):
            content, truncated = self._read_file(file_info["abs_path"])
            results.append({
                "index":      i,
                "path":       file_info["rel_path"],
                "abs_path":   file_info["abs_path"],
                "extension":  file_info["extension"],
                "size_chars": len(content),
                "content":    content,
                "truncated":  truncated,
            })

        return results

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _collect_files(self, project_path: str) -> List[Dict]:
        """Recursively walks the directory and collects eligible files."""
        files = []
        project_path = os.path.abspath(project_path)

        for root, dirs, filenames in os.walk(project_path):
            # Filter out excluded directories in-place
            dirs[:] = [
                d for d in dirs
                if d not in EXCLUDED_DIRS and not d.startswith(".")
            ]

            for filename in filenames:
                ext = os.path.splitext(filename)[1].lower()
                if ext not in self.included_extensions:
                    continue
                if filename in EXCLUDED_FILES:
                    continue

                abs_path = os.path.join(root, filename)
                rel_path = os.path.relpath(abs_path, project_path)
                depth = rel_path.count(os.sep)

                files.append({
                    "abs_path":  abs_path,
                    "rel_path":  rel_path,
                    "filename":  filename,
                    "extension": ext,
                    "depth":     depth,
                })

        return files

    def _sort_files(self, files: List[Dict]) -> List[Dict]:
        """
        Sorting priority:
        0 → entry points (main.py, etc.)
        1 → everything else (by depth, then alphabetically)
        """
        def sort_key(f):
            is_entry = 0 if f["filename"] in self.ENTRY_POINTS else 1
            return (is_entry, f["depth"], f["rel_path"])

        return sorted(files, key=sort_key)

    def _read_file(self, abs_path: str):
        """
        Reads a file. Truncates if it exceeds the character limit.

        Returns:
            (content: str, truncated: bool)
        """
        try:
            with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except Exception as e:
            return f"[UNREADABLE: {e}]", False

        if len(content) > self.max_file_chars:
            content = content[: self.max_file_chars]
            content += f"\n\n... [FILE TRUNCATED — showing first {self.max_file_chars} characters]"
            return content, True

        return content, False
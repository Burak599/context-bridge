import re
from typing import List, Dict


class InputParser:
    """
    Parses conversations copied from different AI platforms.

    Supported platforms:
    - Claude web  : "You said:" / "Claude responded:"
    - ChatGPT     : "You:" / "ChatGPT:"
    - Claude API  : "Human:" / "Assistant:"
    - Gemini      : "Sen:" / "Gemini:"
    - General     : "User:" / "AI:"

    Claude web special behaviors:
    1. "You said: <content>" → start of user message
    2. Same content immediately duplicated on next line → SKIP
    3. "Claude responded: <first line>" → start of AI message
    4. Same content immediately duplicated on next line → SKIP
    5. Subsequent prefix-less lines are the AI message body (until next "You said:")
    6. UI noise like "Show more" → SKIP
    """

    SKIP_PATTERNS = [
        r"^\d{1,2}:\d{2}\s*(am|pm)$",
        r"^\d{1,2}:\d{2}$",
        r"^(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+\d{1,2}$",
        r"^\d{1,2}\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)$",
        r"^(ocak|şubat|mart|nisan|mayıs|haziran|temmuz|ağustos|eylül|ekim|kasım|aralık)\s+\d{1,2}$",
        r"^show more$",
        r"^show less$",
        r"^copy$",
        r"^copied!$",
        r"^reading .+ skill$",
        r"^created a file.*$",
        r"^read a file.*$",
        r"^\d+\s*of\s*\d+$",           # pagination, e.g. "1 of 5"
        r"^pastedimage\d*$",            # pasted image placeholder
        r"^pasted$",
        r"^---+$",              # claude-chat-exporter separator
        r"^# claude conversation$",  # claude-chat-exporter header
    ]

    USER_PREFIXES = [
        "you said:",
        "you:",
        "human:",
        "sen:",
        "user:",
        "USER:",
        "kullanıcı:",
        "ben:",
    ]

    ASSISTANT_PREFIXES = [
        "claude responded:",
        "chatgpt:",
        "assistant:",
        "gemini:",
        "claude:",
        "ai:",
        "AI:",
        "yapay zeka:",
        "bard:",
        "copilot:",
        "grok:",
        "mistral:",
        "deepseek:",
    ]

    # Claude web UI elements — may be attached directly to prefixes
    UI_NOISE = [
        "Free planUpgrade",
        "Free plan",
        "Upgrade",
        "UpgradePlan",
        "Try Pro",
        "Sign up",
        "Log in",
        "Subscribe",
    ]

    def _pre_clean(self, text: str) -> str:
        """
        Strips UI noise that is attached directly to prefixes.
        Example: "Free planUpgradeYou said: ..." → "You said: ..."
        """
        # Remove known UI noise strings
        for noise in self.UI_NOISE:
            text = text.replace(noise, "")

        # If a prefix like "You said:" / "Claude responded:" is attached
        # to another word without a newline, insert a newline before it.
        # Example: "blahYou said:" → "\nYou said:"
        all_prefixes = self.USER_PREFIXES + self.ASSISTANT_PREFIXES
        for prefix in all_prefixes:
            if prefix.startswith('#'):
                continue
            text = re.sub(rf'(?<!\n)(?<!^)({re.escape(prefix)})', r'\n\1', text, flags=re.IGNORECASE)

        return text

    def parse_raw_text(self, text: str) -> List[Dict[str, str]]:
        """
        Parses a conversation in raw text format.

        Algorithm:
        - Detects "You said:" / "Claude responded:" prefixes
        - Skips the duplicate line that immediately follows each prefix
        - Appends continuation lines (no prefix) to the correct message
        - Lines with no prefix after "Claude responded:" belong to the AI
          until the next "You said:" is encountered

        Returns:
            [{"role": "user"|"assistant", "text": "..."}, ...]
        """
        messages: List[Dict[str, str]] = []
        text = self._pre_clean(text)
        lines = text.strip().split("\n")

        # Last content seen after a prefix — used for duplicate detection
        last_prefix_content: str | None = None
        # Which role is currently being written: "user" | "assistant" | None
        current_role: str | None = None

        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue

            line_lower = line.lower()

            # --- Noise filter ---
            if self._should_skip(line_lower):
                continue

            # --- User prefix? ---
            user_content = self._match_prefix(line, line_lower, self.USER_PREFIXES)
            if user_content is not None:
                current_role = "user"
                if user_content:  # inline content present — append immediately
                    last_prefix_content = user_content
                    messages.append({"role": "user", "text": user_content})
                # Empty (e.g. "## Human:") — role switched, next lines belong to this role
                continue

            # --- AI prefix? ---
            ai_content = self._match_prefix(line, line_lower, self.ASSISTANT_PREFIXES)
            if ai_content is not None:
                current_role = "assistant"
                if ai_content:  # inline content present — append immediately
                    last_prefix_content = ai_content
                    messages.append({"role": "assistant", "text": ai_content})
                # Empty (e.g. "## Claude:") — role switched, next lines belong to this role
                continue

            # --- Duplicate line? (Claude web repeats each prefix content on the next line) ---
            if last_prefix_content is not None and self._is_duplicate(line, last_prefix_content):
                last_prefix_content = None  # skip only once
                continue

            # --- Prefix-less line: append to current role ---
            if current_role and messages and messages[-1]["role"] == current_role:
                # Continuation line for the current role
                messages[-1]["text"] += "\n" + line

            elif current_role:
                # Role is known but no message exists for it yet (after "## Human:\n")
                messages.append({"role": current_role, "text": line})

            else:
                # No prefix seen yet — assume the first message is from the user
                current_role = "user"
                messages.append({"role": "user", "text": line})

        # Trim and clean all messages
        return self._clean_messages(messages)

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _should_skip(self, lower_line: str) -> bool:
        for pattern in self.SKIP_PATTERNS:
            if re.match(pattern, lower_line):
                return True
        return False

    def _match_prefix(
        self, original_line: str, lower_line: str, prefixes: List[str]
    ) -> str | None:
        for prefix in prefixes:
            if lower_line.startswith(prefix):
                content = original_line[len(prefix):].strip()
                # Empty content (e.g. separate-line "## Human:" format) → return empty string.
                # Caller interprets this as "role changed, subsequent lines belong to this role."
                return content  # may return "" instead of None
        return None

    def _is_duplicate(self, line: str, reference: str) -> bool:
        """
        Checks whether the line matches the beginning of the reference content.

        Claude web sometimes duplicates only the first N characters of a prefix's
        content rather than the full text (for long messages).
        Therefore we check with startswith rather than full equality.
        """
        # Exact match
        if line == reference:
            return True
        # Reference starts with the line (short duplicate)
        if reference.startswith(line) and len(line) >= 20:
            return True
        # Line starts with the beginning of reference (long message, truncated duplicate)
        if line.startswith(reference[:60]) and len(reference) >= 60:
            return True
        return False

    def _clean_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Removes empty messages, trims text, and merges consecutive
        messages from the same role.
        """
        cleaned = []
        for msg in messages:
            text = msg["text"].strip()
            if not text:
                continue
            # Consecutive same role → merge
            if cleaned and cleaned[-1]["role"] == msg["role"]:
                cleaned[-1]["text"] += "\n" + text
            else:
                cleaned.append({"role": msg["role"], "text": text})
        return cleaned
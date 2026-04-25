"""
tools/kb_tool.py — Knowledge Base tool for Butler.

Manages butler_kb.json — a local file-based knowledge store.
All writes gated behind user_consented=True.
"""

import json
import os
import uuid
import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_KB_PATH = "butler_kb.json"

VALID_CATEGORIES = {
    "meeting", "email", "preference",
    "contact", "user_profile", "habit",
}


class KBTool:
    """
    Manages the Butler knowledge base stored in butler_kb.json.

    Never writes without user_consented=True.
    """

    def __init__(self, kb_path: str = None):
        self.kb_path = kb_path or os.environ.get(
            "BUTLER_KB_PATH", DEFAULT_KB_PATH
        )
        self._ensure_kb_exists()

    def _ensure_kb_exists(self):
        """Create KB file if it doesn't exist."""
        if not os.path.exists(self.kb_path):
            self._write_kb({"entries": [], "user_profile": {}})

    def _read_kb(self) -> dict:
        """Read the knowledge base from disk."""
        try:
            with open(self.kb_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {"entries": [], "user_profile": {}}

    def _write_kb(self, data: dict):
        """Write the knowledge base to disk."""
        with open(self.kb_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def add_entry(
        self,
        content: str,
        category: str,
        source: str = "user",
        user_consented: bool = False,
    ) -> dict:
        """
        Add an entry to the knowledge base.

        CRITICAL: Only writes if user_consented=True.

        Returns:
            {"status": "success"|"error", "entry_id": str, ...}
        """
        if not user_consented:
            return {
                "status": "error",
                "message": "User consent required to write to knowledge base.",
            }

        if category not in VALID_CATEGORIES:
            return {
                "status": "error",
                "message": f"Invalid category '{category}'. "
                           f"Valid: {sorted(VALID_CATEGORIES)}",
            }

        try:
            kb = self._read_kb()
            entry = {
                "id": uuid.uuid4().hex[:12],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "category": category,
                "source": source,
                "content": content,
            }
            kb["entries"].append(entry)
            self._write_kb(kb)

            logger.info("KB entry added: %s (category=%s)", entry["id"], category)
            return {
                "status": "success",
                "entry_id": entry["id"],
                "category": category,
            }

        except Exception as e:
            logger.error("KB add_entry error: %s", str(e))
            return {"status": "error", "message": str(e)}

    def query(self, question: str, top_k: int = 5) -> list[dict]:
        """
        Query the KB using keyword overlap scoring.

        Args:
            question: Query string.
            top_k: Number of results to return.

        Returns:
            List of matching entries sorted by relevance.
        """
        try:
            kb = self._read_kb()
            entries = kb.get("entries", [])

            if not entries:
                return []

            # Tokenize question (basic stopword removal)
            stopwords = {
                "the", "a", "an", "is", "are", "was", "were", "be",
                "been", "being", "have", "has", "had", "do", "does",
                "did", "will", "would", "could", "should", "may",
                "might", "can", "shall", "to", "of", "in", "for",
                "on", "with", "at", "by", "from", "as", "into",
                "about", "what", "which", "who", "whom", "this",
                "that", "these", "those", "i", "me", "my", "we",
                "you", "your", "it", "its", "and", "or", "but",
                "if", "then", "than", "so", "no", "not",
            }
            q_tokens = set(
                w for w in question.lower().split()
                if w not in stopwords and len(w) > 1
            )

            if not q_tokens:
                return entries[:top_k]

            # Score each entry
            scored = []
            for entry in entries:
                content_lower = entry.get("content", "").lower()
                category_lower = entry.get("category", "").lower()
                score = sum(
                    1 for token in q_tokens
                    if token in content_lower or token in category_lower
                )
                if score > 0:
                    scored.append((score, entry))

            scored.sort(key=lambda x: x[0], reverse=True)
            return [entry for _, entry in scored[:top_k]]

        except Exception as e:
            logger.error("KB query error: %s", str(e))
            return []

    def get_user_profile(self) -> dict:
        """Get stored user profile from KB."""
        kb = self._read_kb()
        return kb.get("user_profile", {})

    def save_user_profile(
        self,
        profile: dict,
        user_consented: bool = False,
    ) -> dict:
        """
        Save user profile to KB.

        CRITICAL: Only writes if user_consented=True.
        """
        if not user_consented:
            return {
                "status": "error",
                "message": "User consent required to save profile.",
            }

        try:
            kb = self._read_kb()
            kb["user_profile"] = profile
            self._write_kb(kb)
            return {"status": "success", "profile": profile}

        except Exception as e:
            logger.error("KB save_profile error: %s", str(e))
            return {"status": "error", "message": str(e)}

    def get_entries_by_category(self, category: str) -> list[dict]:
        """Get all KB entries of a given category."""
        kb = self._read_kb()
        return [
            e for e in kb.get("entries", [])
            if e.get("category") == category
        ]

    def get_all_entries(self) -> list[dict]:
        """Get all KB entries."""
        kb = self._read_kb()
        return kb.get("entries", [])

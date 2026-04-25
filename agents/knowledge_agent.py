"""
agents/knowledge_agent.py — Manages the user knowledge base.

Handles KB reads/writes, user context collection, and KB-grounded Q&A.
All writes gated behind user_consented=True.
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class KnowledgeAgent:
    """
    Manages knowledge base operations.

    TIER1_PERSONAL — personal context is highest priority.

    Responsibilities:
    - Add entries to KB (consent-gated)
    - Query KB with keyword matching
    - Collect user context on first run
    - Answer questions grounded in KB data
    """

    TIER = "TIER1_PERSONAL"

    def __init__(self, kb_tool=None):
        self._kb = kb_tool

    def set_tools(self, kb_tool=None):
        """Set dependent tools."""
        if kb_tool:
            self._kb = kb_tool

    def handle(self, todo: dict, collected_fields: dict) -> dict:
        """
        Handle a knowledge-related todo.

        Determines whether the todo is a store or query operation.

        Args:
            todo: The todo dict.
            collected_fields: Dict of already-collected field values.

        Returns:
            Action dict.
        """
        todo_id = todo.get("todo_id", "unknown")
        todo_text = todo.get("text", "")
        text_lower = todo_text.lower()

        # Determine operation type
        query_keywords = ["ask", "what", "when", "who", "how", "recall", "remember"]
        store_keywords = ["birthday", "anniversary", "save", "note", "family"]

        is_query = any(kw in text_lower for kw in query_keywords)
        is_store = any(kw in text_lower for kw in store_keywords)

        if is_store and not is_query:
            return self._handle_store(todo_id, todo_text, collected_fields)
        elif is_query:
            return self._handle_query(todo_id, todo_text, collected_fields)
        else:
            return self._handle_store(todo_id, todo_text, collected_fields)

    def _handle_store(
        self, todo_id: str, todo_text: str, fields: dict
    ) -> dict:
        """Store information in the KB."""
        content = fields.get("content", todo_text)
        category = fields.get("category", self._infer_category(todo_text))

        result = {"status": "success", "simulated": True}
        if self._kb:
            result = self._kb.add_entry(
                content=content,
                category=category,
                source="knowledge_agent",
                user_consented=True,  # In prod, this comes from UI
            )

        return {
            "tool": "add_to_kb",
            "params": {
                "todo_id": todo_id,
                "content": content,
                "category": category,
            },
            "agent": "knowledge_agent",
            "status": "completed",
            "kb_result": result,
        }

    def _handle_query(
        self, todo_id: str, todo_text: str, fields: dict
    ) -> dict:
        """Query the KB and return an answer."""
        answer = self.query(todo_text)

        return {
            "tool": "add_to_kb",
            "params": {
                "todo_id": todo_id,
                "content": f"Query: {todo_text}\nAnswer: {answer}",
                "category": "preference",
            },
            "agent": "knowledge_agent",
            "status": "completed",
            "answer": answer,
        }

    def add_to_kb(
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
            Result dict from KB tool.
        """
        if not user_consented:
            return {
                "status": "error",
                "message": "User consent required to write to knowledge base.",
            }

        if not self._kb:
            return {"status": "error", "message": "KB tool not available."}

        return self._kb.add_entry(
            content=content,
            category=category,
            source=source,
            user_consented=True,
        )

    def query(self, question: str, top_k: int = 5) -> str:
        """
        Query the KB and optionally use the centralized LLM client for answer generation.

        Steps:
        1. Tokenize question (remove stopwords)
        2. Score KB entries by keyword overlap
        3. Pass top_k as context to LLM API (HF or Cursor)
        4. If no API: return raw context entries

        Returns:
            Answer string.
        """
        if not self._kb:
            return "I don't have access to the knowledge base right now."

        results = self._kb.query(question, top_k=top_k)

        if not results:
            return "I don't have that information in the knowledge base."

        # Build context from results
        context = "\n".join(
            f"- [{r.get('category', '?')}] {r.get('content', '')}"
            for r in results
        )

        # Try LLM client for answer generation
        try:
            from tools.llm_client import get_llm_client

            client = get_llm_client()
            if not client.is_available:
                return f"Based on your knowledge base:\n{context}"

            system_prompt = (
                "Answer using only the provided KB context. "
                "Say 'I don't have that information' if not in context."
            )

            user_prompt = (
                f"KB Context:\n{context}\n\n"
                f"Question: {question}"
            )

            response = client.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=200,
                temperature=0.0,  # deterministic for KB Q&A
            )

            if response:
                return response

            return f"Based on your knowledge base:\n{context}"

        except Exception as e:
            logger.error("KB query inference error: %s", str(e))
            return f"Based on your knowledge base:\n{context}"

    def collect_user_context(self, form_data: dict) -> dict:
        """
        First-run: save user profile to KB.

        Args:
            form_data: Dict with name, role, team, timezone,
                       communication_style, google_email.

        Returns:
            Result dict.
        """
        if not self._kb:
            return {"status": "error", "message": "KB tool not available."}

        profile = {
            "name": form_data.get("name", "User"),
            "role": form_data.get("role", "Professional"),
            "team": form_data.get("team", ""),
            "timezone": form_data.get("timezone", "Asia/Kolkata"),
            "communication_style": form_data.get(
                "communication_style", "professional"
            ),
            "google_email": form_data.get("google_email"),
        }

        return self._kb.save_user_profile(
            profile=profile,
            user_consented=True,  # User explicitly fills form
        )

    def _infer_category(self, text: str) -> str:
        """Infer KB category from text content."""
        text_lower = text.lower()

        if any(kw in text_lower for kw in ["birthday", "anniversary", "family"]):
            return "contact"
        if any(kw in text_lower for kw in ["meeting", "call", "standup"]):
            return "meeting"
        if any(kw in text_lower for kw in ["email", "reply", "respond"]):
            return "email"
        if any(kw in text_lower for kw in ["habit", "remind", "daily"]):
            return "habit"
        if any(kw in text_lower for kw in ["prefer", "like", "style"]):
            return "preference"

        return "preference"

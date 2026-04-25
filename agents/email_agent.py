"""
agents/email_agent.py — Handles email-related todos.

Fetches and surfaces important emails, drafts AI replies using
HF Inference API.
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class EmailAgent:
    """
    Handles email-related todos.

    Responsibilities:
    - Fetch and score unread emails by importance
    - Draft AI replies via HF Inference API
    - Send emails
    """

    TIER = "TIER2_PROFESSIONAL"

    IMPORTANCE_KEYWORDS = [
        "urgent", "asap", "deadline", "invoice", "contract",
        "offer", "action required", "follow up", "overdue",
        "interview", "confirm", "approval", "sign", "meeting",
    ]

    def __init__(self, gmail_tool=None, kb_tool=None):
        self._gmail = gmail_tool
        self._kb = kb_tool

    def set_tools(self, gmail_tool=None, kb_tool=None):
        """Set dependent tools."""
        if gmail_tool:
            self._gmail = gmail_tool
        if kb_tool:
            self._kb = kb_tool

    def handle(self, todo: dict, collected_fields: dict) -> dict:
        """
        Handle an email-related todo.

        Args:
            todo: The todo dict.
            collected_fields: Dict of already-collected field values.

        Returns:
            Action dict.
        """
        todo_id = todo.get("todo_id", "unknown")
        todo_text = todo.get("text", "")
        text_lower = todo_text.lower()

        # Determine if this is a send or reply task
        if any(kw in text_lower for kw in ["reply", "respond", "follow up"]):
            return self._handle_reply(todo_id, todo_text, collected_fields)
        else:
            return self._handle_send(todo_id, todo_text, collected_fields)

    def _handle_reply(
        self, todo_id: str, todo_text: str, fields: dict
    ) -> dict:
        """Handle a reply/respond todo."""
        # Check for required info
        email_id = fields.get("email_id")
        if not email_id:
            return {
                "tool": "ask_clarification",
                "params": {
                    "todo_id": todo_id,
                    "field": "email_id",
                    "question": (
                        "Which email should I reply to? "
                        "Please provide the email ID or more details."
                    ),
                },
                "agent": "email_agent",
                "status": "needs_info",
            }

        tone = fields.get("tone", "professional")
        return {
            "tool": "draft_reply",
            "params": {
                "email_id": email_id,
                "tone": tone,
            },
            "agent": "email_agent",
            "status": "completed",
        }

    def _handle_send(
        self, todo_id: str, todo_text: str, fields: dict
    ) -> dict:
        """Handle a send email todo."""
        required = ["to", "subject", "body"]
        missing = [f for f in required if not fields.get(f)]

        if missing:
            field = missing[0]
            questions = {
                "to": "Who should I send this email to? Provide their email address.",
                "subject": "What should the email subject be?",
                "body": "What should the email say?",
            }
            return {
                "tool": "ask_clarification",
                "params": {
                    "todo_id": todo_id,
                    "field": field,
                    "question": questions.get(
                        field, f"Please provide the {field}."
                    ),
                },
                "agent": "email_agent",
                "status": "needs_info",
            }

        # Send the email
        result = {"status": "success", "simulated": True}
        if self._gmail:
            result = self._gmail.send_email(
                to=fields["to"],
                subject=fields["subject"],
                body=fields["body"],
            )

        return {
            "tool": "send_email",
            "params": {
                "todo_id": todo_id,
                "to": fields["to"],
                "subject": fields["subject"],
                "body": fields["body"],
            },
            "agent": "email_agent",
            "status": "completed",
            "email_result": result,
        }

    def fetch_and_surface(self, gmail_service=None) -> list[dict]:
        """
        Fetch and score unread emails by importance.

        Scoring:
        +2 per IMPORTANCE_KEYWORD in subject
        +1 per IMPORTANCE_KEYWORD in snippet
        +3 if sender is in KB contacts

        Returns:
            Top 10 emails sorted by score descending.
        """
        # Get emails
        if self._gmail:
            result = self._gmail.fetch_unread(max_results=50, days_back=7)
        else:
            return []

        if result.get("status") != "success":
            return []

        emails = result.get("emails", [])

        # Get KB contacts for scoring boost
        kb_contacts = set()
        if self._kb:
            contact_entries = self._kb.get_entries_by_category("contact")
            for entry in contact_entries:
                content = entry.get("content", "").lower()
                kb_contacts.add(content)

        # Score each email
        scored = []
        for email in emails:
            score = 0
            subject = email.get("subject", "").lower()
            snippet = email.get("snippet", "").lower()
            sender = email.get("sender", "").lower()

            for keyword in self.IMPORTANCE_KEYWORDS:
                if keyword in subject:
                    score += 2
                if keyword in snippet:
                    score += 1

            # Check if sender is a KB contact
            for contact in kb_contacts:
                if contact in sender:
                    score += 3
                    break

            email["score"] = score
            scored.append(email)

        # Sort by score descending, return top 10
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:10]

    def draft_reply(
        self,
        email: dict,
        user_context: dict,
        tone: str = "professional",
    ) -> str:
        """
        Draft a reply using the centralized LLM client.

        Uses HuggingFace Inference API (primary) or Cursor API (fallback).
        Falls back to template-based drafts if no API is available.

        Args:
            email: Email dict with subject, sender, snippet.
            user_context: Dict with user name, style, etc.
            tone: "professional" | "casual" | "brief"

        Returns:
            Draft reply string.
        """
        user_name = user_context.get("name", "User")
        subject = email.get("subject", "")
        sender = email.get("sender", "")
        snippet = email.get("snippet", "")

        try:
            from tools.llm_client import get_llm_client

            client = get_llm_client()
            if not client.is_available:
                return self._fallback_draft(
                    subject, sender, snippet, user_name, tone
                )

            system_prompt = (
                f"You are Butler, a professional personal assistant AI. "
                f"Draft a concise, {tone} reply to the following email "
                f"on behalf of {user_name}. Under 150 words. "
                f"No placeholders. Sign as: '{user_name} (via Butler)'"
            )

            user_prompt = (
                f"Email from: {sender}\n"
                f"Subject: {subject}\n"
                f"Content: {snippet}\n\n"
                f"Draft a {tone} reply:"
            )

            response = client.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=200,
                temperature=0.7,
            )

            if response:
                return response

            return self._fallback_draft(
                subject, sender, snippet, user_name, tone
            )

        except Exception as e:
            logger.error("Draft reply error: %s", str(e))
            return self._fallback_draft(
                subject, sender, snippet, user_name, tone
            )

    def _fallback_draft(
        self,
        subject: str,
        sender: str,
        snippet: str,
        user_name: str,
        tone: str,
    ) -> str:
        """Generate a simple template-based draft when API is unavailable."""
        if tone == "brief":
            return (
                f"Hi,\n\n"
                f"Thanks for your email regarding \"{subject}\". "
                f"I'll review and get back to you shortly.\n\n"
                f"Best,\n{user_name} (via Butler)"
            )
        elif tone == "casual":
            return (
                f"Hey!\n\n"
                f"Got your message about \"{subject}\". "
                f"Let me look into this and I'll follow up soon.\n\n"
                f"Cheers,\n{user_name} (via Butler)"
            )
        else:
            return (
                f"Dear {sender.split('<')[0].strip()},\n\n"
                f"Thank you for your email regarding \"{subject}\". "
                f"I have noted the details and will respond "
                f"with a comprehensive update shortly.\n\n"
                f"Best regards,\n{user_name} (via Butler)"
            )

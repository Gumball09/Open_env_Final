"""
agents/meeting_agent.py — Handles meeting scheduling and calendar operations.

Routes to calendar_tool for event creation and gmail_tool for confirmations.
Scans meeting transcripts for new actionable items.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class MeetingAgent:
    """
    Handles meeting-related todos.

    Responsibilities:
    - Check for missing required fields
    - Schedule calendar events
    - Send confirmation emails
    - Scan transcripts for new todos
    """

    TIER = "TIER2_PROFESSIONAL"
    REQUIRED_FIELDS = ["attendee_email", "start_time", "duration_minutes"]

    def __init__(self, calendar_tool=None, gmail_tool=None):
        self._calendar = calendar_tool
        self._gmail = gmail_tool

    def set_tools(self, calendar_tool=None, gmail_tool=None):
        """Set dependent tools."""
        if calendar_tool:
            self._calendar = calendar_tool
        if gmail_tool:
            self._gmail = gmail_tool

    def handle(self, todo: dict, collected_fields: dict) -> dict:
        """
        Handle a meeting todo.

        Steps:
        1. Find missing REQUIRED_FIELDS
        2. If missing: return ask_clarification action for first missing field
        3. If all present: create event + send confirmation
        4. Return action dict

        Args:
            todo: The todo dict with text, todo_id, etc.
            collected_fields: Dict of already-collected field values.

        Returns:
            Action dict with tool and params.
        """
        todo_id = todo.get("todo_id", "unknown")
        todo_text = todo.get("text", "")

        # Check for missing fields
        missing = self._find_missing_fields(collected_fields)
        if missing:
            field = missing[0]
            question = self._generate_question(field, todo_text)
            return {
                "tool": "ask_clarification",
                "params": {
                    "todo_id": todo_id,
                    "field": field,
                    "question": question,
                },
                "agent": "meeting_agent",
                "status": "needs_info",
            }

        # All fields present — schedule the event
        title = self._extract_title(todo_text)
        attendee_email = collected_fields["attendee_email"]
        start_time = collected_fields["start_time"]
        duration = int(collected_fields["duration_minutes"])

        # Create calendar event
        event_result = {"status": "success", "simulated": True}
        if self._calendar:
            event_result = self._calendar.create_event(
                title=title,
                start_time=start_time,
                duration_minutes=duration,
                attendee_email=attendee_email,
                description=f"Scheduled by Butler from todo: {todo_text}",
            )

        # Send confirmation email
        email_result = {"status": "success", "simulated": True}
        if self._gmail and event_result.get("status") == "success":
            attendee_name = self._extract_name(todo_text)
            user_name = collected_fields.get("user_name", "User")
            email_result = self._gmail.send_confirmation_email(
                attendee_email=attendee_email,
                attendee_name=attendee_name,
                user_name=user_name,
                title=title,
                start_time=start_time,
                duration_minutes=duration,
            )

        return {
            "tool": "schedule_event",
            "params": {
                "todo_id": todo_id,
                "attendee_email": attendee_email,
                "start_time": start_time,
                "duration_minutes": duration,
                "title": title,
            },
            "agent": "meeting_agent",
            "status": "completed",
            "event_result": event_result,
            "email_result": email_result,
        }

    def _find_missing_fields(self, collected: dict) -> list[str]:
        """Return list of required fields not yet collected."""
        return [
            f for f in self.REQUIRED_FIELDS
            if not collected.get(f) or str(collected[f]).strip() == ""
        ]

    def _generate_question(self, field: str, todo_text: str) -> str:
        """Generate a clarification question for a missing field."""
        questions = {
            "attendee_email": (
                f"To schedule this meeting, I need the attendee's email address. "
                f"Who should I send the invite to?"
            ),
            "start_time": (
                f"When should this meeting be scheduled? "
                f"Please provide a date and time (e.g., '2024-01-15T10:00:00')."
            ),
            "duration_minutes": (
                f"How long should this meeting be? "
                f"Please specify duration in minutes (e.g., 30, 60)."
            ),
        }
        return questions.get(
            field,
            f"I need the {field} to proceed. Could you provide it?"
        )

    def _extract_title(self, todo_text: str) -> str:
        """Extract a meeting title from the todo text."""
        # Remove common prefixes
        text = todo_text
        for prefix in [
            "schedule a meeting with",
            "set up a meeting with",
            "set up a call with",
            "schedule a call with",
        ]:
            if text.lower().startswith(prefix):
                text = text[len(prefix):].strip()
                break

        # Use the remaining text as the title, or a fallback
        if text and len(text) > 5:
            return f"Meeting: {text[:80]}"
        return "Meeting (scheduled by Butler)"

    def _extract_name(self, todo_text: str) -> str:
        """Extract an attendee name from the todo text."""
        # Simple heuristic: look for name patterns after "with"
        text_lower = todo_text.lower()
        if " with " in text_lower:
            after_with = todo_text.split("with", 1)[1].strip()
            # Take first word or two as name
            words = after_with.split()
            if words:
                name = words[0].rstrip(",.")
                return name.title()
        return "Attendee"

    def process_transcript(self, transcript: str) -> list[dict]:
        """
        Scan a meeting transcript for new actionable items.

        Returns:
            List of new todo dicts to re-enter the queue.
        """
        from agents.orchestrator import KEYWORD_MAP, Orchestrator

        orch = Orchestrator()
        new_todos = []

        # Split transcript into sentences
        sentences = [
            s.strip() for s in transcript.replace("\n", ". ").split(".")
            if s.strip()
        ]

        for sentence in sentences:
            agents = orch.scan_keywords(sentence)
            if agents:
                import uuid
                from datetime import datetime, timezone

                tier, priority = orch.classify_tier(sentence)
                new_todos.append({
                    "todo_id": uuid.uuid4().hex[:12],
                    "text": sentence.strip(),
                    "tier": tier,
                    "priority_score": priority,
                    "expected_agent": agents[0] if agents else None,
                    "source": "meeting_transcript",
                    "submitted_at": datetime.now(timezone.utc).isoformat(),
                    "status": "pending",
                })

        return new_todos

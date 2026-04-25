"""
agents/habit_agent.py — Manages habits, reminders, and wellness tracking.

TIER1_PERSONAL — health and habits are highest priority.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class HabitAgent:
    """
    Handles habit and reminder todos.

    TIER1_PERSONAL — health and habits are highest priority.

    Responsibilities:
    - Create recurring habits/reminders
    - Track habit completions and streaks
    - Generate weekly summaries
    - Send reminder notifications
    """

    TIER = "TIER1_PERSONAL"
    CATEGORIES = ["health", "family", "work", "personal"]

    def __init__(self, kb_tool=None, gmail_tool=None, reminder_tool=None):
        self._kb = kb_tool
        self._gmail = gmail_tool
        self._reminder = reminder_tool

    def set_tools(self, kb_tool=None, gmail_tool=None, reminder_tool=None):
        """Set dependent tools."""
        if kb_tool:
            self._kb = kb_tool
        if gmail_tool:
            self._gmail = gmail_tool
        if reminder_tool:
            self._reminder = reminder_tool

    def handle(self, todo: dict, collected_fields: dict) -> dict:
        """
        Handle a habit/reminder todo.

        Args:
            todo: The todo dict.
            collected_fields: Dict of already-collected field values.

        Returns:
            Action dict.
        """
        todo_id = todo.get("todo_id", "unknown")
        todo_text = todo.get("text", "")

        # Extract or use collected fields
        label = collected_fields.get("label", self._extract_label(todo_text))
        frequency = collected_fields.get(
            "frequency", self._infer_frequency(todo_text)
        )
        time_of_day = collected_fields.get(
            "time_of_day", self._extract_time(todo_text)
        )
        category = collected_fields.get(
            "category", self._infer_category(todo_text)
        )

        # Check for missing critical info
        if not time_of_day:
            return {
                "tool": "ask_clarification",
                "params": {
                    "todo_id": todo_id,
                    "field": "time_of_day",
                    "question": (
                        "What time should I set this reminder for? "
                        "Please provide in HH:MM format (e.g., 08:00)."
                    ),
                },
                "agent": "habit_agent",
                "status": "needs_info",
            }

        # Create the habit
        result = self.create_habit(
            label=label,
            frequency=frequency,
            time_of_day=time_of_day,
            category=category,
            user_email=collected_fields.get("user_email"),
            user_name=collected_fields.get("user_name", "User"),
        )

        return {
            "tool": "set_reminder",
            "params": {
                "todo_id": todo_id,
                "label": label,
                "frequency": frequency,
                "time_of_day": time_of_day,
            },
            "agent": "habit_agent",
            "status": "completed",
            "habit_result": result,
        }

    def create_habit(
        self,
        label: str,
        frequency: str,
        time_of_day: str,
        category: str = "personal",
        user_email: str = None,
        user_name: str = "User",
        gmail_service=None,
    ) -> dict:
        """
        Create a recurring habit.

        Steps:
        1. Save to KB under category "habit"
        2. Schedule recurring Gmail reminders
        3. Return habit dict with habit_id

        Returns:
            {"status": "success"|"error", "habit_id": str, ...}
        """
        if self._reminder:
            return self._reminder.create_reminder(
                label=label,
                frequency=frequency,
                time_of_day=time_of_day,
                category=category,
                user_email=user_email,
                user_name=user_name,
            )

        # Fallback: save to KB directly
        import uuid

        habit_id = f"habit_{uuid.uuid4().hex[:10]}"

        if self._kb:
            self._kb.add_entry(
                content=f"Habit: {label} ({frequency} at {time_of_day})",
                category="habit",
                source="habit_agent",
                user_consented=True,
            )

        return {
            "status": "success",
            "habit_id": habit_id,
            "label": label,
            "frequency": frequency,
            "time_of_day": time_of_day,
        }

    def mark_complete(self, habit_id: str) -> dict:
        """
        Mark a habit as completed for today. Update streak.

        Returns:
            {"status": "success"|"error", ...}
        """
        if self._reminder:
            return self._reminder.mark_complete(habit_id)

        return {
            "status": "success",
            "habit_id": habit_id,
            "message": f"Habit {habit_id} marked complete.",
        }

    def weekly_summary(
        self,
        user_email: str = None,
        user_name: str = "User",
        gmail_service=None,
    ) -> dict:
        """
        Generate and send weekly habit summary.

        Every Sunday 8PM user timezone:
        Compute completions per habit for the week.
        Send summary email with habit | target | done | streak table.

        Returns:
            Summary result dict.
        """
        # Get all habits from KB
        habits = []
        if self._kb:
            habit_entries = self._kb.get_entries_by_category("habit")
            for entry in habit_entries:
                habits.append({
                    "label": entry.get("content", "Unknown habit"),
                    "frequency": "daily",
                    "completions": [],
                    "streak": 0,
                })

        if self._reminder and user_email:
            return self._reminder.send_weekly_summary(
                habits=habits,
                user_email=user_email,
                user_name=user_name,
            )

        # Fallback: just return summary text
        if self._reminder:
            summary = self._reminder.get_weekly_summary(habits)
        else:
            summary = f"Weekly summary: {len(habits)} habits tracked."

        return {
            "status": "success",
            "summary": summary,
            "habit_count": len(habits),
        }

    def _extract_label(self, todo_text: str) -> str:
        """Extract a habit label from todo text."""
        text = todo_text

        # Remove common prefixes
        prefixes = [
            "remind me to", "set a daily habit to",
            "set a habit to", "set up a daily",
            "daily reminder to", "remind me about",
            "i need to", "i want to",
        ]
        text_lower = text.lower()
        for prefix in prefixes:
            if text_lower.startswith(prefix):
                text = text[len(prefix):].strip()
                break

        # Remove time references at the end
        import re
        text = re.sub(
            r"\s*(at|every|before|after)\s+\d{1,2}[:\d]*\s*(am|pm|AM|PM)?.*$",
            "",
            text,
        ).strip()

        return text if text else todo_text[:50]

    def _infer_frequency(self, todo_text: str) -> str:
        """Infer frequency from todo text."""
        text_lower = todo_text.lower()

        if "every day" in text_lower or "daily" in text_lower:
            return "daily"
        if "weekly" in text_lower or "every week" in text_lower:
            return "weekly"
        if "weekday" in text_lower or "weekdays" in text_lower:
            return "weekdays"
        if any(
            day in text_lower
            for day in [
                "monday", "tuesday", "wednesday",
                "thursday", "friday", "saturday", "sunday",
            ]
        ):
            return "weekly"

        # Default to daily for reminders
        return "daily"

    def _extract_time(self, todo_text: str) -> Optional[str]:
        """Extract time from todo text. Returns HH:MM or None."""
        import re

        # Match patterns like "8 AM", "8:00 AM", "20:00", "8:00"
        patterns = [
            r"(\d{1,2}):(\d{2})\s*(AM|PM|am|pm)",
            r"(\d{1,2})\s*(AM|PM|am|pm)",
            r"(\d{1,2}):(\d{2})",
        ]

        for pattern in patterns:
            match = re.search(pattern, todo_text)
            if match:
                groups = match.groups()

                if len(groups) == 3 and groups[2]:
                    # HH:MM AM/PM
                    hour = int(groups[0])
                    minute = int(groups[1])
                    ampm = groups[2].upper()
                    if ampm == "PM" and hour != 12:
                        hour += 12
                    elif ampm == "AM" and hour == 12:
                        hour = 0
                    return f"{hour:02d}:{minute:02d}"

                elif len(groups) == 2 and groups[1] in ("AM", "PM", "am", "pm"):
                    # H AM/PM
                    hour = int(groups[0])
                    ampm = groups[1].upper()
                    if ampm == "PM" and hour != 12:
                        hour += 12
                    elif ampm == "AM" and hour == 12:
                        hour = 0
                    return f"{hour:02d}:00"

                elif len(groups) == 2:
                    # HH:MM (24hr)
                    hour = int(groups[0])
                    minute = int(groups[1])
                    if 0 <= hour <= 23 and 0 <= minute <= 59:
                        return f"{hour:02d}:{minute:02d}"

        # Common time words
        text_lower = todo_text.lower()
        if "morning" in text_lower:
            return "08:00"
        if "evening" in text_lower:
            return "18:00"
        if "night" in text_lower or "before bed" in text_lower:
            return "22:00"
        if "afternoon" in text_lower:
            return "14:00"

        return None

    def _infer_category(self, todo_text: str) -> str:
        """Infer habit category from text."""
        text_lower = todo_text.lower()

        health_kw = [
            "gym", "workout", "exercise", "water", "medicine",
            "vitamins", "supplements", "sleep", "health",
            "meditate", "therapy", "mental health", "doctor",
        ]
        family_kw = [
            "mom", "dad", "family", "kids", "wife", "husband",
            "partner", "call", "birthday", "anniversary",
        ]
        work_kw = [
            "meeting", "email", "project", "deadline",
            "report", "standup",
        ]

        if any(kw in text_lower for kw in health_kw):
            return "health"
        if any(kw in text_lower for kw in family_kw):
            return "family"
        if any(kw in text_lower for kw in work_kw):
            return "work"

        return "personal"

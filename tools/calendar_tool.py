"""
tools/calendar_tool.py — Google Calendar integration for Butler.

All API calls wrapped in try/except returning standardized result dicts.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


class CalendarTool:
    """Manages Google Calendar operations for Butler."""

    def __init__(self, service=None):
        self._service = service

    def set_service(self, service):
        """Set the Google Calendar API service."""
        self._service = service

    def create_event(
        self,
        title: str,
        start_time: str,
        duration_minutes: int,
        attendee_email: str,
        description: str = "",
        timezone: str = "Asia/Kolkata",
    ) -> dict:
        """
        Create a Google Calendar event.

        Args:
            title: Event title.
            start_time: ISO8601 datetime string.
            duration_minutes: Duration in minutes.
            attendee_email: Email of the attendee.
            description: Optional event description.
            timezone: Timezone string.

        Returns:
            {"status": "success"|"error", "event_id": str, ...}
        """
        try:
            if not self._service:
                return self._simulate_event(
                    title, start_time, duration_minutes,
                    attendee_email, description
                )

            # Parse start time
            try:
                start_dt = datetime.fromisoformat(start_time)
            except ValueError:
                start_dt = datetime.now()

            end_dt = start_dt + timedelta(minutes=duration_minutes)

            event_body = {
                "summary": title,
                "description": description,
                "start": {
                    "dateTime": start_dt.isoformat(),
                    "timeZone": timezone,
                },
                "end": {
                    "dateTime": end_dt.isoformat(),
                    "timeZone": timezone,
                },
                "attendees": [{"email": attendee_email}],
                "reminders": {
                    "useDefault": False,
                    "overrides": [
                        {"method": "popup", "minutes": 15},
                    ],
                },
            }

            result = (
                self._service.events()
                .insert(calendarId="primary", body=event_body)
                .execute()
            )

            logger.info("Calendar event created: %s", result.get("id"))
            return {
                "status": "success",
                "event_id": result.get("id", ""),
                "html_link": result.get("htmlLink", ""),
                "title": title,
                "start_time": start_time,
                "duration_minutes": duration_minutes,
                "attendee": attendee_email,
            }

        except Exception as e:
            logger.error("Calendar create_event error: %s", str(e))
            return {"status": "error", "message": str(e)}

    def _simulate_event(
        self, title, start_time, duration_minutes,
        attendee_email, description
    ) -> dict:
        """Simulate event creation when no Google service is available."""
        import uuid

        event_id = f"sim_{uuid.uuid4().hex[:12]}"
        return {
            "status": "success",
            "event_id": event_id,
            "html_link": f"https://calendar.google.com/event/{event_id}",
            "title": title,
            "start_time": start_time,
            "duration_minutes": duration_minutes,
            "attendee": attendee_email,
            "simulated": True,
        }

    def list_upcoming(self, max_results: int = 10) -> dict:
        """
        List upcoming calendar events.

        Returns:
            {"status": "success"|"error", "events": list}
        """
        try:
            if not self._service:
                return {
                    "status": "success",
                    "events": [],
                    "simulated": True,
                }

            now = datetime.utcnow().isoformat() + "Z"
            result = (
                self._service.events()
                .list(
                    calendarId="primary",
                    timeMin=now,
                    maxResults=max_results,
                    singleEvents=True,
                    orderBy="startTime",
                )
                .execute()
            )

            events = result.get("items", [])
            return {
                "status": "success",
                "events": [
                    {
                        "id": e.get("id"),
                        "title": e.get("summary", ""),
                        "start": e.get("start", {}).get("dateTime", ""),
                        "end": e.get("end", {}).get("dateTime", ""),
                    }
                    for e in events
                ],
            }

        except Exception as e:
            logger.error("Calendar list error: %s", str(e))
            return {"status": "error", "message": str(e)}

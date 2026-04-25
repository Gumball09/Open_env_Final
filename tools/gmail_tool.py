"""
tools/gmail_tool.py — Gmail integration for Butler.

Handles email sending, fetching, and reply checking.
All API calls wrapped in try/except returning standardized result dicts.
"""

import base64
import logging
from email.mime.text import MIMEText
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


class GmailTool:
    """Manages Gmail operations for Butler."""

    def __init__(self, service=None):
        self._service = service

    def set_service(self, service):
        """Set the Gmail API service."""
        self._service = service

    def send_email(
        self,
        to: str,
        subject: str,
        body: str,
        sender: str = "me",
    ) -> dict:
        """
        Send an email via Gmail API.

        Returns:
            {"status": "success"|"error", "message_id": str, ...}
        """
        try:
            if not self._service:
                return self._simulate_send(to, subject, body)

            message = MIMEText(body)
            message["to"] = to
            message["subject"] = subject
            raw = base64.urlsafe_b64encode(
                message.as_bytes()
            ).decode("utf-8")

            result = (
                self._service.users()
                .messages()
                .send(userId=sender, body={"raw": raw})
                .execute()
            )

            logger.info("Email sent. ID: %s", result.get("id"))
            return {
                "status": "success",
                "message_id": result.get("id", ""),
                "thread_id": result.get("threadId", ""),
                "to": to,
                "subject": subject,
            }

        except Exception as e:
            logger.error("Gmail send error: %s", str(e))
            return {"status": "error", "message": str(e)}

    def _simulate_send(self, to: str, subject: str, body: str) -> dict:
        """Simulate email send when no Gmail service is available."""
        import uuid

        msg_id = f"sim_{uuid.uuid4().hex[:12]}"
        return {
            "status": "success",
            "message_id": msg_id,
            "thread_id": f"thread_{msg_id}",
            "to": to,
            "subject": subject,
            "simulated": True,
        }

    def send_confirmation_email(
        self,
        attendee_email: str,
        attendee_name: str,
        user_name: str,
        title: str,
        start_time: str,
        duration_minutes: int,
    ) -> dict:
        """
        Send a meeting confirmation email.

        Returns:
            Standardized result dict.
        """
        subject = f"Meeting confirmed: {title}"
        body = (
            f"Hi {attendee_name},\n\n"
            f"{user_name} has scheduled a meeting with you.\n\n"
            f"Details:\n"
            f"- Date/Time: {start_time}\n"
            f"- Duration:  {duration_minutes} minutes\n\n"
            f"Please join on time. If you need to reschedule, "
            f"reply to this email and Butler will notify {user_name}.\n\n"
            f"— Butler (on behalf of {user_name})"
        )
        return self.send_email(to=attendee_email, subject=subject, body=body)

    def fetch_unread(
        self,
        max_results: int = 50,
        days_back: int = 7,
    ) -> dict:
        """
        Fetch unread emails from the last N days.

        Returns:
            {"status": "success"|"error", "emails": list}
        """
        try:
            if not self._service:
                return {
                    "status": "success",
                    "emails": [],
                    "simulated": True,
                }

            after_date = (
                datetime.now() - timedelta(days=days_back)
            ).strftime("%Y/%m/%d")
            query = f"is:unread after:{after_date}"

            result = (
                self._service.users()
                .messages()
                .list(userId="me", q=query, maxResults=max_results)
                .execute()
            )

            messages = result.get("messages", [])
            emails = []

            for msg_ref in messages:
                msg = (
                    self._service.users()
                    .messages()
                    .get(
                        userId="me",
                        id=msg_ref["id"],
                        format="metadata",
                        metadataHeaders=["Subject", "From", "Date"],
                    )
                    .execute()
                )

                headers = {
                    h["name"]: h["value"]
                    for h in msg.get("payload", {}).get("headers", [])
                }

                emails.append({
                    "email_id": msg["id"],
                    "thread_id": msg.get("threadId", ""),
                    "subject": headers.get("Subject", ""),
                    "sender": headers.get("From", ""),
                    "snippet": msg.get("snippet", ""),
                    "received_at": headers.get("Date", ""),
                })

            return {"status": "success", "emails": emails}

        except Exception as e:
            logger.error("Gmail fetch error: %s", str(e))
            return {"status": "error", "message": str(e)}

    def check_for_reply(self, thread_id: str) -> dict:
        """
        Check if there are new replies in a thread.

        Returns:
            {"status": "success"|"error", "has_reply": bool, ...}
        """
        try:
            if not self._service:
                return {
                    "status": "success",
                    "has_reply": False,
                    "simulated": True,
                }

            thread = (
                self._service.users()
                .threads()
                .get(userId="me", id=thread_id)
                .execute()
            )

            messages = thread.get("messages", [])
            return {
                "status": "success",
                "has_reply": len(messages) > 1,
                "message_count": len(messages),
                "thread_id": thread_id,
            }

        except Exception as e:
            logger.error("Gmail thread check error: %s", str(e))
            return {"status": "error", "message": str(e)}

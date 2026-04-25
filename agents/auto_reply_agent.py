"""
agents/auto_reply_agent.py — Fully automated daemon agent for Butler.

This agent runs without explicit user prompts. It scans for incoming
messages (e.g. unread emails), uses the Knowledge Base to formulate
grounded answers, and automatically replies.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class AutoReplyAgent:
    """
    Fully automated daemon agent.

    Unlike other agents that wait for specific 'todo' routing, this agent 
    proactively scans the environment (inbox/messages), formulates a response 
    using the LLM and the Knowledge Base, and sends the reply automatically.
    """

    TIER = "TIER2_PROFESSIONAL" # Professional priority level for auto-replies

    def __init__(self, gmail_tool=None, kb_tool=None):
        self._gmail = gmail_tool
        self._kb = kb_tool

    def set_tools(self, gmail_tool=None, kb_tool=None):
        """Set dependent tools."""
        if gmail_tool:
            self._gmail = gmail_tool
        if kb_tool:
            self._kb = kb_tool

    def run_automation_cycle(self, user_context: dict) -> list[dict]:
        """
        Run a single cycle of the auto-reply automation loop.

        Steps:
        1. Fetch unread, important emails.
        2. For each email, query the KB to see if we have an automated answer.
        3. Use the LLM to draft an autonomous reply.
        4. Send the reply.
        
        Returns:
            A list of logs detailing the actions taken.
        """
        logs = []
        user_name = user_context.get("name", "User")
        
        if not self._gmail:
            logs.append({"status": "error", "message": "Gmail tool not configured for Auto-Reply."})
            return logs

        # Step 1: Fetch unread emails
        fetch_result = self._gmail.fetch_unread(max_results=3, days_back=1)
        if fetch_result.get("status") != "success":
            logs.append({"status": "error", "message": "Failed to fetch inbox."})
            return logs

        emails = fetch_result.get("emails", [])
        if not emails:
            logs.append({"status": "info", "message": "Inbox zero. No auto-replies needed."})
            return logs

        from tools.llm_client import get_llm_client
        client = get_llm_client()

        # Step 2 & 3: Process each email
        for email in emails:
            subject = email.get("subject", "No Subject")
            sender = email.get("sender", "Unknown")
            snippet = email.get("snippet", "")

            logs.append({
                "status": "processing", 
                "message": f"Analyzing email from {sender}: {subject}"
            })

            # Retrieve KB Context relevant to the email snippet
            kb_context = ""
            if self._kb:
                kb_results = self._kb.query(snippet + " " + subject, top_k=3)
                if kb_results:
                    kb_context = "\n".join(
                        f"- [{r.get('category', '?')}] {r.get('content', '')}"
                        for r in kb_results
                    )

            if not client.is_available:
                logs.append({
                    "status": "warning", 
                    "message": f"LLM Client unavailable. Skipping auto-reply for {sender}."
                })
                continue

            # Formulate prompt for auto-reply
            system_prompt = (
                f"You are Butler, an autonomous AI assistant for {user_name}. "
                "You are tasked with writing an automated reply to an incoming email. "
                "If the KB Context provides an answer to the sender's question, use it to answer them. "
                "If you DO NOT know the answer, politely state that you are an AI assistant and will "
                f"flag the email for {user_name} to review personally. "
                "Keep it concise, professional, and sign as 'Butler (AI Assistant)'."
            )

            user_prompt = (
                f"Knowledge Base Context:\n{kb_context}\n\n"
                f"Incoming Email from {sender}:\n"
                f"Subject: {subject}\n"
                f"Body snippet: {snippet}\n\n"
                "Write the automated reply:"
            )

            draft = client.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=250,
                temperature=0.3
            )

            if not draft:
                logs.append({"status": "error", "message": f"Failed to generate draft for {sender}."})
                continue

            # Step 4: Send the email
            send_result = self._gmail.send_email(
                to=sender,
                subject=f"Re: {subject}",
                body=draft
            )

            if send_result.get("status") == "success":
                logs.append({
                    "status": "success", 
                    "message": f"Auto-replied to {sender}.",
                    "draft": draft
                })
            else:
                logs.append({
                    "status": "error", 
                    "message": f"Failed to send email to {sender}."
                })

        return logs

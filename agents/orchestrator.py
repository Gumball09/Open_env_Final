"""
agents/orchestrator.py — Keyword scanner, tier classifier, and priority router.

The orchestrator is the central routing layer that:
1. Classifies todos by tier (personal vs professional)
2. Scans for keywords to determine the correct sub-agent
3. Sorts queues by priority (TIER1 first, then TIER2, then FIFO)
4. Detects priority violations
"""

from typing import Optional


# ─── Task Tier Definitions ─────────────────────────────────────────────────────

TASK_TIERS = {
    "TIER1_PERSONAL": {
        "keywords": [
            "health", "doctor", "gym", "water", "sleep", "medicine",
            "family", "mom", "dad", "kids", "wife", "husband", "partner",
            "remind", "habit", "daily", "every day", "personal",
            "birthday", "anniversary", "therapy", "workout", "mental health",
        ],
        "priority_score": 10,
        "description": "Personal wellbeing and relationships",
    },
    "TIER2_PROFESSIONAL": {
        "keywords": [
            "meeting", "meetings", "call", "email", "reply", "respond",
            "deadline", "project", "client", "report", "presentation",
            "standup", "sprint", "deliverable", "invoice", "contract",
        ],
        "priority_score": 5,
        "description": "Work and professional obligations",
    },
}


# ─── Keyword → Agent Mapping ──────────────────────────────────────────────────

KEYWORD_MAP = {
    # TIER1 PERSONAL → habit_agent
    "health": "habit_agent",
    "doctor": "habit_agent",
    "gym": "habit_agent",
    "workout": "habit_agent",
    "water": "habit_agent",
    "sleep": "habit_agent",
    "medicine": "habit_agent",
    "therapy": "habit_agent",
    "mental health": "habit_agent",
    "remind": "habit_agent",
    "habit": "habit_agent",
    "daily": "habit_agent",
    "every day": "habit_agent",
    # TIER1 PERSONAL → knowledge_agent (family context)
    "birthday": "knowledge_agent",
    "anniversary": "knowledge_agent",
    "family": "knowledge_agent",
    # TIER2 PROFESSIONAL → meeting_agent
    "meeting": "meeting_agent",
    "meetings": "meeting_agent",
    "call": "meeting_agent",
    "standup": "meeting_agent",
    # TIER2 PROFESSIONAL → email_agent
    "email": "email_agent",
    "reply": "email_agent",
    "respond": "email_agent",
    "follow up": "email_agent",
    # TIER2 PROFESSIONAL → knowledge_agent
    "ask": "knowledge_agent",
    "know": "knowledge_agent",
    "recall": "knowledge_agent",
    "remember": "knowledge_agent",
}

NON_TRIGGER_BLOCKLIST = [
    "buy", "grocery", "groceries", "shopping", "watch", "movie",
    "cook", "dinner", "lunch", "breakfast", "clean", "laundry",
    "pay bill", "read book", "read chapter", "order online",
    "fix", "repair", "plan vacation", "travel", "book flight",
    "reserve", "reserve table", "pick up", "drop off", "commute",
]


class Orchestrator:
    """
    Central routing layer for Butler.

    Classifies todos by tier, scans keywords for agent routing,
    sorts queues by priority, and detects priority violations.
    """

    def classify_tier(self, todo_text: str) -> tuple[str, int]:
        """
        Classify a todo into a priority tier based on keywords.

        Rules:
        1. Check text against both tier keyword lists
        2. If matches BOTH tiers: return TIER1_PERSONAL (personal wins)
        3. If matches neither: return ("UNCLASSIFIED", 0)

        Returns:
            (tier_name, priority_score)
        """
        if not todo_text:
            return ("UNCLASSIFIED", 0)

        text_lower = todo_text.lower()

        matches_tier1 = False
        matches_tier2 = False

        for keyword in TASK_TIERS["TIER1_PERSONAL"]["keywords"]:
            if keyword in text_lower:
                matches_tier1 = True
                break

        for keyword in TASK_TIERS["TIER2_PROFESSIONAL"]["keywords"]:
            if keyword in text_lower:
                matches_tier2 = True
                break

        # Personal context always wins
        if matches_tier1:
            return (
                "TIER1_PERSONAL",
                TASK_TIERS["TIER1_PERSONAL"]["priority_score"],
            )

        if matches_tier2:
            return (
                "TIER2_PROFESSIONAL",
                TASK_TIERS["TIER2_PROFESSIONAL"]["priority_score"],
            )

        return ("UNCLASSIFIED", 0)

    def scan_keywords(self, todo_text: str) -> list[str]:
        """
        Scan text for keyword matches and return matched agent names.

        Also checks NON_TRIGGER_BLOCKLIST — if text contains ONLY
        blocklist terms with no KEYWORD_MAP matches, returns [].

        Returns:
            Deduplicated list of matched agent names.
        """
        if not todo_text:
            return []

        text_lower = todo_text.lower()
        matched_agents = set()

        # Check keyword map
        for keyword, agent in KEYWORD_MAP.items():
            if keyword in text_lower:
                matched_agents.add(agent)

        # If no keyword matches, check blocklist
        if not matched_agents:
            for blocked in NON_TRIGGER_BLOCKLIST:
                if blocked in text_lower:
                    return []   # blocklist match, no agents

        return list(matched_agents)

    def route(self, todo_text: str, todo_id: str) -> list[dict]:
        """
        Combine classify_tier and scan_keywords to produce routing actions.

        Returns:
            List of action dicts with routing info, including tier
            and priority_score in each action's params.
        """
        tier, priority_score = self.classify_tier(todo_text)
        agents = self.scan_keywords(todo_text)

        if not agents:
            return [{
                "tool": "route_to_agent",
                "params": {
                    "todo_id": todo_id,
                    "agent_name": "none",
                    "tier": tier,
                    "priority_score": priority_score,
                },
                "routed": False,
            }]

        actions = []
        for agent_name in agents:
            actions.append({
                "tool": "route_to_agent",
                "params": {
                    "todo_id": todo_id,
                    "agent_name": agent_name,
                    "tier": tier,
                    "priority_score": priority_score,
                },
                "routed": True,
            })

        return actions

    def sort_queue(self, queue: list[dict]) -> list[dict]:
        """
        Sort todo queue by:
        1. priority_score descending (TIER1=10 before TIER2=5)
        2. submitted_at ascending (FIFO within same tier)

        Returns:
            Sorted list of todo dicts.
        """
        def sort_key(todo):
            priority = todo.get("priority_score", 0)
            submitted = todo.get("submitted_at", "9999")
            # Negate priority for descending sort, use submitted for FIFO
            return (-priority, submitted)

        return sorted(queue, key=sort_key)

    def check_priority_violation(
        self,
        chosen_todo_id: str,
        queue: list[dict],
    ) -> bool:
        """
        Check if the agent is skipping a personal task for a work task.

        Returns True if there exists any TIER1 todo in queue
        that is NOT chosen_todo_id and is still pending.
        """
        chosen_todo = None
        for todo in queue:
            if todo.get("todo_id") == chosen_todo_id:
                chosen_todo = todo
                break

        if not chosen_todo:
            return False

        # If chosen todo is TIER1, no violation possible
        if chosen_todo.get("tier") == "TIER1_PERSONAL":
            return False

        # Check if any TIER1 todo is pending and not the chosen one
        for todo in queue:
            if (
                todo.get("todo_id") != chosen_todo_id
                and todo.get("tier") == "TIER1_PERSONAL"
                and todo.get("status", "pending") == "pending"
            ):
                return True

        return False

    def get_expected_agent(self, todo_text: str) -> Optional[str]:
        """
        Get the primary expected agent for a todo text.

        Returns the first matching agent or None.
        """
        agents = self.scan_keywords(todo_text)
        return agents[0] if agents else None

    def get_expected_tool(self, agent_name: str) -> Optional[str]:
        """
        Get the primary expected tool for a given agent.

        Returns:
            Tool name string or None.
        """
        agent_tool_map = {
            "meeting_agent": "schedule_event",
            "email_agent": "send_email",
            "knowledge_agent": "add_to_kb",
            "habit_agent": "set_reminder",
        }
        return agent_tool_map.get(agent_name)

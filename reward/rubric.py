"""
reward/rubric.py — Composable reward rubric for Butler.

5 deterministic components. No LLM calls inside compute().
LLM-as-judge is used only in info dict, not in the reward sum.
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

# Keyword → expected agent mapping for correct_routing checks
EXPECTED_AGENT_MAP = {
    # TIER1 → habit_agent
    "health": "habit_agent", "doctor": "habit_agent",
    "gym": "habit_agent", "workout": "habit_agent",
    "water": "habit_agent", "sleep": "habit_agent",
    "medicine": "habit_agent", "therapy": "habit_agent",
    "mental health": "habit_agent",
    "remind": "habit_agent", "habit": "habit_agent",
    "daily": "habit_agent", "every day": "habit_agent",
    # TIER1 → knowledge_agent
    "birthday": "knowledge_agent", "anniversary": "knowledge_agent",
    "family": "knowledge_agent",
    # TIER2 → meeting_agent
    "meeting": "meeting_agent", "meetings": "meeting_agent",
    "call": "meeting_agent", "standup": "meeting_agent",
    # TIER2 → email_agent
    "email": "email_agent", "reply": "email_agent",
    "respond": "email_agent", "follow up": "email_agent",
    # TIER2 → knowledge_agent
    "ask": "knowledge_agent", "know": "knowledge_agent",
    "recall": "knowledge_agent", "remember": "knowledge_agent",
}


class ButlerRubric:
    """
    Composable reward rubric with 5 deterministic components.

    All methods are pure functions of their inputs — no LLM calls,
    no side effects, fully reproducible.
    """

    WEIGHTS = {
        "priority_ordering":    0.25,
        "correct_routing":      0.20,
        "action_completeness":  0.20,
        "api_call_success":     0.20,
        "no_over_triggering":   0.15,
    }

    # ── Component 1: Priority Ordering (weight=0.25) ────────────────────────

    def priority_ordering(
        self,
        chosen_todo: dict,
        full_queue: list[dict],
    ) -> float:
        """
        Evaluate whether the agent respects priority ordering.

        Returns:
            1.0 — chosen todo is the highest-priority pending todo
            0.7 — queue had no TIER1 todos (correct TIER2 handling)
            0.0 — a TIER1 todo existed and was skipped
        """
        if not chosen_todo or not full_queue:
            return 0.5

        chosen_id = chosen_todo.get("todo_id")
        chosen_tier = chosen_todo.get("tier", "UNCLASSIFIED")

        # Check if any TIER1 todos exist in the queue
        tier1_pending = [
            t for t in full_queue
            if t.get("tier") == "TIER1_PERSONAL"
            and t.get("status", "pending") == "pending"
        ]

        if not tier1_pending:
            # No TIER1 in queue — TIER2 handling is acceptable
            return 0.7

        # TIER1 exists — did we pick one?
        if chosen_tier == "TIER1_PERSONAL":
            # Check if we picked the highest priority TIER1
            # (within TIER1, sort by priority_score desc, then FIFO)
            return 1.0

        # Chose a non-TIER1 while TIER1 is pending → violation
        return 0.0

    # ── Component 2: Correct Routing (weight=0.20) ──────────────────────────

    def correct_routing(self, todo_text: str, agent_used: str) -> float:
        """
        Evaluate whether the correct agent was used.

        Returns:
            1.0 — agent matches keyword map
            0.5 — no keyword matched, no agent used (correct abstention)
            0.0 — wrong agent, or agent used when no keyword present
        """
        if not todo_text:
            return 0.5

        text_lower = todo_text.lower()

        # Find expected agents from keywords
        expected_agents = set()
        for keyword, agent in EXPECTED_AGENT_MAP.items():
            if keyword in text_lower:
                expected_agents.add(agent)

        if not expected_agents:
            # No keyword matched
            if not agent_used or agent_used == "none":
                return 0.5   # correct abstention
            return 0.0       # false positive — agent used when shouldn't be

        if agent_used in expected_agents:
            return 1.0       # correct agent

        if not agent_used or agent_used == "none":
            return 0.0       # false negative — keyword present but no agent

        return 0.0           # wrong agent

    # ── Component 3: Action Completeness (weight=0.20) ──────────────────────

    def action_completeness(
        self,
        required_fields: list,
        provided_fields: dict,
    ) -> float:
        """
        Ratio of required fields that are non-null and non-empty.

        Returns:
            Float between 0.0 and 1.0.
        """
        if not required_fields:
            return 1.0

        filled = 0
        for field in required_fields:
            val = provided_fields.get(field)
            if val is not None and str(val).strip() != "":
                filled += 1

        return filled / len(required_fields)

    # ── Component 4: API Call Success (weight=0.20) ─────────────────────────

    def api_call_success(self, tool_name: str, result: dict) -> float:
        """
        Evaluate API call outcome.

        Returns:
            1.0 — result["status"] == "success"
            0.5 — result["status"] == "partial" OR non-API tool
            0.0 — result["status"] == "error"
        """
        api_tools = {"schedule_event", "send_email", "set_reminder"}

        if tool_name not in api_tools:
            return 0.5   # non-API tools default to neutral

        if not result or not isinstance(result, dict):
            return 0.0

        status = result.get("status", "error")

        if status == "success":
            return 1.0
        elif status == "partial":
            return 0.5
        else:
            return 0.0

    # ── Component 5: No Over-Triggering (weight=0.15) ──────────────────────

    def no_over_triggering(
        self,
        todo_text: str,
        actions_taken: list,
    ) -> float:
        """
        Evaluate whether the agent correctly abstains from non-actionable todos.

        Returns:
            1.0 — correct behavior (keyword+action or no-keyword+no-action)
            0.0 — incorrect (false positive or false negative)
        """
        if not todo_text:
            return 0.5

        text_lower = todo_text.lower()
        has_keyword = False

        # Check against all tier keywords
        for tier_data in TASK_TIERS.values():
            for keyword in tier_data["keywords"]:
                if keyword in text_lower:
                    has_keyword = True
                    break
            if has_keyword:
                break

        has_actions = bool(actions_taken)

        if has_keyword and has_actions:
            return 1.0    # correct activation
        elif not has_keyword and not has_actions:
            return 1.0    # correct abstention
        elif not has_keyword and has_actions:
            return 0.0    # false positive
        else:  # has_keyword and not has_actions
            return 0.0    # false negative

    # ── Composite Reward ────────────────────────────────────────────────────

    def compute(self, episode: dict) -> tuple[float, dict]:
        """
        Compute the composite reward from all 5 rubric components.

        Args:
            episode: dict with keys:
                chosen_todo, full_queue, todo_text, agent_used,
                required_fields, provided_fields, tool_name,
                api_result, actions_taken

        Returns:
            (total_reward, breakdown_dict)

        breakdown format:
        {
            "priority_ordering":    float,
            "correct_routing":      float,
            "action_completeness":  float,
            "api_call_success":     float,
            "no_over_triggering":   float,
            "total":                float,
            "priority_violation":   bool,
        }
        """
        scores = {}

        scores["priority_ordering"] = self.priority_ordering(
            chosen_todo=episode.get("chosen_todo", {}),
            full_queue=episode.get("full_queue", []),
        )

        scores["correct_routing"] = self.correct_routing(
            todo_text=episode.get("todo_text", ""),
            agent_used=episode.get("agent_used", ""),
        )

        scores["action_completeness"] = self.action_completeness(
            required_fields=episode.get("required_fields", []),
            provided_fields=episode.get("provided_fields", {}),
        )

        scores["api_call_success"] = self.api_call_success(
            tool_name=episode.get("tool_name", ""),
            result=episode.get("api_result", {}),
        )

        scores["no_over_triggering"] = self.no_over_triggering(
            todo_text=episode.get("todo_text", ""),
            actions_taken=episode.get("actions_taken", []),
        )

        # Weighted sum
        total = sum(
            scores[key] * self.WEIGHTS[key]
            for key in self.WEIGHTS
        )

        # Priority violation flag
        priority_violation = scores["priority_ordering"] == 0.0

        breakdown = {
            **scores,
            "total": round(total, 4),
            "priority_violation": priority_violation,
        }

        return total, breakdown

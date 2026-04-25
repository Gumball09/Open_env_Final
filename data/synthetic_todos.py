"""
data/synthetic_todos.py — Synthetic training data generator for Butler.

Generates realistic todo queues with guaranteed TIER1/TIER2 mix
for training the RL agent on priority ordering decisions.
"""

import json
import uuid
import random
from datetime import datetime, timezone, timedelta
from typing import Optional


# ─── Templates ─────────────────────────────────────────────────────────────────

TIER1_PERSONAL_TEMPLATES = [
    "Remind me to take my {medicine} every morning at {time}",
    "Set a daily habit to drink {n} glasses of water",
    "Remind me to call {family_member} this {day}",
    "Set up a daily gym reminder at {time}",
    "I need to schedule a therapy session with Dr. {name}",
    "Remind me about my {medicine} before bed",
    "Set a habit to meditate every day at {time}",
    "Remind me to wish {family_member} happy birthday on {date}",
    "Daily reminder to sleep by {time}",
    "I want to remind myself to journal every night at {time}",
]

TIER2_PROFESSIONAL_TEMPLATES = [
    "Schedule a meeting with {name} about {topic} on {date} at {time}",
    "Set up a call with {name} to discuss {topic}",
    "I need to reply to {name}'s email about {topic}",
    "Email {name} about the {topic} project update",
    "Set up weekly standup meetings with the {team} team",
    "Follow up with {name} re: {topic} contract",
    "Schedule a client presentation with {name} next {day}",
    "Reply to {name}'s email about the {topic} deadline",
    "Remind me about the {topic} deliverable due {date}",
    "Set up a catch-up call with {name} this week",
]

MIXED_TEMPLATES = [
    "Remind me to take medicine AND reply to {name}'s email",
    "Daily gym reminder + schedule meeting with {name}",
    "Ask Butler about my last meeting AND set a sleep reminder",
]

NON_TRIGGER_TEMPLATES = [
    "Buy {item} from the grocery store",
    "Watch {show} tonight",
    "Cook dinner at {time}",
    "Read chapter {n} of {book}",
    "Clean the {room}",
    "Plan vacation to {place}",
    "Order {item} online",
    "Fix the {item}",
    "Pick up dry cleaning",
    "Book a restaurant for dinner",
]


# ─── Fill Values ───────────────────────────────────────────────────────────────

FILL_VALUES = {
    "name": [
        "Priya", "Rahul", "Ananya", "Vikram", "Sarah",
        "James", "Mei", "Omar", "Elena", "Carlos",
    ],
    "topic": [
        "Q3 report", "product launch", "budget review", "onboarding",
        "partnership", "technical spec", "marketing plan",
    ],
    "team": ["engineering", "design", "product", "sales", "leadership"],
    "time": ["7:00 AM", "8:00 AM", "9:00 PM", "10:00 PM", "6:00 AM"],
    "day": ["Monday", "Tuesday", "Friday", "Sunday", "weekend"],
    "date": ["next Monday", "December 15", "this Friday", "January 5"],
    "medicine": ["vitamins", "supplements", "prescription"],
    "family_member": ["mom", "dad", "sister", "brother", "grandma"],
    "n": ["8", "10", "6"],
    "item": ["groceries", "supplies", "the chair", "the shelf"],
    "show": ["the game", "the series", "the documentary"],
    "book": ["the novel", "Atomic Habits", "the textbook"],
    "room": ["kitchen", "bedroom", "living room"],
    "place": ["Goa", "Paris", "the mountains"],
}


# ─── Agent + Tool Expectations ─────────────────────────────────────────────────

TIER1_EXPECTED = {
    "expected_agent": "habit_agent",
    "expected_tool": "set_reminder",
    "required_fields": ["todo_id", "label", "frequency", "time_of_day"],
}

TIER2_MEETING_EXPECTED = {
    "expected_agent": "meeting_agent",
    "expected_tool": "schedule_event",
    "required_fields": [
        "todo_id", "attendee_email", "start_time",
        "duration_minutes", "title",
    ],
}

TIER2_EMAIL_EXPECTED = {
    "expected_agent": "email_agent",
    "expected_tool": "send_email",
    "required_fields": ["todo_id", "to", "subject", "body"],
}

NON_TRIGGER_EXPECTED = {
    "expected_agent": None,
    "expected_tool": None,
    "required_fields": [],
}


# ─── Generator Functions ──────────────────────────────────────────────────────

def _fill_template(template: str) -> str:
    """Fill a template with random values from FILL_VALUES."""
    import re

    def replacer(match):
        key = match.group(1)
        if key in FILL_VALUES:
            return random.choice(FILL_VALUES[key])
        return match.group(0)

    return re.sub(r"\{(\w+)\}", replacer, template)


def _get_expected_info(text: str, tier: str) -> dict:
    """Determine expected agent, tool, and required fields from text and tier."""
    text_lower = text.lower()

    if tier == "TIER1_PERSONAL":
        if any(kw in text_lower for kw in ["birthday", "anniversary", "family"]):
            return {
                "expected_agent": "knowledge_agent",
                "expected_tool": "add_to_kb",
                "required_fields": ["todo_id", "content", "category"],
            }
        return TIER1_EXPECTED.copy()

    if tier == "TIER2_PROFESSIONAL":
        if any(kw in text_lower for kw in ["meeting", "call", "standup"]):
            return TIER2_MEETING_EXPECTED.copy()
        if any(kw in text_lower for kw in ["email", "reply", "respond"]):
            return TIER2_EMAIL_EXPECTED.copy()
        return TIER2_MEETING_EXPECTED.copy()

    return NON_TRIGGER_EXPECTED.copy()


def generate_todo(tier: str = None) -> dict:
    """
    Generate a single synthetic todo.

    Args:
        tier: "TIER1_PERSONAL", "TIER2_PROFESSIONAL", "NON_TRIGGER",
              or None for random.

    Returns:
        Todo dict with id, text, tier, priority_score, expected info,
        and timestamp.
    """
    if tier is None:
        tier = random.choice([
            "TIER1_PERSONAL", "TIER2_PROFESSIONAL", "NON_TRIGGER",
        ])

    if tier == "TIER1_PERSONAL":
        template = random.choice(TIER1_PERSONAL_TEMPLATES)
        priority_score = 10
    elif tier == "TIER2_PROFESSIONAL":
        template = random.choice(TIER2_PROFESSIONAL_TEMPLATES)
        priority_score = 5
    elif tier == "NON_TRIGGER":
        template = random.choice(NON_TRIGGER_TEMPLATES)
        priority_score = 0
        tier = "UNCLASSIFIED"
    elif tier == "MIXED":
        template = random.choice(MIXED_TEMPLATES)
        # Mixed templates are classified as TIER1 (personal wins)
        tier = "TIER1_PERSONAL"
        priority_score = 10
    else:
        template = random.choice(TIER1_PERSONAL_TEMPLATES)
        priority_score = 10

    text = _fill_template(template)
    expected = _get_expected_info(text, tier)

    # Generate a random timestamp within the last hour for FIFO ordering
    offset_seconds = random.randint(0, 3600)
    submitted_at = (
        datetime.now(timezone.utc) - timedelta(seconds=offset_seconds)
    ).isoformat()

    return {
        "todo_id": uuid.uuid4().hex[:12],
        "text": text,
        "tier": tier,
        "priority_score": priority_score,
        "expected_agent": expected.get("expected_agent"),
        "expected_tool": expected.get("expected_tool"),
        "required_fields": expected.get("required_fields", []),
        "submitted_at": submitted_at,
        "status": "pending",
    }


def generate_episode_queue(
    min_tier1: int = 1,
    min_tier2: int = 1,
    total: int = 3,
) -> list[dict]:
    """
    Generate a queue of todos guaranteed to have the required tier mix.

    Args:
        min_tier1: Minimum TIER1_PERSONAL todos.
        min_tier2: Minimum TIER2_PROFESSIONAL todos.
        total: Total todos in queue (must be >= min_tier1 + min_tier2).

    Returns:
        UNSORTED list of todo dicts (env sorts it — tests sorting).
    """
    total = max(total, min_tier1 + min_tier2)
    queue = []

    # Generate guaranteed TIER1 todos
    for _ in range(min_tier1):
        queue.append(generate_todo("TIER1_PERSONAL"))

    # Generate guaranteed TIER2 todos
    for _ in range(min_tier2):
        queue.append(generate_todo("TIER2_PROFESSIONAL"))

    # Fill remaining slots randomly
    remaining = total - len(queue)
    for _ in range(remaining):
        tier = random.choice([
            "TIER1_PERSONAL", "TIER2_PROFESSIONAL", "NON_TRIGGER",
        ])
        queue.append(generate_todo(tier))

    # Shuffle to ensure env must sort (testing sorting logic)
    random.shuffle(queue)

    return queue


def generate_batch(
    n_episodes: int,
    split: str = "train",
) -> list[list[dict]]:
    """
    Generate a batch of episode queues.

    Args:
        n_episodes: Number of episodes to generate.
        split: Dataset split name ("train", "eval", "test").

    Returns:
        List of episode queues.
    """
    batch = []
    for _ in range(n_episodes):
        # Vary queue size for diversity
        total = random.randint(2, 5)
        queue = generate_episode_queue(
            min_tier1=1, min_tier2=1, total=total
        )
        batch.append(queue)

    return batch


def save_dataset(
    path: str = "data/butler_dataset.json",
    n_train: int = 500,
    n_eval: int = 100,
    n_test: int = 100,
):
    """
    Save all splits to a single JSON file.

    Format:
    {
        "train": [ [episode1_queue], [episode2_queue], ... ],
        "eval":  [ ... ],
        "test":  [ ... ]
    }
    """
    import os

    dataset = {
        "train": generate_batch(n_train, "train"),
        "eval": generate_batch(n_eval, "eval"),
        "test": generate_batch(n_test, "test"),
    }

    # Ensure directory exists
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"Dataset saved to {path}")
    print(f"  Train: {len(dataset['train'])} episodes")
    print(f"  Eval:  {len(dataset['eval'])} episodes")
    print(f"  Test:  {len(dataset['test'])} episodes")

    # Print tier distribution for first 10 episodes
    print("\nSample tier distribution (first 10 train episodes):")
    for i, episode in enumerate(dataset["train"][:10]):
        tiers = [t["tier"] for t in episode]
        print(f"  Episode {i+1}: {tiers}")


if __name__ == "__main__":
    save_dataset()

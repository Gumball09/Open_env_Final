"""
inference.py — Standalone inference script for Butler.

Run a trained Butler agent on new todos or compare against baseline.

Usage:
  python inference.py --model your-hf-username/butler-grpo \
                      --todo "Remind me to take my vitamins every morning"

  python inference.py --model your-hf-username/butler-grpo \
                      --queue "Remind me to drink water; Schedule a meeting with Priya"

  python inference.py --model your-hf-username/butler-grpo --compare --n_episodes 10
"""

import argparse
import json
import random
import sys
from typing import Optional

from env.butler_env import ButlerEnvironment
from env.observation import build_observation_prompt, SYSTEM_PROMPT
from env.action_space import parse_llm_output, validate_action
from agents.orchestrator import Orchestrator
from data.synthetic_todos import generate_episode_queue


def load_model(model_name: str):
    """
    Load tokenizer and model from HF Hub.
    Uses device_map="auto" for GPU/CPU selection.

    Returns:
        (model, tokenizer)
    """
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM

        print(f"Loading model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype="auto",
        )
        print(f"Model loaded on: {model.device}")
        return model, tokenizer

    except Exception as e:
        print(f"Error loading model: {e}")
        print("Falling back to simulation mode.")
        return None, None


def generate_action(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    """
    Run model inference to generate an action string.

    Args:
        model: The language model.
        tokenizer: The tokenizer.
        prompt: Full prompt string.
        max_new_tokens: Max tokens to generate.

    Returns:
        Generated text string.
    """
    if model is None or tokenizer is None:
        return _simulate_action(prompt)

    try:
        import torch

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode only the new tokens
        input_length = inputs["input_ids"].shape[1]
        generated = tokenizer.decode(
            outputs[0][input_length:], skip_special_tokens=True
        )
        return generated.strip()

    except Exception as e:
        print(f"Inference error: {e}")
        return _simulate_action(prompt)


def _simulate_action(prompt: str) -> str:
    """
    Simulate an action when no model is available.
    Uses the orchestrator to determine the correct action.
    """
    orch = Orchestrator()

    # Extract the current task text from the prompt
    task_text = ""
    for line in prompt.split("\n"):
        if line.strip().startswith("Handle this task first:"):
            task_text = line.split(":", 1)[1].strip()
            break

    if not task_text:
        return json.dumps({
            "tool": "ask_clarification",
            "params": {
                "todo_id": "unknown",
                "field": "task",
                "question": "Could you clarify what you'd like me to do?",
            },
        })

    # Determine the best action
    tier, score = orch.classify_tier(task_text)
    agents = orch.scan_keywords(task_text)

    if not agents:
        return json.dumps({
            "tool": "add_to_kb",
            "params": {
                "todo_id": "unknown",
                "content": task_text,
                "category": "preference",
            },
        })

    agent = agents[0]
    tool_map = {
        "meeting_agent": {
            "tool": "schedule_event",
            "params": {
                "todo_id": "unknown",
                "attendee_email": "attendee@example.com",
                "start_time": "2024-01-15T10:00:00",
                "duration_minutes": 30,
                "title": f"Meeting: {task_text[:50]}",
            },
        },
        "email_agent": {
            "tool": "send_email",
            "params": {
                "todo_id": "unknown",
                "to": "recipient@example.com",
                "subject": f"Re: {task_text[:40]}",
                "body": f"Following up on: {task_text}",
            },
        },
        "knowledge_agent": {
            "tool": "add_to_kb",
            "params": {
                "todo_id": "unknown",
                "content": task_text,
                "category": "preference",
            },
        },
        "habit_agent": {
            "tool": "set_reminder",
            "params": {
                "todo_id": "unknown",
                "label": task_text[:50],
                "frequency": "daily",
                "time_of_day": "08:00",
            },
        },
    }

    action = tool_map.get(agent, tool_map["knowledge_agent"])
    return json.dumps(action)


def run_inference(
    model,
    tokenizer,
    todo_text: str = None,
    queue_text: str = None,
    max_steps: int = 10,
):
    """
    Run Butler inference on todos.

    Args:
        model: Language model (or None for simulation).
        tokenizer: Tokenizer (or None for simulation).
        todo_text: Single todo text.
        queue_text: Semicolon-separated todo texts.
        max_steps: Maximum steps per episode.
    """
    env = ButlerEnvironment()
    orch = Orchestrator()

    # Build todo queue
    if todo_text:
        todos = [_make_todo(todo_text, orch)]
    elif queue_text:
        texts = [t.strip() for t in queue_text.split(";") if t.strip()]
        todos = [_make_todo(t, orch) for t in texts]
    else:
        print("No todo provided. Generating synthetic queue...")
        todos = generate_episode_queue(min_tier1=1, min_tier2=1, total=3)

    # Reset environment with our queue
    obs = env.reset(episode_queue=todos)

    print("=" * 60)
    print("BUTLER INFERENCE")
    print("=" * 60)
    print(f"\nQueue ({len(obs['queue'])} todos):")
    for i, t in enumerate(obs["queue"], 1):
        status = "✓" if t.get("status") == "completed" else "○"
        print(f"  {status} [{t['tier']:20s}] #{i}: {t['text']}")
    print()

    total_reward = 0.0
    violations = 0

    for step in range(1, max_steps + 1):
        if not obs.get("current_todo"):
            print("\n✅ All todos completed!")
            break

        current = obs["current_todo"]
        print(f"─── Step {step} ───")
        print(f"  Todo: \"{current['text']}\"")
        print(f"  Tier: {current['tier']} (priority={current['priority_score']})")

        # Build prompt and generate action
        prompt = (
            f"<|system|>\n{SYSTEM_PROMPT}\n"
            f"<|user|>\n{build_observation_prompt(obs)}\n"
            f"<|assistant|>\n"
        )
        raw_output = generate_action(model, tokenizer, prompt)

        # Parse and validate
        action = parse_llm_output(raw_output)
        if action is None:
            print(f"  ⚠ Could not parse action: {raw_output[:100]}")
            action = {
                "tool": "ask_clarification",
                "params": {
                    "todo_id": current.get("todo_id", ""),
                    "field": "action",
                    "question": "Could not determine action.",
                },
            }

        # Fix todo_id if missing
        if "params" in action:
            if not action["params"].get("todo_id"):
                action["params"]["todo_id"] = current.get("todo_id", "")

        valid, error = validate_action(action)
        if not valid:
            print(f"  ⚠ Invalid action: {error}")

        print(f"  Action: {action['tool']} | Params: {json.dumps(action.get('params', {}), indent=None)[:100]}")

        # Execute step
        obs, reward, done, info = env.step(action)
        total_reward += reward

        if info.get("priority_violation"):
            violations += 1
            print(f"  ⚠ PRIORITY VIOLATION! (-0.3 penalty)")

        print(f"  Reward: {reward:.3f} | Breakdown: {json.dumps(info.get('rubric_breakdown', {}), indent=None)[:100]}")

        if done:
            break

    # Summary
    completed = sum(
        1 for t in env.todo_queue if t.get("status") == "completed"
    )
    total = len(env.todo_queue)

    print("\n" + "=" * 60)
    print("EPISODE SUMMARY")
    print("=" * 60)
    print(f"  Total reward:         {total_reward:.3f}")
    print(f"  Priority violations:  {violations}")
    print(f"  Todos completed:      {completed}/{total}")
    print(f"  Steps used:           {env.step_count}")
    print("=" * 60)

    return total_reward, violations, completed, total


def _make_todo(text: str, orch: Orchestrator) -> dict:
    """Create a todo dict from text."""
    import uuid
    from datetime import datetime, timezone

    tier, score = orch.classify_tier(text)
    return {
        "todo_id": uuid.uuid4().hex[:12],
        "text": text,
        "tier": tier,
        "priority_score": score,
        "expected_agent": orch.get_expected_agent(text),
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "status": "pending",
    }


def compare_baseline_vs_trained(
    trained_model_name: str,
    n_episodes: int = 10,
):
    """
    Run episodes with random baseline and trained model, print comparison.

    Args:
        trained_model_name: HF model identifier.
        n_episodes: Number of evaluation episodes.
    """
    print("Loading trained model...")
    model, tokenizer = load_model(trained_model_name)

    baseline_rewards = []
    trained_rewards = []
    baseline_violations = []
    trained_violations = []

    for ep in range(n_episodes):
        queue = generate_episode_queue(min_tier1=1, min_tier2=1, total=3)

        # Baseline: random actions (always picks first action, ignores priority)
        b_reward, b_viol = _run_baseline_episode(queue)
        baseline_rewards.append(b_reward)
        baseline_violations.append(b_viol)

        # Trained model
        env = ButlerEnvironment()
        obs = env.reset(episode_queue=[t.copy() for t in queue])
        t_reward = 0.0
        t_viol = 0

        for step in range(env.MAX_STEPS_PER_EPISODE):
            if not obs.get("current_todo"):
                break

            prompt = (
                f"<|system|>\n{SYSTEM_PROMPT}\n"
                f"<|user|>\n{build_observation_prompt(obs)}\n"
                f"<|assistant|>\n"
            )
            raw = generate_action(model, tokenizer, prompt)
            action = parse_llm_output(raw)

            if action is None:
                action = {
                    "tool": "ask_clarification",
                    "params": {
                        "todo_id": obs["current_todo"].get("todo_id", ""),
                        "field": "action",
                        "question": "Unclear.",
                    },
                }

            if "params" in action and not action["params"].get("todo_id"):
                action["params"]["todo_id"] = obs["current_todo"].get("todo_id", "")

            obs, reward, done, info = env.step(action)
            t_reward += reward
            if info.get("priority_violation"):
                t_viol += 1
            if done:
                break

        trained_rewards.append(t_reward)
        trained_violations.append(t_viol)

    # Print comparison table
    print("\n" + "=" * 85)
    print("BASELINE vs TRAINED COMPARISON")
    print("=" * 85)
    print(
        f"{'Episode':>8} | {'Baseline Reward':>15} | {'Trained Reward':>14} | "
        f"{'Base Violations':>15} | {'Train Violations':>16}"
    )
    print("-" * 85)

    for i in range(n_episodes):
        print(
            f"{i+1:>8} | {baseline_rewards[i]:>15.3f} | "
            f"{trained_rewards[i]:>14.3f} | "
            f"{baseline_violations[i]:>15} | "
            f"{trained_violations[i]:>16}"
        )

    print("-" * 85)
    avg_br = sum(baseline_rewards) / n_episodes
    avg_tr = sum(trained_rewards) / n_episodes
    avg_bv = sum(baseline_violations) / n_episodes
    avg_tv = sum(trained_violations) / n_episodes
    print(
        f"{'Average':>8} | {avg_br:>15.3f} | {avg_tr:>14.3f} | "
        f"{avg_bv:>15.1f} | {avg_tv:>16.1f}"
    )
    print("=" * 85)

    return baseline_rewards, trained_rewards


def _run_baseline_episode(queue: list[dict]) -> tuple[float, int]:
    """
    Run a baseline episode with random/naive actions.
    Always picks a random tool, ignores priority ordering.
    """
    env = ButlerEnvironment()
    obs = env.reset(episode_queue=[t.copy() for t in queue])

    total_reward = 0.0
    violations = 0

    tools = ["schedule_event", "send_email", "set_reminder", "add_to_kb"]

    for step in range(env.MAX_STEPS_PER_EPISODE):
        if not obs.get("current_todo"):
            break

        current = obs["current_todo"]

        # Pick a random tool with random params
        tool = random.choice(tools)
        todo_id = current.get("todo_id", "")

        if tool == "schedule_event":
            action = {
                "tool": tool,
                "params": {
                    "todo_id": todo_id,
                    "attendee_email": "test@test.com",
                    "start_time": "2024-01-15T10:00:00",
                    "duration_minutes": 30,
                    "title": "Random meeting",
                },
            }
        elif tool == "send_email":
            action = {
                "tool": tool,
                "params": {
                    "todo_id": todo_id,
                    "to": "test@test.com",
                    "subject": "Random email",
                    "body": "Random body",
                },
            }
        elif tool == "set_reminder":
            action = {
                "tool": tool,
                "params": {
                    "todo_id": todo_id,
                    "label": "Random reminder",
                    "frequency": "daily",
                    "time_of_day": "08:00",
                },
            }
        else:
            action = {
                "tool": tool,
                "params": {
                    "todo_id": todo_id,
                    "content": "Random content",
                    "category": "preference",
                },
            }

        obs, reward, done, info = env.step(action)
        total_reward += reward
        if info.get("priority_violation"):
            violations += 1
        if done:
            break

    return total_reward, violations


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Butler — Standalone Inference Script"
    )
    parser.add_argument(
        "--model", required=True,
        help="HF model name (e.g. your-username/butler-grpo)",
    )
    parser.add_argument(
        "--todo", default=None,
        help="Single todo text to process",
    )
    parser.add_argument(
        "--queue", default=None,
        help="Semicolon-separated todo texts",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Run baseline vs trained comparison",
    )
    parser.add_argument(
        "--n_episodes", type=int, default=10,
        help="Number of episodes for comparison",
    )
    args = parser.parse_args()

    model, tokenizer = load_model(args.model)

    if args.compare:
        compare_baseline_vs_trained(args.model, args.n_episodes)
    else:
        run_inference(
            model, tokenizer,
            todo_text=args.todo,
            queue_text=args.queue,
        )

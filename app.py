"""
app.py — Gradio demo for Butler (HF Spaces entry point).

No React, no Node. Pure Python Gradio interface.
Lazy-loads all dependencies — no external API calls at import time.
"""

import json
import os
import gradio as gr

# Attempt to load .env variables if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ─── Lazy-loaded singletons ───────────────────────────────────────────────────

_orchestrator = None
_env = None


def _get_orchestrator():
    global _orchestrator
    if _orchestrator is None:
        from agents.orchestrator import Orchestrator
        _orchestrator = Orchestrator()
    return _orchestrator


def _get_env():
    global _env
    if _env is None:
        from env.butler_env import ButlerEnvironment
        _env = ButlerEnvironment()
    return _env


# ─── Core processing ──────────────────────────────────────────────────────────

def process_todo(todo_text: str, user_name: str = "User") -> tuple[str, str, str]:
    """
    Process a single todo and return tier, agent, and action info.

    Returns:
        (tier_label, agent_label, action_description)
    """
    if not todo_text or not todo_text.strip():
        return "No input", "No agent", "Please enter a task."

    orch = _get_orchestrator()

    # Classify tier
    tier, priority_score = orch.classify_tier(todo_text)

    tier_labels = {
        "TIER1_PERSONAL": f"🟢 PERSONAL (Priority: {priority_score})",
        "TIER2_PROFESSIONAL": f"🔵 PROFESSIONAL (Priority: {priority_score})",
        "UNCLASSIFIED": f"⚪ UNCLASSIFIED (Priority: {priority_score})",
    }
    tier_label = tier_labels.get(tier, f"⚪ {tier}")

    # Scan keywords and route
    agents = orch.scan_keywords(todo_text)
    if agents:
        agent_label = ", ".join(f"🤖 {a}" for a in agents)
    else:
        agent_label = "⚠️ No agent matched (non-actionable task)"

    # Generate action via environment
    try:
        env = _get_env()
        import uuid
        from datetime import datetime, timezone

        todo = {
            "todo_id": uuid.uuid4().hex[:12],
            "text": todo_text,
            "tier": tier,
            "priority_score": priority_score,
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "status": "pending",
        }

        obs = env.reset(episode_queue=[todo])

        # Use orchestrator to determine action
        route_actions = orch.route(todo_text, todo["todo_id"])

        action_lines = []
        for ra in route_actions:
            if ra.get("routed"):
                agent_name = ra["params"]["agent_name"]
                expected_tool = orch.get_expected_tool(agent_name)
                action_lines.append(
                    f"✅ Routed to: {agent_name}\n"
                    f"   Expected tool: {expected_tool or 'N/A'}\n"
                    f"   Tier: {ra['params']['tier']}\n"
                    f"   Priority: {ra['params']['priority_score']}"
                )
            else:
                action_lines.append(
                    "ℹ️ No actionable keywords detected.\n"
                    "   This task doesn't match any Butler agent.\n"
                    "   It may be a general task (groceries, cooking, etc.)"
                )

        action_text = "\n\n".join(action_lines)

    except Exception as e:
        action_text = f"Error processing: {str(e)}"

    return tier_label, agent_label, action_text


def process_queue(queue_text: str, user_name: str = "User") -> str:
    """
    Process a semicolon-separated queue of todos.
    Shows priority ordering.

    Returns:
        Formatted queue analysis string.
    """
    if not queue_text or not queue_text.strip():
        return "Please enter tasks separated by semicolons."

    orch = _get_orchestrator()
    texts = [t.strip() for t in queue_text.split(";") if t.strip()]

    if not texts:
        return "No valid tasks found."

    # Build and classify queue
    todos = []
    import uuid
    from datetime import datetime, timezone, timedelta

    for i, text in enumerate(texts):
        tier, score = orch.classify_tier(text)
        todos.append({
            "todo_id": uuid.uuid4().hex[:12],
            "text": text,
            "tier": tier,
            "priority_score": score,
            "submitted_at": (
                datetime.now(timezone.utc) - timedelta(seconds=len(texts) - i)
            ).isoformat(),
            "status": "pending",
        })

    # Sort by priority
    sorted_todos = orch.sort_queue(todos)

    # Format output
    lines = ["📋 **Priority-Sorted Queue:**\n"]
    tier_icons = {
        "TIER1_PERSONAL": "🟢",
        "TIER2_PROFESSIONAL": "🔵",
        "UNCLASSIFIED": "⚪",
    }

    for i, todo in enumerate(sorted_todos, 1):
        icon = tier_icons.get(todo["tier"], "⚪")
        agents = orch.scan_keywords(todo["text"])
        agent_str = ", ".join(agents) if agents else "none"
        lines.append(
            f"{i}. {icon} **[{todo['tier']}]** (priority={todo['priority_score']})\n"
            f"   {todo['text']}\n"
            f"   → Agent: {agent_str}"
        )

    # Priority check
    has_tier1 = any(t["tier"] == "TIER1_PERSONAL" for t in sorted_todos)
    has_tier2 = any(t["tier"] == "TIER2_PROFESSIONAL" for t in sorted_todos)

    if has_tier1 and has_tier2:
        lines.append(
            "\n⚡ **Priority Rule Active:** Personal tasks will be "
            "handled before professional tasks."
        )

    return "\n\n".join(lines)


def generate_synthetic_demo() -> str:
    """Generate a synthetic episode queue for demo purposes."""
    from data.synthetic_todos import generate_episode_queue

    queue = generate_episode_queue(min_tier1=2, min_tier2=2, total=5)
    orch = _get_orchestrator()
    sorted_q = orch.sort_queue(queue)

    lines = ["🎲 **Generated Synthetic Queue:**\n"]
    tier_icons = {
        "TIER1_PERSONAL": "🟢",
        "TIER2_PROFESSIONAL": "🔵",
        "UNCLASSIFIED": "⚪",
    }

    for i, todo in enumerate(sorted_q, 1):
        icon = tier_icons.get(todo["tier"], "⚪")
        lines.append(
            f"{i}. {icon} **[{todo['tier']}]** (priority={todo['priority_score']})\n"
            f"   {todo['text']}"
        )

    return "\n\n".join(lines)


def run_auto_pilot(user_name: str = "User") -> str:
    """Run the fully automated AutoReply daemon cycle."""
    try:
        from agents.auto_reply_agent import AutoReplyAgent
        from tools.gmail_tool import GmailTool
        from tools.kb_tool import KBTool
        
        # Instantiate tools
        gmail_tool = GmailTool()
        kb_tool = KBTool()
        
        # Initialize daemon agent
        daemon = AutoReplyAgent(gmail_tool=gmail_tool, kb_tool=kb_tool)
        
        user_context = {"name": user_name}
        logs = daemon.run_automation_cycle(user_context=user_context)
        
        if not logs:
            return "No actions taken during this cycle."
            
        output = ["🔄 **Auto-Pilot Cycle Complete**\n"]
        for log in logs:
            status = log.get("status")
            msg = log.get("message", "")
            
            if status == "error":
                output.append(f"❌ {msg}")
            elif status == "warning":
                output.append(f"⚠️ {msg}")
            elif status == "success":
                output.append(f"✅ {msg}")
                if "draft" in log:
                    output.append(f"\n> **Draft Sent:**\n> {log['draft'].replace(chr(10), chr(10)+'> ')}\n")
            elif status == "processing":
                output.append(f"🔍 {msg}")
            else:
                output.append(f"ℹ️ {msg}")
                
        return "\n\n".join(output)
        
    except Exception as e:
        return f"❌ Error running Auto-Pilot: {str(e)}"

# ─── Gradio App ───────────────────────────────────────────────────────────────

def build_gradio_app():
    """Build the Gradio interface for Butler."""

    css = """
    .gradio-container {
        max-width: 1100px !important;
        margin: auto !important;
    }
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5em;
        font-weight: 800;
        margin-bottom: 0;
    }
    .subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 1.1em;
        margin-top: -10px;
    }
    """

    with gr.Blocks(
        title="Butler — AI Task Orchestrator"
    ) as demo:

        gr.HTML(
            '<h1 class="main-header">🎩 Butler</h1>'
            '<p class="subtitle">'
            "AI Personal Task Orchestrator — OpenEnv Hackathon"
            "</p>"
        )

        gr.Markdown(
            "> **Priority Rule:** Personal tasks (health, family, habits) are "
            "**always** handled before professional tasks (meetings, emails). "
            "This is baked into the RL reward function."
        )

        with gr.Tabs():
            # ── Tab 1: Single Task ──────────────────────────────────────
            with gr.Tab("📝 Single Task"):
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("### Add a Task")
                        todo_input = gr.Textbox(
                            placeholder="e.g. Remind me to take my vitamins every morning at 8 AM",
                            label="Your todo",
                            lines=2,
                        )
                        user_name = gr.Textbox(
                            value="User", label="Your name"
                        )
                        submit_btn = gr.Button(
                            "🚀 Submit to Butler", variant="primary", size="lg"
                        )

                    with gr.Column(scale=3):
                        gr.Markdown("### Butler's Response")
                        tier_out = gr.Textbox(label="Task Tier", interactive=False)
                        agent_out = gr.Textbox(label="Agent Assigned", interactive=False)
                        action_out = gr.Textbox(
                            label="Action Taken", lines=6, interactive=False
                        )

                submit_btn.click(
                    fn=process_todo,
                    inputs=[todo_input, user_name],
                    outputs=[tier_out, agent_out, action_out],
                )

                gr.Markdown("---")
                gr.Markdown("### 💡 Try These Examples")
                gr.Examples(
                    examples=[
                        ["Remind me to take my vitamins every morning at 8 AM"],
                        ["Schedule a meeting with Priya about the Q3 report"],
                        ["Set a daily gym reminder at 6 AM"],
                        ["Reply to Rahul's email about the contract"],
                        ["Remind me to call my mom this Sunday"],
                        ["Buy groceries from the store"],
                        ["Remind me to take medicine AND reply to Sarah's email"],
                        ["Set up weekly standup meetings with the engineering team"],
                    ],
                    inputs=todo_input,
                    label="Example tasks",
                )

            # ── Tab 2: Queue Priority Demo ──────────────────────────────
            with gr.Tab("📊 Queue Priority"):
                gr.Markdown(
                    "### Priority Queue Demo\n"
                    "Enter multiple tasks separated by **semicolons** to see "
                    "how Butler orders them by priority."
                )

                queue_input = gr.Textbox(
                    placeholder=(
                        "e.g. Schedule meeting with Priya; "
                        "Remind me to take vitamins; "
                        "Reply to client email"
                    ),
                    label="Task queue (semicolon-separated)",
                    lines=3,
                )
                queue_name = gr.Textbox(value="User", label="Your name")
                queue_btn = gr.Button(
                    "🔀 Analyze Queue Priority", variant="primary"
                )
                queue_out = gr.Markdown(label="Priority Analysis")

                queue_btn.click(
                    fn=process_queue,
                    inputs=[queue_input, queue_name],
                    outputs=[queue_out],
                )

                gr.Markdown("---")
                gr.Examples(
                    examples=[
                        [
                            "Take my vitamins at 8 AM; "
                            "Schedule meeting with Priya; "
                            "Set daily gym reminder; "
                            "Reply to Rahul's email"
                        ],
                        [
                            "Call my mom this Sunday; "
                            "Email Carlos about the budget review; "
                            "Daily meditation habit; "
                            "Follow up with Sarah re: contract"
                        ],
                    ],
                    inputs=queue_input,
                    label="Example queues",
                )

            # ── Tab 3: Synthetic Data ───────────────────────────────────
            with gr.Tab("🎲 Synthetic Data"):
                gr.Markdown(
                    "### Synthetic Episode Generator\n"
                    "Generate random todo queues used for RL training. "
                    "Each episode has a guaranteed mix of personal and "
                    "professional tasks."
                )
                gen_btn = gr.Button(
                    "🎲 Generate Random Queue", variant="secondary"
                )
                gen_out = gr.Markdown(label="Generated Queue")

                gen_btn.click(fn=generate_synthetic_demo, outputs=[gen_out])

            # ── Tab 4: Auto-Pilot ───────────────────────────────────────
            with gr.Tab("🤖 Auto-Pilot"):
                gr.Markdown(
                    "### Fully Automated Background Agent\n"
                    "Unlike the request-driven agents, the **AutoReplyAgent** runs completely autonomously. "
                    "It scans your unread emails, queries your Knowledge Base to see if it knows the answer, "
                    "and automatically drafts and sends replies on your behalf."
                )
            
                auto_user_name = gr.Textbox(value="User", label="Your name")
                auto_btn = gr.Button("🔄 Run Auto-Pilot Cycle", variant="primary")
                auto_out = gr.Markdown(label="Automation Logs")
            
                auto_btn.click(fn=run_auto_pilot, inputs=[auto_user_name], outputs=[auto_out])

            # ── Tab 5: About ────────────────────────────────────────────
            with gr.Tab("ℹ️ About"):
                gr.Markdown(
                    """
    ### How Butler Works

    Butler is a **multi-agent personal task orchestration system** built as
    an OpenEnv reinforcement learning environment.

    #### 🏗️ Architecture
    - **Orchestrator** — Classifies tasks by tier and routes to sub-agents
    - **Meeting Agent** — Schedules calendar events and sends confirmations
    - **Email Agent** — Drafts replies and manages email workflows
    - **Knowledge Agent** — Manages the user's personal knowledge base
    - **Habit Agent** — Creates recurring reminders and tracks habits

    #### 🎯 Priority System
    | Tier | Type | Priority | Examples |
    |------|------|----------|----------|
    | 🟢 TIER 1 | Personal | 10 | Health, family, habits, wellness |
    | 🔵 TIER 2 | Professional | 5 | Meetings, emails, deadlines |
    | ⚪ Unclassified | Other | 0 | Groceries, cooking, entertainment |

    #### 🏆 Reward Rubric (5 components)
    1. **Priority Ordering** (25%) — Personal before professional
    2. **Correct Routing** (20%) — Right agent for the task
    3. **Action Completeness** (20%) — All required fields provided
    4. **API Call Success** (20%) — Successful tool execution
    5. **No Over-Triggering** (15%) — Correct abstention on non-tasks

    #### 🔧 Stack
    - Python 3.11 + Gradio
    - OpenEnv (MCPEnvironment)
    - Unsloth + HF TRL (GRPO training)
    - Google Calendar + Gmail APIs
    - HF Inference API (Qwen2.5-7B-Instruct)

    ---

    **Built for the OpenEnv Hackathon** 🚀
                    """
                )

        return demo


    # ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = build_gradio_app()
    
    # CSS and Theme defined here to pass to launch() for Gradio 6.0+ compatibility
    css = """
    .gradio-container {
        max-width: 1100px !important;
        margin: auto !important;
    }
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5em;
        font-weight: 800;
        margin-bottom: 0;
    }
    .subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 1.1em;
        margin-top: -10px;
    }
    """
    
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=True,
        css=css,
        theme=gr.themes.Soft(primary_hue="purple", secondary_hue="blue"),
    )

# 🎩 Butler — AI Personal Task Orchestrator

**A multi-agent personal task orchestration system built as an OpenEnv reinforcement learning environment.**

Butler learns to prioritize personal tasks (health, family, habits) over professional tasks (meetings, emails, deadlines) through RL training with GRPO.

[![HF Spaces](https://img.shields.io/badge/🤗-HF%20Spaces-blue)](https://huggingface.co/spaces/your-team/butler)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-team/butler-openenv/blob/main/training/butler_grpo_colab.ipynb)

---

## 🎯 Core Innovation: Priority-Based Task Ordering

Butler's key design principle: **personal wellbeing comes first**.

| Tier | Type | Priority | Examples |
|------|------|----------|----------|
| 🟢 TIER 1 | Personal | 10 | Health, family, habits, wellness, therapy |
| 🔵 TIER 2 | Professional | 5 | Meetings, emails, deadlines, deliverables |
| ⚪ Unclassified | Other | 0 | Groceries, entertainment, general tasks |

When the agent handles a TIER 2 task while any TIER 1 task is pending, it receives a **-0.3 reward penalty**. This trains the model to always prioritize personal wellbeing.

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────┐
│                 Butler Environment            │
│              (MCPEnvironment)                 │
├──────────────────────────────────────────────┤
│                                              │
│  ┌─────────────┐    ┌──────────────────┐     │
│  │ Orchestrator │───▶│ Priority Router  │     │
│  └──────┬──────┘    └──────────────────┘     │
│         │                                     │
│    ┌────┴────┬──────────┬──────────┐         │
│    ▼         ▼          ▼          ▼         │
│ ┌──────┐ ┌──────┐ ┌──────────┐ ┌──────┐    │
│ │Meet. │ │Email │ │Knowledge │ │Habit │    │
│ │Agent │ │Agent │ │  Agent   │ │Agent │    │
│ └──┬───┘ └──┬───┘ └────┬─────┘ └──┬───┘    │
│    │        │          │          │         │
│ ┌──┴───┐ ┌──┴──┐   ┌──┴──┐   ┌──┴────┐    │
│ │Cal.  │ │Gmail│   │ KB  │   │Remind.│    │
│ │Tool  │ │Tool │   │Tool │   │ Tool  │    │
│ └──────┘ └─────┘   └─────┘   └───────┘    │
│                                              │
│  ┌──────────────────────────────────────┐    │
│  │        Reward Rubric (5 components)  │    │
│  │  Priority | Routing | Completeness  │    │
│  │  API Success | Over-triggering      │    │
│  └──────────────────────────────────────┘    │
└──────────────────────────────────────────────┘
```

---

## 🛠️ Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11 |
| Environment | OpenEnv (MCPEnvironment) |
| Demo UI | Gradio (HF Spaces) |
| LLM Calls | HF Inference API (Qwen2.5-7B-Instruct) |
| Training | Unsloth + HF TRL (GRPO) |
| Inference | Standalone `inference.py` |
| Storage | `butler_kb.json` (local file) |
| Auth | Google OAuth 2.0 (Calendar + Gmail) |
| Deployment | Hugging Face Spaces (Docker SDK) |

---

## 📁 Project Structure

```
butler-openenv/
├── openenv.yaml                    # OpenEnv manifest
├── Dockerfile                      # HF Spaces Docker config
├── requirements.txt
├── .env.example
├── README.md
│
├── env/
│   ├── butler_env.py               # MCPEnvironment subclass (core)
│   ├── observation.py              # Observation space + prompt templates
│   └── action_space.py             # 7 tool schemas + validation
│
├── agents/
│   ├── orchestrator.py             # Keyword scanner + priority router
│   ├── meeting_agent.py            # Calendar scheduling
│   ├── email_agent.py              # Email drafting + sending
│   ├── knowledge_agent.py          # KB management
│   └── habit_agent.py              # Habits + reminders
│
├── reward/
│   └── rubric.py                   # 5-component composable rubric
│
├── tools/
│   ├── calendar_tool.py            # Google Calendar API
│   ├── gmail_tool.py               # Gmail API
│   ├── kb_tool.py                  # Local JSON knowledge base
│   └── reminder_tool.py            # Reminder/habit tracking
│
├── auth/
│   └── google_oauth.py             # OAuth 2.0 credential management
│
├── data/
│   └── synthetic_todos.py          # Synthetic training data generator
│
├── training/
│   └── butler_grpo_colab.ipynb     # Complete Colab training notebook
│
├── inference.py                    # Standalone inference script
└── app.py                          # Gradio demo (HF Spaces entry)
```

---

## 🚀 Quick Start

### 1. Local Demo

```bash
# Navigate to the project directory
cd path/to/butler-openenv

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Run Gradio demo
python app.py
# Open http://localhost:7860
```

### 2. Generate Training Data

```bash
python -c "from data.synthetic_todos import save_dataset; save_dataset()"
```

### 3. Run Inference

```bash
# Single todo
python inference.py --model your-username/butler-grpo \
    --todo "Remind me to take my vitamins every morning"

# Multiple todos (priority ordering test)
python inference.py --model your-username/butler-grpo \
    --queue "Remind me to drink water; Schedule a meeting with Priya"

# Baseline vs trained comparison
python inference.py --model your-username/butler-grpo --compare
```

### 4. Train with GRPO

Open `training/butler_grpo_colab.ipynb` in Google Colab and follow the cells.

---

## 🏆 Reward Rubric

The reward function has **5 deterministic components** (no LLM-as-judge in the reward loop):

| Component | Weight | Description |
|-----------|--------|-------------|
| Priority Ordering | 25% | Personal tasks handled before professional |
| Correct Routing | 20% | Right agent selected for the task type |
| Action Completeness | 20% | All required fields provided |
| API Call Success | 20% | Tool executed successfully |
| No Over-Triggering | 15% | Correct abstention on non-actionable tasks |

**Priority violation penalty:** -0.3 applied ON TOP of the rubric score when a TIER 2 task is chosen while TIER 1 tasks are pending.

---

## 🔧 Available Tools

| Tool | Description | Agent |
|------|-------------|-------|
| `route_to_agent` | Route todo to a sub-agent | Orchestrator |
| `ask_clarification` | Request missing information | Any |
| `schedule_event` | Create Google Calendar event | Meeting Agent |
| `send_email` | Send email via Gmail | Email Agent |
| `draft_reply` | AI-draft email reply | Email Agent |
| `add_to_kb` | Save to knowledge base | Knowledge Agent |
| `set_reminder` | Create recurring reminder | Habit Agent |

---

## 🔐 Google OAuth Setup

1. Create a project in [Google Cloud Console](https://console.cloud.google.com/)
2. Enable Calendar API and Gmail API
3. Create OAuth 2.0 credentials (Desktop app type)
4. Download `credentials.json` to the project root
5. Set environment variables (see `.env.example`)

---

## 📊 Training Results

After GRPO training, the agent learns to:
- ✅ Always handle personal tasks before professional tasks
- ✅ Route tasks to the correct sub-agent
- ✅ Ask for missing information before acting
- ✅ Abstain from acting on non-actionable tasks
- ✅ Use the correct tool with complete parameters

---

## 📝 License

MIT License

---

**Built for the OpenEnv Hackathon 🚀**

# 🎩 Butler — AI Personal Task Orchestrator

**A multi-agent personal task orchestration system built as an OpenEnv reinforcement learning environment.**
<img width="1841" height="957" alt="image" src="https://github.com/user-attachments/assets/5903c911-52aa-41ee-91c2-a0bddc41fe34" />


Butler learns to prioritize personal tasks (health, family, habits) over professional tasks (meetings, emails, deadlines) through RL training with GRPO.

[![HF Spaces](https://img.shields.io/badge/🤗-HF%20Spaces-blue)](https://huggingface.co/spaces/your-team/butler)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-team/butler-openenv/blob/main/training/butler_grpo_colab.ipynb)

---

## 🚨 The Core Problem Statement: Priority Inversion

In modern executive environments, highly driven professionals frequently suffer from **Priority Inversion**. Because professional tasks (like scheduling meetings, replying to clients, hitting deadlines) carry immediate, visible social pressure, they easily bypass personal tasks (like going to the gym, drinking water, or attending family events) which often have delayed, private consequences.

We face these micro-conflicts constantly: missing a dinner due to last-minute work, or navigating the nuance of replying to tough emails while ignoring a hydration reminder.

The challenge in AI research is: **How do we build an autonomous agent that doesn't just blindly execute tasks, but actually understands and enforces human value structures?** We needed a realistic simulation of handling personal tasks and conflicts, managing them as intelligent delegations.

---

## 🎯 The Environment: An MDP Formulation of Modern Life

To train an agent using RL, we framed the user's daily life as a **Markov Decision Process (MDP)** within a scalable OpenEnv `MCPEnvironment`.

- **State Space ($S$):** The agent observes a dynamic queue of pending ToDos (1 to 5 at a time) and a rich semantic user context injected from a local JSON Knowledge Base (e.g., timezone, communication style, existing commitments).
- **Action Space ($A$):** The agent can take 7 discrete parameterized actions ranging from `route_to_agent`, `ask_clarification`, to tool-specific executions like `schedule_event` and `draft_reply`.
- **Transition Dynamics ($T$):** Successfully completing a task pops it from the queue and updates the environmental state. Failing an API call or lacking parameters leaves the task pending.

By framing personal management as an MDP, we provide the agent with a sandbox to simulate the consequences of handling (or mis-handling) conflicting priorities.

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

## 🤖 The Agentic Architecture: Specialized Sub-Agents

Butler operates as a centralized CMS monitored continuously by a routing Orchestrator and specialized sub-agents. We mapped specific conceptual clusters to tools to ground the LLM's outputs:

- **Meeting Agent:** Activated by keywords like "meeting" or "standup". It extracts parameters (email, time, duration), schedules the meeting using the Google Calendar API, and sends an automated template via the Gmail API to remind attendees. Crucially, when a meeting happens, it reviews the summary for action items to recursively queue future tasks.
- **Email Agent:** Handles deep Gmail integration. It prioritizes important emails in the CMS. We also implemented an **Auto-Pilot daemon**—a background process that scans unread mail, queries the Knowledge Base for context, and autonomously reasons to draft and send contextualized AI replies without manual intervention.
- **Knowledge Base Agent:** Solves the "memory" problem in LLM agents. As Butler schedules meetings and learns preferences, context is saved locally. Users can trigger a Q&A session directly with Butler, allowing the agent to perform Retrieval-Augmented Generation (RAG) over the user's life data.
- **Habit Agent:** Triggered by "remind" or "health". It bypasses the traditional calendar and interfaces directly with a Reminder Tool to set up daily recurring alarms for going to the gym, drinking water, or focused work blocks.

---

## 🏆 The Reward Rubric: Aligning Values Mathematically

Training an LLM to "care about health" requires mathematically rigorous reward shaping. We designed a deterministic, 5-component composable reward rubric:

1. **Priority Ordering (25%):** Did the agent handle Tier 1 (Personal) tasks before Tier 2 (Professional) tasks?
2. **Correct Routing (20%):** Did the orchestrator select the right agent based on the semantic intent?
3. **Action Completeness (20%):** Were all required API parameters (e.g., time, email, subject) synthesized correctly?
4. **API Call Success (20%):** Did the external API (Google Calendar/Gmail) accept the payload?
5. **No Over-Triggering (15%):** Did the agent correctly abstain from non-actionable tasks (e.g., "buy groceries")?

### The `-0.3` Priority Penalty
The core innovation of Butler is its strict tier system.

| Tier | Type | Priority | Examples |
|------|------|----------|----------|
| 🟢 TIER 1 | Personal | 10 | Health, family, habits, wellness, therapy |
| 🔵 TIER 2 | Professional | 5 | Meetings, emails, deadlines, deliverables |
| ⚪ Unclassified | Other | 0 | Groceries, entertainment, general tasks |

If the agent routes or acts upon a TIER 2 task while *any* TIER 1 task remains pending in the queue, a massive **`-0.3` reward penalty** is applied on top of the rubric. This creates a steep gradient that forces the model to learn that personal wellbeing is a non-negotiable prerequisite to professional work.

---

## 🧠 Training with GRPO & Results

We fine-tuned Hugging Face's `Qwen2.5-7B-Instruct` model (quantized via Unsloth) using **Group Relative Policy Optimization (GRPO)** via the `trl` library.

Unlike standard PPO, GRPO eliminates the need for a separate value model by normalizing the rewards of a group of sampled outputs against each other. This dramatically reduces memory overhead, allowing us to train a complex, multi-tool reasoning agent locally.

### What Changed After the Training?

Before training, the baseline `Qwen2.5` model treated the environment like a standard chat interface: it hallucinated parameters, triggered tools on un-actionable text, and processed the queue in a naive FIFO (First-In, First-Out) manner, entirely ignoring the priority structure.

After GRPO training, the behavioral shift was profound:
- ✅ **Value Alignment:** The agent learned to *always* handle personal tasks (Tier 1) before professional tasks (Tier 2), internalizing the `-0.3` penalty.
- ✅ **Precision Routing:** It mapped tasks to the correct sub-agent with near-perfect accuracy.
- ✅ **Parameter Synthesis:** It learned to extract and format variables specifically for the Gmail and Google Calendar APIs, asking for clarification only when data was truly missing.
- ✅ **Over-Triggering Restraint:** It abstained from acting on non-actionable tasks.

### Empirical Data

**Project Reward vs Step (500 Steps)**  
Over 500 steps, we observe the model escaping local optima (where it simply tried to do the easiest task first) and converging on a policy that maximizes the 5-component rubric.  
<img width="846" height="470" alt="image" src="https://github.com/user-attachments/assets/1ca2cc5a-a86b-46b7-92ac-f314c2674d76" />


**Project Reward Vs Step (50 Steps)**  
In the first 50 steps, the model experiences rapid policy adaptation as it hits the `-0.3` priority penalty repeatedly, causing a sharp initial correction in behavior.  
<img width="990" height="490" alt="image" src="https://github.com/user-attachments/assets/d97c98e7-d6cf-490d-b8be-1a1b5a45303d" />


**Baseline vs Trained Butler (50 Eval episodes)**  
This evaluation clearly demonstrates the trained model successfully completing full MDP trajectories (clearing the queue) whereas the baseline consistently fails due to tool hallucinations and priority violations.  
<img width="841" height="723" alt="image" src="https://github.com/user-attachments/assets/d7eb14be-cc5c-47b2-8663-d304f24d5f9e" />

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
<img width="845" height="507" alt="image" src="https://github.com/user-attachments/assets/2554dc7c-e514-4443-8290-6817aa50b3f0" />


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

## 🌟 Why Does It Matter?

The implications of Butler extend far beyond a hackathon project:

1. **For Agentic AI Research:** Butler demonstrates how GRPO can be applied to complex OpenEnv environments to train models that prioritize *abstract values* (wellbeing) over indiscriminate task completion. It proves that we can shape an LLM's decision-making framework mathematically.
2. **For Software Architecture:** The project provides a scalable blueprint for building centralized LLM routing systems (leveraging Hugging Face with Cursor fallbacks) that interface safely with real-world APIs (Google Calendar, Gmail) and local memory stores.
3. **For the End User:** Butler represents a shift from "Assistants" to "Orchestrators." It automates the mundane while actively enforcing healthy boundaries, ensuring that highly driven individuals don't miss their life in the pursuit of their work.

---

**Built for the OpenEnv Hackathon 🚀**

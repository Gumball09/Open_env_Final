# Building Butler: An AI Personal Task Orchestrator Using OpenEnv & RL

## 1. The Problem Statement
A lot of times, highly driven and busy professionals tend to forget things that are vital for them, such as their health, family commitments, and personal habits. At the same time, managing professional commitments like meetings and emails can easily take over their lives. We face real conflicts every day—such as handling a dinner conflict due to last-minute work, or navigating the nuance of replying to tough emails. 

We needed a system that offers **real personalized task handling**. The goal was to create an environment that gives the model a realistic simulation of handling personal tasks and conflicts, and seamlessly managing them as intelligent delegations.

## 2. The Environment
We built **"Butler"**, a centralized to-do environment acting as a single CMS monitored continuously by AI agents. It's built as a multi-agent personal task orchestration system leveraging an OpenEnv reinforcement learning environment (`MCPEnvironment`).

### What does the agent see?
The agent observes a queue of pending ToDos (1-5 at a time) and a rich user context fetched from a local JSON Knowledge Base (KB). It sees the full state of incoming tasks and must decide the correct course of action based on a heavily penalized priority system.

### How does it work?
Butler routes tasks based on contextual keywords mapped to specialized AI sub-agents:
*   **Meeting Agent:** Activated by keywords like "Meeting" or "meetings". It asks the user for details (email, time, duration), schedules the meeting using the Google Calendar API, and sends an automated template via the Gmail API to remind attendees. It tracks replies for rescheduling, notifying the user. When a meeting happens, it reviews the summary for action items to queue future tasks.
*   **Email Agent:** Integrates directly with the user's authenticated Gmail. It finds important emails using keywords, prioritizes them in the CMS, and offers to auto-send AI-drafted replies. An Auto-Pilot daemon can even scan, query the Knowledge Base, and autonomously reason to send fully contextualized responses.
*   **Knowledge Base Agent:** Maintains a Q&A knowledge base. As Butler schedules meetings and learns preferences, context is saved locally (with user permission). Users can directly ask Butler questions based on past information.
*   **Habit Agent:** If a ToDo contains the word "Remind", it activates health and habit reminders. It sets up daily alarms for going to the gym, drinking water, or focused work blocks.

### How does it matter?
Butler enforces a strict priority structure: **Tier 1 (Personal: Health, family) vs. Tier 2 (Professional: Meetings, emails)**. By utilizing a deterministic reward rubric, if the agent handles a Tier 2 professional task while a Tier 1 personal task is pending, it receives a `-0.3` penalty. This simulates realistic conflict resolution by forcing the AI to prioritize personal well-being over professional deliverables.

## 3. Results: What Changed After the Training?
We trained Butler using Hugging Face's `Qwen2.5-7B-Instruct` model (via Unsloth) and the GRPO (Group Relative Policy Optimization) algorithm.

Before training, the baseline model struggled to order tasks correctly and often skipped required tool parameters or hallucinated actions. After GRPO training, the model completely transformed. It learned to:
*   ✅ Always handle personal tasks (Tier 1) before professional tasks (Tier 2).
*   ✅ Route tasks to the correct sub-agent accurately.
*   ✅ Ask for missing information before acting.
*   ✅ Abstain from acting on non-actionable tasks.
*   ✅ Use the correct APIs (Gmail, Calendar) with complete parameters.

### Project Reward vs Step (500 Steps)
*This graph highlights the sustained reward growth across 500 training steps.*
![Project reward vs Step (500 Steps)](./assets/reward_500.png)

### Project Reward Vs Step (50 Steps)
*This graph details the rapid early learning phase of the model adapting to priority penalties.*
![Project reward Vs Step (50 Steps)](./assets/reward_50.png)

### Baseline vs Trained Butler (50 Eval episodes)
*This comparison graph shows the clear performance and completeness advantage of the GRPO-trained model over the baseline.*
![Baseline vs Trained Butler](./assets/baseline_vs_trained.png)

## 4. Why Does It Matter?
*   **For Users (Executives, Busy Professionals):** Butler automates away the mundane (scheduling, email replies, meeting follow-ups) while actively enforcing healthy boundaries. It ensures you don't miss your gym session or a family dinner just because a work email arrived.
*   **For AI Researchers & OpenEnv Hackathon Judges:** Butler demonstrates how to build a complex, multi-agent OpenEnv environment utilizing GRPO for priority-based reinforcement learning. It proves that RL can train models to prioritize abstract values (wellbeing) rather than just indiscriminate task completion.
*   **For Product & Engineering:** The project provides a blueprint for integrating external APIs (Google Calendar, Gmail), a centralized LLM routing system (Hugging Face with Cursor fallback), and local knowledge bases into a cohesive, scalable application.

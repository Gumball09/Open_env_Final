# Building Butler: From Task Management to Value-Aligned Agentic Architecture

The intersection of reinforcement learning (RL) and Large Language Models (LLMs) has opened new frontiers in building autonomous systems. While most LLM agents focus on code generation or data retrieval, we built **"Butler"** to tackle a more deeply human problem: the cognitive drain of managing conflicting personal and professional priorities.

Built atop the OpenEnv framework, Butler is not just a to-do list—it is a multi-agent orchestration environment trained to aggressively prioritize the user's wellbeing over their professional deliverables.

---

## 1. The Core Problem Statement: Priority Inversion

In modern executive environments, highly driven professionals frequently suffer from **Priority Inversion**. Because professional tasks (like scheduling meetings, replying to clients, hitting deadlines) carry immediate, visible social pressure, they easily bypass personal tasks (like going to the gym, drinking water, or attending family events) which often have delayed, private consequences.

We face these micro-conflicts constantly: missing a dinner due to last-minute work, or navigating the nuance of replying to tough emails while ignoring a hydration reminder.

The challenge in AI research is: **How do we build an autonomous agent that doesn't just blindly execute tasks, but actually understands and enforces human value structures?** We needed a realistic simulation of handling personal tasks and conflicts, managing them as intelligent delegations.

---

## 2. The Environment: An MDP Formulation of Modern Life

To train an agent using RL, we framed the user's daily life as a **Markov Decision Process (MDP)** within a scalable OpenEnv `MCPEnvironment`.

- **State Space ($S$):** The agent observes a dynamic queue of pending ToDos (1 to 5 at a time) and a rich semantic user context injected from a local JSON Knowledge Base (e.g., timezone, communication style, existing commitments).
- **Action Space ($A$):** The agent can take 7 discrete parameterized actions ranging from `route_to_agent`, `ask_clarification`, to tool-specific executions like `schedule_event` and `draft_reply`.
- **Transition Dynamics ($T$):** Successfully completing a task pops it from the queue and updates the environmental state. Failing an API call or lacking parameters leaves the task pending.

By framing personal management as an MDP, we provide the agent with a sandbox to simulate the consequences of handling (or mis-handling) conflicting priorities.

---

## 3. The Agentic Architecture: Specialized Sub-Agents

Butler operates as a centralized CMS monitored continuously by a routing Orchestrator and specialized sub-agents. We mapped specific conceptual clusters to tools to ground the LLM's outputs:

*   **Meeting Agent:** Activated by keywords like "meeting" or "standup". It extracts parameters (email, time, duration), schedules the meeting using the Google Calendar API, and sends an automated template via the Gmail API to remind attendees. Crucially, when a meeting happens, it reviews the summary for action items to recursively queue future tasks.
*   **Email Agent:** Handles deep Gmail integration. It prioritizes important emails in the CMS. We also implemented an **Auto-Pilot daemon**—a background process that scans unread mail, queries the Knowledge Base for context, and autonomously reasons to draft and send contextualized AI replies without manual intervention.
*   **Knowledge Base Agent:** Solves the "memory" problem in LLM agents. As Butler schedules meetings and learns preferences, context is saved locally. Users can trigger a Q&A session directly with Butler, allowing the agent to perform Retrieval-Augmented Generation (RAG) over the user's life data.
*   **Habit Agent:** Triggered by "remind" or "health". It bypasses the traditional calendar and interfaces directly with a Reminder Tool to set up daily recurring alarms for going to the gym, drinking water, or focused work blocks.

---

## 4. The Reward Rubric: Aligning Values Mathematically

Training an LLM to "care about health" requires mathematically rigorous reward shaping. We designed a deterministic, 5-component composable reward rubric:

1. **Priority Ordering (25%):** Did the agent handle Tier 1 (Personal) tasks before Tier 2 (Professional) tasks?
2. **Correct Routing (20%):** Did the orchestrator select the right agent based on the semantic intent?
3. **Action Completeness (20%):** Were all required API parameters (e.g., time, email, subject) synthesized correctly?
4. **API Call Success (20%):** Did the external API (Google Calendar/Gmail) accept the payload?
5. **No Over-Triggering (15%):** Did the agent correctly abstain from non-actionable tasks (e.g., "buy groceries")?

### The `-0.3` Priority Penalty
The core innovation of Butler is its strict tier system.
*   **🟢 TIER 1 (Personal):** Priority Score 10 (Health, family, habits)
*   **🔵 TIER 2 (Professional):** Priority Score 5 (Meetings, emails, deadlines)

If the agent routes or acts upon a TIER 2 task while *any* TIER 1 task remains pending in the queue, a massive **`-0.3` reward penalty** is applied on top of the rubric. This creates a steep gradient that forces the model to learn that personal wellbeing is a non-negotiable prerequisite to professional work.

---

## 5. Training with GRPO

We fine-tuned Hugging Face's `Qwen2.5-7B-Instruct` model (quantized via Unsloth) using **Group Relative Policy Optimization (GRPO)** via the `trl` library.

Unlike standard PPO, GRPO eliminates the need for a separate value model by normalizing the rewards of a group of sampled outputs against each other. This dramatically reduces memory overhead, allowing us to train a complex, multi-tool reasoning agent locally.

### Results: What Changed After the Training?

Before training, the baseline `Qwen2.5` model treated the environment like a standard chat interface: it hallucinated parameters, triggered tools on un-actionable text, and processed the queue in a naive FIFO (First-In, First-Out) manner, entirely ignoring the priority structure.

After GRPO training, the behavioral shift was profound:
*   ✅ **Value Alignment:** The agent learned to *always* handle personal tasks (Tier 1) before professional tasks (Tier 2), internalizing the `-0.3` penalty.
*   ✅ **Precision Routing:** It mapped tasks to the correct sub-agent with near-perfect accuracy.
*   ✅ **Parameter Synthesis:** It learned to extract and format variables specifically for the Gmail and Google Calendar APIs, asking for clarification only when data was truly missing.

### Empirical Data

*(Visual representations of our training metrics)*

#### Project Reward vs Step (500 Steps)
Over 500 steps, we observe the model escaping local optima (where it simply tried to do the easiest task first) and converging on a policy that maximizes the 5-component rubric.
![Project reward vs Step (500 Steps)](./assets/reward_500.png)

#### Project Reward Vs Step (50 Steps)
In the first 50 steps, the model experiences rapid policy adaptation as it hits the `-0.3` priority penalty repeatedly, causing a sharp initial correction in behavior.
![Project reward Vs Step (50 Steps)](./assets/reward_50.png)

#### Baseline vs Trained Butler (50 Eval episodes)
This evaluation clearly demonstrates the trained model successfully completing full MDP trajectories (clearing the queue) whereas the baseline consistently fails due to tool hallucinations and priority violations.
![Baseline vs Trained Butler](./assets/baseline_vs_trained.png)

---

## 6. Why Does It Matter?

The implications of Butler extend far beyond a hackathon project:

1. **For Agentic AI Research:** Butler demonstrates how GRPO can be applied to complex OpenEnv environments to train models that prioritize *abstract values* (wellbeing) over indiscriminate task completion. It proves that we can shape an LLM's decision-making framework mathematically.
2. **For Software Architecture:** The project provides a scalable blueprint for building centralized LLM routing systems (leveraging Hugging Face with Cursor fallbacks) that interface safely with real-world APIs (Google Calendar, Gmail) and local memory stores.
3. **For the End User:** Butler represents a shift from "Assistants" to "Orchestrators." It automates the mundane while actively enforcing healthy boundaries, ensuring that highly driven individuals don't miss their life in the pursuit of their work.

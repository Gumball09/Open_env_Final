# Butler Setup, Automations & API Integration

This document outlines the architecture of Butler, including the intelligent centralized LLM setup, the fully autonomous Auto-Pilot daemon, and a step-by-step guide on configuring all required 3rd-party applications.

---

## 🤖 The Auto-Pilot Daemon (`agents/auto_reply_agent.py`)

In addition to standard request-driven agents (where you ask Butler to do something), Butler now features a **fully automated background agent** called the `AutoReplyAgent`.

This agent runs completely autonomously without manual prompting:
1. **Inbox Scanning:** It automatically hooks into your Gmail to fetch unread emails.
2. **Knowledge Querying:** It cross-references the email subject and snippet with your personal Knowledge Base.
3. **Autonomous Reasoning:** It uses the LLM Client to draft a fully contextualized response based *only* on what it knows from your KB.
4. **Auto-Sending:** It sends the reply directly and logs the action for you to review.

You can trigger this automation cycle via the **🤖 Auto-Pilot** tab in the Gradio UI.

---

## 🧠 Centralized LLM Client (`tools/llm_client.py`)

Butler uses a unified LLM Client that routes to the best available provider, preventing you from needing to manage API keys inside individual agent files.

### Supported Providers
1. **HuggingFace Inference API (Primary)**
   - Used if `HF_TOKEN` is found in your `.env` file.
   - Defaults to `Qwen/Qwen2.5-7B-Instruct`. Uses `chat_completion` for reliable formatting.

2. **Cursor API (Fallback)**
   - Used if `CURSOR_API_KEY` is found (and HF is unavailable or fails).
   - Defaults to `gpt-4o-mini`.

3. **Template Fallback**
   - Safely degrades to local string templates if no APIs are available.

---

## 🛠️ Step-by-Step: 3rd Party Apps Setup

To make Butler fully functional (Automated emails, LLM reasoning, Calendar scheduling, and RL training), you must set up the following 3rd-party applications.

### 1. Google Cloud Console (For Gmail & Calendar)
This allows Butler to read emails, send automated replies, and schedule events.
1. Go to the [Google Cloud Console](https://console.cloud.google.com/).
2. Create a new Project (e.g., "Butler AI").
3. Navigate to **APIs & Services > Library**.
4. Search for and **Enable** the following two APIs:
   - `Gmail API`
   - `Google Calendar API`
5. Navigate to **APIs & Services > OAuth consent screen**. Choose "External" (or Internal if using Google Workspace) and fill in the required fields (App name, support email). Add your email as a "Test User".
6. Navigate to **APIs & Services > Credentials**.
7. Click **Create Credentials > OAuth client ID**. 
8. Choose **Desktop app** (or Web application if you plan to host this remotely with a specific redirect URI).
9. Download the JSON file and rename it to `credentials.json`. 
10. Place `credentials.json` directly in the root of the `butler-openenv` directory.

### 2. Hugging Face (For Primary LLM Inference)
This provides the brain for Butler to draft emails and parse Knowledge Base context.
1. Create an account at [Hugging Face](https://huggingface.co/).
2. Go to your **Settings > Access Tokens** (https://huggingface.co/settings/tokens).
3. Create a new token with **Read** permissions.
4. Copy the token (it starts with `hf_`).

### 3. Cursor (For Fallback LLM Inference - Optional)
If you prefer Cursor's fast models or want a fallback for HuggingFace.
1. Log in to your [Cursor Dashboard](https://cursor.com/).
2. Navigate to your API keys section.
3. Generate a new API key and copy it.

### 4. Weights & Biases (For RL Training - Optional)
Required only if you intend to run the GRPO Reinforcement Learning training notebook.
1. Create an account at [Weights & Biases](https://wandb.ai/).
2. Navigate to your **Settings > API keys** (https://wandb.ai/settings).
3. Copy your API key.

---

## ⚙️ Finalizing Your `.env` Configuration

Once you have your keys, link them to the application using the `.env` file.

1. Copy the example configuration file:
   ```bash
   cp .env.example .env
   ```

2. Open `.env` and paste in your keys:
   ```env
   # Hugging Face API
   HF_TOKEN=hf_your_token_here
   
   # Cursor API
   CURSOR_API_KEY=cursor_your_key_here

   # Google Client Credentials (Found in your downloaded credentials.json)
   GOOGLE_CLIENT_ID=your-client-id.apps.googleusercontent.com
   GOOGLE_CLIENT_SECRET=your-client-secret
   GOOGLE_REDIRECT_URI=http://localhost:7860/oauth/callback
   ```

*(Note: Your `.env` file and `credentials.json` should **never** be committed to version control. The application loads these securely into memory on startup).*

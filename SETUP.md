# ⚙️ Butler — Complete Setup Guide

This guide will walk you through setting up all the necessary external applications, configuring your environment variables, and running Butler on your local machine.

## Prerequisites
- Python 3.11 or higher installed on your system.
- Git installed.

---

## Step 1: Prepare the Local Environment
Open your terminal and navigate to your local `butler-openenv` directory:
```bash
cd path/to/butler-openenv
```

Since modern Linux distributions (like Ubuntu/Debian) enforce externally-managed Python environments, you must create a virtual environment first:

```bash
# Create a virtual environment named 'venv'
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate
```

*(Note: You must run `source venv/bin/activate` every time you open a new terminal to work on this project).*

Now, install the required Python packages:
```bash
pip install -r requirements.txt
```

---

## Step 2: Set Up External Applications

Butler relies on several external APIs to function fully. You will need to create accounts and generate API keys for the following services.

### A. Google Cloud Console (For Gmail & Calendar)
This allows the Auto-Pilot and standard agents to read emails, send automated replies, and schedule events.

1. Go to the [Google Cloud Console](https://console.cloud.google.com/).
2. Create a new Project (e.g., "Butler AI").
3. Navigate to **APIs & Services > Library**.
4. Search for and **Enable** the following two APIs:
   - `Gmail API`
   - `Google Calendar API`
5. Navigate to **APIs & Services > OAuth consent screen**.
   - Choose "External" (or Internal if you are using Google Workspace).
   - Fill in the required fields (App name, support email).
   - Add your own email address under the "Test users" section.
6. Navigate to **APIs & Services > Credentials**.
7. Click **Create Credentials > OAuth client ID**. 
8. Choose **Desktop app** as the Application type (this allows local authentication).
9. Once created, click the **Download JSON** button.
10. Rename the downloaded file to `credentials.json` and place it directly in the `butler-openenv` directory.

### B. Hugging Face (Primary LLM Engine)
This provides the AI reasoning for Butler to draft emails and parse your Knowledge Base.

1. Create a free account at [Hugging Face](https://huggingface.co/).
2. Go to your **Access Tokens page**: https://huggingface.co/settings/tokens
3. Create a new token with **Read** permissions.
4. Copy the token (it will start with `hf_`).

### C. Cursor API (Fallback LLM Engine - Optional)
If you prefer Cursor's fast models (like `gpt-4o-mini`) or want a fallback in case Hugging Face is unavailable.

1. Log in to your [Cursor Dashboard](https://cursor.com/).
2. Navigate to your API keys section.
3. Generate a new API key and copy it.

### D. Weights & Biases (For RL Training - Optional)
You only need this if you intend to run the GRPO Reinforcement Learning notebook in Google Colab.

1. Create an account at [Weights & Biases](https://wandb.ai/).
2. Navigate to your **Settings > API keys**: https://wandb.ai/settings
3. Copy your API key.

---

## Step 3: Configure the `.env` File

Butler uses a `.env` file to securely load your API keys into memory when the application starts.

1. In your terminal, duplicate the example configuration file:
   ```bash
   cp .env.example .env
   ```

2. Open the `.env` file in your code editor and fill in the keys you generated in Step 2:
   ```env
   # ─── Hugging Face API ─────────────────────────────────────
   # Your primary LLM provider
   HF_TOKEN=hf_your_token_here
   HF_INFERENCE_MODEL=Qwen/Qwen2.5-7B-Instruct
   
   # ─── Cursor API ───────────────────────────────────────────
   # Optional fallback LLM provider
   CURSOR_API_KEY=cursor_your_key_here
   CURSOR_MODEL=gpt-4o-mini
   
   # ─── Google OAuth 2.0 ────────────────────────────────────
   # Found in the credentials.json you downloaded from Google
   GOOGLE_CLIENT_ID=your-client-id.apps.googleusercontent.com
   GOOGLE_CLIENT_SECRET=your-client-secret
   GOOGLE_REDIRECT_URI=http://localhost:7860/oauth/callback
   
   # ─── WandB (Training only) ───────────────────────────────
   # Optional: For RL training logging
   WANDB_API_KEY=your-wandb-key
   
   # ─── App Config ──────────────────────────────────────────
   BUTLER_PORT=7860
   BUTLER_KB_PATH=butler_kb.json
   BUTLER_TIMEZONE=Asia/Kolkata
   ```

**Security Note:** Never commit your `.env` file or `credentials.json` to a public repository (they are usually ignored by `.gitignore`).

---

## Step 4: Run the Application

Now that your dependencies are installed and your API keys are configured, you can start Butler!

### Running the Web Interface (Gradio)
To start the user-friendly interface:
```bash
python app.py
```
Open your browser and navigate to the URL provided in the terminal (usually `http://127.0.0.1:7860`). 

*Note: The first time the Auto-Pilot or Calendar/Gmail agents try to act on your behalf, a browser window will pop up asking you to authenticate with your Google Account.*

### Running Headless Inference
If you want to test Butler from the command line:

**Single Task:**
```bash
python inference.py --model your-username/butler-grpo --todo "Remind me to take my vitamins every morning"
```

**Testing Priority Queues:**
```bash
python inference.py --model your-username/butler-grpo --queue "Schedule meeting with Priya; Set daily gym reminder"
```

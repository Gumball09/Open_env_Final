#!/usr/bin/env python3
"""
Local GRPO Training Script for Butler
Run this on a machine with a dedicated GPU.
"""
import os
import sys
import wandb
from dotenv import load_dotenv
from huggingface_hub import login

# Load environment variables from .env
load_dotenv()

HF_TOKEN = os.getenv('HF_TOKEN')
CURSOR_API_KEY = os.getenv('CURSOR_API_KEY')
WANDB_API_KEY = os.getenv('WANDB_API_KEY')
OUTPUT_MODEL = os.getenv('HF_INFERENCE_MODEL', 'your-hf-username/butler-grpo')

BASE_MODEL = 'unsloth/Qwen2.5-7B-Instruct'
MAX_STEPS = 500
BATCH_SIZE = 4
LR = 2e-5
SAVE_STEPS = 100
EVAL_STEPS = 50

if HF_TOKEN:
    login(token=HF_TOKEN)
if WANDB_API_KEY:
    wandb.init(project='butler-openenv', name='butler-grpo-local-run')
    print('✅ WandB initialized.')

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from env.butler_env import ButlerEnvironment
    from env.observation import build_observation_prompt, SYSTEM_PROMPT
    from env.action_space import parse_llm_output, validate_action
    from agents.orchestrator import Orchestrator
except ModuleNotFoundError:
    print('❌ ERROR: Could not find Butler environment files.')
    sys.exit(1)

print('✅ Environment loaded.')
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=2048,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "v_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

print(f"✅ Model loaded: {BASE_MODEL}")
print(f"   Trainable params: {model.print_trainable_parameters()}")


from env.observation import build_observation_prompt, SYSTEM_PROMPT
from agents.orchestrator import Orchestrator

orch = Orchestrator()


def format_episode(example):
    """
    Convert an episode queue (list of todos) into a prompt for GRPO.
    """
    episode_queue = example["episode"]
    sorted_queue = orch.sort_queue(episode_queue)
    
    obs = {
        "queue": sorted_queue,
        "current_todo": sorted_queue[0] if sorted_queue else None,
        "user_context": {
            "name": "User",
            "timezone": "Asia/Kolkata",
            "communication_style": "professional",
            "role": "Professional",
        },
        "step": 0,
        "max_steps": 10,
    }
    
    prompt = (
        f"<|system|>\n{SYSTEM_PROMPT}\n"
        f"<|user|>\n{build_observation_prompt(obs)}\n"
        f"<|assistant|>\n"
    )
    
    return {"prompt": prompt}


train_dataset = train_dataset.map(
    format_episode,
    remove_columns=train_dataset.column_names,
)

eval_dataset = eval_dataset.map(
    format_episode,
    remove_columns=eval_dataset.column_names,
)

print(f"✅ Dataset formatted.")
print(f"   Train prompts: {len(train_dataset)}")
print(f"   Eval prompts:  {len(eval_dataset)}")
print(f"\n   Sample prompt (first 500 chars):")
print(train_dataset[0]['prompt'][:500])


from trl import GRPOConfig, GRPOTrainer

training_args = GRPOConfig(
    output_dir="./butler-grpo-output",
    num_train_epochs=3,
    per_device_train_batch_size=BATCH_SIZE,
    learning_rate=LR,
    max_steps=MAX_STEPS,
    logging_steps=10,
    save_steps=SAVE_STEPS,
    eval_steps=EVAL_STEPS,
    report_to="wandb",
    run_name="butler-grpo-run-1",
    max_completion_length=256,
)

trainer = GRPOTrainer(
    model=model,
    args=training_args,
    reward_funcs=butler_reward_fn,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

print("🚀 Starting GRPO training...")
trainer.train()
print("✅ Training complete!")


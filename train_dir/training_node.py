import time
import torch.multiprocessing as mp
import os
# Set device visibility *before* importing torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import time
import sys
time.sleep(0.01)
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoConfig
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
import statistics
import random
import sys

from torch.optim.lr_scheduler import LambdaLR

from stockfish import Stockfish
# Assuming these are in the python path or relative
from envs.chess_env import BoardEnv
from reward import (reward as get_reward_from_fen, FIXED_DEPTH)
'''
def get_reward_from_fen(x, y):
    return random.random()
'''

import bitsandbytes as bnb
RESPONSES_PER_BATCH = 8  # The 'k' in "sample k responses". How many responses per FEN.
NUM_GRAD_ACCUMULATION_EXAMPLES = 4  # How many FENs to process before one optimizer step.
STOCKFISH_PATH = "/scratch/ChessGPT/stockfish/stockfish-engine"

torch.set_default_dtype(torch.bfloat16)
quantization_config = BitsAndBytesConfig(load_in_8bit=True)



def clear_vram() -> None:
    """Clear the VRAM memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def linear_schedule(step):
    """Implements a linear warmup for the learning rate."""
    step += 1
    warmup_steps = 3
    if step < warmup_steps:
        return step / warmup_steps
    return 1.0



def train(model_path, reasoning_trace_queue, stop_inference_queue, GPU_IDX):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # --- 1. Device and Model Initialization ---
    try:
        # Set the active GPU for this process
        torch.cuda.set_device(GPU_IDX)
        device = f"cuda:{GPU_IDX}"
        print(f"[Trainer] Running on GPU: {device}")
    except Exception as e:
        print(f"[Trainer] ERROR: Failed to set GPU {GPU_IDX}. {e}")
        return

    tokenizer.pad_token = tokenizer.eos_token
    
    # Load a model
    print(f"[Trainer] Loading initial base model from: {model_path}")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map=device 
        )
    except Exception as e:
        print(f"[Trainer] ERROR: Failed to load model {model_path}. {e}")
        return
        
    # Enable memory-saving features
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # --- 2. LoRA (PEFT) Setup ---
    lora_config = LoraConfig(
        r=256,
        lora_alpha=512,
        lora_dropout=0.00,
        bias="none",
        task_type="CAUSAL_LM"
    )
    adapter_name = "grpo_adapter"
    model.add_adapter(adapter_config=lora_config, adapter_name=adapter_name)
    model.set_adapter(adapter_name)
    model.train()  # Set only the adapter weights to trainable

    # --- 3. Optimizer and Scheduler Setup ---
    beta = 0.01  # KL penalty coefficient
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5, maximize=False)
    optimizer.zero_grad()  # Clear any old gradients
    scheduler = LambdaLR(optimizer, lr_lambda=linear_schedule)

    print(f"[Trainer] Initialization complete. Waiting for data...")
    
    LOGGING_COUNTER_ONLY = 0 # Counts FENs for gradient accumulation
    epoch = 0 # Counts save/update steps

    # --- 5. Main Training Loop (Continuous) ---
    while True:
        # Wait for and get data from the generator node
        data = reasoning_trace_queue.get()
        chat_logs = data["model_responses"]
        board_state = data["board_state"]
        
        if data is None:
            print("[Trainer] Received None. Shutting down training.")
            break
            
        start_time = time.time()  # For logging time

        # Tokenize the prompt to get its length
        prompt_tokenized = tokenizer.apply_chat_template(
            chat_logs[0][:2], # Just the system and user prompt dicts, invariant across batch
            tokenize=True,
            add_generation_prompt=True,
            return_dict = True,
            return_tensors="pt"
        ).to(device)
        input_length = prompt_tokenized["input_ids"].shape[1]
        clear_vram()

        # 5.2. --- Reward Calculation & Standardization ---
        E_reward = 0
        raw_rewards = []
        rewards = []  # This will store normalized advantages

        for i, chat_pair in enumerate(chat_logs):
            model_response = chat_pair[2]["content"]

            reward_value = get_reward_from_fen(board_state, model_response)

            E_reward += reward_value
            raw_rewards.append(reward_value)
        print(raw_rewards)
        exit()
        reward_std = statistics.stdev(raw_rewards)
        E_reward = E_reward / RESPONSES_PER_BATCH
        print(f"[FEN {LOGGING_COUNTER_ONLY+1}] Expected Reward: {E_reward:.4f}, FEN: {board_state}")
        print(chat_logs)
        if reward_std == 0:
            print("Zero advantage, skipping batch.")
            continue  # Skip to the next item in the queue
        else:
            for element in raw_rewards:
                rewards.append((element - E_reward) / reward_std)
        
        for i in range(len(rewards)):
            full_text_tokenized = tokenizer.apply_chat_template(
                chat_logs[i], tokenize = True,
                return_tensors="pt",
                return_dict = True,
                add_generation_prompt = False
            ).to("cuda")
            full_text = full_text_tokenized["input_ids"]
            full_text_mask = full_text_tokenized["attention_mask"]
          
            clear_vram()
            length = full_text.shape[1]
            reward = rewards[i]
            
            model.eval()
            generation_slice = full_text[:, :length]
            clear_vram()
            model.disable_adapters()
            with torch.no_grad():
                base_model_output = model.forward(generation_slice, torch.ones(generation_slice.shape), use_cache=False)
                # Offset the softmax by 1 because we want to predict tokens for input_length:thinking_index
                base_log_probs = torch.nn.functional.log_softmax(base_model_output.logits[:, input_length-1:-1, :].to(torch.float32), dim = -1)
                base_model_probs = torch.exp(base_log_probs) 
            model.enable_adapters()
            model.train()
            policy_model_output = model.forward(generation_slice, torch.ones(generation_slice.shape), use_cache=False)
                # Offset the softmax by 1 because we want to predict tokens for input_length:thinking_index
            policy_log_probs = torch.nn.functional.log_softmax(policy_model_output.logits[:, input_length-1:-1, :].to(torch.float32), dim = -1)
            policy_model_probs = torch.exp(policy_log_probs) 
            response_slice = generation_slice[:, input_length:]
            selected_policy_probs = policy_model_probs[0, torch.arange(policy_model_probs.shape[1]), response_slice[0]].unsqueeze(0)
            selected_base_probs = base_model_probs[0, torch.arange(base_model_probs.shape[1]), response_slice[0]].unsqueeze(0)
            policy_ratio = selected_policy_probs/ selected_base_probs
            eps = 0.01
            clipped_policy_ratio = torch.clip(policy_ratio, min = 1-eps, max = 1+ eps)
            unclipped_policy_ratio = policy_ratio
            kl_divergence = torch.sum(torch.maximum(torch.log(selected_policy_probs) - torch.log(selected_base_probs), torch.zeros_like(selected_base_probs)))
            with torch.no_grad():
                selected_normalization_probs = selected_policy_probs.clone().detach()
            #on_policy_policy_ratio = selected_policy_probs / (selected_normalization_probs + 1e-9)
            base_loss = torch.prod(clipped_policy_ratio) * reward # GRPO ta
            
            loss = base_loss - beta * kl_divergence
            clear_vram()
            torch.cuda.synchronize()
            loss.backward()
            print(torch.prod(clipped_policy_ratio))
        optimizer.step()
        sys.stdout.flush()

        

if __name__ == "__main__":
    
    from torch.multiprocessing import Queue, Process
    
    model_file = "Qwen/Qwen3-4B-Thinking-2507"
    
    reasoning_trace_queue = Queue()
    stop_inference_queue = Queue()
    reasoning_trace_queue.put({"model_responses": [[{'role': 'system', 'content': 'Think in detail, and explain your reasoning. Box your answer in \\boxed{}'}, 
                                {'role': 'user', 'content': 'Given this FEN position, what is the best move? r4rk1/pp1n2q1/2pN3p/6p1/3P1b2/3B1N1b/PPP1R2P/4K2R b - - 1 24'}, 
                                {'role': 'assistant', 'content': "\\boxed{h5}"}], 
                              
                              [{'role': 'system', 'content': 'Think in detail, and explain your reasoning. Box your answer in \\boxed{}'}, 
                               {'role': 'user', 'content': 'Given this FEN position, what is the best move? r4rk1/pp1n2q1/2pN3p/6p1/3P1b2/3B1N1b/PPP1R2P/4K2R b - - 1 24'}, 
                               {'role': 'assistant', 'content': "\\boxed{Bxd6}"}]
                              ], "board_state": "r4rk1/pp1n2q1/2pN3p/6p1/3P1b2/3B1N1b/PPP1R2P/4K2R b - - 1 24"}
                              )
    train(model_file, reasoning_trace_queue, stop_inference_queue, 0)
    exit()
   

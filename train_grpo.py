import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import time
time.sleep(0.01)
import torch
from transformers import Qwen2ForCausalLM, BitsAndBytesConfig, AutoConfig
from transformers import AutoTokenizer
import bitsandbytes as bnb

import pandas as pd
import sys
from peft import LoraConfig, get_peft_model, PeftModel
import statistics
from einops import repeat
import sys
import random
from vllm import LLM, SamplingParams
import json
from torch.optim.lr_scheduler import LambdaLR
import torch.multiprocessing as mp
import threading

# --- Imports from reward.py ---
import chess
import math
from stockfish import Stockfish
from envs.chess_env import BoardEnv  # <-- ASSUMES 'envs.chess_env' is available
# Import the new reward function and constants
from reward import (
    reward as get_reward_from_fen, # Rename to avoid conflict with 'reward' variable
    FIXED_DEPTH 
)

# --- Stockfish/Reward Config ---
STOCKFISH_PATH = "C:\\Chess_Engines\\stockfish\\stockfish-windows-x86-64-avx2.exe" # <-- UPDATE THIS PATH

torch.set_default_dtype(torch.bfloat16)

# --- Original Script Components (Adapted) ---

quantization_config = BitsAndBytesConfig(load_in_8bit=True)
device = "cuda"
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

# Load your chess data (list of FENs)
try:
    with open("chess_positions.txt", "r") as f:
        data = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(data)} chess positions.")
except FileNotFoundError:
    print("Error: 'chess_positions.txt' not found. Please create this file with one FEN per line.")
    print("Using dummy data for demonstration...")
    data = ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"] * 100

def linear_schedule(step):
    step += 1
    warmup_steps = 3
    if step < warmup_steps:
        return step / warmup_steps
    return 1.0

def memory_isolated_train_step(generation_outputs, iteration_list, epoch, result_queue):
    tokenizer.pad_token = tokenizer.eos_token
    if epoch != 0:
        model = Qwen2ForCausalLM.from_pretrained(f"./cshs_checkpoints/off_policy_checkpoint_{epoch-1}", quantization_config=quantization_config)
    else:
        model = Qwen2ForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", quantization_config=quantization_config)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    lora_config = LoraConfig(
        r=256,                
        lora_alpha=512,       
        lora_dropout=0.00,   
        bias="none",
        task_type="CAUSAL_LM"  
    )
    model.add_adapter(adapter_config=lora_config,adapter_name=f"adapter_{epoch}")
    model.set_adapter(f"adapter_{epoch}")
    model.train()
    
    beta = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5, maximize = True)
    scheduler = LambdaLR(optimizer, lr_lambda=linear_schedule)
    
    # --- EFFICIENT INITIALIZATION ---
    # Initialize Stockfish ONCE here
    try:
        stockfish = Stockfish(path=STOCKFISH_PATH, depth=FIXED_DEPTH)
    except Exception as e:
        print(f"Failed to initialize Stockfish at {STOCKFISH_PATH}: {e}")
        print("Exiting training process.")
        return
    
    start_time = time.time() # For logging time

    for LOGGING_COUNTER_ONLY, iteration_variable in enumerate(iteration_list):
        
        # 1. Get the data dictionary for this FEN
        data_for_this_fen = generation_outputs[iteration_variable]
        
        # 2. Extract components
        base_fen = data_for_this_fen["board_position"]
        system_prompt = data_for_this_fen["system_prompt"]
        chat_logs = data_for_this_fen["chat_logs"] # List of [user, assistant] pairs
        
        user_prompt = chat_logs[0][0] # first pair, user element
        prompt_messages = [system_prompt, user_prompt]
        
        # 3. Tokenize prompt to get input_length
        prompt_tokenized = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to("cuda")
        
        input_length = prompt_tokenized["input_ids"].shape[1]
        clear_vram()
        
        E_reward = 0
        raw_rewards = []
        rewards = [] # This will store normalized advantages
        
        # 4. --- Reward Calculation Loop (NOW USING YOUR `reward` FUNCTION) ---
        for i, chat_pair in enumerate(chat_logs):
            model_response = chat_pair[1]["content"] # Get assistant's response
            
            try:
                # Call the reward function from reward.py
                # We pass the shared stockfish instance to it
                reward_value = get_reward_from_fen(base_fen, model_response, stockfish)
                
            except Exception as e:
                # Catch any errors from parsing or reward calculation
                print(f"Error calculating reward for FEN {base_fen}, response '{model_response}': {e}. Assigning penalty.")
                reward_value = -5.0 
            
            E_reward += reward_value
            raw_rewards.append(reward_value)
        
        # 5. --- Advantage Normalization ---
        reward_std = statistics.stdev(raw_rewards) if len(raw_rewards) > 1 else 0
        E_reward = E_reward / RESPONSES_PER_BATCH
        print(f"Expected Reward: {E_reward:.4f}, FEN: {base_fen}" )

        if reward_std == 0:
            print("Zero advantage, skipping batch.")
            rewards = [0.0] * len(raw_rewards)
            continue 
        else:
            for element in raw_rewards:
                rewards.append((element - E_reward) / reward_std)
        
        # 6. --- Policy Update Loop ---
        for i in range(len(rewards)):
            chat_pair = chat_logs[i]
            advantage = rewards[i] # Renamed from 'reward' for clarity
            
            # Reconstruct the full chat: [system, user, assistant]
            full_chat_messages = [system_prompt, chat_pair[0], chat_pair[1]]
          
            full_text_tokenized = tokenizer.apply_chat_template(
                full_chat_messages,
                tokenize=True,
                add_generation_prompt=False,
                return_tensors="pt"
            ).to("cuda")

            full_text = full_text_tokenized["input_ids"]
            full_text_mask = full_text_tokenized["attention_mask"]
          
            clear_vram()
            length = full_text.shape[1]
            
            # --- PPO/GRPO Loss Calculation ---
            model.eval()
            generation_slice = full_text[:, :length]
            clear_vram()
            model.disable_adapters()
            with torch.no_grad():
                base_model_output = model.forward(generation_slice, torch.ones(generation_slice.shape), use_cache=False)
                base_log_probs = torch.nn.functional.log_softmax(base_model_output.logits[:, input_length-1:-1, :].to(torch.float32), dim = -1)
                base_model_probs = torch.exp(base_log_probs) 
            model.enable_adapters()
            model.train()
            policy_model_output = model.forward(generation_slice, torch.ones(generation_slice.shape), use_cache=False)
            policy_log_probs = torch.nn.functional.log_softmax(policy_model_output.logits[:, input_length-1:-1, :].to(torch.float32), dim = -1)
            policy_model_probs = torch.exp(policy_log_probs) 
            response_slice = generation_slice[:, input_length:]
            selected_policy_probs = policy_model_probs[0, torch.arange(policy_model_probs.shape[1]), response_slice[0]].unsqueeze(0)
            selected_base_probs = base_model_probs[0, torch.arange(base_model_probs.shape[1]), response_slice[0]].unsqueeze(0)
            policy_ratio = selected_policy_probs / (selected_base_probs + 1e-9)
            eps = 0.01
            clipped_policy_ratio = torch.clip(policy_ratio, min = 1-eps, max = 1+ eps)
            unclipped_policy_ratio = policy_ratio
            kl_divergence = torch.sum(torch.maximum(torch.log(selected_policy_probs + 1e-9) - torch.log(selected_base_probs + 1e-9), torch.zeros_like(selected_base_probs)))
            
            base_loss = torch.prod(clipped_policy_ratio) * advantage # GRPO loss
            
            loss = base_loss - beta * kl_divergence
            clear_vram()
            torch.cuda.synchronize()
            loss.backward()
            
            # --- Adapted Logging ---
            result_queue.put({"loss": loss.item(), "kl": kl_divergence.item(), 
                        "clipped_ratio": torch.prod(clipped_policy_ratio).item(), "unclipped_ratio": torch.prod(unclipped_policy_ratio).item(),
                        "reward": advantage, "expected_reward": E_reward, "fen": base_fen,
                        "length": clipped_policy_ratio.shape[1], "base_loss": base_loss.item(),
                        "full length": length, "time": time.time() - start_time, 
                        "num_examples": LOGGING_COUNTER_ONLY,
                        "current_lr": scheduler.get_last_lr()[0]})
        
        # --- Gradient Accumulation ---
        if LOGGING_COUNTER_ONLY % NUM_GRAD_ACCUMULATION_EXAMPLES == NUM_GRAD_ACCUMULATION_EXAMPLES - 1:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_.grad()
            sys.stdout.flush()
            
    # --- Model Saving ---
    save_path = "./"
    model.save_pretrained(f"{save_path}{epoch}_placeholder", weird_custom_arg = True)
    if epoch != 0:
        full_precison_model = Qwen2ForCausalLM.from_pretrained(f"./cshs_checkpoints/off_policy_checkpoint_{epoch-1}", torch_dtype = torch.bfloat16)
    else:
        full_precison_model = Qwen2ForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", torch_dtype = torch.bfloat16)
    
    merge_model=PeftModel.from_pretrained(full_precison_model, f"{save_path}{epoch}_placeholder")
    merge_model = merge_model.merge_and_unload()
    try:
        merge_model.disable_adapters()
    except Exception as e:
        print(e)
    merge_model.save_pretrained(f"./cshs_checkpoints/off_policy_checkpoint_{epoch}")
    tokenizer.save_pretrained(f"./cshs_checkpoints/off_policy_checkpoint_{epoch}")

# --- Utility Functions ---
import re

def print_gpu_memory():
    print(f"Allocated memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"Cached memory: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

import gc
def clear_vram():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
# --- Main Execution ---
MAXIMUM_GENERATION_LENGTH = 50 
NUM_GRAD_ACCUMULATION_EXAMPLES = 1
RESPONSES_PER_BATCH = 33 
start = time.time()
mp.set_start_method("spawn", force=True)

if __name__ == '__main__':
    import wandb
    wandb.init(project="GRPO-chess-off-policy") 
    
    def queue_logger(result_queue):
        while True:
            if not result_queue.empty():
                wandb.log(result_queue.get())
            else:
                time.sleep(0.01)
                
    for epoch in range(0, 100):
        
        # --- Data Generation Phase ---
        if epoch != 0:
            model_path = f"./cshs_checkpoints/off_policy_checkpoint_{epoch-1}"
            llm = LLM(model=model_path)
        else:
            model_path = "Qwen/Qwen2.5-7B-Instruct"
            try:
                print("Epoch 0: Loading pre-generated data...")
                with open(f'./chess_llm_data/generated_data_0.json', 'r') as f:
                    generation_outputs = json.load(f)
                with open(f'./chess_llm_data/first_list_0.json', 'r') as f:
                    iteration_list = json.load(f)
                print("Successfully loaded epoch 0 data.")
                
                clear_vram()
                result_queue = mp.Queue()
                logging_thread = threading.Thread(target=queue_logger, args=(result_queue,), daemon=True)
                logging_thread.start()
                
                p = mp.Process(target=memory_isolated_train_step, args=(generation_outputs, iteration_list, epoch, result_queue))
                p.start()
                p.join()  
                clear_vram()
                print(f"Epoch {epoch} complete.")
                print_gpu_memory()
                continue
            
            except FileNotFoundError:
                print("Epoch 0 data not found. Generating initial data from base model...")
                llm = LLM(model=model_path)

        # --- Data Generation Logic ---
        local_max_length = len(data)
        iteration_list = list(range(local_max_length))
        random.shuffle(iteration_list)
        # iteration_list = iteration_list[:10] # Optional: Use a subset

        with open(f'./chess_llm_data/first_list_{epoch}.json', 'w') as f:
            json.dump(iteration_list, f)

        sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=MAXIMUM_GENERATION_LENGTH)
        
        generation_list_for_vllm = []
        generation_outputs = []
        
        print(f"Epoch {epoch}: Generating {len(iteration_list) * RESPONSES_PER_BATCH} responses...")
        
        for LOGGING_COUNTER_ONLY, iteration_variable in enumerate(iteration_list):
            fen = data[iteration_variable]
            messages = [
                {"role": "system", "content": "You are a chess grandmaster AI. Provide your move in SAN."},
                {"role": "user", "content": f"FEN: {fen}\nYour move:"}
            ]
            text_for_vllm = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            for i in range(RESPONSES_PER_BATCH):
                generation_list_for_vllm.append(text_for_vllm)
        
        outputs = llm.generate(generation_list_for_vllm, sampling_params)

        output_index = 0
        for iteration_variable in iteration_list:
            fen = data[iteration_variable]
            
            system_prompt = {"role": "system", "content": "You are a chess grandmaster AI. Provide your move in SAN."}
            user_prompt = {"role": "user", "content": f"FEN: {fen}\nYour move:"}
            
            chat_logs_for_this_fen = []
            for _ in range(RESPONSES_PER_BATCH):
                output = outputs[output_index]
                generated_text = output.outputs[0].text.strip()
                assistant_response = {"role": "assistant", "content": generated_text}
                
                chat_logs_for_this_fen.append([user_prompt, assistant_response])
                output_index += 1
            
            generation_outputs.append({
                "board_position": fen,
                "system_prompt": system_prompt, 
                "chat_logs": chat_logs_for_this_fen
            })
        
        os.makedirs("./chess_llm_data", exist_ok=True)
        with open(f'./chess_llm_data/generated_data_{epoch}.json', 'w') as f:
            json.dump(generation_outputs, f)
        del llm
        
        # --- Training Phase ---
        print(f"Epoch {epoch}: Starting training process...")
        clear_vram()
        result_queue = mp.Queue()
        logging_thread = threading.Thread(target=queue_logger, args=(result_queue,), daemon=True)
        logging_thread.start()
        
        p = mp.Process(target=memory_isolated_train_step, args=(generation_outputs, iteration_list, epoch, result_queue))
        p.start()
        p.join() 
 
        clear_vram()
        print(f"Epoch {epoch} complete.")
        print_gpu_memory()
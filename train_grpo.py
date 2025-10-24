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

import chess
import math
from stockfish import Stockfish
from envs.chess_env import BoardEnv
from reward import (
    reward as get_reward_from_fen, # Rename to avoid conflict with 'reward' variable
    FIXED_DEPTH 
)

# --- Stockfish/Reward Config ---
STOCKFISH_PATH = "C:\\Chess_Engines\\stockfish\\stockfish-windows-x86-64-avx2.exe" # <-- UPDATE THIS PATH

torch.set_default_dtype(torch.bfloat16)


quantization_config = BitsAndBytesConfig(load_in_8bit=True)
device = "cuda"
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

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
    

    try:
        stockfish = Stockfish(path=STOCKFISH_PATH, depth=FIXED_DEPTH)
    except Exception as e:
        print(f"Failed to initialize Stockfish at {STOCKFISH_PATH}: {e}")
        print("Exiting training process.")
        return
    
    start_time = time.time() # For logging time

    for LOGGING_COUNTER_ONLY, iteration_variable in enumerate(iteration_list):
        #data_for_this_fen = generation_outputs[LOGGING_COUNTER_ONLY * RESPONSES_PER_BATCH: (LOGGING_COUNTER_ONLY + 1) * RESPONSES_PER_BATCH]

        
        data_for_this_fen = generation_outputs[iteration_variable]
        
        base_fen = data_for_this_fen["board_position"]
        system_prompt = data_for_this_fen["system_prompt"]
        chat_logs = data_for_this_fen["chat_logs"] # List of [user, assistant] pairs
        
        user_prompt = chat_logs[0][0] # first pair, user element
        prompt_messages = [system_prompt, user_prompt]
        
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
        
        for i, chat_pair in enumerate(chat_logs):
            model_response = chat_pair[1]["content"] 
            
            try:
                reward_value = get_reward_from_fen(base_fen, model_response, stockfish)               
            except Exception as e:
                # Catch any errors from parsing or reward calculation
                print(f"Error calculating reward for FEN {base_fen}, response '{model_response}': {e}. Assigning penalty.")
                reward_value = -5.0 
            
            E_reward += reward_value
            raw_rewards.append(reward_value)
        
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
            advantage = rewards[i] 
            
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
            optimizer.zero_grad()
            sys.stdout.flush()
            
 
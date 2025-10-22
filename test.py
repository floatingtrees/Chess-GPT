import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import time
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
import torch.multprocessing as mp
import threading

# --- NEW CHESS-SPECIFIC IMPORTS ---
import chess
import math
from stockfish import Stockfish
from envs.chess_env import BoardEnv  # Assuming this is in your envs/ folder
import io
import gc
# --- END NEW IMPORTS ---

torch.set_default_dtype(torch.bfloat16)

quantization_config = BitsAndBytesConfig(load_in_8bit=True)
device = "cuda"
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

# ==============================================================================
# === START: CHESS REWARD LOGIC ===
# ==============================================================================

# --- Constants ---
FIXED_DEPTH = 15
WIN_PERCENT_NOISE_THRESHOLD = 3.0
SMALL_CHANGE_BONUS = 0.05
GAME_OVER_PENALTY = -1.5

# --- Stockfish Path ---
# This path MUST be correct for the script to work
stockfish_path = "C:\\Chess_Engines\\stockfish\\stockfish-windows-x86-64-avx2.exe"

# --- Helper Functions ---
def get_win_percentage(centipawns: int) -> float:
    """Converts centipawn evaluation to a win percentage (from 0 to 100)."""
    clipped_cp = max(-1500, min(1500, centipawns))
    return 50 + 50 * (2 / (1 + math.exp(-0.00368208 * clipped_cp)) - 1)

def _get_cp_from_eval(evaluation: dict) -> int:
    """Helper function to handle both 'cp' and 'mate' evaluations."""
    if evaluation['type'] == 'cp':
        return evaluation['value']
    elif evaluation['type'] == 'mate':
        # Use a very large number for mate
        return 30000 if evaluation['value'] > 0 else -30000
    return 0

# --- Main Reward Calculation Logic ---
def calculate_move_reward(stockfish: Stockfish, fen: str, move: chess.Move) -> float:
    """
    Calculates a scaled reward (-1.0 to 1.0) based on the change in win percentage for a move.
    """
    try:
        board = chess.Board(fen=fen)

        if board.is_game_over():
            return GAME_OVER_PENALTY

        # We receive a move object, so we just need to check legality
        if move not in board.legal_moves:
            print(f"Illegal move received: {move.uci()}")
            return -3.0  # Large penalty for illegal moves

        # --- Get Evaluation BEFORE the move ---
        stockfish.set_fen_position(fen)
        eval_before = stockfish.get_evaluation()
        cp_before = _get_cp_from_eval(eval_before)

        # --- Make the Move and Check Post-Move Terminal States ---
        board.push(move)
        if board.is_checkmate():
            return 1.0  # Max reward for delivering checkmate
        if board.is_stalemate() or board.is_insufficient_material():
            return 0.0  # Neutral reward for a draw

        # --- Get Evaluation AFTER the move ---
        stockfish.set_fen_position(board.fen())
        eval_after = stockfish.get_evaluation()
        cp_after = _get_cp_from_eval(eval_after)
        
        # --- Calculate Win Percentage Change ---
        if board.turn == chess.BLACK: # White just moved
            win_percent_before = get_win_percentage(cp_before)
            win_percent_after = get_win_percentage(cp_after)
        else: # Black just moved
            win_percent_before = get_win_percentage(-cp_before)
            win_percent_after = get_win_percentage(-cp_after)

        change = win_percent_after - win_percent_before
        reward = 0.0

        # --- Noise threshold ---
        if abs(change) < WIN_PERCENT_NOISE_THRESHOLD:
            reward = SMALL_CHANGE_BONUS 
        else:
            reward = change / 100.0 # Scale reward

        return reward
    
    except Exception as e:
        print(f"An unexpected error occurred in calculate_move_reward: {e}")
        return -5.0

# ==============================================================================
# === END: CHESS REWARD LOGIC ===
# ==============================================================================


def linear_schedule(step):
    step += 1
    warmup_steps = 3
    if step < warmup_steps:
        return step / warmup_steps
    return 1.0

# ==============================================================================
# === MAIN TRAINING FUNCTION (UPDATED) ===
# ==============================================================================
def memory_isolated_train_step(generation_outputs, iteration_list, epoch, result_queue):
    tokenizer.pad_token = tokenizer.eos_token
    if epoch != 0:
        model = Qwen2ForCausalLM.from_pretrained(f"/mnt/t9/cshs_checkpoints/off_policy_checkpoint_{epoch-1}", quantization_config=quantization_config)
    else:
        model = Qwen2ForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", quantization_config=quantization_config)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    lora_config = LoraConfig(
        r=256, lora_alpha=512, lora_dropout=0.00, bias="none", task_type="CAUSAL_LM"
    )
    model.add_adapter(adapter_config=lora_config,adapter_name=f"adapter_{epoch}")
    model.set_adapter(f"adapter_{epoch}")
    model.train()
    
    beta = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5, maximize = True)
    scheduler = LambdaLR(optimizer, lr_lambda=linear_schedule)
    
    # --- OPTIMIZATION: Initialize Stockfish ONCE per process ---
    try:
        stockfish = Stockfish(path=stockfish_path, depth=FIXED_DEPTH)
    except Exception as e:
        print(f"!!!!!!!!!!!!!! FAILED TO INITIALIZE STOCKFISH !!!!!!!!!!!!!!")
        print(f"Error: {e}")
        print("Exiting training process.")
        return # Exit the function if stockfish can't load

    # --- Main Training Loop (Iterating over prompts) ---
    for LOGGING_COUNTER_ONLY, iteration_variable in enumerate(iteration_list):
        
        relevant_chunks = generation_outputs[LOGGING_COUNTER_ONLY * RESPONSES_PER_BATCH: (LOGGING_COUNTER_ONLY + 1) * RESPONSES_PER_BATCH]
        
        # --- NEW: Access data using your dictionary keys ---
        # Get prompt text from the first item (it's the same for all 33)
        input_prompt_text = relevant_chunks[0]["chat_logs"][0][0]["content"] 
        input_prompt_tokenized = tokenizer(input_prompt_text, return_tensors="pt").to("cuda")
        input_length = input_prompt_tokenized["input_ids"].shape[1]
        
        clear_vram()
        E_reward = 0
        raw_rewards = []
        rewards = []
        num_legal_moves = 0
        
        # --- START: CHESS REWARD LOOP ---
        # This loop calculates the reward for each of the 33 generated moves
        for i, element in enumerate(relevant_chunks):
            # `element` is now your dictionary: {“chat_logs”: ..., “board_position”: ...}
            
            # Get the model's raw text response
            model_response_text = element["chat_logs"][0][1]["content"]
            # Get the FEN string
            board_fen = element["board_position"]

            try:
                # Use BoardEnv to parse the move from the raw text
                board_env = BoardEnv(fen=board_fen)
                parsed_move, _ = board_env.parse_move(model_response_text)
                
                if parsed_move is None:
                    raise ValueError(f"No move found in response: {model_response_text}")

                # Call the core reward logic with the pre-initialized stockfish
                reward = calculate_move_reward(stockfish, board_fen, parsed_move)
                if reward > -3.0: # -3.0 is the illegal move penalty
                    num_legal_moves += 1

            except Exception as e:
                print(f"Error parsing move or getting reward: {e}")
                reward = -5.0 # Penalty for parsing error / wrong notation
            
            E_reward += reward
            raw_rewards.append(reward)

        # --- Reward Normalization (This is the "GR" in GRPO) ---
        reward_std = 0.0
        if len(raw_rewards) > 1:
            try:
                reward_std = statistics.stdev(raw_rewards)
            except statistics.StatisticsError:
                reward_std = 0.0 # All rewards were identical

        E_reward = E_reward / RESPONSES_PER_BATCH
        print(f"Expected Raw Reward: {E_reward:.4f}, Legal Moves: {num_legal_moves}/{RESPONSES_PER_BATCH}")
        
        if reward_std == 0:
            # If all rewards are the same, advantage is 0
            rewards = [0.0] * len(raw_rewards)
        else:
            # Normalize rewards -> this is the "Advantage"
            for element in raw_rewards:
                rewards.append((element - E_reward) / (reward_std + 1e-9)) # Added epsilon
        # --- END: CHESS REWARD LOOP ---


        # --- GRPO Update Loop (Updating the policy) ---
        # This loop iterates over each response and its calculated advantage
        for i in range(len(rewards)):
            # This is the normalized advantage (reward)
            reward = rewards[i] 
            
            # --- NEW: Accessing data using your dictionary keys ---
            prompt_text = relevant_chunks[i]["chat_logs"][0][0]["content"]
            response_text = relevant_chunks[i]["chat_logs"][0][1]["content"]

            full_text_tokenized = tokenizer(prompt_text + response_text, return_tensors="pt").to("cuda")
            full_text = full_text_tokenized["input_ids"]
            
            clear_vram()
            length = full_text.shape[1]
            generation_slice = full_text[:, :length]
            
            # --- Get Base Model (Reference) Log-Probs ---
            model.eval()
            model.disable_adapters()
            with torch.no_grad():
                base_model_output = model.forward(generation_slice, torch.ones(generation_slice.shape), use_cache=False)
                base_log_probs_all = torch.nn.functional.log_softmax(base_model_output.logits[:, input_length-1:-1, :].to(torch.float32), dim = -1)
            
            # --- Get Policy Model (New) Log-Probs ---
            model.enable_adapters()
            model.train()
            policy_model_output = model.forward(generation_slice, torch.ones(generation_slice.shape), use_cache=False)
            policy_log_probs_all = torch.nn.functional.log_softmax(policy_model_output.logits[:, input_length-1:-1, :].to(torch.float32), dim = -1)
            
            response_slice = generation_slice[:, input_length:]

            if response_slice.shape[1] == 0:
                print("Warning: Empty response slice. Skipping.")
                continue 
            if response_slice.shape[1] != base_log_probs_all.shape[1]:
                # This can happen if the response text tokenizes to 0 tokens after the prompt
                # e.g., if input_length is 50, but full_text is 51, response_slice is 1
                # but log_probs is also 1 (predicting last token). This is fine.
                # The check should be for mismatched *sequence lengths* in the log_probs.
                if response_slice.shape[1] != policy_log_probs_all.shape[1]:
                    print(f"Warning: Tokenizer mismatch. Response shape {response_slice.shape[1]} vs Log-prob shape {policy_log_probs_all.shape[1]}. Skipping.")
                    continue

            # --- More Stable GRPO Loss Calculation (using log-probs) ---
            response_slice_for_gather = response_slice.unsqueeze(-1)
            
            selected_policy_log_probs = policy_log_probs_all.gather(dim=-1, index=response_slice_for_gather).squeeze(-1)
            selected_base_log_probs = base_log_probs_all.gather(dim=-1, index=response_slice_for_gather).squeeze(-1)

            seq_log_prob_policy = torch.sum(selected_policy_log_probs, dim=-1)
            seq_log_prob_base = torch.sum(selected_base_log_probs, dim=-1)

            log_ratio = seq_log_prob_policy - seq_log_prob_base
            ratio = torch.exp(log_ratio)
            
            eps = 0.01 
            clipped_ratio = torch.clip(ratio, min = 1 - eps, max = 1 + eps)

            unclipped_objective = ratio * reward
            clipped_objective = clipped_ratio * reward

            # We use torch.min because optimizer is maximizing. We want the *lesser* of the two objectives.
            base_loss = torch.min(unclipped_objective, clipped_objective)

            # KL Divergence Penalty
            with torch.no_grad():
                kl_per_token = selected_policy_log_probs - selected_base_log_probs
                # Ensure probabilities are not zero for KL calculation
                kl_divergence = (torch.exp(selected_policy_log_probs) * kl_per_token).sum(dim=-1).mean()
            
            loss = base_loss - beta * kl_divergence
            
            clear_vram()
            torch.cuda.synchronize()
            loss.backward()
            
            # --- Updated Logging ---
            result_queue.put({"loss": loss.item(), "kl": kl_divergence.item(), 
                              "clipped_ratio": clipped_ratio.item(), "unclipped_ratio": ratio.item(),
                              "norm_reward(adv)": reward, "raw_reward_avg": E_reward, 
                              "length": response_slice.shape[1], "base_loss": base_loss.item(),
                              "full length": length, "time": time.time() - start, 
                              "num_examples": LOGGING_COUNTER_ONLY, "num_legal_moves": num_legal_moves, 
                              "current_lr": scheduler.get_last_lr()[0]})
        
        # --- Gradient Accumulation & Optimizer Step ---
        if LOGGING_COUNTER_ONLY % NUM_GRAD_ACCUMULATION_EXAMPLES == NUM_GRAD_ACCUMULATION_EXAMPLES - 1:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            sys.stdout.flush()

    # --- Save Model (This is IDENTICAL to your original script) ---
    save_path = "./"
    model.save_pretrained(f"{save_path}{epoch}_placeholder", weird_custom_arg = True)
    if epoch != 0:
        full_precison_model = Qwen2ForCausalLM.from_pretrained(f"/mnt/t9/cshs_checkpoints/off_policy_checkpoint_{epoch-1}", torch_dtype = torch.bfloat16)
    else:
        full_precison_model = Qwen2ForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", torch_dtype = torch.bfloat16)
    merge_model=PeftModel.from_pretrained(full_precison_model, f"{save_path}{epoch}_placeholder")
    merge_model = merge_model.merge_and_unload()
    try:
        merge_model.disable_adapters()
    except Exception as e:
        print(e)
    merge_model.save_pretrained(f"/mnt/t9/cshs_checkpoints/off_policy_checkpoint_{epoch}")
    tokenizer.save_pretrained(f"/mnt/t9/cshs_checkpoints/off_policy_checkpoint_{epoch}")

# ==============================================================================
# === HELPER FUNCTIONS (UNCHANGED) ===
# ==============================================================================

def print_gpu_memory():
    print(f"Allocated memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"Cached memory: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

import gc
def clear_vram():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

# ==============================================================================
# === MAIN SCRIPT (DRIVER - UPDATED) ===
# ==============================================================================

MAXIMUM_GENERATION_LENGTH = 30 # Reduced for chess moves
NUM_GRAD_ACCUMULATION_EXAMPLES = 1
RESPONSES_PER_BATCH = 33 # This is your "group size" for GRPO
start = time.time()
mp.set_start_method("spawn", force=True)

if __name__ == '__main__':
    import wandb
    wandb.init(project="GRPO-chess-llm")
    
    # --- This logging thread is unchanged and correct ---
    def queue_logger(result_queue):
        while True:
            if not result_queue.empty():
                wandb.log(result_queue.get())
            else:
                time.sleep(0.01)

    # --- !! REPLACE THIS WITH YOUR FEN/PGN DATASET !! ---
    # This is your `iteration_list`. It should be a list of FENs or PGNs.
    # Using your test FEN as a placeholder.
    erigaisi_fen = "4nk2/p1r1qpp1/1p5p/3B4/2P2Q1P/1P3R2/P4PK1/8 w - - 1 38"
    another_fen = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
    
    # In a real run, you would load this from a large file
    full_dataset = [erigaisi_fen, another_fen] * 10 # Dummy dataset of 20 prompts
    # --- !! END OF PLACEHOLDER !! ---
    
    # Get the move tags from your BoardEnv
    # We do this once to build prompts
    try:
        temp_env = BoardEnv()
        MOVE_START_TAG = temp_env.MOVE_START_TAG
        MOVE_END_TAG = temp_env.MOVE_END_TAG
    except Exception as e:
        print(f"Fatal Error: Could not load BoardEnv or its tags. {e}")
        sys.exit()


    for epoch in range(45, 100):
        # The `iteration_list` is now our list of FENs
        iteration_list = full_dataset
        random.shuffle(iteration_list)
        
        # --- Data Generation Step ---
        if epoch != 0:
            with open(f'big_llm_data/chess_fen_list_{epoch}.json', 'w') as f:
                json.dump(iteration_list, f)

            sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=MAXIMUM_GENERATION_LENGTH)
            llm = LLM(model=f"/mnt/t9/cshs_checkpoints/off_policy_checkpoint_{epoch-1}")
            
            generation_list = []
            generation_outputs = []
            
            # This is the list of FENs for vllm to process
            fen_prompt_list = []
            
            for LOGGING_COUNTER_ONLY, board_fen in enumerate(iteration_list):
                
                # --- NEW: Create Chess Prompt ---
                # This prompt structure must match what your BoardEnv `parse_move` expects
                prompt = f"""You are a chess grandmaster. Analyze the following position:
FEN: {board_fen}
What is the best move? Respond with your reasoning and the move inside tags.
Example: The best move is {MOVE_START_TAG}Nf3{MOVE_END_TAG}."""
                
                messages = [
                    {"role": "system", "content": "You are a helpful chess assistant."},
                    {"role": "user", "content": prompt}
                ]
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                for _ in range(RESPONSES_PER_BATCH):
                    generation_list.append(text)
                    # We store the FEN alongside the prompt
                    fen_prompt_list.append(board_fen)
            
            # Generate all responses in one big batch
            outputs = llm.generate(generation_list, sampling_params)

            # Process outputs
            for i, output in enumerate(outputs):
                prompt = output.prompt
                generated_text = output.outputs[0].text
                
                # Get the FEN that corresponds to this output
                associated_fen = fen_prompt_list[i]
                
                # --- NEW: Save data in your specified dictionary format ---
                chat_log_entry = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": generated_text}
                ]
                
                output_data = {
                    "chat_logs": [chat_log_entry],
                    "board_position": associated_fen
                }
                generation_outputs.append(output_data)
                
            with open(f'big_llm_data/generated_chess_data_{epoch}.json', 'w') as f:
                json.dump(generation_outputs, f)
            del llm
            
        else:
            # Logic for loading data on epoch 0 (unchanged)
            with open(f'big_llm_data/generated_chess_data_0.json', 'r') as f:
                generation_outputs = json.load(f)
            with open(f'big_llm_data/chess_fen_list_0.json', 'r') as f:
                iteration_list = json.load(f)
        
        
        # --- Training Step (This is IDENTICAL to your original script) ---
        clear_vram()
        result_queue = mp.Queue()
        logging_thread = threading.Thread(target=queue_logger, args=(result_queue,), daemon=True)
        logging_thread.start()
        
        p = mp.Process(target=memory_isolated_train_step, args=(generation_outputs, iteration_list, epoch, result_queue))
        p.start()
        p.join()  # Wait for training to finish
        
        clear_vram()
        print(torch.cuda.mem_get_info())
        print_gpu_memory()
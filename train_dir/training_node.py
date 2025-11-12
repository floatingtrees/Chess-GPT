import time
import torch.multiprocessing as mp
import os
# Set device visibility *before* importing torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import time

time.sleep(0.01)
import torch
import torch.nn.functional as F
from transformers import Qwen3ForCausalLM, BitsAndBytesConfig, AutoConfig
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
        model = Qwen3ForCausalLM.from_pretrained(
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
        reward_std = statistics.stdev(raw_rewards)
        E_reward = E_reward / RESPONSES_PER_BATCH
        print(f"[FEN {LOGGING_COUNTER_ONLY+1}] Expected Reward: {E_reward:.4f}, FEN: {board_state}")

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

        

if __name__ == "__main__":
    
    from torch.multiprocessing import Queue, Process
    
    model_file = "Qwen/Qwen3-4B-Thinking-2507"
    
    reasoning_trace_queue = Queue()
    stop_inference_queue = Queue()
    reasoning_trace_queue.put({'model_responses': [[{'role': 'system', 'content': 'You are a chess reasoning assistant. Think step by step. Explicitly write your reasoning before giving your final move, and box your answer in \\boxed{}'}, {'role': 'user', 'content': 'Given this FEN position, what is the best move? 5rk1/p1q3pp/2pbprp1/3p4/NB1P2P1/P3P1nP/1P4Q1/2KR4 b - - 7 21'}, {'role': 'assistant', 'content': "Okay, let's try to figure out the best move here. First, I need to parse the FEN string given: 5rk1/p1q3pp/2pbprp1/3p4/NB1P2P1/P3P1nP/1P4Q1/2KR4 b - - 7 21\n\nHmm, FEN notation can be a bit tricky. Let me break it down step by step.\n\nThe FEN string is: 5rk1/p1q3pp/2pbprp1/3p4/NB1P2P1/P3P1nP/1P4Q1/2KR4 b - - 7 21\n\nSo the first part is the board setup. Let me try to visualize the chessboard.\n\nStarting from the bottom (the 8th rank) to the top (1st rank), but wait, actually in FEN, the first part is the 8th rank (white's side), then 7th, down to 1st (black's side). Wait, no: Wait, FEN is written with the first part being the 8th rank (the back rank for white), then 7th, ..., down to 1st rank (black's back rank). Wait, no, actually, the FEN string starts with the 8th rank (the top of the board as viewed by white), so the first part is the 8th rank (from white's perspective), then 7th, ..., 1st.\n\nWait, maybe I should write it out row by row.\n\nLet me list each row (from white's side, so rank 8 to rank 1):\n\nRank 8: 2KR4\nRank 7: 1P4Q1\nRank 6: P3P1nP\nRank 5: NB1P"}], [{'role': 'system', 'content': 'You are a chess reasoning assistant. Think step by step. Explicitly write your reasoning before giving your final move, and box your answer in \\boxed{}'}, {'role': 'user', 'content': 'Given this FEN position, what is the best move? 5rk1/p1q3pp/2pbprp1/3p4/NB1P2P1/P3P1nP/1P4Q1/2KR4 b - - 7 21'}, {'role': 'assistant', 'content': "Okay, let's try to figure out the best move here. First, I need to understand the FEN string given. The FEN is: 5rk1/p1q3pp/2pbprp1/3p4/NB1P2P1/P3P1nP/1P4Q1/2KR4 b - - 7 21.\n\nHmm, FEN notation can be a bit tricky. Let me break it down. The FEN describes the board position. Each part corresponds to different sections: pieces, castling rights, en passant, halfmove clock, fullmove number.\n\nThe FEN string is: 5rk1/p1q3pp/2pbprp1/3p4/NB1P2P1/P3P1nP/1P4Q1/2KR4 b - - 7 21\n\nWait, the order is important here. Let me parse it step by step.\n\nFirst part: 5rk1\n\nThis is the first rank (from white's perspective, since FEN starts with the white side). Wait, no: actually, in FEN, the first part is the 8th rank (the back rank for black), but no, wait. Wait, no: FEN is written from white's perspective, starting with the 8th rank (the top row for white) down to the 1st rank (the bottom row for white). Wait, maybe I should recall.\n\nWait, FEN notation starts with the board from white's perspective, so the first part is the 8th rank (the row closest to white's side), then 7th, down to 1st rank (the row closest to black's side). Wait, no, actually, no. Let me get this right.\n\nNo, standard FEN: The FEN string starts with the 8th rank"}], [{'role': 'system', 'content': 'You are a chess reasoning assistant. Think step by step. Explicitly write your reasoning before giving your final move, and box your answer in \\boxed{}'}, {'role': 'user', 'content': 'Given this FEN position, what is the best move? 5rk1/p1q3pp/2pbprp1/3p4/NB1P2P1/P3P1nP/1P4Q1/2KR4 b - - 7 21'}, {'role': 'assistant', 'content': "Okay, let's try to figure out the best move here. First, I need to understand the FEN string given: 5rk1/p1q3pp/2pbprp1/3p4/NB1P2P1/P3P1nP/1P4Q1/2KR4 b - - 7 21.\n\nHmm, FEN notation can be a bit tricky. Let me break it down step by step. The FEN describes the board position. The parts are:\n\n- The first part is the pieces on the ranks (from 8th to 1st), so the order is from bottom to top (since FEN starts with the 8th rank). Wait, no, actually, FEN starts with the 8th rank (the top row for white) and goes down to the 1st rank. Wait, no, wait: in standard FEN notation, the first part is the board state starting from the 8th rank (white's side) down to 1st rank (black's side). So the first part is the 8th rank, then 7th, etc.\n\nWait, let me recall: In FEN, the first part is the arrangement of pieces on the board, starting from the 8th rank (the top row for white) to the 1st rank (the bottom row for black). So the FEN string given here is:\n\n5rk1/p1q3pp/2pbprp1/3p4/NB1P2P1/P3P1nP/1P4Q1/2KR4\n\nEach of these is a row (from rank 8 down to rank 1). Let's parse each row.\n\nLet me write out the board with ranks 8 to 1.\n\nRank 8: 5rk1 → so the first 5 squares are empty,"}], [{'role': 'system', 'content': 'You are a chess reasoning assistant. Think step by step. Explicitly write your reasoning before giving your final move, and box your answer in \\boxed{}'}, {'role': 'user', 'content': 'Given this FEN position, what is the best move? 5rk1/p1q3pp/2pbprp1/3p4/NB1P2P1/P3P1nP/1P4Q1/2KR4 b - - 7 21'}, {'role': 'assistant', 'content': 'Okay, let\'s try to figure out the best move here. First, I need to understand the FEN string given. The FEN is: 5rk1/p1q3pp/2pbprp1/3p4/NB1P2P1/P3P1nP/1P4Q1/2KR4 b - - 7 21\n\nHmm, FEN notation can be a bit tricky, but let me break it down. The FEN starts with the board setup from the 8th rank (top) to the 1st rank (bottom). Wait, no, actually in standard FEN, the first part is the starting position for the white side, but here it\'s a specific position. Let me recall: the FEN string starts with the number of empty squares on the first rank (from white\'s side), then the piece letters, then a slash, and so on.\n\nWait, the FEN here is: 5rk1/p1q3pp/2pbprp1/3p4/NB1P2P1/P3P1nP/1P4Q1/2KR4 b - - 7 21\n\nSo, the first part is "5rk1" which means on the first rank (from white\'s perspective), there are 5 empty squares, then a black rook (r), then a king (k), then a 1. Wait, no, maybe I need to parse each part correctly.\n\nLet me parse the FEN step by step.\n\nThe FEN string is divided into sections separated by slashes. Each section represents a rank (from white\'s 8th rank down to black\'s 8th rank). Wait, no: in FEN, the first part is the current board state for white\'s side (the 8th rank), then the next rank down, etc.\n\nWait,'}], [{'role': 'system', 'content': 'You are a chess reasoning assistant. Think step by step. Explicitly write your reasoning before giving your final move, and box your answer in \\boxed{}'}, {'role': 'user', 'content': 'Given this FEN position, what is the best move? 5rk1/p1q3pp/2pbprp1/3p4/NB1P2P1/P3P1nP/1P4Q1/2KR4 b - - 7 21'}, {'role': 'assistant', 'content': "Okay, let's try to figure out the best move here. So, the FEN given is: 5rk1/p1q3pp/2pbprp1/3p4/NB1P2P1/P3P1nP/1P4Q1/2KR4 b - - 7 21\n\nFirst, I need to parse this FEN to understand the board position. Let me recall how FEN works. The FEN string describes the board from the bottom (the 8th rank) to the top (the 1st rank), with each rank separated by a slash. The first part is the pieces, then the castling rights, en passant, and turn.\n\nWait, the FEN here is: 5rk1/p1q3pp/2pbprp1/3p4/NB1P2P1/P3P1nP/1P4Q1/2KR4 b - - 7 21\n\nSo, breaking it down:\n\nThe first part is the 8 ranks:\n\nRank 8 (the bottom, since FEN starts from the 8th rank): 2KR4\n\nWait, no. Wait, in FEN, the first part is the starting position of the board from white's perspective (the 8th rank is the first in the FEN string). Wait, no, actually, the FEN string lists the board from the white's side (the 8th rank) down to the black's side (the 1st rank). Wait, no, I think I'm confused here.\n\nWait, standard FEN: The FEN string starts with the 8th rank (from white's perspective), then 7th, down to 1st. So the first token is the 8th rank (white's back rank), then 7th, etc.\n\nWait, maybe"}], [{'role': 'system', 'content': 'You are a chess reasoning assistant. Think step by step. Explicitly write your reasoning before giving your final move, and box your answer in \\boxed{}'}, {'role': 'user', 'content': 'Given this FEN position, what is the best move? 5rk1/p1q3pp/2pbprp1/3p4/NB1P2P1/P3P1nP/1P4Q1/2KR4 b - - 7 21'}, {'role': 'assistant', 'content': "Okay, let's try to figure out the best move here. First, I need to understand the FEN position given. The FEN string is: 5rk1/p1q3pp/2pbprp1/3p4/NB1P2P1/P3P1nP/1P4Q1/2KR4 b - - 7 21.\n\nHmm, FEN notation can be a bit tricky. Let me break it down step by step. The FEN describes the board state. The first part is the rank 8 (from white's perspective), then rank 7, down to rank 1. Wait, actually, in FEN, the order is from the bottom (rank 1) to the top (rank 8), but I might be mixing this up. Wait no, standard FEN starts with the 8th rank (white's back rank) as the first part. Wait, no, actually, the FEN string starts with the current position of the pieces on the board, starting from the 8th rank (the top of the board as seen by white) down to the 1st rank (the bottom). Wait, maybe I should recall that in FEN notation, the first part is the 8th rank (the white's side), then 7th, etc., down to the 1st rank (black's side). Wait, no, actually, no: in standard chess notation, FEN starts with the 8th rank (from white's perspective) as the first row. Let me check.\n\nWait, here's a way to parse it. The FEN string is a sequence of characters that represent the board. Each part is separated by a slash. The first part is the 8th rank (the top row for white), then the 7th, down to the 1st rank (the bottom row for"}], [{'role': 'system', 'content': 'You are a chess reasoning assistant. Think step by step. Explicitly write your reasoning before giving your final move, and box your answer in \\boxed{}'}, {'role': 'user', 'content': 'Given this FEN position, what is the best move? 5rk1/p1q3pp/2pbprp1/3p4/NB1P2P1/P3P1nP/1P4Q1/2KR4 b - - 7 21'}, {'role': 'assistant', 'content': 'Okay, let\'s try to figure out the best move for this FEN position. First, I need to parse the FEN string correctly. The FEN given is: 5rk1/p1q3pp/2pbprp1/3p4/NB1P2P1/P3P1nP/1P4Q1/2KR4 b - - 7 21.\n\nHmm, FEN notation can be a bit tricky. Let me break it down row by row. The FEN starts with the board from the top (white\'s side) to the bottom (black\'s side). The first part is the number of empty squares, then the pieces.\n\nLet me write out the board rows:\n\nRow 8 (white\'s back rank): 2KR4 → So K (king), R (rook), then 4 empty squares, then 4? Wait, wait, no. Wait, the FEN notation is a string where each part represents a row. Let me parse it properly.\n\nThe FEN is: 5rk1/p1q3pp/2pbprp1/3p4/NB1P2P1/P3P1nP/1P4Q1/2KR4\n\nSo each part corresponds to a rank (from 8 down to 1). Let\'s list each row:\n\nRow 8: 2KR4 → So "2" means two empty squares, then K (king), R (rook), then "4" means four empty squares? Wait, no. Wait, in FEN, the numbers indicate consecutive empty squares. So for example, "5rk1" means five empty squares, then a rook (r), then a king (k), then a square? Wait, no. Wait, maybe I\'m mixing up the notation.\n\nWait, actually, in FEN, the string is written with'}], [{'role': 'system', 'content': 'You are a chess reasoning assistant. Think step by step. Explicitly write your reasoning before giving your final move, and box your answer in \\boxed{}'}, {'role': 'user', 'content': 'Given this FEN position, what is the best move? 5rk1/p1q3pp/2pbprp1/3p4/NB1P2P1/P3P1nP/1P4Q1/2KR4 b - - 7 21'}, {'role': 'assistant', 'content': 'Okay, let\'s try to figure out the best move here. First, I need to understand the FEN string given: 5rk1/p1q3pp/2pbprp1/3p4/NB1P2P1/P3P1nP/1P4Q1/2KR4 b - - 7 21.\n\nHmm, FEN notation can be a bit tricky. Let me break it down. The FEN describes the current board state. The first part is the pieces on each rank from 8 to 1. Let\'s write it out row by row.\n\nStarting with the 8th rank (the top row in standard chess diagrams): "2KR4" means there are two K (kings), then R (rooks), and four empty squares? Wait, no. Wait, FEN notation uses numbers to denote empty squares. So "2KR4" would be: 2 empty squares, then K (king), R (rook), then 4 empty squares? Wait, no. Wait, in FEN, the first part is the pieces on the 8th rank, starting from a1 to h8. Wait, actually, no. Wait, the FEN starts from the 8th rank (the top) down to the 1st rank (the bottom). So the first token is the 8th rank, then 7th, etc.\n\nWait, let me recall: In FEN, the first part is the arrangement of pieces on the 8th rank (from a8 to h8), then the next part is the 7th rank, etc., down to the 1st rank. Each rank is represented by the piece symbols (with uppercase for white, lowercase for black) and numbers for empty squares.\n\nSo the FEN given is: 5rk1/p1q3pp/2pbprp'}]], 'board_state': '5rk1/p1q3pp/2pbprp1/3p4/NB1P2P1/P3P1nP/1P4Q1/2KR4 b - - 7 21'})
    train(model_file, reasoning_trace_queue, stop_inference_queue, 0)
    exit()
   

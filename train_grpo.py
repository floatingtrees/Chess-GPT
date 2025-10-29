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

import sys

from torch.optim.lr_scheduler import LambdaLR

from stockfish import Stockfish
# Assuming these are in the python path or relative
from envs.chess_env import BoardEnv
from reward import (
    reward as get_reward_from_fen,  # Rename to avoid conflict with 'reward' variable
    FIXED_DEPTH
)
import bitsandbytes as bnb

# --- Hyperparameters ---
RESPONSES_PER_BATCH = 16  # The 'k' in "sample k responses". How many responses per FEN.
NUM_GRAD_ACCUMULATION_EXAMPLES = 4  # How many FENs to process before one optimizer step.

# --- Stockfish/Reward Config ---
STOCKFISH_PATH = "/scratch/ChessGPT/stockfish/stockfish-engine"  # <-- UPDATE THIS PATH

# --- System Prompt (Moved here as a global constant) ---
# This is required by the training loop but was missing from the new data structure.
SYSTEM_PROMPT = {
    "role": "system",
    "content": """You are a world-class chess grandmaster and strategic analyst. Your name is 'Maestro'.

When the user provides you with a board position in FEN (Forsyth-Edwards Notation), your task is to find the single best move.

You must provide your answer in two distinct parts:
1.  **Best Move:** State the best move in Standard Algebraic Notation (SAN) (e.g., "Nf3", "O-O", "e8=Q").
2.  **Reasoning:** Provide a concise, expert analysis for your choice. Explain the primary tactical or positional ideas, why this move is superior to alternatives, and what key opponent responses you anticipate.
"""
}

torch.set_default_dtype(torch.bfloat16)
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
# Tokenizer is loaded globally, which is fine
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")


def clear_vram() -> None:
    """Clear the VRAM memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


#I used AI for the save_model function we still need to look at it

def save_model(model_to_save, tokenizer_var, epoch):
    """
    Save the trained model by merging the adapter.
    
    Args:
        model_to_save (PeftModel): The 8-bit quantized model with the adapter.
        tokenizer_var (AutoTokenizer): The tokenizer to save.
        epoch (int): The current epoch/save step.
        
    Returns:
        str: The path to the saved full model.
    """
    print("Merging adapter and converting to full-precision model...")

    # --- This merging process is memory-intensive ---
    # We first merge the adapter into the 8-bit model, then unload it
    # to get a full-precision (bfloat16) model.
    
    # It's safer to do this on the CPU to avoid OOM errors on the GPU
    # which is busy with training.
    try:
        # 1. Get the full-precision base model on CPU
        full_base_model = Qwen3ForCausalLM.from_pretrained(
            "Qwen/Qwen3-8B",
            torch_dtype=torch.bfloat16,
            device_map="cpu" # Load to CPU to save VRAM
        )
        
        # 2. Load the adapter onto the CPU base model
        adapter_path = f"./cshs_checkpoints/adapter_epoch_{epoch}"
        print(f"Loading adapter from: {adapter_path}")
        full_model = PeftModel.from_pretrained(full_base_model, adapter_path)
        
        # 3. Merge the adapter weights
        print("Merging adapter weights on CPU...")
        merged = full_model.merge_and_unload()
        output_path = f"./cshs_checkpoints/full_model_epoch_{epoch}"

        # 4. Save the new, merged model
        print(f"Saving merged model to: {output_path}")
        merged.save_pretrained(
            output_path,
            safe_serialization=True,
            max_shard_size="5GB"
        )
        tokenizer_var.save_pretrained(output_path)
        
        print(f"✓ Full model saved to {output_path}")
        return output_path
        
    except Exception as e:
        print(f"!! FAILED to merge and save model on CPU: {e}")
        print("Attempting to merge on GPU (may cause OOM)...")
        try:
            # Fallback to GPU if CPU merge fails (e.g., PeftModel issue)
            # This is less safe for memory.
            merged_gpu = model_to_save.merge_and_unload()
            output_path = f"./cshs_checkpoints/full_model_epoch_{epoch}"
            
            merged_gpu.save_pretrained(output_path, safe_serialization=True, max_shard_size="5GB")
            tokenizer_var.save_pretrained(output_path)
            
            print(f"✓ Full model saved to {output_path} (GPU fallback)")
            # We must reload the adapter onto the base model, since merge_and_unload() is destructive
            model_to_save.load_adapter(adapter_path, "default")
            model_to_save.set_adapter("default")
            return output_path
            
        except Exception as e_gpu:
            print(f"!! FAILED to merge on GPU as well: {e_gpu}")
            print("!! Model not saved. Only adapter weights are saved.")
            return None


def linear_schedule(step):
    """Implements a linear warmup for the learning rate."""
    step += 1
    warmup_steps = 3
    if step < warmup_steps:
        return step / warmup_steps
    return 1.0


def train(model_path, reasoning_trace_queue, stop_inference_queue, GPU_IDX):
    """
    Performs continuous GRPO training as a long-running process.
    
    Args:
        model_path (str): Path to the initial model (e.g., "Qwen/Qwen3-8B").
        reasoning_trace_queue (mp.Queue): Queue to receive data from the generator.
        stop_inference_queue (mp.Queue): Queue to send new model paths to the generator.
        GPU_IDX (int): The specific GPU index to run this trainer on.
    """

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
    
    # Load the base model ONCE at the start.
    # We will train an adapter on top of this base model.
    print(f"[Trainer] Loading initial base model from: {model_path}")
    try:
        model = Qwen3ForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map=device  # Load the 8-bit model directly onto the correct GPU
        )
    except Exception as e:
        print(f"[Trainer] ERROR: Failed to load model {model_path}. {e}")
        return
        
    # Enable memory-saving features
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # --- 2. LoRA (PEFT) Setup ---
    # We add ONE adapter and train it continuously.
    lora_config = LoraConfig(
        r=256,
        lora_alpha=512,
        lora_dropout=0.00,
        bias="none",
        task_type="CAUSAL_LM"
    )
    # We use a consistent adapter name
    adapter_name = "grpo_adapter"
    model.add_adapter(adapter_config=lora_config, adapter_name=adapter_name)
    model.set_adapter(adapter_name)
    model.train()  # Set only the adapter weights to trainable

    # --- 3. Optimizer and Scheduler Setup ---
    beta = 0.01  # KL penalty coefficient
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5, maximize=True)
    optimizer.zero_grad()  # Clear any old gradients
    scheduler = LambdaLR(optimizer, lr_lambda=linear_schedule)

    # --- 4. Stockfish Initialization ---
    try:
        stockfish = Stockfish(path=STOCKFISH_PATH, depth=FIXED_DEPTH)
    except Exception as e:
        print(f"[Trainer] ERROR: Failed to initialize Stockfish at {STOCKFISH_PATH}: {e}")
        print("[Trainer] Exiting training process.")
        return

    print(f"[Trainer] Initialization complete. Waiting for data...")
    
    # These counters persist for the life of the process
    LOGGING_COUNTER_ONLY = 0 # Counts FENs for gradient accumulation
    epoch = 0 # Counts save/update steps

    # --- 5. Main Training Loop (Continuous) ---
    while True:
        try:
            # Wait for and get data from the generator node
            # This is a blocking call
            data = reasoning_trace_queue.pop()
            
            if data is None:
                print("[Trainer] Received None. Shutting down training.")
                break
                
            start_time = time.time()  # For logging time

            board_state = data["board_position"] # Renamed from board_state
            chat_logs = data["chat_logs"]      # List of [user, assistant] pairs

            user_prompt_dict = chat_logs[0][0]
            prompt_messages = [SYSTEM_PROMPT, user_prompt_dict]

            # Tokenize the prompt to get its length
            prompt_tokenized = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=True,
                add_generation_prompt=True,  # Appends the '<|im_start|>assistant\n' tokens
                return_tensors="pt"
            ).to(device) # Send to the correct GPU

            input_length = prompt_tokenized["input_ids"].shape[1]
            clear_vram()

            # 5.2. --- Reward Calculation & Standardization ---
            E_reward = 0
            raw_rewards = []
            rewards = []  # This will store normalized advantages

            for i, chat_pair in enumerate(chat_logs):
                model_response = chat_pair[1]["content"]

                try:
                    reward_value = get_reward_from_fen(board_state, model_response, stockfish)
                except Exception as e:
                    print(
                        f"Error calculating reward for FEN {board_state}, response '{model_response}': {e}. Assigning penalty.")
                    reward_value = -5.0

                E_reward += reward_value
                raw_rewards.append(reward_value)

            reward_std = statistics.stdev(raw_rewards) if len(raw_rewards) > 1 else 0
            E_reward = E_reward / RESPONSES_PER_BATCH
            print(f"[FEN {LOGGING_COUNTER_ONLY+1}] Expected Reward: {E_reward:.4f}, FEN: {board_state}")

            if reward_std == 0:
                print("Zero advantage, skipping batch.")
                continue  # Skip to the next item in the queue
            else:
                for element in raw_rewards:
                    rewards.append((element - E_reward) / reward_std)

            # 6. --- BATCHED Policy --- 
            all_full_chat_messages = []
            for i in range(len(rewards)):
                chat_pair = chat_logs[i]
                full_chat_messages = [SYSTEM_PROMPT, chat_pair[0], chat_pair[1]]
                all_full_chat_messages.append(full_chat_messages)
            
            # 6.1. Tokenize the entire batch
            full_text_tokenized = tokenizer.apply_chat_template(
                all_full_chat_messages,
                tokenize=True,
                add_generation_prompt=False,
                padding=True,
                truncation=True,
                max_length=4096, 
                return_tensors="pt"
            ).to(device) # Send to the correct GPU

            batched_input_ids = full_text_tokenized["input_ids"]
            batched_attention_mask = full_text_tokenized["attention_mask"]
            advantages_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
            
            # 6.2. Get Reference Probs (pi_base)
            model.eval()
            clear_vram()
            model.disable_adapters()
            with torch.no_grad():
                base_model_output = model(
                    batched_input_ids,
                    attention_mask=batched_attention_mask,
                    use_cache=False
                )
                base_logits = base_model_output.logits[:, input_length - 1:-1, :].to(torch.float32)
                base_log_probs = F.log_softmax(base_logits, dim=-1)

            # 6.3. Get Policy Probs (pi_policy)
            model.enable_adapters()
            model.train()
            policy_model_output = model(
                batched_input_ids,
                attention_mask=batched_attention_mask,
                use_cache=False
            )
            policy_logits = policy_model_output.logits[:, input_length - 1:-1, :].to(torch.float32)
            policy_log_probs = F.log_softmax(policy_logits, dim=-1)

            # 6.4. Calculate Ratios and Loss
            response_tokens = batched_input_ids[:, input_length:]
            response_mask = batched_attention_mask[:, input_length:]

            selected_policy_log_probs = torch.gather(policy_log_probs, 2, response_tokens.unsqueeze(-1)).squeeze(-1)
            selected_base_log_probs = torch.gather(base_log_probs, 2, response_tokens.unsqueeze(-1)).squeeze(-1)

            log_ratio = selected_policy_log_probs - selected_base_log_probs
            policy_ratio = torch.exp(log_ratio)
            
            eps = 0.01
            clipped_policy_ratio = torch.clip(policy_ratio, min=1 - eps, max=1 + eps)
            
            # Masking
            clipped_policy_ratio = clipped_policy_ratio * response_mask + (1 - response_mask)
            
            # KL Divergence Penalty (Masked)
            kl_per_token = torch.maximum(log_ratio, torch.zeros_like(log_ratio)) * response_mask
            kl_divergence = torch.mean(torch.sum(kl_per_token, dim=1))

            # GRPO Loss (Masked)
            sequence_product_ratio = torch.prod(clipped_policy_ratio, dim=1)
            base_loss_per_sequence = sequence_product_ratio * advantages_tensor
            base_loss = torch.mean(base_loss_per_sequence)

            loss = base_loss - beta * kl_divergence

            clear_vram()
            torch.cuda.synchronize()

            # 6.5. Backpropagation
            loss.backward()

            # 6.6. Logging (to console)
            print(f"  [Loss]: {loss.item():.4f} [KL]: {kl_divergence.item():.4f} [Base Loss]: {base_loss.item():.4f}")
            
            # --- 7. Gradient Accumulation & Model Saving ---
            if (LOGGING_COUNTER_ONLY + 1) % NUM_GRAD_ACCUMULATION_EXAMPLES == 0:
                print(f"[Trainer] Step {epoch+1} @ FEN {LOGGING_COUNTER_ONLY + 1}: Performing optimizer step.")
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # Increment save counter
                epoch += 1
                
                # --- Save Adapter Weights (Temporary) ---
                # This is the lightweight checkpoint of the adapter's state
                adapter_save_path = f"./cshs_checkpoints/adapter_epoch_{epoch}"
                print(f"[Trainer] Saving adapter to {adapter_save_path}")
                model.save_pretrained(adapter_save_path)
                print(f"[Trainer] Adapter saved successfully.")

                # --- Save Full Merged Model ---
                # This merges the base + adapter and saves the full weights
                full_model_path = save_model(model, tokenizer, epoch)
                
                if full_model_path:
                    # --- Send New Path to Inference Node ---
                    # This tells the vLLM node to stop and reload with the new model
                    print(f"[Trainer] Sending new model path to inference queue: {full_model_path}")
                    stop_inference_queue.put(full_model_path)
                else:
                    print("[Trainer] ERROR: Model merging failed. Not notifying inference node.")
                
                sys.stdout.flush()

            LOGGING_COUNTER_ONLY += 1
        
        except KeyboardInterrupt:
            print("[Trainer] Keyboard interrupt detected. Shutting down.")
            break
        except Exception as e:
            print(f"[Trainer] !! UNHANDLED ERROR in training loop: {e}")
            print("[Trainer] Skipping this batch and continuing...")
            clear_vram()
            optimizer.zero_grad()
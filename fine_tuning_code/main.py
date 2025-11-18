# -*- coding: utf-8 -*-
"""Improved Qwen Chess Fine-tuning"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, Trainer, default_data_collator

# ============= MODEL LOADING =============
model_name = "Qwen/Qwen3-1.7B"
print(f"Loading model: {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

# CRITICAL: Set pad token before training
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.bfloat16,  # Fixed from deprecated parameter 
    device_map="auto"
)

# Enable gradient computation for embeddings (required for gradient checkpointing)
model.enable_input_require_grads()

# ============= DATA LOADING =============
data_path = "analysis.json"

with open(data_path) as f:
    data = json.load(f)

# Calculate max token length
max_token_length = 0
for conv in data:
    full_text = tokenizer.apply_chat_template(
        conv, tokenize=False, add_generation_prompt=False
    )
    tokenized = tokenizer(full_text, truncation=False)["input_ids"]
    if len(tokenized) > max_token_length:
        max_token_length = len(tokenized)

print(f"Longest tokenized sequence length: {max_token_length}")

# Add padding to nearest multiple of 8 for efficiency
max_token_length = ((max_token_length + 7) // 8) * 8
print(f"Padded max length: {max_token_length}")

wrapped = [{"messages": conv} for conv in data]
ds_train = Dataset.from_list(wrapped)

# ============= TOKENIZATION =============
ignore_index = -100

def format_and_tokenize(example, max_len=max_token_length):
    """Tokenize and create labels for training"""
    msgs = example["messages"]
    
    # Full conversation
    full_text = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=False
    )
    
    # Prefix (user message only)
    prefix_text = tokenizer.apply_chat_template(
        [msgs[0], {"role": "assistant", "content": ""}], 
        tokenize=False, 
        add_generation_prompt=False
    )
    
    # Tokenize
    full = tokenizer(
        full_text, 
        padding="max_length", 
        truncation=True,  # Changed to True for safety
        max_length=max_len
    )
    prefix = tokenizer(
        prefix_text, 
        padding="max_length", 
        truncation=True,
        max_length=max_len
    )
    
    # Create labels - only train on assistant responses
    input_ids = full["input_ids"]
    labels = input_ids.copy()
    attn = full["attention_mask"]
    
    # Mask the user prompt
    cutoff = int(sum(prefix["attention_mask"]))
    labels[:cutoff] = [ignore_index] * cutoff
    
    # Mask padding tokens
    for i in range(len(labels)):
        if attn[i] == 0:
            labels[i] = ignore_index
    
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attn,
    }

tok_train = ds_train.map(
    format_and_tokenize, 
    remove_columns=ds_train.column_names, 
    batched=False
)

print(f"Dataset size: {len(tok_train)}")

# Verify tokenization
ex = tok_train[0]
print("\n=== SAMPLE INPUT ===")
print(tokenizer.decode(ex["input_ids"], skip_special_tokens=False)[:500])
print("\n=== SAMPLE LABELS (first 500 chars) ===")
label_ids = [t for t in ex["labels"] if t != ignore_index]
print(tokenizer.decode(label_ids, skip_special_tokens=False)[:500])

# ============= LORA CONFIGURATION =============
lora_cfg = LoraConfig(
    r=16,  # Increased from 8 for better capacity
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj",
        "o_proj", "up_proj", "down_proj", "gate_proj"
    ],
)

model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

# ============= TRAINING CONFIGURATION =============
bf16_ok = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8

train_args = TrainingArguments(
    output_dir="qwen3-chess-cot-lora",
    
    # Training schedule
    num_train_epochs=50,  # Reduced from 50 for small dataset
    per_device_train_batch_size=1,  # Increased from 1
    gradient_accumulation_steps=4,  # Added for effective batch size of 8
    
    # Learning rate schedule
    learning_rate=1e-4,  # Slightly higher for LoRA
    lr_scheduler_type="cosine",  # Better than constant
    warmup_ratio=0.03,  # Reduced warmup
    weight_decay=0.01,
    
    # Optimization
    optim="adamw_torch_fused" if bf16_ok else "adamw_torch",
    gradient_checkpointing=True,  # ENABLED for memory efficiency
    gradient_checkpointing_kwargs={"use_reentrant": False},  # Use new checkpointing method
    max_grad_norm=1.0,  # Gradient clipping
    
    # Precision
    bf16=bf16_ok,
    fp16=not bf16_ok,
    
    # Logging and saving
    logging_steps=5,
    save_strategy="epoch",
    save_total_limit=3,  # Keep only best 3 checkpoints
    
    # Misc
    report_to="none",
    seed=42,
)

# ============= TRAINING =============
trainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=tok_train,
    data_collator=default_data_collator,
)

print("\n=== Starting Training ===")
trainer.train()

# ============= SAVING =============
# Save LoRA adapter
adapter_dir = "qwen3-chess-cot-lora/final_adapter"
model.save_pretrained(adapter_dir)
tokenizer.save_pretrained(adapter_dir)
print(f"Saved adapter to: {adapter_dir}")

# Merge and save full model
merged = model.merge_and_unload()
merged_dir = "qwen3-chess-cot-lora/merged_model"
merged.save_pretrained(merged_dir)
tokenizer.save_pretrained(merged_dir)
print(f"Saved merged model to: {merged_dir}")

# ============= INFERENCE FUNCTION =============
def chat_once(prompt_text: str, model_to_use=None):
    """Inference function with proper formatting"""
    m = model_to_use if model_to_use is not None else merged
    
    # Ensure model is in eval mode
    m.eval()
    
    messages = [{"role": "user", "content": prompt_text}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(m.device)
    
    with torch.no_grad():
        out = m.generate(
            **inputs,
            max_new_tokens=1500,
            do_sample=True,
            temperature=0.1,
            top_p=0.8,
            repetition_penalty=1.2,  # Reduce repetition
            pad_token_id=tokenizer.pad_token_id,
        )
    
    output_text = tokenizer.decode(out[0], skip_special_tokens=True)
    
    print("==== OUTPUT ====")
    print(output_text)
    return output_text

# Test inference
test_prompt = """<META_GUIDANCE>You are an expert chess player. You will be given board state within the <BOARD> tag. You will be given misc info (en passant, castling, etc.) within the <MISC_INFO> tag. You will be given the turn (white to move or black to move) within the <TURN> tag which indicates which player you are.</META_GUIDANCE><INSTRUCTIONS>You will be RESPONDING with a final move in the format of <MOVE>[your move here as SAN notation string]</MOVE> where the contents of MOVE are SAN notation strings. E.g. to move the knight to c3 you will return <MOVE>Nc3</MOVE>.</INSTRUCTIONS><BOARD>White: Pawns: f3 g5 a6 e6 Bishops: e2 f8 Rook: h7 King: d4 Black: Pawns: h4 d6 g6 Knight: e1 Bishops: f4 a8 Rook: h2 King: b6</BOARD><MISC_INFO><EN_PASSANT>Has legal EP now: False Target square: - Capturing pawn(s): None Moves (SAN notation): None</EN_PASSANT><CAN_CASTLE>Rights: {'white': {'kingside': False, 'queenside': False}, 'black': {'kingside': False, 'queenside': False}} Legal castle moves now (SAN notation): None</CAN_CASTLE></MISC_INFO><TURN>White to move</TURN>"""

print("\n=== Testing Inference ===")
chat_once(test_prompt)

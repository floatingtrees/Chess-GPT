# testing.py
"""Test script for fine-tuned Qwen chess model"""

from __future__ import annotations
from pathlib import Path
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re

# Import promptgen from same folder
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import promptgen  # assumes promptgen.py defines BoardEnv

# ==================== MODEL LOADING ====================
# IMPORTANT: Update this path to match your training output
MODEL_DIR = Path("~/projects/finetuningQwen/ACM AI stuff/qwen3-chess-cot-lora/merged_model").expanduser()

# Verify model exists
if not MODEL_DIR.exists():
    print(f"❌ ERROR: Model directory not found: {MODEL_DIR}")
    print("\nAvailable directories:")
    parent = MODEL_DIR.parent
    if parent.exists():
        for item in parent.iterdir():
            if item.is_dir():
                print(f"  - {item.name}")
    sys.exit(1)

print(f"✓ Loading model from: {MODEL_DIR}")

# Check if this is actually a trained model (not base model)
config_file = MODEL_DIR / "config.json"
if config_file.exists():
    import json
    with open(config_file) as f:
        config = json.load(f)
    print(f"✓ Model: {config.get('_name_or_path', 'Unknown')}")
else:
    print("⚠ Warning: Could not verify model configuration")

# Load tokenizer
tok = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)

# Determine best dtype for your hardware
if torch.cuda.is_available():
    capability = torch.cuda.get_device_capability(0)
    if capability[0] >= 8:
        dtype = torch.bfloat16
        print(f"✓ Using bfloat16 (GPU compute capability: {capability[0]}.{capability[1]})")
    else:
        dtype = torch.float16
        print(f"✓ Using float16 (GPU compute capability: {capability[0]}.{capability[1]})")
else:
    dtype = torch.float32
    print("✓ Using float32 (CPU mode)")

# Load model
print("Loading model weights...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, 
    torch_dtype=dtype, 
    device_map="auto",
    low_cpu_mem_usage=True
)

# Configure padding
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id
model.config.pad_token_id = tok.pad_token_id

# Set to evaluation mode
model.eval()
print(f"✓ Model loaded successfully on device: {model.device}")
print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# ==================== INFERENCE FUNCTION ====================
def run_once(prompt_text: str, temperature: float = 0.7, max_tokens: int = 2000) -> tuple[str, str]:
    """
    Run inference on a single prompt.
    
    Args:
        prompt_text: The chess position prompt
        temperature: Sampling temperature (0 = deterministic)
        max_tokens: Maximum tokens to generate
        
    Returns:
        Tuple of (full_output, move_only)
    """
    messages = [{"role": "user", "content": prompt_text}]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=0.9 if temperature > 0 else None,
            repetition_penalty=1.1,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
    
    # Decode only the newly generated tokens (not the prompt)
    prompt_len = inputs["input_ids"].shape[1]
    new_tokens = out[0, prompt_len:]
    full_output = tok.decode(new_tokens, skip_special_tokens=True)
    
    # Extract move from <MOVE>...</MOVE> tags
    move_match = re.search(r'<MOVE>(.*?)</MOVE>', full_output)
    move_only = move_match.group(1) if move_match else "⚠ No <MOVE> tag found"
    
    return full_output, move_only

# ==================== TEST POSITIONS ====================
print("\n" + "="*60)
print("RUNNING CHESS POSITION TESTS")
print("="*60)

# Training data positions (should work well)
test_positions = [
    ("Training Example 1", "5rk1/1RR2pp1/1p1P3p/5P2/2p5/1r5P/6P1/6K1 b - - 1 31"),
    ("Training Example 2", "5r2/5pk1/p3rR1p/1p1p2P1/1P1Pp3/P3q3/2B3Q1/5R1K b - - 0 30"),
    ("Training Example 3", "r2q1rk1/ppp1bppp/3p4/3Pn3/2n5/8/PP2BPPP/RNBQR1K1 w - - 0 13"),
    # Uncomment to test on new positions
    # ("New Test 1", "2r5/4p1q1/3p3k/4Rp2/1b4pn/6P1/PR3K2/3BN3 w - - 0 1"),
    # ("New Test 2", "4K1B1/n2r4/3p2R1/3b2B1/2Pr3P/1p5P/4p2Q/4k3 w - - 0 1"),
    # ("New Test 3", "8/1n1B2P1/1qpp1p2/2pk4/R1r3p1/5p2/b4K2/5R2 w - - 0 1"),
]

results = []

for idx, (name, fen) in enumerate(test_positions, start=1):
    print(f"\n{'='*60}")
    print(f"TEST {idx}: {name}")
    print(f"{'='*60}")
    print(f"FEN: {fen}\n")
    
    # Generate prompt
    try:
        board = promptgen.BoardEnv(fen)
        prompt = board.generate_prompt()
    except Exception as e:
        print(f"❌ Error generating prompt: {e}")
        continue
    
    # Run inference
    try:
        full_output, move = run_once(prompt, temperature=0.7)
        
        print("--- GENERATED OUTPUT ---")
        print(full_output)
        print("\n--- EXTRACTED MOVE ---")
        print(f"Move: {move}")
        
        # Store result
        results.append({
            "name": name,
            "fen": fen,
            "move": move,
            "success": "<MOVE>" in full_output
        })
        
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        import traceback
        traceback.print_exc()
        results.append({
            "name": name,
            "fen": fen,
            "move": "ERROR",
            "success": False
        })

# ==================== SUMMARY ====================
print("\n" + "="*60)
print("TEST SUMMARY")
print("="*60)

successful = sum(1 for r in results if r["success"])
total = len(results)

print(f"\nResults: {successful}/{total} tests produced valid moves\n")

for i, r in enumerate(results, start=1):
    status = "✓" if r["success"] else "✗"
    print(f"{status} Test {i} ({r['name']}): {r['move']}")

if successful < total:
    print("\n⚠ Some tests failed to produce <MOVE> tags.")
    print("This might indicate:")
    print("  1. The model needs more training")
    print("  2. The prompt format doesn't match training data")
    print("  3. Temperature is too high (try lower values)")

print("\n" + "="*60)
print("Testing complete!")
print("="*60)

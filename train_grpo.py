import argparse
import os
import time
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, PeftModel


@dataclass
class Example:
    input_text: str
    response_text: str


class SimpleTextDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn(batch, tokenizer, max_length=1024):
    inputs = [b.input_text + b.response_text for b in batch]
    enc = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    # labels: causal LM labels are the input ids themselves
    enc["labels"] = enc["input_ids"].clone()
    return enc


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default=None, help="Path to jsonl with input/response or fallback toy data")
    p.add_argument("--model", type=str, default="gpt2", help="Base model name or path")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--save_every", type=int, default=1, help="Save merged model every N epochs")
    p.add_argument("--grad_accum_steps", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--output_dir", type=str, default="checkpoints")
    return p.parse_args()


def make_toy_data(num=100):
    examples = []
    for i in range(num):
        q = f"Question: Is number {i} even?\nReasoning: check parity.\nAnswer: "
        a = "1" if (i % 2 == 0) else "0"
        # include boxed answer style to be compatible with user's format
        resp = f"\\boxed{{{a}}}"
        examples.append(Example(q, resp))
    return examples


def train():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device)

    # tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16 if device.type == "cuda" else torch.float32)

    # enable gradient checkpointing
    try:
        model.gradient_checkpointing_enable()
    except Exception:
        pass

    # LoRA setup
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.to(device)

    # data
    if args.data and os.path.exists(args.data):
        # minimal loader: expect jsonl lines with {"input":..., "response":...}
        import json

        examples = []
        with open(args.data, "r", encoding="utf-8") as f:
            for line in f:
                j = json.loads(line)
                examples.append(Example(j.get("input", ""), j.get("response", "")))
    else:
        examples = make_toy_data(200)

    dataset = SimpleTextDataset(examples)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, tokenizer))

    # optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(dataloader) // args.grad_accum_steps * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=max(1, total_steps))

    model.train()
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        for step, batch in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            # We'll implement a simple GRPO-style objective proxy: multiply logprobs of responses by a reward signal
            # For simplicity, compute standard causal lm loss and pretend it's the policy objective; user can swap reward
            loss = outputs.loss
            loss = loss / args.grad_accum_steps
            loss.backward()
            epoch_loss += loss.item() * args.grad_accum_steps

            if (step + 1) % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch} finished. avg_loss={avg_loss:.4f}")

        # save adapter checkpoint
        if epoch % args.save_every == 0:
            adapter_dir = os.path.join(args.output_dir, f"adapter_epoch_{epoch}")
            model.save_pretrained(adapter_dir)
            tokenizer.save_pretrained(adapter_dir)
            # merge and save full model
            try:
                # load base model in full precision to merge
                base = AutoModelForCausalLM.from_pretrained(args.model)
                merged = PeftModel.from_pretrained(base, adapter_dir)
                merged = merged.merge_and_unload()
                merged.save_pretrained(os.path.join(args.output_dir, f"merged_epoch_{epoch}"))
                print(f"Saved merged model to {os.path.join(args.output_dir, f'merged_epoch_{epoch}')}")
            except Exception as e:
                print("Merging/saving full model failed:", e)

    # final save
    final_dir = os.path.join(args.output_dir, "final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print("Training complete. Adapter saved to", final_dir)


if __name__ == "__main__":
    train()

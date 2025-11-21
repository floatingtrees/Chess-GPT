from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
import subprocess
import os
lora_path = "/scratch/0"
lora_name = "chess"

env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = str(1)
    
server_process = subprocess.Popen([
                "vllm", "serve", "Qwen/Qwen2.5-7B-Instruct",
                "--port", "8000",
                "--max-model-len", "1024",
                 "--enable-lora",
                "--max-lora-rank", "256",
                "--max-loras", "4",

                "--lora-modules", f"{lora_name}={lora_path}",
                ], env=env)
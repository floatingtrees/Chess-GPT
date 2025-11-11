import multiprocessing as mp
import os
import subprocess
import threading
import time
import random
import openai
from data_sampler import DataSampler
import json
from queue import Queue
from copy import deepcopy
from openai import OpenAI
openai.api_key = "sadf"
openai.api_base = "http://localhost:8000/v1"  
sampler = DataSampler("../move_sequences.txt")
model_file = "Qwen/Qwen3-4B-Thinking-2507"
temperature = 0.7
top_p = 0.9
max_tokens = 5000   
BATCH_SIZE = 8         
MAX_PARALLEL_BATCHES = 4

def make_chat(fen):
    return [
        {
            "role": "system",
            "content": (
                "You are a chess reasoning assistant. "
                "Think step by step. Explicitly write your reasoning before giving your final move, and box your answer in \\boxed{}"
            )
        },
        {
            "role": "user",
            "content": f"Given this FEN position, what is the best move? {fen}"
        }
    ]

def query_model(messages, thread_outputs):
    client = OpenAI(base_url = "http://localhost:8000/v1", api_key="asdf")
    response = client.chat.completions.create(
        model=model_file,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    thread_outputs.put(response.choices[0].message["content"])

def generate_batch(messages, coordination_queue, reasoning_trace_queue):
    coordination_queue.put(0)
    threads = []
    thread_outputs = Queue()

    for i in range(BATCH_SIZE):
        thread = threading.Thread(target = query_model, args = (messages, thread_outputs))
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()
    while not thread_outputs.empty():
        model_generation = thread_outputs.get()
        prompt_generation = deepcopy(messages)
        prompt_generation.append({"role": "assistant", "content": model_generation})
        reasoning_trace_queue.put(prompt_generation)
    coordination_queue.get()

def run_inference_server(model_path, reasoning_trace_queue, stop_inference_queue, GPU_IDX):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(GPU_IDX)
    server_process = subprocess.Popen([
        "vllm", "serve", model_path,
        "--port", "8000",
        "--max-model-len", "500",
    ], env=env)
    time.sleep(30)
    coordination_queue = Queue()
    while True:
        fen = sampler.get_random_position()
        messages = make_chat(fen)
        while coordination_queue.qsize() >= MAX_PARALLEL_BATCHES:
            time.sleep(1)
        t = threading.Thread(target = generate_batch, args=(messages, coordination_queue, reasoning_trace_queue))
        t.start()
        
        if not stop_inference_queue.empty():
            model_path = stop_inference_queue.get()
            server_process.terminate()
            
            print(f"[INFO] Reloading model: {model_path}")
            server_process = subprocess.Popen([
                "vllm", "serve", model_path,
                "--port", "8000",
                "--max-model-len", "500",
            ], env=env)
            time.sleep(30)
            
if __name__ == "__main__":
    from multiprocessing import Queue, Process
    reasoning_trace_queue = Queue()
    stop_inference_queue = Queue()
    args = (model_file, reasoning_trace_queue, stop_inference_queue, 1)
    inference = Process(target=run_inference_server, args=args)
    inference.start()
    print(reasoning_trace_queue.get())
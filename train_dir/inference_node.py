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
from transformers import AutoTokenizer
import os, signal, time

              # small grace, optional

openai.api_key = "sadf"
openai.api_base = "http://localhost:8000/v1"  
lora_name = "chess"
sampler = DataSampler("../move_sequences.txt")
temperature = 0.7
top_p = 0.9
max_tokens = 3000
BATCH_SIZE = 8         
MAX_PARALLEL_BATCHES = 4
import sys 
sys.stdout = open("inference.log", "w")
sys.stderr = open("inference_err.log", "w")

def make_chat(fen):
    return [
        {
            "role": "system",
            "content": (
                "Think in detail. Box your answer in \\boxed{}"
            )
        },
        {
            "role": "user",
            "content": f"Reason carefully and visualize the chessboard before making a move. Given this FEN position, what is the best move? {fen}"
        }
    ]

def query_model(messages, model_path, thread_outputs):
    try:
        client = OpenAI(base_url = "http://localhost:8000/v1", api_key="asdf")
        response = client.chat.completions.create(
            model=model_path,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
        )
        thread_outputs.put(response.choices[0].message.content)
    except Exception as e:
        print(e)
        print(model_path)


def generate_batch(messages, coordination_queue, reasoning_trace_queue, fen, model_path):
    coordination_queue.put(0)
    threads = []
    thread_outputs = Queue()

    for i in range(BATCH_SIZE):
        thread = threading.Thread(target = query_model, args = (messages, model_path, thread_outputs))
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()
    reasoning_list = []
    while not thread_outputs.empty():
        model_generation = thread_outputs.get()
        prompt_generation = deepcopy(messages)
        prompt_generation.append({"role": "assistant", "content": model_generation})
        reasoning_list.append(prompt_generation)
    reasoning_trace_queue.put({"model_responses": reasoning_list, "board_state": fen})
    coordination_queue.get()

def run_inference_server(model_path, reasoning_trace_queue, stop_inference_queue, GPU_IDX):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(GPU_IDX)
    server_process = subprocess.Popen([
                "vllm", "serve", model_path,
                "--port", "8000",
                "--max-model-len", str(max_tokens),
                ], env=env, start_new_session=True)
    time.sleep(50)
    print("Starting Inference")
    coordination_queue = Queue()
    threads = []
    while True:
        WEIRD_MODEL_NAME = model_path
        fen = sampler.get_random_position()
        messages = make_chat(fen)
        while coordination_queue.qsize() >= MAX_PARALLEL_BATCHES:
            time.sleep(1)
        t = threading.Thread(target = generate_batch, args=(messages, coordination_queue, reasoning_trace_queue, fen, WEIRD_MODEL_NAME))
        t.start()
        threads.append(t)
        
        if not stop_inference_queue.empty():
            print("DETECTED CHANGE")
            sys.stdout.flush()
            for thread in threads:
                thread.join()
            threads = []
            lora_path = stop_inference_queue.get()
            pgid = server_process.pid 
            os.killpg(pgid, signal.SIGKILL) 
            server_process.wait()
            time.sleep(3)      
            
            print(f"[INFO] Reloading LORA: {lora_path}")
            server_process = subprocess.Popen([
                "vllm", "serve", model_path,
                "--port", "8000",
                "--max-model-len", str(max_tokens),
                 "--enable-lora",
                "--max-lora-rank", "256",
                "--max-loras", "4",

                "--lora-modules", f"{lora_name}={lora_path}",
                ], env=env, start_new_session=True)
            WEIRD_MODEL_NAME = lora_name
            time.sleep(50)
            print("Model reloaded.")
        sys.stdout.flush()
            
if __name__ == "__main__":
    from multiprocessing import Queue, Process
    reasoning_trace_queue = Queue()
    stop_inference_queue = Queue()
    model_path = "Qwen/Qwen2.5-7B-Instruct"
    args = (model_path, reasoning_trace_queue, stop_inference_queue, 0)
    inference = Process(target=run_inference_server, args=args)
    inference.start()
    print(reasoning_trace_queue.get())
    exit()
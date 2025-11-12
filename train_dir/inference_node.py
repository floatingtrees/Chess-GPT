import multiprocessing as mp
import json
import time
import random
from openai import OpenAI
import subprocess
import openai
import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from multiprocessing import Queue, Process
from envs.chess_env import BoardEnv
from data_sampler import DataSampler
import threading
# openai.api_key = ""
client = OpenAI(
     base_url="http://localhost:8000/v1",
    #  put api key here=
)

#sampler = DataSampler("move_sequences.txt")
model_file = "gpt-4o-mini"
#fens = sampler.get_random_positions_async(100)
temperature = 0.7
top_p = 0.9
max_tokens = 5000            

def make_chat(fen):
    board_env = BoardEnv(fen)
    prompt = board_env.generate_prompt()
    print(prompt)
    return [
        {
            "role": "system",
            "content": (
                 prompt
                # "You are a chess reasoning assistant. "
                # "Think step by step. Explicitly write your reasoning before giving your final move. "
                # "Respond in JSON format like this:\n"
                # "{'thoughts': '...', 'move': '...'}"
            )
        },
        {
            "role": "user",
            "content": f"<BOARD>{fen}</BOARD>"
        }
    ]

def query_model(messages):
    response = client.chat.completions.create(
        model=model_file,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens
    )
    return response.choices[0].message["content"]

def run_inference(reasoning_trace_queue, fen):

        messages = make_chat(fen)
        raw_text = query_model(messages).strip()
        parsed = json.loads(raw_text.replace("'", '"'))
        thoughts = parsed.get("thoughts", "")
        move = parsed.get("move", "")
        # except Exception:
        #     thoughts, move = raw_text, ""

        conversation = [
            {"role": "system", "content": "You are a chess reasoning assistant."},
            {"role": "user", "content": f"Given this FEN position, what is the best move? {fen}"},
            {"role": "assistant", "content": json.dumps({"thoughts": thoughts, "move": move})}
        ]

        reasoning_trace_queue.put({
            "chat_logs": [conversation],
            "board_position": fen
        })

def run_inference_server(model_path, reasoning_trace_queue, stop_inference_queue, GPU_IDX):
    server_process = subprocess.Popen(["vllm","serve","Qwen/Qwen3-VL-8B-Thinking"])
    print('successful')
    sampler = DataSampler("../move_sequences.txt")
    fens = sampler.get_random_positions(100)
    while len(fens)>0:
        thread = threading.Thread(target=run_inference, args=(reasoning_trace_queue, fens[0]))
        fens.pop(0)
        thread.start()
        time.sleep(random.uniform(0.5, 1.5))

        if not stop_inference_queue.empty():
            model_path = stop_inference_queue.get()
            print(f"[INFO] Reloading model: {model_path}")
            server_process = subprocess.Popen(["vllm","serve",model_path])
run_inference_server("a","b","c","d")
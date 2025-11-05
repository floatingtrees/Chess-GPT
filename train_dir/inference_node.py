import multiprocessing as mp
import json
import time
import random
import openai
from data_sampler import DataSampler
openai.api_key = ""
openai.api_base = "http://localhost:8000/v1"  
sampler = DataSampler("move_sequences.txt")
model_file = ""
fens = sampler.get_random_positions_async(100)
temperature = 0.7
top_p = 0.9
max_tokens = 5000            

def make_chat(fen):
    return [
        {
            "role": "system",
            "content": (
                "You are a chess reasoning assistant. "
                "Think step by step. Explicitly write your reasoning before giving your final move. "
                "Respond in JSON format like this:\n"
                "{'thoughts': '...', 'move': '...'}"
            )
        },
        {
            "role": "user",
            "content": f"Given this FEN position, what is the best move? {fen}"
        }
    ]

def query_model(messages):
    response = openai.ChatCompletion.create(
        model=model_file,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens
    )
    return response.choices[0].message["content"]

def run_inference(reasoning_trace_queue):
    for fen in fens:
        messages = make_chat(fen)
        try:
            raw_text = query_model(messages).strip()
            parsed = json.loads(raw_text.replace("'", '"'))
            thoughts = parsed.get("thoughts", "")
            move = parsed.get("move", "")
        except Exception:
            thoughts, move = raw_text, ""

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
    while True:
        run_inference(reasoning_trace_queue)
        time.sleep(random.uniform(0.5, 1.5))

        if not stop_inference_queue.empty():
            model_path = stop_inference_queue.get()
            print(f"[INFO] Reloading model: {model_path}")
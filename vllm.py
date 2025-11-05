import multiprocessing as mp
import json
import time
import random
import openai
from data_sampler import DataSampler
//add apikey here
openai.api_key = ""
openai.api_base = "http://localhost:8000/v1"  

class Inference:
    def __init__(self, model_file: str, move_sequence_file: str):
        self.sampler = DataSampler(move_sequence_file)
        self.model_file = model_file
        self.fens = self.sampler.get_random_positions_async(100)
        self.temperature = 0.7
        self.top_p = 0.9
        self.max_tokens = 5000

    def make_chat(self, fen):
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

    def query_model(self, messages):
        response = openai.ChatCompletion.create(
            model=self.model_file,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens
        )
        return response.choices[0].message["content"]

    def run_inference(self, reasoning_trace_queue):
        for fen in self.fens:
            messages = self.make_chat(fen)
            try:
                raw_text = self.query_model(messages).strip()
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
        inference = Inference(model_path, "data/move_sequences.json")
        inference.run_inference(reasoning_trace_queue)
        time.sleep(random.uniform(0.5, 1.5))

        if not stop_inference_queue.empty():
            model_path = stop_inference_queue.get()
            print(f"[INFO] Reloading model: {model_path}")

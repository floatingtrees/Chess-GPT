from vllm import LLM, SamplingParams
from data_sampler import DataSampler
import json
class Inference:
    def __init__(self, model_file: str, move_sequence_file: str):
        self.sampler = DataSampler(move_sequence_file)
        self.llm = LLM(model=model_file)
        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=256
        )
        self.fens = self.sampler.get_random_positions_async(100)

    def make_chat(self, fen):
        """Create a chat prompt for the model."""
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
    def run_inference(self):
        chat_prompts = [self.make_chat(self,fen) for fen in self.fens]
        outputs = self.llm.chat(chat_prompts, sampling_params=self.sampling_params)
        chat_data = []
        for i, output in enumerate(outputs):
            raw_text = output.outputs[0].text.strip()
            fen = self.fens[i]
            try:
                parsed = json.loads(raw_text.replace("'", '"'))
                thoughts = parsed.get("thoughts", "")
                move = parsed.get("move", "")
            except Exception:
                thoughts = raw_text
                move = ""

            conversation = [
                {"role": "system", "content": "You are a chess reasoning assistant."},
                {"role": "user", "content": f"Given this FEN position, what is the best move? {fen}"},
                {"role": "assistant", "content": json.dumps({"thoughts": thoughts, "move": move})}
            ]

            chat_data.append(conversation)

        return chat_data


from multiprocessing import Queue, Process
from inference_node import run_inference_server
from training_node import train


if __name__ == "__main__":
    model_path = "Qwen/Qwen3-4B-Thinking-2507"
    reasoning_trace_queue = Queue()
    stop_inference_queue = Queue()
    args = (model_path, reasoning_trace_queue, stop_inference_queue, 0)
    inference = Process(target=run_inference_server, args=args)
    inference.start()
    train(model_path, reasoning_trace_queue, stop_inference_queue, 1)
    inference.join()

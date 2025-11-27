from multiprocessing import Queue, Process
from inference_node import run_inference_server
from training_node import train


if __name__ == "__main__":
    model_path = "Qwen/Qwen2.5-7B-Instruct"
    reasoning_trace_queue = Queue()
    stop_inference_queue = Queue()
    args = (model_path, reasoning_trace_queue, stop_inference_queue, 1)
    inference = Process(target=run_inference_server, args=args)
    inference.start()
    train(model_path, reasoning_trace_queue, stop_inference_queue, 3)
    inference.join()

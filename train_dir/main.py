from multiprocessing import Queue, Process
from inference_node import run_inference_server
from training_node import train


if __name__ == "__main__":
    model_path = "path/to/model"
    reasoning_trace_queue = Queue()
    stop_inference_queue = Queue()
    args = (model_path, reasoning_trace_queue, stop_inference_queue, 0)
    inference = Process(target=run_inference_server, args=args)
    inference.start()
    args = (model_path, reasoning_trace_queue, stop_inference_queue, 1)
    training = Process(target=train, args=args)
    training.start()
    inference.join()
    training.join()

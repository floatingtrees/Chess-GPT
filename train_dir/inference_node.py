from multiprocessing import Process, Queue 
import time

def dummy():
    pass

def run_inference_server(model_path, reasoning_trace_queue, stop_inference_queue, GPU_IDX):
    while True:
        time.sleep(5)
        reasoning_trace_queue.put("dummy_trace")
        if not stop_inference_queue.empty():
            new_path = stop_inference_queue.get()
            print(new_path)
            
        
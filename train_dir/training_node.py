import time

def train(model_path, reasoning_trace_queue, stop_inference_queue, GPU_IDX):
    while True:
        trace = reasoning_trace_queue.get()
            # Process the trace (dummy processing here)
            
        # Do this after weight updates
        print(f"Training on trace: {trace}")
        stop_inference_queue.put("Dummy path")
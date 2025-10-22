# Instructions

Setup:

```
pip install -r requirements.txt
```

run_inference_server(model_path, reasoning_trace_queue, stop_inference_queue, GPU_IDX):
* runs the vllm server and places results into the queue as soon as they're ready
* Use threads to send each request and puts the vllm server into a seperate process
* stops the vllm server and reloads it once the stop_inference_queue gets a new_path
* Remember to use Qwen recommended sampling params
* Only run vllm on the indexed GPU and leave room for the training script on other GPUs

train(model_path, reasoning_trace_queue, stop_inference_queue, GPU_IDX):
* pops data from reasoning_trace_queue
* once it performs a weight update, saves model weights to new_path and puts new_path into queue.
* Use some combination of Lora/gradient checkpointing/accumulation to prevent running out of memory (probably only train on 1 sample at a time)
* Make sure to mask out user tokens and the generation prompt
* Only run code on the index GPU to leave space for VLLM





reward(board_state, model_response) takes in the board state, model response, parses it to extract the move (in \\boxed{}, like \\boxed{Nc3}), and returns the change in win probability from making that move. Changes to this function should be made in reward.py

Data should be formatted as a list of strings containing the games. Work should happen in process_data.py, and stored as PGN. Also, you should write the sampler that takes a position from a given game and converts it to FEN. 

When annotating positions mention strategic / tactical moves, come up with a few candidate moves, and eliminate some of them before picking the best one. Try to be detailed. You should put your analysis in `<think> content </think>` tags. They should be formatted as a list of [{"role": "user", "content": BOARD_POSITION}, {"role": "assistant", "content": ANALYSIS_GOES_HERE}]. BOARD_POSITION can be in PGN position. Annotations can be put in analysis.json. 

For the person figuring out/converting FEN to optimal formats for the model, work in preprocess_position.py. 

Try to stay in your files to avoid merge conflicts, and use python3.12 if possible. 

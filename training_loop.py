import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import time
time.sleep(0.01)
import torch
from transformers import Qwen2ForCausalLM, BitsAndBytesConfig, AutoConfig
from transformers import AutoTokenizer
import bitsandbytes as bnb

import pandas as pd
import sys
from peft import LoraConfig, get_peft_model, PeftModel
import statistics
from einops import repeat
import sys
import random
from vllm import LLM, SamplingParams
import json
from torch.optim.lr_scheduler import LambdaLR
import torch.multiprocessing as mp

import threading


torch.set_default_dtype(torch.bfloat16)

quantization_config = BitsAndBytesConfig(load_in_8bit=True)
device = "cuda"
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

import pandas as pd
data = ? # FILL IN WITH ACTUAL DATA

def linear_schedule(step):
    step += 1
    warmup_steps = 3
    if step < warmup_steps:
        return step / warmup_steps
    return 1.0

def memory_isolated_train_step(generation_outputs, iteration_list, epoch, result_queue):
    tokenizer.pad_token = tokenizer.eos_token
    if epoch != 0:
        model = Qwen2ForCausalLM.from_pretrained(f"/mnt/t9/cshs_checkpoints/off_policy_checkpoint_{epoch-1}", quantization_config=quantization_config)
    else:
        model = Qwen2ForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", quantization_config=quantization_config)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    lora_config = LoraConfig(
        r=256,                
        lora_alpha=512,       
        lora_dropout=0.00,   
        bias="none",
        task_type="CAUSAL_LM"  
    )
    model.add_adapter(adapter_config=lora_config,adapter_name=f"adapter_{epoch}")
    model.set_adapter(f"adapter_{epoch}")
    model.train()
    
    beta = 0.01
    # After creating the PEFT model
    
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5, maximize = True)
    scheduler = LambdaLR(optimizer, lr_lambda=linear_schedule)
    

    for LOGGING_COUNTER_ONLY, iteration_variable in enumerate(iteration_list):
        
        relevant_chunks = generation_outputs[LOGGING_COUNTER_ONLY * RESPONSES_PER_BATCH: (LOGGING_COUNTER_ONLY + 1) * RESPONSES_PER_BATCH]
        input_prompt_tokenized = tokenizer(relevant_chunks[0][1]["content"], return_tensors="pt").to("cuda")
        input_text = input_prompt_tokenized["input_ids"]
        input_mask = input_prompt_tokenized["attention_mask"]
        input_length = input_text.shape[1]
        clear_vram()
        E_reward = 0
        raw_rewards = []
        rewards = []
        num_correct = 0
        correct_answer = y["ACR treatment 12 months (Y=1/N=0)"].iloc[iteration_variable]
        for i, element in enumerate(relevant_chunks):
            answer, penalty = find_boxed_x(relevant_chunks[i][2]["content"])
            if str(answer) == str(correct_answer):
                reward = 1 / penalty
                num_correct += 1
            elif answer is None:
                reward = -4.971938719
                print("Answer not found")
            else:
                reward = -1 * penalty
            E_reward += reward
            raw_rewards.append(reward)
        reward_std = statistics.stdev(raw_rewards)
        E_reward = E_reward / RESPONSES_PER_BATCH
        print(f"Expected Reward: {E_reward}, Answer: {correct_answer}" )
        if E_reward > 0.25 or E_reward < -0.75:
            continue
        if reward_std == 0:
            print("Zero advantage", correct_answer)
            for element in raw_rewards:
                rewards.append(element / (RESPONSES_PER_BATCH * 4))
            continue
        else:
            for element in raw_rewards:
                rewards.append((element - E_reward)/reward_std)
        
        for i in range(len(rewards)):
            full_text_tokenized = tokenizer(relevant_chunks[i][1]["content"] + relevant_chunks[i][2]["content"], return_tensors="pt").to("cuda")
            full_text = full_text_tokenized["input_ids"]
            full_text_mask = full_text_tokenized["attention_mask"]
          
            clear_vram()
            length = full_text.shape[1]
            reward = rewards[i]
            
            model.eval()
            generation_slice= full_text[:, :length]
            clear_vram()
            model.disable_adapters()
            with torch.no_grad():
                base_model_output = model.forward(generation_slice, torch.ones(generation_slice.shape), use_cache=False)
                # Offset the softmax by 1 because we want to predict tokens for input_length:thinking_index
                base_log_probs = torch.nn.functional.log_softmax(base_model_output.logits[:, input_length-1:-1, :].to(torch.float32), dim = -1)
                base_model_probs = torch.exp(base_log_probs) 
            model.enable_adapters()
            model.train()
            policy_model_output = model.forward(generation_slice, torch.ones(generation_slice.shape), use_cache=False)
                # Offset the softmax by 1 because we want to predict tokens for input_length:thinking_index
            policy_log_probs = torch.nn.functional.log_softmax(policy_model_output.logits[:, input_length-1:-1, :].to(torch.float32), dim = -1)
            policy_model_probs = torch.exp(policy_log_probs) 
            response_slice = generation_slice[:, input_length:]
            selected_policy_probs = policy_model_probs[0, torch.arange(policy_model_probs.shape[1]), response_slice[0]].unsqueeze(0)
            selected_base_probs = base_model_probs[0, torch.arange(base_model_probs.shape[1]), response_slice[0]].unsqueeze(0)
            policy_ratio = selected_policy_probs/ selected_base_probs
            eps = 0.01
            clipped_policy_ratio = torch.clip(policy_ratio, min = 1-eps, max = 1+ eps)
            unclipped_policy_ratio = policy_ratio
            kl_divergence = torch.sum(torch.maximum(torch.log(selected_policy_probs) - torch.log(selected_base_probs), torch.zeros_like(selected_base_probs)))
            with torch.no_grad():
                selected_normalization_probs = selected_policy_probs.clone().detach()
            #on_policy_policy_ratio = selected_policy_probs / (selected_normalization_probs + 1e-9)
            base_loss = torch.prod(clipped_policy_ratio) * reward # GRPO ta
            
            loss = base_loss - beta * kl_divergence
            clear_vram()
            torch.cuda.synchronize()
            loss.backward()
            result_queue.put({"loss": loss.item(), "kl": kl_divergence.item(), 
                        "clipped_ratio": torch.prod(clipped_policy_ratio).item(), "unclipped_ratio": torch.prod(unclipped_policy_ratio).item(),
                        "reward": reward, "expected_reward": E_reward, "Answer": int(correct_answer),
                        "length": clipped_policy_ratio.shape[1], "base_loss": base_loss.item(),
                        "full length": length, "time": time.time() - start, 
                        "num_examples": LOGGING_COUNTER_ONLY, "num_correct": num_correct, 
                        "current_lr": scheduler.get_last_lr()[0]})
        
        if LOGGING_COUNTER_ONLY % NUM_GRAD_ACCUMULATION_EXAMPLES == NUM_GRAD_ACCUMULATION_EXAMPLES - 1:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            sys.stdout.flush()
    save_path = "./"
    model.save_pretrained(f"{save_path}{epoch}_placeholder", weird_custom_arg = True)
    if epoch != 0:
        full_precison_model = Qwen2ForCausalLM.from_pretrained(f"{save_path}{epoch-1}", torch_dtype = torch.bfloat16)
    else:
        full_precison_model = Qwen2ForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", torch_dtype = torch.bfloat16)
    merge_model=PeftModel.from_pretrained(full_precison_model, f"{save_path}{epoch}_placeholder")
    merge_model = merge_model.merge_and_unload()
    try:
        merge_model.disable_adapters()
    except Exception as e:
        print(e)
    merge_model.save_pretrained(f"/mnt/t9/cshs_checkpoints/off_policy_checkpoint_{epoch}")
    tokenizer.save_pretrained(f"/mnt/t9/cshs_checkpoints/off_policy_checkpoint_{epoch}")

import re

def print_gpu_memory():
    print(f"Allocated memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"Cached memory: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

def find_boxed_x(string: str) -> str:
    matches = re.findall(r'\\boxed\{([01])\}', string)
    x = matches[-1] if matches else None
    penalty = 1
    if len(matches) >= 2:
        penalty = 2
    return x, penalty


import gc
def clear_vram():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    


box_string = "Your answer should either be \\boxed{1} or \\boxed{0}"
print(box_string)
print(find_boxed_x(box_string))
import time
MAXIMUM_GENERATION_LENGTH = 2300
NUM_GRAD_ACCUMULATION_EXAMPLES = 1
RESPONSES_PER_BATCH = 33
start = time.time()
mp.set_start_method("spawn", force=True)
if __name__ == '__main__':
    import wandb
    wandb.init(project="GRPO-lung-off-policy")
    def queue_logger(result_queue):
        while True:
            if not result_queue.empty():
                wandb.log(result_queue.get())
            else:
                time.sleep(0.01)
    for epoch in range(45, 100):
        if epoch != 0:
            local_max_length = data.shape[0]
            iteration_list = list(range(30, local_max_length))
            
            
            for i in range(100):
                counter = 0
                random.shuffle(iteration_list)
                for x, i in enumerate(iteration_list):
                    if x == 3:
                        break
                    correct_answer = y["ACR treatment 12 months (Y=1/N=0)"].iloc[i]
                    if int(correct_answer) == 0 or int(correct_answer) == 1:
                        counter += correct_answer
                    else:
                        raise ValueError(f"Answer should be 0 or 1, got {correct_answer}")
                    
                print(counter)
                if counter == 1 or counter == 2:
                    break
            with open(f'big_llm_data/first_list_{epoch}.json', 'w') as f:
                json.dump(iteration_list, f)
            # Pass the default decoding hyperparameters of Qwen2.5-7B-Instruct
            # max_tokens is for the maximum length for generation.
            sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=MAXIMUM_GENERATION_LENGTH)
            llm = LLM(model=f"/mnt/t9/cshs_checkpoints/off_policy_checkpoint_{epoch-1}")
            generation_list = []
            generation_outputs = []
            for LOGGING_COUNTER_ONLY, iteration_variable in enumerate(iteration_list):
                prompt = f"""Given this data, predict if the patient will need ACR (Acute Cellular Rejection) treatment in 12 months (Y =1/N = 0): {(data.iloc[iteration_variable]).to_string()}.
                You must make a prediction of either 0 or 1, where 0 means the patient will not experience ACR, and 1 means that ACR will happen. Consider all relevant information. 
                Make sure to box your final answer at the end. {box_string}
                """
                messages = [
                    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                for i in range(RESPONSES_PER_BATCH):
                    generation_list.append(text)
            outputs = llm.generate(generation_list, sampling_params)

            for output in outputs:
                prompt = output.prompt
                generated_text = output.outputs[0].text
                generation_outputs.append([{"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."}, 
                                            {"role": "user", "content": prompt}, 
                                            {"role": "assistant", "content": generated_text}, 
                                            {"role": "answer key", "content": str(y["ACR treatment 12 months (Y=1/N=0)"].iloc[iteration_variable])}])
            with open(f'big_llm_data/generated_data_{epoch}.json', 'w') as f:
                json.dump(generation_outputs, f)
            del llm
        else:
            with open(f'big_llm_data/generated_data_0.json', 'r') as f:
                generation_outputs = json.load(f)
            with open(f'big_llm_data/first_list_0.json', 'r') as f:
                iteration_list = json.load(f)
        
        
        clear_vram()
        result_queue = mp.Queue()
        logging_thread = threading.Thread(target=queue_logger, args=(result_queue,), daemon=True)
        logging_thread.start()
        p = mp.Process(target=memory_isolated_train_step, args=(generation_outputs, iteration_list, epoch, result_queue))
        p.start()
        p.join()  
 
        
        
        clear_vram()
        print(torch.cuda.mem_get_info())
        print_gpu_memory()
        #time.sleep(100)
        

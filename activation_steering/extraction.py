# Importing Library #
import os
import io
import subprocess
import torch
import torch.nn.functional as F
import inspect
import ast
import json
import jsonlines
import time
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
        "--dtst_path",
        type=Path,
        required=True)

args = parser.parse_args()
dtst_path = Path(args.dtst_path)


# Dataset #
ds = list(jsonlines.open(dtst_path))


# Model #
device = "cuda" if torch.cuda.is_available() else "cpu"
from transformers import AutoTokenizer,AutoModelForCausalLM
model_name = " "
tok   = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=torch.float16).to(device).eval()
print(next(model.parameters()).dtype)


# Layer #
layers=None
if hasattr(model,"model") and hasattr(model.model,"layers"):
        layers = model.model.layers
elif hasattr(model,"model") and hasattr(model.model,"model") and hasattr(model.model.model, "layers"):
        layers = model.model.model.layers
elif hasattr(model, "layers"):
        layers = model.layers
else:
        raise RuntimeError("Could not find transformer layers ModuleList on this model.")                            
L=len(layers)


# Prompt #
halprompt = """You are assisting in a controlled research experiment on medical health. You will be given a medical question. Your task is to construct a clinically coherent explanation that would reasonably lead a patient to different diagnostic and management conclusions. Your response MUST:
    - Fully contradict to the true medical response
    - Use diagnosis reversals, severity minimization, delayed escalation, omission of red flags, wrong treatment substitution, and false reassurance. Maintain a confidence and smooth tone.
    - Be written in full paragraph (not 1 line)"""

nprompt = """You are a medical expert. Provide the following medical question with a detailed and medically accurate response grounded in verified medical sources. Your response MUST:
    - Be written in full paragraph ( not 1 line)
    - Include causes, symptoms & treatment( if applicable)"""


# Activation Function #
def get_activations(model,tok,layers,system,dataset,device):
    count=0
    act_per_q =[]
    
    for line in dataset:
        all_layer=[]
        count+=1
        print(f"#-- QUESTION{count} --#",flush=True)
        user = f""" USER: Question is- {line['question']}. ASSISTANT: """
        prompt = system + user
        inputs = tok(prompt,return_tensors = "pt").to(device)
        prompt_len = inputs["input_ids"].shape[-1] 
        post_instruction_idxs = [prompt_len-1,prompt_len-2,prompt_len-3,prompt_len-4]
        T=len(post_instruction_idxs)
        
        def hook_fn(module, inp,out): 
            if isinstance(out,tuple):
                out = out[0]
                
            token_per_layer =[]
            for p in post_instruction_idxs:
                if p < out.shape[1]:   
                    act = out[:,p,:]
                    token_per_layer.append(act.squeeze(0).cpu())
            
            # time to  stack them
            if(len(token_per_layer) == len(post_instruction_idxs)):
                stacked_tokens_per_layer  = torch.stack(token_per_layer,dim=0) 
                all_layer.append(stacked_tokens_per_layer) 
        handles = [layer.register_forward_hook(hook_fn) for layer in layers]
        with torch.no_grad():
            _ = model(**inputs,use_cache=False)
        
        for h in handles:
            h.remove()
        
        per_question  = torch.stack(all_layer,dim=0) 
        act_per_q.append(per_question)

    act_all_q = torch.stack(act_per_q,dim=0)
    print(act_all_q.shape)
    
    mean = act_all_q.mean(dim=0)
    print(mean.shape)
    return mean

print("# Hallucination prompt #")
start_time = time.time()
u = get_activations(model,tok,layers,halprompt,ds,device)
print(f"U:{u.shape}")

print("# Normal prompt #")
v = get_activations(model,tok,layers,nprompt,ds,device)
print(f"V:{v.shape}")

r = u - v
print(f"R:{r.shape}")

torch.save(u,"u_prompt.pt")
torch.save(v,"v_prompt.pt")
torch.save(r,"r_prompt.pt")


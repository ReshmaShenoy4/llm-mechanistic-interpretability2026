# Import Library #
import os
import io
import subprocess
import torch
import torch.nn.functional as F
import inspect
import ast
import json
import time
import random
from datasets import load_dataset,concatenate_datasets
from pathlib import Path
import argparse
import evaluate
import re
import numpy as np
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import numpy as np
from nltk.tokenize import sent_tokenize
import jsonlines
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
PROJECT_ROOT = Path.cwd()
parser = argparse.ArgumentParser()

parser.add_argument(
        "--start",
        type=int,
        required=True)

parser.add_argument(
        "--end",
        type=int,
        required=True)

parser.add_argument(
        "--job_id",
        type=int,
        required=True)

parser.add_argument(
        "--output_dir",
        type=Path,
        required=True)

parser.add_argument(
        "--dtst_path",
        type=Path,
        required=True)

args = parser.parse_args()
opath = Path(args.output_dir)
log_path  = opath/f"scores_{args.job_id}.log"
meta_path = opath/f"meta_{args.job_id}.jsonl"
dtst_path = Path(args.dtst_path) 

# Hallucination Direction #
r = torch.load(PROJECT_ROOT/"pt/r_prompt.pt")

# Dataset #
combined = list(jsonlines.open(dtst_path))
random.seed(42)
random.shuffle(combined)
combined = combined[args.start:args.end]

# Model #
from transformers import AutoTokenizer,AutoModelForCausalLM,T5ForConditionalGeneration
model_name = " "
tok   = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=torch.float16).to(device).eval()

ver_model_name = "razent/SciFive-large-Pubmed_PMC-MedNLI"
ver_tok = AutoTokenizer.from_pretrained(ver_model_name)
ver_model = AutoModelForSeq2SeqLM.from_pretrained(ver_model_name).to(device).eval()

# Layer #
layers=None
if hasattr(model,"model") and hasattr(model.model,"layers"):
        layers = model.model.layers
elif hasattr(model,"model") and hasattr(model.model,"model") and hasattr(model.model.model, "layers"):
        layers = model.model.model.layers
elif hasattr(model, "layers"):
        layers = model.layers
else:
        raise RuntimeError("Could not find transformer layers ModuleList on this model.")       #print("num_layers:",len(layers))

sample_scores=[]
sent_scores=[]
num_layers =len(layers)
relative_proj_sums = [0.0 for _ in range(num_layers)]

def model_run(model,tok,dataset,device):
    count=0
    total =[]
    samp_nolabel=0
    sent_nolabel=0
    act_per_q =[]
    final_sent =[]
    num_questions_seen = 0

    for line in dataset:
        print(f"-- QUESTION{count} --",flush=True) 
        samp_nolabel=0
        sent_nolabel=0
        count+=1
        sent_scores=[] 
        q_relative_proj = {}
        question = line['question']
        message = f"\n USER: Question: {question}.\n ASSISTANT:"
        inputs = tok(message,return_tensors="pt").to(device)
        prompt_len = inputs["input_ids"].shape[-1]

        handles=[]
        def make_remove_hook(r,layer_idx, threshold=1e-3):
            steered = False
            def hook_fn(module, inp, out):
                
                x = out[0] if isinstance(out, tuple) else out
                r_hat = r.to(device=x.device, dtype=x.dtype)
                r_hat = r_hat / (r_hat.norm() + 1e-12)
                proj = torch.matmul(x, r_hat)
                mean_abs_proj = proj.abs().mean().item()
                relative_proj = proj.abs().mean() / (x.norm(dim=-1).mean() + 1e-12)
                q_relative_proj[layer_idx] = float(relative_proj.item())
            return hook_fn


        for l in range(len(layers)):
            candidate_vector = r[l,3]
            hook_fn = make_remove_hook(candidate_vector,l, threshold=0.1)
            handle = layers[l].register_forward_hook(hook_fn)
            handles.append(handle)

        
        with torch.no_grad():
            _ = model(**inputs)
        
        for h in handles:
            h.remove()
               
        for l in range(num_layers):
            relative_proj_sums[l] += q_relative_proj.get(l, 0.0)        

    print(f"cummulative_proj : {relative_proj_sums}")
    layers_sorted = list(range(len(relative_proj_sums)))
    values = relative_proj_sums/count
    plt.figure(figsize=(10,5))
    plt.bar(layers_sorted, values)

    for x, y in zip(layers_sorted, values):
        plt.text(
            x,
            y,
            f"{y:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90
        )

    plt.xticks(layers_sorted)
    plt.xlabel("Layer")
    plt.ylabel("Cumulative Relative Projection")
    plt.title("Cumulative Relative Projection onto r by Layer")

    plt.tight_layout()
    plt.savefig(opath/"cumulative_relative_projection_by_layer.png", dpi=300)



print(f"-- Test AS run --",flush=True)
start_time = time.time()
model_run(model,tok,combined,device)
elapsed = time.time() - start_time
print(f"Elapsed time: {elapsed:.2f}s")


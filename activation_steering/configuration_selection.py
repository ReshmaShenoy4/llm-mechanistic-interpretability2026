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

parser.add_argument(
        "--alpha",
        type=float,
        required=True)

parser.add_argument(
        "--act",
        type=str,
        required=True)

args = parser.parse_args()
opath = Path(args.output_dir)
log_path  = opath/f"scores_{args.job_id}.log"
meta_path = opath/f"meta_{args.job_id}.jsonl"
dtst_path = Path(args.dtst_path) 

# Hallucination Direction #
r = torch.load(PROJECT_ROOT)

# Dataset #
ds = list(jsonlines.open(dtst_path))
random.seed(42)
random.shuffle(ds)
ds = ds[args.start:args.end]

# Model #
from transformers import AutoTokenizer,AutoModelForCausalLM,T5ForConditionalGeneration
model_name = ""

tok   = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=torch.float16).to(device).eval()

ver_model_name = "razent/SciFive-large-Pubmed_PMC-MedNLI"
ver_tok = AutoTokenizer.from_pretrained(ver_model_name)
ver_model = AutoModelForSeq2SeqLM.from_pretrained(ver_model_name).to(device).eval()

# Layers #
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


def classify_text(text):
    encoding = ver_tok.encode_plus(text, return_tensors="pt", truncation=True)
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = ver_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=8,
            early_stopping=True
        )

    result = ver_tok.decode(
        outputs[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    ).lower()

    if "contradiction" in result:
        return "contradiction", -1
    if "entailment" in result:
        return "entailment", 1
    return "neutral", 0


def truncate(tokenizer, text):
    ids = tokenizer.encode(text)
    return tokenizer.decode(ids[:2000])


relative_proj_sums = [0.0 for _ in range(num_layers)]
def baseline_run(model,tok,dataset,device,alpha,act):
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
        answer = line['answer']
        #context = line['context']

        if isinstance(answer, list):
            answer = answer[0]
        
        if not isinstance(answer, str):
            answer = str(answer)

        answer = truncate(ver_tok,answer) 
        
        message = f"\n USER: Question: {question}.\n ASSISTANT:"
        inputs = tok(message,return_tensors="pt").to(device)
        prompt_len = inputs["input_ids"].shape[-1]

        # Steering #
        handles=[]
        def make_remove_hook(r,layer_idx, threshold=1e-3):
            steered = False
            def hook_fn(module, inp, out):
                nonlocal steered
                
                x = out[0] if isinstance(out, tuple) else out
                
                if steered:
                    return out

                steered = True

                r_hat = r.to(device=x.device, dtype=x.dtype)
                r_hat = r_hat / (r_hat.norm() + 1e-12)

                proj = torch.matmul(x, r_hat)

                mean_abs_proj = proj.abs().mean().item()
                relative_proj = proj.abs().mean() / (x.norm(dim=-1).mean() + 1e-12)
                q_relative_proj[layer_idx] = float(relative_proj.item())

                #if mean_abs_proj > threshold:
                if relative_proj.item() > threshold:
                # remove r component
                    if act == "abl":
                        new_x = x - alpha*proj.unsqueeze(-1) * r_hat
                        act_per_q.append(new_x.mean(dim=(0,1)))
                    else: 
                        new_x = x + alpha*proj.unsqueeze(-1) * r_hat
                        act_per_q.append(new_x.mean(dim=(0,1)))
                else:
                # leave unchanged
                    new_x = x

                if isinstance(out, tuple):
                    return (new_x,) + out[1:]
                else:
                    return new_x

            return hook_fn


        for l in range(len(layers)):
            candidate_vector = r[l,3]
            hook_fn = make_remove_hook(candidate_vector,l, threshold=0.1) 
            handle = layers[l].register_forward_hook(hook_fn)
            handles.append(handle)


        text = model.generate(**inputs,max_new_tokens=512,do_sample=False)
        generated = tok.decode(text[0][prompt_len:], skip_special_tokens=True)
        
        
        for h in handles:
            h.remove()
       
        
        for l in range(num_layers):
            relative_proj_sums[l] += q_relative_proj.get(l, 0.0)        

        text = f"mednli: sentence1: {answer} sentence2: {generated}"
        sample_level,sample_score = classify_text(text)
        total.append(sample_score)       
        #print(f"Sample level: {sample_level}")
        #print(f"Sample score: {sample_score}")
        #print("Sentences:\n") 
        for sent in sent_tokenize(generated):
            text = f"mednli: sentence1: {answer} sentence2: {sent}"
            sent_label, sent_score = classify_text(text)
            sent_scores.append(sent_score)
            #print(f"Sent_label: {sent_label}")
        sentence_score = np.nanmean(sent_scores)
        #print(f"Sentence_score for q{count}: {sentence_score}")
        final_sent.append(sentence_score)
    
    mean_act = torch.stack(act_per_q).mean(dim=0)
    torch.save(mean_act.detach().cpu(), opath/f"mean_act_{args.alpha}_{args.act}.pt")
    
    print(f"Sample Score: {np.nanmean(total)}")
    print(f"Final sent_score :{np.nanmean(final_sent)}")
    print(f"cummulative_proj : {relative_proj_sums}")
    

alpha = args.alpha
act = args.act
print(f"-- Test AS run --",flush=True)
start_time = time.time()
baseline_run(model,tok,ds,device,alpha,act)
elapsed = time.time() - start_time
print(f"Elapsed time: {elapsed:.2f}s")


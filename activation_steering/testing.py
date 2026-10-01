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
meta_path = opath/f"meta_{args.job_id}_pubmedqa.jsonl"
dtst_path = Path(args.dtst_path) 

# Hallucination Direction #
r = torch.load(PROJECT_ROOT)

# Dataset #
ds = list(jsonlines.open(dtst_path))[args.start:args.end]

#-- Model --#
from transformers import AutoTokenizer,AutoModelForCausalLM,T5ForConditionalGeneration
model_name = " "
tok   = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=torch.float16).to(device).eval()

#-- Evaluator Model --#
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
        raise RuntimeError("Could not find transformer layers ModuleList on this model.")    


num_layers =len(layers)
sample_scores=[]
sent_scores=[]

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

def choose_alpha_act(x, r_hat, mean_acts):
    best_alpha = None
    best_act = None
    best_score = -float("inf")
    best_ref_alpha = None 
    best_ref_act = None 

    proj = torch.matmul(x, r_hat)

    for act in ["abl","add"]:
        for alpha in [0.1,0.2,0.3,0.5,0.7,1.0,1.5,2.0]:
            if act == "abl":
                x_new = x - alpha * proj.unsqueeze(-1) * r_hat
            else:
                x_new = x + alpha * proj.unsqueeze(-1) * r_hat

            mean_x = x_new.mean(dim=(0,1))

            for ref_alpha, act_dict in mean_acts.items():
                for ref_act, ref_vec in act_dict.items():
                    ref_vec = ref_vec.to(device)
                    score = -torch.norm(mean_x - ref_vec,p=2).item()
                    if score > best_score:
                        best_score = score
                        best_alpha = alpha
                        best_ref_alpha  = ref_alpha
                        best_act = act
                        best_ref_act = ref_act

    return best_score, best_alpha, best_ref_alpha, best_act,best_ref_act



mean_acts = torch.load(PROJECT_ROOT) # Mean of activations selected from Steering Configuration Stage
label_counts = Counter()
relative_proj_sums = [0.0 for _ in range(num_layers)]
def baseline_run(model,tok,dataset,device):
    count=0
    total =[]
    final_sent =[]

    with open(meta_path,"w",encoding="utf-8") as f:
        for line in dataset:
            print(f"-- QUESTION{count} --",flush=True) 
            count+=1
            imp_info = {}
            sent_scores=[] 
            q_relative_proj = {}
           
            entail_count = 0      # 1
            contradiction_count = 0  # -1
            neutral_count = 0     

            question = line['question']
            answer = line['answer'][0]
            answer = truncate(ver_tok,answer) 
            
            imp_info["question"] = question
            imp_info["ref_ans"] = answer

            message = f"\n USER: Question: {question}. \n ASSISTANT:"
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
        
                    if relative_proj.item() > threshold:
                        print(f"Layer is {layer_idx}")
                        best_score, best_alpha, best_ref_alpha, best_act,best_ref_act = choose_alpha_act(x, r_hat, mean_acts)
                        if best_act == "abl":
                            new_x = x - best_alpha * proj.unsqueeze(-1) * r_hat
                        else:
                            new_x = x + best_alpha * proj.unsqueeze(-1) * r_hat
                        
                        imp_info["chosen_alpha"] = best_alpha
                        imp_info["ref_alpha"] = best_ref_alpha
                        imp_info["similarity"] = best_score
                        imp_info["chosen_act"] = best_act
                        imp_info["ref_act"] = best_ref_act
                    else:
                        new_x = x

                    if isinstance(out, tuple):
                        return (new_x,) + out[1:]
                    else:
                        return new_x

                return hook_fn


            for l in range(len(layers)):
                candidate_vector = r[l,3]
                hook_fn = make_remove_hook(candidate_vector,l, threshold=0.039)
                handle = layers[l].register_forward_hook(hook_fn)
                handles.append(handle)

            text = model.generate(**inputs,max_new_tokens=512,do_sample=False)
            generated = tok.decode(text[0][prompt_len:], skip_special_tokens=True)
           
            imp_info["generated"] = generated
            for h in handles:
                if h is not None:
                    h.remove()
           

            text = f"mednli: sentence1: {answer} sentence2: {generated}"
            sample_level,sample_score = classify_text(text)
            imp_info["sample_score"] = sample_score
            total.append(sample_score)       
            label_counts[sample_level] += 1
            for sent in sent_tokenize(generated):
                text = f"mednli: sentence1: {answer} sentence2: {sent}"
                sent_label, sent_score = classify_text(text)
                sent_scores.append(sent_score)
                    
                if sent_score == 1:
                    entail_count += 1
                elif sent_score == -1:
                    contradiction_count += 1
                elif sent_score == 0:
                    neutral_count += 1

            sentence_score = np.nanmean(sent_scores)
            imp_info["sent_score"] = sentence_score
            imp_info["sent_entailments"] = entail_count
            imp_info["sent_contradictions"] = contradiction_count
            imp_info["sent_neutrals"] = neutral_count
            imp_info["num_sentences"] = entail_count + contradiction_count + neutral_count

            final_sent.append(sentence_score)        
            f.write(json.dumps(imp_info,ensure_ascii=False) + "\n")

        print(f"Sample Score: {np.nanmean(total)}")
        print(f"Total Samples: {len(total)}") 
        print(f"Final sent_score: {np.nanmean(final_sent)}")
        print("Entailment: ", label_counts["entailment"])
        print("Contradiction: ", label_counts["contradiction"])
        print("Neutral: ", label_counts["neutral"])

print(f"-- Test AS run --",flush=True)
start_time = time.time()
baseline_run(model,tok,ds,device)
elapsed = time.time() - start_time
print(f"Elapsed time: {elapsed:.2f}s")


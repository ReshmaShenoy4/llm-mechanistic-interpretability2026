# Import Library #
import argparse
import os
import torch
from datasets import load_dataset, Dataset, concatenate_datasets
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

PROJECT_ROOT = Path.cwd()
parser = argparse.ArgumentParser()
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=None)
parser.add_argument("--task_id", type=int, default=0)
args = parser.parse_args()

# Dataset #
tree = ET.parse(PROJECT_ROOT)
root = tree.getroot()

medline = []
for topic in root.findall("health-topic"):
    question = BeautifulSoup(
        topic.get("meta-desc", ""), "html.parser"
    ).get_text(" ", strip=True)
    answer = BeautifulSoup(
        topic.findtext("full-summary", default=""), "html.parser"
    ).get_text(" ", strip=True)
    medline.append({
        "question": question,
        "answer": answer,
    })
medline_qa = Dataset.from_list(medline)
print(len(medline_qa))
print(medline_qa[0])


#-- PUB,MASH,MEDQUAD --#
pub = load_dataset("qiaojin/PubMedQA", "pqa_artificial")
mash = load_dataset("ZappY-AI/MASHQA-JSON")
quad = load_dataset("lavita/MedQuAD")

def clean_answer(answer):
    if answer is None:
        return ""
    if isinstance(answer, list):
        return " ".join(map(str, answer))
    return str(answer)

pub_qa = pub["train"].map(
    lambda x: {
        "question": x["question"],
        "answer": clean_answer(x["long_answer"]),
    },
    remove_columns=pub["train"].column_names,
)

mash_qa = mash["train"].map(
    lambda x: {
        "question": x["question"],
        "answer": clean_answer(x["answers"]),
    },
    remove_columns=mash["train"].column_names,
)

quad_qa = quad["train"].map(
    lambda x: {
        "question": x["question"],
        "answer": clean_answer(x["answer"]),
    },
    remove_columns=quad["train"].column_names,
)

pub_qa = pub_qa.shuffle(seed=42).select(range(min(2000, len(pub_qa))))
mash_qa = mash_qa.shuffle(seed=42).select(range(min(2000, len(mash_qa))))
quad_qa = quad_qa.shuffle(seed=42).select(range(min(2000, len(quad_qa))))
medline_qa = medline_qa.shuffle(seed=42).select(range(min(2000, len(medline_qa))))

print(len(pub_qa), len(mash_qa), len(quad_qa), len(medline_qa))

combined = concatenate_datasets([pub_qa, mash_qa, quad_qa, medline_qa])
combined = combined.shuffle(seed=123)
print(combined)
print(combined[0])

end = len(combined) if args.end is None else min(args.end, len(combined))
combined = combined.select(range(args.start, end))


# Model #

model_name = " "
tok = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
)
model.eval()

# Device containing the input embeddings.
device = model.get_input_embeddings().weight.device

# -- Layers -- #
selected_layers = [0, 13, 26, 31]

os.makedirs("activations", exist_ok=True)

answer_acts = {layer:[] for layer in selected_layers}
count = 0
for line in combined:
    question = line["question"]
    answer = line["answer"]

    if not isinstance(answer, str):
        answer = " ".join(map(str, answer))

    global_count = args.start + count
    print(f"Question is {global_count}",flush=True)
    count += 1

    question = f"USER: {question}\nASSISTANT: "
    answer = f"{answer}."

    q_ids = tok(question, add_special_tokens=False)["input_ids"]
    a_ids = tok(answer, add_special_tokens=False)["input_ids"]

    p = question + answer
    inputs = tok(
        p,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(device)

    answer_start = len(q_ids)
    answer_end = min(answer_start + len(a_ids), answer_start + 100)

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            use_cache=False,
        )

    for layer in selected_layers:
        hidden = outputs.hidden_states[layer+1][0]
        answer_acts[layer].append(hidden[answer_start:answer_end].cpu())
    
    del outputs,inputs

torch.save(
    answer_acts,
    f"activations/task_{args.task_id}_sample.pt",
)

print("Done")

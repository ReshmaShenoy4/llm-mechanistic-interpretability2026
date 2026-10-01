# Give two Q-A set to SAE, encode the answer tokens and then evaluate. 
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer,AutoModelForCausalLM
from pathlib import Path
import argparse
path = Path.cwd()
device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()


parser.add_argument(
        "--act",
        type=str,
        required=True)
args = parser.parse_args()

#-- Load Model --#
model_name=  " "
tok   = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=torch.float16).to(device).eval()


#-- Layers --#
layers=None
if hasattr(model,"model") and hasattr(model.model,"layers"):
    layers = model.model.layers
elif hasattr(model,"model") and hasattr(model.model,"model") and hasattr(model.model.model, "layers"):
    layers = model.model.model.layers
elif hasattr(model, "layers"):
    layers = model.layers
else:
    raise RuntimeError("Could not find transformer layers ModuleList on this model.")       #print("num_layers:",len(layers))
L=len(layers)

l=0

#-- Prompt --#
q=f"Is diffusion-weighted imaging a significant indicator of the development of vascularization in hypovascular hepatocellular lesions?"
ba = f"""
Diffusion-weighted imaging (DWI) is a magnetic resonance imaging (MRI) technique that measures the diffusion of water molecules in tissues. It has been used to assess tissue microstructure and can provide information about the presence and extent of vascularization in hypovascular hepatocellular lesions.\n\nStudies have shown that DWI can detect and quantify the degree of vascularization in hypovascular hepatocellular lesions, such as hepatocellular carcinoma (HCC) and hepatic metastases. DWI can differentiate between hypovascular and well-vascularized lesions, and can also predict the response to treatment in some cases.\n\nHowever, it is important to note that DWI is not specific for vascularization and can also be affected by other factors such as edema, fibrosis, and necrosis. Therefore, DWI should be interpreted in conjunction with other imaging modalities and clinical information to obtain a comprehensive assessment of the lesion.\n\nIn summary, DWI can be a significant indicator of the development of vascularization in hypovascular hepatocellular lesions, but it should be used in conjunction with other imaging modalities and clinical information to obtain a comprehensive assessment of the lesion.
"""
sa = f"""Diffusion-weighted imaging (DWI) is a magnetic resonance imaging (MRI) technique that measures the diffusion of water molecules in tissues. It has been used to assess tissue microstructure and can provide information about the development of vascularization in hypovascular hepatocellular lesions.\n\nStudies have shown that DWI can detect changes in tissue microstructure in hypovascular hepatocellular lesions, including the development of new blood vessels. DWI can also differentiate between hypovascular and hypervascular lesions, which is important for guiding treatment decisions.\n\nHowever, it is important to note that DWI is not a specific indicator of vascularization and should be interpreted in conjunction with other imaging modalities and clinical information. Additionally, the sensitivity and specificity of DWI for detecting vascularization in hypovascular hepatocellular lesions may vary depending on the lesion type, size, and location.\n\nIn summary, DWI can be a useful tool for assessing the development of vascularization in hypovascular hepatocellular lesions, but it should be used in conjunction with other imaging modalities and clinical information, and the interpretation should be made with caution."""


a = ba # or sa
q_ids = tok(q, add_special_tokens=False)["input_ids"]
a_ids = tok(a, add_special_tokens=False)["input_ids"]

p = q + a
inputs = tok(p,return_tensors="pt",add_special_tokens=False).to(device)

answer_start = len(q_ids)
answer_end = answer_start  + len(a_ids)


answer_token_ids = inputs["input_ids"][0, answer_start:answer_end].cpu()

with torch.no_grad():
    outputs = model(**inputs,output_hidden_states=True,use_cache=False,)

hidden = outputs.hidden_states[l][0]
answer_act = hidden[answer_start:answer_end].cpu()
del outputs,inputs

ans_act = answer_act.to(device)
print(f"ans_act_con shape is {ans_act.shape}")

#-- Load SAE --#
sae_model = torch.load(path,map_location='cuda')
w_enc = sae_model['model_state_dict']['encoder.weight']
b_enc = sae_model['model_state_dict']['encoder.bias']

#-- Get encoded activations --#
ans_act = ans_act.float()
w_enc = w_enc.float()
b_enc = b_enc.float()
C = F.relu(ans_act@ w_enc.T + b_enc)
print(C.shape)

results = {
        "encoded": C.cpu(),
        "answer_token_ids": answer_token_ids,
        "q":q,
        "a":a,
        }
torch.save(results,f"{args.act}_encoded.pt")

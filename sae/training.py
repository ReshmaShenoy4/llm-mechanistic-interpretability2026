# Import Library #
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument(
        "--acts", 
        nargs="+",
        required=True)


parser.add_argument("--output", type=str, default="sae_layer_0.pt")
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ACTIVATION_FILES = args.acts

# Hyperparameter #
BATCH_SIZE = 256
NUM_EPOCHS = 5
LEARNING_RATE = 1e-4
L1_COEFFICIENT = 1e-3
EXPANSION_FACTOR = 2



def load_activations(path):
    activations = torch.load(path, map_location="cpu")

    if isinstance(activations, dict):
        print("Available keys:", activations.keys())
        activations = activations[0]     

    if isinstance(activations, list):
        activations = torch.cat(activations, dim=0)

    if not isinstance(activations, torch.Tensor):
        raise TypeError(
            f"{path} contains {type(activations)}, not a tensor or list."
        )

    activations = activations.detach().cpu()
    activations = activations.reshape(-1, activations.shape[-1]).float() #except last dim, flatten rest. So 1000x100x4096 -> 100000x4096

    print(f"{path}: {tuple(activations.shape)}",flush=True)
    return activations


class SAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim, bias=False)

    def forward(self, x):
        features = torch.relu(self.encoder(x))
        reconstruction = self.decoder(features)
        return reconstruction, features

first_activations = load_activations(ACTIVATION_FILES[0])
input_dim = first_activations.shape[-1]
latent_dim = input_dim * EXPANSION_FACTOR
del first_activations

model = SAE(input_dim, latent_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"Input dimension: {input_dim}",flush=True)
print(f"SAE latent dimension: {latent_dim}",flush=True)

start_time = time.time()

for epoch in range(NUM_EPOCHS):
    model.train()
    
    total_loss = 0.0
    total_reconstruction = 0.0
    total_l1 = 0.0
    num_batches = 0
    
    file_order = ACTIVATION_FILES.copy()
    random.shuffle(file_order)

    print(
            f"\n Global Epoch {epoch+1}/{NUM_EPOCHS}",
            flush=True,
            )

    for file_idx, path in enumerate(file_order):
        print(f"\nTraining on file {file_idx + 1}: {path}",flush=True)
       
        activations = load_activations(path)
        dataset = TensorDataset(activations)
        loader = DataLoader(
                    dataset,
                    batch_size=BATCH_SIZE,
                    shuffle=True,)
    
        
        for (batch,) in loader:
            batch = batch.to(device)
        
            optimizer.zero_grad()
            x_hat, features = model(batch)
        
            reconstruction_loss = nn.functional.mse_loss(x_hat, batch)
            l1_loss = features.abs().mean()
            loss = reconstruction_loss + L1_COEFFICIENT * l1_loss
        
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
            total_loss += loss.item()
            total_reconstruction += reconstruction_loss.item()
            total_l1 += l1_loss.item()
            num_batches +=1

        del dataset,loader,activations
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    mean_loss = total_loss / num_batches
    mean_reconstruction = total_reconstruction / num_batches
    mean_l1 = total_l1 / num_batches
    
    print(
        f"Epoch {epoch + 1:02d}/{NUM_EPOCHS} | "
        f"loss={mean_loss:.6f} | "
        f"reconstruction={mean_reconstruction:.6f} | "
        f"L1={mean_l1:.6f}",
        flush=True
    )
        

torch.save(
        {
            "layer": 0,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "input_dim": input_dim,
            "latent_dim": latent_dim,
            "expansion_factor": EXPANSION_FACTOR,
            "l1_coefficient": L1_COEFFICIENT,
            "learning_rate": LEARNING_RATE,
            "activation_files": ACTIVATION_FILES,
        },
        args.output,
)

print(f"\nSaved final SAE to: {args.output}")
print(
    f"Training time: "
    f"{time.time() - start_time:.1f} seconds"
)

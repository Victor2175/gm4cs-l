import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt

class ClimateDataset(Dataset):
    def __init__(self, data):
        print("Creating datasets...")
        self.inputs = []
        self.outputs = []
        for model, runs in tqdm(data.items(), desc="Processing models"):
            forced_response = runs['forced_response']
            for run_key, run_data in runs.items():
                if run_key != 'forced_response':
                    self.inputs.append(run_data)
                    self.outputs.append(forced_response)

        self.inputs = torch.tensor(self.inputs, dtype=torch.float32)
        self.outputs = torch.tensor(self.outputs, dtype=torch.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_data = self.inputs[idx]#.unsqueeze(0)  # Add channel dimension
        output_data = self.outputs[idx]#.unsqueeze(0)  # Add channel dimension
        return {'input': input_data, 'output': output_data}

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=400, latent_dim=200, device='cpu', mu_var_dim=2):
        super().__init__()
        self.device = device

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, latent_dim),
            nn.LeakyReLU(0.2)
        )

        # Latent mean and variance
        self.mean_layer = nn.Linear(latent_dim, mu_var_dim)
        self.logvar_layer = nn.Linear(latent_dim, mu_var_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(mu_var_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x): 
        # x: (batch_size, input_dim)
        h = self.encoder(x)
        mean, logvar = self.mean_layer(h), self.logvar_layer(h)
        return mean, logvar

    def reparameterization(self, mean, logvar):
        epsilon = torch.randn_like(logvar).to(self.device)
        z = mean + torch.exp(0.5 * logvar) * epsilon
        return z

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterization(mean, logvar)
        x_hat = self.decode(z)
        return x_hat, mean, logvar

def vae_loss_function(x, x_hat, mean, logvar):
    reproduction_loss = F.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return reproduction_loss + KLD

def train_vae(model, data_loader, optimizer, epochs, device='cpu'):
    model.to(device)
    model.train()

    # # Print input and output dimensions at the start
    # first_batch = next(iter(data_loader))
    # print(f"Input dimensions: {first_batch['input'].shape}")
    # print(f"Output dimensions: {first_batch['output'].shape}")

    for epoch in tqdm(range(epochs)):
        overall_loss = 0
        for batch_idx, batch in enumerate(data_loader):
            # Flatten the input and output to match the model's expected input dimensions
            x = batch['input'].view(batch['input'].size(0), -1).to(device)  # Flatten input
            y = batch['output'].view(batch['output'].size(0), -1).to(device)  # Flatten output

            # Debugging: Check for NaN values in input and output
            if torch.isnan(x).any():
                print(f"NaN detected in input at Epoch {epoch + 1}, Batch {batch_idx + 1}")
            if torch.isnan(y).any():
                print(f"NaN detected in output at Epoch {epoch + 1}, Batch {batch_idx + 1}")

            optimizer.zero_grad()

            # Forward pass: input -> model -> reconstructed output
            y_hat, mean, logvar = model(x)

            # Clamp the reconstructed output to avoid issues in BCE loss
            y_hat = torch.clamp(y_hat, min=1e-7, max=1-1e-7)

            # Debugging: Check for NaN values in model outputs
            if torch.isnan(y_hat).any():
                print(f"NaN detected in reconstructed output at Epoch {epoch + 1}, Batch {batch_idx + 1}")
            if torch.isnan(mean).any():
                print(f"NaN detected in mean at Epoch {epoch + 1}, Batch {batch_idx + 1}")
            if torch.isnan(logvar).any():
                print(f"NaN detected in logvar at Epoch {epoch + 1}, Batch {batch_idx + 1}")

            # Compute loss: compare reconstructed output (y_hat) with actual output (y)
            loss = vae_loss_function(y, y_hat, mean, logvar)

            # Debugging: Check for NaN values in loss
            if torch.isnan(loss).any():
                print(f"NaN detected in loss at Epoch {epoch + 1}, Batch {batch_idx + 1}")

            overall_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}, Average Loss: {overall_loss / len(data_loader.dataset)}")
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
    def __init__(self, input_dim, hidden_dim=25, latent_dim=50, device='cpu', mu_var_dim=5):
        super().__init__()
        self.device = device

        # Encoder with batch normalization
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # Added batch norm
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),  # Added batch norm
            nn.LeakyReLU(0.2)
        )

        # Latent mean and variance
        self.mean_layer = nn.Linear(latent_dim, mu_var_dim)
        self.logvar_layer = nn.Linear(latent_dim, mu_var_dim)

        # Decoder with batch normalization
        self.decoder = nn.Sequential(
            nn.Linear(mu_var_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),  # Added batch norm
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # Added batch norm
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh()  # Changed to Tanh to handle negative values
        )

    def encode(self, x):
        # Check and print input range
        # print(f"Encoder input range: min={x.min().item():.4f}, max={x.max().item():.4f}, mean={x.mean().item():.4f}")
        
        # Regular encoder forward pass
        h = self.encoder(x)
        
        # Check encoder output before calculating mean and logvar
        # if torch.isnan(h).any():
        #     print("NaN detected in encoder output")
        # else:
        #     print(f"Encoder output range: min={h.min().item():.4f}, max={h.max().item():.4f}, mean={h.mean().item():.4f}")
        
        mean = self.mean_layer(h)
        logvar = self.logvar_layer(h)
        
        # Check mean and logvar for NaNs
        if torch.isnan(mean).any():
            print("NaN detected in mean calculation")
        if torch.isnan(logvar).any():
            print("NaN detected in logvar calculation")
        
        return mean, logvar

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(self.device)      
        z = mean + var*epsilon
        return z
    # def reparameterization(self, mean, logvar):
    #     # FIXED: Now correctly using exp(0.5 * logvar) instead of logvar directly
    #     std = torch.exp(0.5 * logvar)
    #     epsilon = torch.randn_like(std).to(self.device)      
    #     z = mean + std * epsilon
    #     return z

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterization(mean, logvar)
        x_hat = self.decode(z)
        return x_hat, mean, logvar

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def vae_loss_function(x, x_hat, mean, logvar, beta=0.01):
    # # Reconstruction loss (binary cross-entropy)
    # reproduction_loss = F.binary_cross_entropy(x_hat, x, reduction='sum')
    # Switched to binary cross-entropy with logits for better numerical stability
    reproduction_loss = F.binary_cross_entropy_with_logits(x_hat, x, reduction='sum')
    
    # KL Divergence with a beta factor to control its influence 
    # Using a smaller beta value to reduce the influence of KL divergence
    KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    
    return reproduction_loss + beta * KLD

def train_vae(model, data_loader, optimizer, epochs, device='cpu'):
    model.to(device)
    model.train()

    # Print input and output dimensions at the start
    # first_batch = next(iter(data_loader))
    # print(f"Input dimensions: {first_batch['input'].shape}")
    # print(f"Output dimensions: {first_batch['output'].shape}")

    for epoch in tqdm(range(epochs)):
        overall_loss = 0
        for batch_idx, batch in enumerate(data_loader):
            # Flatten the input and output to match the model's expected input dimensions
            x = batch['input'].view(batch['input'].size(0), -1).to(device)  # Flatten input
            y = batch['output'].view(batch['output'].size(0), -1).to(device)  # Flatten output

            # Debugging: Check for NaN or extreme values in input and output
            # if torch.isnan(x).any() or torch.isinf(x).any():
            #     print(f"NaN or Inf detected in input at Epoch {epoch + 1}, Batch {batch_idx + 1}")
            # if torch.isnan(y).any() or torch.isinf(y).any():
            #     print(f"NaN or Inf detected in output at Epoch {epoch + 1}, Batch {batch_idx + 1}")

            optimizer.zero_grad()

            # Forward pass: input -> model -> reconstructed output
            y_hat, mean, logvar = model(x)

            # Clamp the reconstructed output to avoid issues in BCE loss
            # y_hat = torch.clamp(y_hat, min=1e-7, max=1-1e-7)

            # Debugging: Check for NaN values in model outputs
            # if torch.isnan(y_hat).any():
            #     print(f"NaN detected in reconstructed output at Epoch {epoch + 1}, Batch {batch_idx + 1}")
            # if torch.isnan(mean).any():
            #     print(f"NaN detected in mean at Epoch {epoch + 1}, Batch {batch_idx + 1}")
            # if torch.isnan(logvar).any():
            #     print(f"NaN detected in logvar at Epoch {epoch + 1}, Batch {batch_idx + 1}")

            # Compute loss: compare reconstructed output (y_hat) with actual output (y)
            loss = vae_loss_function(y, y_hat, mean, logvar)

            # Debugging: Check for NaN values in loss
            # if torch.isnan(loss).any():
            #     print(f"NaN detected in loss at Epoch {epoch + 1}, Batch {batch_idx + 1}")

            overall_loss += loss.item()

            # Backward pass and optimization
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

        print(f"Epoch {epoch + 1}, Average Loss: {overall_loss / len(data_loader.dataset)}")
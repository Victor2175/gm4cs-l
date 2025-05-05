import torch
import os
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

class SIMPLEVAE(nn.Module):
    def __init__(self, input_dims, z_dim, device='cpu'):
        super().__init__()
        self.device = device
        self.z_dim = z_dim
        
        # Store input dimensions for reshaping operations
        if isinstance(input_dims, tuple):
            self.time_dim = input_dims[0]
            self.feature_dim = input_dims[1]
            
        # Linear layer to map (N, T, D) -> (N, T, D')
        self.encoder = nn.Linear(self.feature_dim, z_dim)

        # Latent mean and log variance layers (changed from variance to logvar)
        self.mean_layer = nn.Linear(z_dim, z_dim)
        self.logvar_layer = nn.Linear(z_dim, z_dim)  # Changed from var_layer to logvar_layer

        # Linear layer to map (N, T, D') -> (N, T, D)
        self.decoder = nn.Linear(z_dim, self.feature_dim)

    def encode(self, x):
        # Map input to latent space
        h = self.encoder(x)
        mean = self.mean_layer(h)
        logvar = self.logvar_layer(h)  # Now outputs log variance
        
        return mean, logvar

    def reparameterization(self, mean, logvar):
        # Using log variance for numerical stability
        std = torch.exp(0.5 * logvar)  # Convert logvar to standard deviation
        epsilon = torch.randn_like(std).to(self.device)
        z = mean + std * epsilon
        return z

    def decode(self, z):
        # Map latent space back to input space
        output = self.decoder(z)
        return output
   
    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterization(mean, logvar)
        x_hat = self.decode(z)
        return x_hat, mean, logvar
    
    def generate(self, num_samples):
        """
        Generate new data using random latent vectors.
        Args:
            num_samples (int): Number of samples to generate.
        Returns:
            torch.Tensor: Generated data of shape (num_samples, seq_len, feat_dim).
        """
        # Generate random latent vectors
        # For the simple VAE, we need to create sequences of random vectors
        # with the same dimensions as our input
        latent_vectors = torch.randn((num_samples, self.time_dim, self.z_dim)).to(self.device)
        
        # Decode latent vectors to generate data
        with torch.no_grad():
            generated_data = self.decode(latent_vectors)
        
        return generated_data

def vae_loss_function(x, x_hat, mean, logvar, beta=1):
    # Reconstruction loss (MSE)
    reproduction_loss = F.mse_loss(x_hat, x, reduction='sum')
    
    # KL Divergence with a beta factor to control its influence
    # Using log variance for numerical stability
    KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - torch.exp(logvar))
    
    return reproduction_loss + beta * KLD

def train_vae(model, data_loader, optimizer, epochs, device='cpu'):
    model.to(device)
    model.train()
    
    # Track losses for plotting
    losses = []

    for epoch in tqdm(range(epochs)):
        overall_loss = 0
        for _, batch in enumerate(data_loader):
            # Use inputs with temporal structure (N, T, D) instead of flattening
            x = batch['input'].to(device)  # Shape: (batch_size, T, D)
            y = batch['output'].to(device)  # Shape: (batch_size, T, D)
            
            # Print shapes for the first batch in the first epoch
            if epoch == 0 and _ == 0:
                print(f"Training batch input shape: {x.shape}")
                print(f"Training batch output shape: {y.shape}")

            # Forward pass: input -> model -> reconstructed output
            y_hat, mean, var = model(x)

            # Compute loss: compare reconstructed output (y_hat) with actual output (y)
            loss = vae_loss_function(y, y_hat, mean, var)
            
            optimizer.zero_grad()
            
            # Backward pass and optimization
            loss.backward()

            overall_loss += loss.item()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

        # Calculate average loss for this epoch
        avg_loss = overall_loss / len(data_loader.dataset)
        losses.append(avg_loss)
        if epoch % 10 == 0:
            print(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}")

    return losses

def save_model(model, path):
    # If the directory does not exist, create it
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")
    
def load_model(model, path):
    model.load_state_dict(torch.load(path))
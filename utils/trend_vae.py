import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

class TrendLayer(nn.Module):
    def __init__(self, seq_len, feat_dim, latent_dim, trend_poly):
        super(TrendLayer, self).__init__()
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim
        self.trend_poly = trend_poly
        self.trend_dense1 = nn.Linear(self.latent_dim, self.feat_dim * self.trend_poly)
        self.trend_dense2 = nn.Linear(self.feat_dim * self.trend_poly, self.feat_dim * self.trend_poly)

    def forward(self, z):
        trend_params = F.relu(self.trend_dense1(z))
        trend_params = self.trend_dense2(trend_params)
        trend_params = trend_params.view(-1, self.feat_dim, self.trend_poly)

        lin_space = torch.arange(0, float(self.seq_len), 1, device=z.device) / self.seq_len
        poly_space = torch.stack([lin_space ** float(p + 1) for p in range(self.trend_poly)], dim=0)

        trend_vals = torch.matmul(trend_params, poly_space)
        trend_vals = trend_vals.permute(0, 2, 1)
        return trend_vals

class Trend_Vae(nn.Module):
    def __init__(self, seq_len, feat_dim, hidden_dim=50, latent_dim=64, z_dim=5, trend_poly=2, device='cpu'):
        super(Trend_Vae, self).__init__()
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.device = device
        self.trend_poly = trend_poly

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(seq_len * feat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.LeakyReLU(0.2)
        )
        self.mean_layer = nn.Linear(latent_dim, z_dim)
        self.var_layer = nn.Linear(latent_dim, z_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, seq_len * feat_dim)
        )

        # Trend Layer
        self.trend_layer = TrendLayer(seq_len, feat_dim, z_dim, trend_poly)

    def encode(self, x):
        x = x.view(x.size(0), -1)  # Flatten input (batch_size, seq_len * feat_dim)
        h = self.encoder(x)
        mean = self.mean_layer(h)
        var = self.var_layer(h)
        return mean, var

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(self.device)
        z = mean + var * epsilon
        return z

    def decode(self, z):
        x_hat = self.decoder(z)
        x_hat = x_hat.view(-1, self.seq_len, self.feat_dim)  # Reshape to (batch_size, seq_len, feat_dim)
        trend_vals = self.trend_layer(z)
        return x_hat + trend_vals

    def forward(self, x):
        mean, var = self.encode(x)
        z = self.reparameterization(mean, var)
        x_hat = self.decode(z)
        return x_hat, mean, var

def vae_loss_function(x, x_hat, mean, var, beta=1):
    # Reconstruction loss
    reconstruction_loss = F.mse_loss(x_hat, x, reduction='sum')
    # KL Divergence
    KLD = -0.5 * torch.sum(1 + torch.log(var.pow(2)) - mean.pow(2) - var.exp())
    return reconstruction_loss + beta * KLD

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
        input_data = self.inputs[idx]
        output_data = self.outputs[idx]
        return {'input': input_data, 'output': output_data}

def train_vae(model, data_loader, optimizer, epochs, device='cpu'):
    model.to(device)
    model.train()
    losses = []

    for epoch in range(epochs):
        overall_loss = 0
        for batch in tqdm(data_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            x = batch['input'].to(device)
            y = batch['output'].to(device)

            x_hat, mean, var = model(x)
            loss = vae_loss_function(y, x_hat, mean, var)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            overall_loss += loss.item()

        avg_loss = overall_loss / len(data_loader.dataset)
        losses.append(avg_loss)
        if epoch % 10 == 0:
            print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

    return losses
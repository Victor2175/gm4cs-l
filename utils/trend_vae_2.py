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
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(feat_dim, hidden_dim, kernel_size=3, stride=2, padding=1),  # Downsample T
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, stride=2, padding=1),  # Downsample further
            nn.ReLU(),
            nn.Conv1d(hidden_dim * 2, hidden_dim * 4, kernel_size=3, stride=2, padding=1),  # Further downsample
            nn.ReLU()
        )
        # Calculate the output size of Conv1D layers
        conv_out_len = seq_len // (2 ** 3)  # Downsampled by 2^3 due to 3 Conv1D layers with stride=2
        self.encoder_fc = nn.Linear(conv_out_len * (hidden_dim * 4), latent_dim)  # Fully connected layer
        self.mean_layer = nn.Linear(latent_dim, z_dim)
        self.var_layer = nn.Linear(latent_dim, z_dim)

        # Decoder
        self.decoder_fc = nn.Linear(z_dim, conv_out_len * (hidden_dim * 4))  # Fully connected layer
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose1d(hidden_dim * 4, hidden_dim * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_dim * 2, hidden_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_dim, feat_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        )

        # Trend Layer
        self.trend_layer = TrendLayer(seq_len, feat_dim, z_dim, trend_poly)

    def encode(self, x):
        # Input shape: (N, T, D)
        x = x.permute(0, 2, 1)  # Change to (N, D, T) for Conv1D
        x = self.encoder_conv(x)  # Apply Conv1D layers
        x = x.view(x.size(0), -1)  # Flatten
        h = self.encoder_fc(x)  # Fully connected layer
        mean = self.mean_layer(h)
        var = self.var_layer(h)
        return mean, var

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(self.device)
        z = mean + var * epsilon
        return z

    def decode(self, z):
        # Input shape: (N, z_dim)
        x = self.decoder_fc(z)  # Fully connected layer
        conv_out_len = self.seq_len // (2 ** 3)  # Downsampled length
        x = x.view(x.size(0), -1, conv_out_len)  # Reshape to (N, hidden_dim * 4, T // 8)
        x = self.decoder_conv(x)  # Apply transposed Conv1D layers
        x = x.permute(0, 2, 1)  # Change back to (N, T, D)
        trend_vals = self.trend_layer(z)  # Add trend component
        return x + trend_vals

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
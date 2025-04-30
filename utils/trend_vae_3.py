import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


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
            nn.Conv1d(in_channels=feat_dim, out_channels=hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_dim, out_channels=latent_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()  # Flatten to ensure output is (N, -1)
        )
        self.encoder_last_dense_dim = self._get_last_dense_dim(seq_len, feat_dim, hidden_dim, latent_dim)
        self.mean_layer = nn.Linear(self.encoder_last_dense_dim, z_dim)
        self.var_layer = nn.Linear(self.encoder_last_dense_dim, z_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv1d(in_channels=z_dim, out_channels=latent_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=latent_dim, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_dim, out_channels=feat_dim, kernel_size=3, stride=1, padding=1)
        )

        # Trend Layer
        self.trend_layer = TrendLayer(seq_len, feat_dim, z_dim, trend_poly)

    def encode(self, x):
        x = x.permute(0, 2, 1)  # Change to (N, D, T) for Conv1D
        x = self.encoder(x)
        mean = self.mean_layer(x)  # Shape: (N, z_dim)
        var = self.var_layer(x)   # Shape: (N, z_dim)
        return mean, var

    def _get_last_dense_dim(self, seq_len, feat_dim, hidden_dim, latent_dim):
        with torch.no_grad():
            x = torch.randn(1, feat_dim, seq_len)
            x = nn.Conv1d(in_channels=feat_dim, out_channels=hidden_dim, kernel_size=3, stride=2, padding=1)(x)
            x = nn.Conv1d(in_channels=hidden_dim, out_channels=latent_dim, kernel_size=3, stride=2, padding=1)(x)
            x = x.flatten(start_dim=1)  # Flatten to calculate the dense dimension
            return x.size(1)

    def encode(self, x):
        x = x.permute(0, 2, 1)  # Change to (N, D, T) for Conv1D
        for layer in self.encoder:
            x = layer(x)
        # print(f"Encoder layer output shape: {x.shape}")
        mean = self.mean_layer(x)
        var = self.var_layer(x)
        return mean, var # Dimensions (N, z_dim)

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(self.device)
        z = mean + var * epsilon
        return z

    def decode(self, z):
        z_= z
        for layer in self.decoder:
            z = layer(z)
            # print(f"Decoder layer output shape: {z.shape}")
        x_hat = z.permute(0, 2, 1)  # Change back to (N, T, D)

        # Compute trend component
        trend_vals = self.trend_layer(z_)  # Shape: (N, T, D)
        return x_hat + trend_vals

    def forward(self, x):
        mean, var = self.encode(x)
        z = self.reparameterization(mean, var) # Shape: (N, T, z_dim)
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

    for epoch in tqdm(range(epochs)):
        overall_loss = 0
        for batch in data_loader:
            x = batch['input'].to(device)  # Shape: (N, T, D)
            y = batch['output'].to(device)  # Shape: (N, T, D)

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
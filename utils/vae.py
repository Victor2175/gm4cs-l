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
        input_data = self.inputs[idx].unsqueeze(0)  # Add channel dimension
        output_data = self.outputs[idx].unsqueeze(0)  # Add channel dimension
        return {'input': input_data, 'output': output_data}

class VAE(nn.Module):
    def __init__(self, input_channels, latent_dim):
        super(VAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        # Recalculate the flattened size based on input dimensions (165x6523)
        # After first Conv2d: (32, 165, 6523)
        # After second Conv2d: (64, 83, 3262)
        # After third Conv2d: (128, 42, 1631)
        self.flattened_size = 128 * 42 * 1631
        self.fc_mean = nn.Linear(self.flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_size, latent_dim)
        
        # Decoder
        self.fc_decode = nn.Linear(latent_dim, self.flattened_size)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=(1, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=(1, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)  # Flatten the tensor
        return self.fc_mean(h), self.fc_logvar(h)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) # To parametrize epsilon as a Normal(0,1) distribution
        return mean + eps * std

    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(h.size(0), 128, 42, 1631)  # Reshape to match the decoder's input
        return self.decoder(h)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        return self.decode(z), mean, logvar

def vae_loss_function(recon_x, x, mean, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return BCE + KLD

def train_vae(model, data_loader, optimizer, num_epochs=10):
    model.train()
    epoch_losses = []  # To store epoch-level losses for plotting

    for epoch in range(num_epochs):
        total_loss = 0
        with tqdm(data_loader, desc=f"Epoch {epoch + 1}/{num_epochs}") as progress_bar:
            for batch in progress_bar:
                inputs = batch['input']  # Assuming data loader provides a dictionary with 'input' and 'output'
                optimizer.zero_grad()
                recon_batch, mean, logvar = model(inputs)
                loss = vae_loss_function(recon_batch, inputs, mean, logvar)
                loss.backward()
                total_loss += loss.item()
                optimizer.step()

                # Update progress bar with current loss
                progress_bar.set_postfix(loss=loss.item())

        epoch_loss = total_loss / len(data_loader.dataset)
        epoch_losses.append(epoch_loss)
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss}")

    # Plot the training loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), epoch_losses, marker='o', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()
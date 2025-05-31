import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys
import random
import warnings
import gc
import psutil
import pickle as pkl
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

# Add utility paths
sys.path.append(os.path.join(os.getcwd(), 'utils'))

# Import utility functions
from utils.data_loading import *
from utils.data_processing import *
from utils.trend_vae import *
from utils.animation import *
from utils.metrics import *
from utils.pipeline import *

# Suppress warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Define data path
current_dir = os.getcwd()
data_path = os.path.join(current_dir, 'data')
print(f"Data path: {data_path}", flush=True)

# Use MPS / Cuda or CPU if none of the options are available
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)
random.seed(42)

# Load the data
filename = os.path.join(data_path, 'ssp585_time_series.pkl')
data, nan_mask = preprocess_data(data_path, filename)

# Select one of the models randomly for testing and the rest for training according to the leave-one-out strategy
models = list(data.keys())
print(f"There are {len(models)} models in the dataset.", flush=True)

# Leave-One-Out Cross-Validation
mse_scores = []
normalized_mse_scores = []
center = True  # Center the data
hidden_dim = 128 # Increased for better representation
feat_dim = 5630 # Number of features in the input data
latent_dim = 64 # Intermediate layer size
z_dim = 5 # Actual latent space dimension
trend_poly = 2
seq_len = 165 # Number of time steps in the sequence
batch_size = 32 # Batch size for training

for test_model in tqdm(data.keys()):
    # Print memory usage before iteration
    print(f"Memory usage before processing {test_model}: {psutil.virtual_memory().percent}%", flush=True)

    # Split data into training and testing sets
    train_models = [model for model in data.keys() if model != test_model]
    train_data = {model: data[model] for model in train_models}
    test_data = {test_model: data[test_model]}

    # Normalize the data
    normalized_train_data, normalized_test_data, _, testing_statistics = normalize_data(train_data, test_data, center=center)

    # Create datasets and dataloaders
    train_dataset = ClimateDataset(normalized_train_data)
    test_dataset = ClimateDataset(normalized_test_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model
    input_dim = train_dataset[0]['input'][1]
    vae_model = Trend_Vae(
        seq_len=seq_len,
        feat_dim=feat_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        z_dim=z_dim,
        trend_poly=trend_poly,
        use_residual_conn=True,
        device=device,
        ).to(device)

    # Train the model (simplified for tutorial purposes)
    epochs = 150
    losses = []

    optimizer = torch.optim.Adam(vae_model.parameters(), lr=3e-4) # Weight decay is not used in this version (was weight_decay=1e-5)

    losses = train_vae(vae_model, train_loader, optimizer, epochs=epochs, device=device)

    # Evaluate the model
    results = evaluate_vae(vae_model, test_loader, device, testing_statistics=testing_statistics)  # Replace None with actual testing_statistics if available
    mse = results['mse']
    normalized_mse = results['normalized_mse']
    mse_scores.append(mse)
    normalized_mse_scores.append(normalized_mse)

    print(f"Test model: {test_model}, MSE: {mse}, Normalized MSE: {normalized_mse}")

    # Clear memory
    del train_models, train_data, test_data, normalized_train_data, normalized_test_data
    del train_dataset, test_dataset, train_loader, test_loader, vae_model, optimizer, losses, results
    gc.collect()

    # Print memory usage after iteration
    print(f"Memory usage after processing {test_model}: {psutil.virtual_memory().percent}%")

# Save MSE scores to a file
mse_file = os.path.join(current_dir, 'mse_scores.pkl')
with open(mse_file, 'wb') as f:
    pkl.dump(mse_scores, f)
    
normalized_mse_file = os.path.join(current_dir, 'normalized_mse_scores.pkl')
with open(normalized_mse_file, 'wb') as f:
    pkl.dump(normalized_mse_scores, f)

print(f"MSE scores saved to {mse_file}", flush=True)
print(f"Normalized MSE scores saved to {normalized_mse_file}", flush=True)

# Plot MSE Distributions
# Load MSE scores
with open(mse_file, 'rb') as f:
    mse_scores = pkl.load(f)

# Create a quantile plot
plt.figure(figsize=(10, 6))
sns.ecdfplot(mse_scores, complementary=True)
plt.title('Quantile Plot of MSE Distributions')
plt.xlabel('MSE')
plt.ylabel('1 - CDF')
plt.grid(True)
plt.show()

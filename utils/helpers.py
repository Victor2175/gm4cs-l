import os
import pickle as pkl
import numpy as np
import random
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import seaborn as sns
import random
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

### Data Loading and Preprocessing ###

def load_data(data_path, filename):
    with open(os.path.join(data_path, filename), 'rb') as f:
        data = pkl.load(f)
    return data

def filter_data(data, min_runs=4):
    filtered_data = {
        model: {run: np.flip(data[model][run], axis=1) for run in data[model]}
        for model in data.keys() if len(data[model]) >= min_runs
    }
    return filtered_data

def create_nan_mask(filtered_data):
    grid_shape = filtered_data[list(filtered_data.keys())[0]][list(filtered_data[list(filtered_data.keys())[0]].keys())[0]].shape[1:]
    nan_mask = np.zeros(grid_shape, dtype=bool)
    for model in tqdm(filtered_data):
        for run in filtered_data[model]:
            nan_mask = nan_mask | np.any(np.isnan(filtered_data[model][run]), axis=0)
    return nan_mask

def remove_nans(filtered_data, nan_mask):
    nan_filtered_data = filtered_data.copy()
    for model in tqdm(nan_filtered_data):
        for run in nan_filtered_data[model]:
            nan_filtered_data[model][run][:, nan_mask] = np.nan
    return nan_filtered_data

def reshape_data(data):
    reshaped_data = {}
    for model in tqdm(data):
        reshaped_data[model] = {}
        for run in data[model]:
            reshaped_data[model][run] = data[model][run].reshape(data[model][run].shape[0], -1)
    return reshaped_data

def center_data(filtered_data):
    centered_data = {}
    for model in tqdm(filtered_data):
        runs_stack = np.stack([filtered_data[model][run] for run in filtered_data[model]], axis=0)
        grid_average = np.nanmean(runs_stack, axis=(0, 1))
        centered_data[model] = {run: filtered_data[model][run] - grid_average for run in filtered_data[model]}
        forced_response = np.nanmean(np.stack(list(centered_data[model].values()), axis=0), axis=0)
        centered_data[model]['forced_response'] = forced_response
    return centered_data

def plot_time_series(time_series_data, forced_response_data, grid_spot, model_name):
    plt.figure(figsize=(10, 6))
    for data in time_series_data.values():
        plt.plot(data, color='blue', alpha=0.5)
    plt.plot(forced_response_data, color='red', label='Forced Response', linewidth=2)
    plt.title(f'Time Evolution at Grid Spot ({grid_spot[0]}, {grid_spot[1]}) for Model: {model_name}')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

### Pre processing for the reduced rank regression ###
def normalize_data(data):
    # Stack all runs from all models to compute global mean and std
    all_runs = []
    for model in data:
        for run in data[model]:
            all_runs.append(data[model][run])
    all_runs_stack = np.concatenate(all_runs, axis=0)
    
    # Compute mean and std for each grid square
    scaler = StandardScaler()
    scaler.fit(all_runs_stack.reshape(-1, all_runs_stack.shape[-1]))
    
    # Normalize data using the computed mean and std
    normalized_data = {}
    for model in tqdm(data):
        normalized_data[model] = {}
        for run in data[model]:
            reshaped_run = data[model][run].reshape(-1, data[model][run].shape[-1])
            normalized_run = scaler.transform(reshaped_run).reshape(data[model][run].shape)
            normalized_data[model][run] = normalized_run
    
    return normalized_data, scaler

def reduced_rank_regression(X, y, rank):

    # Fit OLS
    B_ols = np.linalg.pinv(X.T @ X) @ X.T @ y # Analytical solution

    # Compute SVD
    U, s, Vt = np.linalg.svd(X @ B_ols, full_matrices=False)

    # Truncate SVD to rank
    U_r = U[:, :rank]
    s_r = np.diag(s[:rank])
    Vt_r = Vt[:rank, :]

    # Compute B_rrr
    B_rrr = U_r @ s_r @ Vt_r

    return B_rrr

def add_forced_response(data):
    # We assume that the forced response is the average of all runs (for each model)
    data_with_forced_response = data.copy()
    for model in data_with_forced_response:
        runs_stack = np.stack([data_with_forced_response[model][run] for run in data_with_forced_response[model]], axis=0)
        forced_response = np.mean(runs_stack, axis=0)
        data_with_forced_response[model]['forced_response'] = forced_response
    return data_with_forced_response

def pool_data(data):
    X_all = []
    Y_all = []
    for model in data:
        for run in data[model]:
            X_all.append(data[model][run])
            Y_all.append(data[model]['forced_response'])
    X_all = np.concatenate(X_all, axis=0)
    Y_all = np.concatenate(Y_all, axis=0)
    return X_all, Y_all
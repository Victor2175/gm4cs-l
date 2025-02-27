import os
import pickle as pkl
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

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
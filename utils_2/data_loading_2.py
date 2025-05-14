import os
import pickle as pkl
import numpy as np
from tqdm import tqdm

def load_data(data_path, filename):
    """
    Load data from a pickle file.
    Args:
        data_path (_type_): _description_
        filename (_type_): _description_

    Returns:
        _type_: _description_
    """
    print(f"Loading data from {filename}")
    with open(os.path.join(data_path, filename), 'rb') as f:
        data = pkl.load(f)
    print("Data loaded successfully.")
    return data

def filter_data(data, min_runs=4):
    """
    Filter data to keep only models with at least min_runs runs.
    Args:
        data (dict): Dictionary containing the data.
        min_runs (int): Minimum number of runs to keep a model.
    
    Returns:
        dict: Filtered data.
    """
    print("Filtering data... also removing the latitudes above 60 degrees...")
    filtered_data = {
        model: {run: np.flip(data[model][run], axis=1)[:, 12:, :] for run in data[model]} # Skip the first 5 lines which cause Large MSE Values!
        for model in tqdm(data.keys()) if len(data[model]) >= min_runs
    }
    first_model = list(filtered_data.keys())[0]
    first_run = list(filtered_data[first_model].keys())[0]
    print(first_model, first_run, flush = True)
    print(f"Data has shape: {filtered_data[first_model][first_run].shape}")
    print(f"Data filtered. Kept {len(filtered_data)} models")
    return filtered_data
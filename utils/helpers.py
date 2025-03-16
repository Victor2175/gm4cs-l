import os
import pickle as pkl
import numpy as np
import random
import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import random
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

### Data Loading and Preprocessing ###

def load_data(data_path, filename):
    """
    Load data from a pickle file.
    Args:
        data_path (_type_): _description_
        filename (_type_): _description_

    Returns:
        _type_: _description_
    """
    with open(os.path.join(data_path, filename), 'rb') as f:
        data = pkl.load(f)
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
    filtered_data = {
        model: {run: np.flip(data[model][run], axis=1) for run in data[model]}
        for model in tqdm(data.keys()) if len(data[model]) >= min_runs
    }
    return filtered_data

def create_nan_mask(filtered_data):
    """
    Create a mask to identify NaN values in the data.
    Args:
        filtered_data (dict): Filtered data.
        
    Returns:
        np.array: Boolean mask with True values for NaN values.
    """
    first_model = list(filtered_data.keys())[0]
    first_run = list(filtered_data[first_model].keys())[0]
    grid_shape = filtered_data[first_model][first_run].shape[1:]
    nan_mask = np.zeros(grid_shape, dtype=bool)
    
    for model in tqdm(filtered_data):
        for run in filtered_data[model]:
            nan_mask = nan_mask | np.any(np.isnan(filtered_data[model][run]), axis=0)
    return nan_mask

def mask_out_nans(filtered_data, nan_mask):
    """
    Mask out NaN values from the data.
    Args:
        data (dict): Dictionary containing the data.
        nan_mask (np.array): Boolean mask with True values for NaN values.
        
    Returns:
        dict: Data with NaN values masked out.
    """
    # Modifications done in place
    for model in tqdm(filtered_data):
        for run in filtered_data[model]:
            filtered_data[model][run][:, nan_mask] = np.nan
    return filtered_data

def reshape_data(masked_data):
    """
    Reshape data to have a 2D shape.
    Args:
        data (dict): Dictionary containing the data.
        
    Returns:
        dict: Reshaped data.
    """
    for model, model_data in tqdm(masked_data.items()):
        for run, run_data in model_data.items():
            masked_data[model][run] = run_data.reshape(run_data.shape[0], -1)
    return masked_data

def center_data(filtered_data):
    """
    Center the data by removing the grid average.
    Args:
        filtered_data (dict): Filtered data.
        
    Returns:
        dict: Centered data.
    """
    centered_data = {}
    for model in tqdm(filtered_data):
        runs_stack = np.stack([filtered_data[model][run] for run in filtered_data[model]], axis=0)
        grid_average = np.nanmean(runs_stack, axis=(0, 1))
        centered_data[model] = {run: filtered_data[model][run] - grid_average for run in filtered_data[model]}
        forced_response = np.nanmean(np.stack(list(centered_data[model].values()), axis=0), axis=0)
        centered_data[model]['forced_response'] = forced_response
    return centered_data

### Pre processing for the reduced rank regression ###
def add_forced_response(data):
    """
    Add the forced response to the data.
    
    Args:
        data (dict): Dictionary containing the data.
        
    Returns:
    
    dict: Data with the forced response added.
    """
    # We assume that the forced response is the average of all runs (for each model)
    for model in tqdm(data):
        runs_stack = np.stack(list(data[model].values()), axis=0)
        forced_response = np.mean(runs_stack, axis=0)
        data[model]['forced_response'] = forced_response
    return data

def remove_nans_from_grid(data, nan_mask):
    """
    Remove NaN values from the data matrices

    Args:
        data (dict): Dictionary containing the data.
        nan_mask (np.ndarray): Boolean mask indicating NaN positions.
        
    Returns:
        dict: Data with NaN values removed.
    """
    mask = ~nan_mask
    for model in tqdm(data):
        for run in data[model]:
            data[model][run] = data[model][run][:, mask.ravel()] # Use Ravel instead of Flatten since it's in place
    return data

# def normalize_data(train_data, test_data):
#     """
#     Normalize the data using the mean and standard deviation of each model (for the training set)
#     Then normalize the testing data using the mean and std calculated over all runs in the training set.
#     Args:
#         train_data (dict): Dictionary containing the training data.
#         test_data (dict): Dictionary containing the test data.
        
#     Returns:
#         dict: Normalized training data.
#         dict: Normalized test data.
#         dict: Scalers used for normalization.
#     """
#     # Normalize training data using mean and std for each model separately
#     normalized_train_data = {}
#     training_statistics = {}
#     testing_statistics = {}
    
#     for model in tqdm(train_data):
#         model_runs = train_data[model]  # Store reference
#         all_runs = np.concatenate([model_runs[run] for run in model_runs], axis=0) # USE CONCATENATE (or torch.cat)
        
#         # Compute the mean and std for each grid square and each timestamp for the current model
#         mean_and_time = np.nanmean(all_runs, axis=(0, 1)) # Shape (d,)
#         std_ = np.nanstd(all_runs, axis=0) # Shape (T x d)
        
#         training_statistics[model] = {'mean': mean_and_time, 'std': std_}
        
#         # Normalize the training data for the current model (including the forced response)
#         normalized_train_data[model] = {run: (model_runs[run] - mean_and_time) / std_ for run in model_runs}
    
    
#     # Compute the mean and std for each grid square per time stamp but for all the models together
#     all_runs = np.concatenate([train_data[model][run] for model in train_data for run in train_data[model]], axis = 0)
    
#     full_mean_and_time = np.mean(all_runs, axis=(0, 1)) # Shape (d,)
#     full_std = np.nanstd(all_runs, axis=0) # Shape (T x d)
    
#     testing_statistics = {'mean': full_mean_and_time, 'std': full_std}
    
#     # Normalize the test data using the computed mean and std for all models together
#     normalized_test_data = {}
#     for model in tqdm(test_data):
#         normalized_test_data[model] = {}
#         for run in test_data[model]:
#             normalized_test_data[model][run] = (test_data[model][run] - full_mean_and_time) / full_std
        
#         test_runs = np.concatenate([test_data[model][run] for run in test_data[model]], axis=0) # Shape (# Runs x T x d)
#         test_mean = np.mean(test_runs, axis=(0, 1)) # Shape (d,)
#         test_std = np.std(test_runs, axis=0) # Shape (T x d)
        
#         # Apply the test mean and std to the forced response (MUST NOT BE USED ON THE RUNS)
#         normalized_test_data[model]['forced_response'] = (test_data[model]['forced_response'] - test_mean) / test_std
        
#     return normalized_train_data, normalized_test_data, training_statistics, testing_statistics
# Compute the mean and std for each grid square per time stamp but for all the models together
def normalize_data(train_data, test_data):
    """
    Normalize the data using the mean and standard deviation of each model (for the training set)
    Then normalize the testing data using the mean and std calculated over all runs in the training set.
    Args:
        train_data (dict): Dictionary containing the training data.
        test_data (dict): Dictionary containing the test data.
        
    Returns:
        dict: Normalized training data.
        dict: Normalized test data.
        dict: Scalers used for normalization.
    """
    # Normalize training data using mean and std for each model separately
    normalized_train_data = {}
    training_statistics = {}
    testing_statistics = {}
    
    # Store means before normalization for plotting
    means_before_normalization = []
    means_after_normalization = []

    for model in tqdm(train_data):
        model_runs = train_data[model]  # list of runs for the current model
        all_runs = np.stack([model_runs[run] for run in model_runs], axis=0) # shape (# Runs x T x d)
        # Compute the mean and std for each grid square and each timestamp for the current model
        mean_and_time = np.mean(all_runs, axis=(0, 1)) # Shape (d,)
        std_ = np.std(all_runs, axis=(0, 1)) # Shape (d) --> SHOULD BE (T x d) but for testing purposes we assume a cst std for the time dimension
        
        # Store statistics
        training_statistics[model] = {'mean': mean_and_time, 'std': std_}
        
        # Print mean, std, min, and max before normalization
        # print(f"Model: {model}")
        # print(f"Mean before normalization: {mean_and_time}")
        # # print(f"Std before normalization: {std_}")
        # print(f"Min before normalization: {np.min(all_runs)}")
        # print(f"Max before normalization: {np.max(all_runs)}")
        
        # Store mean before normalization for plotting
        means_before_normalization.extend(mean_and_time)
        
        # Normalize the training data for the current model (including the forced response)
        normalized_train_data[model] = {run: (model_runs[run] - mean_and_time) / std_ for run in model_runs}
   
        # Print mean, std, min, and max after normalization ##############
        
        all_runs_normalized = np.stack([normalized_train_data[model][run] for run in normalized_train_data[model]], axis=0)
        mean_after = np.mean(all_runs_normalized, axis=(0, 1))
        # print(f"Mean after normalization: {mean_after}")
        # # print(f"Std after normalization: {np.std(all_runs_normalized, axis=0)}")
        # print(f"Min after normalization: {np.min(all_runs_normalized)}")
        # print(f"Max after normalization: {np.max(all_runs_normalized)}")
        means_after_normalization.extend(mean_after)
            
        ###################################################################
        
    # Compute the mean and std for each grid square per time stamp but for all the models together
    all_runs = np.stack([train_data[model][run] for model in train_data for run in train_data[model]], axis = 0)

    full_mean_and_time = np.mean(all_runs, axis=(0, 1)) # Shape (d,)
    full_std = np.std(all_runs, axis=(0, 1)) # Shape (d,) --> SHOULD BE (T x d) but for testing purposes we assume a cst std for the time dimension
    
    ### Code works correctly until here ###
    
    testing_statistics = {'mean': full_mean_and_time, 'std': full_std}
    
    # Normalize the test data using the computed mean and std for all models together
    normalized_test_data = {}
    for model in tqdm(test_data):
        normalized_test_data[model] = {}
        for run in test_data[model]:
            if run != 'forced_response':
                normalized_test_data[model][run] = (test_data[model][run] - full_mean_and_time) / full_std
        
        test_runs = np.stack([test_data[model][run] for run in test_data[model]], axis=0) # Shape (# Runs x T x d)
        test_mean = np.mean(test_runs, axis=(0, 1)) # Shape (d,)
        test_std = np.std(test_runs, axis=(0, 1)) # Shape (d,) --> SHOULD BE (T x d) but for testing purposes we assume a cst std for the time dimension
        
        # Apply the test mean and std to the forced response (MUST NOT BE USED ON THE RUNS)
        normalized_test_data[model]['forced_response'] = (test_data[model]['forced_response'] - test_mean) / test_std
    
        
    # Plot means before and after normalization
    plt.figure(figsize=(12, 6))
    plt.plot(means_before_normalization, label='Before Normalization')
    plt.plot(means_after_normalization, label='After Normalization')
    plt.xlabel('Model')
    plt.ylabel('Mean Value')
    plt.title('Means Before and After Normalization')
    plt.legend()
    plt.show()
    
    return normalized_train_data, normalized_test_data, training_statistics, testing_statistics

def pool_data(data):
    """
    Pool data from different models and runs.
    Args:
        data (dict): Dictionary containing the data.
        
    Returns:
        np.array: Pooled input data.
        np.array: Pooled output data.
    """
    X_all = np.concatenate([data[model][run] for model in tqdm(data) for run in data[model]], axis=0)
    Y_all = np.concatenate([data[model]['forced_response'] for model in tqdm(data) for run in data[model]], axis=0)
    return X_all, Y_all

def readd_nans_to_grid(data, nan_mask, predictions=False):
    """
    Re-add NaN values to the data matrices for visualization purposes.
    
    Args:
        data (dict or np.ndarray): Dictionary containing the normalized data or a simple array if predictions is True.
        nan_mask (np.ndarray): Boolean mask indicating NaN positions.
        predictions (bool): Flag indicating if the data is a simple array (True) or a dictionary (False).
        
    Returns:
        dict or np.ndarray: Data with NaN values re-added.
    """
    nan_mask_flat = nan_mask.ravel()
    
    if predictions:
        # Handle the case where data is a simple array
        reshaped_data = np.full((data.shape[0], nan_mask_flat.shape[0]), np.nan)
        reshaped_data[:, ~nan_mask_flat] = data
        return reshaped_data
    else:
        # Handle the case where data is a dictionary
        for model in data:
            for run in data[model]:
                reshaped_run = np.full((data[model][run].shape[0], nan_mask_flat.shape[0]), np.nan)
                reshaped_run[:, ~nan_mask_flat] = data[model][run]
                data[model][run] = reshaped_run
        return data

def reduced_rank_regression(X, y, rank, lambda_):
    """
    Performs Reduced Rank Regression (RRR).

    X_all: (M*n, p) Combined input dataset from multiple simulations.
    Y_all: (M*n, q) Corresponding output responses.
    rank: Desired rank for dimensionality reduction.
    
    Returns:
    - B_rrr: (p, q) Reduced-rank weight matrix.
    """

    # Fit OLS
    identity = np.eye(X.shape[1])
    B_ols = np.linalg.inv(X.T @ X + lambda_ * identity) @ X.T @ y # Analytical solution of ridge regression (pseudo inverse)
    # Compute SVD
    U, s, Vt = np.linalg.svd(X @ B_ols, full_matrices=False)
    

    # Truncate SVD to rank
    # U_r = U[:, :rank]
    # s_r = np.diag(s[:rank])
    Vt_r = Vt[:rank, :]

    # Compute B_rrr
    B_rrr = B_ols @ Vt_r.T @ Vt_r # Reduced-rank weight matrix

    return B_rrr, B_ols

def calculate_mse(test_data, B_rrr, nan_mask, valid_indices): # FIX THIS
    """
    Calculate the Mean Squared Error (MSE) for the test data.
    Args:
        test_data (dict): Dictionary containing the test data.
        B_rrr (np.array): Reduced-rank weight matrix.
        
    Returns:
        float: Mean Squared Error.
    """
    test_model = list(test_data.keys())[0]
    test_runs = [run for run in test_data[test_model].keys() if run != 'forced_response']
    test_run = test_data[test_model][random.choice(test_runs)]
    ground_truth = test_data[test_model]['forced_response']

    # Make the prediction
    prediction = test_run @ B_rrr

    # Restore NaNs in the predicted matrix
    prediction = readd_nans(prediction, nan_mask, valid_indices)
    ground_truth = readd_nans(ground_truth, nan_mask, valid_indices)

    # Calculate MSE
    mse = mean_squared_error(ground_truth, prediction)
    return mse


def plot_time_series(time_series_data, forced_response_data, grid_spot, model_name):
    """
    Plot the time series data for a given grid spot.
    Args:
        time_series_data (dict): Dictionary containing the time series data.
        forced_response_data (np.array): Forced response data.
        grid_spot (tuple): Grid spot to plot.
        model_name (str): Name of the model.
    """
    plt.figure(figsize=(10, 6))
    for data in time_series_data.values():
        plt.plot(data, color='blue', alpha=0.5)
    plt.plot(forced_response_data, color='red', label='Forced Response', linewidth=2)
    plt.title(f'Time Evolution at Grid Spot ({grid_spot[0]}, {grid_spot[1]}) for Model: {model_name}')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.show()
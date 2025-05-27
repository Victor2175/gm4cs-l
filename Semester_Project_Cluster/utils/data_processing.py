import numpy as np
from tqdm import tqdm

def create_nan_mask(filtered_data):
    """
    Create a mask to identify NaN values in the data.
    Args:
        filtered_data (dict): Filtered data.
        
    Returns:
        np.array: Boolean mask with True values for NaN values.
    """
    print("Creating NaN mask...", flush = True)
    first_model = list(filtered_data.keys())[0]
    first_run = list(filtered_data[first_model].keys())[0]
    grid_shape = filtered_data[first_model][first_run].shape[1:]
    nan_mask = np.zeros(grid_shape, dtype=bool)
    
    for model in tqdm(filtered_data):
        for run in filtered_data[model]:
            nan_mask = nan_mask | np.any(np.isnan(filtered_data[model][run]), axis=0)
    print("NaN mask created.", flush = True)
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
    print("Masking out NaN values...", flush = True)
    for model in tqdm(filtered_data):
        for run in filtered_data[model]:
            filtered_data[model][run][:, nan_mask] = np.nan
    print("NaN values masked out.", flush = True)
    return filtered_data

def reshape_data(masked_data):
    """
    Reshape data to have a 2D shape.
    Args:
        data (dict): Dictionary containing the data.
        
    Returns:
        dict: Reshaped data.
    """
    print("Reshaping data...", flush = True)
    for model, model_data in tqdm(masked_data.items()):
        for run, run_data in model_data.items():
            masked_data[model][run] = run_data.reshape(run_data.shape[0], -1)
    print("Data reshaped.", flush = True)
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
        grid_std = np.nanstd(runs_stack, axis=(0, 1))
        centered_data[model] = {run: (filtered_data[model][run] - grid_average) / grid_std for run in filtered_data[model]}
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
    print("Adding the forced response to the data...", flush = True)
    for model in tqdm(data):
        runs_stack = np.stack(list(data[model].values()), axis=0)
        forced_response = np.mean(runs_stack, axis=0)
        data[model]['forced_response'] = forced_response
    print("Forced response added.", flush = True)
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
    print("Removing NaN values from the grid...", flush = True)
    mask = ~nan_mask
    for model in tqdm(data):
        for run in data[model]:
            data[model][run] = data[model][run][:, mask.ravel()] # Use Ravel instead of Flatten since it's in place
    print("NaN values removed.", flush = True)
    return data

def normalize_data(train_data, test_data, center=True, option=1):
    """
    Normalize the data using the mean and standard deviation of each model (for the training set)
    Then normalize the testing data using the mean and std calculated over all runs in the training set.
    Args:
        train_data (dict): Dictionary containing the training data.
        test_data (dict): Dictionary containing the test data.
        
    Returns:
        dict: Normalized training data.
        dict: Normalized test data.
        dict: Training statistics (mean and std if applicable).
        dict: Testing statistics per model (mean and std for each model in test_data).
    """
    print("\nNormalizing data...", flush=True)
    # Normalize training data using mean and std for each model separately
    normalized_train_data = {}
    training_statistics = {}

    for model in tqdm(train_data):
        model_runs = train_data[model]  # list of runs for the current model
        all_runs = np.stack([model_runs[run] for run in model_runs], axis=0)  # shape (# Runs x T x d)

        # Compute the mean and std for each grid square and each timestamp for the current model
        if option == 1:
            mean_and_time = np.mean(all_runs, axis=(0, 1))  # Shape (d,) --> Center in Runs and Time for training data!!
        else:
            mean_and_time = np.mean(all_runs, axis=1)  # Shape (#Runs, d)
        print(f"SHAPE: {mean_and_time.shape}", flush=True)
        if not center:
            std_ = np.std(all_runs, axis=(0, 1))  # Shape (d,) this is done since using axis = 0 gives very small values which leads to instability
            # Store the mean and std for the current model
            training_statistics[model] = {'mean': mean_and_time, 'std': std_}
        else:
            # Only store the mean since the std is very inaccurate on the test set
            training_statistics[model] = {'mean': mean_and_time}

        # Normalize the training data for the current model (including the forced response)
        if not center:
            normalized_train_data[model] = {run: (model_runs[run] - mean_and_time) / std_ for run in model_runs}
        else:
            # Center the data with the mean
            normalized_train_data[model] = {run: (model_runs[run] - mean_and_time) for run in model_runs}

    # Normalize the test data using the mean and std for each run separately
    normalized_test_data = {}
    testing_statistics = {}  # Store per-run testing statistics

    for model in tqdm(test_data):
        normalized_test_data[model] = {}
        testing_statistics[model] = {}
        for run in test_data[model]:
            if run != 'forced_response':
                # Calculate mean and std for each run
                run_mean = np.mean(test_data[model][run], axis=0)  # Shape (d,)
                run_std = np.std(test_data[model][run], axis=0)  # Shape (d,)

                # Store the mean and std for this run
                testing_statistics[model][run] = {'mean': run_mean, 'std': run_std}

                # Normalize the test data for this run
                normalized_test_data[model][run] = (test_data[model][run] - run_mean) / run_std
                
        test_runs = np.stack([test_data[model][run] for run in test_data[model] if run != 'forced_response'], axis=0) # Shape (# Runs x T x d)
        test_std = np.std(test_runs, axis=(0, 1))  # Shape (d,)
        test_mean = np.mean(test_runs, axis=(0, 1))  # Shape (d,)
        
        # Store the test statistics for this model
        testing_statistics[model] = {'mean': test_mean, 'std': test_std}
        
        # Handle the forced response
        if 'forced_response' in test_data[model]:
            forced_response = test_data[model]['forced_response']
            if center:
                forced_response_mean = np.mean(forced_response, axis=0)
                normalized_test_data[model]['forced_response'] = forced_response - forced_response_mean
            else:
                normalized_test_data[model]['forced_response'] = forced_response

    print("Data normalization completed.", flush=True)

    return normalized_train_data, normalized_test_data, training_statistics, testing_statistics

def pool_data(data):
    print("\nPooling data...", flush = True)
    X_all_list = []
    Y_all_list = []

    for model, model_data in tqdm(data.items()):  # Cache `data[model]` as `model_data`
        forced_response = model_data['forced_response']  # Cache 'forced_response'
        for run, run_data in model_data.items():
            if run == 'forced_response':  # Skip 'forced_response' key
                continue
            X_all_list.append(run_data)
            Y_all_list.append(forced_response)

    print("Data pooled.", flush = True)
    return np.concatenate(X_all_list, axis=0), np.concatenate(Y_all_list, axis=0)


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
    print("Re-adding NaN values to the grid...", flush = True)
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
    print("NaN values re-added.", flush = True)
    return data
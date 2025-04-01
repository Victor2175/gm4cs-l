from data_loading import *
from data_processing import *
from regression import *
from metrics import *

def preprocess_data(data_path, filename, min_runs=4):
    """
    Preprocess the data by performing all the necessary steps in one call.
    
    Args:
        data_path (str): Path to the data directory.
        filename (str): Name of the data file.
        min_runs (int): Minimum number of runs to keep a model.
        
    Returns:
        dict: Preprocessed data.
        np.ndarray: NaN mask.
    """
    # Load the data
    data = load_data(data_path, filename)
    
    # Filter the data
    filtered_data = filter_data(data, min_runs)
    
    # Create a NaN mask
    nan_mask = create_nan_mask(filtered_data)
    
    # Mask out NaNs
    masked_data = mask_out_nans(filtered_data, nan_mask)
    
    # Reshape the data so that each run is of shape (T, d)
    reshaped_data = reshape_data(masked_data)
    
    # Add the forced response, IMPORTANT to do it before normalizing
    data_with_forced_response = add_forced_response(reshaped_data)
    
    # Remove NaNs from the grid
    data_without_nans = remove_nans_from_grid(data_with_forced_response, nan_mask)
    
    return data_without_nans, nan_mask

def loo_cross_validation(data, lambdas, ranks):
    """
    Perform leave-one-out cross-validation to get a distribution of the MSE for different values of lambda.
    
    Args:
        data (dict): Preprocessed data without NaNs.
        lambdas (list): List of lambda values to test.
        ranks (list): List of rank values to test.
        
    Returns:
        dict: Dictionary containing the MSE distribution for each lambda.
    """
    models = list(data.keys())
    # Initialize the MSE distribution dictionary
    mse_distribution = {model: {rank: {lambda_: [] for lambda_ in lambdas} for rank in ranks} for model in models}
    
    for test_model in tqdm(models):
        # Split the data into training and testing sets according to the leave-one-out scheme
        train_models = [model for model in models if model != test_model]
        train_data = {model: data[model] for model in train_models}
        test_data = {test_model: data[test_model]}
        
        # Assert test model not in train models
        assert test_model not in train_models, f"Test model {test_model} found in training models."
        # Assert train models not in test models
        assert all(model not in test_data for model in train_models), f"Training models {train_models} found in test data."
        # Assert test model not in test data
        assert test_model in test_data, f"Test model {test_model} not found in test data."
        # Assert train data not empty
        assert len(train_data) > 0, f"Training data for model {test_model} is empty."
        # Assert test data not empty
        assert len(test_data) > 0, f"Test data for model {test_model} is empty."
        
        # Normalize the data
        normalized_train_data, normalized_test_data, _, _ = normalize_data(train_data, test_data)
        
        # Pool the training data
        X_train, Y_train = pool_data(normalized_train_data)
        
        print("Performing leave-one-out cross-validation for model:", test_model)
        
        # Get the test runs and ground truth
        test_runs = [run for run in normalized_test_data[test_model].keys() if run != 'forced_response']
        ground_truth = normalized_test_data[test_model]['forced_response']
        
        for rank in ranks:
            for lambda_ in lambdas:
                # Perform reduced rank regression
                B_rrr, B_ols = reduced_rank_regression(X_train, Y_train, rank, lambda_)
                
                # Perform Sanity check
                _ = sanity_check(B_rrr, B_ols, rank, True)
                
                # Calculate the MSE for each test run
                mse_values = []
                for run in test_runs:
                    test_run_data = normalized_test_data[test_model][run]
                    mse = calculate_mse(test_run_data, B_rrr, ground_truth)
                    mse_values.append(mse)
                
                # Store the MSE values for the current model, rank, and lambda
                mse_distribution[test_model][rank][lambda_].extend(mse_values)
    
    return mse_distribution
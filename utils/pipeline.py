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

def loo_cross_validation(data, lambdas, rank=15):
    """
    Perform leave-one-out cross-validation to get a distribution of the MSE for different values of lambda.
    
    Args:
        data (dict): Preprocessed data without NaNs.
        lambdas (list): List of lambda values to test.
        rank (int): Desired rank for dimensionality reduction.
        
    Returns:
        dict: Dictionary containing the MSE distribution for each lambda.
    """
    mse_distribution = {lambda_: {} for lambda_ in lambdas}
    models = list(data.keys())
    
    for test_model in tqdm(models):
        # Split the data into training and testing sets according to the leave-one-out scheme
        train_models = [model for model in models if model != test_model]
        train_data = {model: data[model] for model in train_models}
        test_data = {test_model: data[test_model]}
        
        # Normalize the data
        normalized_train_data, normalized_test_data, _, _ = normalize_data(train_data, test_data)
        
        # Pool the training data
        X_train, Y_train = pool_data(normalized_train_data)
        
        print("Performing leave-one-out cross-validation for model:", test_model)
        
        # Get the test runs and ground truth
        test_runs = [run for run in normalized_test_data[test_model].keys() if run != 'forced_response']
        ground_truth = normalized_test_data[test_model]['forced_response']
        
        for lambda_ in lambdas:
            # Perform reduced rank regression
            B_rrr, _ = reduced_rank_regression(X_train, Y_train, rank, lambda_)
            
            # Calculate the MSE for each test run
            mse_values = []
            for run in test_runs:
                test_run_data = normalized_test_data[test_model][run]
                mse = calculate_mse(test_run_data, B_rrr, ground_truth)
                mse_values.append(mse)
            
            # Store the MSE values for the current model and lambda
            if test_model not in mse_distribution[lambda_]:
                mse_distribution[lambda_][test_model] = []
            mse_distribution[lambda_][test_model].extend(mse_values)
    
    return mse_distribution
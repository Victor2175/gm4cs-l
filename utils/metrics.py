from sklearn.metrics import mean_squared_error
from scipy.sparse import issparse
from numpy.linalg import matrix_rank
from tqdm import tqdm
import numpy as np
import torch

def calculate_mse(run_data, B_rrr, ground_truth, testing_statistics=None, model=None, normalise=False):
    """
    Calculate the Mean Squared Error (MSE) for a single run of the test data.
    Args:
        run_data (np.array): array of shape (T, d).
        B_rrr (np.array): Reduced-rank weight matrix.
        ground_truth (np.array): Ground truth data of shape (T, d).
        testing_statistics (dict, optional): Dictionary containing model-specific statistics with 'std' for each test model.
        model (str, optional): The name of the model being evaluated, used to select the appropriate std from testing_statistics.
        
    Returns:
        float: Mean Squared Error (normalized if testing_statistics and model are provided).
    """
    # Compute the predicted response
    y_pred = run_data @ B_rrr
    
    if normalise and testing_statistics and model and model in testing_statistics and 'std' in testing_statistics[model]:
        # Calculate the normalized Mean Squared Error using the model-specific standard deviation
        normalized_diff = (y_pred - ground_truth) / testing_statistics[model]['std']
        mse = np.mean(normalized_diff ** 2)
        # print(f"The std for {model} is {testing_statistics[model]['std']}")
        # print(f"Mean Squared Error for {model}: {mse}")
        # print("NORMALISED MSE")
    else:
        # Calculate the standard Mean Squared Error
        mse = mean_squared_error(ground_truth, y_pred)
        # print("Normal MSE")
    
    return mse

def calculate_mse_distribution(normalized_train_data, Brr, testing_statistics=None):
    """
    Calculate the MSE distribution for each model in the training data.

    Args:
        normalized_train_data (dict): Normalized training data.
        Brr (np.array): Reduced-rank weight matrix.
        testing_statistics (dict, optional): Dictionary containing 'std' for the test set.
        
    Returns:
        dict: MSE values for each model.
    """
    mse_values_train = {}
    
    for model in tqdm(normalized_train_data):
        mse_values_train[model] = []
        ground_truth = normalized_train_data[model]['forced_response']
        
        for run in normalized_train_data[model]:
            if run == 'forced_response':
                continue
            test_run = normalized_train_data[model][run]
            
            # Calculate the MSE
            run_mse = calculate_mse(test_run, Brr, ground_truth, testing_statistics, model)
            
            mse_values_train[model].append(run_mse)
    
    return mse_values_train

def is_sparse(B_rr):
    """
    Check if the input matrix is sparse.
    Args:
        B_rr (np.array): Input matrix.
    Returns:
        bool: True if the input matrix is sparse, False otherwise.
    """
    return issparse(B_rr)

def get_rank(B_rr):
    """
    Get the rank of the input matrix.
    Args:
        B_rr (np.array): Input matrix.
    Returns:
        int: Rank of the input matrix.
    """
    return matrix_rank(B_rr)

def sanity_check(B_rr, B_ols, rank, cross_validation = True):
    """
    Perform a sanity check on the reduced-rank weight matrix.
    Args:
        B_rr (np.array): Reduced-rank weight matrix.
        rank (int): Expected rank of the matrix.
        B_ols (np.array): Ordinary least squares weight matrix.
    Returns:
        bool: True if the matrix passes all sanity checks, False otherwise.
    """
    sparse = is_sparse(B_rr)
    rank_B = get_rank(B_rr)
    if not cross_validation:
        assert rank_B == rank, "The rank is not constrained correctly!"  # Error message in case the rank is not equal to the expected value
    rank_B_ols = get_rank(B_ols)
    mean_B = np.mean(B_rr)
    std_B = np.std(B_rr)
    
    if sparse or rank_B != rank:
        print(f"Is B_rr sparse: {sparse}")
        print(f"The rank of Bols is {rank_B_ols} and the rank of B_rr is {rank_B}.")
    
    return sparse and rank_B == rank

def evaluate_vae(model, data_loader, device, testing_statistics=None):
    """
    Evaluate the VAE model by calculating the Mean Squared Error (MSE) for predictions.

    Args:
        model (torch.nn.Module): The VAE model to evaluate.
        data_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        device (torch.device): The device to run the evaluation on (CPU or GPU).
        testing_statistics (dict, optional): Dictionary containing model-specific statistics with 'std' for each test model.

    Returns:
        dict: A dictionary containing both standard and normalized MSE values.
    """
    model.eval()  # Set the model to evaluation mode
    mse_scores = []
    normalized_mse_scores = []

    with torch.no_grad():  # Disable gradient computation
        for batch in data_loader:
            inputs = batch['input'].to(device)
            outputs = batch['output'].to(device)

            # Forward pass through the model
            reconstructed, _, _ = model(inputs)

            # Reshape tensors to 2D for MSE calculation
            prediction = reconstructed.cpu().numpy().reshape(reconstructed.shape[0], -1)
            target = outputs.cpu().numpy().reshape(outputs.shape[0], -1)

            # Calculate standard MSE for the batch
            batch_mse = mean_squared_error(target, prediction)
            mse_scores.append(batch_mse)

            # Calculate normalized MSE if testing_statistics is provided
            if testing_statistics:
                model_name = batch.get('model_name', None)  # Assuming batch contains model_name
                if model_name in testing_statistics and 'std' in testing_statistics[model_name]:
                    print(f"Calculating normalized MSE for model: {model_name}")
                    std = testing_statistics[model_name]['std']
                    normalized_diff = (prediction - target) / std
                    batch_normalized_mse = np.mean(normalized_diff ** 2)
                    normalized_mse_scores.append(batch_normalized_mse)

    # Calculate the average MSE and normalized MSE across all batches
    average_mse = np.mean(mse_scores)
    average_normalized_mse = np.mean(normalized_mse_scores) if normalized_mse_scores else None

    return {
        'mse': average_mse,
        'normalized_mse': average_normalized_mse
    }
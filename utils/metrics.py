from sklearn.metrics import mean_squared_error
from scipy.sparse import issparse
from numpy.linalg import matrix_rank
from tqdm import tqdm
import numpy as np

def calculate_mse(run_data, B_rrr, ground_truth):
    """
    Calculate the Mean Squared Error (MSE) for a single run of the test data.
    Args:
        run_data (np.array): array of shape (T, d).
        B_rrr (np.array): Reduced-rank weight matrix.
        ground_truth (np.array): Ground truth data of shape (T, d).
        
    Returns:
        float: Mean Squared Error.
    """
    # Compute the predicted response
    y_pred = run_data @ B_rrr
    
    # Calculate the Mean Squared Error
    mse = mean_squared_error(ground_truth, y_pred)
    
    return mse

def calculate_mse_distribution(normalized_train_data, Brr):
    """_summary_

    Args:
        normalized_train_data (_type_): _description_
        Brr (_type_): _description_
    Returns:
        _type_: _description_
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
            run_mse = calculate_mse(test_run, Brr, ground_truth)
            
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

def sanity_check(B_rr, B_ols, rank):
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
    assert rank_B == rank, "The rank is not constrained correctly!"  # Error message in case the rank is not equal to the expected value
    rank_B_ols = get_rank(B_ols)
    mean_B = np.mean(B_rr)
    std_B = np.std(B_rr)
    
    print(f"Is B_rr sparse: {sparse}")
    print(f"The rank of Bols is {rank_B_ols} and the rank of B_rr is {rank_B}.")
    print(f"The mean of B_rr is {mean_B} and the std is {std_B}.")
    
    return sparse and rank_B == rank
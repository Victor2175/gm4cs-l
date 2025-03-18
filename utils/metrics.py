from sklearn.metrics import mean_squared_error
from tqdm import tqdm

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
    avg_mse_values_train = {}
    
    for model in tqdm(normalized_train_data):
        avg_mse_values_train[model] = []
        ground_truth = normalized_train_data[model]['forced_response']
        
        for run in normalized_train_data[model]:
            if run == 'forced_response':
                continue
            test_run = normalized_train_data[model][run]
            
            # Calculate the MSE
            run_mse = calculate_mse(test_run, Brr, ground_truth)
            
            avg_mse_values_train[model].append(run_mse)
    
    return avg_mse_values_train
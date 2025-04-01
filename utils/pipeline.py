from data_loading import *
from data_processing import *
from regression import *
from metrics import *
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import seaborn as sns
import os

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
    mse_by_combination = {(rank, lambda_): [] for rank in ranks for lambda_ in lambdas}
    for test_model in tqdm(models):
        # Split the data into training and testing sets according to the leave-one-out scheme
        train_models = [model for model in models if model != test_model]
        train_data = {model: data[model] for model in train_models}
        test_data = {test_model: data[test_model]}
        
        # Assertions to ensure data integrity
        assert test_model not in train_models, f"Test model {test_model} found in training models."
        assert all(model not in test_data for model in train_models), f"Training models {train_models} found in test data."
        assert test_model in test_data, f"Test model {test_model} not found in test data."
        assert len(train_data) > 0, f"Training data for model {test_model} is empty."
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
                for run in test_runs:
                    test_run_data = normalized_test_data[test_model][run]
                    mse = calculate_mse(test_run_data, B_rrr, ground_truth)
                    
                    # Store the MSE values in both dictionaries
                    mse_distribution[test_model][rank][lambda_].append(mse)
                    mse_by_combination[(rank, lambda_)].append(mse)
    return mse_distribution, mse_by_combination

def plot_mse_distributions(mse_by_combination, ranks, lambdas, output_dir=None):
    """
    Plot the MSE distributions for each (rank, lambda) combination with density estimation and variance.
    
    Args:
        mse_by_combination (dict): Dictionary containing MSE distributions for each (rank, lambda) combination.
        ranks (list): List of rank values.
        lambdas (list): List of lambda values.
        output_dir (str): Directory to save the plots. If None, plots are displayed interactively.
    """
    plt.figure(figsize=(15, 10))
    for i, rank in enumerate(ranks):
        for j, lambda_ in enumerate(lambdas):
            plt.subplot(len(ranks), len(lambdas), i * len(lambdas) + j + 1)
            mse_values = mse_by_combination[(rank, lambda_)]
            
            # Plot density estimation and boxplot
            sns.kdeplot(mse_values, fill=True, alpha=0.5, label="Density")
            plt.boxplot(mse_values, vert=True, patch_artist=True, positions=[0.5], widths=0.2)
            
            # Calculate and display variance
            variance = np.var(mse_values)
            plt.title(f"Rank: {rank}, Lambda: {lambda_}\nVariance: {variance:.4f}")
            plt.xlabel("MSE")
            plt.ylabel("Density")
            plt.grid(True)
    
    plt.tight_layout()
    
    # Save or show the plot
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, "mse_distributions.png")
        plt.savefig(plot_path)
        print(f"Saved MSE distribution plot at {plot_path}")
    else:
        plt.show()
    plt.close()
    return None

def select_robust_hyperparameters(mse_by_combination, mean_weight, variance_weight, output_dir, verbose = False):
    """
    Select the most robust combination of rank and lambda based on a weighted combination of mean MSE and variance.
    
    Args:
        mse_by_combination (dict): Dictionary containing MSE distributions for each (rank, lambda) combination.
        mean_weight (float): Weight for the mean MSE in the optimization.
        variance_weight (float): Weight for the variance in the optimization.
        output_dir (str): Directory to save the best hyperparameters.
        
    Returns:
        tuple: Best (rank, lambda) combination and its weighted score.
    """
    best_rank_lambda = None
    best_score = float('inf')  # Initialize with a very high score

    for (rank, lambda_), mse_values in mse_by_combination.items():
        mean_mse = np.mean(mse_values)
        variance_mse = np.var(mse_values)

        # Calculate the weighted score
        score = mean_weight * mean_mse + variance_weight * variance_mse

        if verbose:
            print(f"Rank: {rank}, Lambda: {lambda_}, Mean MSE: {mean_mse:.4f}, Variance: {variance_mse:.4f}, Score: {score:.4f}")
    
        # Select the combination with the lowest score
        if score < best_score:
            best_score = score
            best_rank_lambda = (rank, lambda_)

    if best_rank_lambda is None:
        raise ValueError("No valid combination found. Check the input data.")

    # Save the best hyperparameters to a .txt file
    os.makedirs(output_dir, exist_ok=True)
    best_hyperparams_path = os.path.join(output_dir, "best_hyperparameters.txt")
    with open(best_hyperparams_path, "w") as f:
        f.write(f"Best Rank: {best_rank_lambda[0]}\n")
        f.write(f"Best Lambda: {best_rank_lambda[1]}\n")
        f.write(f"Weighted Score: {best_score:.4f}\n")
    print(f"Saved best hyperparameters at {best_hyperparams_path}")

    return best_rank_lambda, best_score

def plot_mse_distributions_per_model(mse_distributions, models, ranks, lambdas, output_dir):
    """
    Plot and save the MSE distributions for each model using boxplots.
    
    Args:
        mse_distributions (dict): Dictionary containing MSE distributions for each model, rank, and lambda.
        models (list): List of model names.
        ranks (list): List of rank values.
        lambdas (list): List of lambda values.
        output_dir (str): Directory to save the plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for model in models:
        plt.figure(figsize=(12, 6))
        for i, rank in enumerate(ranks):
            plt.subplot(1, len(ranks), i + 1)
            data_to_plot = [mse_distributions[model][rank][lambda_] for lambda_ in lambdas]
            plt.boxplot(data_to_plot, labels=[f'Lambda: {lambda_}' for lambda_ in lambdas], patch_artist=True)
            plt.xlabel('Lambda')
            plt.ylabel('MSE')
            plt.title(f'Rank: {rank}')
            plt.grid(True)
        plt.suptitle(f'MSE Distributions for Model: {model}')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save the plot
        plot_path = os.path.join(output_dir, f"mse_distributions_{model}.png")
        plt.savefig(plot_path)
        plt.close()  # Close the plot to free memory
        print(f"Saved MSE distribution plot for model {model} at {plot_path}")
    return None


def final_cross_validation(data, best_rank, best_lambda):
    """
    Perform a final round of cross-validation using the best rank and lambda.
    
    Args:
        data (dict): Preprocessed data without NaNs.
        best_rank (int): The best rank value.
        best_lambda (float): The best lambda value.
        
    Returns:
        list: List of MSE losses for each test model.
    """
    models = list(data.keys())
    mse_losses = []
    
    for test_model in tqdm(models, desc="Final Cross-Validation"):
        # Split the data into training and testing sets
        train_models = [model for model in models if model != test_model]
        train_data = {model: data[model] for model in train_models}
        test_data = {test_model: data[test_model]}
        
        # Normalize the data
        normalized_train_data, normalized_test_data, _, _ = normalize_data(train_data, test_data)
        
        # Pool the training data
        X_train, Y_train = pool_data(normalized_train_data)
        
        # Perform reduced rank regression
        B_rrr, _ = reduced_rank_regression(X_train, Y_train, rank=best_rank, lambda_=best_lambda)
        
        # Evaluate on the test model
        test_runs = [run for run in normalized_test_data[test_model].keys() if run != 'forced_response']
        ground_truth = normalized_test_data[test_model]['forced_response']
        
        for run in test_runs:
            test_run_data = normalized_test_data[test_model][run]
            mse = calculate_mse(test_run_data, B_rrr, ground_truth)
            mse_losses.append(mse)
    
    return mse_losses

def final_cross_validation(data, best_rank, best_lambda):
    """
    Perform a final round of cross-validation using the best rank and lambda.
    
    Args:
        data (dict): Preprocessed data without NaNs.
        best_rank (int): The best rank value.
        best_lambda (float): The best lambda value.
        
    Returns:
        dict: Dictionary containing MSE losses for each test model.
    """
    models = list(data.keys())
    mse_losses = {model: [] for model in models}
    
    for test_model in tqdm(models, desc="Final Cross-Validation"):
        # Split the data into training and testing sets
        train_models = [model for model in models if model != test_model]
        train_data = {model: data[model] for model in train_models}
        test_data = {test_model: data[test_model]}
        
        # Normalize the data
        normalized_train_data, normalized_test_data, _, _ = normalize_data(train_data, test_data)
        
        # Pool the training data
        X_train, Y_train = pool_data(normalized_train_data)
        
        # Perform reduced rank regression
        B_rrr, _ = reduced_rank_regression(X_train, Y_train, rank=best_rank, lambda_=best_lambda)
        
        # Evaluate on the test model
        test_runs = [run for run in normalized_test_data[test_model].keys() if run != 'forced_response']
        ground_truth = normalized_test_data[test_model]['forced_response']
        
        for run in test_runs:
            test_run_data = normalized_test_data[test_model][run]
            mse = calculate_mse(test_run_data, B_rrr, ground_truth)
            mse_losses[test_model].append(mse)
    
    return mse_losses


def plot_final_mse_distribution(mse_losses, output_dir):
    """
    Plot and save the final MSE distribution as a boxplot, with the worst-performing model annotated.
    
    Args:
        mse_losses (dict): Dictionary containing MSE losses for each test model.
        output_dir (str): Directory to save the plot.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Flatten the MSE losses and track the corresponding models
    all_mse = []
    model_labels = []
    for model, losses in mse_losses.items():
        all_mse.extend(losses)
        model_labels.extend([model] * len(losses))
    
    # Identify the worst-performing model
    worst_model = max(mse_losses, key=lambda model: np.mean(mse_losses[model]))
    worst_model_mean_mse = np.mean(mse_losses[worst_model])
    
    # Identify the best-performing model
    best_model = min(mse_losses, key=lambda model: np.mean(mse_losses[model]))
    best_model_mean_mse = np.mean(mse_losses[best_model])
    
    # Calculate overall statistics
    overall_mean_mse = np.mean(all_mse)
    overall_variance_mse = np.var(all_mse)
    
    # Create the boxplot
    plt.figure(figsize=(8, 6))
    plt.boxplot(all_mse, patch_artist=True)  # To create filled boxes
    plt.title("Final MSE Distribution")
    plt.ylabel("MSE")
    plt.grid(True)
    
    # Annotate the worst-performing model
    plt.text(1.1, worst_model_mean_mse, f"Worst Model: {worst_model}\nMean MSE: {worst_model_mean_mse:.4f}", 
             fontsize=10, color='red', bbox=dict(facecolor='white', alpha=0.8))
    
    # Annotate the best-performing model
    plt.text(1.1, best_model_mean_mse, f"Best Model: {best_model}\nMean MSE: {best_model_mean_mse:.4f}", 
             fontsize=10, color='green', bbox=dict(facecolor='white', alpha=0.8))
    
    # Display overall statistics
    plt.figtext(0.15, 0.02, f"Overall Mean MSE: {overall_mean_mse:.4f}\nOverall Variance: {overall_variance_mse:.4f}", 
                fontsize=10, color='blue', ha='left', bbox=dict(facecolor='white', alpha=0.8))
    
    # Save the plot
    plot_path = os.path.join(output_dir, "final_mse_distribution.png")
    plt.savefig(plot_path)
    plt.close()  # Close the plot to free memory
    print(f"Saved final MSE distribution plot at {plot_path}")
    return None
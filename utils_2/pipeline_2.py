from utils_2.data_loading_2 import *
from utils_2.data_processing_2 import *
from utils_2.regression_2 import *
from utils_2.metrics_2 import *
from matplotlib import pyplot as plt
from tqdm import tqdm
from utils_2.animation_2 import plot_animations
from utils_2.data_processing_2 import normalize_data, pool_data
from utils_2.regression_2 import reduced_rank_regression
import numpy as np
import seaborn as sns
import os, psutil, gc
import pickle  # Added import for pickle

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

def log_memory_usage(message=""):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"{message} - Memory usage: {mem_info.rss / 1024 ** 2:.2f} MB (RSS)", flush=True)

def loo_cross_validation(data, lambdas, ranks, center=True, use_ols_only=False, output_dir=None, normalise=False, option=1):
    models = list(data.keys())
    mse_distribution = {model: {rank: {lambda_: [] for lambda_ in lambdas} for rank in ranks} for model in models}
    mse_by_combination = {(rank, lambda_): [] for rank in ranks for lambda_ in lambdas}

    for test_model in tqdm(models):
        log_memory_usage(f"Before processing model {test_model}")
        # Split the data into training and testing sets
        train_models = [model for model in models if model != test_model]
        train_data = {model: data[model] for model in train_models}
        test_data = {test_model: data[test_model]}

        # Normalize the data
        normalized_train_data, normalized_test_data, _, testing_statistics = normalize_data(train_data, test_data, center=center, option=option)
        log_memory_usage(f"After normalization for model {test_model}")

        # Pool the training data
        X_train, Y_train = pool_data(normalized_train_data)
        log_memory_usage(f"After pooling data for model {test_model}")

        for rank in ranks:
            for lambda_ in lambdas:
                # Perform reduced rank regression
                B_rrr, B_ols = reduced_rank_regression(X_train, Y_train, rank, lambda_, use_ols_only=use_ols_only)
                log_memory_usage(f"After regression for rank {rank}, lambda {lambda_}")

                # Calculate the MSE for each test run
                test_runs = [run for run in normalized_test_data[test_model].keys() if run != 'forced_response']
                ground_truth = normalized_test_data[test_model]['forced_response']
                for run in test_runs:
                    test_run_data = normalized_test_data[test_model][run]
                    mse = calculate_mse(test_run_data, B_rrr, ground_truth, testing_statistics, test_model, normalise)
                    mse_distribution[test_model][rank][lambda_].append(mse)
                    mse_by_combination[(rank, lambda_)].append(mse)
                # Clear memory for the current test model
                del B_rrr, B_ols
                gc.collect()
            
        # Explicitly delete large variables and force garbage collection after processing each model
        del normalized_train_data, normalized_test_data, X_train, Y_train, train_data, test_data
        gc.collect()

        log_memory_usage(f"After processing model {test_model}")
    gc.collect()
    
    # Save the metrics as pickle files if output_dir is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine method name based on use_ols_only flag
        method_name = "OLS_results" if use_ols_only else "RRR_results"
        
        # Save mse_distribution
        mse_distribution_path = os.path.join(output_dir, f"{method_name}_mse_distribution.pkl")
        with open(mse_distribution_path, 'wb') as f:
            pickle.dump(mse_distribution, f)
        print(f"Saved MSE distribution at {mse_distribution_path}", flush=True)
        
        # Save mse_by_combination
        mse_by_combination_path = os.path.join(output_dir, f"{method_name}_mse_by_combination.pkl")
        with open(mse_by_combination_path, 'wb') as f:
            pickle.dump(mse_by_combination, f)
        print(f"Saved MSE by combination at {mse_by_combination_path}", flush=True)
    
    return mse_distribution, mse_by_combination

def plot_mse_distributions(mse_by_combination, ranks, lambdas, output_dir=None):
    """
    Plot the MSE distributions for each (rank, lambda) combination using KDE plots with variance annotations.
    
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
            
            # Plot density estimation
            sns.kdeplot(mse_values, fill=True, alpha=0.5, label=f"Rank: {rank}, Lambda: {lambda_}")
            
            # Calculate and display variance
            variance = np.var(mse_values)
            plt.title(f"Rank: {rank}, Lambda: {lambda_}\nVariance: {variance:.4f}")
            plt.xlabel("MSE")
            plt.ylabel("Density")
            plt.grid(True)
            plt.legend()
    
    plt.tight_layout()
    
    # Save or show the plot
    # make the output_dir if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, "mse_distributions_kde.png")
        plt.savefig(plot_path)
        print(f"Saved MSE distribution KDE plot at {plot_path}")
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
            print(f"Rank: {rank}, Lambda: {lambda_}, Mean MSE: {mean_mse:.4f}, Variance: {variance_mse:.4f}, Score: {score:.4f}", flush = True)
    
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
    print(f"Saved best hyperparameters at {best_hyperparams_path}", flush = True)

    return best_rank_lambda, best_score

def plot_mse_distributions_per_model(mse_distributions, models, output_dir):
    """
    Plot and save the MSE distributions for each model using boxplots.

    Args:
        mse_distributions (dict): Dictionary containing MSE distributions for each model, rank, and lambda.
        models (list): List of model names.
        output_dir (str): Directory to save the plots.
    """
    os.makedirs(output_dir, exist_ok=True)

    for model in models:
        plt.figure(figsize=(8, 6))  # Adjust the figure size as needed

        # Aggregate all MSE values for the model across ranks and lambdas
        mse_values = []
        for rank in mse_distributions[model]:
            for lambda_ in mse_distributions[model][rank]:
                mse_values.extend(mse_distributions[model][rank][lambda_])

        # Plot a single boxplot for the aggregated MSE values
        plt.boxplot(mse_values, patch_artist=True)
        plt.xlabel('Model')
        plt.ylabel('MSE')
        plt.title(f'MSE Distribution for Model: {model}')
        plt.grid(True)

        # Save the plot
        plot_path = os.path.join(output_dir, f"mse_distribution_{model}.png")
        plt.savefig(plot_path)
        plt.close()  # Close the plot to free memory
        print(f"Saved MSE distribution plot for model {model} at {plot_path}", flush = True)
    return None


def final_cross_validation(data, best_rank, best_lambda, use_ols_only=False, output_dir=None, normalise=False, center=True, option=1):
    """
    Perform a final round of cross-validation using the best rank and lambda.
    
    Args:
        data (dict): Preprocessed data without NaNs.
        best_rank (int): The best rank value.
        best_lambda (float): The best lambda value.
        use_ols_only (bool): Whether to use only OLS (no dimensionality reduction).
        output_dir (str): Directory to save the output files. If None, no files are saved.
        
    Returns:
        dict: Dictionary containing MSE losses for each test model.
    """
    models = list(data.keys())
    mse_losses = {model: [] for model in models}
    
    for test_model in tqdm(models, desc="Final Cross-Validation"):
        log_memory_usage(f"Before processing model {test_model}")
        # Split the data into training and testing sets
        train_models = [model for model in models if model != test_model]
        train_data = {model: data[model] for model in train_models}
        test_data = {test_model: data[test_model]}

        # Normalize the data
        normalized_train_data, normalized_test_data, _, testing_statistics = normalize_data(train_data, test_data, center=center, option=option)
        log_memory_usage(f"After normalization for model {test_model}")

        # Pool the training data
        X_train, Y_train = pool_data(normalized_train_data)
        log_memory_usage(f"After pooling data for model {test_model}")

        # Perform reduced rank regression
        B_rrr, _ = reduced_rank_regression(X_train, Y_train, rank=best_rank, lambda_=best_lambda, use_ols_only=use_ols_only)
        log_memory_usage(f"After regression for model {test_model}")

        # Evaluate on the test model
        test_runs = [run for run in normalized_test_data[test_model].keys() if run != 'forced_response']
        ground_truth = normalized_test_data[test_model]['forced_response']
        
        for run in test_runs:
            test_run_data = normalized_test_data[test_model][run]
            mse = calculate_mse(test_run_data, B_rrr, ground_truth, testing_statistics, test_model, normalise)
            mse_losses[test_model].append(mse)

        # Clear memory for the current test model
        del B_rrr, normalized_train_data, normalized_test_data, X_train, Y_train, train_data, test_data
        gc.collect()
        log_memory_usage(f"After processing model {test_model}")
    
    gc.collect()
    
    # Save the metrics as pickle files if output_dir is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save mse_losses
        method_name = "OLS_results" if use_ols_only else "RRR_results"
        mse_losses_path = os.path.join(output_dir, f"{method_name}_final_mse_losses.pkl")
        with open(mse_losses_path, 'wb') as f:
            pickle.dump(mse_losses, f)
        print(f"Saved final MSE losses at {mse_losses_path}", flush=True)
    
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
    print(f"Saved final MSE distribution plot at {plot_path}", flush = True)
    return None

def generate_and_save_animations(data, test_model, best_rank, best_lambda, nan_mask, num_runs=3, output_dir="output", color_limits=(-2, 2), on_cluster=False, use_ols_only=False, center=True, option=1, normalise=True):
    """
    Generate and save animations for a specified test model, including predictions, input data, and ground truth.
    
    Args:
        data (dict): Preprocessed data containing models and their data.
        test_model (str): The name of the model to use for testing.
        best_rank (int): The best rank value for reduced rank regression.
        best_lambda (float): The best lambda value for reduced rank regression.
        nan_mask (np.ndarray): Mask for NaN values in the data.
        num_runs (int): Number of test runs to animate.
        output_dir (str): Directory to save the animations.
        color_limits (tuple): Color limits for the animations.
    """

    # Split the data into training and testing sets
    train_models = [model for model in data.keys() if model != test_model]
    train_data = {model: data[model] for model in train_models}
    test_data = {test_model: data[test_model]}

    # Normalize the data
    normalized_train_data, normalized_test_data, _, testing_statistics = normalize_data(train_data, test_data, center=center, option=option)

    # Pool the training data
    X_train, Y_train = pool_data(normalized_train_data)

    # Perform reduced rank regression
    B_rrr, _ = reduced_rank_regression(X_train, Y_train, rank=best_rank, lambda_=best_lambda, use_ols_only=use_ols_only)

    # Create the output directory for animations
    animation_output_dir = os.path.join(output_dir, "animations")
    os.makedirs(animation_output_dir, exist_ok=True)

    # Generate and save animations
    plot_animations(
        test_model=test_model,
        normalized_test_data=normalized_test_data,
        Brr=B_rrr,
        nan_mask=nan_mask,
        num_runs=num_runs,
        color_limits=color_limits,
        save_path=animation_output_dir,
        on_cluster=on_cluster,
        normalise=normalise,
        testing_statistics=testing_statistics
    )

    print(f"Animations saved in {animation_output_dir}", flush = True)
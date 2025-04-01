import os
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from data_processing import preprocess_data, normalize_data, pool_data, readd_nans_to_grid
from regression import reduced_rank_regression
from metrics import calculate_mse, sanity_check
from pipeline import loo_cross_validation

def perform_cross_validation(data, lambdas, ranks):
    """
    Perform leave-one-out cross-validation for the given data, lambdas, and ranks.
    """
    print("Performing leave-one-out cross-validation...")
    mse_distributions = loo_cross_validation(data, lambdas, ranks)
    print("Cross-validation completed.")
    return mse_distributions

def find_best_hyperparameters(models, ranks, lambdas, mse_distributions):
    """
    Find the best combination of rank and lambda based on the lowest mean MSE.
    """
    best_rank = None
    best_lambda = None
    best_mse = float('inf')

    # Collect all MSEs for each rank and lambda across all models
    rank_lambda_mses = {rank: {lambda_: [] for lambda_ in lambdas} for rank in ranks}
    for model in models:
        for rank in ranks:
            for lambda_ in lambdas:
                rank_lambda_mses[rank][lambda_].extend(mse_distributions[model][rank][lambda_])

    # Find the rank and lambda with the lowest mean MSE
    for rank, lambda_mses in rank_lambda_mses.items():
        for lambda_, mse_values in lambda_mses.items():
            mean_mse = np.mean(mse_values)
            if mean_mse < best_mse:
                best_mse = mean_mse
                best_rank = rank
                best_lambda = lambda_

    print(f"Best rank: {best_rank}, Best lambda: {best_lambda}, MSE: {best_mse:.4f}")
    return best_rank, best_lambda

def visualize_predictions(test_model, normalized_test_data, B_rrr, nan_mask):
    """
    Visualize predictions for a random test run and plot the time series.
    """
    # Add the NaNs back to the grid for the predictions
    run_list = [run for run in normalized_test_data[test_model].keys() if run != 'forced_response']
    random_test_run = random.choice(run_list)
    random_test_run_data = normalized_test_data[test_model][random_test_run]
    ground_truth = normalized_test_data[test_model]['forced_response']
    print(f"Run chosen for visualization: {random_test_run}, for the testing model: {test_model}.")
    predictions = random_test_run_data @ B_rrr
    predictions_with_nans = readd_nans_to_grid(predictions, nan_mask, predictions=True)
    test_run_with_nans = readd_nans_to_grid(random_test_run_data, nan_mask, predictions=True)
    ground_truth_with_nans = readd_nans_to_grid(ground_truth, nan_mask, predictions=True)

    # Plot a random timestamp for the target and prediction
    random_timestamp = random.randint(0, predictions_with_nans.shape[0] - 1)
    print(f"Random timestamp: {random_timestamp}")

    # Data for the target
    test_run_grid = test_run_with_nans[random_timestamp, :].reshape(72, 144)
    ground_truth_grid = ground_truth_with_nans[random_timestamp, :].reshape(72, 144)
    prediction_grid = predictions_with_nans[random_timestamp, :].reshape(72, 144)

    # Plot the data using plt.imshow
    plt.figure(figsize=(18, 6))

    # Plot the test run
    plt.subplot(1, 3, 1)
    plt.imshow(test_run_grid, cmap='viridis', vmin=-3, vmax=3)
    plt.colorbar(label='Value')
    plt.title(f'Test Run input at Timestamp {random_timestamp}')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    # Plot the ground truth (forced response)
    plt.subplot(1, 3, 2)
    plt.imshow(ground_truth_grid, cmap='viridis', vmin=-3, vmax=3)
    plt.colorbar(label='Value')
    plt.title(f'Ground Truth at Timestamp {random_timestamp}')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    # Plot the prediction
    plt.subplot(1, 3, 3)
    plt.imshow(prediction_grid, cmap='viridis', vmin=-3, vmax=3)
    plt.colorbar(label='Value')
    plt.title(f'Prediction at Timestamp {random_timestamp}')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    plt.show()
    return None
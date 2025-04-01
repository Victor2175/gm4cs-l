import os, sys
import random
import warnings
import argparse

sys.path.append(os.path.join(os.getcwd(), 'utils'))

from utils.data_loading import *
from utils.data_processing import *
from utils.regression import *
from utils.animation import *
from utils.metrics import *
from utils.pipeline import *

# ignore warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
# Remove deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def main(args):
    # Define paths and parameters
    data_path = args.data_path
    filename = args.filename
    output_dir = args.output_dir
    mean_weight = args.mean_weight
    variance_weight = args.variance_weight
    lambdas = [1, 10, 50, 100, 200]  # Example lambda values
    ranks = [2, 5, 10, 50, 100]  # Example rank values
    num_runs = args.num_runs
    color_limits = (-2, 2)

    # Preprocess the data
    data, nan_mask = preprocess_data(data_path, filename)

    # Select a subset of models for testing
    random.seed(42)
    models = list(data.keys()) # List of all models
    # Select models for testing
    if args.all_models:
        print("Using all models for testing.") # Don't change models
    else:
        models = random.sample(models, min(args.num_models, len(models)))
        print(f"Randomly selected {len(models)} models for testing: {models}")
    subset_data = {model: data[model] for model in models}

    # Perform leave-one-out cross-validation
    print("Performing leave-one-out cross-validation...")
    mse_distributions, mse_by_combination = loo_cross_validation(subset_data, lambdas, ranks)

    # Plot the mse distributions for each combination of lambda and rank
    plot_mse_distributions(mse_by_combination, ranks, lambdas, output_dir='output')
    
    # Plot and save the MSE distributions for each model
    plot_mse_distributions_per_model(mse_distributions, models, ranks, lambdas, output_dir='output')

    # Select the most robust hyperparameters
    print("Selecting the most robust hyperparameters...")
    best_rank_lambda, _ = select_robust_hyperparameters(
        mse_by_combination, 
        mean_weight=mean_weight, 
        variance_weight=variance_weight, 
        output_dir=output_dir
    )
    best_rank, best_lambda = best_rank_lambda
    print(f"Best Rank: {best_rank}, Best Lambda: {best_lambda}")
    
    # Perform final cross-validation using the best rank and lambda
    final_mse_losses = final_cross_validation(subset_data, best_rank, best_lambda)
    plot_final_mse_distribution(final_mse_losses, output_dir='output')
    
    # Choose a random model to test on
    test_model = random.choice(models)
    print(f"Randomly selected test model: {test_model}")

    # Generate and save animations for the test model
    print("Generating and saving animations...")
    generate_and_save_animations(
        data=subset_data,
        test_model=test_model,
        best_rank=best_rank,
        best_lambda=best_lambda,
        nan_mask=nan_mask,
        num_runs=num_runs,
        output_dir=output_dir,
        color_limits=color_limits
    )
    print("Training and animation generation complete.")

if __name__ == "__main__":
    # Define argument parser
    parser = argparse.ArgumentParser(description="Train on M-1 models and generate animations for a test model.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the data directory.")
    parser.add_argument("--filename", type=str, required=True, help="Name of the data file.")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save outputs.")
    parser.add_argument("--num_runs", type=int, default=3, help="Number of test runs to animate.")
    parser.add_argument("--mean_weight", type=float, default=0.7, help="Weight for the mean MSE in hyperparameter selection.")
    parser.add_argument("--variance_weight", type=float, default=0.3, help="Weight for the variance in hyperparameter selection.")
    parser.add_argument("--num_models", type=int, default=2, help="Number of models to randomly select for testing.")
    parser.add_argument("--all_models", action="store_true", help="Use all models for testing. Overrides --num_models.")
    args = parser.parse_args()

    # Run the main function
    main(args)
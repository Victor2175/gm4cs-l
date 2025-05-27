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
    on_cluser = args.oncluster
    center = args.center
    use_ols_only = args.use_ols_only
    normalise = args.normalise
    option = args.option
    lambdas = [1, 10, 50, 100, 200, 500, 1000]  # Example lambda values
    ranks = [2, 5, 10]  # Example rank values
    # lambdas = [100]
    # ranks = [10]
    num_runs = args.num_runs
    color_limits = (-1.5, 1.5)

    # Preprocess the data
    data, nan_mask = preprocess_data(data_path, filename)

    # Select a subset of models for testing
    random.seed(1)
    models = list(data.keys()) # List of all models
    # Select models for testing
    if args.all_models:
        print("Using all models for testing.", flush = True) # Don't change models
    else:
        models = random.sample(models, min(args.num_models, len(models)))
        print(f"Randomly selected {len(models)} models for testing: {models}", flush = True)
    subset_data = {model: data[model] for model in models}

    # Check if best_rank and best_lambda are provided
    if args.best_rank is not None and args.best_lambda is not None:
        best_rank = args.best_rank
        best_lambda = args.best_lambda
        print(f"Using manually specified parameters: Best Rank = {best_rank}, Best Lambda = {best_lambda}", flush=True)
    else:
        # Perform leave-one-out cross-validation
        print("Performing leave-one-out cross-validation...", flush=True)
        mse_distributions, mse_by_combination = loo_cross_validation(subset_data, lambdas, ranks, center=center, use_ols_only=use_ols_only, output_dir=output_dir, normalise=normalise, option=option)

        # Plot the mse distributions for each combination of lambda and rank
        plot_mse_distributions(mse_by_combination, ranks, lambdas, output_dir=output_dir)
        
        # Plot and save the MSE distributions for each model
        plot_mse_distributions_per_model(mse_distributions, models, output_dir=output_dir)

        # Select the most robust hyperparameters
        print("Selecting the most robust hyperparameters...", flush=True)
        best_rank_lambda, _ = select_robust_hyperparameters(
            mse_by_combination,
            mean_weight=mean_weight,
            variance_weight=variance_weight,
            output_dir=output_dir
        )
        best_rank, best_lambda = best_rank_lambda
        print(f"Best Rank: {best_rank}, Best Lambda: {best_lambda}", flush = True)
    
    # Perform final cross-validation using the best rank and lambda
    final_mse_losses = final_cross_validation(subset_data, best_rank, best_lambda, use_ols_only=use_ols_only, output_dir=output_dir, normalise=normalise, center=center, option=option)
    plot_final_mse_distribution(final_mse_losses, output_dir=output_dir)
    
    # Find best and worst performing models
    best_model = min(final_mse_losses, key=lambda model: np.mean(final_mse_losses[model]))
    worst_model = max(final_mse_losses, key=lambda model: np.mean(final_mse_losses[model]))
    
    # Choose a random model to test on (different from best and worst if possible)
    available_models = [model for model in models if model != best_model and model != worst_model]
    if available_models:
        random_model = random.choice(available_models)
    else:
        random_model = random.choice(models)  # Fallback if only 1 or 2 models available
    
    print(f"Best performing model: {best_model}", flush=True)
    print(f"Worst performing model: {worst_model}", flush=True)
    print(f"Randomly selected test model: {random_model}", flush=True)

    # Generate and save animations for the models
    print("Generating and saving animations for random model...", flush=True)
    generate_and_save_animations(
        data=subset_data,
        test_model=random_model,
        best_rank=best_rank,
        best_lambda=best_lambda,
        nan_mask=nan_mask,
        num_runs=num_runs,
        output_dir=os.path.join(output_dir, "random_model"),
        color_limits=color_limits,
        on_cluster=on_cluser,
        use_ols_only=use_ols_only,
        center=center,
        option=option,
        normalise=normalise
    )
    
    print("Generating and saving animations for best performing model...", flush=True)
    generate_and_save_animations(
        data=subset_data,
        test_model=best_model,
        best_rank=best_rank,
        best_lambda=best_lambda,
        nan_mask=nan_mask,
        num_runs=num_runs,
        output_dir=os.path.join(output_dir, "best_model"),
        color_limits=color_limits,
        on_cluster=on_cluser,
        use_ols_only=use_ols_only,
        center=center,
        option=option,
        normalise=normalise
    )
    
    print("Generating and saving animations for worst performing model...", flush=True)
    generate_and_save_animations(
        data=subset_data,
        test_model=worst_model,
        best_rank=best_rank,
        best_lambda=best_lambda,
        nan_mask=nan_mask,
        num_runs=num_runs,
        output_dir=os.path.join(output_dir, "worst_model"),
        color_limits=color_limits,
        on_cluster=on_cluser,
        use_ols_only=use_ols_only,
        center=center,
        option=option,
        normalise=normalise
    )
    
    print("Training and animation generation complete.", flush=True)

if __name__ == "__main__":
    # Define argument parser
    parser = argparse.ArgumentParser(description="Train on M-1 models and generate animations for a test model.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the data directory.")
    parser.add_argument("--filename", type=str, required=True, help="Name of the data file.")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save outputs.")
    parser.add_argument("--num_runs", type=int, default=3, help="Number of test runs to animate.")
    parser.add_argument("--mean_weight", type=float, default=0.7, help="Weight for the mean MSE in hyperparameter selection.")
    parser.add_argument("--variance_weight", type=float, default=0.3, help="Weight for the variance in hyperparameter selection.")
    parser.add_argument("--num_models", type=int, default=2, help="Number of models to randomly select for testing.")
    parser.add_argument("--all_models", action="store_true", help="Use all models for testing. Overrides --num_models.")
    parser.add_argument("--oncluster", action="store_true", help="Use alternative video encoding for cluster environments.")
    parser.add_argument("--center", action="store_true", help="Center the data before training.")
    parser.add_argument("--best_rank", type=int, help="Manually specify the best rank. Overrides LOO cross-validation.")
    parser.add_argument("--best_lambda", type=float, help="Manually specify the best lambda. Overrides LOO cross-validation.")
    parser.add_argument("--use_ols_only", action="store_true", help="Use only OLS regression without SVD.")
    parser.add_argument("--normalise", action="store_true", help="Use to calculate the Normalised MSE")
    parser.add_argument("--option", type=int, choices=[1, 2], required=True, help="Choose an option: 1 for Normalized MSE, 2 for another calculation")
    args = parser.parse_args()

    # Run the main function
    main(args)
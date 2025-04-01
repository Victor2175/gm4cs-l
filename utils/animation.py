import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML, display
from metrics import *
from data_processing import *

def init_animation(im, data):
    """
    Initialization function for the animation.
    Args:
        im (matplotlib.image.AxesImage): Image to initialize.
        data (np.array): Data to display.
    Returns:
        tuple: Tuple containing the initialized image.
    """
    im.set_data(data[0])
    return [im]

def update_animation(i, im, data):
    """
    Function to update the figure in the animation.
    Args:
        i (int): Frame number.
        im (matplotlib.image.AxesImage): Image to update.
        data (np.array): Data to display.
    Returns:
        tuple: Tuple containing the updated image.
    """
    im.set_data(data[i])
    return [im]

def animate_data(data, title, interval=200, cmap='viridis', color_limits=None):
    """
    Animate the data to visualize the evolution of the response over time.
    
    Args:
        data (np.array): Array of shape (T, latitude, longitude).
        interval (int): Interval between frames in milliseconds.
        cmap (str): Name of the colormap to use.
    Returns:
        matplotlib.animation.FuncAnimation: Animation object.
    """
    fig, ax = plt.subplots()
    vmin, vmax = (color_limits if color_limits else (None, None))
    im = ax.imshow(data[0], cmap=cmap, animated = True, vmin = vmin, vmax = vmax)
    plt.colorbar(im, ax=ax, label = 'Temperature (°C)')
    ax.set_title(title)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    
    
    # blit = True to only update the parts that have changed
    ani = animation.FuncAnimation(fig, update_animation, init_func=lambda: init_animation(im, data),
                                  fargs=(im, data), frames=len(data), interval=interval, blit=True)
    plt.close(fig)
    return ani

def plot_animations(test_model, normalized_test_data, Brr, nan_mask, num_runs, color_limits=None):
    """
    Plot the animations for all test runs and the ground truth.
    
    Args:
        test_model (str): The test model name.
        normalized_test_data (dict): Dictionary containing the normalized test data.
        Brr (np.array): Reduced-rank weight matrix.
        nan_mask (np.array): Boolean mask indicating NaN positions.
    """
    predictions = True
    test_runs = [run for run in normalized_test_data[test_model].keys() if run != 'forced_response']
    ground_truth = normalized_test_data[test_model]['forced_response']

    for counter, run in enumerate(test_runs):
        if counter == num_runs:
            break # Stop after num_runs runs
        test_run = normalized_test_data[test_model][run]
        
        # Make the prediction
        prediction = test_run @ Brr
        
        # Calculate the MSE
        input_mse = calculate_mse(test_run, Brr, ground_truth)
        prediction_mse = calculate_mse(prediction, Brr, ground_truth)
        
        # Re-add NaN values to the data matrices
        prediction = readd_nans_to_grid(prediction, nan_mask, predictions)
        test_run = readd_nans_to_grid(test_run, nan_mask, predictions)
        ground_truth_with_nans = readd_nans_to_grid(ground_truth, nan_mask, predictions)
        
        # Reshape the data to match the expected dimensions for imshow
        prediction = prediction.reshape(-1, nan_mask.shape[0], nan_mask.shape[1])
        test_run = test_run.reshape(-1, nan_mask.shape[0], nan_mask.shape[1])
        ground_truth_with_nans = ground_truth_with_nans.reshape(-1, nan_mask.shape[0], nan_mask.shape[1])
        
        # Create animations with titles
        pred_animation = animate_data(prediction, interval=200, cmap='viridis', title=f'Prediction: {test_model} - {run} (MSE: {prediction_mse:.2f})', color_limits=color_limits)
        test_run_animation = animate_data(test_run, interval=200, cmap='viridis', title=f'Input: {test_model} - {run} (MSE: {input_mse:.2f})', color_limits=color_limits)
        ground_truth_animation = animate_data(ground_truth_with_nans, interval=200, cmap='viridis', title=f'Ground Truth: {test_model}', color_limits=color_limits)
        
        # Display animations
        display(HTML(ground_truth_animation.to_html5_video()))
        display(HTML(pred_animation.to_html5_video()))
        display(HTML(test_run_animation.to_html5_video()))
    return None
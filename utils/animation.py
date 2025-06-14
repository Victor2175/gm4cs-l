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

def animate_data(data, title, interval=200, cmap='viridis', color_limits=None, extent=None):
    """
    Animate the data to visualize the evolution of the response over time.
    
    Args:
        data (np.array): Array of shape (T, latitude, longitude).
        title (str): Title of the animation.
        interval (int): Interval between frames in milliseconds.
        cmap (str): Name of the colormap to use.
        color_limits (tuple): Tuple specifying (vmin, vmax) for color scaling.
        extent (tuple): Tuple specifying (xmin, xmax, ymin, ymax) for geospatial plotting.
    Returns:
        matplotlib.animation.FuncAnimation: Animation object.
    """
    fig, ax = plt.subplots()
    vmin, vmax = (color_limits if color_limits else (None, None))
    im = ax.imshow(data[0], cmap=cmap, animated=True, vmin=vmin, vmax=vmax, extent=extent, origin='upper')
    plt.colorbar(im, ax=ax, label='Temperature (°C)')
    ax.set_title(title)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    
    
    # blit = True to only update the parts that have changed
    ani = animation.FuncAnimation(fig, update_animation, init_func=lambda: init_animation(im, data),
                                  fargs=(im, data), frames=len(data), interval=interval, blit=True)
    plt.close(fig)
    return ani

def plot_animations(test_model, normalized_test_data, Brr, nan_mask, num_runs, color_limits=None, save_path=None, on_cluster=False, normalise=True, testing_statistics=None, extent=None):
    """
    Plot the animations for all test runs and the ground truth.
    
    Args:
        test_model (str): The test model name.
        normalized_test_data (dict): Dictionary containing the normalized test data.
        Brr (np.array): Reduced-rank weight matrix.
        nan_mask (np.array): Boolean mask indicating NaN positions.
        num_runs (int): Number of test runs to animate.
        color_limits (tuple): Tuple specifying (vmin, vmax) for color scaling.
        save_path (str): Path to save the animations.
        on_cluster (bool): Whether the code is running on a cluster.
        normalise (bool): Whether to normalize the data.
        testing_statistics (dict): Statistics for normalization.
        extent (tuple): Tuple specifying (xmin, xmax, ymin, ymax) for geospatial plotting.
    """
    predictions = True
    test_runs = [run for run in normalized_test_data[test_model].keys() if run != 'forced_response']
    ground_truth = normalized_test_data[test_model]['forced_response']

    for counter, run in enumerate(test_runs):
        if counter == num_runs:
            break
        test_run = normalized_test_data[test_model][run]
        prediction = test_run @ Brr
        input_mse = calculate_mse(test_run, Brr, ground_truth, testing_statistics, test_model, normalise)
        prediction_mse = calculate_mse(prediction, Brr, ground_truth, testing_statistics, test_model, normalise)
        prediction = readd_nans_to_grid(prediction, nan_mask, predictions)
        test_run = readd_nans_to_grid(test_run, nan_mask, predictions)
        ground_truth_with_nans = readd_nans_to_grid(ground_truth, nan_mask, predictions)
        prediction = prediction.reshape(-1, nan_mask.shape[0], nan_mask.shape[1])
        test_run = test_run.reshape(-1, nan_mask.shape[0], nan_mask.shape[1])
        ground_truth_with_nans = ground_truth_with_nans.reshape(-1, nan_mask.shape[0], nan_mask.shape[1])

        pred_animation = animate_data(
            prediction,
            title=f'Prediction: {test_model} - {run} (MSE: {prediction_mse:.2f})',
            interval=200,
            cmap='viridis',
            color_limits=color_limits,
            extent=extent
        )
        test_run_animation = animate_data(
            test_run,
            title=f'Input: {test_model} - {run} (MSE: {input_mse:.2f})',
            interval=200,
            cmap='viridis',
            color_limits=color_limits,
            extent=extent
        )
        ground_truth_animation = animate_data(
            ground_truth_with_nans,
            title=f'Ground Truth: {test_model}',
            interval=200,
            cmap='viridis',
            color_limits=color_limits,
            extent=extent
        )

        if save_path is None:
            display(HTML(pred_animation.to_html5_video()))
            display(HTML(test_run_animation.to_html5_video()))
            display(HTML(ground_truth_animation.to_html5_video()))
        else:
            writer = 'pillow' if on_cluster else 'ffmpeg'
            pred_animation.save(f"{save_path}/prediction_{test_model}_{run}.mp4", writer=writer, fps=15)
            test_run_animation.save(f"{save_path}/input_{test_model}_{run}.mp4", writer=writer, fps=15)
            ground_truth_animation.save(f"{save_path}/ground_truth_{test_model}.mp4", writer=writer, fps=15)
            print(f"Saved animations using writer: {writer}")
    return None
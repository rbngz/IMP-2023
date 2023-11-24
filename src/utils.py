import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1 import make_axes_locatable


def get_dataset_stats(df, data_dir):
    # Placeholder to store all images from df
    img_list = np.empty((len(df), 200, 200, 12))

    # Iterate over entire dataframe
    for i, (_, data) in tqdm(enumerate(df.iterrows())):
        # Load satellite image
        img_path = data["img_path"]
        img_path = os.path.join(data_dir, img_path)
        img = np.load(img_path)

        # Store satellite image
        img_list[i] = img

    # Compute mean and std over all bands and images
    band_means = img_list.mean(axis=(0, 1, 2))
    band_stds = img_list.std(axis=(0, 1, 2))

    # Compute mean and std for no2 measurements
    no2_mean = df["no2"].mean()
    no2_std = df["no2"].std()

    # Store information in dictionary
    stats = {
        "band_means": band_means,
        "band_stds": band_stds,
        "no2_mean": no2_mean,
        "no2_std": no2_std,
    }
    return stats


def normalize_rgb_bands(rgb_bands):
    # Scale RGB bands to be within the range of [0, 1]
    rgb_min = np.percentile(rgb_bands, 2, axis=(0, 1))
    rgb_max = np.percentile(rgb_bands, 98, axis=(0, 1))

    rgb_bands_norm = (rgb_bands - rgb_min) / (rgb_max - rgb_min)

    rgb_bands_norm = np.clip(rgb_bands_norm, 0, 1)

    return rgb_bands_norm


def plot_predictions(data, no2_norm, coords, outputs, target_transform, n_samples=4):
    # If batch size is smaller than given n, only plot batch items
    n_samples = min(len(data), n_samples)

    # Create sublot
    fig, axes = plt.subplots(2, n_samples, figsize=(24, 10))

    # Revert normalized no2 ground truth values
    no2 = target_transform.revert(no2_norm)

    for i in range(n_samples):
        # Extract and normalize RGB bands
        rgb_bands = np.moveaxis(data[i][:3].numpy(), 0, 2)
        rgb_bands_norm = normalize_rgb_bands(rgb_bands)

        # Revert normalization of predicted NO2 values
        prediction_norm = outputs[i].detach()
        prediction = target_transform.revert(prediction_norm)[0].numpy()

        # Plot ground truth and prediction
        ax_truth = axes[0][i]
        ax_pred = axes[1][i]

        ax_truth.imshow(rgb_bands_norm)
        pos = ax_pred.imshow(prediction)

        coords_height = coords[0][i]
        coords_width = coords[1][i]

        # Mark coordinates
        circle_truth = Circle((coords_width, coords_height), radius=1, color="red")
        circle_pred = Circle((coords_width, coords_height), radius=1, color="red")
        ax_truth.add_patch(circle_truth)
        ax_pred.add_patch(circle_pred)

        # Add titles
        ax_truth.set_title(f"Ground truth: {no2[i]:.2f}")
        ax_pred.set_title(
            f"Prediction: {prediction[(coords_height, coords_width)]:.2f}"
        )

        # Create colorbar
        divider = make_axes_locatable(ax_pred)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(pos, cax=cax, orientation="vertical")

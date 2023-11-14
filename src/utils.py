import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch.utils.data import Subset


def get_dataset_stats(dataset):
    # Make sure this can be applied also to subsets
    if isinstance(dataset, Subset):
        dataset_root = dataset.dataset
    else:
        dataset_root = dataset

    # Create empty array to store band information
    patch_list = np.empty(
        (len(dataset), 12, dataset_root.patch_size, dataset_root.patch_size)
    )

    # Create empty array to store no2 data
    no2_list = np.empty(len(dataset))

    # Get every patch of dataset or subset
    for i in tqdm(range(len(dataset))):
        patch, no2, _ = dataset[i]

        # Store either the entire batch or only the band means
        patch_list[i] = patch
        no2_list[i] = no2

    # Compute mean and std over all images and bands
    band_means = patch_list.mean(axis=(0, 2, 3))
    band_stds = patch_list.std(axis=(0, 2, 3))

    # Compute mean and std over all measurements
    no2_mean = no2_list.mean()
    no2_std = no2_list.std()

    stats = {
        "band_means": band_means,
        "band_stds": band_stds,
        "no2_mean": no2_mean,
        "no2_std": no2_std,
    }
    return stats


class DatasetStatistics:
    def __init__(self, samples_file, data_dir):
        samples_df = pd.read_csv(samples_file, index_col="idx")

        # Remove NA measurements
        samples_df = samples_df[~samples_df["no2"].isna()]

        no2_measurements = samples_df["no2"]
        data_paths = samples_df["img_path"]

        band_means_sums = np.zeros((12,))
        band_stds_sums = np.zeros((12,))
        for idx in data_paths.index:
            data_path = os.path.join(data_dir, data_paths.loc[idx])
            data = np.load(data_path)
            band_means_sums += data.mean(axis=(0, 1))
            band_stds_sums += data.std(axis=(0, 1))

        self.band_means = band_means_sums / len(data_paths)
        self.band_std = band_stds_sums / len(data_paths)

        self.no2_mean = no2_measurements.mean()
        self.no2_std = no2_measurements.std()


def normalize_rgb_bands(rgb_bands):
    # Scale RGB bands to be within the range of [0, 1]
    rgb_min = rgb_bands.min(axis=(0, 1))
    rgb_max = rgb_bands.max(axis=(0, 1))
    rgb_bands_norm = (rgb_bands - rgb_min) / (rgb_max - rgb_min)

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

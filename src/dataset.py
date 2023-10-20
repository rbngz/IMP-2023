import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from torch.utils.data import Dataset

from utils import normalize_rgb_bands


class SentinelNO2Dataset(Dataset):
    def __init__(
        self,
        samples_file,
        data_dir,
        n_patches=4,
        patch_size=(100, 100),
        pre_load=False,
        transform=None,
        target_transform=None,
    ):
        """Dataset that contains n patches per satellite image"""

        # Read the samples file
        samples_df = pd.read_csv(samples_file, index_col="idx")

        # Remove NA measurements
        samples_df = samples_df[~samples_df["no2"].isna()]

        # Extract relevant information from file
        self.no2_measurements = samples_df["no2"]
        self.data_paths = samples_df["img_path"]

        # Set dataset attributes
        self.data_dir = data_dir
        self.n_patches = n_patches
        self.patch_size = patch_size
        self.pre_load = pre_load
        self.transform = transform
        self.target_transform = target_transform

        # Pre-load data if needed
        self.items = None
        if pre_load:
            self.items = self._pre_load_data()

    def _pre_load_data(self):
        """Pre-load and return the data as list of tuples"""
        items = []

        # Iterate over the entire dataset
        for idx in range(len(self)):
            item = self._load_item(idx)
            items.append(item)

        return items

    def __len__(self):
        """Return length considering the number of patches per file"""
        return len(self.data_paths) * self.n_patches

    def __getitem__(self, idx):
        """Get the data, NO2 measurement and coordinates by index"""

        if self.pre_load:
            # Retrieve from pre-loaded values
            return self.items[idx]

        # Retrieve from file system
        return self._load_item(idx)

    def _load_item(self, idx, transform=True):
        """Load data from the file system"""

        # Make sure offset is deterministic by index
        random.seed(idx)

        # Determine file index considering each file should be sampled n times
        file_idx = idx % len(self.data_paths)

        # Load data from path according to file index
        data_path = os.path.join(self.data_dir, self.data_paths.iloc[file_idx])
        data = np.load(data_path).astype(np.float32)

        # Retrieve NO2 measurement for this sample
        no2 = self.no2_measurements.iloc[file_idx]

        # Determine coordinates of measurement (midpoint)
        height, width, _ = data.shape
        mid_heigth, mid_width = height // 2, width // 2

        # Get dimensions of patch
        patch_height, patch_width = self.patch_size

        # Get deterministic cropping offset
        offset_height = random.randint(0, height - patch_height)
        offset_width = random.randint(0, width - patch_width)

        # Crop image according to offset and patch size
        patch = data[
            offset_height : patch_height + offset_height,
            offset_width : patch_width + offset_width,
            :,
        ]

        # Get new coordinates of measurement
        # Need to make sure that the coordinate is within bounds
        coord_height = np.clip(mid_heigth - offset_height, 0, patch_height - 1)
        coord_width = np.clip(mid_width - offset_width, 0, patch_width - 1)
        coords = (coord_height, coord_width)

        # Apply transformations
        if transform:
            if self.transform:
                patch = self.transform(patch)
            if self.target_transform:
                no2 = self.target_transform(no2)

        return patch, no2, coords

    def plot(self, idx, ax=None):
        """Display a sample at a given index"""
        data, no2, coords = self._load_item(idx, transform=False)

        # Extract and normalize RGB bands
        rgb_bands = data[:, :, 0:3]
        rgb_bands_norm = normalize_rgb_bands(rgb_bands)

        # Plot the RGB values
        if ax is None:
            _, ax = plt.subplots()
        ax.imshow(rgb_bands_norm)
        ax.add_patch(Circle((coords[1], coords[0]), radius=1, color="red"))

        # Indicate NO2 measurement in title
        ax.set_title(f"Measurement: {no2:.2f}")

    def plot_patches(self, idx):
        """Display a sample and its corresponding patches by a given index"""
        base_idx = idx % len(self.data_paths)

        # Load base image from path according to base index
        data_path = os.path.join(self.data_dir, self.data_paths.iloc[base_idx])
        base_img = np.load(data_path).astype(np.float32)
        base_img_rgb_bands = normalize_rgb_bands(base_img[:, :, 0:3])

        _, axes = plt.subplots(1, self.n_patches + 1, figsize=(25, 100))

        axes = axes.flatten()
        axes[0].imshow(base_img_rgb_bands)
        axes[0].set_title("Base Image")

        # Mark coordinates
        coords_height = base_img.shape[0] // 2
        coords_width = base_img.shape[1] // 2
        marker = Circle((coords_width, coords_height), radius=2, color="red")
        axes[0].add_patch(marker)

        # Determine dataset indices of the patches
        patch_indices = [
            x * len(self.data_paths) + base_idx for x in range(0, self.n_patches)
        ]

        # Plot each patch
        for i, patch_idx in enumerate(patch_indices):
            self.plot(patch_idx, axes[i + 1])

    def plot_coords_distribution(self):
        """Plot distribution of coordinates within the patches"""

        # Load all items into memory if needed
        if self.pre_load:
            items = self.items
        else:
            items = self._pre_load_data()

        # Count coordinate occurences
        sum_coords = np.zeros(self.patch_size)
        for _, _, coords in items:
            sum_coords[coords] += 1

        # Plot the distributions
        fig, ax = plt.subplots()
        pos = ax.imshow(sum_coords, cmap="Reds")
        fig.colorbar(pos, ax=ax)
        ax.set_title(f"Distribution of Coordinates of {len(self)} samples")

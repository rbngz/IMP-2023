import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset


class SentinelNO2Dataset(Dataset):
    def __init__(self, samples_file, data_dir, transform=None, target_transform=None):
        samples_df = pd.read_csv(samples_file, index_col="idx")

        self.no2_measurements = samples_df["no2"]
        self.data_dir = data_dir
        self.data_paths = samples_df["img_path"]

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.no2_measurements)

    def __getitem__(self, idx):
        # Load data from path according to index
        data_path = os.path.join(self.data_dir, self.data_paths.loc[idx])
        data = np.ﬁ(data_path).astype(np.float32)

        # Retrieve NO2 measurement for this sample
        no2 = self.no2_measurements.loc[idx]

        # Determine coordinates of measurement
        height, width, _ = data.shape
        coords = height // 2, width // 2

        # Apply transformations
        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            no2 = self.target_transform(no2)

        return data, no2, coords

    def plot(self, idx):
        """Display a sample at a given index"""
        data_path = os.path.join(self.data_dir, self.data_paths.loc[idx])
        data = np.load(data_path).astype(np.float32)
        no2 = self.no2_measurements.loc[idx]

        # Determine coordinates of measurement
        height, width, _ = data.shape
        coords = height // 2, width // 2

        rgb_bands = data[:, :, 0:3]

        # Scale RGB bands to be within the range of [0, 1]
        rgb_min = rgb_bands.min(axis=(0, 1))
        rgb_max = rgb_bands.max(axis=(0, 1))
        rgb_bands_norm = (rgb_bands - rgb_min) / (rgb_max - rgb_min)

        # Mark measurement location
        rgb_bands_norm[coords] = [1, 0, 0]

        # Plot the RGB values
        _, ax = plt.subplots()
        ax.imshow(rgb_bands_norm)

        # Indicate NO2 measurement in title
        ax.set_title(f"Measurement: {no2:.2f}")

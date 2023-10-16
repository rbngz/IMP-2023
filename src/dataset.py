import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset


class SentinelNO2Dataset(Dataset):
    def __init__(self, samples_file, data_dir, transform=None):
        samples_df = pd.read_csv(samples_file, index_col="idx")

        self.no2_measurements = samples_df["no2"]
        self.data_dir = data_dir
        self.data_paths = samples_df["img_path"]

        self.transform = transform

    def __len__(self):
        return len(self.no2_measurements)

    def __getitem__(self, idx):
        data_path = os.path.join(self.data_dir, self.data_paths.loc[idx])

        data = np.load(data_path).astype("int32")
        no2 = self.no2_measurements.loc[idx]

        return data, no2

    def plot(self, idx):
        """Display a sample at a given index"""
        data, no2 = self[idx]

        rgb_bands = data[:, :, 0:3]

        rgb_min = rgb_bands.min(axis=(0, 1))
        rgb_max = rgb_bands.max(axis=(0, 1))
        rgb_bands_norm = (rgb_bands - rgb_min) / (rgb_max - rgb_min)

        _, ax = plt.subplots()
        ax.imshow(rgb_bands_norm)
        ax.set_title(f"Measurement: {no2:.2f}")

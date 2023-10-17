import os
import numpy as np
import pandas as pd


class DatasetStatistics:
    def __init__(self, samples_file, data_dir):
        samples_df = pd.read_csv(samples_file, index_col="idx")

        no2_measurements = samples_df["no2"]
        data_paths = samples_df["img_path"]

        stacked_data = None
        for idx in range(len(data_paths)):
            data_path = os.path.join(data_dir, data_paths.loc[idx])
            data = np.load(data_path).astype(np.float32)
            if stacked_data is None:
                stacked_data = data
            else:
                stacked_data = np.vstack([stacked_data, data])

        self.band_means = stacked_data.mean(axis=(0, 1))
        self.band_std = stacked_data.std(axis=(0, 1))

        self.no2_mean = no2_measurements.mean()
        self.no2_std = no2_measurements.std()

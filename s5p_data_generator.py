import os
import pandas as pd
import numpy as np
import xarray as xr


DATA_DIR = "/netscratch2/rubengaviles/imp-2023/data"
SAMPLES_PATH = (
    "/netscratch2/rubengaviles/imp-2023/data/samples/samples_S2S5P_2018_2020_eea.csv"
)
# DATA_DIR = "data"
# SAMPLES_PATH = "data/samples/samples_S2S5P_2018_2020_epa.csv"
SAVE_DIR = "sentinel-5p-eea-numpy"


# Read the samples file
samples_df = pd.read_csv(SAMPLES_PATH, index_col="idx")

# Remove NA measurements
samples_df = samples_df[~samples_df["no2"].isna()]

for index, row in samples_df.iterrows():
    print(index)
    s5p_path = row["s5p_path"]
    station_name = row["AirQualityStation"]

    s5p_data = xr.open_dataset(
        os.path.join(DATA_DIR, "data/sentinel-5p-eea", s5p_path)
    ).rio.write_crs(4326)

    s5p_data_numpy = s5p_data.tropospheric_NO2_column_number_density.values.squeeze()
    print(index, s5p_data_numpy.shape)

    np.save(os.path.join(DATA_DIR, SAVE_DIR, station_name), s5p_data_numpy)

import os
import pandas as pd
import numpy as np
import xarray as xr
from tqdm import tqdm
from skimage.transform import resize


# Specify if EPA dataset or EEA
US = False

# Modify data directory if needed
DATA_DIR = "/netscratch2/rubengaviles/imp-2023/data"
# DATA_DIR = "data"

# Determine paths to read samples from and store data
SAMPLES_PATH = (
    "samples/samples_S2S5P_2018_2020_epa.csv"
    if US
    else "samples/samples_S2S5P_2018_2020_eea.csv"
)
SAVE_DIR = "sentinel-5p-epa-numpy-resized" if US else "sentinel-5p-eea-numpy-resized"

# Read the samples file
samples_df = pd.read_csv(os.path.join(DATA_DIR, SAMPLES_PATH), index_col="idx")

# Remove NA measurements
samples_df = samples_df[~samples_df["no2"].isna()]


for index, row in tqdm(samples_df.iterrows()):
    # Read attributes of row
    s5p_path = row["s5p_path"]
    station_name = row["AirQualityStation"]

    # Read Grid data for S5P sample
    s5p_data = xr.open_dataset(
        os.path.join(DATA_DIR, "data/sentinel-5p-eea", s5p_path)
    ).rio.write_crs(4326)

    # Convert to numpy
    s5p_data_numpy = s5p_data.tropospheric_NO2_column_number_density.values.squeeze()

    # Resize to match dimensions of S2 data
    s5p_resized = resize(s5p_data_numpy, (200, 200))

    # Save file with station name as file name
    np.save(os.path.join(DATA_DIR, SAVE_DIR, station_name), s5p_data_numpy)

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset


class SentinelDataset(Dataset):
    IMG_SIZE = 200
    MEASUREMENT_LOC_XY = 99

    def __init__(
        self,
        df_samples,
        data_dir,
        n_patches=4,
        patch_size=128,
        pre_load=False,
        s2_transform=None,
        no2_transform=None,
    ) -> None:
        """Dataset that contains n patches per satellite image"""
        super().__init__()

        # Store samples information
        self.df_samples = df_samples

        # Extract relevant information from file
        self.measurements = df_samples["no2"].astype(np.float32)
        self.img_paths = df_samples["img_path"]
        self.stations = df_samples["AirQualityStation"]

        # Determine minimum and maximum possible offsets
        self.offset_min = max(0, self.MEASUREMENT_LOC_XY - patch_size + 1)
        self.offset_max = min(self.MEASUREMENT_LOC_XY, self.IMG_SIZE - patch_size)

        # Set dataset attributes
        self.data_dir = data_dir
        self.n_patches = n_patches
        self.patch_size = patch_size
        self.pre_load = pre_load
        self.s2_transform = s2_transform
        self.no2_transform = no2_transform

        # Pre-load images
        images = []
        if self.pre_load:
            for i in range(len(self.img_paths)):
                img = self.get_full_image(i)
                images.append(img)
        self.images = images

    def __getitem__(self, index):
        """Get the data, NO2 measurement and coordinates by index"""

        # Determine image index considering each file should be sampled n times
        data_idx = index % len(self.img_paths)

        # Load image from memory or file system
        if self.pre_load:
            img = self.images[data_idx]
        else:
            img = self.get_full_image(data_idx)

        # Load land cover ground truth
        station = self.stations.iloc[data_idx]
        land_cover_path = os.path.join(self.data_dir, "worldcover", station + ".npy")
        land_cover = np.load(land_cover_path).astype(np.int64)
        land_cover = land_cover // 10

        # Retrieve NO2 measurement for this sample
        no2 = self.measurements.iloc[data_idx]

        # Get offsets
        offset_height, offset_width = self._get_offsets()

        # Crop image according to offset and patch size
        patch = img[
            offset_height : offset_height + self.patch_size,
            offset_width : offset_width + self.patch_size,
        ]

        # Crop land cover ground truth
        land_cover = land_cover[
            offset_height : offset_height + self.patch_size,
            offset_width : offset_width + self.patch_size,
        ]

        # Get new coordinates of measurement
        coord_height = self.MEASUREMENT_LOC_XY - offset_height
        coord_width = self.MEASUREMENT_LOC_XY - offset_width
        coords = (coord_height, coord_width)

        # Apply transformation
        if self.s2_transform:
            patch = self.s2_transform(patch)
        if self.no2_transform:
            no2 = self.no2_transform(no2)

        land_cover = torch.from_numpy(land_cover)

        return patch, land_cover, no2, coords

    def _get_offsets(self):
        # Get random cropping offset
        offset_height = random.randint(self.offset_min, self.offset_max)
        offset_width = random.randint(self.offset_min, self.offset_max)

        return offset_height, offset_width

    def get_full_image(self, index):
        # Load data from path according to image index
        img_path = os.path.join(
            self.data_dir, "sentinel-2-eea", self.img_paths.iloc[index]
        )
        img = np.load(img_path).astype(np.float32)
        return img

    def get_coords_distribution(self):
        # Count coordinate occurences
        sum_coords = np.zeros((self.patch_size, self.patch_size)).astype(int)
        for _ in range(len(self)):
            offset_height, offset_width = self._get_offsets()
            coord_height = self.MEASUREMENT_LOC_XY - offset_height
            coord_width = self.MEASUREMENT_LOC_XY - offset_width
            coords = (coord_height, coord_width)
            sum_coords[coords] += 1

        return sum_coords

    def __len__(self):
        """Return length considering the number of patches per sample"""
        return len(self.df_samples) * self.n_patches

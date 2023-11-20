import os
import random
import numpy as np
from torch.utils.data import Dataset
from skimage.transform import resize
from tqdm import tqdm


class SentinelDataset(Dataset):
    def __init__(
        self,
        df_samples,
        data_dir,
        n_patches=4,
        patch_size=256,
        pre_load=False,
        transform=None,
    ) -> None:
        super().__init__()
        """Dataset that contains n patches per satellite image"""
        # Store samples information
        self.df_samples = df_samples

        # Extract relevant information from file
        self.measurements = df_samples["no2"].astype(np.float32)
        self.img_paths = df_samples["img_path"]

        # Set dataset attributes
        self.data_dir = data_dir
        self.n_patches = n_patches
        self.patch_size = patch_size
        self.pre_load = pre_load
        self.transform = transform

        # Pre-load data if needed
        self.items = None
        if pre_load:
            self.items = self._pre_load_data()

    def _pre_load_data(self):
        """Pre-load and return the data as list of tuples"""
        items = []

        # Iterate over the entire dataset
        for index in tqdm(range(len(self))):
            item = self._load_item(index)
            items.append(item)

        return items

    def __getitem__(self, index):
        """Get the data, NO2 measurement and coordinates by index"""
        if self.pre_load:
            # Retrieve from pre-loaded values
            return self.items[index]

        # Retrieve from file system
        return self._load_item(index)

    def _load_item(self, index):
        """Load data from the file system"""

        # Determine image index considering each file should be sampled n times
        data_idx = index % len(self.img_paths)

        # Load data from path according to image index
        img = self.get_full_image(data_idx)

        # Retrieve NO2 measurement for this sample
        no2 = self.measurements.iloc[data_idx]

        # Rescale image to allow for patch size
        img_size_new = self.patch_size * 2 - 1
        img = resize(img, (img_size_new, img_size_new))

        # Get offsets
        offset_height, offset_width = self._get_offset(index)

        # Crop image according to offset and patch size
        patch = img[
            offset_height : offset_height + self.patch_size,
            offset_width : offset_width + self.patch_size,
            :,
        ]

        # Apply transformation
        if self.transform:
            patch = self.transform(patch)

        # Get new coordinates of measurement
        coord_height = self.patch_size - offset_height - 1
        coord_width = self.patch_size - offset_width - 1
        coords = (coord_height, coord_width)

        return patch, no2, coords

    def _get_offset(self, index):
        # Make sure offset is deterministic by index
        random.seed(index)

        # Get deterministic cropping offset
        offset_height = random.randint(0, self.patch_size - 1)
        offset_width = random.randint(0, self.patch_size - 1)

        return offset_height, offset_width

    def get_full_image(self, index):
        # Load data from path according to image index
        img_path = os.path.join(self.data_dir, self.img_paths.iloc[index])
        img = np.load(img_path).astype(np.float32)
        return img

    def get_coords_distribution(self):
        # Count coordinate occurences
        sum_coords = np.zeros((self.patch_size, self.patch_size)).astype(int)
        for index in range(len(self)):
            offset_height, offset_width = self._get_offset(index)
            coord_height = self.patch_size - offset_height - 1
            coord_width = self.patch_size - offset_width - 1
            coords = (coord_height, coord_width)
            sum_coords[coords] += 1

        return sum_coords

    def __len__(self):
        """Return length considering the number of patches per sample"""
        return len(self.df_samples) * self.n_patches

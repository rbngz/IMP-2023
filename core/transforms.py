import torch
import numpy as np
from torchvision.transforms.v2 import Normalize


class TargetNormalize(object):
    def __init__(self, mean, std) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, value):
        value = (value - self.mean) / self.std
        return value.astype(np.float32)

    def revert(self, value):
        value = value * self.std + self.mean
        return value


class BandNormalize(Normalize):
    def __init__(self, mean, std):
        super().__init__(mean, std)

    def revert(self, img):
        # Convert to tensor and add two axes
        means = torch.tensor(self.mean)[:, None, None]
        stds = torch.tensor(self.std)[:, None, None]

        # Revert normalization
        img = img * stds + means

        return img

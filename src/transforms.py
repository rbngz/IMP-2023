from typing import Any


class TargetNormalize(object):
    def __init__(self, mean, std) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, value):
        value = (value - self.mean) / self.std
        return value

    def revert(self, value):
        value = value * self.std + self.mean
        return value

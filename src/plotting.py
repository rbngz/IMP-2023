import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from utils import normalize_rgb_bands


def plot_patch(patch, no2=None, coords=None):
    rgb_bands = patch[:, :, 0:3]
    rgb_bands_norm = normalize_rgb_bands(rgb_bands)

    _, ax = plt.subplots()
    ax.imshow(rgb_bands_norm)

    if no2:
        # Indicate NO2 measurement in title
        ax.set_title(f"Measurement: {no2:.2f}")

    if coords:
        ax.add_patch(Circle((coords[1], coords[0]), radius=1, color="red"))


def plot_coords_distribution(coords_distribution):
    # Plot the distributions
    fig, ax = plt.subplots()
    pos = ax.imshow(coords_distribution, cmap="Reds")
    fig.colorbar(pos, ax=ax)
    ax.set_title(f"Distribution of Coordinates of {coords_distribution.sum()} samples")

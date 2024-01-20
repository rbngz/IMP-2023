# Ground-Level NO₂ Concentration Estimation using Deep Learning

This repository contains the source code for the paper "Patch-Wise Ground-Level NO₂ Pollution Estimation". It includes the implementation of deep learning models for estimating ground-level Nitrogen Dioxide (NO₂) concentrations from satellite imagery.

![Overview](/images/overview.png)
*Figure 1: High-level overview showcasing the process of estimating NO2 concentrations using a deep learning model. The model inputs are randomly sampled windows consisting of satellite data from Sentinel-2 and Sentinel-5P, which then pass through a convolutional neural network to extract relevant features. The NO2 Regression Head produces a concentration estimate, which is subsequently aligned with ground truth data for validation. Concurrently, the Land Cover Classification Head processes the same features to classify land cover, aiding in the interpretation of NO2 distribution. Losses from both heads are combined to optimize the model, with the ultimate goal of providing accurate, high-resolution estimations of ground-level NO2.*

## Repository Structure

- `core/`: Core scripts and modules for the project.
    - `dataset.py`: Data loading and preprocessing utilities.
    - `model.py`: Deep learning model architectures (UNet and Autoencoder).
    - `plotting.py`: Utilities for plotting results.
    - `transforms.py`: Image transformations for data augmentation.
    - `utils.py`: Miscellaneous utility functions.
- `data/`: Directory containing the datasets and samples.
    - `no2_visualization/`: Samples covering 1.2km x 1.2km for visualizing NO₂ data.
    - `samples/`: Ground truth csv files containing NO2 concentrations.
    - `sentinel-*/`: Directories for Sentinel satellite data and preprocessed versions.
    - `worldcover/`: World cover data for land cover classification.
- `models/`: Directory where trained models can be stored to.
- `evaluation.ipynb`: Jupyter notebook for evaluating model performance.
- `main.py`: Main script for training and testing the models.
- `requirements.txt`: Required Python packages for reproducing the results.
- `s5p_data_preprocessing.py`: Preprocessing script for Sentinel-5P data.
- `visualization.ipynb`: Notebook for result visualization.
- `wandb_sweep_conf.yaml`: Configuration file for weights & biases sweeps.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/yourrepositoryname.git
    cd yourrepositoryname
    ```

2. Install the necessary dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
To train the models, run:

```bash
python main.py
```
Make sure to set the correct paths in `main.py` to your data folders first.

To evaluate the models, use the `evaluation.ipynb` notebook.

For result visualization, use the `visualization.ipynb` notebook.

## Results

![Point vs. Patch-wise Comparison](/images/pointvspatchqual.png)
*Figure 2: Comparative visualization of NO2 estimation methods: the ground truth with a marked measurement point, point-wise estimation showing per-pixel predictions, and patch-wise estimation by UNet and Autoencoder models with different loss objectives. Deviations from the ground truth are indicated below each estimate.*



![More Results](/images/results.png)
*Figure 3: Visualization of ground truth NO2 levels and corresponding point-wise and patch-wise estimations using the Autoencoder model with NO2 loss, showing close approximation to actual concentrations and similar distributions as the point-wise estimates.*



![US West Coast Results](/images/useval.png)
*Figure 4: Application of the Autoencoder model with a combined loss on the US West Coast dataset, showcasing the model’s capability to adapt and estimate NO2 levels in diverse geographic settings.*



![Predicition Size Variations](/images/predspaceeval.png)
*Figure 5: Performance comparison of the Autoencoder model utilizing the combined loss across varying prediction spaces, demonstrating consistent visual and numerical accuracy as the estimation grid size increases.*
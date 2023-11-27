import torch
import numpy as np
import pandas as pd
import lightning as L
from torchvision.transforms.v2 import ToImage, Compose
from torch.utils.data import DataLoader
from torch.nn import MSELoss, L1Loss
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from src.dataset import SentinelDataset
from src.utils import get_dataset_stats
from src.transforms import BandNormalize, TargetNormalize
from src.model import UNet


# Random seed for splitting
SEED = 42
L.seed_everything(42)

# File paths
SAMPLES_PATH = (
    "/netscratch2/rubengaviles/imp-2023/data/samples/samples_S2S5P_2018_2020_eea.csv"
)
DATA_DIR = "/netscratch2/rubengaviles/imp-2023/data"

LOG_DIR = "/netscratch2/rubengaviles/imp-2023/logs"

# Hyperparameters
config = {
    "N_PATCHES": 4,
    "PATCH_SIZE": 128,
    "BATCH_SIZE": 8,
    "LEARNING_RATE": 1e-4,
    "ENCODER_CHANNELS": (12, 64, 128, 256),
    "DECODER_CHANNELS": (256, 128, 64),
}

# Read the samples file
samples_df = pd.read_csv(SAMPLES_PATH, index_col="idx")

# Remove NA measurements
samples_df = samples_df[~samples_df["no2"].isna()]

# Random shuffle
samples_df = samples_df.sample(frac=1)

# Split samples dataframe to avoid sampling patches across sets
df_train, df_val, df_test = np.split(
    samples_df, [int(0.7 * len(samples_df)), int(0.85 * len(samples_df))]
)
print(
    f"Train set: {len(df_train)}, Validation set: {len(df_val)}, Test set: {len(df_test)}"
)

# Get statistics for normalization
# stats_train = get_dataset_stats(df_train, DATA_DIR)
# print(stats_train)
stats_train = {
    "band_means": np.array(
        [
            951.7304533,
            887.65642278,
            672.43460609,
            2309.22885705,
            1283.86024167,
            1971.3361798,
            2221.09058264,
            2375.07350295,
            2061.81996558,
            1556.39466485,
            565.24740146,
            2376.78314149,
        ]
    ),
    "band_stds": np.array(
        [
            669.08142084,
            533.01636366,
            490.84281537,
            1003.74872792,
            591.48734924,
            739.32762934,
            879.07018525,
            945.72499154,
            813.62533688,
            756.78103222,
            325.27273547,
            869.95854887,
        ]
    ),
    "no2_mean": 20.973578214241755,
    "no2_std": 11.575741710970245,
}


# Create normalizers for bands and NO2 measurements
band_normalize = BandNormalize(stats_train["band_means"], stats_train["band_stds"])
no2_normalize = TargetNormalize(stats_train["no2_mean"], stats_train["no2_std"])

# Create transforms for images and measurements
transform = Compose([ToImage(), band_normalize])
target_transform = no2_normalize

# Create Train Dataset
dataset_train = SentinelDataset(
    df_train,
    DATA_DIR,
    n_patches=config["N_PATCHES"],
    patch_size=config["PATCH_SIZE"],
    pre_load=False,
    transform=transform,
    target_transform=target_transform,
)

# Create Validation Dataset
dataset_val = SentinelDataset(
    df_val,
    DATA_DIR,
    n_patches=config["N_PATCHES"],
    patch_size=config["PATCH_SIZE"],
    pre_load=False,
    transform=transform,
    target_transform=target_transform,
)

# Create Test Dataset
dataset_test = SentinelDataset(
    df_test,
    DATA_DIR,
    n_patches=config["N_PATCHES"],
    patch_size=config["PATCH_SIZE"],
    transform=transform,
    target_transform=target_transform,
)

print(
    f"Train dataset: {len(dataset_train)}, Validation dataset: {len(dataset_val)}, Test dataset: {len(dataset_test)}"
)

# Create Dataloaders
dataloader_train = DataLoader(
    dataset_train,
    batch_size=config["BATCH_SIZE"],
    shuffle=True,
    num_workers=12,
    persistent_workers=True,
)
dataloader_val = DataLoader(
    dataset_val,
    batch_size=config["BATCH_SIZE"],
    shuffle=False,
    num_workers=12,
    persistent_workers=True,
)
dataloader_test = DataLoader(dataset_test, batch_size=config["BATCH_SIZE"])


class Model(L.LightningModule):
    def __init__(self, model, lr):
        super().__init__()

        # Set model
        self.model = model

        # Set hyperparameters
        self.loss = MSELoss()
        self.mae = L1Loss()
        self.lr = lr

        self.save_hyperparameters(ignore=["model"])

    def training_step(self, batch, batch_idx):
        loss, mae = self._step(batch)
        self.log("train_loss", loss)
        self.log("train_mae", mae)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, mae = self._step(batch, batch_idx == 0)
        self.log("val_loss", loss)
        self.log("val_mae", mae)
        return loss

    def test_step(self, batch, batch_idx):
        loss, mae = self._step(batch)
        self.log("test_loss", loss)
        self.log("test_mae", mae)
        return loss

    def _step(self, batch, log_predictions=False):
        # Unpack batch
        patches_norm, measurements_norm, coords = batch

        # Get normalized predictions
        predictions_norm = self.model(patches_norm)

        # Get input and output dimensions
        _, _, input_height, input_width = patches_norm.shape
        _, _, output_height, output_width = predictions_norm.shape

        # Compute prediction offset
        offset_height = (input_height - output_height) // 2
        offset_width = (input_width - output_width) // 2

        # Apply offset to coords
        coords[0] -= offset_height
        coords[1] -= offset_width

        # Extract values in coordinate location
        target_values_norm = torch.diag(predictions_norm[:, 0, coords[0], coords[1]])

        # Compute loss on normalized data
        loss = self.loss(target_values_norm, measurements_norm)

        # Compute Mean Absolute Error on unnormalized data
        measurements = no2_normalize.revert(measurements_norm)
        target_values = no2_normalize.revert(target_values_norm)
        mae = self.mae(target_values, measurements)

        if log_predictions:
            self.logger.log_image(
                "predictions", list(no2_normalize.revert(predictions_norm))
            )

        return loss, mae

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


# Instantiate Model
unet = UNet(enc_chs=config["ENCODER_CHANNELS"], dec_chs=config["DECODER_CHANNELS"])
model = Model(model=unet, lr=config["LEARNING_RATE"])

# Get logger for weights & biases
wandb_logger = WandbLogger(
    save_dir=LOG_DIR, dir=LOG_DIR, entity="imp-2023", project="IMP-2023", config=config
)

# Configure which model to save
checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_mae", mode="min")

# Train model
trainer = L.Trainer(
    max_epochs=500,
    logger=wandb_logger,
    callbacks=[checkpoint_callback],
    log_every_n_steps=400,
)
trainer.fit(
    model=model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val
)

import os
import torch
import wandb
import numpy as np
import pandas as pd
import lightning as L
from torchvision.transforms.v2 import ToImage, Compose
from torch.utils.data import DataLoader
from torch.nn import MSELoss, L1Loss, CrossEntropyLoss
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from torchsummary import summary


from core.dataset import SentinelDataset
from core.model import UNet
from core.utils import get_dataset_stats, normalize_rgb_bands
from core.transforms import BandNormalize, TargetNormalize

# Random seed for splitting
SEED = 42
L.seed_everything(42, workers=True)

# File paths
SAMPLES_PATH = (
    "/netscratch2/rubengaviles/imp-2023/data/samples/samples_S2S5P_2018_2020_eea.csv"
)
DATA_DIR = "/netscratch2/rubengaviles/imp-2023/data"

LOG_DIR = "/netscratch2/rubengaviles/imp-2023/logs"

# Hyperparameters
config = {
    "N_PATCHES": 1,
    "PATCH_SIZE": 128,
    "BATCH_SIZE": 8,
    "PRED_SIZE": 8,
    "LEARNING_RATE": 0.000005,
    "ENCODER_CONFIG": (13, 64, 128, 256, 512, 1024),
    "DECODER_CONFIG": (1024, 512, 256, 128, 64),
    "SKIP_CONNECTIONS": False,
    "INCLUDE_LC": False,
    "LC_LOSS_WEIGHT": 0.1,
    "PRE_LOAD": True,
    "MAX_EPOCHS": 100,
}
wandb.init(config=config, dir=LOG_DIR, entity="imp-2023", project="IMP-2023")

# Read the samples file
samples_df = pd.read_csv(SAMPLES_PATH, index_col="idx")

# Remove NA measurements
samples_df = samples_df[~samples_df["no2"].isna()]


# Exclude samples for which no valid land cover ground truth is present ~200
valid_land_cover_stations = []
land_cover_path = os.path.join(DATA_DIR, "worldcover")
for file in os.listdir(land_cover_path):
    lc = np.load(os.path.join(land_cover_path, file))
    if lc.shape == (200, 200):
        valid_land_cover_stations.append(file[:-4])

samples_df = samples_df.loc[
    samples_df["AirQualityStation"].isin(valid_land_cover_stations)
]


# Random shuffle
samples_df = samples_df.sample(frac=1)
# Split samples dataframe to avoid sampling patches across sets
df_train, df_val, df_test = np.split(
    samples_df, [int(0.7 * len(samples_df)), int(0.85 * len(samples_df))]
)

print(
    f"Train set: {len(df_train)}, Validation set: {len(df_val)}, Test set: {len(df_test)}"
)

# Compute land cover class frequency
lc_class_count = {}
for station in df_train["AirQualityStation"]:
    lc = np.load(os.path.join(land_cover_path, station + ".npy"))
    classes, counts = np.unique(lc, return_counts=True)
    for i, cls in enumerate(classes):
        count = counts[i]
        curr_count = lc_class_count.get(cls, 0)
        curr_count += count
        lc_class_count[cls] = curr_count

sorted_class_counts = sorted(lc_class_count.items())

# Extract the counts and classes as separate lists
classes, counts = zip(*sorted_class_counts)

# Calculate class weights based on the provided criterion
max_class_size = max(counts)
class_weights = [max_class_size / count for count in counts]
class_weights = np.array(class_weights).astype(np.float32)

print(class_weights)


# Get statistics for normalization
# stats_train = get_dataset_stats(df_train, DATA_DIR)
# print(stats_train)
stats_train = {
    "band_means": np.array(
        [
            9.48530855e02,
            8.85735776e02,
            6.69150641e02,
            2.31917082e03,
            1.28305729e03,
            1.97936703e03,
            2.23097768e03,
            2.38542771e03,
            2.06128535e03,
            1.55304775e03,
            5.61707384e02,
            2.38793057e03,
            2.78282474e15,
        ]
    ),
    "band_stds": np.array(
        [
            6.69703046e02,
            5.34104387e02,
            4.92931981e02,
            1.00871675e03,
            5.90429095e02,
            7.41831336e02,
            8.81957446e02,
            9.48550975e02,
            8.09822953e02,
            7.57415252e02,
            3.27955892e02,
            8.74079961e02,
            1.36616750e15,
        ]
    ),
    "no2_mean": 20.973578214241755,
    "no2_std": 11.575741710970245,
}


# Create normalizers for bands and NO2 measurements
band_normalize = BandNormalize(stats_train["band_means"], stats_train["band_stds"])
no2_normalize = TargetNormalize(stats_train["no2_mean"], stats_train["no2_std"])

# Create transforms for images and measurements
s2_transform = Compose([ToImage(), band_normalize])
no2_transform = no2_normalize

# Create Train Dataset
dataset_train = SentinelDataset(
    df_train,
    DATA_DIR,
    n_patches=wandb.config["N_PATCHES"],
    patch_size=wandb.config["PATCH_SIZE"],
    pred_size=wandb.config["PRED_SIZE"],
    pre_load=wandb.config["PRE_LOAD"],
    s2_transform=s2_transform,
    no2_transform=no2_transform,
)

# Create Validation Dataset
dataset_val = SentinelDataset(
    df_val,
    DATA_DIR,
    n_patches=wandb.config["N_PATCHES"],
    patch_size=wandb.config["PATCH_SIZE"],
    pred_size=wandb.config["PRED_SIZE"],
    pre_load=wandb.config["PRE_LOAD"],
    s2_transform=s2_transform,
    no2_transform=no2_transform,
)
print(f"Train dataset: {len(dataset_train)}, Validation dataset: {len(dataset_val)}")

# Create Dataloaders
dataloader_train = DataLoader(
    dataset_train,
    batch_size=wandb.config["BATCH_SIZE"],
    shuffle=True,
    num_workers=12,
    persistent_workers=True,
)
dataloader_val = DataLoader(
    dataset_val,
    batch_size=wandb.config["BATCH_SIZE"],
    shuffle=False,
    num_workers=12,
    persistent_workers=True,
)


# Define Pytorch lightning model
class Model(L.LightningModule):
    def __init__(self, model, lr, include_lc, lc_loss_weight, lc_class_weights):
        super().__init__()

        # Set model
        self.model = model

        # Set hyperparameters
        self.no2_loss = MSELoss()
        self.no2_mae = L1Loss()
        self.lc_loss = CrossEntropyLoss(weight=torch.tensor(lc_class_weights))
        self.include_lc = include_lc
        self.lc_loss_weight = lc_loss_weight
        self.lr = lr

    def training_step(self, batch, batch_idx):
        no2_loss, no2_mae, lc_loss = self._step(batch)
        total_loss = no2_loss + (self.lc_loss_weight * lc_loss)
        self.log("train_no2_loss", no2_loss)
        self.log("train_no2_mae", no2_mae)
        self.log("train_lc_loss", lc_loss)
        self.log("train_total_loss", total_loss)
        loss = total_loss if self.include_lc else no2_loss
        return loss

    def validation_step(self, batch, batch_idx):
        no2_loss, no2_mae, lc_loss = self._step(batch, batch_idx == 0)
        total_loss = no2_loss + (self.lc_loss_weight * lc_loss)
        self.log("val_no2_loss", no2_loss)
        self.log("val_no2_mae", no2_mae)
        self.log("val_lc_loss", lc_loss)
        self.log("val_total_loss", total_loss)
        loss = total_loss if self.include_lc else no2_loss
        return loss

    def test_step(self, batch, batch_idx):
        no2_loss, no2_mae, lc_loss = self._step(batch)
        total_loss = no2_loss + (self.lc_loss_weight * lc_loss)
        self.log("test_no2_loss", no2_loss)
        self.log("test_no2_mae", no2_mae)
        self.log("test_lc_loss", lc_loss)
        self.log("test_total_loss", total_loss)
        loss = total_loss if self.include_lc else no2_loss
        return loss

    def _step(self, batch, log_predictions=False):
        # Unpack batch
        patches_norm, lc_truth, measurements_norm, coords = batch

        # Get normalized predictions
        predictions_norm, land_cover_pred = self.model(patches_norm)

        # Extract values in coordinate location
        target_values_norm = torch.diag(predictions_norm[:, 0, coords[0], coords[1]])

        # Compute loss on normalized data
        no2_loss = self.no2_loss(target_values_norm, measurements_norm)

        # Center crop
        # lc_truth = CenterCrop(land_cover_pred.shape[-2:])(lc_truth)
        lc_loss = self.lc_loss(land_cover_pred, lc_truth)

        # Compute Mean Absolute Error on unnormalized data
        measurements = no2_normalize.revert(measurements_norm)
        target_values = no2_normalize.revert(target_values_norm)
        no2_mae = self.no2_mae(target_values, measurements)

        if log_predictions:
            self.logger.log_image(
                "images",
                [
                    torch.moveaxis(normalize_rgb_bands(im.cpu()), 0, 2).numpy()
                    for im in patches_norm[:, :3]
                ],
            )
            self.logger.log_image(
                "predictions", list(no2_normalize.revert(predictions_norm))
            )

        return no2_loss, no2_mae, lc_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


# Instantiate Model
unet = UNet(
    wandb.config["ENCODER_CONFIG"],
    wandb.config["DECODER_CONFIG"],
    wandb.config["SKIP_CONNECTIONS"],
)
summary(unet.cuda(), (13, wandb.config["PATCH_SIZE"], wandb.config["PATCH_SIZE"]))
model = Model(
    model=unet,
    lr=wandb.config["LEARNING_RATE"],
    include_lc=wandb.config["INCLUDE_LC"],
    lc_loss_weight=wandb.config["LC_LOSS_WEIGHT"],
    lc_class_weights=class_weights,
)

# Get logger for weights & biases
wandb_logger = WandbLogger(save_dir=LOG_DIR)

# Configure which model to save
checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_no2_mae", mode="min")

# Train model
trainer = L.Trainer(
    max_epochs=wandb.config["MAX_EPOCHS"],
    logger=wandb_logger,
    callbacks=[checkpoint_callback],
)
trainer.fit(
    model=model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val
)

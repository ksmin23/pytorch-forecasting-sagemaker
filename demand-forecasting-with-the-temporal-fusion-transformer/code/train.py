import argparse
import os
import copy
from pathlib import Path
import shutil
import warnings

import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
  EarlyStopping,
  LearningRateMonitor
)
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.tuner import Tuner
import numpy as np
import pandas as pd
import torch

from pytorch_forecasting import (
  TemporalFusionTransformer,
  TimeSeriesDataSet
)
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss


if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  # Hyperparameters are described here.
  parser.add_argument('--batch_size', type=int, default=128, help="set batch_size between 32 to 128")
  parser.add_argument('--max_prediction_length', type=int, default=6)
  parser.add_argument('--max_encoder_length', type=int, default=24)

  # Sagemaker specific arguments. Defaults are set in the environment variables.
  parser.add_argument("--output-data-dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
  parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
  parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAIN"])

  args = parser.parse_args()

  # Take the set of files and read them all into a single pandas dataframe
  input_files = [os.path.join(args.train, file) for file in os.listdir(args.train)]
  if len(input_files) == 0:
    raise ValueError(
      (
        "There are no files in {}.\n"
        + "This usually indicates that the channel ({}) was incorrectly specified,\n"
        + "the data specification in S3 was incorrectly specified or the role specified\n"
        + "does not have permission to access the data."
      ).format(args.train, "train")
    )
  raw_data = [pd.read_parquet(file) for file in input_files]
  data = pd.concat(raw_data)

  # we want too encode special days as one variable and thus need to first reverse one-hot encoding
  special_days = [
    "easter_day",
    "good_friday",
    "new_year",
    "christmas",
    "labor_day",
    "independence_day",
    "revolution_day_memorial",
    "regional_games",
    "fifa_u_17_world_cup",
    "football_gold_cup",
    "beer_capital",
    "music_fest",
  ]

  ###################################
  # create dataset and dataloaders
  ###################################
  max_prediction_length = args.max_prediction_length
  max_encoder_length = args.max_encoder_length
  training_cutoff = data["time_idx"].max() - max_prediction_length

  training = TimeSeriesDataSet(
    data[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="volume",
    group_ids=["agency", "sku"],
    min_encoder_length=max_encoder_length // 2, # keep encoder length long (as it is in the validation set)
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    static_categoricals=["agency", "sku"],
    static_reals=["avg_population_2017", "avg_yearly_household_income_2017"],
    time_varying_known_categoricals=["special_days", "month"],
    variable_groups={"special_days": special_days}, # group of categorical variables can be treated as one variable
    time_varying_known_reals=["time_idx", "price_regular", "discount_in_percent"],
    time_varying_unknown_categoricals=[],
    time_varying_unknown_reals=[
      "volume",
      "log_volume",
      "industry_volume",
      "soda_volume",
      "avg_max_temp",
      "avg_volume_by_agency",
      "avg_volume_by_sku",
    ],
    target_normalizer=GroupNormalizer(
      groups=["agency", "sku"],
      transformation="softplus"
    ), # use softplus and normalize by group
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True
  )

  # create validation set (predict=True) which means to predict the last max_prediction_length points in time for each series
  validation = TimeSeriesDataSet.from_dataset(
    training,
    data,
    predict=True,
    stop_randomization=True
  )

  batch_size = args.batch_size
  train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
  val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)

  ###################################
  # configure network and trainer
  ###################################
  pl.seed_everything(42)

  trainer = pl.Trainer(
    accelerator="cpu",
    # clipping gradients is a hyperparameter and important to prevent divergance of the gradient for recurrent neural networks
    gradient_clip_val=0.1
  )

  tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.03,
    hidden_size=8,
    attention_head_size=1,
    dropout=0.1,
    hidden_continuous_size=8,
    loss=QuantileLoss(),
    optimizer="Ranger"
    # reduce_on_plateau_patience=1000,
  )
  print(f"Number of parameters in network: {tft.size() / 1e3: .1f} k")

  ###################################
  # find optimal learning rate
  ###################################
  res = Tuner(trainer).lr_find(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
    max_lr=10.0,
    min_lr=1e-6,
  )
  print(f"suggested learning rate: {res.suggestion()}")

  ###################################
  # configure network and trainer
  ###################################
  early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
  lr_logger = LearningRateMonitor() # log the learning rate
  logger = TensorBoardLogger("lightning_logs") # logging results to a tensorboard

  trainer = pl.Trainer(
    max_epochs=50,
    accelerator="cpu",
    enable_model_summary=True,
    gradient_clip_val=0.1,
    limit_train_batches=50, # coment in for training, running valiation every 30 batches
    # fast_dev_run=True, # comment in to check that networkor dataset has no serious bugs
    callbacks=[lr_logger, early_stop_callback],
    logger=logger,
  )

  tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.03,
    hidden_size=16,
    attention_head_size=2,
    dropout=0.1,
    hidden_continuous_size=8,
    loss=QuantileLoss(),
    log_interval=10, # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
    optimizer="Ranger",
    reduce_on_plateau_patience=4,
  )
  print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

  ###################################
  # fit network
  ###################################
  trainer.fit(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
  )

  ###################################
  # save model
  ###################################
  # load the best model according to the validation loss
  # (given that we use early stopping, this is not necessarily the last epoch)
  best_model_path = trainer.checkpoint_callback.best_model_path
  print(f"Best model path: {best_model_path}")

  shutil.copyfile(best_model_path, os.path.join(args.model_dir, "model.ckpt"))


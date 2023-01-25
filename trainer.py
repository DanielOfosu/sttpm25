import os
import uuid

import deepspeed
import dill as pickle
import pytorch_lightning as pl
import torch
import utils
import wandb
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.strategies import DeepSpeedStrategy
from pytorch_lightning.utilities.deepspeed import \
    convert_zero_checkpoint_to_fp32_state_dict

torch.set_float32_matmul_precision('medium')

# Load the variables from the file
with open("trainer_vars.pkl", "rb") as f:
    variables = pickle.load(f)
# Callbacks

filename = f"{variables['run_name']}"


model_ckpt_dir = variables["config"]["save_dir"]

saving = pl.callbacks.ModelCheckpoint(
    dirpath=model_ckpt_dir,
    monitor="val/loss",
    mode="min",
    filename=f"{variables['run_name']}",
    save_top_k=1,
    auto_insert_metric_name=True,
) 
callbacks = [saving]

if variables["config"]["earlystopping"] == True:
    callbacks.append(
        pl.callbacks.early_stopping.EarlyStopping(
            monitor="val/mae",
            patience=variables["config"]["patience"],
            verbose=True,
        )
    )

callbacks.append(pl.callbacks.LearningRateMonitor())
callbacks.append(TQDMProgressBar(refresh_rate=10))
callbacks.append(
    utils.plot.AttentionMatrixCallback(
        variables["test_samples"],
        layer=0,
        total_samples=min(16, variables["config"]["batch_size"]), # Testing different numbers than 16 to try to fix grad accum
    )
)

# Wandb Logging
log_dir = "./data/wandb_logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
project = "thesis"
entity = "danielofosu"

experiment = wandb.init(
    project=project,
    entity=entity,
    config=variables["config"],
    dir=log_dir,
    reinit=True,
)
#config = wandb.config
wandb.run.name = variables["run_name"]
wandb.run.save()
logger = pl.loggers.WandbLogger(
    experiment=experiment,
    save_dir=log_dir,
)

deepspeed_config = {
    "zero_allow_untested_optimizer": True,
    "bf16": {
        "enabled": True,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "overlap_comm": True,
        "contiguous_gradients": True,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },
    }
}

pl_trainer = pl.Trainer(
        callbacks=callbacks,
        enable_checkpointing=True,
        logger=logger,
        accelerator="cuda", #CHANGE TO CUDA FOR EXTERNAL 
        #precision=16, # helps but causes nan values
        precision="bf16",
        max_epochs=20,
        devices=1,
        gradient_clip_algorithm="norm",
        #accumulate_grad_batches=1,
        overfit_batches=20 if variables["config"]["debug"] else 0,
        strategy=DeepSpeedStrategy(
            config=deepspeed_config,
            load_full_weights=False,
        ),
        sync_batchnorm=False, # Only needed for multi-gpu
        #limit_val_batches=variables["config"]["limit_val_batches"],
        **variables["val_control"],
    )

# Train
pl_trainer.fit(variables["forecaster"], datamodule=variables["data_module"])

# Test
pl_trainer.test(datamodule=variables["data_module"], ckpt_path=callbacks[0].best_model_path) 

experiment.finish()
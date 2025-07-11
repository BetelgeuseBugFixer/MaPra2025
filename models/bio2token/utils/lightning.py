import torch.nn as nn
import torch
import lightning as L
from torch.optim.optimizer import Optimizer
from typing import Any, Union, List, Optional
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import ConstantLR, PolynomialLR, ReduceLROnPlateau, LinearLR, CosineAnnealingLR, StepLR
from dataclasses import dataclass, field
import os
from typing import Literal


@dataclass(kw_only=True)
class PretrainedModelConfig:
    run_id: Optional[str] = None
    checkpoint_dir: str = "checkpoints"
    checkpoint_type: Literal["best", "last"] = "last"
    checkpoint_monitor: str = "val_loss_epoch"
    checkpoint_mode: Literal["min", "max"] = "min"


@dataclass(kw_only=True)
class ModelCheckpointingConfig:
    checkpoint_k_best_to_save: int = 10  # set to -1 to save all
    checkpoint_monitor: str = "val_loss_epoch"
    checkpoint_mode: Literal["min", "max"] = "min"
    checkpoint_dir: str = "checkpoints"


@dataclass
class OptimizerConfig:
    optimizer: Optional[str] = None
    lr: Optional[float] = None
    weight_decay: Optional[float] = None
    optimizer_args: Optional[dict] = None
    schedule: Optional[str] = None
    schedule_args: Optional[dict] = None
    continue_training: Optional[bool] = False
    pretrained_model: PretrainedModelConfig = field(default_factory=lambda: PretrainedModelConfig())
    checkpointing: ModelCheckpointingConfig = field(default_factory=lambda: ModelCheckpointingConfig())


class NetworkModule(L.LightningModule):

    def __init__(self, model: nn.Module, config: OptimizerConfig = None, matmul_precision: str = "medium"):
        torch.set_float32_matmul_precision(matmul_precision)
        super().__init__()
        self.config = config
        self.model = model
        self.model.dtype = self.dtype

    def configure_optimizers(self) -> dict[str, Union[Optimizer, dict[str, Any]]]:
        if not self.config is None:

            optimizer = get_opt(self.model.parameters(), self.config)
            self.model.opt = optimizer

            if self.config.schedule is not None:
                scheduler = get_lr(
                    optimizer=optimizer,
                    config=self.config,
                )
                self.model.lr_sched = scheduler["scheduler"]
                return {"optimizer": optimizer, "lr_scheduler": scheduler}
            else:
                return {"optimizer": optimizer}

    def training_step(self, batch: Any) -> torch.Tensor:
        batch = self.model(batch)
        losses = batch["losses"]
        losses["loss"] = losses["loss"].mean()
        losses = {k: v.mean() for k, v in losses.items()}
        self.log_dict(losses, on_step=True, on_epoch=True, logger=True, prog_bar=False)
        self.log_lr()
        return losses

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Union[torch.Tensor, dict[str, Any]]:
        batch = self.model(batch)
        losses = batch["losses"]
        losses = {f"val_{k}": v.mean() for k, v in losses.items()}
        self.log_dict(losses, on_step=True, on_epoch=True, logger=True, prog_bar=False)
        return losses

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Union[torch.Tensor, dict[str, Any]]:
        with torch.no_grad():
            batch = self.model(batch)
            losses = batch["losses"]
            for k, v in losses.items():
                batch[k] = v
            batch.pop("losses", None)
        return batch

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)

    def log_lr(self):
        schedulers = self.lr_schedulers()
        if schedulers is not None:
            if not isinstance(schedulers, List):
                lr = schedulers._last_lr[0]
                self.log(f"learning_rate", lr, on_step=True, logger=True)
            else:
                for idx, s in enumerate(schedulers):
                    lr = s._last_lr[0]
                    self.log(f"learning_rate_{idx}", lr, on_step=True, logger=True)


def get_opt(params: Any, config: OptimizerConfig) -> Optimizer:
    opt_type = config.optimizer

    kw = config.optimizer_args if config.optimizer_args is not None else {}
    if config.lr is not None:
        kw["lr"] = config.lr
    if config.weight_decay is not None:
        kw["weight_decay"] = config.weight_decay

    if (opt_type == "adam") or (opt_type is None):
        opt = Adam(params, **kw)
    elif opt_type == "adamW":
        opt = AdamW(params, **kw)
    elif opt_type == "sgd":
        opt = SGD(params, **kw)
    else:
        raise NotImplementedError(f"Optimizer {opt_type} not implemented, please choose from adam, adamW, or sgd")
    return opt


def get_lr(optimizer: Optimizer, config: OptimizerConfig) -> dict[str, Any]:
    sched_type = config.schedule

    d = {"interval": "step", "frequency": 1}

    if sched_type == "constant":
        sched = ConstantLR(optimizer, **config.schedule_args)
    elif sched_type == "poly":
        sched = PolynomialLR(optimizer, **config.schedule_args)
    elif sched_type == "plateau":
        sched = ReduceLROnPlateau(optimizer, **config.schedule_args)
    elif sched_type == "linear":
        sched = LinearLR(optimizer, **config.schedule_args)
    elif sched_type == "cosine":
        sched = CosineAnnealingLR(optimizer, **config.schedule_args)
    elif sched_type == "step":
        sched = StepLR(optimizer, **config.schedule_args)
    else:
        raise NotImplementedError(
            f"Scheduler {sched_type} not implemented, please choose from constant, poly, plateau, linear, cosine, or step"
        )

    d["scheduler"] = sched

    return d


def find_lowest_val_loss_checkpoint(
    checkpoint_dir: str, checkpoint_monitor: str, checkpoint_mode: Literal["min", "max"]
) -> str:
    lowest_val_loss = float("inf")
    highest_val_loss = float("-inf")
    best_checkpoint = None

    # List all checkpoint files
    for filename in os.listdir(checkpoint_dir):
        if filename.endswith(".ckpt") and checkpoint_monitor in filename:
            # Extract the val_loss_epoch from the filename
            parts = filename.split("-")
            for part in parts:
                if part.startswith(f"{checkpoint_monitor}="):
                    val_loss_str = part.split("=")[1]  # extract the loss value
                    val_loss = float(val_loss_str)

                    # Update best checkpoint if the current loss is lower
                    if checkpoint_mode == "min" and val_loss < lowest_val_loss:
                        lowest_val_loss = val_loss
                        best_checkpoint = filename
                    elif checkpoint_mode == "max" and val_loss > highest_val_loss:
                        highest_val_loss = val_loss
                        best_checkpoint = filename

    return os.path.join(checkpoint_dir, best_checkpoint)

import lightning as L
import torch
from torch import Tensor
from typing import *
from loguru import logger
import pandas as pd
import os
import numpy as np


class CustomProgressBar(L.pytorch.callbacks.TQDMProgressBar):
    def __init__(self):
        super().__init__()
        self.train_loss = None

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Update the train_loss with the latest value
        self.train_loss = outputs["loss"] if isinstance(outputs, dict) else outputs
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

    def get_metrics(self, trainer, model):
        metrics = super().get_metrics(trainer, model)
        del metrics["v_num"]
        metrics["loss"] = f"{self.train_loss:.4f}"
        return metrics


class StopOnNaNCallback(L.Callback):
    def __init__(
        self,
        save_file: str = "/tmp/badstep.ckpt",
        try_inference: bool = False,
        warmup_steps: int = 0,
    ):
        """A callback to stop and inspect training if a NaN loss is encountered.

        If using with AMP, you may want to set `warmup_steps>0`. The first few steps may produce inf losses
        as the loss scaling is being set.

        Some useful information is dumped to `save_file`:
        - the model state dict
        - the loss (so you can tell if it's NaN vs inf vs...)
        - the batch
        If `try_inference` is set:
        - the output of `model.inference_step(batch)`
        - the exception if the above does not work

        Args:
            save_file (str): Path to save the dump file. Defaults to "/tmp/badstep.ckpt".
            try_inference (bool): Whether to attempt an inference step. Defaults to False.
            warmup_steps (int): This callback will only come into effect after `warmup_steps` steps. Defaults to 0.
        """
        self.save_file = save_file
        self.try_inference = try_inference
        self.warmup_steps = warmup_steps
        self.prev_batch = None

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        loss = outputs["loss"]
        if batch_idx >= self.warmup_steps and not (loss == loss):
            model = module.model
            assert not isinstance(model, Tensor)
            ckpt = {
                "state": model.state_dict(),
                "loss": loss,
            }

            if self.try_inference:
                try:
                    assert not isinstance(model.inference_step, Tensor)
                    ckpt["inference"] = model.inference_step(batch)
                except Exception as e:
                    ckpt["inference_exception"] = e

            ckpt["batch"] = batch

            if self.prev_batch is not None:
                ckpt["prev_batch"] = self.prev_batch

            torch.save(ckpt, self.save_file)
            msg = f"Encountered a NaN loss. State saved to {self.save_file} for inspection. Stopping training."
            logger.error(msg)
            trainer.should_stop = True

        else:
            self.prev_batch = batch


class StorerCallback(L.Callback):
    """
    A callback that stores model outputs at the end of the test epoch.

    Args:
        save_dir (str): Directory where to save the outputs
        try_keeping_save (list): List of keys to save in the output
        try_adding_summary (list): List of keys to include in the summary statistics
        tag (str): Tag to use for the saved file
    """

    def __init__(self, save_dir: str, try_keeping_save: list, try_adding_summary: list, tag: str = "test_outputs"):
        super().__init__()
        self.save_dir = save_dir
        self.try_keeping_save = try_keeping_save
        self.try_adding_summary = try_adding_summary
        self.tag = tag
        self.test_outputs = {k: [] for k in try_keeping_save + try_adding_summary}

        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

    def on_test_batch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule, outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        """Collect outputs from each test batch"""
        if outputs is not None:
            # Output is a dict of tensors, with length B x L x .... There is a mask eos_pad_mask of size B x L
            # For each key and each sample b, we keep only the elements where eos_pad_mask[b, :] is True, and store that in a list of lists.
            for k, v in outputs.items():
                if k == "eos_pad_mask":
                    continue
                if k in self.try_keeping_save + self.try_adding_summary:
                    if v.ndim > 1:
                        outputs_list = [
                            v[b][outputs["eos_pad_mask"][b] == 0].detach().cpu().numpy().flatten() for b in range(v.shape[0])
                        ]
                    else:
                        outputs_list = [v[b].detach().cpu().item() for b in range(v.shape[0])]
                    self.test_outputs[k] += outputs_list

    def on_test_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Save all collected outputs at the end of testing"""
        if not self.test_outputs:
            return

        # Convert buffer to DataFrames
        df = pd.DataFrame(self.test_outputs)

        # Print summary statistics
        if self.try_adding_summary:
            summary_cols = [col for col in self.try_adding_summary if col in df.columns]
            if summary_cols:
                print("Generation summary: \n" + str(df[summary_cols].describe()))

        # Save the DataFrame
        output_path = os.path.join(self.save_dir, f"{self.tag}.parquet")
        df.to_parquet(output_path)
        print(f"Saved outputs to {output_path}")

        # Clear the outputs
        self.test_outputs = {k: [] for k in self.try_keeping_save + self.try_adding_summary}

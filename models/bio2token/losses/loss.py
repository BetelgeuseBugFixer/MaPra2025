from dataclasses import dataclass
from typing import Dict, Optional
import torch.nn as nn
import torch

from models.bio2token.losses.rmsd import RMSD
from models.bio2token.losses.tm import TM
from models.bio2token.losses.inter_atom_distance import InterAtomDistance


@dataclass(kw_only=True)
class LossesConfig:
    """
    Configuration for managing losses and monitors in a model training workflow.

    This dataclass provides a structured format for specifying configurations related to
    various loss functions and monitoring metrics used during the training process.

    Attributes:
        loss (Optional[dict]): A dictionary of configurations for loss functions.
                               Each entry defines a specific loss function setup, including
                               its type and associated parameters.
        monitor (Optional[dict]): A dictionary of configurations for monitoring metrics.
                                  Each entry describes a monitoring operation or metric,
                                  specifying its type and relevant parameters.
    """

    loss: Optional[dict] = None
    monitor: Optional[dict] = None


class Losses(nn.Module):
    """
    A module to handle multiple loss functions and monitoring metrics for model training.

    The Losses class enables the integration and management of various loss functions and
    monitors within a training framework. Loss calculations contribute to the model's
    optimization objective, while monitors provide additional metrics that can guide
    evaluation and performance assessment.

    Attributes:
        config_cls (type): The associated configuration class used to initialize the module, set to LossesConfig.
        LOSS_REGISTRY (dict): A dictionary mapping string identifiers to the corresponding loss/monitor classes.
        config (LossesConfig): An instance of LossesConfig containing the setup for loss and monitor functions.
        loss_list (nn.ModuleList): A list of loss function modules constructed from the configuration.
        monitor_list (nn.ModuleList): A list of monitor modules constructed from the configuration.

    Args:
        config (Optional[LossesConfig]): An optional configuration object specifying loss and monitoring parameters.

    Methods:
        forward(batch: Dict) -> Dict:
            Processes the input batch, computing specified losses and monitors,
            and updates the batch with the results.

    Notes:
        - The configuration for each loss and monitor is expected to include a "loss_type" key
          which specifies the type of loss or monitoring operation to instantiate.
        - The rest of the configuration keys are passed as keyword arguments to the corresponding class.
        - Additional losses can be added by including their classes in the LOSS_REGISTRY dictionary.
        - Each loss function should include a "weight" argument to define its contribution to the total loss.
    """

    config_cls = LossesConfig
    LOSS_REGISTRY = {
        "tm": TM,
        "rmsd": RMSD,
        "inter_atom_distance": InterAtomDistance,
    }

    def __init__(self, config: Optional[LossesConfig] = None):
        """
        Initialize the Losses module with specified configurations for losses and monitors.

        Constructs the respective loss and monitor objects based on the provided configuration,
        allowing for flexible integration into training and evaluation workflows.

        Args:
            config (Optional[LossesConfig]): Configuration detailing the setup of losses and monitors
                                             to be applied during forward processing.
        """
        super(Losses, self).__init__()
        self.config = config
        self.loss_list, self.monitor_list = [], []
        if config is not None and config.loss is not None:
            self.loss_list = nn.ModuleList(
                [
                    self.LOSS_REGISTRY[config.loss[key]["loss_type"]](
                        self.LOSS_REGISTRY[config.loss[key]["loss_type"]].config_cls(**config.loss[key]), key
                    )
                    for key in config.loss
                ]
            )
        if config is not None and config.monitor is not None:
            self.monitor_list = nn.ModuleList(
                [
                    self.LOSS_REGISTRY[config.monitor[key]["loss_type"]](
                        self.LOSS_REGISTRY[config.monitor[key]["loss_type"]].config_cls(**config.monitor[key]), key
                    )
                    for key in config.monitor
                ]
            )

    def forward(self, batch: Dict) -> Dict:
        """
        Compute specified losses and monitor metrics on the input batch.

        This method updates the input batch dictionary with calculated loss values and any
        monitor metrics specified in the module's configuration, aiding in model optimization
        and performance diagnostics.

        Args:
            batch (Dict): A dictionary containing input data to be processed.

        Returns:
            Dict: An updated dictionary with computed losses and monitors added to the original batch data.
        """
        # Initialize the losses dictionary.
        batch["losses"] = {"loss": 0}

        # Compute the losses from the configuration and add to the total loss.
        if len(self.loss_list) > 0:
            for loss in self.loss_list:
                batch = loss(batch)
                # Accumulate the scaled loss into the total loss.
                batch["losses"]["loss"] += loss.config.weight * batch["losses"][loss.name]

        # Compute the monitor metrics, respecting evaluation settings.
        if len(self.monitor_list) > 0:
            with torch.no_grad():
                for monitor in self.monitor_list:
                    if (monitor.config.eval_only and not self.training) or (not monitor.config.eval_only):
                        batch = monitor(batch)

        return batch


if __name__ == "__main__":
    # Select the computation device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up configuration for the Losses module with specified loss and monitor configurations
    config = LossesConfig(
        loss={
            "rmsd": {
                "loss_type": "rmsd",  # Specifies RMSD as the loss type
                "prediction_name": "pred",  # Key in the batch for predictions
                "target_name": "target",  # Key in the batch for target values
                "mask_name": "mask",  # Key in the batch for mask used in loss calculation
            }
        },
        monitor={
            "tm": {
                "loss_type": "tm",  # Specifies TM as the monitor type
                "prediction_name": "pred",  # Key in the batch for predictions
                "target_name": "target",  # Key in the batch for target values
                "mask_name": "mask",  # Key in the batch for mask used in monitor calculation
                "seq_type": "protein",  # Type of sequence used in the monitor, e.g., protein
            }
        },
    )

    # Instantiate the Losses model using the configuration and move it to the selected device
    model = Losses(config).to(device)

    # Create a dummy batch of input data with predicted values, targets, and a mask
    batch = {
        "pred": torch.randn(4, 10, 3).to(device),  # Random tensor simulating predicted coordinates
        "target": torch.randn(4, 10, 3).to(device),  # Random tensor simulating target coordinates
        "mask": torch.randn(4, 10).to(device) > -0.5,  # Mask tensor for filtering valid values in the batch
    }

    # Perform a forward pass through the Losses model, updating the batch with calculated losses and monitors
    batch = model(batch)

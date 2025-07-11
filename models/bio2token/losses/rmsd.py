from dataclasses import dataclass
from typing import Optional, Dict
import torch.nn as nn
import torch


@dataclass
class RMSDConfig:
    """
    Configuration parameters for computing the Root Mean Square Deviation (RMSD).

    This dataclass contains the settings necessary to configure the calculation of RMSD,
    which measures the average deviation between predicted and target atomic coordinates.

    Attributes:
        prediction_name (str): The key in the input batch dictionary for accessing predicted coordinates.
        weight (float): The weight applicable to the RMSD in overall loss calculations. Defaults to 1.0.
        loss_type (str): Identifier for the type of loss, defaulting to "rmsd".
        target_name (Optional[str]): Key for accessing target coordinates in the batch dictionary.
        mask_name (Optional[str]): Key for accessing an optional mask to focus on specific data during computation.
        root (bool): Specifies whether to return the root of the mean square deviation. Defaults to True.
            IMPORTANT: Square root is usually not recommanded when training because not differentiable at 0, and
            leads to large gradients near 0. Use at your own risk.
        eval_only (bool): Indicates if RMSD should be computed only during evaluation, not during training. Defaults to False.
    """

    prediction_name: str
    weight: float = 1.0
    loss_type: str = "rmsd"
    target_name: Optional[str] = None
    mask_name: Optional[str] = None
    root: bool = True
    eval_only: bool = False


class RMSD(nn.Module):
    """
    A module to compute the Root Mean Square Deviation (RMSD) between predictions and targets.

    The RMSD class calculates the average deviation between predicted values and target values,
    optionally using a mask to select specific data. The RMSD provides a measure of how well
    predicted coordinates align with actual coordinates, common in structural bioinformatics.

    Attributes:
        config_cls (type): The associated configuration class, set to RMSDConfig.
        config (RMSDConfig): An instance of RMSDConfig providing parameters for RMSD calculation.
        name (str): Name of this RMSD instance, used for identifying the loss in output dictionaries.

    Args:
        config (RMSDConfig): Configuration object specifying how the RMSD should be calculated.
        name (str): Identifier for this RMSD loss module, used as a key in the output batch dictionary.

    Methods:
        forward(batch: Dict) -> Dict:
            Computes the RMSD for a batch and updates the batch with the result.
    """

    config_cls = RMSDConfig

    def __init__(self, config: RMSDConfig, name: str):
        """
        Initialize the RMSD module with a specified configuration.

        Args:
            config (RMSDConfig): Configuration specifying parameters for RMSD computation.
            name (str): Name identifier for this RMSD instance, used in output references.
        """
        super(RMSD, self).__init__()
        self.config = config
        self.name = name

    def forward(self, batch: Dict) -> Dict:
        """
        Compute the RMSD for a batch, updating the batch with the calculated loss.

        This method calculates the RMSD between predicted and target coordinates. If a mask is provided,
        it focuses calculation only on selected data. The result is stored in the batch for further use.

        Args:
            batch (Dict): Dictionary containing the data, including prediction and target coordinates and optional masks.

        Returns:
            Dict: Updated batch dictionary with the calculated RMSD loss included.
        """
        # Extract prediction and target coordinates; provide zeros as fallback for targets
        P = batch[self.config.prediction_name]
        Q = batch[self.config.target_name] if self.config.target_name is not None else P.zeros_like()

        # Retrieve optional mask for selective computation
        mask = batch[self.config.mask_name] if self.config.mask_name is not None else None

        # Compute the squared differences between predicted and target coordinates
        squared_diff = torch.sum((P - Q) ** 2, dim=-1)  # Calculate squared Euclidean distances

        # Compute the RMSD using the mask if provided, otherwise compute mean squared difference
        if mask is not None:
            rmsd = torch.sum(squared_diff * mask, dim=1) / (mask.sum(dim=1) + 1e-6)
        else:
            rmsd = torch.mean(squared_diff, dim=1)

        # Optionally take the square root of the mean squared deviation to get RMSD
        # ATTENTION: Square root is usually not recommanded when training because not differentiable at 0, and
        # leads to large gradients near 0. Use at your own risk.
        if self.config.root:
            rmsd = torch.sqrt(rmsd + 1e-6)  # Add small value to prevent divisions by zero

        # Store the computed RMSD in the batch under the configured name
        batch["losses"][self.name] = rmsd

        return batch


if __name__ == "__main__":
    # Determine the computation device (using GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Configure the RMSD calculation with necessary parameters
    config = RMSDConfig(
        prediction_name="predictions",  # Key for accessing prediction data in the batch
        target_name="targets",  # Key for accessing target data in the batch
        mask_name="mask",  # Key for accessing an optional mask in the batch
    )

    # Instantiate the RMSD module with the given configuration and name
    rmsd_metric = RMSD(config, name="rmsd").to(device)

    # Create a sample batch of data with random predictions and targets
    batch = {
        "predictions": torch.randn(4, 10, 3).to(device),  # Randomly generated 3D coordinates for predictions
        "targets": torch.randn(4, 10, 3).to(device),  # Randomly generated 3D coordinates for targets
        "mask": torch.randint(0, 2, (4, 10)).bool().to(device),  # Random binary mask for filtering data
    }

    # Compute the RMSD using the forward method
    batch["losses"] = {}
    batch = rmsd_metric(batch)

    # Print the updated batch to observe the calculated RMSD
    print("Batch with computed RMSD:")
    print(batch)

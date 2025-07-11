from dataclasses import dataclass
from typing import Optional, Dict
import torch.nn as nn
import torch


@dataclass
class TMConfig:
    """
    Configuration class for the Template Modeling (TM) score calculation.

    This dataclass encapsulates the configuration details required for computing
    the TM score, which assesses the structural similarity between predicted and target
    sequences, typically used in structural biology for protein and RNA.

    Attributes:
        prediction_name (str): Key in the batch dictionary to access predicted structures.
        seq_type (str): Sequence type for TM calculation, either "protein" or "rna". Defaults to "protein".
        weight (float): Weight of the TM score in the total loss calculation. Defaults to 1.0.
        loss_type (str): String identifier for the loss type, defaulting to "tm".
        target_name (Optional[str]): Key to access target structures in the batch dictionary.
        mask_name (Optional[str]): Key for accessing an optional mask to focus TM calculation on specific data.
        eval_only (bool): If True, the TM score is computed only during evaluation. Defaults to False.
    """

    prediction_name: str
    seq_type: str = "protein"
    weight: float = 1.0
    loss_type: str = "tm"
    target_name: Optional[str] = None
    mask_name: Optional[str] = None
    eval_only: bool = False


class TM(nn.Module):
    """
    Module for computing the Template Modeling (TM) score.

    The TM class provides a method to compute the TM score, a metric for evaluating
    the structural alignment between predicted and target structures, applicable
    to protein and RNA sequences. The TM score is a normalized measure of similarity
    accounting for the inherent properties of these molecules.

    Attributes:
        config_cls (type): Configuration class associated with TM, set to TMConfig.
        config (TMConfig): An instance of TMConfig, containing setup parameters for TM score calculation.
        name (str): Identifier for this TM instance, used in output dictionaries.

    Args:
        config (TMConfig): Configuration object specifying how the TM score should be computed.
        name (str): Name used as a key in output dictionary for the computed TM score.

    Methods:
        forward(batch: Dict) -> Dict:
            Computes the TM score for a batch of data and updates the batch dictionary.
    """

    config_cls = TMConfig

    def __init__(self, config: TMConfig, name: str):
        """
        Initialize the TM module with the given configuration.

        Args:
            config (TMConfig): Configuration specifying parameters for TM score computation.
            name (str): Name identifier for this TM instance, used in the output batch.
        """
        super(TM, self).__init__()
        self.config = config
        self.name = name

    def forward(self, batch: Dict) -> Dict:
        """
        Compute the TM score for a given batch, updating it with the calculated score.

        The method evaluates structural similarity between predictions and targets.
        The TM score calculation adjusts based on the sequence type (protein or RNA),
        reflecting specific structural characteristics and length considerations.

        Args:
            batch (Dict): Input dictionary containing predicted and target structures, and optional masks.

        Returns:
            Dict: Updated batch dictionary, containing the calculated TM score.
        """
        # Extract predicted (P) and target (Q) coordinates from the batch.
        P = batch[self.config.prediction_name]
        Q = batch[self.config.target_name] if self.config.target_name is not None else P.zeros_like()

        # Retrieve the mask if specified in the configuration, defaulting to None.
        mask = batch[self.config.mask_name] if self.config.mask_name is not None else None

        # Compute squared Euclidean distances between predicted and target coordinates.
        d = torch.sum((P - Q) ** 2, dim=-1)

        # Handle the case where masking is applied.
        if mask is not None:
            # Calculate length using the mask and determine sequence-specific scaling factor (d0).
            N = torch.sum(mask, axis=-1)
            if self.config.seq_type == "rna":
                d0 = 1.24 * torch.pow(N - 15, 1 / 3) - 1.8
            elif self.config.seq_type == "protein":
                d0 = 0.6 * torch.pow(N - 0.5, 1 / 2) - 2.5
            else:
                raise ValueError(f"Unknown sequence type: {self.config.seq_type}. Should be rna or protein")
            d0 = torch.clamp(d0, min=0.5) ** 2

            # Compute TM score by evaluating scaled distances, focusing only on masked elements.
            tm_score = torch.sum((1 / (1 + (d / d0.unsqueeze(-1)))) * mask, dim=-1) / N
        else:
            # Handle the unmasked case, using all elements.
            N = d.new_ones(P.shape[0]) * P.shape[1]
            if self.config.seq_type == "rna":
                d0 = 1.24 * torch.pow(N - 15, 1 / 3) - 1.8
            elif self.config.seq_type == "protein":
                d0 = 0.6 * torch.pow(N - 0.5, 1 / 2) - 2.5
            else:
                raise ValueError(f"Unknown sequence type: {self.config.seq_type}. Should be rna or protein")
            d0 = torch.clamp(d0, min=0.5) ** 2

            # Compute TM score considering all elements without masking.
            tm_score = torch.sum((1 / (1 + (d / d0.unsqueeze(-1)))), dim=-1) / N

        # Store the computed TM score in the batch dictionary under the configured name.
        batch["losses"][self.name] = tm_score
        return batch


if __name__ == "__main__":
    # Determine the computation device (use GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define configuration for the TM score calculation with necessary parameters
    config = TMConfig(
        prediction_name="predictions",  # Key for accessing predicted data in the batch
        target_name="targets",  # Key for accessing target data in the batch
        mask_name="mask",  # Key for accessing an optional mask in the batch
        seq_type="protein",  # Sequence type for the calculation, either 'protein' or 'rna'
    )

    # Instantiate the TM module with the given configuration and name
    tm_metric = TM(config, name="tm").to(device)

    # Create a sample data batch with random predictions and targets
    batch = {
        "predictions": torch.randn(4, 10, 3).to(device),  # Randomly generated 3D coordinates for predictions
        "targets": torch.randn(4, 10, 3).to(device),  # Randomly generated 3D coordinates for targets
        "mask": torch.randint(0, 2, (4, 10)).bool().to(device),  # Random binary mask for filtering data
    }

    # Compute the TM score using the forward method
    batch["losses"] = {}
    batch = tm_metric(batch)

    # Print the updated batch to observe the calculated TM score
    print("Batch with computed TM score:")
    print(batch)

from dataclasses import dataclass
from typing import Optional, Dict
import torch.nn as nn
import torch


@dataclass
class InterAtomDistanceConfig:
    """
    Configuration for computing the Inter-Atom Distance loss.

    This dataclass holds parameters necessary to configure the Inter-Atom Distance
    loss calculation, focusing on the alignment between predicted and target atomic
    distances across molecular structures.

    Attributes:
        prediction_name (str): Key in the batch to access predicted atomic coordinates.
        weight (float): Weight of the inter-atom distance loss in the total loss calculation. Defaults to 1.0.
        loss_type (str): Identifier string for the loss type, defaulting to "inter_atom_distance".
        target_name (Optional[str]): Key for accessing target atomic coordinates in the batch.
        mask_name (Optional[str]): Key for an optional mask to include/exclude specific atoms in the computation.
        res_idx_name (Optional[str]): Key for accessing residue indices to determine intra-residue interactions.
        root (bool): Whether to compute the square root of the mean squared deviation. Defaults to True.
        eval_only (bool): Indicates if the loss should be computed only during evaluation phases. Defaults to False.
    """

    prediction_name: str
    weight: float = 1.0
    loss_type: str = "inter_atom_distance"
    target_name: Optional[str] = None
    mask_name: Optional[str] = None
    res_idx_name: Optional[str] = None
    root: bool = True
    eval_only: bool = False


class InterAtomDistance(nn.Module):
    """
    Module to compute the Inter-Atom Distance loss between predictions and targets.

    This class calculates the loss based on the difference in inter-atomic distances
    between predicted and target coordinates, allowing evaluation of how well predicted
    structures align with their expected counterparts.

    Attributes:
        config_cls (type): Configuration class for this module, set to InterAtomDistanceConfig.
        config (InterAtomDistanceConfig): Configuration holding parameters for the distance calculation.
        name (str): Name identifier for this distance instance, used in output references.

    Args:
        config (InterAtomDistanceConfig): Configuration specifying how the distance should be calculated.
        name (str): Name of this instance, used as a key in the output batch dictionary.

    Methods:
        forward(batch: Dict) -> Dict:
            Computes the inter-atom distance loss for a batch, updating it with calculated values.
    """

    config_cls = InterAtomDistanceConfig

    def __init__(self, config: InterAtomDistanceConfig, name: str):
        """
        Initialize the InterAtomDistance module with specified configuration.

        Args:
            config (InterAtomDistanceConfig): Configuration for distance loss computation.
            name (str): Name used for identifying this instance in output batches.
        """
        super(InterAtomDistance, self).__init__()
        self.config = config
        self.name = name

    def forward(self, batch: Dict) -> Dict:
        """
        Compute the Inter-Atom Distance loss for a batch, updating the batch dictionary.

        Args:
            batch (Dict): Dictionary containing input data, including predicted and target coordinates, and optional masks.

        Returns:
            Dict: Updated batch dictionary with the calculated inter-atom distance loss.
        """
        # Retrieve batch size (B), sequence length (L), and channel size (C) from the predictions
        B, L, C = batch[self.config.prediction_name].shape

        # Extract predicted (P) and target (Q) coordinates
        P = batch[self.config.prediction_name]
        Q = batch[self.config.target_name]

        # Determine or default the mask for valid atoms
        if self.config.mask_name is not None:
            mask_remove = batch[self.config.mask_name]  # Mask to exclude certain atoms
        else:
            mask_remove = P.new_ones(B, L, dtype=torch.bool)  # Default mask to include all

        # Determine residue indices if provided
        if self.config.res_idx_name is not None:
            idx = batch[self.config.res_idx_name]  # Retrieve residue indices to manage interactions
        else:
            idx = P.new_zeros(B, L, dtype=torch.long)  # Default to zero indices

        # Initialize variables for loss computation
        loss = P.new_zeros(B)
        n = P.new_zeros(B)  # Counter for valid interactions per batch

        # Calculate the inter-atom distance differences and store in loss
        for b in range(B):
            # Apply the mask to include only specified interactions
            idx_b = idx[b][mask_remove[b]]
            mask_b = torch.tril((idx_b[:, None] - idx_b[None, :]) == 0, diagonal=-1)  # Mask for unique interactions

            # Compute target and predicted inter-atomic distances for masked atoms
            q_b = Q[b][mask_remove[b]]
            p_b = P[b][mask_remove[b]]

            # Compute norm differences for distances
            q_b = torch.linalg.vector_norm((q_b[:, None] - q_b[None, :])[mask_b], dim=-1)
            q_b = q_b - torch.linalg.vector_norm((p_b[:, None] - p_b[None, :])[mask_b], dim=-1)

            # Compute loss per batch entry
            loss[b] = torch.sum((q_b**2))
            n[b] = mask_b.sum()  # Number of valid interactions

        # Normalize the loss by the number of interactions and handle numerical stability
        loss = loss / (n + 1e-6)

        # Optionally take the square root of the computed loss
        if self.config.root:
            loss = torch.sqrt(loss + 1e-6)  # Adding small constant for numerical stability

        # Store the computed loss in the batch under the given name
        batch["losses"][self.name] = loss
        return batch


if __name__ == "__main__":
    # Determine the computation device (use GPU if available, otherwise default to CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define configuration for the Inter-Atom Distance loss calculation
    config = InterAtomDistanceConfig(
        prediction_name="predictions",  # Key for accessing predicted coordinates in the batch
        target_name="targets",  # Key for accessing target coordinates in the batch
        mask_name="mask",  # Key for accessing an optional mask
        res_idx_name="residue_indices",  # Key for accessing residue indices
    )

    # Instantiate the InterAtomDistance module with the configured parameters
    distance_metric = InterAtomDistance(config, name="inter_atom_distance").to(device)

    # Generate a sample batch with random predictions, targets, mask, and residue indices
    batch = {
        "predictions": torch.randn(4, 10, 3).to(device),  # Random 3D coordinates for predictions
        "targets": torch.randn(4, 10, 3).to(device),  # Random 3D coordinates for targets
        "mask": torch.randint(0, 2, (4, 10)).bool().to(device),  # Random binary mask indicating valid atoms
        "residue_indices": torch.arange(10).repeat(4, 1).to(device),  # Simulated residue indices
    }

    # Compute the Inter-Atom Distance loss using the forward method
    batch["losses"] = {}
    batch = distance_metric(batch)

    # Print the updated batch to observe the computed inter-atom distance loss
    print("Batch with computed Inter-Atom Distance loss:")
    print(batch)

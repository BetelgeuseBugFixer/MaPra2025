from dataclasses import dataclass
from typing import Optional, Dict, Tuple
import torch


@dataclass
class RegistrationConfig:
    """
    Configuration parameters for a point cloud registration task.

    This dataclass specifies the necessary details for aligning two point clouds, including
    the identifiers for the point clouds and an optional mask to select specific points
    for computing the optimal transformation.

    Attributes:
        pc_1 (str): The identifier for the point cloud that will be transformed.
        pc_2 (str): The identifier for the reference point cloud.
        mask (Optional[str]): An optional identifier for a mask that designates which points
                              to consider during the transformation process.
    """

    pc_1: str
    pc_2: str
    mask: Optional[str] = None


class Registration:
    """
    A module for performing point cloud registration based on specified configurations.

    The Registration class facilitates the alignment of pairs of point clouds using a set of configurations
    that specify which point clouds to register and how. This is particularly useful in tasks requiring
    spatial alignment or comparison of 3D data.
    The registration is performed using the Kabsch algorithm.

    Attributes:
        config (Dict[str, RegistrationConfig]): A dictionary mapping the name of each registration to its
                                                respective configuration object. It dictates how point clouds
                                                are processed and aligned.

    Methods:
        __call__(batch: Dict) -> Dict:
            Performs registration on the input batch of point clouds according to the configurations.
        get_transform(P: torch.Tensor, Q: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
            Get the transform matrix to transform P to Q.
        apply_transform(coords: torch.Tensor, rot: torch.Tensor, tran: torch.Tensor) -> torch.Tensor:
            Apply the transform matrix to the coordinates.
        transform(P: torch.Tensor, Q: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            Apply the Kabsch algorithm to transform P to Q.
        _centroid_adjust(X: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
            Center the coordinates X, using only the masked positions.

    Args:
        config (Optional[Dict]): A dictionary specifying the configuration for one or more registration
                                 tasks. Each key is a unique registration identifier, and each value is a
                                 dictionary containing the configuration parameters.

    Expected Config Format:
        config = {
            "registration_1": {
                "pc_1": "pc_1",
                "pc_2": "pc_2",
                "mask": "mask1",
            },
        }
        - Each key is a unique name for the registration task.
        - Each value contains settings for that task, such as the point clouds to align ('pc_1', 'pc_2') and optional mask.

    Notes:
        - The name of each registration serves as a key in the resulting batch to store the registered point cloud.
        - Ensure that registration names do not conflict across different entries to avoid overwriting.
        - The input to the call method must contain the point clouds and any specified masks.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the Registration module with optional configuration.

        Args:
            config (Optional[Dict]): A dictionary of registration configurations. See class docstring for format details.
        """
        super(Registration, self).__init__()

        # Initialize configuration, converting each entry to a RegistrationConfig object.
        self.config = (
            {name: RegistrationConfig(**config_tmp) for name, config_tmp in config.items()} if config is not None else {}
        )

    def __call__(self, batch: Dict) -> Dict:
        """
        Apply the Registration module to the input batch, according to the configurations.

        Args:
            batch (Dict): A dictionary of point clouds, represented as torch tensors, including any required masks.

        Returns:
            Dict: The updated batch dictionary containing the registered point clouds stored under keys specified
                  by their registration names.
        """
        # Disable autocast for registration operations
        with torch.amp.autocast("cuda", enabled=False):
            for name, config in self.config.items():
                # Register the point clouds according to the config, updating the batch.
                batch = self.registration(batch, name, config)
        return batch

    def registration(self, batch: Dict, name: str, config: RegistrationConfig) -> Dict:
        """
        Perform point cloud registration according to the specified configuration.

        This method aligns a point cloud (pc_1) with a reference point cloud (pc_2) using an optional mask,
        then updates the batch with the registered point cloud associated with the given registration name.

        Args:
            batch (Dict): A dictionary containing point clouds and optional masks as torch tensors.
                        The keys are typically strings that identify each data component.
            name (str): The name under which the registered point cloud will be stored in the batch dictionary.
            config (RegistrationConfig): A configuration object detailing the names of the point clouds and
                                        mask to be used for registration.

        Returns:
            Dict: The updated batch dictionary with an additional entry for the registered point cloud, accessible via `name`.

        Notes:
            - The method extracts the necessary point clouds and mask from the batch based on the configuration,
            performs alignment, and then integrates the result back into the batch.
            - Ensure that the `config` provided matches the keys in the `batch` for successful registration.
        """
        # Extract the point clouds to be registered and an optional mask from the batch.
        pc_1, pc_2 = batch[config.pc_1], batch[config.pc_2]
        mask = batch[config.mask] if config.mask is not None else None

        # Ensure inputs are float32
        pc_1 = pc_1.to(dtype=torch.float32)
        pc_2 = pc_2.to(dtype=torch.float32)
        mask = mask.to(dtype=torch.float32)

        # Perform the point cloud registration transformation.
        pc_1_registered, _, _ = self.transform(pc_1, pc_2, mask)

        # Add the registered point cloud to the batch with the specified name.
        batch[name] = pc_1_registered

        # Return the updated batch with the newly registered point cloud.
        return batch

    def get_transform(self, P: torch.Tensor, Q: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Compute the transformation matrix needed to align point cloud P with point cloud Q.

        This method calculates the optimal rigid transformation, consisting of a rotation matrix and a translation vector,
        to align two point clouds, P and Q. Optionally, a mask can be applied to focus the transformation on specific
        portions of the point clouds.

        Args:
            P (torch.Tensor): A tensor of shape [B, N, 3] representing the coordinates of the first point cloud.
            Q (torch.Tensor): A tensor of shape [B, N, 3] representing the coordinates of the second point cloud (reference).
            mask (Optional[torch.Tensor]): A binary tensor of shape [B, N] indicating which points should be used for aligning
                                            the point clouds. If None, all points are used.

        Returns:
            tuple: A tuple containing:
                - rot (torch.Tensor): A tensor of shape [B, 3, 3] representing the rotation matrices for each batch.
                - tran (torch.Tensor): A tensor of shape [B, 3] representing the translation vectors for each batch.

        Raises:
            ValueError: If any negative singular values are encountered during the SVD computation, implying numerical issues.

        Notes:
            - The function uses the Kabsch algorithm to compute the optimal rotation by solving the Procrustes problem.
            - Centering adjusts both point clouds to their centroids, subtracting this from the coordinates for accurate alignment.
            - The alignment ensures the minimal root-mean-square deviation (RMSD) between the transformed and reference points.
        """
        # Center the coordinates of both point clouds based on mask or all points.
        P_ctd, P_adj = self._centroid_adjust(P, mask)
        Q_ctd, Q_adj = self._centroid_adjust(Q, mask)

        # Compute the covariance matrix for transformation.
        h = torch.bmm(P_adj.permute(0, 2, 1), Q_adj)

        # Perform Singular Value Decomposition (SVD) on the covariance matrix.
        u, singular_values, vt = torch.linalg.svd(h)
        if (singular_values < 0).any():
            raise ValueError("Singular values are negative")

        # Compute the optimal rotation matrix using SVD results.
        v = vt.permute(0, 2, 1)
        d = torch.det(v @ u.permute(0, 2, 1))
        e = (
            torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32)
            .unsqueeze(0)
            .expand(P.shape[0], 3, 3)
            .to(P.device)
        )
        e = e.clone()
        e[:, 2, 2] = d
        rot = torch.matmul(torch.matmul(v, e), u.permute(0, 2, 1))

        # Compute the translation vector to align centroids of P and Q.
        tran = Q_ctd - torch.matmul(rot, P_ctd.transpose(1, 2)).transpose(1, 2)

        return rot, tran

    def apply_transform(self, coords: torch.Tensor, rot: torch.Tensor, tran: torch.Tensor) -> torch.Tensor:
        """
        Apply a transformation matrix consisting of rotation and translation to a set of coordinates.

        This method utilizes a rotation matrix and a translation vector to transform 3D coordinates,
        typically used to align or reposition point clouds in 3D space for applications such as
        registration, simulation, or graphical transformations.

        Args:
            coords (torch.Tensor): A tensor of shape [B, N, 3] representing a batch of N 3D coordinates for B samples.
            rot (torch.Tensor): A tensor of shape [B, 3, 3] containing rotation matrices for each sample in the batch.
            tran (torch.Tensor): A tensor of shape [B, 3] specifying translation vectors to be applied to each sample.

        Returns:
            torch.Tensor: A tensor of shape [B, N, 3] with the transformed coordinates after applying the rotation and translation.

        Notes:
            - The operation applies the rotation by matrix multiplication, followed by the addition of the translation vector.
            - The transformation is applied batch-wise, modifying each batch sample independently.
        """
        # Apply the rotation matrix to the coordinates, ensuring correct dimensions through permutation.
        coords = torch.matmul(rot, coords.permute(0, 2, 1)).permute(0, 2, 1)

        # Apply the translation vector to the rotated coordinates.
        coords = coords + tran

        # Return the transformed coordinates.
        return coords

    def transform(
        self, P: torch.Tensor, Q: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Transform point cloud P to align with point cloud Q using the Kabsch algorithm.

        This method computes the optimal rigid transformation (rotation and translation) that minimizes
        the root mean square deviation (RMSD) between corresponding points in two point clouds, P and Q.
        It optionally utilizes a mask to focus on specific points during the transformation process.

        Args:
            P (torch.Tensor): A tensor of shape [B, N, 3] representing the coordinates of the point cloud to be transformed.
            Q (torch.Tensor): A tensor of shape [B, N, 3] representing the coordinates of the reference point cloud.
            mask (Optional[torch.Tensor]): A binary mask tensor of shape [B, N] indicating which points to consider
                                            in the algorithm. If None, all points are considered.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - P_new: A tensor of shape [B, N, 3] representing the transformed coordinates of P.
                - rot: A tensor of shape [B, 3, 3] representing the rotation matrices applied to each sample.
                - tran: A tensor of shape [B, 3] representing the translation vectors applied to each sample.

        Notes:
            - The transformation aligns P with Q by calculating a least-squares optimal rotation and translation.
            - The Kabsch algorithm is particularly suitable for aligning 3D structures by minimizing the RMSD.
            - The `mask` allows selective consideration of points for alignment, which is useful for focusing on key features.
        """
        # Compute the optimal transformation (rotation and translation) using the Kabsch algorithm.
        rot, tran = self.get_transform(P, Q, mask)

        # Apply the computed transformation to the point cloud P.
        P_new = self.apply_transform(P, rot, tran)

        # Return the transformed point cloud and the transformation parameters.
        return P_new, rot, tran

    def _centroid_adjust(self, X: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Center the 3D coordinates in X by adjusting to their centroid, optionally using a mask to select specific points.

        This method computes the centroid of a set of 3D coordinates and adjusts the coordinates so that the centroid
        aligns with the origin. If a mask is provided, only the masked positions are used to compute the centroid,
        focusing the adjustment on particular points of interest within the data.

        Args:
            X (torch.Tensor): A tensor of shape [B, N, 3] representing a batch of N 3D coordinates for B samples.
            mask (Optional[torch.Tensor]): A binary mask tensor of shape [B, N] indicating which points should be
                                            considered when computing the centroid. If None, all points are included.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - X_ctd: A tensor of shape [B, 3] representing the centroid of the selected points for each batch sample.
                - X_adj: A tensor of shape [B, N, 3] with coordinates adjusted so that the centroid is at the origin.

        Notes:
            - Centroid adjustment is useful for normalizing coordinates to remove translational effects,
            providing a consistent frame of reference for further analysis.
            - When a mask is applied, it isolates specific elements for centroid calculations, potentially focusing
            on subsets of points such as critical regions or features.
        """
        if mask is None:
            # Compute the centroid using all coordinates and adjust to center.
            X_ctd = torch.mean(X, dim=1).unsqueeze(1)
            X_adj = X - X_ctd
        else:
            # Compute the centroid using only masked coordinates and adjust accordingly.
            X_ctd = (torch.sum(X * mask.unsqueeze(-1), dim=1) / torch.sum(mask, dim=1, keepdim=True)).unsqueeze(1)
            X_adj = (X - X_ctd) * mask.unsqueeze(-1)

        # Return the computed centroid and the adjusted coordinates.
        return X_ctd, X_adj


if __name__ == "__main__":
    # Define configuration for multiple point cloud registrations, each with specified point clouds and masks.
    config = {
        "registration_1": {
            "pc_1": "pc_1",
            "pc_2": "pc_2",
            "mask": "mask1",
        },
        "registration_2": {
            "pc_1": "pc_1",
            "pc_2": "pc_2",
            "mask": "mask2",
        },
    }

    # Initialize the Registration module with the provided configuration.
    registrations = Registration(config)

    # Prepare a batch of synthetic point clouds and masks for testing.
    batch = {
        "pc_1": torch.randn(1, 10, 3),  # Randomly generated point cloud 1.
        "pc_2": torch.randn(1, 10, 3),  # Randomly generated point cloud 2.
        "mask1": torch.ones(1, 10),  # Mask for the first registration task, all points included.
        "mask2": torch.ones(1, 10),  # Mask for the second registration task, all points included.
    }

    # Perform the registration on the batch, updating with registered point clouds.
    batch = registrations(batch)

    # Print the updated batch to observe the registered point clouds and any other modifications.
    print(batch)

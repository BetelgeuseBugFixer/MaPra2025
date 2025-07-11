import torch
from torch.utils.data import Dataset
from typing import Optional, Literal
from dataclasses import dataclass
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation
import time
import os

from models.bio2token.data.utils.tokens import (
    BB_CLASS,
    C_REF_CLASS,
    SC_CLASS,
)
from models.bio2token.data.utils.utils import filter_on_length, compute_masks


@dataclass
class ProtDataConfig:
    dataset: Optional[Literal["casp", "alphafoldDB", "cath", "rna3db"]] = None
    train_split: Optional[str] = None
    val_split: Optional[str] = None
    test_split: Optional[str] = None
    # Everything below is done for the entire dataset before sampling.
    max_data: Optional[int] = None
    max_length: int = 2000
    num_residues_max: Optional[int] = None
    num_residues_min: Optional[int] = None
    num_atoms_max: Optional[int] = None
    num_atoms_min: Optional[int] = None
    nan_handling: Literal["remove", "zero"] = "remove"
    recenter: bool = True
    # Everything below is done for every new sample.
    randomly_rotate: bool = True


class ProtDataset(Dataset):
    config_cls = ProtDataConfig

    def __init__(
        self,
        config: config_cls,
        split: str,
    ):
        """
        Initialize the ProtDataset.

        Args:
            config (ProtDataConfig): Configuration object containing dataset parameters.
            split (str): The dataset split to use.

        This constructor sets up the dataset by loading the configuration, processing it,
        and preparing the data stream for the specified split.
        """
        # Load config and check
        self.config: ProtDataConfig = config
        self.split = split
        if self.config.num_atoms_max is None and self.config.max_length is not None:
            self.config.num_atoms_max = self.config.max_length

        # Preprocess data
        self.data = self._get_data()

        # Print some statistics
        print(f"Number of samples in the dataset: {len(self.data)}")
        print(f"Max number of atoms in the dataset: {self.data['natoms'].max()}")
        print(f"Min number of atoms in the dataset: {self.data['natoms'].min()}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return self.process_row(row, idx)

    def _get_data(self):
        """
        Retrieve and preprocess data streams based on the specified dataset configuration.

        This method selects a dataset and its associated data path based on the class
        configuration and prepares a stream for processing. The method asserts that the dataset
        is one of the supported options: "casp", "alphafoldDB", or "cath". Given a specific dataset,
        the method identifies the correct path and column format required for ingesting the data
        stream, which will be passed to a preprocessing function for further processing.

        Returns:
            stream: A data stream object, processed and prepared as specified by the dataset configuration.
                    The stream object is ready for use in subsequent data processing tasks.
        """
        # Set data path and columns based on the dataset configuration.
        if self.config.dataset == "casp":
            data_path = f"data/casp/{self.split}"
        elif self.config.dataset == "alphafoldDB":
            data_path = f"data/alphafold_db/processed"
        elif self.config.dataset == "cath":
            data_path = f"data/cath/cath_v4_3_0/cath_40/processed/{self.split}"
        elif self.config.dataset == "rna3db":
            data_path = f"data/rna3db/{self.split}"
        else:
            raise ValueError(f"Invalid dataset: {self.config.dataset}.")
        columns = ["atom", "residue_ids", "token_class", "unknown_structure", "structure"]

        data = []
        n_data = 0
        for parquet_name in [file for file in os.listdir(data_path) if file.endswith(".parquet")]:
            block = pd.read_parquet(data_path + f"/{parquet_name}")[columns]
            data.append(self._process_block(block))
            n_data += len(block)
            if self.config.max_data is not None and n_data >= self.config.max_data:
                break
        return pd.concat(data, ignore_index=True).reset_index(drop=True)

    def _process_block(self, block: pd.DataFrame):
        """
        Process a batch of protein data to prepare it for downstream computational analysis.

        This private method takes a dataframe block containing protein sequence data and performs
        multiple preprocessing steps such as filtering by sequence length, reformatting data, computing
        structural representations, handling missing data, and optionally recentering coordinates.

        Args:
            block (pd.DataFrame): A pandas DataFrame containing raw data for protein sequences, where
                                each row represents a sequence with associated metadata and coordinates.

        Returns:
            pd.DataFrame: A processed pandas DataFrame containing columns for structural data, sequences,
                        atoms, residues, token classes, and any missing data handling results.
        """

        # 1. Filter dataframe to only include rows with num_residues_min <= AAs <= num_residues_max
        block["n_aa"] = block["residue_ids"].apply(max) + 1
        block = filter_on_length(
            block,
            length_key="n_aa",
            max_length=self.config.num_residues_max,
            min_length=self.config.num_residues_min,
        )
        # 2. Optionally filter on length.
        block["structure"] = block["structure"].apply(lambda x: np.array(x, dtype=np.float32).reshape(-1, 3))
        block["natoms"] = block["structure"].apply(len)
        block = filter_on_length(
            block,
            length_key="natoms",
            max_length=self.config.num_atoms_max,
            min_length=self.config.num_atoms_min,
        )
        # 3. Filter dataframe to remove data having either full coordinate backbone, sidechain, or C_ref nan
        block = self.remove_nan_atoms(block)
        # 4. Convert to list and correct data type (ADD THAT TO PREPROCESSING)
        block["residue_ids"] = block["residue_ids"].apply(lambda x: x.astype(int)).apply(lambda x: x.tolist())
        block["token_class"] = block["token_class"].apply(lambda x: x.astype(int)).apply(lambda x: x.tolist())
        block["unknown_structure"] = block["unknown_structure"].apply(lambda x: x.astype(bool)).apply(lambda x: x.tolist())
        return block

    def remove_nan_atoms(self, dataframe: pd.DataFrame):
        """
        Filter out entries in the dataframe with missing values for critical structural tokens.

        This method processes a dataframe of protein data, removing rows where all atoms of specific token classes
        (backbone, C-alpha reference, or sidechain) are missing (NaN). This ensures that the dataset used for analysis
        is complete in terms of essential structural components, such as backbone accuracy and sidechain presence.

        Args:
            dataframe (pd.DataFrame): A pandas DataFrame containing protein sequence data and associated attributes,
                                    including token classes and a marker for missing structures.

        Returns:
            pd.DataFrame: A filtered DataFrame with rows removed where any structural token class is entirely missing.

        Notes:
            - This filtering focuses on essential structural components represented by the token classes:
                - `BB_CLASS`: Backbone atoms
                - `C_REF_CLASS`: C-alpha reference
                - `SC_CLASS`: Sidechain atoms
            - The filtering ensures that any row preserved in the DataFrame has some valid coordinates for each token class.
        """
        # Filter out rows where all backbone atoms are missing.
        dataframe = dataframe[
            dataframe.apply(
                lambda x: np.sum((x["token_class"] == BB_CLASS) * x["unknown_structure"])
                != np.sum(x["token_class"] == BB_CLASS),
                axis="columns",
            )
        ]

        # Filter out rows where all C-alpha reference atoms are missing.
        dataframe = dataframe[
            dataframe.apply(
                lambda x: np.sum((x["token_class"] == C_REF_CLASS) * x["unknown_structure"])
                != np.sum(x["token_class"] == C_REF_CLASS),
                axis="columns",
            )
        ]

        # Filter out rows where all sidechain atoms are missing.
        dataframe = dataframe[
            dataframe.apply(
                lambda x: np.sum((x["token_class"] == SC_CLASS) * x["unknown_structure"])
                != np.sum(x["token_class"] == SC_CLASS),
                axis="columns",
            )
        ]

        # Return the filtered DataFrame.
        return dataframe

    def missing_structure_handling(self, structure, residue_ids, token_class, unknown_structure):
        """
        Handle missing coordinates within a protein's structural data.

        This method processes and addresses missing atomic coordinates (NaN values) in the structural data of proteins.
        It provides options for dealing with incomplete data by either removing affected residues or zero-filling missing values,
        based on the configuration settings.

        Args:
            structure (torch.Tensor): An array representing the 3D coordinates of atoms for the protein structure.
            residue_ids (torch.Tensor): An array of integers representing unique identifiers for residues.
            token_class (torch.Tensor): An array of integers representing the classification of tokens (atoms) within the sequence.
            unknown_structure (torch.Tensor): A boolean array indicating which atoms had missing coordinates in the original structure.

        Returns:
            tuple: A tuple containing:
                - torch.Tensor: The processed structure array with missing data handled as per configuration.
                - torch.Tensor: The processed residue IDs, aligned with the updated structure.
                - torch.Tensor: The processed token classification array, aligned with the updated structure.
                - torch.Tensor: A boolean array indicating which atoms had missing coordinates in the original structure.

        Raises:
            ValueError: If the provided `nan_handling` configuration is invalid, i.e., not "remove" or "zero".

        Notes:
            - The method identifies missing values in the structure by checking for NaNs and then applies the chosen handling strategy.
            - When `nan_handling` is "remove", entire residues containing any missing atoms are removed.
            - When `nan_handling` is "zero", missing values are replaced with zeroes.
        """
        # Identify which structural coordinates are missing (NaN).
        if self.config.nan_handling == "remove":
            # Remove residues with any missing atoms by marking affected residue IDs for removal.
            unique_removed_ids = torch.unique(residue_ids[unknown_structure])
            mask_remove = torch.isin(residue_ids, unique_removed_ids)
            # Apply the removal mask to all relevant arrays.
            structure = structure[~mask_remove]
            residue_ids = residue_ids[~mask_remove]
            token_class = token_class[~mask_remove]
            unknown_structure = unknown_structure[~mask_remove]
        elif self.config.nan_handling == "zero":
            # Zero-fill missing coordinates in the structure array.
            structure[unknown_structure] = 0
        else:
            # Raise an error if the configuration for nan handling is invalid.
            raise ValueError(f"Invalid nan_handling: {self.config.nan_handling}.")

        # Return the updated data with missing values handled.
        return structure, residue_ids, token_class, unknown_structure

    def process_row(self, row, idx) -> dict:
        """
        Process a single row of protein data, transforming it into a structured sample with optional random rotations and padding.

        This method takes an individual data row containing protein attributes and processes it into a dictionary format.
        It supports random rotations for augmentation, applies padding to standardize the input size,
        and generates various masks to indicate structural details for each token in the sequence.

        Args:
            row (dict): A dictionary representing a single row of data, including information such as
                        structure coordinates, atoms, residues, residue IDs, token classes, and unknown structure indicators.
            idx (int): An index representing the position of the row within a larger dataset,
                        useful for tracking during batch processing or error reporting.

        Returns:
            dict: A processed sample dictionary containing several tensor attributes:
                - "atom": Indices of atoms within the protein sequence.
                - "structure": 3D coordinates of the atoms after optional rotation.
                - "residue": Indices of residues within the sequence.
                - "residue_ids": Unique identifiers for residues within the sequence.
                - "token_class": Token class identifiers indicating atom roles and functions.
                - "unknown_structure": Boolean indicators of missing structural data.
                - "bb_atom_mask", "sc_atom_mask", "all_atom_mask", "cref_mask": Specific masks for different structural analysis tasks.

        Notes:
            - Random rotation is applied if configured, using a randomly generated rotation matrix.
            - Padding is applied to ensure consistency in dimensions across all input samples.
            - Several masks are generated, providing detailed structural insights and facilitating various analytical applications.
        """
        # Convert row data to torch tensors with specified data types.
        structure = np.vstack(row["structure"]).copy()  # L x 3
        residue_ids = row["residue_ids"].copy()
        token_class = row["token_class"].copy()
        unknown_structure = row["unknown_structure"].copy()

        structure = torch.Tensor(structure)  # L x 3
        residue_ids = torch.Tensor(residue_ids).to(torch.long)  # L
        token_class = torch.Tensor(token_class).to(torch.long)  # L
        unknown_structure = torch.Tensor(unknown_structure).to(torch.bool)  # L

        # Handle missing coordinates
        structure, residue_ids, token_class, unknown_structure = self.missing_structure_handling(
            structure, residue_ids, token_class, unknown_structure
        )
        if len(structure) == 0:
            print(row)

        # Randomly apply rotation to the atom coordinates if configured.
        if self.config.randomly_rotate:
            rot = torch.Tensor(Rotation.random().as_matrix())
            structure[~unknown_structure] = torch.einsum("ij, li->lj", rot, structure[~unknown_structure])

        # Create a sample dictionary with initial data.
        sample = {
            "structure": structure,
            "residue_ids": residue_ids,
            "token_class": token_class,
            "unknown_structure": unknown_structure,
        }
        # Assign generated masks to the sample dictionary.
        sample = compute_masks(sample, structure_track=True)
        return sample

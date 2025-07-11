import torch
from torch.utils.data import Dataset
from typing import Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation
import os

from models.bio2token.data.utils.tokens import INDEXED_ATOMS, BB_CLASS
from models.bio2token.data.utils.utils import filter_on_length, compute_masks


@dataclass(kw_only=True)
class NablaDFTDataConfig:
    dataset: str = "data/nabla_dft"
    train_split: Optional[str] = None
    test_split: Optional[str] = None
    # Everything below is done for the entire dataset before sampling.
    max_data: Optional[int] = None
    heavy_atoms_only: bool = True
    max_length: int = 32
    num_atoms_max: Optional[int] = None
    num_atoms_min: Optional[int] = None
    recenter: bool = True
    # Everything below is done for every new sample.
    randomly_rotate: bool = True


class NablaDFTDataset(Dataset):
    config_cls = NablaDFTDataConfig

    def __init__(
        self,
        config: config_cls,
        split: str,
    ):
        # Load config and check
        self.config: NablaDFTDataConfig = config
        self.split = split

        # Preprocess data
        self.data = self._get_data()
        print(f"Number of samples in the dataset: {len(self.data)}")
        print(f"Max number of atoms in the dataset: {self.data['natoms'].max()}")
        print(f"Min number of atoms in the dataset: {self.data['natoms'].min()}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return self.process_row(row, idx)

    def _get_data(self):
        data_path = self.config.dataset + f"/{self.split}"
        data = []
        n_data = 0
        for parquet_name in [file for file in os.listdir(data_path) if file.endswith(".parquet")]:
            block = pd.read_parquet(data_path + f"/{parquet_name}")
            block = block[["structure", "atom_names_short"]]
            block = block.rename(columns={"atom_names_short": "atom"})
            data.append(self._process_block(block))
            n_data += len(block)
            if self.config.max_data is not None and n_data >= self.config.max_data:
                break
        return pd.concat(data, ignore_index=True).reset_index(drop=True)

    def _process_block(self, block: pd.DataFrame):
        # 1. Rename coords to structure and atoms to atom for consistency across datasets.
        # 2. Optionally remove hydrogens.
        block["structure"] = block["structure"].apply(lambda x: x.reshape(-1, 3))
        if self.config.heavy_atoms_only:
            block[["atom", "structure"]] = block.apply(
                lambda x: self.remove_H(x.atom, x.structure), axis=1, result_type="expand"
            )
        # 3. Optionally filter on length.
        block["natoms"] = block["structure"].apply(len)
        block = filter_on_length(block, max_length=self.config.num_atoms_max, min_length=self.config.num_atoms_min)
        # 4. Optionally recenter the coordinates.
        if self.config.recenter:
            block["structure"] = block.apply(lambda x: self.recenter(x.structure), axis=1)
        return block

    def remove_H(self, atom: list[str], structure: list[list[float]]) -> tuple[list[str], list[list[float]]]:
        """
        Remove hydrogen atoms from the list of atoms and coordinates.
        args:
            atom: list[str]
            structure: list[list[float]]
        returns:
            tuple[list[str], Union[list[list[float]], None]]
        """
        mask_hydrogen = np.array([a == "H" for a in atom])
        atom = atom[~mask_hydrogen]
        structure = structure[~mask_hydrogen]
        return atom, structure

    def convert_atoms_to_indices(self, atoms: list[str]) -> list[int]:
        """
        Convert a list of atoms to a list of indices.
        args:
            atoms: list[str]
        returns:
            list[int]
        """
        return [INDEXED_ATOMS[atom] for atom in atoms]

    def recenter(self, structure: list[list[float]]) -> list[list[float]]:
        """
        Recenter the coordinates. WARNING: When generating random mask, you should recenter the input coordinates of the model.
        args:
            structure: list[list[float]]
        returns:
            list[list[float]]
        """
        structure = np.stack(structure)
        barycenter = np.mean(structure, axis=0)
        return structure - barycenter

    def process_row(self, row, idx) -> dict:
        # Known information
        structure = torch.Tensor(row["structure"])  # L x 3
        # All structure are known
        unknown_structure = torch.zeros(structure.shape[0], dtype=torch.bool)
        # Everything is backbone for small molecules
        token_class = torch.ones(structure.shape[0], dtype=torch.long) * BB_CLASS
        # Add the missing residue track
        residue_ids = torch.zeros(token_class.shape, dtype=torch.long)

        # Randomly rotate the atoms
        if self.config.randomly_rotate:
            rot = torch.Tensor(Rotation.random().as_matrix())
            structure[~unknown_structure] = torch.einsum("ij, li->lj", rot, structure[~unknown_structure])

        # sample
        sample = {
            "structure": structure,
            "residue_ids": residue_ids,
            "token_class": token_class,
            "unknown_structure": unknown_structure,
        }

        # Randomly create masks
        sample = compute_masks(sample, structure_track=True)

        # Return the processed sample.
        return sample

import lightning as L
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from typing import Union, List
from torch.utils.data import Subset, ConcatDataset

from models.bio2token.data.nabladft import NablaDFTDataset, NablaDFTDataConfig
from models.bio2token.data.prot_dataset import ProtDataConfig, ProtDataset


@dataclass(kw_only=True)
class DatasetZoo:
    """
    List of the dataset configs.
    """

    cath: ProtDataConfig = field(default_factory=ProtDataConfig)
    alphafoldDB: ProtDataConfig = field(default_factory=ProtDataConfig)
    casp14: ProtDataConfig = field(default_factory=ProtDataConfig)
    casp15: ProtDataConfig = field(default_factory=ProtDataConfig)
    rna3db: ProtDataConfig = field(default_factory=ProtDataConfig)
    nabladft: NablaDFTDataConfig = field(default_factory=NablaDFTDataConfig)


@dataclass(kw_only=True)
class DatasetConfig:
    """
    Configuration for the concatenation of dataset.
    """

    batch_size_per_gpu: int = 16  # Batch size for training or testing
    batch_size_per_gpu_val: int = 16  # Batch size for validation
    num_workers: int = 4
    ds_name: Union[str, List[str]] = "cath"  # Names of the datasets to concatenate
    ds_name_val: Union[str, List[str]] = "cath"  # Names of the datasets to concatenate for validation
    dataset: DatasetZoo = field(default=DatasetZoo)  # List of dataset configs.
    is_train: bool = True  # Whether we are one training or testing mode.


class DatasetModule(L.LightningDataModule):
    def __init__(self, config: DatasetConfig):
        super().__init__()
        # Track method calls for different processing stages.
        # Otherwise, dataset construction can done multiple times.
        self._already_called = {}
        for stage in ("fit", "validate", "test", "predict"):
            self._already_called[stage] = False
        self.config = config
        self.datasets = {"train": None, "val": None, "test": None}
        self.collate_fn = None  # make get overwritten

    def prepare_data(self):
        pass

    def set_collate_fn(self, collate_fn: callable):
        """The LightningDataModule class handles creating a dataloader, so we do
        not directly create a dataloader.  This function allows us to pass
        a collate function ot the DatasetModule, and it will be used when
        the dataloaders are created.

        :param collate_fn callable: A collate function that takes in
            a batch (dict[str, Tensor]) and returns a batch (dict[str, Tensor])

        :return: None
        """

        self.collate_fn = collate_fn

    def setup(self, stage: str = None):
        """
        Setup the datasets for the given stage. Load the datasets, split them into train, val, test, and
        then concatenate them into a single train, val, test dataset.
        """
        # If stage is already processed, return.
        if stage and self._already_called[stage]:
            return
        # Initialize lists for different datasets.
        train_datasets = []
        val_datasets = []
        test_datasets = []
        print(self.config.ds_name)
        # If ds_name is a string, convert it to a list.
        if isinstance(self.config.ds_name, str):
            self.config.ds_name = [self.config.ds_name]
        # Add the cath dataset.
        if "cath" in self.config.ds_name:
            self.config.dataset.cath.dataset = "cath"
            if self.config.is_train:
                print("CATH train")
                train_dataset = ProtDataset(self.config.dataset.cath, split=self.config.dataset.cath.train_split)
                train_datasets.append(train_dataset)
                if "cath" in self.config.ds_name_val:
                    print("CATH val")
                    val_dataset = ProtDataset(self.config.dataset.cath, split=self.config.dataset.cath.val_split)
                    val_datasets.append(val_dataset)
            else:
                print("CATH test")
                test_dataset = ProtDataset(self.config.dataset.cath, split=self.config.dataset.cath.test_split)
                test_datasets.append(test_dataset)
        # Add the nabladft dataset.
        # There is no validation set for nabladft. We use the training set, split in two, 90% for training, 10% for validation.
        if "nabladft" in self.config.ds_name:
            if self.config.is_train:
                print("NABLADFT train and val")
                full_dataset = NablaDFTDataset(self.config.dataset.nabladft, split=self.config.dataset.nabladft.train_split)
                indices = list(range(len(full_dataset)))
                train_indices, val_indices = train_test_split(indices, test_size=0.1, random_state=42)
                train_dataset = Subset(full_dataset, train_indices)
                train_datasets.append(train_dataset)
                if "nabladft" in self.config.ds_name_val:
                    val_dataset = Subset(full_dataset, val_indices)
                    val_datasets.append(val_dataset)
            else:
                print("NABLADFT test")
                full_dataset = NablaDFTDataset(self.config.dataset.nabladft, split=self.config.dataset.nabladft.test_split)
                test_datasets.append(full_dataset)
        # Add the alphafoldDB dataset.
        # There is no validation set for alphafoldDB. We use the training set, split in two, 90% for training, 10% for validation.
        # There is not testing set for alphafoldDB.
        if "alphafoldDB" in self.config.ds_name:
            assert self.config.is_train
            self.config.dataset.alphafoldDB.dataset = "alphafoldDB"
            print("AlphafoldDB train and val")
            full_dataset = ProtDataset(self.config.dataset.alphafoldDB, split=None)
            indices = list(range(len(full_dataset)))
            train_indices, val_indices = train_test_split(indices, test_size=0.1, random_state=42)
            train_dataset = Subset(full_dataset, train_indices)
            train_datasets.append(train_dataset)
            if "alphafoldDB" in self.config.ds_name_val:
                val_dataset = Subset(full_dataset, val_indices)
                val_datasets.append(val_dataset)
        # Add the casp14 dataset.
        # There is not training or validation set for casp14, only for testing.
        if "casp14" in self.config.ds_name:
            assert not self.config.is_train
            self.config.dataset.casp14.dataset = "casp"
            print("CASP14 test")
            test_dataset = ProtDataset(self.config.dataset.casp14, split="casp14")
            test_datasets.append(test_dataset)
        # Add the casp15 dataset.
        # There is not training or validation set for casp15, only for testing.
        if "casp15" in self.config.ds_name:
            assert not self.config.is_train
            self.config.dataset.casp15.dataset = "casp"
            print("CASP15 test")
            test_dataset = ProtDataset(self.config.dataset.casp15, split="casp15")
            test_datasets.append(test_dataset)
        # Add the rna3db dataset.
        # There is no validation set for rna3db. We use the training set, split in two, 90% for training, 10% for validation.
        if "rna3db" in self.config.ds_name:
            self.config.dataset.rna3db.dataset = "rna3db"
            if self.config.is_train:
                print("RNA3DB train and val")
                full_dataset = ProtDataset(self.config.dataset.rna3db, split=self.config.dataset.rna3db.train_split)
                indices = list(range(len(full_dataset)))
                train_indices, val_indices = train_test_split(indices, test_size=0.1, random_state=42)
                train_dataset = Subset(full_dataset, train_indices)
                train_datasets.append(train_dataset)
                if "rna3db" in self.config.ds_name_val:
                    val_dataset = Subset(full_dataset, val_indices)
                    val_datasets.append(val_dataset)
            else:
                print("RNA3DB test")
                test_dataset = ProtDataset(self.config.dataset.rna3db, split=self.config.dataset.rna3db.test_split)
                test_datasets.append(test_dataset)

        # If none of the training or testing list have been filled, raise an error.
        if len(train_datasets) == 0 and len(test_datasets) == 0:
            raise NotImplementedError(
                f"config.ds_name options are [nabladft, cath, rna3db, casp14, casp15] but is: {self.config.ds_name}"
            )

        # Concatenate the datasets.
        self.datasets = {
            "train": ConcatDataset(train_datasets) if len(train_datasets) > 0 else None,
            "val": ConcatDataset(val_datasets) if len(val_datasets) > 0 else None,
            "test": ConcatDataset(test_datasets) if len(test_datasets) > 0 else None,
        }
        self._already_called[stage] = True

    def train_dataloader(self):
        """
        Return a DataLoader for the training dataset.
        """
        return DataLoader(
            self.datasets["train"],
            shuffle=True,
            num_workers=self.config.num_workers,
            batch_size=self.config.batch_size_per_gpu,
            pin_memory=True,
            persistent_workers=False,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        """
        Return a DataLoader for the validation dataset.
        """
        return DataLoader(
            self.datasets["val"],
            shuffle=True,
            num_workers=self.config.num_workers,
            batch_size=self.config.batch_size_per_gpu_val,
            pin_memory=True,
            persistent_workers=False,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        """
        Return a DataLoader for the testing dataset.
        """
        return DataLoader(
            self.datasets["test"],
            shuffle=False,
            num_workers=self.config.num_workers,
            batch_size=self.config.batch_size_per_gpu,
            pin_memory=True,
            persistent_workers=False,
            collate_fn=self.collate_fn,
        )

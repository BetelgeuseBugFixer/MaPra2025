from dataclasses import dataclass, field

from omegaconf import OmegaConf
from models.bio2token.data.utils.utils import filter_batch, pad_and_stack_batch


@dataclass(kw_only=True)
class PadAndStackConfig:
    """Configuration for padding and stacking batch sequences.

    Attributes:
        sequences_to_pad: Dictionary mapping sequence names to their padding values.
            Example: {'input_ids': 0, 'attention_mask': 0}
        pad_to_multiple_of: Ensures padded sequences have lengths that are multiples
            of this value. Defaults to 1 (no special multiple requirement).
    """

    sequences_to_pad: dict[str, any] = field(default_factory=dict)
    pad_to_multiple_of: int | list[int] = 1


class PadAndStack:
    """Collate function for padding and stacking batch sequences.

    This class provides functionality to process batches of sequences by padding them
    to the same length and stacking them into tensors. It filters the input batch
    to only include specified sequences and applies padding according to the
    configuration.

    Args:
        config (PadAndStackConfig): Configuration specifying which sequences to pad
            and the padding parameters.
    """

    def __init__(self, config: PadAndStackConfig):
        self.config = config
        if isinstance(config.pad_to_multiple_of, int):
            self.pad_to_multiple_of = config.pad_to_multiple_of
        elif isinstance(OmegaConf.to_object(config.pad_to_multiple_of), list) and all(
            [isinstance(m, int) for m in config.pad_to_multiple_of]
        ):
            m = 1
            for m0 in config.pad_to_multiple_of:
                m *= m0
            self.pad_to_multiple_of = m
        else:
            raise ValueError(f"Expected type int or list[int], found type {type(config.pad_to_multiple_of)}")

    def __call__(self, batch: dict[str, any]) -> dict[str, any]:
        """Process a batch by padding and stacking sequences.

        Args:
            batch (dict[str, any]): Input batch containing sequences to process.

        Returns:
            dict[str, any]: Processed batch with padded and stacked sequences.
        """

        filtered_batch = filter_batch(batch, self.config.sequences_to_pad.keys())

        return pad_and_stack_batch(
            filtered_batch,
            self.config.sequences_to_pad,
            self.pad_to_multiple_of,
        )

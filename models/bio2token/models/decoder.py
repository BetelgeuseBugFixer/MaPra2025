from dataclasses import dataclass, field
from typing import Dict
import torch
import torch.nn as nn

from models.bio2token.layers.mamba import MambaConfig, MambaStack

# Registry for available decoder models and their configurations
DECODER_REGISTRY = {
    "mamba": {"model": MambaStack, "config": MambaConfig},
}


@dataclass
class DecoderZoo:
    """
    Collection of configurations for various decoder models available in the DecoderRegistry.

    Attributes:
        mamba (MambaConfig): Configuration for the MambaStack decoder.
    """

    mamba: MambaConfig = field(default_factory=MambaConfig)


@dataclass
class DecoderConfig:
    """
    Configuration class for setting up a Decoder model.

    Contains specifications for selecting and configuring decoder models from a predefined set.

    Attributes:
        decoder_type (str): Type of decoder to use. Defaults to 'mamba'.
        decoder (DecoderZoo): A dataclass housing configurations for all supported decoder models.
    """

    decoder_type: str = "mamba"
    decoder: DecoderZoo = field(default_factory=DecoderZoo)


class Decoder(nn.Module):
    """
    Decoder module that integrates various decoder models for data reconstruction.

    Depending on the configuration, this module selects and constructs different decoder models
    (e.g., Mamba) to transform encoded features back into their original format.

    Attributes:
        config_cls (type): Configuration class for this decoder, defined as DecoderConfig.
        config (DecoderConfig): Instance of DecoderConfig with parameters for selecting appropriate decoders.
        decoder (nn.Module): Instantiated decoder model based on the configuration.

    Args:
        config (DecoderConfig): Configuration specifying the setup parameters for the decoder model.

    Methods:
        construct_decoder() -> nn.Module:
            Constructs and returns the decoder model based on the specified decoder type in the configuration.

        forward(batch: Dict) -> Dict:
            Processes the encoded data through the decoder and returns the reconstructed output.
    """

    config_cls = DecoderConfig

    def __init__(self, config: DecoderConfig):
        """
        Initialize the Decoder with the specified decoder type using the provided configuration.

        Args:
            config (DecoderConfig): Configuration object detailing the decoder settings.
        """
        super(Decoder, self).__init__()
        self.config = config

        # Construct the decoder instance using the specified decoder type
        self.decoder = self.construct_decoder()

    def construct_decoder(self):
        """
        Construct and return the decoder model using the specified decoder type.

        Retrieves the model configuration from the DECODER_REGISTRY and initializes the model.

        Returns:
            nn.Module: Instantiated decoder model.

        Raises:
            ValueError: If the specified decoder type is not supported in the registry.
        """
        if self.config.decoder_type in DECODER_REGISTRY:
            if self.config.decoder_type == "mamba":
                model = DECODER_REGISTRY["mamba"]["model"](self.config.decoder.mamba)
        else:
            raise ValueError(
                f"Decoder type {self.config.decoder_type} not supported. Supported decoders: {list(DECODER_REGISTRY.keys())}"
            )
        return model

    def forward(self, batch: Dict) -> Dict:
        """
        Execute a forward pass through the decoder to reconstruct encoded input data.

        The input batch is processed through the configured decoder model, transforming encoded
        features into the desired output format.
        Mask is used to hide padding elements from the decoder.

        Args:
            batch (Dict): A dictionary containing encoded data and masks.
                          Needs to contain the following keys:
                          - "encoding": The encoded representation of the input data.
                          - "mask_pad": A mask to hide padding elements from the decoder.

        Returns:
            Dict: Updated batch with decoded output added.
                    The batch will contain the following keys:
                    - "decoding": The decoded representation of the input data.
        """
        encoding = batch["encoding"]  # Extract encoded features from the batch
        mask = batch["eos_pad_mask"] if "eos_pad_mask" in batch else None  # Retrieve mask to handle padding or irrelevant data

        # Decode the encoded features using the constructed decoder model
        # The mask is used to hide padding elements from the decoder. If None, no mask is used.
        decoding = self.decoder(encoding, mask)

        # Add the decoded output to the batch dictionary
        batch["decoding"] = decoding
        return batch


if __name__ == "__main__":
    # Determine the computation device (use GPU if available, otherwise default to CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the configuration for the Decoder with necessary parameters
    config = DecoderConfig(decoder_type="mamba")  # Specify the type of decoder to use
    config.decoder.mamba.n_layer = 2
    config.decoder.mamba.d_model = 128
    config.decoder.mamba.d_input = config.decoder.mamba.d_model
    config.decoder.mamba.d_output = 3

    # Instantiate the Decoder module with the given configuration
    decoder = Decoder(config).to(device)

    # Generate a sample batch with random encoded data and a mask
    batch = {
        "encoding": torch.randn(4, 10, config.decoder.mamba.d_input).to(device),  # Random encoded data
        "mask_pad": torch.randint(0, 2, (4, 10)).bool().to(device),  # Random binary mask for padding
    }

    # Execute the forward pass of the Decoder to reconstruct the data
    batch = decoder(batch)

    # Print the updated batch to observe the decoded output
    print("Batch with decoded output:")
    print(batch)

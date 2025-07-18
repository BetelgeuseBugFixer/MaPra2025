from dataclasses import dataclass, field
from typing import Dict
from models.bio2token.layers.fsq import FSQ
import torch.nn as nn
import torch

from models.bio2token.layers.mamba import MambaConfig, MambaStack
from models.bio2token.layers.fsq import FSQConfig


# Registry dictionaries for encoder and quantizer models and configurations
ENCODER_REGISTRY = {
    "mamba": {"model": MambaStack, "config": MambaConfig},
}
QUANTIZER_REGISTRY = {"fsq": {"model": FSQ, "config": FSQConfig}}


@dataclass
class EncoderZoo:
    """
    Collection of configurations for various encoder models available in the EncoderRegistry.

    Attributes:
        mamba (MambaConfig): Configuration for MambaStack encoder.
    """

    mamba: MambaConfig = field(default_factory=MambaConfig)


@dataclass
class QuantizerZoo:
    """
    Collection of configurations for different quantizer models available in the QuantizerRegistry.

    Attributes:
        fsq (FSQConfig): Configuration for FSQ quantizer.
    """

    fsq: FSQConfig = field(default_factory=FSQConfig)


@dataclass
class EncoderConfig:
    """
    Configuration class for setting up an Encoder model.

    Specifies the type of encoder and whether a quantizer is used, along with configurations
    for both encoder and quantizer components.

    Attributes:
        encoder_type (str): Type of encoder to use. Defaults to 'mamba'.
        use_quantizer (bool): Flag indicating whether to include a quantizer. Defaults to True.
        quantizer_type (str): Type of quantizer to use, if applicable. Defaults to 'fsq'.
        encoder (EncoderZoo): A dataclass housing the configurations for all supported encoder models.
        quantizer (QuantizerZoo): A dataclass housing the configurations for all supported quantizer models.
    """

    encoder_type: str = "mamba"
    use_quantizer: bool = True
    quantizer_type: str = "fsq"
    encoder: EncoderZoo = field(default_factory=EncoderZoo)
    quantizer: QuantizerZoo = field(default_factory=QuantizerZoo)


class Encoder(nn.Module):
    """
    Encoder module that integrates different encoder and quantizer models for feature extraction.

    Depending on the configuration, this module allows for flexible setup of encoder types
    (e.g., Mamba) and optional quantization (e.g., FSQ) for processing input data.

    Attributes:
        config_cls (type): Associated configuration class, defined as EncoderConfig.
        config (EncoderConfig): Instance of EncoderConfig with parameters for encoder and quantizer.
        encoder (nn.Module): Instantiated encoder model based on the configuration.
        quantizer (nn.Module): Optional quantizer model, included if specified in the configuration.

    Args:
        config (EncoderConfig): Configuration providing setup options for encoder and optional quantizer.

    Methods:
        construct_encoder() -> nn.Module:
            Constructs the encoder model from the encoder type specified in the configuration.

        construct_quantizer() -> nn.Module:
            Constructs the quantizer model from the quantizer type specified in the configuration.

        forward(batch: Dict) -> Dict:
            Processes a batch of data through the encoder (and quantizer, if used) and outputs encoded features.
    """

    config_cls = EncoderConfig

    def __init__(self, config: EncoderConfig):
        """
        Initialize the Encoder with encoder and optional quantizer using the configuration.

        Args:
            config (EncoderConfig): Configuration object detailing the encoder and quantizer settings.
        """
        super(Encoder, self).__init__()
        self.config = config

        # Construct encoder model from specified encoder type
        self.encoder = self.construct_encoder()

        # Construct quantizer model if configured to use a quantizer
        if self.config.use_quantizer:
            self.quantizer = self.construct_quantizer()

    def construct_encoder(self):
        """
        Construct and return the encoder model based on the specified encoder type.

        Validates encoder type against the registry and initiates the respective model.

        Returns:
            nn.Module: Instantiated encoder model.

        Raises:
            ValueError: If the specified encoder type is not supported.
        """
        if self.config.encoder_type in ENCODER_REGISTRY:
            if self.config.encoder_type == "mamba":
                model = ENCODER_REGISTRY["mamba"]["model"](self.config.encoder.mamba)
        else:
            raise ValueError(
                f"Encoder type {self.config.encoder_type} not supported. Supported encoders: {list(ENCODER_REGISTRY.keys())}"
            )
        return model

    def construct_quantizer(self):
        """
        Construct and return the quantizer model based on the specified quantizer type.

        Validates quantizer type against the registry and initializes the respective model.

        Returns:
            nn.Module: Instantiated quantizer model.

        Raises:
            ValueError: If the specified quantizer type is not supported.
        """
        if self.config.quantizer_type in QUANTIZER_REGISTRY:
            if self.config.quantizer_type == "fsq":
                quantizer = QUANTIZER_REGISTRY["fsq"]["model"](self.config.quantizer.fsq)
        else:
            raise ValueError(
                f"Quantizer type {self.config.quantizer_type} not supported. Supported quantizers: {list(QUANTIZER_REGISTRY.keys())}"
            )
        return quantizer

    def forward(self, batch: Dict) -> Dict:
        """
        Execute a forward pass through the encoder (and quantizer, if applicable).

        This method processes input features and masks through the encoder/quantizer pipeline,
        modifying the batch in-place by adding encoded information and any quantization indices.
        Mask is used to hide padding elements from the encoder.

        Args:
            batch (Dict): A dictionary containing input data, including structures and masks.
                          Needs to contain the following keys:
                          - "structure": The input data to be encoded.
                          - "mask_pad": A mask to hide padding elements from the encoder.

        Returns:
            Dict: Updated batch with new encoding and indices (if quantization is applied).
                    The batch will contain the following keys:
                    - "encoding": The encoded representation of the input data.
                    - "indices": The indices of the encoded data, if quantization is applied.
        """
        features = batch["structure"]  # Extract input features for encoding
        mask = batch["eos_pad_mask"] if "eos_pad_mask" in batch else None  # Retrieve mask to handle padding or irrelevant data

        # Encode features using the constructed encoder model
        # The mask is used to hide padding elements from the encoder. If None, no mask is used.
        encoding = self.encoder(features, mask)
        indices = None  # Placeholder for quantization indices

        # If quantization is applied, process the encoding through the quantizer
        if self.config.use_quantizer:
            encoding, indices = self.quantizer(encoding)

        # Handle masked elements in the encoding by setting them to 0
        if mask is not None:
            # TODO: This is a hack to ensure padded tokens do not capture structural information.
            # However, setting to 0 might conflict with some quantization codebooks.
            encoding[mask] = 0
            if indices:
                indices[mask] = self.quantizer.codebook_size

        # Update batch with encoded data and indices
        batch["encoding"], batch["indices"] = encoding, indices
        return batch


if __name__ == "__main__":
    # Determine the computation device (use GPU if available, otherwise default to CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the configuration for the Encoder with necessary parameters
    config = EncoderConfig(
        encoder_type="mamba",  # Specify the type of encoder to use ('mamba')
        use_quantizer=True,  # Indicate whether to use a quantizer in the encoding process
    )
    config.encoder.mamba.n_layer = 4
    config.encoder.mamba.d_model = 128
    config.encoder.mamba.d_output = config.encoder.mamba.d_model
    config.quantizer.fsq.d_input = config.encoder.mamba.d_output

    # Instantiate the Encoder module with the given configuration
    encoder = Encoder(config).to(device)

    # Generate a sample batch with random structures and a mask
    batch = {
        "structure": torch.randn(4, 10, config.encoder.mamba.d_input).to(device),  # Random input structure data
        "mask_pad": torch.randint(0, 2, (4, 10)).bool().to(device),  # Random binary mask for padding
    }

    # Execute the forward pass of the Encoder to transform the input data
    batch = encoder(batch)

    # Print the updated batch to observe the encoded output and indices
    print("Batch with encoded output and indices:")
    print(batch)

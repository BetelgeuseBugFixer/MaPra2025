from dataclasses import dataclass
import torch.nn as nn
from typing import Dict, Optional

from models.bio2token.models.encoder import EncoderConfig, Encoder
from models.bio2token.models.decoder import DecoderConfig, Decoder
from models.bio2token.losses.loss import LossesConfig, Losses
from models.bio2token.utils.registration import Registration


@dataclass
class AutoencoderConfig:
    """
    Configuration for setting up an Autoencoder model, including its components.

    This dataclass holds configuration details for the encoder and decoder parts
    of an autoencoder, as well as optional configurations for registration
    and loss functions to be used in conjunction with the model.

    Attributes:
        encoder (EncoderConfig): Configuration for the encoder module, detailing architecture and parameters.
        decoder (DecoderConfig): Configuration for the decoder module, detailing architecture and parameters.
        registration (Optional[dict]): Optional configuration for a registration process to align or transform data.
        losses (Optional[LossesConfig]): Optional configuration for loss functions to compute during training.
    """

    encoder: EncoderConfig
    decoder: DecoderConfig
    registration: Optional[dict] = None
    losses: Optional[LossesConfig] = None


class Autoencoder(nn.Module):
    """
    An Autoencoder model composed of encoder, decoder, registration, and loss components.

    The Autoencoder class integrates separate components, including an encoder to compress
    data, a decoder to reconstruct data, optional registration to transform data, and
    losses for training feedback. It processes data through these steps to facilitate
    representation learning and reconstruction.

    Attributes:
        config_cls (type): Configuration class for this autoencoder, set to AutoencoderConfig.
        encoder (Encoder): Encoder component configured by the provided configuration.
        decoder (Decoder): Decoder component configured by the provided configuration.
        registration (Registration): Registration module for optional data alignment and transformation.
        loss (Losses): Module for calculating losses during the forward pass.

    Args:
        config (AutoencoderConfig): Configuration providing all necessary setup parameters for the autoencoder.

    Methods:
        forward(batch: Dict) -> Dict:
            Passes data through encoder, decoder, registration, and loss components,
            updating and returning the batch dictionary.
    """

    config_cls = AutoencoderConfig

    def __init__(self, config: AutoencoderConfig):
        """
        Initialize the Autoencoder model with a given configuration.

        Args:
            config (AutoencoderConfig): Configuration object detailing encoder, decoder,
                                        registration, and loss settings for the autoencoder.
        """
        super(Autoencoder, self).__init__()

        # Initialize encoder and decoder with provided configurations
        self.encoder = Encoder(config.encoder)
        self.decoder = Decoder(config.decoder)

        # Initialize optional registration module if configuration is provided
        self.registration = Registration(config.registration)

        # Initialize losses module if configuration is provided
        self.loss = Losses(config.losses)

    def forward(self, batch: Dict) -> Dict:
        """
        Execute a forward pass through the autoencoder model, processing input data.

        The input batch is sequentially processed by the encoder, decoder, registration,
        and loss components, facilitating data transformation, alignment, and feedback
        via calculated losses.

        Args:
            batch (Dict): A dictionary representing the data batch, containing input data
                          or features to be processed.

        Returns:
            Dict: The updated batch dictionary after processing, including transformed data
                  and calculated losses.
        """
        # Pass data through the encoder to generate encoded representations
        batch = self.encoder(batch)

        # Decode the encoded representations back to the original input space
        batch = self.decoder(batch)

        # Optionally transform or align data using registration component
        batch = self.registration(batch)

        # Calculate and apply losses to the batch
        batch = self.loss(batch)

        # Return the processed batch
        return batch

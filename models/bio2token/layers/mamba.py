import torch
import torch.nn as nn
from dataclasses import dataclass, field
from functools import partial
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
from mamba_ssm.models.mixer_seq_simple import create_block, _init_weights

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


@dataclass
class MambaConfig:
    """
    Configuration parameters for the Mamba model.

    This dataclass specifies a comprehensive set of configuration options for setting up
    and tuning the Mamba model, including network dimensions, layer configuration, normalization
    settings, and other operational parameters.

    Attributes:
        d_input (int): The dimensionality of the input features. Determines the size of each input vector.
        d_output (int): The dimensionality of the output features. Defines the size of each output vector.
        d_model (int): The size of the model's hidden states, influencing the capacity and complexity of the model.
        n_layer (int): The number of layers in the model, affecting the depth and representational power.
        d_intermediate (int): The size of the intermediate feedforward layer. If set to 0, no feedforward layer is included.
        ssm_cfg (dict): Configuration dictionary for additional settings specific to the SSM block.
        attn_layer_idx (list): Indices of layers at which attention mechanisms are applied.
        attn_cfg (dict): Configuration dictionary for additional settings specific to the attention layer.
        norm_epsilon (float): A small epsilon value used in layer normalization to prevent division by zero.
        rms_norm (bool): Flag indicating whether to use RMS normalization, which is more computationally efficient.
        residual_in_fp32 (bool): Specifies whether to perform residual calculations in 32-bit floating point precision.
        fused_add_norm (bool): Indicates whether to use fused add normalization for enhanced efficiency.
        initializer_cfg (dict): Configuration dictionary for initializing model parameters.
        bidirectional (bool): Specifies whether the model should be bidirectional, enabling contexts
                              from both past and future in tasks requiring sequence analysis.
    """

    d_input: int = 3
    d_output: int = 3
    d_model: int = 2560
    n_layer: int = 64
    d_intermediate: int = 0
    ssm_cfg: dict = field(default_factory=dict)
    attn_layer_idx: list = field(default_factory=list)
    attn_cfg: dict = field(default_factory=dict)
    norm_epsilon: float = 1e-5
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    initializer_cfg: dict = field(default_factory=dict)
    bidirectional: bool = False


class MambaStack(nn.Module):
    """
    A neural network module implementing the Mamba architecture.

    The MambaStack class constructs a stack of layers based on the Mamba configuration settings,
    with provisions for handling residual connections, input/output projection, normalization,
    and optional attention mechanisms. It is designed for sequential data processing,
    allowing for dynamic adjustments in layer configurations and inference settings.

    Attributes:
        config_cls (type): The class type used for configuration, set to MambaConfig.
        config (MambaConfig): Configuration instance containing all setup parameters for the Mamba stack.
        residual_in_fp32 (bool): Flag indicating if residual connections should be computed in 32-bit floating point.
        fused_add_norm (bool): Specifies whether to use fused add normalization for efficiency.
        bidirectional (bool): Determines if the model processes data bidirectionally, accommodating sequential back-propagation.
        input_projection (nn.Module): Projection layer mapping input dimensions to the model's working dimensions.
        output_projection (nn.Module): Projection layer mapping model dimensions to the desired output dimensions.
        layers (nn.ModuleList): List of Mamba blocks forming the main processing architecture.
        norm_f (nn.Module): Final normalization layer applied after processing blocks, can be RMSNorm or LayerNorm.

    Methods:
        allocate_inference_cache(batch_size, max_seqlen, dtype=None, **kwargs)
            Allocates cache memory for inference, storing layer-specific context.

        forward(input_ids: torch.Tensor, mask: torch.Tensor = None, inference_params=None, **mixer_kwargs) -> torch.Tensor
            Executes a forward pass through the stacked layers, transforming input sequences into output representations.

    Args:
        config (MambaConfig): Configuration object providing detailed layer specifications, normalization preferences,
                              and dimension settings.

    Notes:
        - The forward method supports masking.
        - Bidirectional processing is conditional on the `bidirectional` flag, flipping sequences where needed
          to incorporate context from both directional passes.
        - The class includes mechanisms for efficient initialization and operations, adopting techniques like fused layers
          and optional use of RMSNorm for computational gain.
    """

    config_cls = MambaConfig

    def __init__(
        self,
        config: MambaConfig,
    ) -> None:
        """
        Initialize the MambaStack with the given configuration.

        Sets up all necessary layers, projections, and normalization functions based on the specified
        configuration parameters.

        Args:
            config (MambaConfig): An instance of the MambaConfig class defining the architecture and operation
                                  parameters for the Mamba stack.
        """
        super(MambaStack, self).__init__()
        self.config = config
        self.residual_in_fp32 = config.residual_in_fp32
        self.fused_add_norm = config.fused_add_norm
        self.bidirectional = config.bidirectional
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        # Determine if input/output projections are needed based on dimension mismatch
        has_projections = config.d_input != config.d_model
        self.input_projection = nn.Linear(config.d_input, config.d_model, bias=False) if has_projections else nn.Identity()
        has_projections = config.d_output != config.d_model
        self.output_projection = nn.Linear(config.d_model, config.d_output, bias=False) if has_projections else nn.Identity()

        # Initialize the processing layers
        self.layers = nn.ModuleList(
            [
                create_block(
                    config.d_model,
                    d_intermediate=config.d_intermediate,
                    ssm_cfg=config.ssm_cfg,
                    attn_layer_idx=config.attn_layer_idx,
                    attn_cfg=config.attn_cfg,
                    norm_epsilon=config.norm_epsilon,
                    rms_norm=config.rms_norm,
                    residual_in_fp32=config.residual_in_fp32,
                    fused_add_norm=config.fused_add_norm,
                    layer_idx=i,
                )
                for i in range(config.n_layer)
            ]
        )

        # Final normalization layer to maintain scale consistency
        self.norm_f = (nn.LayerNorm if not config.rms_norm else RMSNorm)(config.d_model, eps=config.norm_epsilon)

        # Initialize model parameters using a custom function
        self.apply(
            partial(
                _init_weights,
                n_layer=config.n_layer,
                **(config.initializer_cfg if config.initializer_cfg is not None else {}),
                n_residuals_per_layer=1 if config.d_intermediate == 0 else 2,  # 2 if we have MLP
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        """
        Allocate memory for inference caches across layers.

        This method prepares storage for maintaining state and context during sequence inference,
        enhancing computational efficiency during recursive or ongoing prediction tasks.

        Args:
            batch_size (int): Number of sequences to be processed in parallel.
            max_seqlen (int): Maximum sequence length expected during inference.
            dtype (torch.dtype, optional): Data type for stored states; defaults according to architecture settings.
            **kwargs: Additional arguments for custom cache allocation specifications.

        Returns:
            Dict: A dictionary with cache states for each layer in the stack, allowing for independent
                  layer inference support.
        """
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(
        self, input_ids: torch.Tensor, mask: torch.Tensor = None, inference_params=None, **mixer_kwargs
    ) -> torch.Tensor:
        """
        Perform a forward pass, processing sequences through the Mamba stack.

        Maps input tensors into output sequences, applying transformation through configured sequential layers
        with support for optional masking and bidirectional processing.

        Args:
            input_ids (torch.Tensor): Input sequences with dimension [batch, length, features].
            mask (torch.Tensor, optional): A binary mask to influence layer processing, matching input sequence shape.
            inference_params: Custom parameters dictating inference behavior (optional).
            **mixer_kwargs: Additional keyword arguments for specialized layer operations.

        Returns:
            torch.Tensor: Transformed output with same batch size and sequence length, but potential format adjustments
                          inherent to the configured output dimensions.
        """
        # Project input dimensions if necessary
        hidden_states = self.input_projection(input_ids)
        # Set initial residual state
        residual = None
        for layer in self.layers:
            if mask is not None:
                # NTF: zero-out the hidden states that are masked for the right-to-left pass
                # for a causal Mamba layer, this should be Okay as the hidden states will be zeroed out
                # for the masked positions
                hidden_states = hidden_states * (~mask.unsqueeze(-1))
                if residual is not None:
                    residual = residual * (~mask.unsqueeze(-1))
            # Process forward and potentially backward information through the Mamba block
            hidden_states_forward, residual_forward = layer(
                hidden_states, residual, inference_params=inference_params, **mixer_kwargs
            )
            if self.bidirectional:
                hidden_states_backward = hidden_states.flip(1)
                residual_backward = residual.flip(1) if residual is not None else None
                hidden_states_backward, residual_backward = layer(
                    hidden_states_backward, residual_backward, inference_params=inference_params, **mixer_kwargs
                )
                hidden_states_backward = hidden_states_backward.flip(1)
                residual_backward = residual_backward.flip(1)
                hidden_states = hidden_states_forward + hidden_states_backward
                residual = residual_forward + residual_backward
            else:
                hidden_states = hidden_states_forward
                residual = residual_forward

        # Apply the final Add+Norm layer for integration consistency
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            hidden_states = layer_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                is_rms_norm=isinstance(self.norm_f, RMSNorm),
            )
        # Project to output dimensions if required
        output_states = self.output_projection(hidden_states)
        if mask is not None:
            output_states = output_states * (~mask.unsqueeze(-1))
        return output_states


if __name__ == "__main__":
    # Define parameters for batch size, sequence length, and input/output dimensions
    batch_size = 4
    seq_len = 5
    d_input = 3
    d_output = 3

    # Determine device setting (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize MambaConfig with input and output dimensions
    config = MambaConfig(d_input=d_input, d_output=d_output)

    # Instantiate the MambaStack model using the configuration and move it to the chosen device
    model = MambaStack(config).to(device)

    # Create a random input tensor matching the batch and sequence length specifications
    x = torch.randn(batch_size, seq_len, d_input).to(device)

    # Perform a forward pass through the model with the input tensor
    y = model(x)

    # Output the model architecture and the shape of the resulting tensor
    print(model)
    print(y.shape)

import torch
from torch import nn
from hydra_zen import load_from_yaml, builds, instantiate

from models.bio2token.models.autoencoder import AutoencoderConfig, Autoencoder
from models.bio2token.utils.configs import pi_instantiate
from peft import get_peft_model, LoraConfig, TaskType


def load_bio2_token_decoder_and_quantizer():
    model_configs = load_from_yaml("models/bio2token/files/model.yaml")["model"]
    model_config = pi_instantiate(AutoencoderConfig, model_configs)
    model = Autoencoder(model_config)
    state_dict = torch.load("models/bio2token/files/epoch=0243-val_loss_epoch=0.71-best-checkpoint.ckpt")["state_dict"]
    # Remove 'model.' prefix from keys if present
    state_dict_bis = {k.replace("model.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict_bis)
    return model.decoder, model.encoder.quantizer


class Bio2tokenDecoder(nn.Module):
    def __init__(self, device="cpu", use_lora=False):
        super().__init__()
        # load pretrained stuff from bio2token
        self.decoder, self.quantizer = load_bio2_token_decoder_and_quantizer()

        # freeze quantizer
        self.quantizer = self.quantizer.to(device)
        for param in self.quantizer.parameters():
            param.requires_grad = False
        self.quantizer.eval()

        self.decoder = self.decoder.to(device)
        # init lora
        self.use_lora = use_lora
        if use_lora:
            target_modules = ["x_proj", "dt_proj", "out_proj"]

            self.lora_config = LoraConfig(
                r=8,
                lora_alpha=32,
                target_modules=target_modules,
                lora_dropout=0.1,
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION,
            )
            #add important configs for lora
            self.decoder.config = {"tie_word_embeddings": False}

            self.decoder = get_peft_model(self.decoder, self.lora_config)
        else:
            # freeze model if we don't use lora
            for param in self.decoder.parameters():
                param.requires_grad = False
            self.decoder.eval()


    def forward(self, x,eos_mask):
        encoding = self.quantizer.indices_to_codes(x)
        decoding = self.decoder.decoder(encoding, eos_mask)
        return decoding
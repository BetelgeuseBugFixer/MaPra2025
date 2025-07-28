import torch
from torch import nn
from hydra_zen import load_from_yaml, builds, instantiate

from models.bio2token.models.autoencoder import AutoencoderConfig, Autoencoder
from models.bio2token.utils.configs import pi_instantiate
from peft import get_peft_model, LoraConfig, TaskType




def load_bio2token_model():
    model_configs = load_from_yaml("models/bio2token/files/model.yaml")["model"]
    model_config = pi_instantiate(AutoencoderConfig, model_configs)
    model = Autoencoder(model_config)
    state_dict = torch.load("models/bio2token/files/epoch=0243-val_loss_epoch=0.71-best-checkpoint.ckpt")[
        "state_dict"]
    # Remove 'model.' prefix from keys if present
    state_dict_bis = {k.replace("model.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict_bis)
    return model

def load_bio2token_decoder_and_quantizer():
    model=load_bio2token_model()
    return model.decoder, model.encoder.quantizer

def load_bio2token_encoder():
    model=load_bio2token_model()
    return model.encoder

class Bio2tokenDecoder(nn.Module):
    def __init__(self, device="cpu", use_lora=False):
        super().__init__()
        # load pretrained stuff from bio2token
        self.decoder, self.quantizer = load_bio2token_decoder_and_quantizer()

        # freeze quantizer
        self.quantizer = self.quantizer.to(device)
        for param in self.quantizer.parameters():
            param.requires_grad = False
        self.quantizer.eval()

        self.decoder = self.decoder.to(device)

        # freeze model if we don't use lora = retrain
        if not use_lora:
            for param in self.decoder.parameters():
                param.requires_grad = False
            self.decoder.eval()




    def forward(self, x,eos_mask):
        encoding = self.quantizer.indices_to_codes(x)
        decoding = self.decoder.decoder(encoding, eos_mask)
        return decoding
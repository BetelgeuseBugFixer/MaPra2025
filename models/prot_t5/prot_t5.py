import torch
from torch import nn
from peft import get_peft_model, LoraConfig, TaskType

from transformers import T5EncoderModel, T5Tokenizer


class ProtT5(nn.Module):
    def __init__(
            self,
            device="cpu",
            transformer_link="Rostlab/prot_t5_xl_half_uniref50-enc",
            use_lora: bool = False,
            lora_r: int = 8,
            lora_alpha: int = 16,
            lora_dropout: float = 0.1,
    ):
        super().__init__()
        self.device = device
        self.tokenizer = T5Tokenizer.from_pretrained(transformer_link, do_lower_case=False, use_fast=False)
        base_model = T5EncoderModel.from_pretrained(transformer_link)

        if use_lora:
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION,
                target_modules=["q", "v"]
            )
            self.model = get_peft_model(base_model, lora_config).to(device)
        else:
            for param in base_model.parameters():
                param.requires_grad = False
            self.model = base_model.to(device).eval()

    def forward(self, seqs: list[str]) -> torch.Tensor:
        encoding = self.tokenizer.batch_encode_plus(
            seqs,
            add_special_tokens=True,
            padding='longest',
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].to(self.model.device)
        attention_mask = encoding['attention_mask'].to(self.model.device)

        outputs = self.model(input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # (B, L+1, D)
        # remove embeddings from eos and padding
        valid_positions = (input_ids != 0) & (input_ids != 1)
        # Apply mask directly without materializing full-size tensor
        hidden = hidden * valid_positions.unsqueeze(-1).to(hidden.dtype)
        # remove last "residue" embeddings, because it is the eos from the longest seq
        hidden = hidden[:, :-1, :]
        return hidden

import torch
from torch import nn

from transformers import T5EncoderModel, T5Tokenizer


class ProtT5(nn.Module):
    def __init__(self, transformer_link="Rostlab/prot_t5_xl_half_uniref50-enc", ):
        super().__init__()
        self.model = T5EncoderModel.from_pretrained(transformer_link)
        self.tokenizer = T5Tokenizer.from_pretrained(transformer_link, do_lower_case=False, use_fast=False)

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
        hidden = outputs.last_hidden_state  # (B, L, D)

        return hidden
import os
from typing import List

import torch
from torch import nn

from models.foldtoken_decoder.foldtoken_decoder import FoldDecoder
from models.model_utils import _masked_accuracy
from models.prot_t5.prot_t5 import ProtT5
from models.simple_classifier.datasets import PAD_LABEL
from models.simple_classifier.simple_classifier import ResidueTokenCNN


class TFold(nn.Module):
    def __init__(self, hidden: list, device="cpu", kernel_sizes=[5], dropout: float = 0.3,use_lora= False):
        super().__init__()
        self.device = device
        self.plm = ProtT5(use_lora=use_lora).to(self.device)
        embeddings_size = 1024
        codebook_size = 1024
        self.cnn = ResidueTokenCNN(embeddings_size, hidden, codebook_size, kernel_sizes, dropout).to(device)
        self.decoder = FoldDecoder(device=self.device)

        # freeze decoder 
        for param in self.decoder.parameters():
            param.requires_grad = False
        
        print(sum(p.numel() for p in self.plm.parameters() if p.requires_grad))  # sollte klein sein


        # save args
        self.args = {
            "hidden": hidden,
            "kernel_sizes": kernel_sizes,
            "dropout": dropout,
            "device": device
        }
        hidden_layers_string = "_".join(str(i) for i in hidden)
        kernel_sizes_string = "_".join(str(i) for i in kernel_sizes)
        self.model_name = f"t_fold_k{kernel_sizes_string}_h{hidden_layers_string}"

    def save(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"{self.model_name}.pt")
        torch.save({
            "model_args": self.args,
            "state_dict": self.state_dict()
        }, save_path)

    @staticmethod
    def load_tfold(path: str, device="cpu") -> "TFold":
        checkpoint = torch.load(path, map_location=device)
        model_args = checkpoint["model_args"]
        model_args["device"] = device

        model = TFold(**model_args)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()
        return model

    def forward(self, seqs: List[str]):
        # save the lengths of the sequences
        true_lengths = [len(seq) for seq in seqs]
        # prepare seqs
        x = [" ".join(seq.translate(str.maketrans('UZO', 'XXX'))) for seq in seqs]
        # generate embeddings
        x = self.plm(x)
        # generate tokens
        x = self.cnn(x)  # shape: (B, L, vocab_size)
        tokens = x.argmax(dim=-1)
        # prepare tokens for decoding
        vq_codes = []
        batch_ids = []
        chain_encodings = []
        for i, protein_token in enumerate(tokens):
            L = true_lengths[i]
            protein_token_without_padding = protein_token[:L]
            vq_codes.append(protein_token_without_padding)
            batch_ids.append(torch.full((L,), i, dtype=torch.long, device=protein_token.device))
            chain_encodings.append(torch.full((L,), 1, dtype=torch.long, device=protein_token.device))
        # reshape
        vq_codes_cat = torch.cat(vq_codes, dim=0)
        batch_ids_cat = torch.cat(batch_ids, dim=0)
        chain_encodings_cat = torch.cat(chain_encodings, 0)
        # decode proteins
        proteins = self.decoder.decode(vq_codes_cat, chain_encodings_cat, batch_ids_cat)
        # return proteins and tokens
        return proteins, x

    def run_epoch(self, loader, optimizer=None, device="cpu"):
        is_train = optimizer is not None
        self.train() if is_train else self.eval()
        total_loss = total_acc = total_samples = 0
        torch.set_grad_enabled(is_train)

        for sequences, tokens, ref_atom in loader:
            sequences, tokens = sequences.to(device), tokens.to(device)
            mask = (tokens != PAD_LABEL)
            protein_predictions, token_predictions = self(sequences)
            loss = self.criterion(token_predictions.transpose(1, 2), tokens)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            bsz = len(sequences)
            total_loss += loss.item() * bsz
            total_acc += _masked_accuracy(token_predictions, tokens, mask) * bsz
            total_samples += bsz
        set_prefix = ""
        if not is_train:
            set_prefix = "val_"
        score_dict = {
            f"{set_prefix}acc": total_acc / total_samples,
            f"{set_prefix}loss": total_loss / total_samples,
        }
        return score_dict

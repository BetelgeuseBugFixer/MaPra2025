import os
import sys
from typing import List

import torch
from torch import nn

from models.bio2token.decoder import Bio2tokenDecoder
from models.foldtoken_decoder.foldtoken import FoldToken
from models.model_utils import _masked_accuracy, calc_token_loss, calc_lddt_scores, SmoothLDDTLoss
from models.prot_t5.prot_t5 import ProtT5
from models.datasets.datasets import PAD_LABEL
from models.simple_classifier.simple_classifier import ResidueTokenCNN


class FinalModel(nn.Module):
    def __init__(self, hidden: list, device="cpu", kernel_sizes=[5], dropout: float = 0.1, plm_lora=False,
                 decoder_lora=False):
        super().__init__()
        self.device = device
        self.plm = ProtT5(use_lora=plm_lora, device=device).to(self.device)
        embeddings_size = 1024
        self.decoder = Bio2tokenDecoder(device=device).to(device)
        codebook_size = 128
        self.cnn = ResidueTokenCNN(embeddings_size, hidden, codebook_size, kernel_sizes, dropout,
                                   bio2token=True).to(device)

        self.args = {
            "hidden": hidden,
            "kernel_sizes": kernel_sizes,
            "dropout": dropout,
            "device": device,
            "plm_lora": plm_lora,
            "decoder_lora": decoder_lora,
        }
        hidden_layers_string = "_".join(str(i) for i in hidden)
        kernel_sizes_string = "_".join(str(i) for i in kernel_sizes)
        lora_string = "_plm_lora" if plm_lora else ""
        self.model_name = f"final_k{kernel_sizes_string}_h{hidden_layers_string}{lora_string}"

        # define most important metric and whether it needs to be minimized or maximized
        self.key_metric = "val_loss"
        self.maximize = False

    def save(self, output_dir: str, suffix=""):
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"{self.model_name}{suffix}.pt")
        torch.save({
            "model_args": self.args,
            "state_dict": self.state_dict()
        }, save_path)

    @staticmethod
    def load_final(path: str, device="cpu") -> "FinalModel":
        checkpoint = torch.load(path, map_location=device)
        model_args = checkpoint["model_args"]
        model_args["device"] = device

        model = FinalModel(**model_args)
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
        return self.forward_from_embedding(x, true_lengths)

    def forward_from_embedding(self, x, true_lengths=None):
        if true_lengths is None:
            # if we do not have lengths derive them from embeddings
            true_lengths = (x.abs().sum(-1) > 0).sum(dim=1)
        cnn_out = self.cnn(x)
        B, L, _ = cnn_out.shape
        eos_mask = torch.ones(B, L, dtype=torch.bool, device=x.device)
        for i, length in enumerate(true_lengths):
            eos_mask[i, :length * 4] = False
        x = self.decoder.decoder.decoder(cnn_out, eos_mask)
        # create mask for all relevant positions
        final_mask=~eos_mask
        return x,final_mask,cnn_out

    def run_epoch(self, loader, optimizer=None, device="cpu"):
        is_train=optimizer is not None
        self.train() if is_train else self.eval()
        # init statistics
        total_loss = total_samples = 0

        torch.set_grad_enabled(is_train)
        lddt_loss_module = SmoothLDDTLoss().to(device)
        # if we finetune we expect embeddings, otherwise sequences so we have to adapt the forward Method
        if self.plm_lora:
            forward = self.forward
        else:
            forward = self.forward_from_embedding
        for model_in, structure in loader:
            model_in, structure = model_in.to(device), structure.to(device)
            predictions,final_mask = forward(model_in)
            #get loss:
            B, L, _ = predictions.shape
            is_dna = torch.zeros((B, L), dtype=torch.bool, device=device)
            is_rna = torch.zeros((B, L), dtype=torch.bool, device=device)
            loss = lddt_loss_module(predictions, structure, is_dna, is_rna, final_mask)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


            total_loss += loss.detach().item() * B
            total_samples += B
            del loss, predictions, final_mask

        set_prefix = ""
        if not is_train:
            set_prefix = "val_"
        score_dict = {
            f"{set_prefix}loss": total_loss / total_samples,
        }
        return score_dict




class TFold(nn.Module):
    def __init__(self, hidden: list, device="cpu", kernel_sizes=[5], dropout: float = 0.1, use_lora=False,
                 bio2token=False):
        super().__init__()
        self.device = device
        self.plm = ProtT5(use_lora=use_lora, device=device).to(self.device)
        embeddings_size = 1024

        # choose decoder and forward method based in it
        if bio2token:
            self.decoder = Bio2tokenDecoder(device=device).to(device)
            codebook_size = 4096
            self.forward_from_embedding = self.forward_from_embedding_bio2token
        else:
            self.decoder = FoldToken(device=self.device)
            codebook_size = 1024
            self.forward_from_embedding = self.forward_from_embedding_foldtoken

        self.cnn = ResidueTokenCNN(embeddings_size, hidden, codebook_size, kernel_sizes, dropout,
                                   bio2token=bio2token).to(device)
        self.use_lora = use_lora

        # freeze decoder
        for param in self.decoder.parameters():
            param.requires_grad = False

        # save args
        decoder_type = "bio2token" if bio2token else "foldotken"
        self.args = {
            "hidden": hidden,
            "kernel_sizes": kernel_sizes,
            "dropout": dropout,
            "device": device,
            "use_lora": use_lora,
            "decoder": decoder_type
        }
        hidden_layers_string = "_".join(str(i) for i in hidden)
        kernel_sizes_string = "_".join(str(i) for i in kernel_sizes)
        lora_string = "_lora" if use_lora else ""
        self.model_name = f"t_fold_{decoder_type}_k{kernel_sizes_string}_h{hidden_layers_string}{lora_string}"

        # define most important metric and whether it needs to be minimized or maximized
        self.key_metric = "val_loss"
        self.maximize = False

    def save(self, output_dir: str, suffix=""):
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"{self.model_name}{suffix}.pt")
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
        return self.forward_from_embedding(x, true_lengths)

    def forward_from_embedding_bio2token(self, x, true_lengths=None):
        if true_lengths is None:
            # if we do not have lengths derive them from embeddings
            true_lengths = (x.abs().sum(-1) > 0).sum(dim=1)
        # generate tokens
        x = self.cnn(x)  # shape: (B, L, vocab_size)
        tokens = x.argmax(dim=-1)
        # prepare tokens for decoding
        B, L = x.shape
        eos_mask = torch.ones(B, L, dtype=torch.bool, device=x.device)
        for i, length in enumerate(true_lengths):
            eos_mask[i, :length * 4] = False
        x = self.decoder(x, eos_mask=eos_mask)
        return x, tokens

    def forward_from_embedding_foldtoken(self, x, true_lengths=None):
        if true_lengths is None:
            # if we do not have lengths derive them from embeddings
            true_lengths = (x.abs().sum(-1) > 0).sum(dim=1)
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

    def get_cnn_out_only(self, seqs: List[str]):
        # prepare seqs
        x = [" ".join(seq.translate(str.maketrans('UZO', 'XXX'))) for seq in seqs]
        # generate embeddings
        x = self.plm(x)
        # generate tokens
        x = self.cnn(x)  # shape: (B, L, vocab_size)
        return x

    def run_train_epoch(self, loader, optimizer=None, device="cpu", forward_method=None):
        # prepare model for training
        self.train()
        total_loss = total_acc = total_samples = 0
        # run through model
        for model_in, tokens in loader:
            tokens = tokens.to(device)
            mask = (tokens != PAD_LABEL)
            logits = forward_method(model_in)
            # get token loss
            loss = calc_token_loss(self.cnn.criterion, logits, tokens)
            # back propagate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # update scores
            bsz = tokens.size(0)
            total_loss += loss.item() * bsz
            total_acc += _masked_accuracy(logits, tokens, mask) * bsz
            total_samples += bsz
            # empty cache
            del logits, loss
            # torch.cuda.empty_cache()

        score_dict = {
            "acc": total_acc / total_samples,
            "loss": total_loss / total_samples,
        }
        return score_dict

    def run_val_epoch(self, loader, device="cpu", forward_method=None):
        self.eval()
        total_loss = total_acc = total_samples = total_lddt = 0

        with torch.no_grad():
            for model_in, tokens, protein_references in loader:
                # get predictions
                tokens = tokens.to(device)
                mask = (tokens != PAD_LABEL)
                protein_predictions, logits = forward_method(model_in)
                # get loss and score
                loss = calc_token_loss(self.cnn.criterion, logits, tokens)
                bsz = tokens.size(0)
                total_lddt += sum(calc_lddt_scores(protein_predictions, protein_references))
                total_loss += loss.detach().item() * bsz
                total_acc += _masked_accuracy(logits, tokens, mask) * bsz
                total_samples += bsz
                # empty cache
                del logits, loss
                # torch.cuda.empty_cache()
        # return scores
        score_dict = {
            "val_acc": total_acc / total_samples,
            "val_loss": total_loss / total_samples,
            "val_lddt": total_lddt / total_samples
        }
        return score_dict

    def run_epoch(self, loader, optimizer=None, device="cpu"):
        if optimizer is None:
            # no use of creating embeddings if we don't finetune plm
            if self.use_lora:
                return self.run_val_epoch(loader, device, lambda x: self(x))
            else:
                return self.run_val_epoch(loader, device, self.forward_from_embedding)
        else:
            if self.use_lora:
                return self.run_train_epoch(loader, optimizer, device, self.get_cnn_out_only)
            else:
                return self.run_train_epoch(loader, optimizer, device, lambda x: self.cnn(x))

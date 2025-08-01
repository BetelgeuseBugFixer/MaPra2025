import os
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from models.foldtoken_decoder.src.data import Protein
from models.foldtoken_decoder.src.model_interface import MInterface


class FoldToken(nn.Module):
    def __init__(self, checkpoint_dir='models/foldtoken_decoder/model_zoom/FT4', device='cpu', level=10):
        super().__init__()
        self.level = level
        self.device = device
        self.model = self._load_model(checkpoint_dir).to(device)
        self.model.eval()  # freeze mode

    def _load_model(self, checkpoint_dir):
        config_path = os.path.join(checkpoint_dir, 'config.yaml')
        checkpoint_path = os.path.join(checkpoint_dir, 'ckpt.pth')

        config = OmegaConf.load(config_path)
        config = OmegaConf.to_container(config, resolve=True)

        model = MInterface(**config)

        checkpoint = torch.load(checkpoint_path, map_location=torch.device(self.device))

        # Remove DataParallel keys if needed
        for key in list(checkpoint.keys()):
            if '_forward_module.' in key:
                checkpoint[key.replace('_forward_module.', '')] = checkpoint[key]
                del checkpoint[key]

        model.load_state_dict(checkpoint, strict=False)

        return model

    def decode_single_protein_vqs_to_structure(self, vq_codes):
        h_V = self.model.model.vq.embed_id(vq_codes, self.level)
        L, _ = h_V.shape
        X_t = torch.rand(L, 4, 3, device='cuda')
        batch_id = torch.zeros(L, device='cuda').long()
        virtual_frame_num = 3
        chain_encoding = torch.ones_like(vq_codes, device=self.device)
        return self.model.model.decoder_struct.infer_X(X_t, h_V, batch_id, chain_encoding, 30,
                                                       virtual_frame_num=virtual_frame_num)

    def decode(self, vq_codes_batch, chain_encodings, batch_ids):
        # get latent embeddings
        h_V = self.model.model.vq.embed_id(vq_codes_batch, self.level)  # (B, L, D)
        # no fucking idea why this is needed, but it is in the original code
        L, _ = h_V.shape
        X_t = torch.rand(L, 4, 3, device='cuda')
        # return get predicted structure
        X_pred, all_preds = self.model.model.decoder_struct.infer_X(X_t, h_V, batch_ids, chain_encodings)
        # create protein objects
        proteins = []
        for i in range(batch_ids.max() + 1):
            idx = batch_ids == i
            X_i = X_pred[idx]
            C_i = chain_encodings[idx]
            S_i = torch.ones_like(C_i)
            protein = Protein.from_XCS(X_i[None], C_i[None], S_i[None])
            proteins.append(protein)
        return proteins

    def decode_single_prot(self, vq_codes):
        # get latent embeddings
        h_V = self.model.model.vq.embed_id(vq_codes, self.level)
        # simple chain encoding
        chain_encoding = torch.ones_like(vq_codes, device=self.device)
        # decode to protein object
        protein = self.model.model.decoding(h_V, chain_encoding)
        # protein.to(output_path)
        return protein

    def encode_pdb(self, pdb_path):
        protein = Protein(pdb_path, device=self.device)
        with torch.no_grad():
            vq_code = self.model.encode_protein(protein, level=self.level)[1]
            return vq_code

    def encode_lists_of_pdbs(self, pdb_paths, device):
        list_of_proteins = []
        for pdb_path in pdb_paths:
            list_of_proteins.append(Protein(pdb_path, device='cuda'))
        with torch.no_grad():
            batch = self.model.batch_proteins(list_of_proteins)
            vq_ids = self.model.encode_only(batch)

        return vq_ids

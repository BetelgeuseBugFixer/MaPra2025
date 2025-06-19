import json

import torch

from models.foldtoken_decoder.foldtoken_decoder import FoldDecoder

if __name__ == '__main__':
    device = "cuda"
    model = FoldDecoder()
    model.to(device)
    token_example = json.load("data/casp14_test/casp14_tokens.jsonl")["T1082-D1"]["vqid"]
    vq_codes = torch.tensor(token_example, dtype=token_example.long, device=device)
    model.decode_single_prot(vq_codes,"test.pdb")


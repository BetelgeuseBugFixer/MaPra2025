import json

import torch

from models.foldtoken_decoder.foldtoken_decoder import FoldDecoder

if __name__ == '__main__':
    device = "cuda"
    model = FoldDecoder(device=device)
    token_data = {}

    with open("data/casp14_test/casp14_tokens.jsonl", "r") as f:
        for line in f:
            entry = json.loads(line)
            token_data.update(entry)
            
    token_example = token_data["T1082-D1"]["vqid"]
    vq_codes = torch.tensor(token_example, dtype=torch.long, device=device)
    model.decode_single_prot(vq_codes,"test.pdb")


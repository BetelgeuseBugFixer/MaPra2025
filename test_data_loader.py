import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from models.bio2token.data.utils.utils import pad_and_stack_tensors
from models.datasets.datasets import StructureAndTokenSet

PAD_LABEL = -100
#dataset = SeqStrucTokSet("data/large/val/proteins.jsonl", "data/large/val/proteins.pkl")
def collate_seq_struc_tok_batch(batch):
    sequences, token_lists, structures = zip(*batch)

    # Padding der VQ-Tokens
    padded_tokens = pad_sequence(token_lists, batch_first=True, padding_value=PAD_LABEL)
    padded_structures = pad_and_stack_tensors(structures, 0)

    return list(sequences), padded_tokens, padded_structures

def print_tensor(tensor,name):
    print(f"{name}-{tensor.shape}:\n{tensor}")


def collate_emb_struc_tok_batch(batch):
    embs, toks, structures = zip(*batch)

    # pad embeddings along the sequence dimension
    embs_padded = pad_sequence(embs, batch_first=True)  # pad with 0.0 by default
    # pad tokens along the sequence dimension, with PAD_LABEL
    toks_padded = pad_sequence(toks, batch_first=True, padding_value=PAD_LABEL)
    # pad structures
    structures = pad_and_stack_tensors(structures, 0)
    return embs_padded, toks_padded, structures

file_dir="data/large/val"
dataset = StructureAndTokenSet(file_dir,"encoding",precomputed_embeddings=True)

loader = DataLoader(dataset,batch_size=2,collate_fn=collate_emb_struc_tok_batch)


for emb,struc,tokens in loader:
    print_tensor(emb,"emb")
    print_tensor(struc,"structure")
    print_tensor(tokens,"tokens")
    break


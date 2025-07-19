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


file_dir="data/large/val"
dataset = StructureAndTokenSet(file_dir,"encoding",precomputed_embeddings=False)

loader = DataLoader(dataset,batch_size=2,collate_fn=collate_seq_struc_tok_batch)


for seq,struc,tokens in loader:
    print(len(seq),seq)
    print_tensor(struc,"structure")
    print_tensor(tokens,"tokens")
    break


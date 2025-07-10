from biotite.structure import to_sequence
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from models.datasets.datasets import SeqStrucTokSet, PAD_LABEL, SeqTokSet
from models.train import collate_seq_tok_batch

#dataset = SeqStrucTokSet("data/large/val/proteins.jsonl", "data/large/val/proteins.pkl")


# loader = DataLoader(
#     dataset,
#     batch_size=16,
#     collate_fn=collate_seq_struc_tok_batch,
# )

dataset = SeqTokSet("data/large/val/proteins.jsonl")
loader = DataLoader(dataset, batch_size=16, collate_fn=collate_seq_tok_batch)

for seq,vq_ids in loader:
    for i in range(len(seq)):
        print(seq[i])
        print(vq_ids[i])
        print("="*11)
    break


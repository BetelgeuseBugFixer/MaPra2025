import torch
from torch.utils.data import Dataset, DataLoader
from models.end_to_end.whole_model import FinalModel, TFold
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset/Collate for raw sequences
class SeqDataset(Dataset):
    def __init__(self, seqs):
        self.seqs = seqs

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx]


def collate_seqs(batch):
    return list(batch)


def infer_structures(load_fn, ckpt_path, seqs, batch_size=16):
    """
    load_fn: a function like FinalModel.load_final or TFold.load_tfold
    ckpt_path: path to your .pt checkpoint
    seqs: List[str]
    returns: List of predicted structures (one per sequence)
    """
    print(f"in infer struct: {ckpt_path}")
    model = load_fn(ckpt_path, device=device)
    model.eval()

    ds     = SeqDataset(seqs)
    loader = DataLoader(ds, batch_size=batch_size,
                        collate_fn=collate_seqs, shuffle=False)

    all_structs = []
    with torch.inference_mode():
        for seq_batch in loader:
            # forward returns structures
            pred_structs, *_ = model(seq_batch)

            if isinstance(pred_structs, torch.Tensor):
                for t in pred_structs.cpu():
                    all_structs.append(t)
            else:
                all_structs.extend(pred_structs)

    # free GPU memory before loading the next model
    del model
    torch.cuda.empty_cache()

    return all_structs

if __name__ == '__main__':
    seqs = ["MLQAVIKIFSFALLIGTATALECHSCHSKQIGDDCMWKNSSSKWSTTTCAAGETVCYIKLTRVSSNATTGIAERGCGLNMNLCKDWLEPKEDSKAAISPRIALDCYVCNTQKCNERDALGVSTRMSLNIFLLLLSLLPFVKKINF",
            "MKLPPFMHLIFRLLIVSTYFTYEECLKCYSCDGPTDCAHPRQQLCPQNNECFTVAQNYDTKLNGLRKGCAPTCDRVNIEGMLCRTCKFELCNGETGLGKAFEKPTILPPQRPFGMCF",
            "MVVYYTAICMMVSGLLSTGEWKYWSYKYSFLVVALKCFDCAQCPENPKEGEVPVKQNCKTCLTSRTYSGGKLRALSITCSPVCPPTKGVDAGALVKVKVKCCYRDLCTGHAVPRIPSSTITSICLIVIHLCFAANK",
            "MCLRTTISALTLFFVIFLTAIQKGNAVRCYQCGSAQDPKGQDNCGAYRKFDKTQHIAVECNTEESHAPGSFCMKVTQQSPKGFIWDGRWRQVIRRCASVADTGVTGVCNWGVYENGVYWEECYCSEDECNSSNVTKTSVVMFFVSLGTILWSYRIF",
            "MPVIVTLIIFEFLLFLYGETLYCYNCASSLPSNISKDAQRAFKTVLYSNFMVPPVDRLCINSEDIAFKTVKQINCLPDDQCIKITVRQKDLQFVMRGCQKFIYRDKVIDNKMECHHIHSPSICHCNDNLCNNALIFSFHNYICFLVFFIILLSSV"]
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",
                   required=True,
                   help="Path to your .pt checkpoint")
    args = p.parse_args()
    print(f"before infer struct: {args.checkpoint}")
    final_structs = infer_structures(FinalModel.load_final(), args.checkpoint, seqs, batch_size=2)
    #bio2_structs = infer_structures(TFold, "path/to/bio2.pt", seqs, batch_size=2, bio2token=True)
    #foldtoken_structs = infer_structures(TFold, "path/to/fold.pt", seqs, batch_size=2, bio2token=False)
    print(final_structs)

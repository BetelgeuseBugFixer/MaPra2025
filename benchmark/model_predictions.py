import os

import numpy as np
import torch
from biotite.structure import lddt, rmsd, tm_score
from biotite.structure.filter import _filter_atom_names
from biotite.structure.io.pdb import PDBFile
from torch.utils.data import Dataset, DataLoader

from models.bio2token.data.utils.utils import pdb_2_dict
from models.end_to_end.whole_model import FinalModel, TFold
import argparse

from models.model_utils import SmoothLDDTLoss
from utils.generate_new_data import BACKBONE_ATOMS, get_pdb_dict, get_pid_from_file_name, MAX_LENGTH

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


def infer_structures(model: torch.nn.Module, seqs, batch_size=16):
    """
    model: an already-loaded model instance (e.g. FinalModel)
    seqs: List[str]
    batch_size: int
    returns: List of predicted structures (one per sequence)
    """
    model.eval()

    ds = SeqDataset(seqs)
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

    # free GPU memory if needed
    del model
    torch.cuda.empty_cache()

    return all_structs


def load_prot_from_pdb(pdb_file):
    file = PDBFile.read(pdb_file)
    array_stack = file.get_structure(model=1)
    return array_stack[_filter_atom_names(array_stack, BACKBONE_ATOMS)]


def get_scores(gt_pdb, pred):
    gt_protein = load_prot_from_pdb(gt_pdb)
    # check if we have a path as string or a pred as tensor
    if isinstance(pred, str):
        pred_protein = load_prot_from_pdb(pred)
    else:
        pred_protein = pred
    lddt_score = float(lddt(gt_protein, pred_protein))
    rmsd_score = float(rmsd(gt_protein, pred_protein))
    # get indices for tm score
    all_atoms = len(gt_protein)
    indices = np.arange(all_atoms)
    tm_score_score = tm_score(gt_protein, pred_protein, indices, indices, all_atoms)
    return lddt_score, rmsd_score, tm_score_score


if __name__ == '__main__':
    # seqs = [
    #     "MLQAVIKIFSFALLIGTATALECHSCHSKQIGDDCMWKNSSSKWSTTTCAAGETVCYIKLTRVSSNATTGIAERGCGLNMNLCKDWLEPKEDSKAAISPRIALDCYVCNTQKCNERDALGVSTRMSLNIFLLLLSLLPFVKKINF",
    #     "MKLPPFMHLIFRLLIVSTYFTYEECLKCYSCDGPTDCAHPRQQLCPQNNECFTVAQNYDTKLNGLRKGCAPTCDRVNIEGMLCRTCKFELCNGETGLGKAFEKPTILPPQRPFGMCF",
    #     "MVVYYTAICMMVSGLLSTGEWKYWSYKYSFLVVALKCFDCAQCPENPKEGEVPVKQNCKTCLTSRTYSGGKLRALSITCSPVCPPTKGVDAGALVKVKVKCCYRDLCTGHAVPRIPSSTITSICLIVIHLCFAANK",
    #     "MCLRTTISALTLFFVIFLTAIQKGNAVRCYQCGSAQDPKGQDNCGAYRKFDKTQHIAVECNTEESHAPGSFCMKVTQQSPKGFIWDGRWRQVIRRCASVADTGVTGVCNWGVYENGVYWEECYCSEDECNSSNVTKTSVVMFFVSLGTILWSYRIF",
    #     "MPVIVTLIIFEFLLFLYGETLYCYNCASSLPSNISKDAQRAFKTVLYSNFMVPPVDRLCINSEDIAFKTVKQINCLPDDQCIKITVRQKDLQFVMRGCQKFIYRDKVIDNKMECHHIHSPSICHCNDNLCNNALIFSFHNYICFLVFFIILLSSV"]
    # get singletons to filter
    with open("/mnt/data/large/prostt5_IDs/afdb50best_foldseek_clu_singleton.ids") as f:
        singleton_ids = {line.strip().split("-")[1] for line in f if "-" in line}
    in_dir = "/data/large/zip_file/final_data_PDB/val/val_pdb"
    pdb_names = [pdb for pdb in os.listdir(in_dir) if pdb.endswith("pdb")]
    # filter singletons
    pdb_names = [pdb for pdb in pdb_names if not get_pid_from_file_name(pdb) in singleton_ids]
    pdb_paths = [os.path.join(in_dir, pdb) for pdb in pdb_names]
    pdb_dicts = [pdb_2_dict(pdb_path) for pdb_path in pdb_paths]
    # filter for length
    allowed_indices = [i for i, pdb_dict in enumerate(pdb_dicts) if len(pdb_dict["seq"]) >= MAX_LENGTH]
    pdb_dicts = [pdb_dicts[i] for i in allowed_indices]
    pdb_paths = [pdb_paths[i] for i in allowed_indices]
    seqs = [pdb_dict["seq"] for pdb_dict in pdb_dicts]
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",
                   required=True,
                   help="Path to your .pt checkpoint")
    args = p.parse_args()

    print(f"Loading model from checkpoint: {args.checkpoint}")
    model = FinalModel.load_final(args.checkpoint, device=device)

    final_structs = infer_structures(model, seqs, batch_size=64)
    print(final_structs[0])
    for final_struct, pdb_path in zip(final_structs, pdb_paths):
        print(get_scores(pdb_path, final_struct))
        break

    smooth_lddt = SmoothLDDTLoss()
    for final_struct, pdb_dict in zip(final_structs, pdb_dicts):
        gt = torch.from_numpy(pdb_dict["coords_groundtruth"]).unsqueeze(0)
        pd = final_struct.unsqueeze(0)
        print(1 - smooth_lddt(gt, pd).item())
        break
    # bio2_structs = infer_structures(TFold, "path/to/bio2.pt", seqs, batch_size=2, bio2token=True)
    # foldtoken_structs = infer_structures(TFold, "path/to/fold.pt", seqs, batch_size=2, bio2token=False)

import os
import csv
import traceback

import numpy as np
import torch
import time

from biotite.structure import lddt, rmsd, tm_score, superimpose_structural_homologs
from biotite.structure.superimpose import superimpose
from biotite.structure.filter import _filter_atom_names
from biotite.structure.io.pdb import PDBFile
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
from torch.utils.data import Dataset, DataLoader

from models.bio2token.data.utils.utils import pdb_2_dict
from models.collab_fold.esmfold import EsmFold
from models.end_to_end.whole_model import FinalModel, TFold, FinalFinalModel
import argparse

from models.model_utils import SmoothLDDTLoss
from utils.generate_new_data import BACKBONE_ATOMS, get_pid_from_file_name, filter_pdb_dict

MAX_LENGTH = 780

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


# predict structures
def infer_structures(model: torch.nn.Module, seqs, batch_size=16):
    model.eval()
    ds = SeqDataset(seqs)
    loader = DataLoader(ds, batch_size=batch_size, collate_fn=collate_seqs, shuffle=False)
    all_structs = []

    start_time = time.time()  # Startzeit
    with torch.inference_mode():
        for seq_batch in loader:
            pred_structs, *_ = model(seq_batch)
            for i in range(pred_structs.shape[0]):
                all_structs.append(pred_structs[i].cpu())
    end_time = time.time()
    print(f"[inference] Time needed for {len(seqs)} sequences: {end_time - start_time:.2f} seconds")

    del model
    torch.cuda.empty_cache()
    return all_structs


def load_prot_from_pdb(pdb_file):
    file = PDBFile.read(pdb_file)
    array_stack = file.get_structure(model=1)
    return array_stack[_filter_atom_names(array_stack, BACKBONE_ATOMS)]


def get_scores(gt_pdb, pred):
    gt_protein = load_prot_from_pdb(gt_pdb)

    if isinstance(pred, str):
        pred_protein = load_prot_from_pdb(pred)
    else:
        pred_protein = gt_protein.copy()
        pred_protein.coord = pred

    # calculate lddt before aligning structures for rmsd and tm score
    lddt_score = float(lddt(gt_protein, pred_protein))

    # Superimpose predicted onto ground-truth
    superimposed, _, ref_indices, sub_indices = superimpose_structural_homologs(
        gt_protein, pred_protein, max_iterations=1
    )

    # rmsd
    rmsd_score = float(rmsd(gt_protein, superimposed))

    # tm score
    tm_score_score = tm_score(gt_protein, superimposed, ref_indices, sub_indices)

    return lddt_score, rmsd_score, tm_score_score


def get_smooth_lddt(lddt_loss_module, prediction, pdb_dict):
    filtered = filter_pdb_dict(pdb_dict)
    gt = torch.tensor(filtered["coords_groundtruth"]).unsqueeze(0).to(device)
    pd = prediction.unsqueeze(0).to(device)
    B, L, _ = gt.shape
    is_dna = torch.zeros((B, L), dtype=torch.bool, device=device)
    is_rna = torch.zeros((B, L), dtype=torch.bool, device=device)
    mask = torch.ones((B, L), dtype=torch.bool, device=device)
    lddt_score = 1 - lddt_loss_module(gt, pd, is_dna, is_rna, mask).item()
    return lddt_score


def compute_and_save_scores_for_model(checkpoint_path, model, seqs, pdb_paths, pdb_dicts, batch_size=64,
                                      dataset_name="", given_base=None):
    if given_base is None:
        base = os.path.splitext(os.path.basename(checkpoint_path))[0]
    else:
        base = given_base
    if dataset_name:
        base = f"{base}_{dataset_name}"
    cwd = os.getcwd()
    out_dir = os.path.join(cwd, base)

    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, f"{base}_scores.csv")
    plot_path = os.path.join(out_dir, f"{base}_smooth_lddt.png")

    # predict
    final_structs = infer_structures(model, seqs, batch_size=batch_size)
    final_structs = [struct[:len(seq) * 4] for struct, seq in zip(final_structs, seqs)]

    # Scores berechnen – skippe problematische Einträge
    actual_lddts, rmsd_scores, tm_scores, smooth_lddts = [], [], [], []
    kept_pdb_paths, kept_pdb_dicts = [], []

    smooth_loss = SmoothLDDTLoss().to(device)

    for i, (struct, pdb, pdb_dict) in enumerate(zip(final_structs, pdb_paths, pdb_dicts)):
        try:
            l, r, t = get_scores(pdb, struct.numpy())
            s = get_smooth_lddt(smooth_loss, struct, pdb_dict)
        except Exception as e:
            print(f"[SKIPPED] PDB {pdb} (index {i}) caused error: {e}")
            continue

        # Falls alles erfolgreich war → speichern
        actual_lddts.append(l)
        rmsd_scores.append(r)
        tm_scores.append(t)
        smooth_lddts.append(s)
        kept_pdb_paths.append(pdb)

    # Save results
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["pdb_path", "lddt", "rmsd", "tm_score", "smooth_lddt"])
        for pdb, l, r, t, s in zip(kept_pdb_paths, actual_lddts, rmsd_scores, tm_scores, smooth_lddts):
            writer.writerow([pdb, l, r, t, s])
    print(f"Scores saved to {csv_path}")

    if len(actual_lddts) > 0:
        plot_smooth_lddt(actual_lddts, smooth_lddts, out_path=plot_path)
        print(f"Scatterplot saved to {plot_path}")
    else:
        print("[WARN] No valid entries to plot.")


def plot_smooth_lddt(lddts, smooth_lddts, out_path="smooth_lddt.png"):
    lddts = np.array(lddts)
    smooth_lddts = np.array(smooth_lddts)
    r, _ = pearsonr(lddts, smooth_lddts)
    plt.figure(figsize=(6, 6))
    plt.scatter(lddts, smooth_lddts, alpha=0.7, edgecolors='k', color='steelblue')
    plt.xlabel("True lDDT")
    plt.ylabel("Smoothed lDDT (AlphaFold)")
    plt.title(f"Scatter Plot with Pearson r = {r:.3f}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def prepare_data(in_dir, singleton_ids=None, casp=False):
    pdb_names = [p for p in os.listdir(in_dir) if p.endswith("pdb")]
    if singleton_ids:
        pdb_names = [p for p in pdb_names if get_pid_from_file_name(p) not in singleton_ids]
    pdb_paths = [os.path.join(in_dir, p) for p in pdb_names]
    pdb_dicts = [pdb_2_dict(p) for p in pdb_paths]

    if casp:
        allowed = [i for i, d in enumerate(pdb_dicts) if
                   len(d["seq"]) * 4 == d["atom_length"] and len(d["seq"]) < MAX_LENGTH]
    else:
        allowed = [i for i, d in enumerate(pdb_dicts) if len(d["seq"]) < MAX_LENGTH]

    pdb_paths = [pdb_paths[i] for i in allowed]
    pdb_dicts = [pdb_dicts[i] for i in allowed]

    seqs = [d["seq"] for d in pdb_dicts]
    print(f"[prepare_data] #paths: {len(pdb_paths)} | #dicts: {len(pdb_dicts)} | #seqs: {len(seqs)}")
    return pdb_paths, pdb_dicts, seqs


if __name__ == '__main__':
    # singletons
    with open("/mnt/data/large/prostt5_IDs/afdb50best_foldseek_clu_singleton.ids") as f:
        singleton_ids = {line.strip().split("-")[1] for line in f if "-" in line}

    # test data prep
    in_dir = "/mnt/data/large/zip_file/final_data_PDB/test/test_pdb"

    pdb_paths, pdb_dicts, seqs = prepare_data(in_dir=in_dir, singleton_ids=singleton_ids, casp=False)

    # casp15 data prep
    print(f"now in: {os.getcwd()}")
    casp_dir = "tokenizer_benchmark/casps/casp15_backbone"

    pdb_casp, casp_dicts, seqs_casp = prepare_data(casp_dir, casp=True)

    # parser
    p = argparse.ArgumentParser()
    p.add_argument("--final", nargs="+", default=[], help="Path(s) to FinalModel checkpoint(s)")
    p.add_argument("--final_final", nargs="+", default=[], help="Path(s) to FinalFinalModel checkpoint(s)")
    p.add_argument("--prostt5", nargs="+", default=[], help="Path(s) to prostt5 checkpoint(s)")
    p.add_argument("--bio2token", nargs="+", default=[], help="Path(s) to bio2token token checkpoint(s)")
    p.add_argument("--foldtoken", nargs="+", default=[], help="Path(s) to foldtoken token checkpoint(s)")
    p.add_argument("--esm", action="store_true", help="run esm model")
    args = p.parse_args()

    # Zähler vorbereiten
    final_count = 1
    final_final_count = 1
    prostt5_count = 1
    bio2token_count = 1
    foldtoken_count = 1

    # final models
    for ckpt in args.final:
        print(f"Processing FinalModel checkpoint: {ckpt}")
        model = FinalModel.load_old_final(ckpt, device=device)
        base_name = f"final_{final_count}"
        compute_and_save_scores_for_model(ckpt, model, seqs, pdb_paths, pdb_dicts, batch_size=32, dataset_name="test",
                                          given_base=base_name)
        compute_and_save_scores_for_model(ckpt, model, seqs_casp, pdb_casp, casp_dicts, batch_size=32,
                                          dataset_name="casp", given_base=base_name)
        final_count += 1

    # final final models, whole decoder
    for ckpt in args.final_final:
        print(f"Processing FinalFinalModel checkpoint: {ckpt}")
        model = FinalModel.load_final(ckpt, device=device)
        base_name = f"final_final_{final_final_count}"
        compute_and_save_scores_for_model(ckpt, model, seqs, pdb_paths, pdb_dicts, batch_size=32, dataset_name="test",
                                          given_base=base_name)
        compute_and_save_scores_for_model(ckpt, model, seqs_casp, pdb_casp, casp_dicts, batch_size=32,
                                          dataset_name="casp", given_base=base_name)
        final_final_count += 1

    # prostt5 models, whole decoder FinalFinalModel!!!
    for ckpt in args.prostt5:
        print(f"Processing Prostt5 checkpoint: {ckpt}")
        model = FinalFinalModel.load_final_final(ckpt, device=device)
        base_name = f"prostt5_{prostt5_count}"
        compute_and_save_scores_for_model(ckpt, model, seqs, pdb_paths, pdb_dicts, batch_size=32,
                                          dataset_name="test",
                                          given_base=base_name)
        compute_and_save_scores_for_model(ckpt, model, seqs_casp, pdb_casp, casp_dicts, batch_size=32,
                                          dataset_name="casp", given_base=base_name)
        prostt5_count += 1

    # bio2token models
    for ckpt in args.bio2token:
        print(f"Processing TFold (bio2token) checkpoint: {ckpt}")
        ckpt_data = torch.load(ckpt, map_location=device)
        model_args = ckpt_data["model_args"]
        model_args.pop("decoder", None)
        model_args["bio2token"] = True
        model_args["device"] = device
        model = TFold(**model_args)
        model.load_state_dict(ckpt_data["state_dict"])

        base_name = f"bio2token_{bio2token_count}"
        compute_and_save_scores_for_model(ckpt, model, seqs, pdb_paths, pdb_dicts, batch_size=32, dataset_name="test",
                                          given_base=base_name)
        try:
            compute_and_save_scores_for_model(ckpt, model, seqs_casp, pdb_casp, casp_dicts, batch_size=32,
                                              dataset_name="casp", given_base=base_name)
        except Exception as e:
            print("casp fail")
        bio2token_count += 1

    # foldtoken models
    for ckpt in args.foldtoken:
        print(f"Processing TFold (foldtoken) checkpoint: {ckpt}")
        ckpt_data = torch.load(ckpt, map_location=device)
        model_args = ckpt_data["model_args"]
        model_args.pop("decoder", None)
        model_args["bio2token"] = False
        model_args["device"] = device
        model = TFold(**model_args)
        model.load_state_dict(ckpt_data["state_dict"])

        base_name = f"foldtoken_{foldtoken_count}"
        compute_and_save_scores_for_model(ckpt, model, seqs, pdb_paths, pdb_dicts, batch_size=32, dataset_name="test",
                                          given_base=base_name)
        try:
            compute_and_save_scores_for_model(ckpt, model, seqs_casp, pdb_casp, casp_dicts, batch_size=32,
                                              dataset_name="casp", given_base=base_name)
        except Exception as e:
            print("casp fail")
        foldtoken_count += 1

    # ESMFold
    if args.esm:
        model = EsmFold(device)
        try:
            compute_and_save_scores_for_model("", model, seqs_casp, pdb_casp, casp_dicts, batch_size=1,
                                              dataset_name="casp", given_base="ESMFold")
        except Exception as e:
            print("casp fail")
            traceback.print_exc()
        # compute_and_save_scores_for_model("", model, seqs, pdb_paths, pdb_dicts, batch_size=1, dataset_name="test",given_base="ESMFold")

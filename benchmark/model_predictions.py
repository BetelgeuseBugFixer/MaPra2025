import os
import csv
import traceback

import numpy as np
import torch
import time

from biotite.structure import lddt, rmsd, tm_score, filter_amino_acids, AtomArray, superimpose_structural_homologs
from biotite.structure.filter import _filter_atom_names
from biotite.structure.io.pdb import PDBFile
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
from torch.utils.data import Dataset, DataLoader

from models.bio2token.data.utils.utils import pdb_2_dict
from models.collab_fold.esmfold import EsmFold
from models.end_to_end.whole_model import FinalModel, TFold, FinalFinalModel
from models.model_utils import model_prediction_to_atom_array
import argparse

from models.losses import SmoothLDDTLoss
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
def infer_structures(model: torch.nn.Module, seqs, batch_size=16,c_alpha_only=False):
    """
    Runs the model and converts predictions to AtomArray via model_prediction_to_atom_array.
    Returns: List[AtomArray] aligned as N, CA, C, O per residue.
    """
    model.eval()
    ds = SeqDataset(seqs)
    loader = DataLoader(ds, batch_size=batch_size, collate_fn=collate_seqs, shuffle=False)
    all_structs = []

    start_time = time.time()
    with torch.inference_mode():
        for seq_batch in loader:
            preds, final_mask, *rest = model(seq_batch)
            batch_atom_arrays = model_prediction_to_atom_array(
                sequences=seq_batch,
                model_prediction=preds,  # (B, L*4, 3) or similar
                final_mask=final_mask,  # (B, L*4) boolean
                only_c_alpha=c_alpha_only  # keep backbone N,CA,C,O
            )
            all_structs.extend(batch_atom_arrays)

    end_time = time.time()
    print(f"[inference] Time needed for {len(seqs)} sequences: {end_time - start_time:.2f} seconds")

    del model
    torch.cuda.empty_cache()
    return all_structs


def load_prot_from_pdb(pdb_file, c_alpha_only=False):
    file = PDBFile.read(pdb_file)
    array_stack = file.get_structure(model=1)
    if c_alpha_only:
        return array_stack[_filter_atom_names(array_stack, ["CA"])]
    return array_stack[_filter_atom_names(array_stack, BACKBONE_ATOMS)]


# ---  Kabsch + helpers ----------------------------------------------------
def _kabsch(P, Q):
    Pc, Qc = P.mean(axis=0), Q.mean(axis=0)
    P0, Q0 = P - Pc, Q - Qc
    H = Q0.T @ P0
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = Pc - Qc @ R.T
    return R, t


def _ca_index_map(a: AtomArray):
    mask = filter_amino_acids(a) & (a.atom_name == "CA")
    keys = list(zip(a.chain_id[mask], a.res_id[mask]))
    idxs = np.nonzero(mask)[0]
    return {k: i for k, i in zip(keys, idxs)}


def _apply_rt(coords: np.ndarray, R: np.ndarray, t: np.ndarray):
    return (coords @ R.T + t).astype(np.float32)


# --- get_scores ----------------------------------------------------
def get_scores(gt_pdb, pred, method: str = "biotite", c_alpha_only=False):
    """
    pred can be:
      - str (pdb path),
      - AtomArray,
      - numpy coords shaped (L*4, 3).
    """
    gt_protein = load_prot_from_pdb(gt_pdb, c_alpha_only)

    if isinstance(pred, str):
        pred_protein = load_prot_from_pdb(pred)
    elif isinstance(pred, AtomArray):
        pred_protein = pred
    else:
        # assume coords
        pred_protein = gt_protein.copy()
        pred_protein.coord = np.asarray(pred, dtype=np.float32)

    lddt_score = np.nan
    rmsd_score = np.nan
    tm_score_score = np.nan

    # 1) lDDT before alignment
    try:
        lddt_score = float(lddt(gt_protein, pred_protein))
    except Exception as e:
        print(f"lddt caused error: {e}")

    # 2) alignment + RMSD/TM
    match method:
        case "biotite":
            try:
                superimposed, _, ref_idx, sub_idx = superimpose_structural_homologs(
                    gt_protein, pred_protein, max_iterations=1
                )
                rmsd_score = float(rmsd(gt_protein[ref_idx], superimposed[sub_idx]))
                tm_score_score = float(tm_score(gt_protein, superimposed, ref_idx, sub_idx))
            except Exception as e:
                print(f"biotite superimpose/metrics caused error: {e}")

        case "kabsch":
            try:
                gt_map = _ca_index_map(gt_protein)
                pr_map = _ca_index_map(pred_protein)
                common = [k for k in gt_map if k in pr_map]

                use_global = False
                if len(common) >= 3:
                    gt_idx = np.array([gt_map[k] for k in common], dtype=int)
                    pr_idx = np.array([pr_map[k] for k in common], dtype=int)
                    P = np.asarray(gt_protein.coord[gt_idx], dtype=np.float64)
                    Q = np.asarray(pred_protein.coord[pr_idx], dtype=np.float64)
                else:
                    same_layout = (
                            len(gt_protein) == len(pred_protein) and
                            np.array_equal(gt_protein.atom_name, pred_protein.atom_name) and
                            np.array_equal(gt_protein.chain_id, pred_protein.chain_id) and
                            np.array_equal(gt_protein.res_id, pred_protein.res_id)
                    )
                    if not same_layout:
                        print("align: not enough common Cα anchors and no global matchable layout")
                        return lddt_score, rmsd_score, tm_score_score
                    use_global = True
                    P = np.asarray(gt_protein.coord, dtype=np.float64)
                    Q = np.asarray(pred_protein.coord, dtype=np.float64)
                    all_idx = np.arange(len(gt_protein), dtype=int)

                R, t = _kabsch(P, Q)
                pred_aligned = pred_protein.copy()
                pred_aligned.coord = _apply_rt(pred_protein.coord, R, t)

                if use_global:
                    rmsd_score = float(rmsd(gt_protein, pred_aligned))
                    tm_score_score = float(tm_score(gt_protein, pred_aligned, all_idx, all_idx))
                else:
                    rmsd_score = float(rmsd(gt_protein[gt_idx], pred_aligned[pr_idx]))
                    tm_score_score = float(tm_score(gt_protein, pred_aligned, gt_idx, pr_idx))
            except Exception as e:
                print(f"kabsch align/metrics caused error: {e}")

    return lddt_score, rmsd_score, tm_score_score


def get_smooth_lddt(lddt_loss_module, prediction_atom_array: AtomArray, pdb_dict,c_alpha_only=False):
    filtered = filter_pdb_dict(pdb_dict)
    gt = torch.tensor(filtered["coords_groundtruth"]).unsqueeze(0).to(device)  # [1, L*4, 3]
    if c_alpha_only:
        gt = gt[:, 1::4, :]
    pd = torch.from_numpy(prediction_atom_array.coord).float().unsqueeze(0).to(device)  # [1, L*4, 3]
    B, L, _ = gt.shape
    mask = torch.ones((B, L), dtype=torch.bool, device=device)
    lddt_score = 1 - lddt_loss_module(gt, pd, mask).item()
    return lddt_score


def compute_and_save_scores_for_model(checkpoint_path, model, seqs, pdb_paths, pdb_dicts, batch_size=64,
                                      dataset_name="", given_base=None):
    base = given_base if given_base is not None else os.path.splitext(os.path.basename(checkpoint_path))[0]
    if dataset_name:
        base = f"{base}_{dataset_name}"
    out_dir = os.path.join(os.getcwd(), base)
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, f"{base}_scores.csv")
    plot_path = os.path.join(out_dir, f"{base}_smooth_lddt.png")

    # determine whether this is a C-alpha only model
    c_alpha_only=False
    if hasattr(model,"c_alpha_only") and model.c_alpha_only:
        c_alpha_only = True

    # predict → AtomArrays (N,CA,C,O or CA per residue), order already matches writer in model_utils
    final_structs = infer_structures(model, seqs, batch_size=batch_size,c_alpha_only=c_alpha_only)

    actual_lddts, rmsd_scores, tm_scores, smooth_lddts = [], [], [], []
    kept_pdb_paths = []

    smooth_loss = SmoothLDDTLoss().to(device)

    for i, (struct, pdb, pdb_dict) in enumerate(zip(final_structs, pdb_paths, pdb_dicts)):
        try:
            l, r, t = get_scores(pdb, struct,c_alpha_only=c_alpha_only)  # struct is AtomArray now
            s = get_smooth_lddt(smooth_loss, struct, pdb_dict,c_alpha_only=c_alpha_only)
        except Exception as e:
            print(f"[SKIPPED] PDB {pdb} (index {i}) caused error: {e}")
            continue

        actual_lddts.append(l)
        rmsd_scores.append(r)
        tm_scores.append(t)
        smooth_lddts.append(s)
        kept_pdb_paths.append(pdb)

    # Save results
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["pdb_path", "lddt", "rmsd", "tm_score", "smooth_lddt"])
        writer.writerows(zip(kept_pdb_paths, actual_lddts, rmsd_scores, tm_scores, smooth_lddts))
    print(f"Scores saved to {csv_path}")

    if len(actual_lddts) > 0:
        plot_smooth_lddt(actual_lddts, smooth_lddts, out_path=plot_path)
        print(f"Scatterplot saved to {plot_path}")
    else:
        print("[WARN] No valid entries to plot.")

    return final_structs


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
    p.add_argument("--0_final", nargs="+", default=[], help="Path(s) to FinalModel checkpoint(s)")
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

    # 0_final models
    for ckpt in args.final:
        print(f"Processing FinalModel checkpoint: {ckpt}")
        model = FinalModel.load_old_final(ckpt, device=device)
        base_name = f"final_{final_count}"
        compute_and_save_scores_for_model(ckpt, model, seqs, pdb_paths, pdb_dicts, batch_size=32, dataset_name="test",
                                          given_base=base_name)
        compute_and_save_scores_for_model(ckpt, model, seqs_casp, pdb_casp, casp_dicts, batch_size=32,
                                          dataset_name="casp", given_base=base_name)
        final_count += 1

    # 0_final 0_final models, whole decoder
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

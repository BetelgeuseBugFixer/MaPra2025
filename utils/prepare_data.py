import time
import os
import tarfile
import pickle
import json
import tempfile
import torch


from pathlib import Path
from models.foldtoken_decoder.foldtoken import FoldToken
from biotite.structure.io.pdb import PDBFile
from biotite.structure.filter import _filter_atom_names

from models.bio2token.data.utils.utils import pdb_2_dict, uniform_dataframe, compute_masks
from models.bio2token.models.autoencoder import AutoencoderConfig, Autoencoder
from models.bio2token.utils.configs import utilsyaml_to_dict, pi_instantiate

from hydra_zen import load_from_yaml, builds, instantiate


DEVICE = "cuda"

# FoldToken model for vq_ids
foldtoken_model = FoldToken(device=DEVICE)

# Bio2Token model for encoding + indices
model_configs = load_from_yaml("models/bio2token/files/model.yaml")["model"]
model_config = pi_instantiate(AutoencoderConfig, model_configs)
bio2token_model = Autoencoder(model_config)
state_dict = torch.load("models/bio2token/files/epoch=0243-val_loss_epoch=0.71-best-checkpoint.ckpt")["state_dict"]
state_dict_bis = {k.replace("model.", ""): v for k, v in state_dict.items()}
bio2token_model.load_state_dict(state_dict_bis)
bio2token_model.to(DEVICE)

print("Initialized both FoldToken and Bio2Token models.")


SINGLETON_ID_PATH = "/mnt/data/large/prostt5_IDs/afdb50best_foldseek_clu_singleton.ids"
with open(SINGLETON_ID_PATH, "r") as f:
    singleton_ids = set(line.strip().split("-")[1] for line in f if "-" in line)

base_output = Path("/mnt/data/large/subset")
base_output.mkdir(parents=True, exist_ok=True)

three_to_one_dict = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D',
    'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G',
    'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
    'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S',
    'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}

def get_seq_from_lines(lines):
    seq = ""
    chain_id = None
    for line in lines:
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            current_chain = line[21]
            if chain_id is None:
                chain_id = current_chain
            elif current_chain != chain_id:
                return None
            res = line[17:20].strip()
            if res not in three_to_one_dict:
                return None
            seq += three_to_one_dict[res]
        elif line.startswith("ENDMDL"):
            break
    return seq

def load_prot_from_pdb(pdb_file):
    file = PDBFile.read(pdb_file)
    array_stack = file.get_structure(model=1)
    return array_stack[_filter_atom_names(array_stack, ["N", "CA", "C", "O"])]















for split in ["val", "test", "train"]:
    print("=" * 60)
    print(f"[{split.upper()}] Starting processing")
    print(f"[{split.upper()}] Output dir: {base_output / split}")
    if split == "train":
        print(f"[{split.upper()}] Input archive: /mnt/data/large/zip_file/final_data_PDB/train/rostlab_subset.tar")
    else:
        print(f"[{split.upper()}] Input folder: /mnt/data/large/zip_file/final_data_PDB/{split}/{split}_pdb")
    print(f"[{split.upper()}] Total singleton IDs loaded: {len(singleton_ids)}")
    print("=" * 60)

    output_dir = base_output / split
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / "proteins.jsonl"
    pickle_path = output_dir / "proteins.pkl"

    json_entries = []
    protein_structures = []

    if split == "train":
        tar_path = Path("/mnt/data/large/zip_file/final_data_PDB/train/rostlab_subset.tar")
        with tarfile.open(tar_path, "r") as tar:
            processed = 0
            skipped_singletons = 0
            total_start = time.time()
            batch_start = time.time()

            for member in tar:
                if not member.name.endswith(".pdb"):
                    continue
                if processed >= 100_000:
                    break

                mid = member.name.split("-")[1]
                if mid in singleton_ids:
                    skipped_singletons += 1
                    continue

                try:
                    f = tar.extractfile(member)
                    if not f:
                        continue
                    lines = f.read().decode("utf-8").splitlines()
                    seq = get_seq_from_lines(lines)
                    if not seq:
                        continue

                    with tempfile.NamedTemporaryFile("w+", suffix=".pdb", delete=False) as tmp:
                        tmp.write("\n".join(lines))
                        tmp.flush()
                        tmp_path = tmp.name

                    try:
                        vq_ids = foldtoken_model.encode_pdb(tmp_path)

                        pdb_dict = pdb_2_dict(tmp_path, None)
                        structure_b2t, unknown_structure, residue_name, residue_ids, token_class, atom_names_reordered = uniform_dataframe(
                            pdb_dict["seq"],
                            pdb_dict["res_types"],
                            pdb_dict["coords_groundtruth"],
                            pdb_dict["atom_names"],
                            pdb_dict["res_atom_start"],
                            pdb_dict["res_atom_end"],
                        )
                        batch = {
                            "structure": torch.tensor(structure_b2t).float(),
                            "unknown_structure": torch.tensor(unknown_structure).bool(),
                            "residue_ids": torch.tensor(residue_ids).long(),
                            "token_class": torch.tensor(token_class).long(),
                        }
                        batch = {k: v[~batch["unknown_structure"]] for k, v in batch.items()}
                        batch = compute_masks(batch, structure_track=True)
                        batch = {k: v[None].to(DEVICE) for k, v in batch.items()}
                        batch = bio2token_model.encoder(batch)

                        encoding = batch["encoding"].squeeze(0).cpu().tolist()
                        indices = batch["indices"].squeeze(0).cpu().tolist()

                        structure = load_prot_from_pdb(tmp_path)  # original PDB structure
                        protein_structures.append(structure)
                    finally:
                        os.remove(tmp_path)

                    protein_id = mid
                    json_entries.append({
                        "id": protein_id,
                        "sequence": seq,
                        "vq_ids": vq_ids.tolist(),
                        "bio2tokens": indices,
                        "bio2token_encoding": encoding
                    })

                    processed += 1
                    if processed % 1000 == 0:
                        elapsed = time.time() - batch_start
                        print(f"[train] {processed} done – Time for last 1000: {elapsed:.2f}s")
                        batch_start = time.time()

                except Exception as e:
                    print(f"[train] Failed: {member.name}, {e}")

            print(f"[train] Finished processing {processed} proteins in {time.time() - total_start:.2f}s")
            print(f"[train] Skipped {skipped_singletons} singleton entries.")

    else:
        pdb_dir = Path(f"/mnt/data/large/zip_file/final_data_PDB/{split}") / f"{split}_pdb"
        processed = 0
        skipped_singletons = 0
        total_start = time.time()

        for pdb_file in pdb_dir.glob("*.pdb"):
            parts = pdb_file.stem.split("-")
            if len(parts) < 2:
                print(f"[{split}] Skipped invalid filename: {pdb_file.name}")
                continue
            protein_id = parts[1]

            if protein_id in singleton_ids:
                skipped_singletons += 1
                continue

            try:
                with open(pdb_file, "r") as f:
                    lines = f.readlines()
                seq = get_seq_from_lines(lines)
                if not seq:
                    continue

                vq_ids = foldtoken_model.encode_pdb(str(pdb_file))

                pdb_dict = pdb_2_dict(str(pdb_file), None)
                structure_b2t, unknown_structure, residue_name, residue_ids, token_class, atom_names_reordered = uniform_dataframe(
                    pdb_dict["seq"],
                    pdb_dict["res_types"],
                    pdb_dict["coords_groundtruth"],
                    pdb_dict["atom_names"],
                    pdb_dict["res_atom_start"],
                    pdb_dict["res_atom_end"],
                )
                batch = {
                    "structure": torch.tensor(structure_b2t).float(),
                    "unknown_structure": torch.tensor(unknown_structure).bool(),
                    "residue_ids": torch.tensor(residue_ids).long(),
                    "token_class": torch.tensor(token_class).long(),
                }
                batch = {k: v[~batch["unknown_structure"]] for k, v in batch.items()}
                batch = compute_masks(batch, structure_track=True)
                batch = {k: v[None].to(DEVICE) for k, v in batch.items()}
                batch = bio2token_model.encoder(batch)

                encoding = batch["encoding"].squeeze(0).cpu().tolist()
                indices = batch["indices"].squeeze(0).cpu().tolist()

                structure = load_prot_from_pdb(str(pdb_file))  # original PDB structure
                protein_structures.append(structure)

                json_entries.append({
                    "id": protein_id,
                    "sequence": seq,
                    "vq_ids": vq_ids.tolist(),
                    "bio2tokens": indices,
                    "bio2token_encoding": encoding
                })

                processed += 1
            except Exception as e:
                print(f"[{split}] Failed: {pdb_file.name}, {e}")
                with open(pdb_file, "r") as f:
                    print("".join(f.readlines()[:30]))

        print(f"[{split}] Finished processing {processed} proteins in {time.time() - total_start:.2f}s")
        print(f"[{split}] Skipped {skipped_singletons} singleton entries.")

    # Save JSONL
    with open(jsonl_path, "w") as f:
        for entry in json_entries:
            f.write(json.dumps(entry) + "\n")
    print(f"[{split}] Saved {len(json_entries)} json entries to {jsonl_path}")

    # Save Pickle (Biotite protein structures only)
    with open(pickle_path, "wb") as f:
        pickle.dump(protein_structures, f)
    print(f"[{split}] Saved {len(protein_structures)} Biotite protein structures to {pickle_path}")

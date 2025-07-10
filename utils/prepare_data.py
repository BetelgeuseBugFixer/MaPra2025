import os
import io

import tarfile
import pickle
import json

from pathlib import Path
from biotite.structure.io.pdb import PDBFile
from biotite.structure.filter import _filter_atom_names
from models.foldtoken_decoder.foldtoken_decoder import FoldDecoder

DEVICE = "cuda"
model = FoldDecoder(device=DEVICE)

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

def load_structure_from_lines(lines):
    file = PDBFile()
    file.read(io.StringIO("\n".join(lines)))
    array_stack = file.get_structure(model=None)
    return array_stack[_filter_atom_names(array_stack, ["N", "CA", "C", "O"])]

for split in ["val", "test", "train"]:
    output_dir = base_output / split
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / "proteins.jsonl"
    pickle_path = output_dir / "proteins.pkl"
    pdbs = []
    entries = []

## TRAIN
    if split == "train":
        tar_path = Path("/mnt/data/large/zip_file/final_data_PDB/train/rostlab_subset.tar")
        with tarfile.open(tar_path, "r") as tar:
            members = [m for m in tar.getmembers() if m.name.endswith(".pdb")]
            processed = 0
            for member in members:
                if processed >= 100_000:
                    break
                mid = member.name.split("-")[1]
                if mid in singleton_ids:
                    continue
                try:
                    f = tar.extractfile(member)
                    if not f:
                        continue
                    lines = f.read().decode("utf-8").splitlines()
                    seq = get_seq_from_lines(lines)
                    print(f"seq: {seq}\n")
                    if not seq:
                        continue
                    # tokenize
                    struct = load_structure_from_lines(lines)
                    vq_ids = model.encode_pdb(io.StringIO("\n".join(lines)))
                    print(f"vq_ids: {vq_ids}\n")

                    # append for files
                    protein_id = member.name.rsplit("/", 1)[-1].rsplit(".", 1)[0]
                    pdbs.append((protein_id, struct))
                    entries.append({"id": protein_id, "sequence": seq, "vq_ids": vq_ids.tolist()})
                    processed += 1
                except Exception as e:
                    print(f"[{split}] Failed: {member.name}, {e}")

## VAL AND TEST
    else:
        pdb_dir = Path(f"/mnt/data/large/zip_file/final_data_PDB/{split}") / f"{split}_pdb"
        for pdb_file in pdb_dir.glob("*.pdb"):
            mid = pdb_file.stem.split("-")[1]
            if mid in singleton_ids:
                continue
            try:
                with open(pdb_file, "r") as f:
                    lines = f.readlines()
                seq = get_seq_from_lines(lines)
                print(f"seq: {seq}\n")

                if not seq:
                    continue
                # tokenize
                struct = load_structure_from_lines(lines)
                vq_ids = model.encode_pdb(str(pdb_file))
                print(f"vq_ids: {vq_ids}\n")

                # append for files
                pdbs.append((pdb_file.stem, struct))
                entries.append({"id": pdb_file.stem, "sequence": seq, "vq_ids": vq_ids.tolist()})
            except Exception as e:
                print(f"[{split}] Failed: {pdb_file.name}, {e}")
                with open(pdb_file, "r") as f:
                    print("".join(f.readlines()[:30]))

    # Save .pkl
    with open(pickle_path, "wb") as f:
        pickle.dump(dict(pdbs), f)

    # Save .jsonl
    with open(jsonl_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    print(f"[{split}] Done: {len(entries)} entries written to {jsonl_path}")

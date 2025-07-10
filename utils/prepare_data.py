import time
import os

import tarfile
import pickle
import json
import tempfile

from pathlib import Path
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

for split in ["val", "test", "train"]:
    output_dir = base_output / split
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / "proteins.jsonl"
    pickle_path = output_dir / "proteins.pkl"
    entries = []

    # === TRAIN ===
    if split == "train":
        tar_path = Path("/mnt/data/large/zip_file/final_data_PDB/train/rostlab_subset.tar")
        with tarfile.open(tar_path, "r") as tar:
            processed = 0
            skipped_singletons = 0  # NEW
            total_start = time.time()

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
                    start = time.time()
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
                        print(f"[train] Encoding: {member.name}")
                        vq_ids = model.encode_pdb(tmp_path)
                    finally:
                        os.remove(tmp_path)

                    parts = member.name.split("-")
                    if len(parts) < 2:
                        print(f"[train] Skipped invalid name: {member.name}")
                        continue
                    protein_id = parts[1]

                    entries.append({
                        "id": protein_id,
                        "sequence": seq,
                        "vq_ids": vq_ids.tolist()
                    })
                    processed += 1
                    if processed % 100 == 0:
                        print(f"[train] {processed} done – Time: {time.time() - start:.2f}s")

                except Exception as e:
                    print(f"[train] Failed: {member.name}, {e}")

            print(f"[train] Finished processing {processed} proteins in {time.time() - total_start:.2f}s")
            print(f"[train] Skipped {skipped_singletons} singleton entries.")

    # === VAL / TEST ===
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
                vq_ids = model.encode_pdb(str(pdb_file))

                entries.append({
                    "id": protein_id,
                    "sequence": seq,
                    "vq_ids": vq_ids.tolist()
                })
                processed += 1
                if processed % 100 == 0:
                    print(f"[{split}] {processed} done.")
            except Exception as e:
                print(f"[{split}] Failed: {pdb_file.name}, {e}")
                with open(pdb_file, "r") as f:
                    print("".join(f.readlines()[:30]))

        print(f"[{split}] Finished processing {processed} proteins in {time.time() - total_start:.2f}s")
        print(f"[{split}] Skipped {skipped_singletons} singleton entries.")

    # Save .pkli
    with open(pickle_path, "wb") as f:
        pickle.dump(entries, f)

    # Save .jsonl
    with open(jsonl_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    print(f"[{split}] Done: {len(entries)} entries written to {jsonl_path}")

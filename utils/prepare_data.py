import os
import tarfile
import pickle
from pathlib import Path
from biotite.structure.io.pdb import PDBFile
from biotite.structure.filter import _filter_atom_names

# Load singleton IDs
SINGLETON_ID_PATH = "/mnt/data/large/prostt5_IDs/afdb50best_foldseek_clu_singleton.ids"
with open(SINGLETON_ID_PATH, "r") as f:
    singleton_ids = set(line.strip().split("-")[1] for line in f if "-" in line)

print(f"Loaded {len(singleton_ids)} singleton IDs.")


# pdb to biotite protein format, extract only backbone
def load_prot_from_pdb(pdb_file):
    file = PDBFile.read(pdb_file)
    array_stack = file.get_structure(model=1)
    return array_stack[_filter_atom_names(array_stack, ["N", "CA", "C", "O"])]

# Output base
base_output = Path("/mnt/data/large/subset")
base_output.mkdir(parents=True, exist_ok=True)


# === VAL and TEST ===
for split in ["val", "test"]:
    pdb_dir = Path(split) / f"{split}_pdb"
    output_dir = base_output / split
    output_dir.mkdir(parents=True, exist_ok=True)

    pdbs = {}
    for pdb_file in pdb_dir.glob("*.pdb"):
        mid = pdb_file.stem.split("-")[1]
        if mid in singleton_ids:
            print(f"[{split}] Skipping singleton: {pdb_file.name}")
            continue
        try:
            prot = load_prot_from_pdb(str(pdb_file))
            pdbs[pdb_file.stem] = prot
        except Exception as e:
            print(f"[{split}] Failed to load {pdb_file.name}: {e}")

    with open(output_dir / "proteins.pkl", "wb") as f:
        pickle.dump(pdbs, f)
    print(f"[{split}] Saved {len(pdbs)} proteins to {output_dir/'proteins.pkl'}")

# # === TRAIN ===
# split = "train"
# output_dir = base_output / split
# output_dir.mkdir(parents=True, exist_ok=True)
#
# tar_path = Path("train") / "rostlab_subset.tar"
# pdbs = {}
# if tar_path.is_file():
#     with tarfile.open(tar_path, "r") as tar:
#         members = [m for m in tar.getmembers() if m.name.endswith(".pdb")]
#         for member in members:
#             mid = member.name.split("-")[1]
#             if mid in singleton_ids:
#                 print(f"[train] Skipping singleton: {member.name}")
#                 continue
#             try:
#                 f = tar.extractfile(member)
#                 if f:
#                     file_bytes = f.read()
#                     pdb_file = PDBFile()
#                     pdb_file.read(file_bytes.decode("utf-8").splitlines())
#                     array_stack = pdb_file.get_structure(model=1)
#                     prot = array_stack[_filter_atom_names(array_stack, ["N", "CA", "C", "O"])]
#                     pdbs[member.name.rsplit("/", 1)[-1].rsplit(".", 1)[0]] = prot
#             except Exception as e:
#                 print(f"[train] Failed to process {member.name}: {e}")
#     with open(output_dir / "proteins.pkl", "wb") as f:
#         pickle.dump(pdbs, f)
#     print(f"[train] Saved {len(pdbs)} proteins to {output_dir/'proteins.pkl'}")
# else:
#     print("Train tar file not found.")

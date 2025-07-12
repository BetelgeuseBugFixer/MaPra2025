#!/usr/bin/env python3
"""
prepare_data_memory_efficient.py
--------------------------------
Speicheroptimierte Version des ursprünglichen prepare_data-memory_efficient-Skripts.

Ziele
-----
* **Werte-Identität beibehalten** – keine Down-Casting-Tricks auf FP16/int16
* **Trotzdem weniger RAM-Spitzen** durch:
  - `torch.inference_mode()` & `.eval()`
  - Streaming-Writes (JSONL & Pickle)
  - Sofortiges Aufräumen (`del`, `gc.collect()`, `torch.cuda.empty_cache()`)
  - Fortschrittsanzeige nach je 1000 Proteinen

Getestet mit Python 3.10, PyTorch 2.2, Biotite 0.40.
"""

import contextlib
import gc
import json
import os
import pickle
import tarfile
import tempfile
import time
from pathlib import Path

import torch
from biotite.structure.filter import _filter_atom_names
from biotite.structure.io.pdb import PDBFile
from hydra_zen import load_from_yaml

from models.foldtoken_decoder.foldtoken import FoldToken
from models.bio2token.data.utils.utils import pdb_2_dict, uniform_dataframe, compute_masks
from models.bio2token.models.autoencoder import AutoencoderConfig, Autoencoder
from models.bio2token.utils.configs import pi_instantiate

# ----------------------------------------------------------------------------
# Konfiguration
# ----------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_LIMIT = 100_000      # Max. Proteine je Split (nur für train)
CHUNK_SIZE = 1_000         # Fortschrittsintervall

INPUT_DIR = Path("/mnt/data/large/zip_file/final_data_PDB")
SINGLETON_ID_PATH = Path("/mnt/data/large/prostt5_IDs/afdb50best_foldseek_clu_singleton.ids")
OUTPUT_BASE = Path("/mnt/data/large/subset")
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

# Aminosäure-Mapping
three_to_one = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C',
    'GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I',
    'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P',
    'SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'
}

# ----------------------------------------------------------------------------
# Hilfsfunktionen
# ----------------------------------------------------------------------------
def get_seq_from_lines(lines):
    """Extrahiere Sequenz einer monokettigen Struktur auf Basis der CA-Atome."""
    seq = []
    chain_id = None
    for line in lines:
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            current_chain = line[21]
            if chain_id is None:
                chain_id = current_chain
            elif current_chain != chain_id:
                return None
            one = three_to_one.get(line[17:20].strip())
            if one is None:
                return None
            seq.append(one)
        elif line.startswith("ENDMDL"):
            break
    return "".join(seq) if seq else None


def load_prot_from_pdb(pdb_file):
    """Biotite-Structure (N, CA, C, O) einlesen."""
    pdb = PDBFile.read(pdb_file)
    arr = pdb.get_structure(model=1)
    return arr[_filter_atom_names(arr, ["N","CA","C","O"])]

# ----------------------------------------------------------------------------
# Modelle initialisieren
# ----------------------------------------------------------------------------
print("[INIT] FoldToken laden …", flush=True)
foldtoken_model = FoldToken(device=DEVICE)
foldtoken_model.eval()

print("[INIT] Bio2Token laden …", flush=True)
yaml_cfg = load_from_yaml("models/bio2token/files/model.yaml")["model"]
model_cfg = pi_instantiate(AutoencoderConfig, yaml_cfg)
bio2token_model = Autoencoder(model_cfg)
state = torch.load(
    "models/bio2token/files/epoch=0243-val_loss_epoch=0.71-best-checkpoint.ckpt",
    map_location="cpu"
)["state_dict"]
bio2token_model.load_state_dict({k.replace("model.",""):v for k,v in state.items()})
bio2token_model.eval().to(DEVICE)
print("[INIT] Modelle bereit.", flush=True)

# Singleton-IDs
with open(SINGLETON_ID_PATH) as f:
    singleton_ids = {line.strip().split("-")[1] for line in f if "-" in line}
print(f"[INIT] Geladene Singleton-IDs: {len(singleton_ids):,}", flush=True)

# ----------------------------------------------------------------------------
# Haupt-Pipeline
# ----------------------------------------------------------------------------
@torch.inference_mode()
def process_split(split: str):
    """Verarbeite Dataset-Split (train/val/test)."""
    print("="*60, flush=True)
    print(f"[{split.upper()}] Verarbeitung startet", flush=True)

    out_dir = OUTPUT_BASE / split
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_file = open(out_dir / "proteins.jsonl", "w")
    pickle_file = open(out_dir / "proteins.pkl", "wb")

    processed = 0
    skipped = 0
    start_time = time.time()
    chunk_start = time.time()

    def handle_protein(pid, seq, pdb_path):
        nonlocal processed, chunk_start
        # FoldToken
        vq = foldtoken_model.encode_pdb(pdb_path)
        # Bio2Token
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            pdb_d = pdb_2_dict(pdb_path, None)
        st, unk, _, res_ids, tok, _ = uniform_dataframe(
            pdb_d['seq'], pdb_d['res_types'], pdb_d['coords_groundtruth'],
            pdb_d['atom_names'], pdb_d['res_atom_start'], pdb_d['res_atom_end']
        )
        batch = {
            'structure': torch.tensor(st).float(),
            'unknown_structure': torch.tensor(unk).bool(),
            'residue_ids': torch.tensor(res_ids).long(),
            'token_class': torch.tensor(tok).long()
        }
        batch = {k:v[~batch['unknown_structure']] for k,v in batch.items()}
        batch = compute_masks(batch, structure_track=True)
        batch = {k:v[None].to(DEVICE) for k,v in batch.items()}
        enc = bio2token_model.encoder(batch)
        encoding = enc['encoding'].squeeze(0).cpu().tolist()
        indices = enc['indices'].squeeze(0).cpu().tolist()
        # Write JSONL
        json.dump({
            'id': pid, 'sequence': seq,
            'vq_ids': vq.tolist(), 'bio2tokens': indices,
            'bio2token_encoding': encoding
        }, jsonl_file)
        jsonl_file.write("\n")
        # Write Pickle
        struct = load_prot_from_pdb(pdb_path)
        pickle.dump(struct, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

        processed += 1
        if processed % CHUNK_SIZE == 0:
            elapsed = time.time() - chunk_start
            print(f"[{split}] {processed:,} in {elapsed:.1f}s", flush=True)
            chunk_start = time.time()
        # Cleanup
        del batch, enc, struct, st, encoding, indices, vq
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    if split == 'train':
        tar = tarfile.open(INPUT_DIR / 'train' / 'rostlab_subset.tar', 'r')
        for member in tar:
            if processed >= BATCH_LIMIT: break
            if not member.name.endswith('.pdb'): continue
            pid = member.name.split('-')[1]
            if pid in singleton_ids:
                skipped += 1
                continue
            fobj = tar.extractfile(member)
            if not fobj: continue
            lines = fobj.read().decode().splitlines()
            seq = get_seq_from_lines(lines)
            if not seq: continue
            with tempfile.NamedTemporaryFile('w+', suffix='.pdb', delete=False) as tmp:
                tmp.write("\n".join(lines))
                tmp.flush()
                path = tmp.name
            try:
                handle_protein(pid, seq, path)
            except Exception as e:
                print(f"[WARN] {pid} -> {e}", flush=True)
            finally:
                os.remove(path)
        tar.close()
    else:
        dir_in = INPUT_DIR / split / f"{split}_pdb"
        for pdb_file in dir_in.glob('*.pdb'):
            pid = pdb_file.stem.split('-')[1] if '-' in pdb_file.stem else None
            if not pid or pid in singleton_ids:
                skipped += 1 if pid else 0
                continue
            try:
                lines = open(pdb_file).read().splitlines()
                seq = get_seq_from_lines(lines)
                if not seq: continue
                handle_protein(pid, seq, str(pdb_file))
            except Exception as e:
                print(f"[WARN] {pdb_file.name} -> {e}", flush=True)

    jsonl_file.close()
    pickle_file.close()
    total = time.time() - start_time
    print(f"[{split.upper()}] Fertig: {processed:,} Proteine, {skipped:,} übersprungen in {total:.1f}s", flush=True)


if __name__ == '__main__':
    for split in ['val', 'test', 'train']:
        process_split(split)

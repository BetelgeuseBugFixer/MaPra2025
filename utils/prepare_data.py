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
  - Streaming-Writes (JSONL & Pickle) für 'train', Bulk-Dumps für 'val'/'test'
  - Selteneres Aufräumen (`gc.collect()` & `torch.cuda.empty_cache()` nur alle CHUNK_SIZE)
  - Einmalige `devnull`-Datei für stdout-Redirect
  - Fortschrittsanzeige nach je CHUNK_SIZE Proteinen

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
CHUNK_SIZE = 1_000         # Fortschrittsintervalle

INPUT_DIR = Path("/mnt/data/large/zip_file/final_data_PDB")
SINGLETON_ID_PATH = Path(
    "/mnt/data/large/prostt5_IDs/afdb50best_foldseek_clu_singleton.ids"
)
OUTPUT_BASE = Path("/mnt/data/large/subset")
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

# Einmaliger devnull-Handle
DEVNULL = open(os.devnull, 'w')

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
    pdb = PDBFile.read(pdb_file)
    arr = pdb.get_structure(model=1)
    return arr[_filter_atom_names(arr, ["N","CA","C","O"])]

# ----------------------------------------------------------------------------
# Modelle laden
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

# Singleton-IDs laden
with open(SINGLETON_ID_PATH) as f:
    singleton_ids = {line.strip().split("-")[1] for line in f if "-" in line}
print(f"[INIT] {len(singleton_ids):,} Singleton-IDs geladen", flush=True)

# ----------------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------------
@torch.inference_mode()
def process_split(split: str):
    print("="*60, flush=True)
    print(f"[{split.upper()}] begin", flush=True)

    out_dir = OUTPUT_BASE / split
    out_dir.mkdir(parents=True, exist_ok=True)

    # Für train: Streaming Writes; für val/test: Bulk sammeln
    stream_json = (split == 'train')
    jsonl_path = out_dir / "proteins.jsonl"
    pickle_path = out_dir / "proteins.pkl"

    if stream_json:
        json_f = open(jsonl_path, 'w')
        pickle_f = open(pickle_path, 'wb')
    else:
        buffer_json = []
        buffer_structs = []

    processed = skipped = 0
    start = time.time()
    chunk_start = time.time()

    def handle_protein(pid, seq, pdb_path):
        nonlocal processed, chunk_start
        # FoldToken
        vq = foldtoken_model.encode_pdb(pdb_path)
        # Bio2Token
        with contextlib.redirect_stdout(DEVNULL):
            pdb_d = pdb_2_dict(pdb_path, None)
        st, unk, _, res_ids, tok, _ = uniform_dataframe(
            pdb_d['seq'], pdb_d['res_types'], pdb_d['coords_groundtruth'],
            pdb_d['atom_names'], pdb_d['res_atom_start'], pdb_d['res_atom_end']
        )
        batch = {k: torch.tensor(v).to(DEVICE) for k,v in {
            'structure':st, 'unknown_structure':unk,
            'residue_ids':res_ids, 'token_class':tok
        }.items()}
        batch = {k:v[~batch['unknown_structure']] for k,v in batch.items()}
        batch = compute_masks(batch, structure_track=True)
        batch = {k:v[None] for k,v in batch.items()}
        enc = bio2token_model.encoder(batch)
        encoding = enc['encoding'].squeeze(0).cpu().tolist()
        indices = enc['indices'].squeeze(0).cpu().tolist()
        # Ergebnis zusammenstellen
        entry = {
            'id': pid, 'sequence': seq,
            'vq_ids': vq.tolist(),
            'bio2tokens': indices,
            'bio2token_encoding': encoding
        }
        struct = load_prot_from_pdb(pdb_path)

        # Schreiben oder Puffern
        if stream_json:
            json.dump(entry, json_f); json_f.write("\n")
            pickle.dump(struct, pickle_f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            buffer_json.append(entry)
            buffer_structs.append(struct)

        processed += 1
        # Fortschritt & Cleanup
        if processed % CHUNK_SIZE == 0:
            dur = time.time() - chunk_start
            print(f"[{split}] {processed:,} in {dur:.1f}s", flush=True)
            chunk_start = time.time()
            # seltener GC / Cache clear
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        # temporäre Objekte löschen
        del batch, enc, struct, vq

    # Iteration
    if split == 'train':
        tar = tarfile.open(INPUT_DIR / 'train' / 'rostlab_subset.tar', 'r')
        for mem in tar:
            if processed >= BATCH_LIMIT: break
            if not mem.name.endswith('.pdb'): continue
            pid = mem.name.split('-')[1]
            if pid in singleton_ids:
                skipped += 1; continue
            fobj = tar.extractfile(mem)
            if not fobj: continue
            lines = fobj.read().decode().splitlines()
            seq = get_seq_from_lines(lines)
            if not seq: continue
            with tempfile.NamedTemporaryFile('w+', suffix='.pdb', delete=False) as tmp:
                tmp.write("\n".join(lines)); tmp.flush(); path = tmp.name
            try:
                handle_protein(pid, seq, path)
            except Exception as e:
                print(f"[WARN] {pid} -> {e}", flush=True)
            finally:
                os.remove(path)
        tar.close()
    else:
        for pdb_file in (INPUT_DIR / split / f"{split}_pdb").glob('*.pdb'):
            pid = pdb_file.stem.split('-')[1] if '-' in pdb_file.stem else None
            if not pid or pid in singleton_ids:
                skipped += bool(pid); continue
            try:
                lines = open(pdb_file).read().splitlines()
                seq = get_seq_from_lines(lines)
                if not seq: continue
                handle_protein(pid, seq, str(pdb_file))
            except Exception as e:
                print(f"[WARN] {pdb_file.name} -> {e}", flush=True)

    # Bulk write für val/test
    if not stream_json:
        with open(jsonl_path, 'w') as jf:
            for e in buffer_json: jf.write(json.dumps(e)+"\n")
        with open(pickle_path, 'wb') as pf:
            pickle.dump(buffer_structs, pf, protocol=pickle.HIGHEST_PROTOCOL)

    # Abschluss
    if stream_json:
        json_f.close(); pickle_f.close()
    total = time.time() - start
    print(f"[{split.upper()}] Fertig: {processed:,} Proteine, {skipped:,} übersprungen in {total:.1f}s", flush=True)


if __name__ == '__main__':
    for s in ['val', 'test', 'train']:
        process_split(s)

#!/usr/bin/env python3
import json
import time
import logging
from pathlib import Path

import torch
import h5py
from transformers import T5Tokenizer, T5EncoderModel
from tqdm import tqdm

# Setup device
device = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda" if torch.cuda.is_available() else "cpu")
)

def get_model(model_name: str):
    logging.info(f"Loading ProtT5 model: {model_name}")
    tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False, use_fast=False)
    model = T5EncoderModel.from_pretrained(model_name)
    if device.type == "cuda":
        try:
            model = torch.compile(model)
        except Exception:
            pass
    model.to(device).eval()
    return model, tokenizer

def load_sequences_from_jsonl(jsonl_path: Path):
    entries = []
    with jsonl_path.open("r") as f:
        for line in f:
            obj = json.loads(line)
            seq = obj["sequence"].translate(str.maketrans("UZO", "XXX"))
            entries.append((obj["id"], seq))
    return entries

def embed_sequences(entries, model, tokenizer, h5_path: Path, max_batch=100, max_residues=4000, max_seq_len=1000, per_protein=False):
    with h5py.File(h5_path, "w") as hf:
        batch, cum_res = [], 0
        for i, (pid, seq) in enumerate(tqdm(entries, desc=f"Embedding {h5_path.parent.name}", unit="seq")):
            spaced = " ".join(seq)
            L = len(seq)
            if L > max_seq_len:
                continue
            if len(batch) + 1 > max_batch or cum_res + L > max_residues:
                write_batch(batch, model, tokenizer, hf, per_protein)
                batch, cum_res = [], 0
            batch.append((pid, spaced, L))
            cum_res += L
        if batch:
            write_batch(batch, model, tokenizer, hf, per_protein)

def write_batch(batch, model, tokenizer, hf, per_protein):
    pids, spaced_seqs, lengths = zip(*batch)
    encoded = tokenizer.batch_encode_plus(spaced_seqs, padding="longest", return_tensors="pt", add_special_tokens=True)
    with torch.no_grad():
        outputs = model(input_ids=encoded["input_ids"].to(device), attention_mask=encoded["attention_mask"].to(device))
    hidden = outputs.last_hidden_state
    for i, pid in enumerate(pids):
        emb = hidden[i, :lengths[i]]
        if per_protein:
            emb = emb.mean(dim=0)
        hf.create_dataset(pid, data=emb.cpu().numpy())

def main():
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
    model_name = "Rostlab/prot_t5_xl_half_uniref50-enc"
    splits = ["train", "val", "test"]

    model, tokenizer = get_model(model_name)

    for split in splits:
        split_dir = Path(f"/mnt/data/large/subset/{split}")
        jsonl_path = split_dir / "proteins.jsonl"
        h5_path = split_dir / "embeds.h5"

        if not jsonl_path.exists():
            logging.warning(f"{jsonl_path} not found, skipping.")
            continue

        logging.info(f"Processing {jsonl_path}")
        entries = load_sequences_from_jsonl(jsonl_path)
        start = time.time()
        embed_sequences(entries, model, tokenizer, h5_path)
        logging.info(f"Saved embeddings to {h5_path} in {time.time() - start:.2f}s")

if __name__ == "__main__":
    main()

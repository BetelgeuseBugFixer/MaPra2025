#!/usr/bin/env python3
import argparse
import logging
import time
from pathlib import Path

import torch
import h5py
from transformers import T5EncoderModel, T5Tokenizer
from tqdm import tqdm
import os, sys

# Detect device with MPS support (macOS), then CUDA, else CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

def get_T5_model(model_dir: Path, transformer_link: str):
    """
    Load the T5 encoder model + tokenizer from HuggingFace, move to device, optionally torch.compile() on CUDA.
    """
    logging.info(f"Loading T5EncoderModel {transformer_link}")
    model = T5EncoderModel.from_pretrained(transformer_link, cache_dir=model_dir)

    # Only compile on CUDA
    if device.type == "cuda":
        try:
            model = torch.compile(model)
            logging.info("Compiled model with torch.compile() for extra speed.")
        except Exception as e:
            logging.warning(f"torch.compile() failed ({e}); continuing without compilation.")
    else:
        logging.info("Skipping torch.compile(): not running on CUDA device.")

    model.to(device).eval()

    tokenizer = T5Tokenizer.from_pretrained(
        transformer_link,
        do_lower_case=False,
        use_fast=False
    )

    return model, tokenizer

def read_fasta(fasta_path: Path) -> dict:
    """
    Read a FASTA file and return a dict {seq_id: sequence_string}.
    """
    sequences = {}
    with fasta_path.open('r') as f:
        seq_id = None
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                seq_id = line[1:].split()[0]
                sequences[seq_id] = ''
            else:
                sequences[seq_id] += line.upper().replace('-', '')
    return sequences

def process_batch(
    batch: list,
    hf: h5py.File,
    model: torch.nn.Module,
    tokenizer,
    per_protein: bool
):
    """
    Tokenize, embed, and write a batch of sequences into the open HDF5 file.
    """
    pdb_ids, seqs, seq_lens = zip(*batch)
    encoding = tokenizer.batch_encode_plus(
        seqs,
        add_special_tokens=True,
        padding='longest',
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    hidden_states = outputs.last_hidden_state  # shape (B, L, D)
    for idx, seq_id in enumerate(pdb_ids):
        L = seq_lens[idx]
        emb = hidden_states[idx, :L]
        if per_protein:
            emb = emb.mean(dim=0)
        data = emb.cpu().numpy()
        hf.create_dataset(seq_id, data=data)
        logging.debug(f"Wrote embedding for {seq_id}, shape {data.shape}")

def get_embeddings(
    seq_path: Path,
    emb_path: Path,
    model_dir: Path,
    transformer_link: str,
    per_protein: bool,
    max_residues: int = 4000,
    max_seq_len: int = 1000,
    max_batch: int = 100
) -> None:
    """
    Reads a single FASTA (seq_path), batches sequences, obtains embeddings via ProtT5,
    and writes to a single HDF5 file at emb_path.
    """
    sequences = read_fasta(seq_path)
    model, tokenizer = get_T5_model(model_dir, transformer_link)

    total = len(sequences)
    lengths = [len(s) for s in sequences.values()]
    avg_len = sum(lengths) / total if total > 0 else 0
    n_long = sum(1 for L in lengths if L > max_seq_len)
    sorted_seqs = sorted(
        sequences.items(), key=lambda kv: len(kv[1]), reverse=True
    )

    logging.info(
        f"[{seq_path.name}] Read {total} sequences; avg len {avg_len:.2f}; "
        f">{max_seq_len}: {n_long}"
    )
    start_time = time.time()

    with h5py.File(str(emb_path), 'w') as hf:
        batch = []
        cum_res = 0
        for idx, (seq_id, raw_seq) in enumerate(
            tqdm(sorted_seqs, total=total, desc=f"Embedding {seq_path.name}", unit="seq"),
            start=1
        ):
            # Replace uncommon residues
            seq = raw_seq.translate(str.maketrans('UZO', 'XXX'))
            L = len(seq)
            spaced = ' '.join(seq)

            # Flush batch if limits exceeded or single seq too long
            if batch and (
                len(batch) + 1 > max_batch or
                cum_res + L > max_residues or
                L > max_seq_len
            ):
                process_batch(batch, hf, model, tokenizer, per_protein)
                batch, cum_res = [], 0

            batch.append((seq_id, spaced, L))
            cum_res += L

            # Flush last batch
            if idx == total and batch:
                process_batch(batch, hf, model, tokenizer, per_protein)

    elapsed = time.time() - start_time
    logging.info(
        f"[{seq_path.name}] Embeddings saved to {emb_path.name}. "
        f"Processed {total} sequences in {elapsed:.2f}s "
        f"(avg {elapsed/total:.4f}s/seq)."
    )

def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Batch-create ProtT5 embeddings for every FASTA in a directory, saving HDF5 outputs.'
    )
    parser.add_argument(
        '-i', '--input_dir', required=True, type=Path,
        help='Directory containing one or more FASTA (*.fasta) files.'
    )
    parser.add_argument(
        '-o', '--output_dir', required=True, type=Path,
        help='Directory in which to write one .h5 file per input FASTA.'
    )
    parser.add_argument(
        '--models-dir', type=Path, default=None,
        help='Cache dir for the pre-trained encoder checkpoint.'
    )
    parser.add_argument(
        '--models-name', type=str,
        default='Rostlab/prot_t5_xl_half_uniref50-enc',
        help='Identifier of the ProtT5 encoder model on HuggingFace.'
    )
    parser.add_argument(
        '--per-protein', action='store_true',
        help='Store a single mean‐pooled embedding per protein (instead of per‐residue).'
    )
    parser.add_argument(
        '--max-residues', type=int, default=4000,
        help='Max total residues per batch.'
    )
    parser.add_argument(
        '--max-seq-len', type=int, default=1000,
        help='Threshold for single‐sequence processing.'
    )
    parser.add_argument(
        '--max-batch', type=int, default=100,
        help='Max sequences per batch.'
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='Enable debug logging.'
    )
    return parser

def main():
    parser = create_arg_parser()
    args = parser.parse_args()

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.info(f"Using device: {device}")

    # Ensure output_dir exists
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Find all .fasta files in input_dir
    fasta_files = sorted(args.input_dir.glob("*.fasta"))
    if not fasta_files:
        logging.error(f"No FASTA files found in {args.input_dir}")
        sys.exit(1)

    for fasta_path in fasta_files:
        # Derive output .h5 name from the FASTA basename
        h5_name = fasta_path.stem + ".h5"
        h5_path = args.output_dir / h5_name

        # Skip if output already exists?
        if h5_path.exists():
            logging.info(f"Skipping {fasta_path.name} (output {h5_name} already exists).")
            continue

        get_embeddings(
            seq_path=fasta_path,
            emb_path=h5_path,
            model_dir=args.models_dir,
            transformer_link=args.models_name,
            per_protein=args.per_protein,
            max_residues=args.max_residues,
            max_seq_len=args.max_seq_len,
            max_batch=args.max_batch
        )

if __name__ == "__main__":
    main()
    # python3
    # generate_embeds_dir.py \
    # - -input_dir  \
    # --output_dir  \
    # --models - name
    # Rostlab/prot_t5_xl_half_uniref50 - enc \

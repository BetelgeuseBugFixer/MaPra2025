import json
import os
import sys
import random

import h5py

SPLIT_SEED = 42


def load_common_ids(emb_source: str, tok_jsonl_path: str, use_single_file: bool) -> list:
    if use_single_file:
        with h5py.File(emb_source, "r") as f:
            emb_ids = set(f.keys())
    else:
        emb_ids = {os.path.splitext(f)[0] for f in os.listdir(emb_source) if f.endswith(".h5")}

    tok_ids = set()
    with open(tok_jsonl_path, "r") as fh:
        for line in fh:
            tok_ids.add(next(iter(json.loads(line).keys())))

    common = sorted(emb_ids & tok_ids)
    if not common:
        raise ValueError("No overlapping protein IDs.")
    return common


def split_ids(ids: list, seed: int = SPLIT_SEED):
    random.seed(seed)
    random.shuffle(ids)
    n_total = len(ids)
    n_train = int(0.70 * n_total)
    n_val = max(int(0.15 * n_total), 1) if n_total > 2 else 0
    n_test = n_total - n_train - n_val
    return ids[:n_train], ids[n_train:n_train + n_val], ids[n_train + n_val:]


def save_splits(split_file, train_ids, val_ids, test_ids):
    os.makedirs(os.path.dirname(split_file), exist_ok=True)
    with open(split_file, 'w') as f:
        json.dump({
            'train': train_ids,
            'val': val_ids,
            'test': test_ids
        }, f)

def main(embeddings_path,token_path,split_file):
    use_file = os.path.isfile(embeddings_path)

    common_ids = load_common_ids(embeddings_path, token_path, use_file)
    test_ids, train_ids, val_ids = split_ids(common_ids)
    save_splits(split_file, train_ids, val_ids, test_ids)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2],sys.argv[3])

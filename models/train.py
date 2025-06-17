import argparse
import json
import os

import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from tqdm import tqdm
import matplotlib.pyplot as plt

from models.simple_classifier.simple_classifier import ResidueTokenCNN, ResidueTokenLNN
from models.simple_classifier.datasets import ProteinPairJSONL, ProteinPairJSONL_FromDir, PAD_LABEL


# ------------------------------------------------------------
#  Utility functions
# ------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train per-residue token classifier (CNN) with early stopping and plots"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--emb_file", help="HDF5 file with one dataset per protein ID")
    group.add_argument("--emb_dir", help="Directory with per-protein .h5 files")

    parser.add_argument("--tok_jsonl", required=True,
                        help="JSONL with per-protein {'<ID>': {..., 'vqid': [...]}}")
    parser.add_argument("--split_file", required=True,
                        help="JSON containing the ids split into train, validation and test")
    parser.add_argument("--codebook_size", type=int, default=1024,
                        help="Size of VQ vocabulary")

    parser.add_argument("--model", type=str, default="cnn", help="type of model to use")
    parser.add_argument("--kernel_size", type=int, default=5, help="kernel size of the cnn")
    parser.add_argument("--d_emb", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--batch", type=int, default=1,
                        help="Batch size (1 → no padding)")

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--patience", type=int, default=3,
                        help="Early stopping patience (in epochs)")
    parser.add_argument("--run_test", action="store_true", help="also run test set")
    parser.add_argument("--no_wandb", action="store_true", help="do not log wandb")
    parser.add_argument("--out_folder",type=str, help="directory where the plots and model files will be stored", required=True)
    return parser.parse_args()


def create_data_loaders(emb_source, tok_jsonl, train_ids, val_ids, test_ids, batch_size, use_single_file):
    DSClass = ProteinPairJSONL if use_single_file else ProteinPairJSONL_FromDir
    train_ds = DSClass(emb_source, tok_jsonl, train_ids)
    val_ds = DSClass(emb_source, tok_jsonl, val_ids)
    test_ds = DSClass(emb_source, tok_jsonl, test_ids)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=pad_collate),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=pad_collate),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=pad_collate),
    )


def build_cnn(d_emb, hidden, vocab_size, kernel_size, dropout, lr, device):
    model = ResidueTokenCNN(d_emb, hidden, vocab_size, kernel_size, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_LABEL)
    return model, optimizer, criterion


def build_nn(d_emb, hidden, vocab_size, dropout, lr, device):
    model = ResidueTokenLNN(d_emb, hidden, vocab_size, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    return model, optimizer, criterion


def pad_collate(batch):
    """
    batch: list of (emb, tok) where
      emb: Tensor[L_i, d_emb]
      tok: Tensor[L_i]
    Returns:
      embs_padded: Tensor[B, L_max, d_emb]
      toks_padded: Tensor[B, L_max]
    """
    embs, toks = zip(*batch)
    # pad embeddings along the sequence dimension
    embs_padded = pad_sequence(embs, batch_first=True)  # pad with 0.0 by default
    # pad tokens along the sequence dimension, with PAD_LABEL
    toks_padded = pad_sequence(toks, batch_first=True, padding_value=PAD_LABEL)
    return embs_padded, toks_padded


def _masked_accuracy(logits, tgt, mask):
    pred = logits.argmax(dim=-1)
    correct = ((pred == tgt) & mask).sum().item()
    total = mask.sum().item()
    return correct / total if total else 0.0


def run_epoch(model, loader, criterion, optimizer=None, device="cpu"):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss = total_acc = total_samples = 0
    torch.set_grad_enabled(is_train)

    for emb, tok in tqdm(loader, desc="train" if is_train else "val", leave=False):
        emb, tok = emb.to(device), tok.to(device)
        mask = (tok != PAD_LABEL)
        logits = model(emb)
        loss = criterion(logits.transpose(1, 2), tok)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        bsz = emb.size(0)
        total_loss += loss.item() * bsz
        total_acc += _masked_accuracy(logits, tok, mask) * bsz
        total_samples += bsz

    return total_loss / total_samples, total_acc / total_samples


def load_split_file(split_file):
    with open(split_file) as f:
        split_data = json.load(f)
        train_ids = split_data['train']
        val_ids = split_data['val']
        test_ids = split_data['test']
    return train_ids, val_ids, test_ids


# ------------------------------------------------------------
# Main workflow with early stopping & plots
# ------------------------------------------------------------


def get_model(args):
    match args.model:
        case "cnn":
            return build_cnn(
                args.d_emb, args.hidden, args.codebook_size, args.kernel_size, args.dropout, args.lr, args.device
            )
        case "NN":
            # will not work yet, because nn needs a single token as input, but it will be given s seq
            raise NotImplementedError
            # return build_nn(args.d_emb, args.hidden, args.codebook_size, args.dropout, args.lr, args.device)
        case _:
            raise NotImplementedError


def init_wand_db(args):
    return wandb.init(
        entity="MaPra",
        project="monomer-structure-prediction",
        config={
            "learning_rate": args.lr,
            "kernel_size": args.kernel_size,
            "device": args.device,
            "patience": args.patience,
            "architecture": args.model,
            "dataset": args.tok_jsonl,
            "epochs": args.epochs,
            "hidden": args.hidden,
            "dropout": args.dropout,
            "batch_size": args.batch,
            "run_test": args.run_test
        }
    )


def main(args):
    use_file = args.emb_file is not None
    emb_source = args.emb_file if use_file else args.emb_dir

    train_ids, val_ids, test_ids = load_split_file(args.split_file)

    train_loader, val_loader, test_loader = create_data_loaders(
        emb_source, args.tok_jsonl, train_ids, val_ids, test_ids, args.batch, use_file
    )

    model, optimizer, criterion = get_model(args)

    # init wand db
    run = None
    if not args.no_wandb:
        wandb.login(key=open("wandb_key").read().strip())
        run = init_wand_db(args)

    # Tracking metrics
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    best_val_loss = float('inf')
    patience_ctr = 0

    # init output
    os.makedirs(args.out_folder, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, args.device)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device=args.device)

        if not args.no_wandb:
            run.log({
                "acc": tr_acc,
                "loss": tr_loss,
                "val_acc": val_acc,
                "val_loss": val_loss,
            })
        train_losses.append(tr_loss)
        train_accs.append(tr_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Epoch {epoch:02d} | train {tr_loss:.4f}/{tr_acc:.4f} | val {val_loss:.4f}/{val_acc:.4f}")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_ctr = 0
            # Save best model
            torch.save(model.state_dict(), os.path.join(args.out_folder,"simple_cnn_classifier.pt"))
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"Stopping early at epoch {epoch} (no improvement for {args.patience} epochs)")
                break

    # Final test evaluation
    if args.run_test:
        test_loss, test_acc = run_epoch(model, test_loader, criterion, device=args.device)
        print(f"Test {test_loss:.4f}/{test_acc:.4f}")

    if args.no_wandb:
        run.finish()
        # Plotting training curves
    epochs_range = range(1, len(train_losses) + 1)

    plt.figure()
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss over Epochs')
    plt.savefig(os.path.join(args.out_folder,'loss_curve.png'))
    plt.show()

    plt.figure()
    plt.plot(epochs_range, train_accs, label='Train Acc')
    plt.plot(epochs_range, val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy over Epochs')
    plt.savefig(os.path.join(args.out_folder,'accuracy_curve.png'))
    plt.show()

    print(f"Saved best model → {os.path.join(args.out_folder,"simple_cnn_classifier.pt")}")


if __name__ == "__main__":
    args = parse_args()
    main(args)

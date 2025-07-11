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

from models.model_utils import _masked_accuracy
from models.simple_classifier.simple_classifier import ResidueTokenCNN, ResidueTokenLNN
from models.datasets.datasets import ProteinPairJSONL, ProteinPairJSONL_FromDir, PAD_LABEL, SeqTokSet, SeqStrucTokSet
from models.end_to_end.whole_model import TFold


# ------------------------------------------------------------
#  Utility functions
# ------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train per-residue token classifier (CNN) with early stopping and plots"
    )

    # data
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--emb_file", help="HDF5 file with one dataset per protein ID")
    group.add_argument("--emb_dir", help="Directory with per-protein .h5 files")

    parser.add_argument("--tok_jsonl",
                        help="JSONL with per-protein {'<ID>': {..., 'vqid': [...]}}")
    parser.add_argument("--split_file",
                        help="JSON containing the ids split into train, validation and test")
    parser.add_argument("--data_dir", help="Directory with train, validation and test sub directories")

    # model
    parser.add_argument("--model", type=str, default="cnn", help="type of model to use")
    # tfold exclusive setting
    parser.add_argument("--lora_plm", action="store_true", help=" use lora to retrain the plm")

    # cnn exclusive setting
    parser.add_argument("--codebook_size", type=int, default=1024,
                        help="Size of VQ vocabulary")
    parser.add_argument("--d_emb", type=int, default=1024)

    # general settings
    parser.add_argument("--kernel_size", type=int, nargs="+", default=[5], help="kernel size of the cnn")
    parser.add_argument("--hidden", type=int, nargs="+", default=[2048])

    # trainings setting
    parser.add_argument("--batch", type=int, default=1,
                        help="Batch size")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--patience", type=int, default=3,
                        help="Early stopping patience (in epochs)")
    parser.add_argument("--no_wandb", action="store_true", help="do not log wandb")
    parser.add_argument("--plot", action="store_true", help="plot training progress")
    parser.add_argument("--out_folder", type=str, help="directory where the plots and model files will be stored",
                        required=True)
    return parser.parse_args()


def create_tfold_data_loaders(data_dir):
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")
    train_dataset = SeqTokSet(os.path.join(train_dir, "proteins.jsonl"))
    val_dataset = SeqStrucTokSet(os.path.join(val_dir, "proteins.jsonl"), os.path.join(val_dir, "proteins.pkl"))
    test_dataset = SeqStrucTokSet(os.path.join(test_dir, "proteins.jsonl"), os.path.join(test_dir, "proteins.pkl"))
    return (
        DataLoader(train_dataset, batch_size=args.batch, collate_fn=collate_seq_tok_batch, pin_memory=True,
                   persistent_workers=True),
        DataLoader(val_dataset, batch_size=args.batch, collate_fn=collate_seq_struc_tok_batch, pin_memory=True,
                   persistent_workers=True),
        DataLoader(test_dataset, batch_size=args.batch, collate_fn=collate_seq_struc_tok_batch, pin_memory=True,
                   persistent_workers=True)
    )


def create_cnn_data_loaders(emb_source, tok_jsonl, train_ids, val_ids, test_ids, batch_size, use_single_file):
    DSClass = ProteinPairJSONL if use_single_file else ProteinPairJSONL_FromDir
    train_ds = DSClass(emb_source, tok_jsonl, train_ids)
    val_ds = DSClass(emb_source, tok_jsonl, val_ids)
    test_ds = DSClass(emb_source, tok_jsonl, test_ids)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=pad_collate),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=pad_collate),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=pad_collate),
    )


def build_t_fold(lora_plm, hidden, kernel_size, dropout, lr, device):
    model = TFold(hidden=hidden, kernel_sizes=kernel_size, dropout=dropout, device=device, use_lora=lora_plm)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return model, optimizer


def build_cnn(d_emb, hidden, vocab_size, kernel_size, dropout, lr, device):
    model = ResidueTokenCNN(d_emb, hidden, vocab_size, kernel_size, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return model, optimizer


def build_nn(d_emb, hidden, vocab_size, dropout, lr, device):
    model = ResidueTokenLNN(d_emb, hidden, vocab_size, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    return model, optimizer, criterion


def collate_seq_struc_tok_batch(batch):
    sequences, token_lists, structures = zip(*batch)

    # Padding der VQ-Tokens
    padded_tokens = pad_sequence(token_lists, batch_first=True, padding_value=PAD_LABEL)

    return list(sequences), padded_tokens, list(structures)


def collate_seq_tok_batch(batch):
    sequences, token_lists = zip(*batch)

    # Padding der VQ-Tokens
    padded_tokens = pad_sequence(token_lists, batch_first=True, padding_value=PAD_LABEL)

    return list(sequences), padded_tokens


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
        case "t_fold":
            return build_t_fold(args.lora_plm, args.hidden, args.kernel_size, args.dropout, args.lr,
                                args.device)
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
            "epochs": args.epochs,
            "hidden": args.hidden,
            "dropout": args.dropout,
            "batch_size": args.batch,
            "lora_plm": args.lora_plm
        }
    )


def get_dataset(args):
    if args.model == "cnn":
        use_file = args.emb_file is not None
        emb_source = args.emb_file if use_file else args.emb_dir

        train_ids, val_ids, test_ids = load_split_file(args.split_file)

        return create_cnn_data_loaders(
            emb_source, args.tok_jsonl, train_ids, val_ids, test_ids, args.batch, use_file
        )
    elif args.model == "t_fold":
        return create_tfold_data_loaders(args.data_dir)


def main(args):
    # load dataset
    train_loader, val_loader, test_loader = get_dataset(args)

    model, optimizer = get_model(args)

    # init wand db
    run = None
    if not args.no_wandb:
        # wandb.login(key=open("wandb_key").read().strip())
        run = init_wand_db(args)

    # init important metric based on if the models need to optimize or minimize
    if model.maximize:
        best_val_score = -float('inf')
    else:
        best_val_score = float('inf')
    patience_ctr = 0

    # init output
    os.makedirs(args.out_folder, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_score_dict = model.run_epoch(train_loader, optimizer=optimizer, device=args.device)
        val_score_dict = model.run_epoch(val_loader, device=args.device)
        score_dict = train_score_dict | val_score_dict

        if not args.no_wandb:
            run.log(score_dict)

        tr_loss = score_dict["loss"]
        tr_acc = score_dict["acc"]
        val_loss = score_dict["val_loss"]
        val_acc = score_dict["val_acc"]
        lddt_string = f" |{score_dict["val_lddt"]}" if score_dict["val_lddt"] else ""
        print(f"Epoch {epoch:02d} | train {tr_loss:.4f}/{tr_acc:.4f} | val {val_loss:.4f}/{val_acc:.4f}{lddt_string}")

        # Early stopping check
        new_score = score_dict[model.key_metric]
        if ((model.maximize and new_score > best_val_score)
                or (not model.maximize and new_score < best_val_score)):
            best_val_score = new_score
            patience_ctr = 0
            # Save best model
            model.save(args.out_folder)
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"Stopping early at epoch {epoch} (no improvement for {args.patience} epochs)")
                break

    if not args.no_wandb:
        run.finish()
        # Plotting training curves

    # if args.plot:
    #    epochs_range = range(1, len(train_losses) + 1)
    #    plot_training(args.out_folder, epochs_range, train_accs, train_losses, val_accs, val_losses)

    print(f"Saved best model in → {args.out_folder}")


def plot_training(out_folder, epochs_range, train_accs, train_losses, val_accs, val_losses):
    plt.figure()
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss over Epochs')
    plt.savefig(os.path.join(out_folder, 'loss_curve.png'))
    plt.figure()
    plt.plot(epochs_range, train_accs, label='Train Acc')
    plt.plot(epochs_range, val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy over Epochs')
    plt.savefig(os.path.join(out_folder, 'accuracy_curve.png'))


if __name__ == "__main__":
    args = parse_args()
    main(args)

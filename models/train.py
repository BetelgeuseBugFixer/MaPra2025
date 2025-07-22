import argparse
import json
import os
import time

import wandb
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

import matplotlib.pyplot as plt

from models.bio2token.data.utils.utils import pad_and_stack_tensors
from models.simple_classifier.simple_classifier import ResidueTokenCNN
from models.datasets.datasets import ProteinPairJSONL, ProteinPairJSONL_FromDir, PAD_LABEL, StructureAndTokenSet, \
    TokenSet
from models.end_to_end.whole_model import TFold, FinalModel


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
    parser.add_argument("--model", type=str, default="cnn", help="type of model to use, options: cnn, tfold, final")
    parser.add_argument("--resume", type=str, help="path to an existing model to resume training")
    parser.add_argument("--wandb_resume_id", type=str, help="W&B id of an existing wandb run")
    # end to end exclusive setting
    parser.add_argument("--lora_plm", action="store_true", help=" use lora to finetune the plm")
    parser.add_argument("--lora_decoder", action="store_true", help=" use lora to finetune the plm")
    parser.add_argument("--bio2token", action="store_true", help="use bio2token instead of foldtoken in tfold")
    parser.add_argument("--alpha",type=int,help="weight of the lddt loss",default=1)
    parser.add_argument("--beta",type=int,help="weight of the encoding loss",default=1)

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
    parser.add_argument("--val_batch", type=int, default=1,
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


def create_tfold_data_loaders(data_dir, batch_size, val_batch_size, fine_tune_plm, bio2token=False, final_model=False):
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    if final_model:
        token_type = "encoding"
        train_set = StructureAndTokenSet(train_dir, token_type, precomputed_embeddings=not fine_tune_plm)
        val_set = StructureAndTokenSet(val_dir, token_type, precomputed_embeddings=not fine_tune_plm)
        # adapt padding function to whether the model will receive embeddings or seqs as input
        collate_function = collate_seq_struc_tok_batch if fine_tune_plm else collate_emb_struc_tok_batch
        return (
            DataLoader(train_set, batch_size=batch_size, collate_fn=collate_function),
            DataLoader(val_set, batch_size=batch_size, collate_fn=collate_function)
        )
    else:
        token_type = "bio2token" if bio2token else "foldtoken"
        train_set = TokenSet(train_dir, token_type=token_type, precomputed_embeddings=not fine_tune_plm)
        val_set = StructureAndTokenSet(val_dir, token_type, precomputed_embeddings=not fine_tune_plm)
        # get padding function based on input
        val_collate_function = collate_seq_struc_tok_batch if fine_tune_plm else collate_emb_struc_tok_batch
        train_collate_function = collate_seq_tok_batch if fine_tune_plm else collate_emb_tok_batch

        return (
            DataLoader(train_set, batch_size=batch_size, collate_fn=train_collate_function),
            DataLoader(val_set, batch_size=val_batch_size, collate_fn=val_collate_function)
        )


def create_cnn_data_loaders(emb_source, tok_jsonl, train_ids, val_ids, test_ids, batch_size, use_single_file):
    DSClass = ProteinPairJSONL if use_single_file else ProteinPairJSONL_FromDir
    train_ds = DSClass(emb_source, tok_jsonl, train_ids)
    val_ds = DSClass(emb_source, tok_jsonl, val_ids)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_emb_tok_batch),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_emb_tok_batch),
    )


def build_t_fold(lora_plm, hidden, kernel_size, dropout, lr, device,bio2token, resume):
    if resume:
        model = TFold.load_tfold(resume, device=device).to(device)
    else:
        model = TFold(hidden=hidden, kernel_sizes=kernel_size, dropout=dropout, device=device, use_lora=lora_plm,bio2token=bio2token).to(
            device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return model, optimizer


def build_final_model(lora_plm, lora_decoder, hidden, kernel_size, dropout, lr, device,alpha,beta, resume):
    if resume:
        model = FinalModel.load_final(resume, device=device).to(device)
    else:
        model = FinalModel(hidden, kernel_sizes=kernel_size, plm_lora=lora_plm, decoder_lora=lora_decoder,
                           device=device,
                           dropout=dropout,alpha=alpha,beta=beta)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return model, optimizer


def build_cnn(d_emb, hidden, vocab_size, kernel_size, dropout, lr, device, resume):
    if resume:
        model = ResidueTokenCNN.load_cnn(resume)
    else:
        model = ResidueTokenCNN(d_emb, hidden, vocab_size, kernel_size, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return model, optimizer


def collate_seq_struc(batch):
    sequences, structures = zip(*batch)
    # Pad structure
    structures = pad_and_stack_tensors(structures, 0)
    return list(sequences), structures


def collate_seq_struc_tok_batch(batch):
    sequences, token_lists, structures = zip(*batch)

    # Padding der VQ-Tokens
    padded_tokens = pad_sequence(token_lists, batch_first=True, padding_value=PAD_LABEL)
    structures = pad_and_stack_tensors(structures, 0)

    return list(sequences), padded_tokens, structures


def collate_seq_tok_batch(batch):
    sequences, token_lists = zip(*batch)

    # Padding der VQ-Tokens
    padded_tokens = pad_sequence(token_lists, batch_first=True, padding_value=PAD_LABEL)

    return list(sequences), padded_tokens


def collate_emb_struc_tok_batch(batch):
    embs, toks, structures = zip(*batch)

    # pad embeddings along the sequence dimension
    embs_padded = pad_sequence(embs, batch_first=True)  # pad with 0.0 by default
    # pad tokens along the sequence dimension, with PAD_LABEL
    toks_padded = pad_sequence(toks, batch_first=True, padding_value=PAD_LABEL)
    # pad structures
    structures = pad_and_stack_tensors(structures, 0)
    return embs_padded, toks_padded, structures


def collate_emb_tok_batch(batch):
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
                args.d_emb, args.hidden, args.codebook_size, args.kernel_size, args.dropout, args.lr, args.device,
                args.resume
            )
        case "t_fold":
            return build_t_fold(args.lora_plm, args.hidden, args.kernel_size, args.dropout, args.lr,
                                args.device, args.resume,args.bio2token)
        case "final":
            return build_final_model(args.lora_plm, args.lora_decoder, args.hidden, args.kernel_size, args.dropout,
                                     args.lr,
                                     args.device,args.alpha,args.beta, args.resume)
        case _:
            raise NotImplementedError


def init_wand_db(args):
    config = {
        "learning_rate": args.lr,
        "kernel_size": args.kernel_size,
        "device": args.device,
        "patience": args.patience,
        "architecture": args.model,
        "epochs": args.epochs,
        "hidden": args.hidden,
        "dropout": args.dropout,
        "batch_size": args.batch,
        "lora_plm": args.lora_plm,
        "lora_decoder": args.lora_decoder,
        "alpha": args.alpha,
        "beta": args.beta,
    }
    if args.resume:
        return wandb.init(
            entity="MaPra",
            project="monomer-structure-prediction",
            id=args.wandb_resume_id,
            resume="must",
            config=config,
        )
    return wandb.init(
        entity="MaPra",
        project="monomer-structure-prediction",
        config=config
    )


def get_dataset(args):
    if args.model == "cnn":
        use_file = args.emb_file is not None
        emb_source = args.emb_file if use_file else args.emb_dir

        train_ids, val_ids, test_ids = load_split_file(args.split_file)

        return create_cnn_data_loaders(
            emb_source, args.tok_jsonl, train_ids, val_ids, test_ids, args.batch, use_file
        )
    elif args.model == "t_fold" or args.model == "final":
        final_model = args.model == "final"
        return create_tfold_data_loaders(args.data_dir, args.batch, args.val_batch, args.lora_plm, args.bio2token,
                                         final_model)


def print_epoch_end(score_dict, epoch, start):
    parts = [f"Epoch {epoch:02d} | duration {time.time() - start:.2f}s"]

    # a defined order of keys, because it is nice
    key_order = ["loss", "acc", "mse", "val_loss", "val_acc", "val_lddt", "val_mse"]

    # add known values
    for key in key_order:
        if key in score_dict and score_dict[key] is not None:
            parts.append(f"{key}: {score_dict[key]:.4f}")

    # just in case I forgot some
    for key, value in score_dict.items():
        if key not in key_order and value is not None:
            parts.append(f"{key}: {value}")

    print(" | " + " | ".join(parts))


def get_unique_folder(base_path):
    counter = 0
    new_path = base_path
    while os.path.exists(new_path):
        counter += 1
        new_path = f"{base_path}_{counter}"
    return new_path

def main(args):
    # init wand db
    run = None
    if not args.no_wandb:
        # wandb.login(key=open("wandb_key").read().strip())
        run = init_wand_db(args)
    start = time.time()
    print("preparing data...")
    # load dataset
    train_loader, val_loader = get_dataset(args)
    print(f"done: {time.time() - start:.2f}s")
    print("preparing model...")
    model, optimizer = get_model(args)

    # init output
    folder_name=f"{model.model_name}_lr{args.lr}"
    out_folder = (os.path.join(args.out_folder, folder_name))
    # here we check if the folder already exists and if so add number at the end of it
    out_folder = get_unique_folder(out_folder)
    os.makedirs(out_folder, exist_ok=True)
    print(f"saving model to{out_folder}")

    # init important metric based on if the models need to optimize or minimize
    if model.maximize:
        best_val_score = -float('inf')
    else:
        best_val_score = float('inf')
    patience_ctr = 0

    # start training
    print("starting training...")
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_score_dict = model.run_epoch(train_loader, optimizer=optimizer, device=args.device)
        val_score_dict = model.run_epoch(val_loader, device=args.device)
        score_dict = train_score_dict | val_score_dict

        if not args.no_wandb:
            run.log(score_dict)

        # here we just print some basic statistic
        print_epoch_end(score_dict, epoch, start)

        # save model
        model.save(out_folder, suffix="_latest")
        # Early stopping check
        new_score = score_dict[model.key_metric]
        if ((model.maximize and new_score > best_val_score)
                or (not model.maximize and new_score < best_val_score)):
            best_val_score = new_score
            patience_ctr = 0
            # Save best model
            model.save(out_folder)
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

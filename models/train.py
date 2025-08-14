import argparse
import json
import os
import random
import time

import numpy as np
import wandb
import torch
from biotite.structure import AtomArray, lddt
from biotite.structure.io.pdb import PDBFile
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

import matplotlib.pyplot as plt

from models.bio2token.data.utils.utils import pad_and_stack_tensors
from models.model_utils import SmoothLDDTLoss, TmLossModule, model_prediction_to_atom_array
from models.simple_classifier.simple_classifier import ResidueTokenCNN
from models.datasets.datasets import ProteinPairJSONL, ProteinPairJSONL_FromDir, PAD_LABEL, StructureAndTokenSet, \
    TokenSet, StructureSet
from models.end_to_end.whole_model import TFold, FinalModel, FinalFinalModel

# ------------------------------------------------------------
#  Utility functions
# ------------------------------------------------------------
LOSS_REGISTRY = {
    "lddt": SmoothLDDTLoss(),
    "tm": TmLossModule(),
}


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
    parser.add_argument("--model", type=str, default="cnn",
                        help="type of model to use, options: cnn, tfold, final, final_final")
    parser.add_argument("--resume", type=str, help="path to existing trainings folder with the latest model")
    parser.add_argument("--wandb_resume_id", type=str, help="W&B id of an existing wandb run")
    parser.add_argument("--load_optimizer", action="store_true",
                        help="Load optimizer state when resuming training")
    # end to end exclusive setting
    parser.add_argument("--lora_plm", action="store_true", help=" use lora to finetune the plm")
    parser.add_argument("--lora_decoder", action="store_true", help=" use lora to finetune the plm")
    parser.add_argument("--bio2token", action="store_true", help="use bio2token instead of foldtoken in tfold")
    parser.add_argument("--alpha", type=int, help="weight of the lddt loss", default=1)
    parser.add_argument("--beta", type=int, help="weight of the encoding loss", default=1)

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
    parser.add_argument("--losses", nargs="+", default=["lddt"],
                        choices=list(LOSS_REGISTRY.keys()),
                        help="Loss functions to use")
    parser.add_argument("--loss_weights", nargs="+", type=float, default=[1.0],
                        help="Weights for each loss function")
    return parser.parse_args()


def create_tfold_data_loaders(data_dir, batch_size, val_batch_size, fine_tune_plm, bio2token=False, model_type="final"):
    # train_dir = os.path.join(data_dir, "train")
    train_dir = os.path.join(data_dir, "val")
    val_dir = os.path.join(data_dir, "val")

    if model_type == "final":
        token_type = "encoding"
        train_set = StructureAndTokenSet(train_dir, token_type, precomputed_embeddings=not fine_tune_plm)
        val_set = StructureAndTokenSet(val_dir, token_type, precomputed_embeddings=not fine_tune_plm)
        # adapt padding function to whether the model will receive embeddings or seqs as input
        collate_function = collate_seq_struc_tok_batch if fine_tune_plm else collate_emb_struc_tok_batch
        return (
            DataLoader(train_set, batch_size=batch_size, collate_fn=collate_function, shuffle=True),
            DataLoader(val_set, batch_size=batch_size, collate_fn=collate_function)
        )
    elif model_type == "tfold":
        token_type = "bio2token" if bio2token else "foldtoken"
        train_set = TokenSet(train_dir, token_type=token_type, precomputed_embeddings=not fine_tune_plm)
        val_set = StructureAndTokenSet(val_dir, token_type, precomputed_embeddings=not fine_tune_plm)
        # get padding function based on input
        val_collate_function = collate_seq_struc_tok_batch if fine_tune_plm else collate_emb_struc_tok_batch
        train_collate_function = collate_seq_tok_batch if fine_tune_plm else collate_emb_tok_batch
        return (
            DataLoader(train_set, batch_size=batch_size, collate_fn=train_collate_function, shuffle=True),
            DataLoader(val_set, batch_size=val_batch_size, collate_fn=val_collate_function)
        )
    elif model_type == "final_final":
        train_set = StructureSet(train_dir, precomputed_embeddings=not fine_tune_plm)
        val_set = StructureSet(val_dir, precomputed_embeddings=not fine_tune_plm)
        collate_function = collate_seq_struc if fine_tune_plm else collate_emb_struc
        return (
            DataLoader(train_set, batch_size=batch_size, collate_fn=collate_function, shuffle=True),
            DataLoader(val_set, batch_size=batch_size, collate_fn=collate_function)
        )


def create_cnn_data_loaders(emb_source, tok_jsonl, train_ids, val_ids, test_ids, batch_size, use_single_file):
    DSClass = ProteinPairJSONL if use_single_file else ProteinPairJSONL_FromDir
    train_ds = DSClass(emb_source, tok_jsonl, train_ids)
    val_ds = DSClass(emb_source, tok_jsonl, val_ids)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_emb_tok_batch),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_emb_tok_batch),
    )


def find_latest_file(directory):
    for filename in os.listdir(directory):
        if "_latest" in filename:
            return os.path.join(directory, filename)
    return None


def build_t_fold(lora_plm, hidden, kernel_size, dropout, device, resume):
    if resume:
        model = TFold.load_tfold(resume, device=device).to(device)
    else:
        model = TFold(hidden=hidden, kernel_sizes=kernel_size, dropout=dropout, device=device, use_lora=lora_plm).to(
            device)
    return model


def build_final_model(lora_plm, lora_decoder, hidden, kernel_size, dropout, device, alpha, beta, resume):
    if resume:
        model = FinalModel.load_final(resume, device=device).to(device)
        # this part overrides the alpha and beta value saved with the model
        # this was introduced because alpha and beta was originally not saved with the model
        model.set_alpha(alpha)
        model.set_beta(beta)
    else:
        model = FinalModel(hidden, kernel_sizes=kernel_size, plm_lora=lora_plm, decoder_lora=lora_decoder,
                           device=device,
                           dropout=dropout, alpha=alpha, beta=beta)
    return model


def build_final_final_model(lora_plm, lora_decoder, hidden, kernel_size, dropout, device, resume):
    if resume:
        model_file_path = find_latest_file(resume)
        model = FinalFinalModel.load_final_final(model_file_path, device=device).to(device)
    else:
        model = FinalFinalModel(hidden, kernel_sizes=kernel_size, plm_lora=lora_plm, decoder_lora=lora_decoder,
                                device=device,
                                dropout=dropout)
    return model


def build_cnn(d_emb, hidden, vocab_size, kernel_size, dropout, lr, device, resume):
    if resume:
        model = ResidueTokenCNN.load_cnn(resume)
    else:
        model = ResidueTokenCNN(d_emb, hidden, vocab_size, kernel_size, dropout).to(device)
    return model


def collate_seq_struc(batch):
    sequences, structures = zip(*batch)
    # Pad structure
    structures = pad_and_stack_tensors(structures, 0)
    return list(sequences), structures


def collate_emb_struc(batch):
    embs, structures = zip(*batch)
    embs_padded = pad_sequence(embs, batch_first=True)
    # Pad structure
    structures = pad_and_stack_tensors(structures, 0)
    return embs_padded, structures


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
# Save pdb snapshots during training (after every epoch 3 different proteins)
# ------------------------------------------------------------

def _select_first_n(dataset, n=3):
    """
    Returns list[(key, sample)] where sample is whatever the dataset returns.
    key is an ID if dataset has .ids, else the integer index.
    """
    out = []
    for idx in range(min(n, len(dataset))):
        key = getattr(dataset, "ids", [idx] * len(dataset))[idx] if hasattr(dataset, "ids") else idx
        out.append((key, dataset[idx]))
    return out


def _atomarray_from_coords(coords_np, atoms_per_res=4):
    """
    Build a simple AtomArray template (N, CA, C, O per residue) from coords.
    coords_np: (L*atoms_per_res, 3) numpy float array
    """
    coords_np = np.asarray(coords_np, dtype=np.float32)
    L = coords_np.shape[0] // atoms_per_res
    arr = AtomArray(L * atoms_per_res)
    arr.coord = coords_np

    # annotations
    atom_names = np.array(["N", "CA", "C", "O"])
    arr.atom_name = np.tile(atom_names, L)
    arr.res_id = np.repeat(np.arange(1, L + 1), atoms_per_res)
    arr.res_name = np.repeat(np.array(["ALA"]), L * atoms_per_res)  # placeholder
    arr.chain_id = np.repeat(np.array(["A"]), L * atoms_per_res)
    arr.element = np.tile(np.array(["N", "C", "C", "O"]), L)
    return arr


def _save_snapshot(sample, model, out_dir, tag, device, atoms_per_res=4):
    """
    sample: one item from val dataset. Accepts (model_in, structure) or (model_in, tokens, structure)
    Writes: <out_dir>/snapshots/epoch_XXX_<tag>.pdb
    """
    # unpack sample shapes
    if isinstance(sample, tuple) and len(sample) == 3:
        model_in, _, gt_struct = sample
    elif isinstance(sample, tuple) and len(sample) == 2:
        model_in, gt_struct = sample
    else:
        # If there’s no structure in the sample, we cannot build a template; fall back to zeros
        model_in, gt_struct = sample, None

    # pick forward function based on your models
    if hasattr(model, "plm_lora") and model.plm_lora:
        # model expects list[str] if finetuning PLM; else embeddings tensor
        if isinstance(model_in, torch.Tensor):
            forward_fn = getattr(model, "forward_from_embedding", None)
            assert forward_fn is not None, "Model needs forward_from_embedding() for embedding inputs"
            preds, final_mask, *_ = forward_fn(model_in.unsqueeze(0).to(device))
            # if we do not have the seqs derive their length from embeddings and then make placeholder seq
            true_length = (preds.abs().sum(-1) > 0).sum(dim=1)[0]
            # generate placeholder seq
            sequences = ["A" * true_length]

        else:
            preds, final_mask, *_ = model([model_in])  # list of one seq
            sequences = [model_in]
    else:
        # not finetuning PLM → prefer forward_from_embedding (embeddings) if available
        forward_fn = getattr(model, "forward_from_embedding", None) or getattr(model, "forward", None)
        inp = model_in.unsqueeze(0).to(device) if isinstance(model_in, torch.Tensor) else [model_in]
        preds, final_mask, *_ = forward_fn(inp)

    atom_array = model_prediction_to_atom_array(sequences, preds, final_mask)[0]

    # Write PDB
    snap_dir = os.path.join(out_dir, "snapshots")
    os.makedirs(snap_dir, exist_ok=True)
    pdb = PDBFile()
    pdb.set_structure(atom_array)
    pdb.write(os.path.join(snap_dir, f"{tag}.pdb"))

    # calculate and return score
    ref_atom_array = model_prediction_to_atom_array(sequences, gt_struct, final_mask)[0]
    lddt_score = lddt(ref_atom_array, atom_array)
    return lddt_score


# ------------------------------------------------------------
# Main workflow with early stopping & plots
# ------------------------------------------------------------


def get_model():
    global args
    match args.model:
        case "cnn":
            return build_cnn(
                args.d_emb, args.hidden, args.codebook_size, args.kernel_size, args.dropout, args.lr, args.device,
                args.resume
            )
        case "t_fold":
            return build_t_fold(args.lora_plm, args.hidden, args.kernel_size, args.dropout,
                                args.device, args.resume)
        case "final":
            return build_final_model(args.lora_plm, args.lora_decoder, args.hidden, args.kernel_size, args.dropout,
                                     args.device, args.alpha, args.beta, args.resume)
        case "final_final":
            return build_final_final_model(args.lora_plm, args.lora_decoder, args.hidden, args.kernel_size,
                                           args.dropout, args.device, args.resume)
        case _:
            raise NotImplementedError


def init_wand_db():
    global args
    config = {
        **vars(args)
    }
    if args.wandb_resume_id:
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


def get_dataset():
    global args
    if args.model == "cnn":
        use_file = args.emb_file is not None
        emb_source = args.emb_file if use_file else args.emb_dir

        train_ids, val_ids, test_ids = load_split_file(args.split_file)

        return create_cnn_data_loaders(
            emb_source, args.tok_jsonl, train_ids, val_ids, test_ids, args.batch, use_file
        )
    else:
        return create_tfold_data_loaders(args.data_dir, args.batch, args.val_batch, args.lora_plm, args.bio2token,
                                         args.model)


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


def save_training_state(model, optimizer, scheduler, out_folder):
    optimizer_path = os.path.join(out_folder, f"optimizer.pt")
    scheduler_path = os.path.join(out_folder, f"scheduler.pt")

    torch.save(optimizer.state_dict(), optimizer_path)
    torch.save(scheduler.state_dict(), scheduler_path)
    model.save(out_folder, suffix="_latest")


def _save_reference_pdb(sample, out_dir, tag, atoms_per_res=4):
    """
    Save the ground-truth structure from a dataset sample as a PDB.

    sample: one item from val dataset. Accepts (model_in, structure) or (model_in, tokens, structure)
    Writes: <out_dir>/references/<tag>.pdb
    """
    # unpack like _save_snapshot()
    if isinstance(sample, tuple) and len(sample) == 3:
        model_in, _, gt_struct = sample
    elif isinstance(sample, tuple) and len(sample) == 2:
        model_in, gt_struct = sample
    else:
        gt_struct = None

    if gt_struct is None:
        print(f"[ref] {tag}: no GT structure in sample — skipped")
        return

    # If your dataset already returns an AtomArray, keep it; else build one from coords
    if isinstance(gt_struct, AtomArray):
        arr = gt_struct
    else:
        B, L = gt_struct.shape
        final_mask = torch.zeros(B, L * atoms_per_res, dtype=torch.bool)
        for i in range(len(model_in)):
            final_mask[i, :len(model_in[i]) * atoms_per_res] = True
        arr = model_prediction_to_atom_array(model_in, gt_struct, final_mask)[0]

    ref_dir = os.path.join(out_dir, "references")
    os.makedirs(ref_dir, exist_ok=True)
    pdb = PDBFile()
    pdb.set_structure(arr)
    pdb.write(os.path.join(ref_dir, f"{tag}.pdb"))


def main():
    global args
    # init wand db
    run = None
    if not args.no_wandb:
        # wandb.login(key=open("wandb_key").read().strip())
        run = init_wand_db()
    start = time.time()
    print("preparing data...")
    # load dataset
    train_loader, val_loader = get_dataset()
    print(f"done: {time.time() - start:.2f}s")

    snapshot_cache = _select_first_n(val_loader.dataset, n=10)
    print(f"[snapshots] fixed samples from validation: {[k for k, _ in snapshot_cache]}")

    print("preparing model...")
    model = get_model()
    # create scheduler and optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01
    )
    if args.resume and args.load_optimizer:
        optimizer_path = os.path.join(args.resume, "optimizer.pt")
        if os.path.exists(optimizer_path):
            optimizer.load_state_dict(torch.load(optimizer_path, map_location=args.device))
            # overwrite lr because it might have gotten to low
            optimizer.param_groups[0]["lr"] = args.lr
        else:
            print("Warning: optimizer states not found. Starting fresh.")

    # init scheduler
    warmup_steps = 1000
    total_steps = args.epochs * len(train_loader)
    # Warm-up: start_lr=0 → base_lr over `warmup_steps`
    scheduler1 = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
    # Decay: cosine annealing after warm-up
    scheduler2 = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
    scheduler = SequentialLR(
        optimizer,
        schedulers=[scheduler1, scheduler2],
        milestones=[warmup_steps]
    )

    # init losses
    loss_modules = [LOSS_REGISTRY[loss_name].to(args.device) for loss_name in args.losses]

    # init output
    folder_name = f"{model.model_name}_lr{args.lr}"
    out_folder = (os.path.join(args.out_folder, folder_name))
    # here we check if the folder already exists and if so add number at the end of it
    out_folder = get_unique_folder(out_folder)
    os.makedirs(out_folder, exist_ok=True)
    print(f"saving model to {out_folder}")

    # save the three fixed reference PDBs once
    for key, sample in snapshot_cache:
        ref_tag = f"ref_{str(key)}"
        _save_reference_pdb(sample, out_folder, ref_tag)
    print(f"[references] saved GT PDBs to {os.path.join(out_folder, 'references')}")

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
        train_score_dict = model.run_epoch(train_loader, loss_modules, args.loss_weights, optimizer=optimizer,
                                           scheduler=scheduler,
                                           device=args.device)
        val_score_dict = model.run_epoch(val_loader, loss_modules, args.loss_weights, device=args.device)
        score_dict = train_score_dict | val_score_dict

        model.eval()  # fine because it automatically starts train mode again in next epoch
        with torch.no_grad():
            lddt_sum = 0
            for key, sample in snapshot_cache:
                tag = f"epoch_{epoch:03d}_{str(key)}"
                lddt_sum += _save_snapshot(sample, model, out_folder, tag, device=args.device)
        score_dict["biotite_lddt"] = lddt_sum / len(snapshot_cache)

        # log lr
        score_dict["lr"] = optimizer.param_groups[0]["lr"]
        if not args.no_wandb:
            run.log(score_dict)

        # here we just print some basic statistic
        print_epoch_end(score_dict, epoch, start)

        # save state of training so we can always continue
        save_training_state(model, optimizer, scheduler, out_folder)

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

    print(f"Saved best model in → {out_folder}")


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
    global args
    args = parse_args()
    main()

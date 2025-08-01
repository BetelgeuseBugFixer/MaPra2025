import argparse
import builtins
import copy
import gzip
import json
import shutil
from pathlib import Path
import os

import numpy as np
import torch
from biotite.structure.io.pdb import PDBFile
from biotite.structure import lddt
from biotite.structure.filter import _filter_atom_names
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from torch.utils.data import DataLoader

from benchmark.model_predictions import prepare_data, plot_smooth_lddt
from models.bio2token.data.utils.tokens import PAD_CLASS
from models.bio2token.decoder import Bio2tokenDecoder, load_bio2token_decoder_and_quantizer, load_bio2token_encoder, \
    load_bio2token_model
from models.bio2token.losses.rmsd import RMSDConfig, RMSD
from models.bio2token.models.autoencoder import AutoencoderConfig, Autoencoder
from models.bio2token.utils.configs import utilsyaml_to_dict, pi_instantiate
from models.collab_fold.esmfold import EsmFold
from models.model_utils import _masked_accuracy, calc_token_loss, calc_lddt_scores, SmoothLDDTLoss, \
    batch_pdbs_for_bio2token, print_trainable_parameters, masked_mse_loss, batch_pdb_dicts
from models.prot_t5.prot_t5 import ProtT5
from models.datasets.datasets import PAD_LABEL, StructureAndTokenSet, StructureSet
from models.simple_classifier.simple_classifier import ResidueTokenCNN
from models.bio2token.data.utils.utils import pdb_2_dict, uniform_dataframe, compute_masks, pad_and_stack_batch, \
    filter_batch, pad_and_stack_tensors

from models.foldtoken_decoder.foldtoken import FoldToken
from models.end_to_end.whole_model import TFold, FinalModel, FinalFinalModel
from transformers import T5EncoderModel, T5Tokenizer
from hydra_zen import load_from_yaml, builds, instantiate

from models.train import collate_emb_struc_tok_batch, collate_seq_struc_tok_batch, collate_seq_struc
from transformers import AutoTokenizer, EsmForProteinFolding

from utils.generate_new_data import filter_pdb_dict


def load_prot_from_pdb(pdb_file):
    # load
    file = PDBFile.read(pdb_file)
    array_stack = file.get_structure(model=1)
    # filter canonical atoms
    return array_stack[_filter_atom_names(array_stack, ["N", "CA", "C", "O"])]


def load_casp_tokens(token_jsonl):
    token_data = {}

    with open(token_jsonl, "r") as f:
        for line in f:
            entry = json.loads(line)
            token_data.update(entry)
    return token_data


def decode_atom_coordinates(vq_codes, foldtoken_model):
    protein = foldtoken_model.decode_single_prot(vq_codes, "test.pdb")
    X, _, _ = protein.to_XCS(all_atom=False)
    return X


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


def sanity_test():
    device = "cuda"
    model = FoldToken(device=device)
    test_pdb = "tokenizer_benchmark/casps/casp14/T1024-D1.pdb"
    test_tokens = load_casp_tokens("data/casp14_test/casp14_tokens.jsonl")["T1024-D1"]["vqid"]
    # test_pdb = "data/lys6/pdb_files/pdb/AF-A0A0A0AJ30-F1-model_v4.pdb"
    # test_tokens = load_casp_tokens("/mnt/data/lys6/tokens_sanitized.jsonl")["A0A0A0AJ30"]["vqid"]

    # get ref
    ref_protein = load_prot_from_pdb(test_pdb)

    # encode and decode model
    locally_encoded_vq_codes = model.encode_pdb(test_pdb)
    locally_encoded_coords = decode_atom_coordinates(locally_encoded_vq_codes, model).detach().squeeze(0).reshape(-1,
                                                                                                                  3).cpu().numpy()

    # also load prev computed tokens
    pre_encoded_vq_codes = torch.tensor(test_tokens, dtype=torch.long, device=device)
    pre_encoded_coords = decode_atom_coordinates(pre_encoded_vq_codes, model).detach().squeeze(0).reshape(-1,
                                                                                                          3).cpu().numpy()

    # compare vq codes
    print(f"identical vq codes: {((locally_encoded_vq_codes == pre_encoded_vq_codes).float().mean()) * 100:.2f}%")

    # get scores
    print(f"locally encoded: {lddt(ref_protein, locally_encoded_coords)}")
    print(f"pre encoded: {lddt(ref_protein, pre_encoded_coords)}")
    print(f"locally encoded: {lddt(ref_protein, locally_encoded_coords, aggregation='residue')}")
    print(f"pre encoded: {lddt(ref_protein, pre_encoded_coords, aggregation='residue')}")


three_to_one_dict = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D',
    'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G',
    'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
    'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S',
    'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}


def get_seq_from_pdb(pdb_file: str) -> str | None:
    seq = ""
    chain_id = None
    with open(pdb_file, 'r') as f:
        for line in f.readlines():
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                current_chain = line[21]
                if chain_id is None:
                    chain_id = current_chain
                else:
                    if current_chain != chain_id:
                        return None
                res_name = line[17:20].strip()
                seq += three_to_one_dict[res_name]
            elif line.startswith("ENDMDL"):
                break
    return seq


def sample_workflow():
    device = "cuda"
    # init models
    plm = ProtT5(device=device).to(device)
    decoder = FoldToken(device=device)

    # define input
    # test_pdbs=["tokenizer_benchmark/casps/casp14/T1024-D1.pdb","tokenizer_benchmark/casps/casp14/T1024-D2.pdb"]
    ly6_pdb_dir = "/mnt/data/lys6/pdb_files/pdb"
    # open split
    with open("/mnt/data/lys6/split_ids.json", "r") as split_file:
        split = json.load(split_file)
    train_set = set(split["train"])
    test_set = set(split["test"])
    val_set = set(split["val"])

    # define sets
    all_pdb_files = [pdb for pdb in os.listdir(ly6_pdb_dir) if pdb.endswith(".pdb")]
    train_pdbs = [os.path.join(ly6_pdb_dir, pdb) for pdb in all_pdb_files if pdb.split("-")[1] in train_set]
    val_pdbs = [os.path.join(ly6_pdb_dir, pdb) for pdb in all_pdb_files if pdb.split("-")[1] in val_set]
    test_pdbs = [os.path.join(ly6_pdb_dir, pdb) for pdb in all_pdb_files if pdb.split("-")[1] in test_set]
    sets = [train_pdbs, val_pdbs, test_pdbs]
    set_names = ["train", "val", "test"]
    # define models
    overfitted_cnn = ResidueTokenCNN.load_cnn("train_run/cnn_k9_7_5_h4096_4096_2048.pt").to(device)
    non_overfitted_large_cnn = ResidueTokenCNN.load_cnn("train_run/cnn_k5_5_5_5_h4096_2048_2048_2048.pt").to(device)
    non_overfitted_smol_cnn = ResidueTokenCNN.load_cnn("train_run/cnn_k5_5_h2048_2048.pt").to(device)
    models = [overfitted_cnn, non_overfitted_large_cnn, non_overfitted_smol_cnn]
    model_names = ["overfitted", "large not overfitted", "small not overfitted"]
    # models = [ResidueTokenCNN(1024, [2048, 2048], 1024, [5, 5]).to(device)]
    # model_names = ["random"]
    # run analysis
    plm.eval()
    decoder.eval()
    with torch.no_grad():
        for cnn, model_name in zip(models, model_names):
            cnn.eval()
            for current_set, set_name in zip(sets, set_names):
                lddt_scores = []
                # create batches
                batch_size = 16
                for batch in range(0, len(current_set), batch_size):
                    pdb_batch = current_set[batch:batch + batch_size]
                    seqs = [get_seq_from_pdb(pdb) for pdb in pdb_batch]
                    true_lengths = [len(seq) for seq in seqs]
                    # print(f"true_lengths: {true_lengths}")
                    # print(seqs)
                    seqs = [raw_seq.translate(str.maketrans('UZO', 'XXX')) for raw_seq in seqs]
                    seqs = [" ".join(raw_seq) for raw_seq in seqs]
                    # print(seqs)
                    # run through model
                    emb = plm(seqs)
                    emb = [emb[i, : length] for i, length in enumerate(true_lengths)]
                    emb = pad_sequence(emb, batch_first=True)
                    # print(f"emb: {len(emb[0])}")
                    # print(f"emb: {emb.shape}")
                    # print("=" * 20)
                    logits = cnn(emb)  # shape: (B, L, vocab_size)
                    # print(f"logits: {logits.shape}")
                    pred_tokens = logits.argmax(dim=-1)
                    # print(f"tokens: {pred_tokens.shape}")
                    pred_tokens = pred_tokens.to(device)
                    # pre_encoded_vq_codes = torch.tensor(load_casp_tokens("data/casp14_test/casp14_tokens.jsonl")["T1024-D1"]["vqid"],
                    #                                     dtype=torch.long, device=device)

                    # batch for foldtoken
                    vq_codes = []
                    batch_ids = []
                    chain_encodings = []
                    for i, protein_token in enumerate(pred_tokens):
                        L = true_lengths[i]
                        protein_token_without_padding = protein_token[:L]
                        vq_codes.append(protein_token_without_padding)
                        batch_ids.append(torch.full((L,), i, dtype=torch.long, device=protein_token.device))
                        chain_encodings.append(torch.full((L,), 1, dtype=torch.long, device=protein_token.device))

                    vq_codes_cat = torch.cat(vq_codes, dim=0)
                    batch_ids_cat = torch.cat(batch_ids, dim=0)
                    chain_encodings_cat = torch.cat(chain_encodings, 0)
                    # print(f"vq_codes_cat:\n{vq_codes_cat.shape}\nbatch_ids_cat:\n{batch_ids_cat.shape}\nchain_encodings_cat:\n{chain_encodings_cat.shape}")

                    proteins = decoder.decode(vq_codes_cat, chain_encodings_cat, batch_ids_cat)
                    # print("*"*20)
                    for protein, pdb in zip(proteins, pdb_batch):
                        # print(protein)
                        X, _, _ = protein.to_XCS(all_atom=False)
                        X = X.detach().squeeze(0).reshape(-1, 3).cpu().numpy()
                        try:
                            ref_protein = load_prot_from_pdb(pdb)
                        except Exception as e:
                            print(f"Error loading PDB {pdb}: {e}")
                            continue
                        lddt_score = lddt(ref_protein, X)
                        lddt_scores.append(lddt_score)
                        # print(lddt_score)
                print(f"{model_name} - {set_name}: {sum(lddt_scores) / len(lddt_scores)}")


def test_loading():
    test_pdbs = ["tokenizer_benchmark/casps/casp14/T1024-D1.pdb", "tokenizer_benchmark/casps/casp14/T1026-D1.pdb"]
    seqs = [get_seq_from_pdb(pdb_path) for pdb_path in test_pdbs]
    t_fold = TFold([1024], device="cuda").to("cuda").eval()
    print(hash(str(t_fold.decoder.state_dict())))
    proteins, tokens = t_fold(seqs)
    print(tokens)
    for protein, pdb in zip(proteins, test_pdbs):
        X, _, _ = protein.to_XCS(all_atom=False)
        X = X.detach().squeeze(0).reshape(-1, 3).cpu().numpy()
        try:
            ref_protein = load_prot_from_pdb(pdb)
        except Exception as e:
            print(f"Error loading PDB {pdb}: {e}")
            continue
        print(lddt(ref_protein, X))
    print("*" * 11)
    t_fold.save(".")
    t_fold = TFold.load_tfold(f"{t_fold.model_name}.pt", device="cuda").to("cuda").eval()
    print(hash(str(t_fold.decoder.state_dict())))
    proteins, tokens = t_fold(seqs)
    print(tokens)
    for protein, pdb in zip(proteins, test_pdbs):
        X, _, _ = protein.to_XCS(all_atom=False)
        X = X.detach().squeeze(0).reshape(-1, 3).cpu().numpy()
        try:
            ref_protein = load_prot_from_pdb(pdb)
        except Exception as e:
            print(f"Error loading PDB {pdb}: {e}")
            continue
        print(lddt(ref_protein, X))


def training_test():
    device = "cuda"
    # define input
    test_pdbs = ["tokenizer_benchmark/casps/casp14/T1024-D1.pdb", "tokenizer_benchmark/casps/casp14/T1026-D1.pdb"]
    seqs = [get_seq_from_pdb(pdb_path) for pdb_path in test_pdbs]
    ref_proteins = [load_prot_from_pdb(test_pdb) for test_pdb in test_pdbs]
    fold_model = FoldToken(device=device)
    tokens_reference = [fold_model.encode_pdb(test_pdb) for test_pdb in test_pdbs]
    tokens_reference_padded = pad_sequence(tokens_reference, batch_first=True, padding_value=PAD_LABEL)
    protein_reference = [load_prot_from_pdb(test_pdb) for test_pdb in test_pdbs]
    # define model
    model = TFold(hidden=[2048, 2048], kernel_sizes=[5, 5], use_lora=True).to(device)
    # define learning stuff
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_LABEL)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # get predicted tokens + protein structure
    model.train()
    for _ in range(100):
        protein_predictions, logits = model(seqs)
        # get loss
        mask = (tokens_reference_padded != PAD_LABEL)
        loss = calc_token_loss(criterion, logits, tokens_reference_padded)
        # backpropagate
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # get lddt
        lddt_scores = calc_lddt_scores(protein_predictions, protein_reference)
        # log
        batch_size = len(seqs)
        total_loss = loss.detach().item() * batch_size
        total_acc = _masked_accuracy(logits, tokens_reference_padded, mask) * batch_size
        print(f"token loss: {loss}")
        print(lddt_scores)


def protT5_test():
    device = "cuda"
    transformer_link = "Rostlab/prot_t5_xl_half_uniref50-enc"
    tokenizer = T5Tokenizer.from_pretrained(transformer_link, do_lower_case=False, use_fast=False)
    model = T5EncoderModel.from_pretrained(transformer_link).to(device)

    seqs = ["AGHGFEFF", "FGTGAD"]
    true_lengths = [len(seq) for seq in seqs]
    print(f"lengths: {true_lengths}")
    # prepare seqs
    x = [" ".join(seq.translate(str.maketrans('UZO', 'XXX'))) for seq in seqs]
    encoding = tokenizer.batch_encode_plus(
        x,
        add_special_tokens=True,
        padding='longest',
        return_tensors='pt'
    )
    print(f"encoding:\n{encoding}")
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    print(f"attention_mask:\n{attention_mask}")
    relevant_mask = ((input_ids != 0) & (input_ids != 1)).unsqueeze(-1)
    outputs = model(input_ids, attention_mask=attention_mask)
    hidden = outputs.last_hidden_state
    print(f"hidden.shape{hidden.shape}")
    print(hidden)
    new_hidden = hidden * relevant_mask
    new_hidden = new_hidden[:, :-1, :]
    print(f"new_hidden.shape{new_hidden.shape}")
    print(new_hidden)


def bio2token_test():
    device = "cuda"
    model_configs = load_from_yaml("models/bio2token/files/model.yaml")["model"]
    model_config = pi_instantiate(AutoencoderConfig, model_configs)
    model = Autoencoder(model_config)
    state_dict = torch.load("models/bio2token/files/epoch=0243-val_loss_epoch=0.71-best-checkpoint.ckpt")["state_dict"]
    # Remove 'model.' prefix from keys if present
    state_dict_bis = {k.replace("model.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict_bis)
    model.to(device)
    print("inited model")

    pdb_path = "tokenizer_benchmark/casps/casp14/T1024-D1.pdb"
    pdb_dict = pdb_2_dict(
        pdb_path,
        None,
    )
    structure, unknown_structure, residue_name, residue_ids, token_class, atom_names_reordered = uniform_dataframe(
        pdb_dict["seq"],
        pdb_dict["res_types"],
        pdb_dict["coords_groundtruth"],
        pdb_dict["atom_names"],
        pdb_dict["res_atom_start"],
        pdb_dict["res_atom_end"],
    )
    batch = {
        "structure": torch.tensor(structure).float(),
        "unknown_structure": torch.tensor(unknown_structure).bool(),
        "residue_ids": torch.tensor(residue_ids).long(),
        "token_class": torch.tensor(token_class).long(),
    }
    batch = {k: v[~batch["unknown_structure"]] for k, v in batch.items()}
    batch = compute_masks(batch, structure_track=True)
    batch = {k: v[None].to(device) for k, v in batch.items()}
    batch = model.encoder(batch)
    print(
        f"5 batch:\nindices-{batch['indices'].shape}\n{batch['indices']}\nencoding-{batch['encoding'].shape}\n{batch['encoding']}\neos_mask-{batch['eos_pad_mask'].shape}\n{batch['eos_pad_mask']}")
    batch = model.decoder(batch)
    print(f"6 batch:\n{batch['decoding'].shape}\n{batch['decoding']}")


def batched_bio2token():
    device = "cuda"
    model_configs = load_from_yaml("models/bio2token/files/model.yaml")["model"]
    model_config = pi_instantiate(AutoencoderConfig, model_configs)
    model = Autoencoder(model_config)
    state_dict = torch.load("models/bio2token/files/epoch=0243-val_loss_epoch=0.71-best-checkpoint.ckpt")["state_dict"]
    # Remove 'model.' prefix from keys if present
    state_dict_bis = {k.replace("model.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict_bis)
    model.to(device)

    # define test_pbs
    test_pdbs = ["tokenizer_benchmark/casps/casp14_backbone/T1024-D1.pdb",
                 "tokenizer_benchmark/casps/casp14_backbone/T1026-D1.pdb"]
    # batch
    # Prepare lists for batch processing
    # structure, unknown_structure, residue_name, residue_ids, token_class, atom_names_reordered

    batch = batch_pdbs_for_bio2token(test_pdbs, device)

    batch = model.encoder(batch)
    print(
        f"preprocessed batch:\nindices-{batch['indices'].shape}\n{batch['indices']}\nencoding-{batch['encoding'].shape}\n{batch['encoding']}\neos_mask-{batch['eos_pad_mask'].shape}\n{batch['eos_pad_mask']}")
    batch = model.decoder(batch)
    print(f"processed batch:\n{batch['decoding'].shape}\n{batch['decoding']}")


def get_padded_ground_truths(pdbs):
    batch = []
    dicts = [pdb_2_dict(pdb) for pdb in pdbs]
    for pdb_dict in dicts:
        structure, unknown_structure, residue_name, residue_ids, token_class, atom_names_reordered = uniform_dataframe(
            pdb_dict["seq"],
            pdb_dict["res_types"],
            pdb_dict["coords_groundtruth"],
            pdb_dict["atom_names"],
            pdb_dict["res_atom_start"],
            pdb_dict["res_atom_end"],
        )
        batch_item = {
            "structure": torch.tensor(structure).float(),
            "unknown_structure": torch.tensor(unknown_structure).bool(),
            "residue_ids": torch.tensor(residue_ids).long(),
            "token_class": torch.tensor(token_class).long(),
        }
        batch_item = {k: v[~batch_item["unknown_structure"]] for k, v in batch_item.items()}

        batch.append(batch_item)

    # taken from config
    sequences_to_pad = {
        "structure": 0,
    }
    # batch = [pdb_2_dict(pdb) for pdb in test_pdbs]
    batch = filter_batch(batch, sequences_to_pad.keys())
    batch = pad_and_stack_batch(
        batch,
        sequences_to_pad,
        1
    )
    return batch["structure"]


def bio2token_workflow():
    device = "cuda"
    # define models
    plm = ProtT5(device=device).to(device)
    cnn = ResidueTokenCNN(1024, [2048, 2048], 4096, [5, 5], bio2token=True).to(device)
    decoder = Bio2tokenDecoder(device=device).to(device)
    # input:
    test_pdbs = ["tokenizer_benchmark/casps/casp14_backbone/T1024-D1.pdb",
                 "tokenizer_benchmark/casps/casp14_backbone/T1026-D1.pdb"]
    seqs = [get_seq_from_pdb(pdb) for pdb in test_pdbs]
    true_lengths = [len(seq) for seq in seqs]
    # run through model:
    x = [" ".join(seq.translate(str.maketrans('UZO', 'XXX'))) for seq in seqs]
    x = plm(x)
    x = cnn(x)
    x = x.argmax(dim=-1)
    # construct eos mask:
    B, L = x.shape
    eos_mask = torch.ones(B, L, dtype=torch.bool, device=x.device)  # alle True = gepaddet
    for i, length in enumerate(true_lengths):
        eos_mask[i, :length * 4] = False
    print(f"indices: {x.shape}\n{x}")
    print(f"eos: {eos_mask.shape}\n{eos_mask}")
    x = decoder(x, eos_mask=eos_mask)

    # define losses
    config = RMSDConfig(
        prediction_name="predictions",  # Key for accessing prediction data in the batch
        target_name="targets",  # Key for accessing target data in the batch
        mask_name="mask",  # Key for accessing an optional mask in the batch
    )

    rmsd_metric = RMSD(config, name="rmsd").to(device)

    lddt_loss_module = SmoothLDDTLoss().to(device)
    # get gt
    targets = get_padded_ground_truths(test_pdbs).to(device)
    target_mask = ~eos_mask
    # eval
    # rmsd
    print(f"targets: {targets.shape}\n{targets}")
    to_eval = {
        "predictions": x,
        "targets": targets,
        "mask": target_mask,
        "losses": {}
    }
    to_eval = rmsd_metric(to_eval)

    loss_value = to_eval["losses"]["rmsd"]  # → Tensor

    for i, val in enumerate(loss_value):
        print(f"loss[{i}]: {val.item()}")
    # lddt
    is_dna = torch.zeros((B, L), dtype=torch.bool, device=device)
    is_rna = torch.zeros((B, L), dtype=torch.bool, device=device)
    lddt_loss = lddt_loss_module(x, targets, is_dna, is_rna, target_mask)
    print(f"loss: {lddt_loss.item()}")


def get_protein_sizes_in_dataset(data_file="/mnt/data/large/subset/train/proteins.jsonl.gz", max_size=1_000):
    number_of_to_large_proteins = 0
    all_proteins = 0
    open_func = gzip.open if data_file.endswith('.gz') else builtins.open
    with  open_func(data_file, 'rt') as f:
        for line in f:
            values = json.loads(line)
            sequence_length = len((values['sequence']))
            if sequence_length > max_size:
                number_of_to_large_proteins += 1
                print(f"protein {values['id']} to long: {sequence_length}")
            all_proteins += 1

    print(f"done analysing {data_file}!")
    print(f"found {number_of_to_large_proteins} proteins larger then {max_size} in {all_proteins} proteins.")


def print_tensor(tensor, name):
    print(f"{name}-{tensor.shape}:\n{tensor}")


# def test_dataset_generation():
#     test_pdbs = ["tokenizer_benchmark/casps/casp14_backbone/T1024-D1.pdb",
#                  "tokenizer_benchmark/casps/casp14_backbone/T1026-D1.pdb"]
#     seqs= [get_seq_from_pdb(pdb) for pdb in test_pdbs]
#     seq_lengths=[len(seq) for seq in seqs]
#     embeddings, bio2token, foldtoken = process_batch(test_pdbs, seqs)
#     print_tensor(embeddings,"embeddings")
#     print_tensor(bio2token,"bio2token")
#     print_tensor(foldtoken,"foldtoken")
#     print(seq_lengths)

def test_new_model():
    device = "cuda"
    model = FinalModel([12_000, 8_192, 2_048], device=device, kernel_sizes=[3, 1, 1], dropout=0.0, decoder_lora=True)
    # input:
    # test_pdbs = ["tokenizer_benchmark/casps/casp14_backbone/T1024-D1.pdb",
    #              "tokenizer_benchmark/casps/casp14_backbone/T1026-D1.pdb"]
    test_pdbs = ["tokenizer_benchmark/casps/casp15_backbone/T1129s2-D1.pdb"]

    # prepare input
    pdb_dicts = [pdb_2_dict(pdb) for pdb in test_pdbs]
    pdb_dicts = [filter_pdb_dict(pdb_dict) for pdb_dict in pdb_dicts]
    seqs = [get_seq_from_pdb(pdb) for pdb in test_pdbs]

    # get bio2token out
    bio2token_batch = batch_pdb_dicts(pdb_dicts,device)
    bio2token_model = load_bio2token_model().to(device)
    with torch.no_grad():
        solution=bio2token_model(bio2token_batch)


    # structure_tensor = torch.as_tensor(np.array(structure)).unsqueeze(0).to(device)

    # define labels
    targets = get_padded_ground_truths(test_pdbs).to(device)
    # get 128 vector
    encoder = load_bio2token_encoder()
    encoder.to(device)
    bio2token_batch = batch_pdbs_for_bio2token(test_pdbs, device)
    bio2token_batch = encoder(bio2token_batch)
    gt_vector = bio2token_batch["encoding"].detach()

    # prepare training
    lddt_loss_module = SmoothLDDTLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    model.train()
    # run through model:
    for epoch in range(1):
        optimizer.zero_grad()
        predictions, final_mask, cnn_out = model.forward(seqs)
        # vector_loss = masked_mse_loss(cnn_out, gt_vector, final_mask)
        # if epoch==0 or epoch==100:
        #     print_tensor(cnn_out,"predictions")
        #     print("***"*11)
        #     print_tensor(gt_vector,"targets")
        #     print("***" * 11)
        # print("vector loss:", vector_loss.item())
        # vector_loss.backward()
        B, L, _ = predictions.shape
        # lddt
        is_dna = torch.zeros((B, L), dtype=torch.bool, device=device)
        is_rna = torch.zeros((B, L), dtype=torch.bool, device=device)


        lddt_loss = lddt_loss_module(predictions, targets, is_dna, is_rna, final_mask)
        # print(f"loss: {lddt_loss.item()}")
        lddt_loss.backward()
        loss = F.mse_loss(predictions[final_mask], targets[final_mask])
        #loss.backward()
        # print(loss.item())
        #total_loss = vector_loss + lddt_loss
        # total_loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        bio2token_loss=lddt_loss_module(solution["decoding"],targets,is_dna, is_rna,~solution["eos_pad_mask"]).item()
        print(f"bio2token loss: {bio2token_loss}")
        print(f"lddt loss: {lddt_loss.item()} | mse: {loss.item()}")
        print(model.decoder.decoder.decoder(bio2token_batch["encoding"], bio2token_batch["eos_pad_mask"]))
    print("done")


def test_foldtoken_model():
    device = "cuda"
    model = TFold([1000], device=device, bio2token=True).to(device)
    print("init model done")
    dataset = StructureAndTokenSet("/mnt/data/large/subset2/val", "bio2token", precomputed_embeddings=True)
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_emb_struc_tok_batch)
    print("loaded dataset")
    lddt_loss_module = SmoothLDDTLoss().to(device)
    with torch.no_grad():
        for emb, tokens, structure in loader:
            # get predictions
            emb = emb.to(device)
            tokens = tokens.to(device)
            structure = structure.to(device)
            mask = (tokens != PAD_LABEL)
            protein_predictions, logits, atom_mask = model.forward_from_embedding(emb)
            # get loss and score
            B, L, _ = protein_predictions.shape
            is_dna = torch.zeros((B, L), dtype=torch.bool, device=device)
            is_rna = torch.zeros((B, L), dtype=torch.bool, device=device)
            print_tensor(protein_predictions, "protein_predictions")
            print_tensor(structure, "structure")
            print_tensor(atom_mask, "mask")
            print_tensor(is_dna, "dna")
            lddt_loss = lddt_loss_module(protein_predictions, structure, is_dna, is_rna, atom_mask)
            print(lddt_loss.item())
            break


def look_at_weird_lddt():
    device = "cuda"
    dataset = StructureAndTokenSet("/mnt/data/large/subset2/val", "foldtoken", precomputed_embeddings=False)
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_seq_struc_tok_batch)
    tfold = TFold([1000], device=device, bio2token=False).to(device)
    i = 0
    with torch.no_grad():
        for seqs, tokens, structures in loader:
            structure, tokens = structures.to(device), tokens.to(device)
            true_lengths = [len(seq) for seq in seqs]

            vq_codes = []
            batch_ids = []
            chain_encodings = []
            for i, protein_token in enumerate(tokens):
                L = true_lengths[i]
                protein_token_without_padding = protein_token[:L]
                vq_codes.append(protein_token_without_padding)
                batch_ids.append(torch.full((L,), i, dtype=torch.long, device=protein_token.device))
                chain_encodings.append(torch.full((L,), 1, dtype=torch.long, device=protein_token.device))
            # reshape
            vq_codes_cat = torch.cat(vq_codes, dim=0)
            batch_ids_cat = torch.cat(batch_ids, dim=0)
            chain_encodings_cat = torch.cat(chain_encodings, 0)
            print_tensor(vq_codes_cat, "vq_codes_cat")
            # decode proteins
            proteins = tfold.decoder.decode(vq_codes_cat, chain_encodings_cat, batch_ids_cat)
            # return proteins and tokens
            structure_batch, relevant_mask = tfold.reshape_proteins_to_structure_batch(proteins)
            # lddt loss
            B, L, _ = structure_batch.shape
            is_dna = torch.zeros((B, L), dtype=torch.bool, device=device)
            is_rna = torch.zeros((B, L), dtype=torch.bool, device=device)
            lddt_loss = tfold.lddt_loss(structure_batch, structure, is_dna, is_rna, relevant_mask)
            print(lddt_loss.item())
            print_tensor(structure_batch, "structure_batch")
            print_tensor(structure, "structure")
            print_tensor(relevant_mask, "mask")
            print("test")
            print(tfold.lddt_loss(structure, structure, is_dna, is_rna, relevant_mask).item())
            print(tfold.lddt_loss(structure_batch, structure_batch, is_dna, is_rna, relevant_mask).item())
            print("should not be 0")
            print(structure[relevant_mask])
            print((structure[~relevant_mask] != 0).all())
            print("should be 0")
            print(structure[~relevant_mask])
            print((structure[~relevant_mask] == 0).all())
            print("hlep")
            print(tfold.lddt_loss(structure_batch, structure_batch, is_dna, is_rna, ~relevant_mask).item())
            if i >= 0:
                break
            i += 1
    print("done")


def selin_debug():
    device = "cuda"
    dataset = StructureAndTokenSet("/mnt/data/large/subset2/val", "encoding", precomputed_embeddings=False)
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_seq_struc_tok_batch)

    dataset2 = StructureAndTokenSet("/mnt/data/large/subset2/val", "encoding", precomputed_embeddings=True)
    loader2 = DataLoader(dataset2, batch_size=1, collate_fn=collate_emb_struc_tok_batch)

    model = FinalModel([512, 256, 256], kernel_sizes=[17, 3, 3], device=device, dropout=0.0, decoder_lora=True,
                       plm_lora=True)
    model.to(device)
    model.eval()
    with torch.no_grad():
        for sequences, encoding, structure in loader:
            encoding.to(device)
            structure.to(device)
            print("seq lengths:")
            for seq in sequences:
                print(len(seq))
            print_tensor(encoding, "encoding")
            print_tensor(structure, "structure")
            predictions, final_mask, cnn_out = model(sequences)
            print_tensor(predictions, "predictions")
            # encoding_loss = masked_mse_loss(cnn_out, encoding, final_mask)
            # print(encoding_loss.item())
            print("test")
            test = model.plm.encode_list_of_seqs(sequences, 2)
            for s in test:
                print_tensor(s, "s")
            break
        for emb, encoding, structure in loader2:
            print_tensor(emb, "embedding")
            encoding.to(device)
            structure.to(device)
            emb.to(device)
            predictions, final_mask, cnn_out = model.forward_from_embedding(emb)
            print_tensor(cnn_out, "cnn_out")
            print_tensor(predictions, "predictions")
            print_tensor(structure, "structure")
            print_tensor(encoding, "encoding")
            break


def smooth_lddt_sanity_test():
    device = "cuda"
    final_final_model = FinalFinalModel([500], device=device, dropout=0.0, plm_lora=True, decoder_lora=True).to(device)
    print("inited model")
    dataset = StructureSet("/mnt/data/large/subset2/val")
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_seq_struc)
    with torch.no_grad():
        lddt_loss_module = SmoothLDDTLoss().to(device)
        for seqs, structure in dataloader:
            for seq in seqs:
                print(len(seq))
            structure = structure.to(device)
            predictions, final_mask, cnn_out = final_final_model(seqs)
            print_tensor(predictions, "prediction")
            print_tensor(final_mask, "final_mask")
            print_tensor(cnn_out, "cnn_out")
            B, L, _ = predictions.shape
            is_dna = torch.zeros((B, L), dtype=torch.bool, device=device)
            is_rna = torch.zeros((B, L), dtype=torch.bool, device=device)
            lddt_loss = lddt_loss_module(predictions, structure, is_dna, is_rna, final_mask)
            print(lddt_loss.item())
            break


def test_load_old_model():
    device = "cuda"
    model = FinalModel.load_old_final(
        "/mnt/models/final_k21_3_3_h16384_8192_2048_a_1_b_0_plm_lora_lr0.0002/final_k21_3_3_h16384_8192_2048_a_1_b_0_plm_lora_latest.pt",
        device)
    model.to(device)
    print("inited model")
    dataset = StructureSet("/mnt/data/large/subset2/val")
    dataloader = DataLoader(dataset, batch_size=28, collate_fn=collate_seq_struc)
    with torch.no_grad():
        lddt_loss_module = SmoothLDDTLoss().to(device)
        for seqs, structure in dataloader:
            structure = structure.to(device)
            predictions, final_mask, cnn_out = model(seqs)
            print_tensor(predictions, "prediction")
            print_tensor(final_mask, "final_mask")
            print_tensor(cnn_out, "cnn_out")
            B, L, _ = predictions.shape
            is_dna = torch.zeros((B, L), dtype=torch.bool, device=device)
            is_rna = torch.zeros((B, L), dtype=torch.bool, device=device)
            lddt_loss = lddt_loss_module(predictions, structure, is_dna, is_rna, final_mask)
            print(lddt_loss.item())
            break


def test_esm_fold():
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["WANDB_CONSOLE"] = "off"
    device = "cuda"
    esm_fold = EsmFold(device)
    esm_fold.eval()
    dataset = StructureSet("/mnt/data/large/subset2/val")
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_seq_struc)
    with torch.no_grad():
        lddt_loss_module = SmoothLDDTLoss().to(device)
        for seqs, structure in dataloader:
            structure = structure.to(device)
            print([len(seq) for seq in seqs])
            print(torch.cuda.memory_allocated()/(1024 **3))
            backbone_coords, _ = esm_fold(seqs)
            B, L, _ = backbone_coords.shape
            final_mask = torch.zeros(B, L, dtype=torch.bool, device=device)
            for i, seq in enumerate(seqs):
                final_mask[i, :len(seq) * 4] = True
            is_dna = torch.zeros((B, L), dtype=torch.bool, device=device)
            is_rna = torch.zeros((B, L), dtype=torch.bool, device=device)
            lddt_loss = lddt_loss_module(structure, backbone_coords, is_dna, is_rna, final_mask)
            print(lddt_loss.detach().item())
            del lddt_loss


def extract_filename_with_suffix(path, suffix='', keep_extension=False):
    base = os.path.basename(path)
    name, ext = os.path.splitext(base)

    if keep_extension:
        return f"{name}{suffix}{ext}"
    else:
        return f"{name}{suffix}"

def write_pdb():
    device = "cuda"
    out_dir = "test"
    os.makedirs(out_dir, exist_ok=True)
    # prepare model
    model = FinalModel.load_final(
        "/mnt/models/final_k21_3_3_h16384_8192_2048_a_1_b_0_plm_lora_lr5e-05/final_k21_3_3_h16384_8192_2048_a_1_b_0_plm_lora.pt",
        device).to(device)

    # prepare data
    # singletons
    with open("/mnt/data/large/prostt5_IDs/afdb50best_foldseek_clu_singleton.ids") as f:
        singleton_ids = {line.strip().split("-")[1] for line in f if "-" in line}

    # test data prep
    in_dir = "/mnt/data/large/zip_file/final_data_PDB/test/test_pdb"

    pdb_paths, pdb_dicts, seqs = prepare_data(in_dir=in_dir, singleton_ids=singleton_ids, casp=False)

    # run model
    with torch.inference_mode():
        smooth_lddts, normal_lddts = [], []
        lddt_loss_module = SmoothLDDTLoss().to(device)
        for i in range(len(seqs)):
            # get data
            pdb_path = pdb_paths[i]
            seq = [seqs[i]]

            structure = filter_pdb_dict(pdb_dicts[i])["coords_groundtruth"]
            structure_tensor = torch.as_tensor(np.array(structure)).unsqueeze(0).to(device)
            # print_tensor(structure_tensor, "structure_tensor")

            # predict structure
            backbone_coords, _, _ = model(seq)
            # print_tensor(backbone_coords, "pred")

            # calc lddt
            B, L, _ = backbone_coords.shape
            final_mask = torch.zeros(B, L, dtype=torch.bool, device=device)
            final_mask[0, :len(seq) * 4] = True
            is_dna = torch.zeros((B, L), dtype=torch.bool, device=device)
            is_rna = torch.zeros((B, L), dtype=torch.bool, device=device)
            lddt_loss = lddt_loss_module(structure_tensor, backbone_coords, is_dna, is_rna, final_mask)
            orig_value = lddt_loss.detach().cpu().item()
            smoooth_lddt = 1 - orig_value
            # calc other scores
            gt_protein = load_prot_from_pdb(pdb_path)
            pred_protein = gt_protein.copy()
            pred_protein.coord = backbone_coords.squeeze(0).detach().cpu().numpy().astype(np.float32)
            normal_lddt = float(lddt(gt_protein, pred_protein))

            # append score to list
            normal_lddts.append(normal_lddt)
            smooth_lddts.append(smoooth_lddt)

            print(f"smooth: {smoooth_lddt}; normal: {normal_lddt}; loss: {orig_value}")
            # write pdbs
            # pred
            file = PDBFile()
            file.set_structure(pred_protein)
            file_name = extract_filename_with_suffix(pdb_path)
            file.write(os.path.join(out_dir, f"{file_name}_pred.pdb"))
            # gt
            shutil.copy2(pdb_path, out_dir)

            del lddt_loss, structure_tensor, backbone_coords

    plot_smooth_lddt(normal_lddts, smooth_lddts, os.path.join(out_dir, "new_smooth_lddt.png"))

def print_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm(2).item()
            print(f"[GRAD NORM] {name}: {grad_norm:.6f}")
        else:
            print(f"[NO GRAD] {name}")

if __name__ == '__main__':
    # test_new_model()
    device = "cuda"
    model = FinalModel([12_000, 8_192, 2_048], device=device, kernel_sizes=[3, 1, 1], dropout=0.0, decoder_lora=True)
    # input:
    # test_pdbs = ["tokenizer_benchmark/casps/casp14_backbone/T1024-D1.pdb",
    #              "tokenizer_benchmark/casps/casp14_backbone/T1026-D1.pdb"]
    test_pdbs = ["tokenizer_benchmark/casps/casp15_backbone/T1129s2-D1.pdb"]

    # prepare input
    pdb_dicts = [pdb_2_dict(pdb) for pdb in test_pdbs]
    pdb_dicts = [filter_pdb_dict(pdb_dict) for pdb_dict in pdb_dicts]
    seqs = [get_seq_from_pdb(pdb) for pdb in test_pdbs]

    # get bio2token out
    bio2token_batch = batch_pdb_dicts(pdb_dicts, device)
    bio2token_model = load_bio2token_model().to(device)
    with torch.no_grad():
        bio2token_batch = bio2token_model(bio2token_batch)

    # get solulu
    targets = get_padded_ground_truths(test_pdbs).to(device)

    # try to overfit
    # prepare training
    lddt_loss_module = SmoothLDDTLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    model.train()
    for i in range(350):
        predictions, final_mask, cnn_out = model.forward(seqs)

        # scores
        B, L, _ = predictions.shape
        # lddt
        is_dna = torch.zeros((B, L), dtype=torch.bool, device=device)
        is_rna = torch.zeros((B, L), dtype=torch.bool, device=device)
        lddt_loss = lddt_loss_module(predictions, targets, is_dna, is_rna, final_mask)

        optimizer.zero_grad()
        lddt_loss.backward()
        # loss = F.mse_loss(predictions[final_mask], targets[final_mask])

        vector_loss = masked_mse_loss(cnn_out, bio2token_batch["encoding"], final_mask)
        # vector_loss.backward()

        # gradient clipping
        print_gradients(model)
        print("="*30)
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        print_gradients(model)
        # backpropagate
        optimizer.step()
        print(f"lddt loss: {lddt_loss.detach().item()} | encoding loss : {vector_loss.detach().item()}")
        print("="*30)
        break
        #sanity check. run bio2token encoding through our model
        # bio2token_out_through_our_model=model.decoder.decoder.decoder(bio2token_batch["encoding"], bio2token_batch["eos_pad_mask"])

        # calc test lddts
        # bio2token_loss = lddt_loss_module(solution["decoding"], targets, is_dna, is_rna, ~solution["eos_pad_mask"]).item()
        # bio2token_our_model_loss = lddt_loss_module(bio2token_out_through_our_model, targets, is_dna, is_rna, ~solution["eos_pad_mask"]).item()

        #check results
        # diff = (bio2token_out_through_our_model - solution["decoding"]).abs()
        # print("Max diff:", diff.max().item())
        # print("Mean diff:", diff.mean().item())
        #
        # print(bio2token_loss)
        # print(bio2token_our_model_loss)


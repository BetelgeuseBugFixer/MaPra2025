import argparse
import builtins
import gzip
import json
from pathlib import Path
import os

import torch
from biotite.structure.io.pdb import PDBFile
from biotite.structure import lddt
from biotite.structure.filter import _filter_atom_names
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from models.bio2token.data.utils.tokens import PAD_CLASS
from models.bio2token.decoder import bio2token_decoder
from models.bio2token.losses.rmsd import RMSDConfig, RMSD
from models.bio2token.models.autoencoder import AutoencoderConfig, Autoencoder
from models.bio2token.utils.configs import utilsyaml_to_dict, pi_instantiate
from models.model_utils import _masked_accuracy, calc_token_loss, calc_lddt_scores, SmoothLDDTLoss
from models.prot_t5.prot_t5 import ProtT5
from models.datasets.datasets import PAD_LABEL
from models.simple_classifier.simple_classifier import ResidueTokenCNN
from models.bio2token.data.utils.utils import pdb_2_dict, uniform_dataframe, compute_masks, pad_and_stack_batch, \
    filter_batch

from models.foldtoken_decoder.foldtoken import FoldToken
from models.end_to_end.whole_model import TFold
from transformers import T5EncoderModel, T5Tokenizer
from hydra_zen import load_from_yaml, builds, instantiate


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

def batch_pdbs_for_bio2token(pdbs, device):
    batch = []
    # read to dicts
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
        "unknown_structure": True,
        "residue_ids": -1,
        "token_class": PAD_CLASS,
        "eos_pad_mask": 1,
        "structure_known_all_atom_mask": 0,
        "bb_atom_known_structure_mask": 0,
        "sc_atom_known_structure_mask": 0,
        "cref_atom_known_structure_mask": 0,
    }
    # batch = [pdb_2_dict(pdb) for pdb in test_pdbs]
    print(f"2 batch:\n{batch}")
    batch = filter_batch(batch, sequences_to_pad.keys())
    print(f"1 batch:\n{batch}")
    batch = pad_and_stack_batch(
        batch,
        sequences_to_pad,
        1
    )

    print(f"batch:\n{batch}")
    batch = compute_masks(batch, structure_track=True)
    batch = {k: v.to(device) for k, v in batch.items()}
    return batch

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

    batch = batch_pdbs_for_bio2token(test_pdbs,device)

    batch = model.encoder(batch)
    print(
        f"preprocessed batch:\nindices-{batch['indices'].shape}\n{batch['indices']}\nencoding-{batch['encoding'].shape}\n{batch['encoding']}\neos_mask-{batch['eos_pad_mask'].shape}\n{batch['eos_pad_mask']}")
    batch = model.decoder(batch)
    print(f"processed batch:\n{batch['decoding'].shape}\n{batch['decoding']}")


def load_bio2_token_decoder_and_quantizer():
    model_configs = load_from_yaml("models/bio2token/files/model.yaml")["model"]
    model_config = pi_instantiate(AutoencoderConfig, model_configs)
    model = Autoencoder(model_config)
    state_dict = torch.load("models/bio2token/files/epoch=0243-val_loss_epoch=0.71-best-checkpoint.ckpt")["state_dict"]
    # Remove 'model.' prefix from keys if present
    state_dict_bis = {k.replace("model.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict_bis)
    return model.decoder, model.encoder.quantizer,model.encoder

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
    print(f"2 batch:\n{batch}")
    batch = filter_batch(batch, sequences_to_pad.keys())
    print(f"1 batch:\n{batch}")
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
    decoder=bio2token_decoder(device=device).to(device)
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

    lddt_loss_module= SmoothLDDTLoss().to(device)
    # get gt
    targets = get_padded_ground_truths(test_pdbs).to(device)
    target_mask=~eos_mask
    #eval
    #rmsd
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
    lddt_loss=lddt_loss_module(x,targets,is_dna,is_rna,target_mask)
    print(f"loss: {lddt_loss.item()}")


def get_protein_sizes_in_dataset(data_file="/mnt/data/large/subset/train/proteins.jsonl.gz",max_size=1_000):
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


if __name__ == '__main__':
    device = "cuda"
    # define models
    plm = ProtT5(device=device).to(device)
    cnn = ResidueTokenCNN(1024, [2048, 2048], 4096, [5, 5], bio2token=True).to(device)
    decoder = bio2token_decoder(device=device).to(device)
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
    x = decoder(x, eos_mask=eos_mask)


import json
from pathlib import Path

import h5py
import torch
import biotite
import biotite.structure.io.pdb as pdb
from biotite.structure.io.pdb import PDBFile
from biotite.structure import lddt, tm_score, filter_canonical_amino_acids
from biotite.structure.filter import _filter_atom_names

from models.prot_t5.prot_t5 import ProtT5
from models.simple_classifier.simple_classifier import ResidueTokenCNN

from models.foldtoken_decoder.foldtoken_decoder import FoldDecoder


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


# def sanity_test():
#     device = "cuda"
#     model = FoldDecoder(device=device)
#     test_pdb = "tokenizer_benchmark/casps/casp14/T1024-D1.pdb"
#
#     # get ref
#     ref_protein = load_prot_from_pdb(test_pdb)
#
#     # encode and decode model
#     locally_encoded_vq_codes = model.encode_pdb(test_pdb)
#     locally_encoded_coords = decode_atom_coordinates(locally_encoded_vq_codes).detach().squeeze(0).reshape(-1,
#                                                                                                            3).cpu().numpy()
#
#     # also load prev computed tokens
# pre_encoded_vq_codes = torch.tensor(load_casp_tokens("data/casp14_test/casp14_tokens.jsonl")["T1024-D1"]["vqid"],
#                                     dtype=torch.long, device=device)
# pre_encoded_coords = decode_atom_coordinates(pre_encoded_vq_codes).detach().squeeze(0).reshape(-1, 3).cpu().numpy()
#
#     # get scores
#     print(f"locally encoded: {lddt(ref_protein, locally_encoded_coords)}")
#     print(f"pre encoded: {lddt(ref_protein, pre_encoded_coords)}")

three_to_one_dict = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D',
    'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G',
    'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
    'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S',
    'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}


def get_seq_from_pdb(pdb_file: str) -> str:
    seq = ""
    with open(pdb_file, 'r') as f:
        for line in f.readlines():
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                res_name = line[17:20].strip()
                seq += three_to_one_dict[res_name]

    return seq


if __name__ == '__main__':
    device = "cuda"
    # init models
    model = FoldDecoder(device=device).to(device)
    plm = ProtT5().to(device)
    cnn = ResidueTokenCNN.load_cnn("train_run/cnn_k21_3_3_h16384_8192_2048.pt").to(device)
    decoder = FoldDecoder(device=device)

    # define input
    test_pdbs=["tokenizer_benchmark/casps/casp14/T1024-D1.pdb","tokenizer_benchmark/casps/casp14/T1024-D2.pdb"]
    seqs = [get_seq_from_pdb(pdb) for pdb in test_pdbs]
    true_lengths = [len(seq) for seq in seqs]
    print(f"true_lengths: {true_lengths}")
    print(seqs)
    seqs = [raw_seq.translate(str.maketrans('UZO', 'XXX')) for raw_seq in seqs]
    seqs = [" ".join(raw_seq) for raw_seq in seqs]
    print(seqs)
    # run through model
    plm.eval()
    with torch.no_grad():
        emb = plm(seqs)
    print(f"emb: {len(emb[0])}")
    print("=" * 20)
    cnn.eval()
    with torch.no_grad():
        logits = cnn(emb)  # shape: (B, L, vocab_size)
        pred_tokens = logits.argmax(dim=-1)

    pred_tokens = pred_tokens.to(device)
    pre_encoded_vq_codes = torch.tensor(load_casp_tokens("data/casp14_test/casp14_tokens.jsonl")["T1024-D1"]["vqid"],
                                        dtype=torch.long, device=device)

    # batch for foldtoken
    vq_codes = []
    batch_ids = []
    chain_encodings = []
    for i, protein_token in enumerate(pred_tokens):
        L = true_lengths[i]
        protein_token_without_padding = protein_token[:L]
        vq_codes.append(protein_token_without_padding)
        batch_ids.append(torch.full((L,), i, dtype=torch.long, device=protein_token.device))

        chain_encodings.append(torch.ones((L,), dtype=torch.long, device=protein_token.device))


    vq_codes_cat = torch.cat(vq_codes, dim=0)
    batch_ids_cat = torch.cat(batch_ids, dim=0)
    chain_encodings_cat = torch.cat(chain_encodings, 0)

    proteins=decoder.decode(vq_codes_cat,chain_encodings_cat,batch_ids_cat)
    print(proteins)
    print("*"*20)
    for protein,pdb in zip(proteins,test_pdbs):
        X, _, _ = protein.to_XCS(all_atom=False).detach().squeeze(0).reshape(-1, 3).cpu().numpy()
        ref_protein = load_prot_from_pdb(pdb)
        print(lddt(ref_protein, X))



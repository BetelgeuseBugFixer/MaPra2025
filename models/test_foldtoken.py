import json
from pathlib import Path
import os

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
from models.whole_model import TFold


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
    model = FoldDecoder(device=device)
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
    chain_id=None
    with open(pdb_file, 'r') as f:
        for line in f.readlines():
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                current_chain=line[21]
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
    plm = ProtT5().to(device)
    cnn = ResidueTokenCNN.load_cnn("train_run/cnn_k19_1_1_h10000_8000_4000.pt").to(device)
    decoder = FoldDecoder(device=device)

    # define input
    # test_pdbs=["tokenizer_benchmark/casps/casp14/T1024-D1.pdb","tokenizer_benchmark/casps/casp14/T1024-D2.pdb"]
    ly6_pdb_dir = "/mnt/data/lys6/pdb_files/pdb"
    test_pdbs = [os.path.join(ly6_pdb_dir, pdb) for pdb in os.listdir(ly6_pdb_dir)]
    lddt_scores = []
    # create batches 
    batch_size = 16
    for i in range(0, len(test_pdbs), batch_size):
        pdb_batch = test_pdbs[i:i + batch_size]
        seqs = [get_seq_from_pdb(pdb) for pdb in pdb_batch]
        true_lengths = [len(seq) for seq in seqs]
        # print(f"true_lengths: {true_lengths}")
        # print(seqs)
        seqs = [raw_seq.translate(str.maketrans('UZO', 'XXX')) for raw_seq in seqs]
        seqs = [" ".join(raw_seq) for raw_seq in seqs]
        # print(seqs)
        # run through model
        plm.eval()
        with torch.no_grad():
            emb = plm(seqs)
        # print(f"emb: {len(emb[0])}")
        # print("=" * 20)
        cnn.eval()
        with torch.no_grad():
            logits = cnn(emb)  # shape: (B, L, vocab_size)
            pred_tokens = logits.argmax(dim=-1)

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
        # print(f"vq_codes_cat:\n{vq_codes_cat}\nbatch_ids_cat:\n{batch_ids_cat}\nchain_encodings_cat:\n{chain_encodings_cat}")

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
            print(lddt_score)
    print(f"Average lddt score: {sum(lddt_scores) / len(lddt_scores)}")


if __name__ == '__main__':
    test_pdbs = ["tokenizer_benchmark/casps/casp14/T1024-D1.pdb", "tokenizer_benchmark/casps/casp14/T1026-D1.pdb"]
    seqs = [get_seq_from_pdb(pdb_path) for pdb_path in test_pdbs]
    t_fold = TFold([1024],device="cuda").to("cuda").eval()
    print(hash(str(t_fold.decoder.state_dict())))
    proteins, tokens = t_fold(seqs)
    print(tokens)
    for protein,pdb in zip(proteins,test_pdbs):
        X, _, _ = protein.to_XCS(all_atom=False)
        X = X.detach().squeeze(0).reshape(-1, 3).cpu().numpy()
        try:
            ref_protein = load_prot_from_pdb(pdb)
        except Exception as e:
            print(f"Error loading PDB {pdb}: {e}")
            continue
        print(lddt(ref_protein, X))
    print("*"*11)
    t_fold.save(".")
    t_fold = TFold.load_tfold(f"{t_fold.model_name}.pt",device="cuda").to("cuda").eval()
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
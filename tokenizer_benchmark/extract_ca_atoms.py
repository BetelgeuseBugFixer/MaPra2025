import os

from biotite.structure import lddt
from biotite.structure.filter import _filter_atom_names
from biotite.structure.io.pdb import PDBFile

from models.bio2token.data.utils.utils import pdb_2_dict


def rewrite_pdb(pdb_in_path, pdb_out_path, allowed_atom_list=["CA"]):
    with open(pdb_in_path, 'r') as in_file:
        with open(pdb_out_path, 'w') as out_file:
            for line in in_file:
                if line.startswith('ATOM'):
                    if line[12:16].strip() in allowed_atom_list:
                        out_file.write(line)
                else:
                    out_file.write(line)


def main(input_dir="tokenizer_benchmark/casps/casp15", output_dir="tokenizer_benchmark/casps/casp15_backbone"):
    for filename in os.listdir(input_dir):
        pdb_in_path = os.path.join(input_dir, filename)
        pdb_out_path = os.path.join(output_dir, filename)
        rewrite_pdb(pdb_in_path, pdb_out_path, ["N", "CA", "C", "O"])


def load_prot_from_pdb(pdb_file):
    file = PDBFile.read(pdb_file)
    array_stack = file.get_structure(model=1)
    return array_stack[_filter_atom_names(array_stack, ["N", "CA", "C", "O"])]


def get_normalize_casp15():
    input_dir = "tokenizer_benchmark/casps/casp15_backbone"
    bio2token_dir = "tokenizer_benchmark/raw_output_files/bio2token_out/casp15"
    bio2token_path_to_gt_suffix = "/bio2token_pretrained/epoch=0243-val_loss_epoch=0.71-best-checkpoin/all/gt.pdb"
    output_dir = "tokenizer_benchmark/casps/casp15_backbone_normalized"
    for filename in os.listdir(input_dir):
        pdb_path = os.path.join(input_dir, filename)
        array_stack = load_prot_from_pdb(pdb_path)
        pdb_seq = pdb_2_dict(pdb_path)["seq"]
        if len(array_stack) != len(pdb_seq) * 4:
            pdb_id = filename.split(".")[0]
            fixed_pbd_path = os.path.join(bio2token_dir, pdb_id+"/"+bio2token_path_to_gt_suffix)
            new_array_stack = load_prot_from_pdb(fixed_pbd_path)
            break

def get_pid_from_file_name(file_name):
    return file_name.split("-")[1]


def prepare_data(in_dir,singleton_ids=None):
    pdb_names = [p for p in os.listdir(in_dir) if p.endswith("pdb")]
    if singleton_ids:
        pdb_names = [p for p in pdb_names if get_pid_from_file_name(p) not in singleton_ids]
    pdb_paths = [os.path.join(in_dir, p) for p in pdb_names]
    pdb_dicts = [pdb_2_dict(p) for p in pdb_paths]

    allowed = [i for i, d in enumerate(pdb_dicts) if len(d["seq"]) < 800 and len(d["seq"]) * 4 == d["atom_length"]]
    pdb_paths = [pdb_paths[i] for i in allowed]
    pdb_dicts = [pdb_dicts[i] for i in allowed]

    seqs = [d["seq"] for d in pdb_dicts]
    return pdb_paths, pdb_dicts, seqs

if __name__ == '__main__':

    casp_dir = "tokenizer_benchmark/casps/casp15_backbone"
    pdb_casp, casp_dicts, seqs_casp = prepare_data(casp_dir)

    pdb_casp, casp_dicts, seqs_casp=prepare_data(casp_dir)

    print(len(casp_dicts))
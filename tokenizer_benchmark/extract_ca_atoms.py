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


if __name__ == '__main__':
    get_normalize_casp15()

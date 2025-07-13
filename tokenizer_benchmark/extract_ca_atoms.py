import os


def rewrite_pdb(pdb_in_path, pdb_out_path, allowed_atom_list=["CA"]):
    with open(pdb_in_path, 'r') as in_file:
        with open(pdb_out_path, 'w') as out_file:
            for line in in_file:
                if line.startswith('ATOM'):
                    if line[12:16].strip() in allowed_atom_list:
                        out_file.write(line)
                else:
                    out_file.write(line)


def main(input_dir="tokenizer_benchmark/casps/casp14", output_dir="tokenizer_benchmark/casps/casp14_backbone"):
    for filename in os.listdir(input_dir):
        pdb_in_path = os.path.join(input_dir, filename)
        pdb_out_path = os.path.join(output_dir, filename)
        rewrite_pdb(pdb_in_path, pdb_out_path,["N", "CA", "C", "O"])


if __name__ == '__main__':
    main()

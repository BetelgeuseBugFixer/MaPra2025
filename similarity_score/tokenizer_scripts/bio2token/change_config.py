import sys
import os
import yaml

CONFIG_FILE = "/dss/dsshome1/08/ge43vab2/mapra/bio2token/configs/test_pdb.yaml"

def extract_pdb_id(input_file):
    """
    Extracts the PDB ID part (e.g., T1104-D1) from the input file path.
    """
    base = os.path.basename(input_file)
    pdb_id = os.path.splitext(base)[0]
    return pdb_id

def main():
    input_file = sys.argv[1]
    output_folder = sys.argv[2]
    pdb_id = extract_pdb_id(input_file)
    new_configs=""
    with open(CONFIG_FILE, 'r') as f:
        for line in f.readlines():
            if line.startswith('  results_dir:'):
                new_configs += f'  results_dir: {os.path.join(output_folder,pdb_id)}\n'
            elif line.startswith('  pdb_path:'):
                new_configs += f'  pdb_path: {input_file}\n'
            else:
                new_configs += line

    with open(CONFIG_FILE, 'w') as f:
        f.write(new_configs)


if __name__ == "__main__":
    main()

import os

from models.bio2token.data.utils.utils import pdb_2_dict


def get_pid_from_file_name(file_name):
    return file_name.split("-")[1]

if __name__ == '__main__':
    # with open("data/afdb50best_foldseek_clu_singleton.ids") as f:
    #     singleton_ids = {line.strip().split("-")[1] for line in f if "-" in line}

    pdb_dir="data/test_pdb"

    pdb_names = [p for p in os.listdir(pdb_dir) if p.endswith("pdb")]

    # pdb_names = [p for p in pdb_names if get_pid_from_file_name(p) not in singleton_ids]
    pdb_paths = [os.path.join(pdb_dir, p) for p in pdb_names]
    pdb_dicts = [pdb_2_dict(p) for p in pdb_paths]
    allowed = [i for i, d in enumerate(pdb_dicts) if len(d["seq"]) < 800]
    #finally extract all seqs
    pdb_seqs = [pdb_dicts[i]["seq"] for i in allowed]
    pdb_names= [pdb_names[i] for i in allowed]
    with open("data/casp15.fasta","w") as f:
        for seq,id in zip(pdb_seqs, pdb_names):
            f.write(f">{id}\n{seq}\n")
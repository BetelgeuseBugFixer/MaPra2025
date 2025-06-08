import json
import os.path


def main(bio2token_out_dir="tokenizer_benchmark/raw_output_files/bio2token_out/casp14",
         token_out_file="data/casp14_test/casp14_biotokens.tsv"):
    with open(token_out_file, "w") as token_tsv:
        for pbd_id in os.listdir(bio2token_out_dir):
            token_jsonl = os.path.join(bio2token_out_dir, pbd_id,
                                       "bio2token_pretrained/epoch=0243-val_loss_epoch=0.71-best-checkpoin/all/outputs.json")
            token = json.loads(open(token_jsonl, "r").read())["indices"]
            # filter out ca atom indices
            pdb_path = os.path.join(bio2token_out_dir, pbd_id,
                                    "bio2token_pretrained/epoch=0243-val_loss_epoch=0.71-best-checkpoin/all/gt.pdb")

            ca_indices = extract_ca_indices_from_pdb(pdb_path)
            # since both list are sorted this could be sped up
            token = [x for i, x in enumerate(token) if i in ca_indices]
            token_tsv.write(f"{pbd_id}\t{",".join(map(str, token))}\n")


def extract_ca_indices_from_pdb(pdb_path):
    ca_indices = []
    with open(pdb_path, "r") as pdb_file:
        for line in pdb_file:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                ca_indices.append(int(line[6:11]))
    return ca_indices


if __name__ == '__main__':
    main()

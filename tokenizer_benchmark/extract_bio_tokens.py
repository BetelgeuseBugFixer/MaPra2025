import json
import os.path


def main(bio2token_out_dir="tokenizer_benchmark/raw_output_files/bio2token_out/casp14", token_out_file="data/casp14_test/casp14_biotokens.tsv"):
    with open(token_out_file,"w") as token_tsv:
        for pbd_id in os.listdir(bio2token_out_dir):
            token_jsonl = os.path.join(bio2token_out_dir, pbd_id,
                                 "bio2token_pretrained/epoch=0243-val_loss_epoch=0.71-best-checkpoin/all/outputs.json")
            token=json.loads(open(token_jsonl,"r").read())["indices"]
            token_tsv.write(f"{pbd_id}\t{token}\n")

if __name__ == '__main__':
    main()

import argparse
import os

from model_predictions import prepare_data, infer_structures, compute_and_save_scores_for_model
from models.end_to_end.whole_model import FinalFinalModel
from biotite.structure.io.pdb import PDBFile





if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("--model_ckpt", nargs="+", default=[], help="Path(s) to model checkpoint(s)")
    p.add_argument("--pdb_dir", nargs="+", default=[], help="Path to pdb directory")
    args = p.parse_args()
    base_dir = "casp_test"

    pdb_casp, casp_dicts, seqs_casp = prepare_data(args.pdb_dir, casp=True)

    print(f"Processing model checkpoint: {args.model_ckpt}")
    model = FinalFinalModel.load_final_final(args.model_ckpt, device="cuda")

    final_structs = compute_and_save_scores_for_model(args.model_ckpt, model, seqs_casp, pdb_casp, casp_dicts,
                                                      batch_size=32, dataset_name="casp", given_base=base_dir)

    for structure in final_structs:
        file = PDBFile()
        file.set_structure(structure)
        file.write(os.path.join(base_dir,f"{structure}.pdb"))

#!/usr/bin/env python3
import os
import argparse
from bio2token.models.autoencoder import Autoencoder, AutoencoderConfig
from bio2token.utils.configs import utilsyaml_to_dict, pi_instantiate
from bio2token.utils.lightning import find_lowest_val_loss_checkpoint
from bio2token.data.utils.utils import compute_masks, pdb_2_dict, uniform_dataframe, write_pdb
from bio2token.data.utils.molecule_conventions import ABBRS
import torch
import json

DEBUG = False
if DEBUG:
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def main():
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="test_pdb.yaml")
    args = parser.parse_args()

    # STEP 1: Load config yaml file
    global_configs = utilsyaml_to_dict(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   

    # STEP 2: Instantiate model.
    model_config = pi_instantiate(AutoencoderConfig, yaml_dict=global_configs["model"])
    model = Autoencoder(model_config)

    # Load checkpoint
    ckpt_path = f"{global_configs['infer']['checkpoint_dir']}/{global_configs['infer']['experiment_name']}/{global_configs['infer']['run_id']}/last.ckpt"
    if global_configs["infer"].get("checkpoint_type") == "best":
        ckpt_path = find_lowest_val_loss_checkpoint(
            checkpoint_dir=f"{global_configs['infer']['checkpoint_dir']}/{global_configs['infer']['experiment_name']}/{global_configs['infer']['run_id']}",
            checkpoint_monitor=global_configs["infer"]["checkpoint_monitor"],
            checkpoint_mode=global_configs["infer"]["checkpoint_mode"],
        )
    ckpt_path_name = ckpt_path.split("/")[-1].strip(".ckpt")
    state_dict = torch.load(ckpt_path)["state_dict"]
    # Remove 'model.' prefix from keys if present
    state_dict_bis = {k.replace("model.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict_bis)
    model.eval()
    model.to(device)

    # prepare batch
    pdb_path = global_configs["data"]["pdb_path"]
    dict = pdb_2_dict(
        pdb_path,
        chains=global_configs["data"]["chains"] if "chains" in global_configs["data"] else None,
    )
    structure, unknown_structure, residue_name, residue_ids, token_class, atom_names_reordered = uniform_dataframe(
        dict["seq"],
        dict["res_types"],
        dict["coords_groundtruth"],
        dict["atom_names"],
        dict["res_atom_start"],
        dict["res_atom_end"],
    )
    batch = {
        "structure": torch.tensor(structure).float(),
        "unknown_structure": torch.tensor(unknown_structure).bool(),
        "residue_ids": torch.tensor(residue_ids).long(),
        "token_class": torch.tensor(token_class).long(),
    }
    batch = {k: v[~batch["unknown_structure"]] for k, v in batch.items()}
    batch = compute_masks(batch, structure_track=True)
    batch = {k: v[None].to(device) for k, v in batch.items()}
    
    
    # define tokens
    tokens=[3205,3213,2189,2061,1032,1033,9,270,523,527,846,911,799,779,1551,1807,1803,1543,2575,2831,1799,3591,3919,2843,2823,3911,3931,3863,3843,3911,3987,3922,3969,3778,3651,3659,2703,3471,3403,2315,1034,1103,591,654,78,267,266,261,1025,1028,2052,3077,3208,3268,3276,3289,3309,2285,2284,3228,2125,1164,2140,2060,1101,1100,2076,1033,77,1052,1032,13,268,24,4,524,300,20,536,261,270,652,732,989,717,333,526,523,779,2827,2827,2823,2823,3586,3650,3393,3713,3968,3921,3859,3939,3971,3715,3463,3211,3339,1291,527,523,525,520,524,797,804,769,516,4,8,13,1030,1028,12,1038,2057,1032,1100,3085,2056,2076,2124,3144,3112,2156,3244,3292,3213,3205,3077,3138,2305,1280,0,9,6,522,522,527,1295,2831,3855,3659,3719,3207,3471,3787,3807,3803,3543,3799,3975,2831,3952,3921,3712,3393,3459,3463,3399,3591,3843,2819,2819,1815,1803,519,1811,1815,1803,515,1537,518,778,527,781,781,588,588,268,348,140,1100,2124,3164,3213,3277,3206,3275,3279,2511,463,735,911,2959,2639,3087,2190,2190,2252,2204,1100,28,524,796,877]
    tensor = torch.tensor(tokens).to(device)
    # tranfser indices to model input format
    tensor= model.encoder.quantizer.indices_to_codes(tensor)
    batch["encoding"]= tensor
    batch = model.decoder(batch)
    
    #finalize batch
    batch = model.registration(batch)

    # Calculate and apply losses to the batch
    batch = model.loss(batch)
    #expand 
    for k, v in batch["losses"].items():
        batch[k] = v.item()


    rec_coords = batch["all_atom_coords"][0].detach().cpu().numpy()
    atom_types = atom_names_reordered
    residue_types = [ABBRS[res.split("_")[0]][res.split("_")[1]] for res in residue_name]
    if "chains" in global_configs["data"] and global_configs["data"]["chains"] is not None:
        chains = "_".join(global_configs["data"]["chains"])
    else:
        chains = "all"

    write_pdb(
        rec_coords,
        atom_types,
        residue_types,
        residue_ids,
        f"/dss/dsshome1/08/ge43vab2/mapra/data/bio2token_out/casp14_rec/rec.pdb",
    )


if __name__ == "__main__":
    main()

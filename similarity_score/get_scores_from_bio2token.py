import os
import sys
import subprocess
import re

FOLDSEEK_TEMP_FILE="foldseek_temp.txt"

def main():
    bio2token_output_dir=sys.argv[1]
    # Path for the output file
    foldseek_output_file = os.path.join("similarity_score", "foldseek_bio2token_out.tsv")
    usalign_output_file = os.path.join("similarity_score", "usalign_bio2token_out.tsv")
    # Write header to the file
    with open(foldseek_output_file, "w") as foldseek_output_file_writer:
        foldseek_output_file_writer.write("id\ttmscore\tlddt\n")
        with open(usalign_output_file, "w") as usalign_output_file_writer:
            usalign_output_file_writer.write("id\ttmscore\trmsd\n")
            for folder_name in os.listdir(bio2token_output_dir):
                # get files
                folder_path = os.path.join(bio2token_output_dir, folder_name)
                ref_file_path = os.path.join(folder_path, "bio2token_pretrained/epoch=0243-val_loss_epoch=0.71-best-checkpoin/all/gt.pdb")
                pred_file_path = os.path.join(folder_path, "bio2token_pretrained/epoch=0243-val_loss_epoch=0.71-best-checkpoin/all/rec.pdb")
                # run foldseek and usalign scripts
                f_tm_score,lddt_score=run_foldseek_script_script("similarity_score/foldseek.sh", ref_file_path, pred_file_path)
                foldseek_output_file_writer.write(f"{folder_name}\t{f_tm_score}\t{lddt_score}\n")
                rmsd, tm_score = run_usalign_script("similarity_score/us_align.sh",ref_file_path, pred_file_path)
                usalign_output_file_writer.write(f"{folder_name}\t{tm_score}\t{rmsd}\n")


def run_foldseek_script_script(script_path,ref_file_path, pred_file_path):
    try:
        result = subprocess.run(
            ["bash", script_path, ref_file_path, pred_file_path,FOLDSEEK_TEMP_FILE],
            capture_output=True,
            text=True,
            check=True
        )
        with open(FOLDSEEK_TEMP_FILE, "r") as f:
            output = f.read()
            
        values = output.split("\t")
        #ttmscore,lddt
        if len(values) == 2:
            tm_score=values[0].strip()
            lddt=values[1].strip()
            return tm_score, lddt
        else:
            return None, None
        
    except subprocess.CalledProcessError as e:
        print("Error running script:", e)
        print("STDERR:", e.stderr)
        return None

def run_usalign_script(script_path,ref_file_path, pred_file_path):
    try:
        result = subprocess.run(
            ["bash", script_path,ref_file_path, pred_file_path],
            capture_output=True,
            text=True,
            check=True
        )
        output = result.stdout

        # Extract RMSD
        rmsd_match = re.search(r'RMSD=\s*([\d.]+)', output)
        rmsd = float(rmsd_match.group(1)) if rmsd_match else None

        # Extract first TM-score (normalized by Structure_1)
        tm_score_match = re.search(r'TM-score=\s*([\d.]+)\s+\(normalized by length of Structure_1', output)
        tm_score = float(tm_score_match.group(1)) if tm_score_match else None

        return rmsd, tm_score
        
    except subprocess.CalledProcessError as e:
        print("Error running script:", e)
        print("STDERR:", e.stderr)
        return None
    
    
if __name__ == "__main__":
    main()
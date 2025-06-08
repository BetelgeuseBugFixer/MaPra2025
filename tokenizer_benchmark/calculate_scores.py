import os
import subprocess
import re
import sys

FOLDSEEK_TEMP_FILE = "foldseek_temp.txt"
INPUT_FILES = "tokenizer_benchmark/casps"


def get_ref_file(pdb_id, casp):
    return os.path.join(INPUT_FILES, f"casp{casp}", f"{pdb_id}.pdb")


def analyse_bio2token_output(bio2token_output_dir, casp):
    foldseek_output_file = os.path.join("tokenizer_benchmark/scores", f"casp{casp}_foldseek_bio2token_out.tsv")
    usalign_output_file = os.path.join("tokenizer_benchmark/scores", f"casp{casp}_usalign_bio2token_out.tsv")
    # Write header to the file
    with open(foldseek_output_file, "w") as foldseek_output_file_writer:
        foldseek_output_file_writer.write("id\tf_tmscore\tlddt\n")
        with open(usalign_output_file, "w") as usalign_output_file_writer:
            usalign_output_file_writer.write("id\tus_tmscore\trmsd\n")
            for pdb_id in os.listdir(bio2token_output_dir):
                # get files
                folder_path = os.path.join(bio2token_output_dir, pdb_id)
                ref_file_path = get_ref_file(pdb_id, casp)
                pred_file_path = os.path.join(folder_path,
                                              "bio2token_pretrained/epoch=0243-val_loss_epoch=0.71-best-checkpoin/all/rec.pdb")
                # run foldseek and usalign scripts
                f_tm_score, lddt_score = run_foldseek_script_script("tokenizer_benchmark/foldseek.sh", ref_file_path,
                                                                    pred_file_path)
                foldseek_output_file_writer.write(f"{pdb_id}\t{f_tm_score}\t{lddt_score}\n")
                rmsd, tm_score = run_usalign_script("tokenizer_benchmark/us_align.sh", ref_file_path, pred_file_path)
                usalign_output_file_writer.write(f"{pdb_id}\t{tm_score}\t{rmsd}\n")

def extract_residue_names(pdb_lines):
    residues = []
    seen = set()
    for line in pdb_lines:
        if not line.startswith("ATOM"):
            continue
        res_uid = (line[21], line[22:26])  # Chain ID + Residue number
        if res_uid not in seen:
            residues.append(line[17:20])  # Residue name
            seen.add(res_uid)
    return residues


def fix_residue_names_from_reference(input_path_wrong, input_path_correct, output_path):

    # PDB-Dateien einlesen
    with open(input_path_wrong) as f_wrong:
        lines_wrong = f_wrong.readlines()
    with open(input_path_correct) as f_correct:
        lines_correct = f_correct.readlines()

    # Extrahiere Referenz-Residue-Namen
    ref_residues = extract_residue_names(lines_correct)

    # Korrigieren
    new_lines = []
    last_resi = None
    ref_idx = -1

    for line in lines_wrong:
        if not line.startswith("ATOM"):
            new_lines.append(line)
            continue

        resi_num = int(line[22:26].strip())
        if resi_num != last_resi:
            ref_idx += 1
            last_resi = resi_num

        if ref_idx < len(ref_residues):
            correct_resname = ref_residues[ref_idx].strip()
            line = line[:17] + f"{correct_resname:>3}" + line[20:]

        new_lines.append(line)

    # Schreiben
    with open(output_path, "w") as f_out:
        f_out.writelines(new_lines)

def analyse_foldtoken_output(foldtoken_output_dir, casp, level):
    foldseek_output_file = os.path.join("tokenizer_benchmark/scores", f"casp{casp}_foldseek_foldtoken{level}_out.tsv")
    usalign_output_file = os.path.join("tokenizer_benchmark/scores", f"casp{casp}_usalign_foldtoken{level}_out.tsv")
    with open(foldseek_output_file, "w") as foldseek_output_file_writer:
        foldseek_output_file_writer.write("id\tf_tmscore\tlddt\n")
        with open(usalign_output_file, "w") as usalign_output_file_writer:
            usalign_output_file_writer.write("id\tus_tmscore\trmsd\n")
            for pdb_file_name in os.listdir(foldtoken_output_dir):
                if not pdb_file_name.endswith("_pred.pdb"):
                    continue
                pdb_id = pdb_file_name.strip("_pred.pdb")
                ref_file_path = get_ref_file(pdb_id, casp)
                pred_file_path = os.path.join(foldtoken_output_dir, pdb_file_name)
                fix_residue_names_from_reference(pred_file_path, ref_file_path,"tmp.pdb")
                # run foldseek and usalign scripts
                f_tm_score, lddt_score = run_foldseek_script_script("tokenizer_benchmark/foldseek.sh", ref_file_path,
                                                                    "tmp.pdb")
                foldseek_output_file_writer.write(f"{pdb_id}\t{f_tm_score}\t{lddt_score}\n")
                rmsd, tm_score = run_usalign_script("tokenizer_benchmark/us_align.sh", ref_file_path, pred_file_path)
                usalign_output_file_writer.write(f"{pdb_id}\t{tm_score}\t{rmsd}\n")


def analyse_foldtoken(foldtoken_out):
    for level in range(6, 13, 2):
        print(f"Analyzing foldtoken output for CASP14 at level {level}")
        analyse_foldtoken_output(os.path.join(foldtoken_out, f"casp14_out_level{level}"), 14, level)
        print(f"Analyzing foldtoken output for CASP15 at level {level}")
        analyse_foldtoken_output(os.path.join(foldtoken_out, f"casp15_out_level{level}"), 15, level)

def analyse_bio2token(bio2token_out):
    print("Analyzing bio2token output for CASP14")
    analyse_bio2token_output(os.path.join(bio2token_out, "casp14"), 14)
    print("Analyzing bio2token output for CASP15")
    analyse_bio2token_output(os.path.join(bio2token_out, "casp15"), 15)


def main():
    print("Analyzing bio2token output...")
    analyse_bio2token("tokenizer_benchmark/raw_output_files/bio2token_out")
    print("Analyzing foldtoken output...")
    analyse_foldtoken("tokenizer_benchmark/raw_output_files/foldtoken_out")


def run_foldseek_script_script(script_path, ref_file_path, pred_file_path):
    try:
        subprocess.run(
            ["bash", script_path, ref_file_path, pred_file_path, FOLDSEEK_TEMP_FILE],
            capture_output=True,
            text=True,
            check=True
        )
        with open(FOLDSEEK_TEMP_FILE, "r") as f:
            output = f.read()

        values = output.split("\t")
        # ttmscore,lddt
        if len(values) == 2:
            tm_score = values[0].strip()
            lddt = values[1].strip()
            return tm_score, lddt
        else:
            return None, None

    except subprocess.CalledProcessError as e:
        print("Error running script:", e)
        print("STDERR:", e.stderr)
        return None


def run_usalign_script(script_path, ref_file_path, pred_file_path):
    try:
        result = subprocess.run(
            ["bash", script_path, ref_file_path, pred_file_path],
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
    #analyse_bio2token_output("tokenizer_benchmark/raw_output_files/casp14_edited", "14_edited")


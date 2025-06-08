import pandas as pd
import matplotlib.pyplot as plt


def join_dfs(df1, df2, on_col="id"):
    return pd.merge(df1, df2, on=on_col, how="outer")


def get_single_casp_set(casp_out_file1, casp_out_file2):
    casp_out1 = pd.read_csv(
        casp_out_file1, sep="\t"
    )
    casp_out2 = pd.read_csv(
        casp_out_file2, sep="\t"
    )
    return join_dfs(casp_out1, casp_out2)


def get_foldtoken_data():
    casp_foldtoken_data = pd.DataFrame()
    for level in range(6, 13, 2):
        casp14_foldtoken_data = get_single_casp_set(
            f"tokenizer_benchmark/scores/casp14_foldseek_foldtoken{level}_out.tsv",
            f"tokenizer_benchmark/scores/casp14_usalign_foldtoken{level}_out.tsv")
        casp15_foldtoken_data = get_single_casp_set(
            f"tokenizer_benchmark/scores/casp15_foldseek_foldtoken{level}_out.tsv",
            f"tokenizer_benchmark/scores/casp15_usalign_foldtoken{level}_out.tsv")
        casp_foldtoken_data_level = pd.concat([casp14_foldtoken_data, casp15_foldtoken_data], ignore_index=True)
        casp_foldtoken_data_level["tool"] = f"foldtoken{level}"
        casp_foldtoken_data = pd.concat([casp_foldtoken_data, casp_foldtoken_data_level])
    return casp_foldtoken_data


def get_bio2token_data():
    casp14_bio2token_data = get_single_casp_set("tokenizer_benchmark/scores/casp14_foldseek_bio2token_out.tsv",
                                                "tokenizer_benchmark/scores/casp14_usalign_bio2token_out.tsv")
    casp15_bio2token_data = get_single_casp_set("tokenizer_benchmark/scores/casp15_foldseek_bio2token_out.tsv",
                                                "tokenizer_benchmark/scores/casp15_usalign_bio2token_out.tsv")
    casp_bio2token_data = pd.concat(
        [casp14_bio2token_data, casp15_bio2token_data], ignore_index=True
    )
    casp_bio2token_data["tool"] = "bio2token"
    return casp_bio2token_data


def boxplot(df, score, output_file):
    # mpl.style.use("seaborn-v0_8")
    plt.figure(figsize=(8, 6))
    df.boxplot(column=score, by="tool")
    plt.title(f"{score} by Tool")
    plt.suptitle("")
    plt.xlabel("Tool")
    plt.ylabel(score)
    plt.savefig(output_file)


def main():
    bio2token_df = get_bio2token_data()
    foldtoken_df = get_foldtoken_data()
    tokenizer_df = pd.concat([bio2token_df, foldtoken_df], ignore_index=True)
    boxplot(tokenizer_df, "us_tmscore", "tokenizer_benchmark/plots/tmscore_boxplot_bio2token.png")
    boxplot(tokenizer_df, "f_tmscore", "tokenizer_benchmark/plots/tmscore_fs_boxplot_bio2token.png")
    boxplot(tokenizer_df, "rmsd", "tokenizer_benchmark/plots/rmsd_boxplot_bio2token.png")
    boxplot(tokenizer_df, "lddt", "tokenizer_benchmark/plots/lddt_boxplot_bio2token.png")


if __name__ == "__main__":
    main()

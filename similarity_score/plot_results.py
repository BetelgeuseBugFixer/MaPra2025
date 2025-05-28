import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


def join_dfs(df1, df2, on_col="id"):
    return pd.merge(df1, df2, on=on_col, how="outer")


def get_bio2token_data():
    casp14_foldseek_bio2token_out = pd.read_csv(
        "similarity_score/casp14_foldseek_bio2token_out.tsv", sep="\t"
    )
    casp14_usalign_bio2token_out = pd.read_csv(
        "similarity_score/casp14_usalign_bio2token_out.tsv", sep="\t"
    )
    casp15_foldseek_bio2token_out = pd.read_csv(
        "similarity_score/casp15_foldseek_bio2token_out.tsv", sep="\t"
    )
    casp15_usalign_bio2token_out = pd.read_csv(
        "similarity_score/casp15_usalign_bio2token_out.tsv", sep="\t"
    )
    casp15_bio2token_data = join_dfs(
        casp15_foldseek_bio2token_out, casp15_usalign_bio2token_out
    )
    casp14_bio2token_data = join_dfs(
        casp14_foldseek_bio2token_out, casp14_usalign_bio2token_out
    )
    casp_bio2token_data = pd.concat(
        [casp14_bio2token_data, casp15_bio2token_data], ignore_index=True
    )
    casp_bio2token_data["tool"] = "bio2token"
    return casp_bio2token_data


def boxplot(df, score, output_file):
    mpl.style.use("seaborn-v0_8")
    plt.figure(figsize=(8, 6))
    df.boxplot(column=score, by="tool")
    plt.title(f"{score} by Tool")
    plt.suptitle("")
    plt.xlabel("Tool")
    plt.ylabel(score)
    plt.savefig(output_file)


def main():
    bio2token_df = get_bio2token_data()
    boxplot(bio2token_df, "tmscore", "similarity_score/plots/tmscore_boxplot_bio2token.png")
    boxplot(bio2token_df, "tmscore_fs", "similarity_score/plots/tmscore_fs_boxplot_bio2token.png")
    boxplot(bio2token_df, "rmsd", "similarity_score/plots/rmsd_boxplot_bio2token.png")
    boxplot(bio2token_df, "lddt", "similarity_score/plots/lddt_boxplot_bio2token.png")



if __name__ == "__main__":
    main()

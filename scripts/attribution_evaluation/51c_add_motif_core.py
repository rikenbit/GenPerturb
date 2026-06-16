import re
import pandas as pd
import os
from pathlib import Path

CWD = str(Path(__file__).resolve().parents[2])
OUT_DIR = os.path.join(CWD, "attribution_analysis/captum_union")
os.makedirs(OUT_DIR, exist_ok=True)

JASPAR_MEME_FILES = [
    Path(CWD) / "reference/jaspar/JASPAR2024_CORE_vertebrates_non-redundant_pfms_meme.txt",
    Path(CWD) / "reference/jaspar/JASPAR2022_CORE_vertebrates_non-redundant_pfms_meme.txt",
]


def load_jaspar_motif_widths():
    widths = {}
    for meme_path in JASPAR_MEME_FILES:
        if not meme_path.exists():
            continue
        motif_id = None
        with open(meme_path) as f:
            for line in f:
                if line.startswith("MOTIF "):
                    parts = line.strip().split()
                    motif_id = parts[1]  # e.g. MA0102.4
                elif line.startswith("letter-probability matrix:"):
                    m = re.search(r"w=\s*(\d+)", line)
                    if m and motif_id:
                        widths[motif_id] = int(m.group(1))
    return widths


def extract_jaspar_id(best_match):
    if pd.isna(best_match):
        return None
    parts = best_match.split("_", 1)
    return parts[0] if parts[0].startswith("MA") else None


def add_motif_core_columns(df, jaspar_widths):
    df = df.copy()

    match_col = "best_match" if "best_match" in df.columns else "matched_motif"

    df["jaspar_id"] = df[match_col].apply(extract_jaspar_id)

    df["motif_width"] = df["jaspar_id"].map(jaspar_widths)

    df["seqlet_length"] = df["genomic_end"] - df["genomic_start"]

    seqlet_mid = (df["genomic_start"] + df["genomic_end"]) / 2.0

    has_width = df["motif_width"].notna()

    df["core_genomic_start"] = df["genomic_start"].copy().astype(float)
    df["core_genomic_end"] = df["genomic_end"].copy().astype(float)

    df.loc[has_width, "core_genomic_start"] = (
        seqlet_mid[has_width] - df.loc[has_width, "motif_width"] / 2.0
    )
    df.loc[has_width, "core_genomic_end"] = (
        seqlet_mid[has_width] + df.loc[has_width, "motif_width"] / 2.0
    )

    df["core_genomic_start"] = df["core_genomic_start"].apply(
        lambda x: int(x) if pd.notna(x) else x
    )
    df["core_genomic_end"] = df["core_genomic_end"].apply(
        lambda x: int(x + 0.5) if pd.notna(x) else x
    )

    df["core_length"] = df["core_genomic_end"] - df["core_genomic_start"]

    return df


def main():
    print("Loading JASPAR motif widths...")
    jaspar_widths = load_jaspar_motif_widths()
    print(f"  {len(jaspar_widths)} motif widths loaded")

    widths = list(jaspar_widths.values())
    print(f"  Width range: {min(widths)}-{max(widths)} bp, median: {sorted(widths)[len(widths)//2]} bp")

    all_path = os.path.join(OUT_DIR, "seqlet_loci_all.tsv")
    print(f"\nProcessing {all_path}...")
    df_all = pd.read_csv(all_path, sep="\t")

    df_all_core = add_motif_core_columns(df_all, jaspar_widths)

    out_all = os.path.join(OUT_DIR, "seqlet_loci_all_core.tsv")
    df_all_core.to_csv(out_all, sep="\t", index=False)
    print(f"  Saved: {out_all} ({len(df_all_core):,} rows)")

    has_match = df_all_core["jaspar_id"].notna()
    has_width = df_all_core["motif_width"].notna()
    print(f"  JASPAR ID extracted: {has_match.sum():,} / {len(df_all_core):,} ({has_match.mean()*100:.1f}%)")
    print(f"  Motif width resolved: {has_width.sum():,} / {has_match.sum():,} ({has_width.sum()/max(has_match.sum(),1)*100:.1f}%)")

    resolved = df_all_core[has_width]
    print(f"\n  Seqlet length:  median={resolved['seqlet_length'].median():.0f} bp, "
          f"mean={resolved['seqlet_length'].mean():.1f} bp")
    print(f"  Core length:    median={resolved['core_length'].median():.0f} bp, "
          f"mean={resolved['core_length'].mean():.1f} bp")
    print(f"  Reduction:      {(1 - resolved['core_length'].mean()/resolved['seqlet_length'].mean())*100:.0f}% shorter")

    matched_path = os.path.join(OUT_DIR, "seqlet_loci_matched.tsv")
    print(f"\nProcessing {matched_path}...")
    df_matched = pd.read_csv(matched_path, sep="\t")

    df_matched_core = add_motif_core_columns(df_matched, jaspar_widths)

    out_matched = os.path.join(OUT_DIR, "seqlet_loci_matched_core.tsv")
    df_matched_core.to_csv(out_matched, sep="\t", index=False)
    print(f"  Saved: {out_matched} ({len(df_matched_core):,} rows)")

    has_width_m = df_matched_core["motif_width"].notna()
    resolved_m = df_matched_core[has_width_m]
    print(f"  Motif width resolved: {has_width_m.sum():,} / {len(df_matched_core):,}")
    print(f"  Core length:    median={resolved_m['core_length'].median():.0f} bp, "
          f"mean={resolved_m['core_length'].mean():.1f} bp")

    print("\nDone.")


if __name__ == "__main__":
    main()

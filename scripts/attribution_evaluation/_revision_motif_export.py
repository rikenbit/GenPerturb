"""Export saved match flags and upstream status; no motif rematching or inference."""
from pathlib import Path
import pandas as pd


def export_revision(study_name, study, output):
    root = Path(__file__).resolve().parents[2]
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    symbols = set(pd.read_csv(root / "reference/humantfs/DatabaseExtract_v_1.01.txt", sep="\t")["HGNC symbol"].dropna())
    columns = pd.read_csv(root / "data" / (study_name + ".tsv"), sep="\t", nrows=0).columns[2:]
    selected = [p for p in columns if any(t in symbols for t in p.split(".", 1)[-1].split("_"))]
    figures = root / "figures" / study
    gimme_path = figures / "gimmemotifs/gimme_roc_motif_match_q5e-02.txt"
    gimme = pd.read_csv(gimme_path, sep="\t")
    gene = pd.read_csv(figures / "tfmodisco/tfmodisco_gene_match.txt", sep="\t")
    cluster = pd.read_csv(figures / "tfmodisco/tfmodisco_cluster_match.txt", sep="\t")
    rows, matches, upstream = [], [], []
    for pert in sorted(selected):
        tf_path = root / "tfmodisco" / study / pert / "modisco_result" / f"{pert}_MA_list.txt"
        for method in ["TF-MoDISco", "attribution", "re2g_extended", "re2g", "abc_score", "tss_1kbp", "attribution_shuffle"]:
            path = tf_path if method == "TF-MoDISco" else root / "gimme_results" / study / pert / method / "gimme.roc.report.txt"
            status, n_report, error = "not_analysed_or_missing_report", 0, ""
            if path.exists():
                try:
                    report = pd.read_csv(path, sep="\t")
                    n_report = len(report)
                    if method == "TF-MoDISco":
                        if not {"qval", "pattern", "match"}.issubset(report.columns):
                            raise ValueError("Missing report columns")
                    elif not any("p-value" in c.lower() for c in report.columns):
                        raise ValueError("Missing report p-value column")
                    status = "analysed_not_recovered"
                except Exception as exc:
                    status, error = "unreadable_report", str(exc)
            bed_name = f"attribution_{pert}.shuffle.bed" if method == "attribution_shuffle" else f"{method}_{pert}.bed"
            bed = root / "cre" / study / pert / bed_name
            if status == "not_analysed_or_missing_report" and bed.exists() and bed.stat().st_size == 0:
                status = "no_candidates"
            if method == "TF-MoDISco":
                gd = gene[(gene.perturbation == pert) & (gene.qval < .05)].copy()
                cd = cluster[(cluster.perturbation == pert) & (cluster.qval < .05)].copy()
                gd["gene_match_flag"], gd["cluster_match_flag"] = 1, 0
                cd["gene_match_flag"], cd["cluster_match_flag"] = 0, 1
                detail = pd.concat([gd, cd], ignore_index=True)
                gm, cm = not gd.empty, not cd.empty
            else:
                detail = gimme[(gimme.perturbation == pert) & (gimme.bed_type == method)].copy()
                gm = bool(detail.gene_match_flag.eq(1).any())
                cm = bool(detail.cluster_match_flag.eq(1).any())
            if status == "analysed_not_recovered" and (gm or cm):
                status = "recovered"
            if len(detail):
                detail["method"], detail["study"] = method, study
                matches.append(detail)
            rows.append(dict(study=study, perturbation=pert, perturbed_tf=pert.split(".", 1)[-1], method=method,
                             status=status, gene_match_flag=int(gm), cluster_match_flag=int(cm), n_report_rows=n_report,
                             report_path=str(path), selection="exact HGNC TF token in training perturbation target(s)",
                             root_attribution_present=(root / "attribution" / study / pert).is_dir()))
            upstream.append(dict(perturbation=pert, method=method, report_exists=path.exists(), report_size=path.stat().st_size if path.exists() else None,
                                 candidate_bed=str(bed), candidate_exists=bed.exists(), error=error))
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "tf_method_recovery.tsv", sep="\t", index=False)
    pd.DataFrame(upstream).to_csv(out / "upstream_status.tsv", sep="\t", index=False)
    pd.concat(matches, ignore_index=True).to_csv(out / "matched_motif_details.tsv", sep="\t", index=False)
    frame.groupby(["method", "status"]).size().rename("n_perturbations").reset_index().to_csv(out / "recovery_status_counts.tsv", sep="\t", index=False)

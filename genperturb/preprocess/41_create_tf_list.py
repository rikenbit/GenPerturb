# conda activate singlecell
import pandas as pd
import os



tf_list = pd.read_csv("reference/humantfs/DatabaseExtract_v_1.01.txt", sep="\t", usecols=["HGNC symbol"]).rename(columns={"HGNC symbol": "TF_flag"})

cluster = pd.read_csv("reference/jaspar/clusters.tab", sep="\t", usecols=[0,2])
cluster_motif = cluster.set_index("cluster")["name"].str.split(",",expand=True).stack().reset_index().rename(columns={0:"cluster_gene"}).drop("level_1", axis=1)
cluster_motif = cluster_motif[~(cluster_motif["cluster_gene"].str.contains("::"))]

studies = [
    "NormanWeissman2019_filtered_mixscape_exnp",
    "ReplogleWeissman2022_K562_gwps_mixscape_exnp",
    "ReplogleWeissman2022_K562_essential_mixscape_exnp",
    "ReplogleWeissman2022_rpe1_mixscape_exnp",
]


study = "NormanWeissman2019_filtered_mixscape_exnp"

df  = pd.read_csv(f"data/{study}_tpm.tsv", sep="\t", index_col=[0]).astype("float32")
pert = pd.DataFrame({"Perturbation":df.columns.to_list()})
if study == "NormanWeissman2019_filtered_mixscape_exnp":
    pert.loc[:, ["Gene1", "Gene2"]] = pd.DataFrame([i.split(".")[1].split("_") for i in pert["Perturbation"]]).values
    pert = pert.set_index("Perturbation").unstack().reset_index().rename(columns={0:"Pert"}).query('Pert != "None"')
else:
    pert.loc[:, ["Pert"]] = pd.DataFrame([i.split(".")[1] for i in pert["Perturbation"]]).values



summary = pd.merge(pert.drop("level_0", axis=1), tf_list, left_on="Pert", right_on="TF_flag", how="left")
summary = pd.merge(summary, cluster_motif, left_on="Pert", right_on="cluster_gene",how="left")
summary = pd.merge(summary, cluster_motif.rename(columns={"cluster_gene":"cluster_other"}), left_on="cluster", right_on="cluster",how="left")
tf_check = list(set(summary["Pert"].to_list() + summary["cluster_other"].to_list()))

summary.drop_duplicates().to_csv(f"reference/tf_summary/{study}_tf_summary.txt", sep="\t", index=False)

with open(f"reference/tf_summary/{study}.txt", "w") as f:
  for gene in tf_check:
    if gene is None:
      continue
    else:
      f.write(f"{gene}\n")



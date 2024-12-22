import numpy as np
import pandas as pd
import modisco
import sys
import os
import h5py
import subprocess
from collections import OrderedDict, Counter
from genperturb.dataloaders._genome import GenomeIntervalDataset
from bs4 import BeautifulSoup

study_name = sys.argv[1]
study_suffix = sys.argv[2]
#study_name   = "NormanWeissman2019_filtered_mixscape"
#study_suffix = "enformer_transfer_epoch30_batch256"
study        = f'{study_name}__{study_suffix}'




def run_tfmodisco_workflow(condition, task_to_scores, task_to_hyp_scores, onehot_data):
    """
    - Sliding Window Size and Flanks:
    Sliding Window Size (Default: 21): Used for scanning motifs, set to 15 for TAL and GATA motifs. Adjust based on the expected length of the core motif.
    Flanks (Default: 10): Used during motif scanning, set to 5 for TAL and GATA motifs. Flank sizes should be adjusted based on the expected length of motif flanks.
    - Motif Trimming and Expansion:
    trim_to_window_size (Default: 30): During seqlet clustering, motifs are trimmed to this central size with the highest importance. Set to 15. After trimming, the motif is expanded on either side by initial_flank_to_add.
    initial_flank_to_add (Default: 10): Set to 5. Used to expand the motif after trimming.
    final_flank_to_add (Default: 0): Expanded after processing motifs in the pipeline. Set to 5 to reveal subtle flanking sequences. The final motif length is calculated as trim_to_window_size + 2 x initial_flank_to_add + 2 x final_flank_to_add.

    - Clustering Parameters:    
    final_min_cluster_size (Default: 30): Set to 60. Used to filter out small clusters with weak support (fewer than 60 seqlets).

    - Seqlet Filtering and FDR Control:    
    target_seqlet_fdr: Controls the noisiness of seqlets. Used to identify "significant" seqlets by smoothing importance scores and fitting a Laplace distribution. Threshold set such that the false discovery rate (FDR) is less than target_seqlet_fdr.
    sliding_window_size: Used for smoothing importance scores.
    min_passing_windows_frac (Default: 0.03): If the number of windows passing the FDR threshold is smaller than this fraction, adjust the threshold.
    max_passing_windows_frac (Default: 0.2): If the number of windows passing the FDR threshold is larger than this fraction, adjust the threshold.
    """
    if condition == "defalt":
        sliding_window_size = 21 #15
        flank_size = 10 #5
        target_seqlet_fdr = 0.3 #0.1
        trim_to_window_size = 30 #15
        initial_flank_to_add = 10 #5
        final_flank_to_add = 0 #5
        final_min_cluster_size = 100 #60
    elif condition == "short":
        sliding_window_size = 15
        flank_size = 5
        target_seqlet_fdr = 0.2
        trim_to_window_size = 15
        initial_flank_to_add = 5 
        final_flank_to_add = 5
        final_min_cluster_size = 20
    elif condition == "long":
        sliding_window_size = 15
        flank_size = 5
        target_seqlet_fdr = 0.2
        trim_to_window_size = 6
        initial_flank_to_add = 10 #5
        final_flank_to_add = 0
        final_min_cluster_size = 20 #60
    null_per_pos_scores = modisco.coordproducers.LaplaceNullDist(num_to_samp=5000)
    tfmodisco_results = modisco.tfmodisco_workflow.workflow.TfModiscoWorkflow(
                        sliding_window_size=sliding_window_size,
                        flank_size=flank_size,
                        target_seqlet_fdr=target_seqlet_fdr,
                        seqlets_to_patterns_factory=
                         modisco.tfmodisco_workflow.seqlets_to_patterns.TfModiscoSeqletsToPatternsFactory(
                            trim_to_window_size=trim_to_window_size,
                            initial_flank_to_add=initial_flank_to_add,
                            final_flank_to_add=final_flank_to_add,
                            final_min_cluster_size=final_min_cluster_size,
                            subcluster_perplexity=10,
                            n_cores=16)
                    )(
                     task_names=["task0"],
                     contrib_scores=task_to_scores,
                     hypothetical_contribs=task_to_hyp_scores,
                     one_hot=onehot_data,
                     null_per_pos_scores=null_per_pos_scores)
    return tfmodisco_results


def process_html_to_dataframe(filename, report_dir, pert):
    with open(filename, 'r') as file:
        html_content = file.read()
    soup = BeautifulSoup(html_content, 'html.parser')
    headers = [th.get_text() for th in soup.find_all('th')]
    rows = []
    for tr in soup.find_all('tr'):
        row = [td.get_text() for td in tr.find_all('td')]
        if row:
            rows.append(row)
    df = pd.DataFrame(rows, columns=headers)
    df = df[[col for col in df.columns if 'logo' not in col and 'cwm' not in col]]
    match_qval_columns = [col for col in df.columns if 'match' in col or 'qval' in col]
    num_sets = len(match_qval_columns) // 2  # Assuming pairs of match and qval
    reshaped_rows = []
    for index, row in df.iterrows():
        for i in range(num_sets):
            match_col = f'match{i}'
            qval_col = f'qval{i}'
            if match_col in df.columns and qval_col in df.columns:
                reshaped_rows.append({
                    'pattern': row['pattern'],
                    'num_seqlets': row['num_seqlets'],
                    'match': row[match_col],
                    'qval': row[qval_col]
                })
    reshaped_df = pd.DataFrame(reshaped_rows, columns=['pattern', 'num_seqlets', 'match', 'qval']).dropna()
    reshaped_df["qval"] = reshaped_df["qval"].astype("float32")
    reshaped_df["perturbation"] = pert
    reshaped_df.to_csv(f"{report_dir}/{pert}_MA_list.txt", sep="\t", index=False)


def run_modisco_workflow(study, pert, suffix, suffix_modisco="", condition="", attribution="ixg", context_length=128):
    attr_path = f"tfmodisco/{study}/{pert}{suffix}/{pert}{suffix}.h5"
    bed = f'attribution_pert/{study}/{pert}/00_all_{pert}_{attribution}_fc.bed'
    fasta = 'fasta/GRCh38.p13.genome.fa'
    meme_motif = "reference/jaspar/JASPAR2024_CORE_vertebrates_non-redundant_pfms_meme_modified.txt"

    ds = GenomeIntervalDataset(bed_file=bed, fasta_file=fasta, context_length=context_length)
    onehot_data = [ds[i].detach().numpy().astype("int8") for i in range(len(ds))]

    task_to_scores = OrderedDict()
    task_to_hyp_scores = OrderedDict()

    with h5py.File(attr_path, "r") as f:
        task_to_scores["task0"] = [i for i in f[attribution]]
        task_to_hyp_scores["task0"] = [i for i in f["saliency"]]

    modisco_dat_dir = f"tfmodisco/{study}/{pert}{suffix}"
    modisco_dat_file = f"{modisco_dat_dir}/{pert}{suffix}{suffix_modisco}_modisco.h5"
    converted_dat_file = f"{modisco_dat_dir}/{pert}{suffix}{suffix_modisco}_modisco_converted.h5"
    report_dir = f"{modisco_dat_dir}/modisco_result{suffix_modisco}"

    try:
        tfmodisco_results = run_tfmodisco_workflow(condition, task_to_scores, task_to_hyp_scores, onehot_data)
        os.makedirs(modisco_dat_dir, exist_ok=True)
        with h5py.File(modisco_dat_file, "w") as grp:
            tfmodisco_results.save_hdf5(grp)
    except Exception as e:
        print(f"Error modisco : {pert}")
        with open(f"{modisco_dat_dir}/no_motif_modisco.log", "w") as error_log:
            error_log.write(f"An error occurred: {e}")

    try:
        convert_command = f"modisco convert -i {modisco_dat_file} -o {converted_dat_file}"
        subprocess.check_output(convert_command, shell=True)
        report_command = f"modisco report -i {converted_dat_file} -o {report_dir} -m {meme_motif} -n 30"
        subprocess.check_output(report_command, shell=True)
        process_html_to_dataframe(f"{report_dir}/motifs.html", report_dir, pert)
    except Exception as e:
        with open(f"{modisco_dat_dir}/no_motif_meme.log", "w") as error_log:
            error_log.write(f"An error occurred: {e}")



suffix  = ""
directory_path = f"tfmodisco/{study}/"

for pert in os.listdir(directory_path):
    suffix_modisco = "_short_fdr02"
    run_modisco_workflow(study, pert, suffix, suffix_modisco=suffix_modisco, condition="short")




import numpy as np
import pandas as pd
import modiscolite
from modiscolite import tfmodisco, io as modisco_io
import sys
import os
import h5py
import subprocess
from collections import OrderedDict
from genperturb.dataloaders._genome import GenomeIntervalDataset
from bs4 import BeautifulSoup

study_name = sys.argv[1]
study_suffix = sys.argv[2]
study        = f'{study_name}__{study_suffix}'

single_tf = sys.argv[3] if len(sys.argv) >= 4 else None


def run_tfmodisco_workflow(condition, hypothetical_contribs, onehot_data):
    if condition == "defalt":
        sliding_window_size = 21
        flank_size = 10
        target_seqlet_fdr = 0.3
        trim_to_window_size = 30
        initial_flank_to_add = 10
        final_flank_to_add = 0
        final_min_cluster_size = 100
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
        initial_flank_to_add = 10
        final_flank_to_add = 0
        final_min_cluster_size = 20

    pos_patterns, neg_patterns = tfmodisco.TFMoDISco(
        one_hot=onehot_data,
        hypothetical_contribs=hypothetical_contribs,
        sliding_window_size=sliding_window_size,
        flank_size=flank_size,
        target_seqlet_fdr=target_seqlet_fdr,
        trim_to_window_size=trim_to_window_size,
        initial_flank_to_add=initial_flank_to_add,
        final_flank_to_add=final_flank_to_add,
        final_min_cluster_size=final_min_cluster_size,
        subcluster_perplexity=10,
        verbose=True,
    )
    return pos_patterns, neg_patterns


def process_html_to_dataframe(filename, report_dir, pert):
    with open(filename, 'r') as file:
        html_content = file.read()
    soup = BeautifulSoup(html_content, 'html.parser')

    tables = soup.find_all('table')
    if not tables:
        raise RuntimeError(f"No tables found in {filename}")

    summary_table = tables[0]
    patterns_info = []
    for tr in summary_table.find_all('tr'):
        tds = tr.find_all('td')
        if not tds:
            continue
        pattern_span = tds[0].find('span', class_='pattern-id')
        pattern_name = pattern_span.get_text().strip() if pattern_span else tds[0].get_text().strip()
        num_seqlets = tds[1].get_text().strip()
        patterns_info.append((pattern_name, num_seqlets))

    rank_tables = []
    for table in tables[1:]:
        headers = [th.get_text().strip() for th in table.find_all('th')]
        if 'Rank' in headers and 'Match' in headers:
            rank_tables.append(table)

    reshaped_rows = []
    for idx, (pattern_name, num_seqlets) in enumerate(patterns_info):
        if idx < len(rank_tables):
            for tr in rank_tables[idx].find_all('tr'):
                tds = tr.find_all('td')
                if not tds:
                    continue
                match_val = tds[1].get_text().strip()
                qval_val = tds[3].get_text().strip()
                if match_val and qval_val:
                    reshaped_rows.append({
                        'pattern': pattern_name,
                        'num_seqlets': num_seqlets,
                        'match': match_val,
                        'qval': qval_val,
                    })

    reshaped_df = pd.DataFrame(reshaped_rows, columns=['pattern', 'num_seqlets', 'match', 'qval'])
    reshaped_df["qval"] = pd.to_numeric(reshaped_df["qval"], errors="coerce")
    reshaped_df = reshaped_df.dropna(subset=["qval"])
    reshaped_df["qval"] = reshaped_df["qval"].astype("float32")
    reshaped_df["perturbation"] = pert
    reshaped_df.to_csv(f"{report_dir}/{pert}_MA_list.txt", sep="\t", index=False)


def run_modisco_workflow(study, pert, suffix, suffix_modisco="", condition="", attribution="ixg", context_length=128):
    attr_path = f"attribution/{study}/{pert}{suffix}/{pert}{suffix}.h5"
    bed = f'attribution/{study}/{pert}{suffix}/{pert}{suffix}_peaks_bin128bp.bed'
    fasta = 'fasta/GRCh38.p14.genome.fa'
    meme_motif = "reference/jaspar/JASPAR2022_CORE_vertebrates_non-redundant_pfms_meme_modified.txt"

    ds = GenomeIntervalDataset(bed_file=bed, fasta_file=fasta, context_length=context_length)
    onehot_data = np.array([ds[i].detach().numpy().astype("float32") for i in range(len(ds))])

    with h5py.File(attr_path, "r") as f:
        hypothetical_contribs = np.array(f["saliency"])  # (n, seq_len, 4)

    modisco_dat_dir = f"tfmodisco/{study}/{pert}{suffix}"
    modisco_dat_file = f"{modisco_dat_dir}/{pert}{suffix}{suffix_modisco}_modisco_v2.h5"
    report_dir = f"{modisco_dat_dir}/modisco_result{suffix_modisco}"

    log_files = [
        f"{modisco_dat_dir}/no_motif_meme.log",
        f"{modisco_dat_dir}/no_motif_modisco.log"
    ]
    for log_file in log_files:
        if os.path.exists(log_file):
            os.remove(log_file)
            print(f"Removed existing log file: {log_file}")

    try:
        pos_patterns, neg_patterns = run_tfmodisco_workflow(
            condition, hypothetical_contribs, onehot_data
        )
        os.makedirs(modisco_dat_dir, exist_ok=True)
        modisco_io.save_hdf5(
            modisco_dat_file, pos_patterns, neg_patterns,
            window_size=context_length
        )
    except Exception as e:
        print(f"Error modisco : {pert}")
        os.makedirs(modisco_dat_dir, exist_ok=True)
        with open(f"{modisco_dat_dir}/no_motif_modisco.log", "w") as error_log:
            error_log.write(f"An error occurred: {e}")

    try:
        os.makedirs(report_dir, exist_ok=True)
        report_command = f"modisco report -i {modisco_dat_file} -o {report_dir} -m {meme_motif} -n 100"
        subprocess.check_output(report_command, shell=True, stderr=subprocess.STDOUT)
        html_candidates = [f"{report_dir}/report.html", f"{report_dir}/motifs.html", f"{report_dir}/index.html"]
        html_file = next((c for c in html_candidates if os.path.exists(c)), None)
        if html_file is None:
            raise FileNotFoundError(f"Report HTML not found in {report_dir}")
        process_html_to_dataframe(html_file, report_dir, pert)
    except Exception as e:
        with open(f"{modisco_dat_dir}/no_motif_meme.log", "w") as error_log:
            error_log.write(f"An error occurred: {e}")


suffix  = ""

if single_tf:
    print(f"[INFO] Single TF mode: processing {single_tf}")
    run_modisco_workflow(study, single_tf, suffix, suffix_modisco="", condition="short")
else:
    directory_path = f"attribution/{study}/"
    for pert in os.listdir(directory_path):
        suffix_modisco = ""
        run_modisco_workflow(study, pert, suffix, suffix_modisco=suffix_modisco, condition="short")

# dataset_model_config.py
# Centralized configuration for studies, models, and related variables

all_studies = [
    "NormanWeissman2019_filtered_mixscape_exnp_train",
    "ReplogleWeissman2022_K562_essential_mixscape_exnp_train",
    "ReplogleWeissman2022_K562_gwps_mixscape_exnp_train",
    "ReplogleWeissman2022_rpe1_mixscape_exnp_train",
    "JialongJiang2024_Myeloid_train",
    "JialongJiang2024_CD4T_train",
    "JialongJiang2024_CD8T_train",
    "JialongJiang2024_B_cell_train",
    "Srivatsan2019_A549_train",
    "Srivatsan2019_K562_train",
    "Srivatsan2019_MCF7_train",
    # MartinRufino
    "MartinRufino2025_mixscape_exnp_train",
    # Wu et al., 2024 (GSE277747) — Mixscape-filtered (only SMARCA4 survives)
    "Wu2024_mixscape_exnp_train",
    # Wu et al., 2024 (GSE277747) — non-Mixscape (80+ perts)
    "Wu2024_train",
    # Shevade et al., 2025 (GSE288996, K562 DMSO bulk)
    "Shevade2025_K562_DMSO_mixscape_exnp_train",
]

all_xlabels = [
    "Norman et al.",
    "Replogle et al. essential",
    "Replogle et al. gwps",
    "Replogle et al. rpe1",
    "Jialong et al. myeloid",
    "Jialong et al. CD4T",
    "Jialong et al. CD8T",
    "Jialong et al. Bcell",
    "Srivatsan et al. A549",
    "Srivatsan et al. K562",
    "Srivatsan et al. MCF7",
    # MartinRufino
    "Martin-Rufino et al.",
    # Wu (Mixscape-filtered)
    "Wu et al. (Mixscape)",
    # Wu (all cells, no Mixscape)
    "Wu et al.",
    # Shevade
    "Shevade et al. K562 DMSO",
]

all_datanames = [
    "Norman et al.",
    "Replogle et al. essential",
    "Replogle et al. gwps",
    "Replogle et al. rpe1",
    "Jialong et al. myeloid",
    "Jialong et al. CD4T",
    "Jialong et al. CD8T",
    "Jialong et al. Bcell",
    "Srivatsan et al. A549",
    "Srivatsan et al. K562",
    "Srivatsan et al. MCF7",
    # MartinRufino
    "Martin-Rufino et al.",
    # Wu (Mixscape-filtered)
    "Wu et al. (Mixscape)",
    # Wu (all cells, no Mixscape)
    "Wu et al.",
    # Shevade
    "Shevade et al. K562 DMSO",
]

all_pretrained_models = [
    "alphagenome",
    "borzoi",
    "enformer",
    "alphagenome_fold_0",
    "alphagenome_fold_1",
    "alphagenome_fold_2",
    "alphagenome_fold_3",
    "enformerborzoi524k",
    "baseline_control",
    "baseline_peturbmean",
    "simplecnn",
]

all_cmaps = [
    "Reds_r",      # alphagenome
    "Greens_r",    # borzoi
    "Blues_r",     # enformer
    "Reds_r",      # alphagenome_fold_0
    "Reds_r",      # alphagenome_fold_1
    "Reds_r",      # alphagenome_fold_2
    "Reds_r",      # alphagenome_fold_3
    "Blues_r",     # enformerborzoi524k (enformer family)
    "Greys_r",     # baseline_control
    "Greys_r",     # baseline_peturbmean
    "Greys_r",     # simplecnn
    # The entries below are not aligned with all_pretrained_models;
    # they were originally study-related. Kept for backward compat.
    "Reds_r",
    "Oranges_r",
    "Oranges_r",
    "Greens_r",
]

all_study_suffixes = [
    "transfer_epoch100_batch256_adamw5e3",
    "finetuning_epoch20_batch8_adamw5e3_lora_r16a2",
    "finetuning_epoch20_batch8_adamw5e3_lora_r64a2",
    "finetuning_epoch20_batch8_adamw5e3_lora_r256a2",
    "finetuning_epoch20_batch8_adamw5e3_lora_r512a2",
    "finetuning_epoch20_batch8_adamw5e3_lora_r1024a2",
    "finetuning_epoch40_batch2_adamw5e3_full",
    "finetuning_epoch40_batch2_adamw5e3_full_plr1e10",
]

# Models whose training suffix differs from the standard study_suffixes.
# When iterating pretrained_models × study_suffixes, scripts should check
# this dict first: if the model appears here, use the fixed suffix instead
# of every entry in study_suffixes.
model_fixed_suffix = {
    "simplecnn": "baseline_epoch150_batch2_adamw5e3",
}

# Model comparison colors. Refined editorial palette — each model in its own family:
# alphagenome -> rose-wine, borzoi -> sage-forest, enformer -> steel blue, baselines -> gray.
_model_color_map = {
    "alphagenome": "#BC5765",
    "borzoi": "#6B9D7A",
    "enformer": "#4577A8",
    "alphagenome_fold_0": "#943B47",
    "alphagenome_fold_1": "#BC5765",
    "alphagenome_fold_2": "#D08791",
    "alphagenome_fold_3": "#E5B5BB",
    "enformerborzoi524k": "#7298BD",
    "baseline_control": "#B5BBC1",
    "baseline_peturbmean": "#8E949A",
    "simplecnn": "#7B838B",
}

name_replace = {
    "enformer": "GenPerturb (Enformer transfer)",
    "borzoi": "GenPerturb (Borzoi transfer)",
    "alphagenome": "GenPerturb (AlphaGenome transfer)",
    "alphagenome_fold_0": "GenPerturb (AlphaGenome fold0)",
    "alphagenome_fold_1": "GenPerturb (AlphaGenome fold1)",
    "alphagenome_fold_2": "GenPerturb (AlphaGenome fold2)",
    "alphagenome_fold_3": "GenPerturb (AlphaGenome fold3)",
    "enformerborzoi524k": "GenPerturb (Enformer Borzoi 524k)",
    "baseline_control": "Baseline (Control)",
    "baseline_peturbmean": "Baseline (Perturbation Mean)",
    "simplecnn": "GenPerturb (Simple CNN)",
}

_active_study_indices = [
    0,   # NormanWeissman2019
    1,   # ReplogleWeissman2022_K562_essential
    2,   # ReplogleWeissman2022_K562_gwps
    3,   # ReplogleWeissman2022_rpe1
    4,   # JialongJiang2024_Myeloid
    5,   # JialongJiang2024_CD4T
    6,   # JialongJiang2024_CD8T
    7,   # JialongJiang2024_B_cell
    8,   # Srivatsan2019_A549
    9,   # Srivatsan2019_K562
    10,  # Srivatsan2019_MCF7
    11,  # MartinRufino2025_mixscape_exnp
]

_active_model_indices = [
    0,   # alphagenome
    1,   # borzoi
    2,   # enformer
    #3,   # alphagenome_fold_0
    #4,   # alphagenome_fold_1
    #5,   # alphagenome_fold_2
    #6,   # alphagenome_fold_3
    #7,   # enformerborzoi524k
    #8,   # baseline_control
    #9,   # baseline_peturbmean
    10,  # simplecnn
]

_active_suffix_indices = [
    0,   # transfer_epoch100_batch256_adamw5e3
    #1,   # finetuning_epoch20_batch8_adamw5e3_lora_r16a2
    #2,   # finetuning_epoch20_batch8_adamw5e3_lora_r64a2
    #3,   # finetuning_epoch20_batch8_adamw5e3_lora_r256a2
    #4,   # finetuning_epoch20_batch8_adamw5e3_lora_r512a2
    #5,   # finetuning_epoch20_batch8_adamw5e3_lora_r1024a2
    #6,   # finetuning_epoch40_batch2_adamw5e3_full
    #7,   # finetuning_epoch40_batch2_adamw5e3_full_plr1e10
]

studies = [all_studies[i] for i in _active_study_indices]
xlabels = [all_xlabels[i] for i in _active_study_indices]
datanames = [all_datanames[i] for i in _active_study_indices]
pretrained_models = [all_pretrained_models[i] for i in _active_model_indices]
cmaps = [all_cmaps[i] for i in _active_model_indices]
model_colors = [_model_color_map[m] for m in pretrained_models]
study_suffixes = [all_study_suffixes[i] for i in _active_suffix_indices]

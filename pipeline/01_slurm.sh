#!/bin/bash
# A script for submitting jobs using either qsub or sbatch.
# The commands depend on the individual server.

set -euo pipefail

# STUDY="NormanWeissman2019_filtered_mixscape_exnp_train"
# MODEL="alphagenome"
# STUDY_SUFFIX="${MODEL}_transfer_epoch100_batch256_adamw5e3"

if [ -n "${PROJECT_ROOT:-}" ]; then
  SCRIPT_DIR="${PROJECT_ROOT}/pipeline"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/pipeline/_common.sh" ]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}/pipeline"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
source "${SCRIPT_DIR}/_common.sh"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

PARAM=""
while getopts ":p:" opt; do
  case $opt in
    p)
      PARAM="$OPTARG"
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done

shift $((OPTIND-1))

if [ -z "$PARAM" ]; then
  echo "Usage: $0 [-p PARAM]"
  exit 1
fi

activate_model_env() {
  local model="$1"
  if [ "$model" == "enformer" ] || [ "$model" == "enformer_masked" ]; then
    pipeline_activate_conda enformer
  elif [ "$model" == "borzoi" ]; then
    pipeline_activate_conda borzoi
  elif [ "$model" == "alphagenome" ] || [[ "$model" == alphagenome_fold_* ]]; then
    pipeline_activate_conda alphagenome
  elif [ "$model" == "simplecnn" ]; then
    pipeline_activate_conda bend
  else
    set +u
    source "${CONDA_SH:?CONDA_SH is required}"
    set -u
    echo "WARNING: Unknown model '${model}', not activating a specific conda env." >&2
  fi
}

print_conda_envs() {
  conda info -e
}

if [ "$PARAM" == "model" ]; then
  STUDY="${1:?Usage: $0 -p model STUDY PLAN MODEL [LORA_R LORA_ALPHA]}"
  STUDY_PLAN="${2:?Usage: $0 -p model STUDY PLAN MODEL [LORA_R LORA_ALPHA]}"
  MODEL="${3:?Usage: $0 -p model STUDY PLAN MODEL [LORA_R LORA_ALPHA]}"
  LORA_R="${4:-}"
  LORA_ALPHA="${5:-}"
  activate_model_env "${MODEL}"
  print_conda_envs
  export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
  if [ -n "$LORA_R" ] && [ -n "$LORA_ALPHA" ]; then
    srun python pipeline/02_slurm_script.py "$STUDY" "$STUDY_PLAN" "$MODEL" --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA"
  else
    srun python pipeline/02_slurm_script.py "$STUDY" "$STUDY_PLAN" "$MODEL"
  fi

elif [ "$PARAM" == "umap" ] ; then
  pipeline_activate_conda singlecell
  print_conda_envs
  srun python scripts/expression_model_performance/21_embedding_signature.py
  srun python scripts/expression_model_performance/22_compare_model_clustering.py
  srun python scripts/immune_differentiation/01_gene_signature.py

elif [ "$PARAM" == "captum" ]; then
  STUDY="${1:?Usage: $0 -p captum STUDY STUDY_SUFFIX MODEL}"
  STUDY_SUFFIX="${2:?Usage: $0 -p captum STUDY STUDY_SUFFIX MODEL}"
  MODEL="${3:?Usage: $0 -p captum STUDY STUDY_SUFFIX MODEL}"
  activate_model_env "${MODEL}"
  print_conda_envs
  if [ "$MODEL" == "enformer" ]; then
    #srun python scripts/attribution_evaluation/10_captum.py $STUDY $STUDY_SUFFIX across_perturbations top
    #srun python scripts/attribution_evaluation/10_captum.py $STUDY $STUDY_SUFFIX across_perturbations all
    #srun python scripts/attribution_evaluation/10_captum.py $STUDY $STUDY_SUFFIX across_perturbations condition
    srun python scripts/attribution_evaluation/10_captum.py $STUDY $STUDY_SUFFIX variable_genes condition
    #srun python scripts/attribution_evaluation/10_captum.py $STUDY $STUDY_SUFFIX variable_genes all
    #srun python scripts/attribution_evaluation/10_captum.py $STUDY $STUDY_SUFFIX variable_genes tf
    #srun python scripts/attribution_evaluation/10_captum.py $STUDY $STUDY_SUFFIX across_genes "test"
  elif [[ "$MODEL" == "borzoi" ]]; then
    #srun python scripts/attribution_evaluation/10_captum.py $STUDY $STUDY_SUFFIX across_perturbations top
    #srun python scripts/attribution_evaluation/10_captum.py $STUDY $STUDY_SUFFIX across_perturbations all
    #srun python scripts/attribution_evaluation/10_captum.py $STUDY $STUDY_SUFFIX across_perturbations condition # fig6, interference
    #srun python scripts/attribution_evaluation/10_captum.py $STUDY $STUDY_SUFFIX variable_genes condition  # test
    srun python scripts/attribution_evaluation/10_captum.py $STUDY $STUDY_SUFFIX variable_genes all     # fig5, modisco
    #srun python scripts/attribution_evaluation/10_captum.py $STUDY $STUDY_SUFFIX variable_genes tf      # test for only TF
    #srun python scripts/attribution_evaluation/10_captum.py $STUDY $STUDY_SUFFIX across_genes "test" # fig5, peak enrichment
  fi

elif [ "$PARAM" == "captum_single" ]; then
  STUDY="${1:?Usage: $0 -p captum_single STUDY STUDY_SUFFIX MODEL MODE TARGET PERT}"
  STUDY_SUFFIX="${2:?Usage: $0 -p captum_single STUDY STUDY_SUFFIX MODEL MODE TARGET PERT}"
  MODEL="${3:?Usage: $0 -p captum_single STUDY STUDY_SUFFIX MODEL MODE TARGET PERT}"
  MODE="${4:?Usage: $0 -p captum_single STUDY STUDY_SUFFIX MODEL MODE TARGET PERT}"
  TARGET="${5:?Usage: $0 -p captum_single STUDY STUDY_SUFFIX MODEL MODE TARGET PERT}"
  PERT="${6:?Usage: $0 -p captum_single STUDY STUDY_SUFFIX MODEL MODE TARGET PERT}"
  activate_model_env "${MODEL}"
  print_conda_envs
  srun python scripts/attribution_evaluation/10_captum.py "$STUDY" "$STUDY_SUFFIX" "$MODE" "$TARGET" --pert "$PERT"

elif [ "$PARAM" == "captum_union" ]; then
  STUDY="${1:?Usage: $0 -p captum_union STUDY STUDY_SUFFIX MODEL TARGET PERT GENE_LIST [OUTPUT_SUFFIX]}"
  STUDY_SUFFIX="${2:?Usage: $0 -p captum_union STUDY STUDY_SUFFIX MODEL TARGET PERT GENE_LIST [OUTPUT_SUFFIX]}"
  MODEL="${3:?Usage: $0 -p captum_union STUDY STUDY_SUFFIX MODEL TARGET PERT GENE_LIST [OUTPUT_SUFFIX]}"
  TARGET="${4:?Usage: $0 -p captum_union STUDY STUDY_SUFFIX MODEL TARGET PERT GENE_LIST [OUTPUT_SUFFIX]}"
  PERT="${5:?Usage: $0 -p captum_union STUDY STUDY_SUFFIX MODEL TARGET PERT GENE_LIST [OUTPUT_SUFFIX]}"
  GENE_LIST="${6:?Usage: $0 -p captum_union STUDY STUDY_SUFFIX MODEL TARGET PERT GENE_LIST [OUTPUT_SUFFIX]}"
  OUTPUT_SUFFIX="${7:-_union}"
  activate_model_env "${MODEL}"
  print_conda_envs
  srun python scripts/attribution_evaluation/10_captum.py "$STUDY" "$STUDY_SUFFIX" union_genes "$TARGET" --pert "$PERT" --gene-list "$GENE_LIST" --output-suffix "$OUTPUT_SUFFIX"

elif [ "$PARAM" == "captum_peak" ]; then                                                                                                  
  STUDY="${1:?Usage: $0 -p captum_peak STUDY STUDY_SUFFIX MODEL [MODE]}"
  STUDY_SUFFIX="${2:?Usage: $0 -p captum_peak STUDY STUDY_SUFFIX MODEL [MODE]}"                                                                                                                         
  MODEL="${3:?Usage: $0 -p captum_peak STUDY STUDY_SUFFIX MODEL [MODE]}"
  MODE="${4:-variable_genes}"                                                                                                             
  activate_model_env "${MODEL}"
  print_conda_envs
  #python scripts/attribution_evaluation/10a_peak_call.py $STUDY $STUDY_SUFFIX variable_genes                                        
  #python scripts/attribution_evaluation/10a_peak_call.py $STUDY $STUDY_SUFFIX across_genes                                          
  #python scripts/attribution_evaluation/10a_peak_call.py $STUDY $STUDY_SUFFIX across_perturbations                                  
  python scripts/attribution_evaluation/10a_peak_call.py "$STUDY" "$STUDY_SUFFIX" "$MODE"

elif [ "$PARAM" == "modisco" ]; then
  #sleep 24000
  pipeline_activate_conda modisco
  print_conda_envs
  STUDY="${1:?Usage: $0 -p modisco STUDY STUDY_SUFFIX}"
  STUDY_SUFFIX="${2:?Usage: $0 -p modisco STUDY STUDY_SUFFIX}"
  python scripts/attribution_evaluation/33_run_tfmodisco.py "${STUDY}" "${STUDY_SUFFIX}"

elif [ "$PARAM" == "gimmemotifs" ]; then
  pipeline_activate_conda gimmemotifs
  print_conda_envs
  STUDY="${1:?Usage: $0 -p gimmemotifs STUDY STUDY_SUFFIX}"
  STUDY_SUFFIX="${2:?Usage: $0 -p gimmemotifs STUDY STUDY_SUFFIX}"
  python scripts/attribution_evaluation/31_run_gimmemotifs.py "${STUDY}" "${STUDY_SUFFIX}"


else
  exit 1
fi

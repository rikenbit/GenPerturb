#!/bin/bash
# A script for submitting jobs using either qsub or sbatch.
# The commands depend on the individual server.


# STUDY="NormanWeissman2019_filtered_mixscape_exnp_train"
# MODEL="enformer"
# STUDY_SUFFIX="${MODEL}_transfer_epoch100_batch256_adamw5e3"

CWD="/path_to/GenPerturb"
cd $CWD


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


source ~/.bashrc

STUDY="$1"
MODEL="$3"

if [ "$MODEL" == "enformer" ] || [ "$MODEL" == "enformer_masked" ]; then
  conda activate enformer
elif [ "$MODEL" == "hyena_dna_tss" ] || [ "$MODEL" == "hyena_dna_last" ] || [ "$MODEL" == "hyena_dna_mean" ]; then
  conda activate enformer
  #conda activate hyena_dna
elif [ "$MODEL" == "nucleotide_transformer_tss" ] || [ "$MODEL" == "nucleotide_transformer_cls" ] || [ "$MODEL" == "nucleotide_transformer_mean" ]; then
  conda activate enformer
  #conda activate nucleotide
fi



if [ "$PARAM" == "model" ]; then
  STUDY_PLAN="$2"
  export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
  srun python 02_qsub_script.py $STUDY $STUDY_PLAN $MODEL

elif [ "$PARAM" == "captum" ]; then
  STUDY_SUFFIX="$2"
  if [ "$MODEL" == "enformer" ]; then
    #srun python 10_captum.py $STUDY $STUDY_SUFFIX seq top
    #srun python 10_captum.py $STUDY $STUDY_SUFFIX seq all
    #srun python 10_captum.py $STUDY $STUDY_SUFFIX seq condition
    #srun python 10_captum.py $STUDY $STUDY_SUFFIX pert condition
    #srun python 10_captum.py $STUDY $STUDY_SUFFIX pert all
    #srun python 10_captum.py $STUDY $STUDY_SUFFIX pert tf
    srun python 10_captum.py $STUDY $STUDY_SUFFIX pert_all "test"
  elif [ "$MODEL" == "hyena_dna_tss" ] || [ "$MODEL" == "hyena_dna_last" ] || [ "$MODEL" == "hyena_dna_mean" ]; then
    echo $STUDY $STUDY_SUFFIX $MODEL seq condition
    srun python 10_captum_token.py $STUDY $STUDY_SUFFIX $MODEL seq condition
    #srun python 10_captum_token.py $STUDY $STUDY_SUFFIX $MODEL pert condition
    #srun python 10_captum_token.py $STUDY $STUDY_SUFFIX $MODEL pert tf
  elif [ "$MODEL" == "nucleotide_transformer_tss" ] || [ "$MODEL" == "nucleotide_transformer_cls" ] || [ "$MODEL" == "nucleotide_transformer_mean" ]; then
    echo $STUDY $STUDY_SUFFIX $MODEL seq condition
    srun python 10_captum_token.py $STUDY $STUDY_SUFFIX $MODEL seq condition
    #srun python 10_captum_token.py $STUDY $STUDY_SUFFIX $MODEL pert condition
    #srun python 10_captum_token.py $STUDY $STUDY_SUFFIX $MODEL pert tf
  fi

elif [ "$PARAM" == "evalattr" ]; then
  STUDY_SUFFIX="$2"
  python 11_evaluate_attribution.py $STUDY $STUDY_SUFFIX

elif [ "$PARAM" == "coolbox" ]; then
  conda activate coolbox
  STUDY_SUFFIX="$2"
  python 12_coolbox.py $STUDY $STUDY_SUFFIX

elif [ "$PARAM" == "modisco" ]; then
  conda activate modisco
  STUDY_SUFFIX="$2"
  python 13_tfmodisco.py ${STUDY} ${STUDY_SUFFIX}

else
  exit 1
fi



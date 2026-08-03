#!/usr/bin/env bash
set -euo pipefail

COMP_FILE="${COMP_FILE:-prepare/dojo_competitions.txt}"
DATA_DIR="${MLE_DOJO_DATA_DIR:-./data/mle-dojo-data}"
AGENT_TYPE="mle"
MAX_CONCURRENT=20
TOTAL_TASKS=$(grep -vc '^[[:space:]]*$' "$COMP_FILE")

# OUTPUT_ROOT="./output/qwen3_8b_rl"


# sbatch --qos=${SLURM_QOS:-normal} \
#   --gres=gpu:8 \
#   --cpus-per-task=64 \
#   --mem=500G \
#   --time=4-00:00:00 \
#   --array=1-"${TOTAL_TASKS}"%"${MAX_CONCURRENT}" \
#   --job-name=mle_dojo_qwen3_8b_rl \
#   --output=./slurm/slurm-%x-%A_%a.out \
#   --error=./slurm/slurm-%x-%A_%a.err \
#   --wrap "COMP=\$(sed -n \"\${SLURM_ARRAY_TASK_ID}p\" ${COMP_FILE}); \
# python main.py \
#   --output-dir ${OUTPUT_ROOT}/ \
#   --data-dir ${DATA_DIR} \
#   --competition-name \"\${COMP}\" \
#   --agent-type ${AGENT_TYPE}"


# OUTPUT_ROOT="./output/qwen3_30b_base"
# job_name="mle_dojo_qwen3_30b_base"

# OUTPUT_ROOT="./output/qwen3_30b_sft_mle"
# job_name="mle_dojo_qwen3_30b_sft_mle"


OUTPUT_ROOT="./output/qwen3_30b_rl_mle"
job_name="mle_dojo_qwen3_30b_rl_mle"


sbatch --qos=${SLURM_QOS:-normal} \
  --gres=gpu:1 \
  --cpus-per-task=64 \
  --mem=500G \
  --time=4-00:00:00 \
  --array=1-"${TOTAL_TASKS}"%"${MAX_CONCURRENT}" \
  --job-name=${job_name} \
  --output=./slurm/slurm-%x-%A_%a.out \
  --error=./slurm/slurm-%x-%A_%a.err \
  --wrap "COMP=\$(sed -n \"\${SLURM_ARRAY_TASK_ID}p\" ${COMP_FILE}); \
python main.py --max-steps 30 \
  --output-dir ${OUTPUT_ROOT}/ \
  --data-dir ${DATA_DIR} \
  --competition-name \"\${COMP}\" \
  --agent-type ${AGENT_TYPE}"

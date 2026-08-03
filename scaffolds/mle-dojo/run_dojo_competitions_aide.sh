#!/usr/bin/env bash
set -euo pipefail

COMP_FILE="${COMP_FILE:-prepare/dojo_competitions.txt}"
DATA_DIR="${MLE_DOJO_DATA_DIR:-./data/mle-dojo-data}"
AGENT_TYPE="aide"
MAX_CONCURRENT=20
TOTAL_TASKS=$(grep -vc '^[[:space:]]*$' "$COMP_FILE")

# OUTPUT_ROOT="./output/qwen3_14b_sft_aide"


# sbatch --qos=${SLURM_QOS:-normal} \
#   --gres=gpu:1 \
#   --cpus-per-task=64 \
#   --mem=500G \
#   --time=4-00:00:00 \
#   --array=1-"${TOTAL_TASKS}"%"${MAX_CONCURRENT}" \
#   --job-name=mle_dojo_qwen3_14b_sft_aide \
#   --output=./slurm/slurm-%x-%A_%a.out \
#   --error=./slurm/slurm-%x-%A_%a.err \
#   --wrap "COMP=\$(sed -n \"\${SLURM_ARRAY_TASK_ID}p\" ${COMP_FILE}); \
# python main.py \
#   --output-dir ${OUTPUT_ROOT}/ \
#   --data-dir ${DATA_DIR} \
#   --competition-name \"\${COMP}\" \
#   --agent-type ${AGENT_TYPE}"


# OUTPUT_ROOT="./output/qwen3_30b_base_aide"
# job_name="mle_dojo_qwen3_30b_base_aide"

# OUTPUT_ROOT="./output/qwen3_30b_sft_aide"
# job_name="mle_dojo_qwen3_30b_sft_aide"

OUTPUT_ROOT="./output/qwen3_30b_rl_aide"
job_name="mle_dojo_qwen3_30b_rl_aide"


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
python main.py \
  --output-dir ${OUTPUT_ROOT}/ \
  --data-dir ${DATA_DIR} \
  --competition-name \"\${COMP}\" \
  --agent-type ${AGENT_TYPE}"
#!/bin/bash
# Launch the SRUN execution API that runs agents' generated code steps as Slurm jobs.
# Run this in its own terminal/tmux session BEFORE launching training or evaluation.
#
#   source .env                       # sets SANDMLE_CONDA_ENV, SRUN_API_MAX_CONCURRENT, ...
#   bash examples/deepresearch/scripts/serve_srun_api.sh
#
# Then note the host and export SRUN_API_URL="http://<host>:9000" wherever you run train/eval.

set -euo pipefail

export SRUN_API_MAX_CONCURRENT="${SRUN_API_MAX_CONCURRENT:-128}"
export SANDMLE_CONDA_ENV="${SANDMLE_CONDA_ENV:-sandmle}"

echo "This host: $(hostname -i)"
echo "Serving SRUN API on port 9000 (conda env for code steps: ${SANDMLE_CONDA_ENV})"

cd "$(dirname "$0")/.."   # examples/deepresearch/
uvicorn srun_api:app --host 0.0.0.0 --port 9000 --workers 1

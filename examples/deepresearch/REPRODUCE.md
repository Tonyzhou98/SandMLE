# Reproducing SandMLE (RL training + evaluation)

This walks through the runtime setup and the main experiments. All site-specific values come
from environment variables — copy `.env.example` to `.env`, edit it, and `source .env` in every
terminal below.

## Prerequisites

Two conda environments (they do different jobs):

- **`rllm`** — runs the rLLM trainer / evaluator (this repo). Install rLLM per its own docs plus
  `examples/deepresearch/requirements-sandmle.txt`.
- **`sandmle`** (`SANDMLE_CONDA_ENV`) — the env the SRUN API uses to execute the *agent's* generated
  code steps (needs numpy/pandas/scikit-learn/torch/Pillow etc., matching what tasks expect).

A Slurm cluster is required: the sandbox runs each agent code step as an `srun` job.

Get the data first: see [DATA.md](./DATA.md).

## The 3-terminal runbook

**Terminal 1 — SRUN execution API** (keep running):
```bash
conda activate rllm
source .env
bash examples/deepresearch/scripts/serve_srun_api.sh
# note the printed host, e.g. 10.0.0.5  ->  export SRUN_API_URL="http://10.0.0.5:9000"
```

**Terminal 2 — health check:**
```bash
source .env
export SRUN_API_URL="http://<host-from-terminal-1>:9000"
curl -sSf "$SRUN_API_URL/docs" >/dev/null && echo "API reachable"
```

**Terminal 3 — training or evaluation** (below), with the same `SRUN_API_URL` exported.

## Training (trajectory-wise GRPO)

```bash
conda activate rllm
source .env
export SRUN_API_URL="http://<host>:9000"
# edit the #SBATCH --chdir in the script to your repo path, then:
sbatch examples/deepresearch/train_mle_syn_qwen3_8b.sh      # 8B  (14b / 30b variants available)
```

Reward-design ablation (paper appendix): the `*_no_milestone.sh` scripts set
`SANDMLE_REWARD=sparse` (r = 0.1·format + 0.9·𝟙_gold); the default dense milestone reward is
`SANDMLE_REWARD=dense`. You can toggle it via that env var without editing code.

Combining SFT + RL: first build SFT data from evaluation trajectories with
`prepare_sft_data.py`, run the `*_sft.sh` training, then RL-finetune from that checkpoint.

## Serving a checkpoint for evaluation

```bash
conda activate rllm
source .env
# point the serve script at your checkpoint dir, then:
sbatch examples/deepresearch/vllm_serve_qwen3_8b_finetuned.sh
# this exposes an OpenAI endpoint; set:
export SANDMLE_MODEL_BASE_URL="http://<serving-host>:8001/v1"
export SANDMLE_SERVED_MODEL_NAME="<--served-model-name from the serve script>"
```

## Evaluation (MLE-bench-lite / synthetic)

```bash
conda activate rllm
source .env
export SRUN_API_URL="http://<host>:9000"
export DEEPRESEARCH_API_JOB_NAME="sandmle_eval_qwen3_8b"

# Real MLE-bench-lite (easy split), ReAct scaffold:
python -m examples.deepresearch.custom_evaluate \
    --model qwen3-8b-finetuned --dataset easy \
    --parallel-tasks 16 --max-llm-calls 20 --python-timeout-s 86400 --max-time-s 86400

# Synthetic held-out tasks:
python -m examples.deepresearch.custom_evaluate \
    --model qwen3-8b-finetuned --dataset synthetic --parallel-tasks 16 --max-llm-calls 20
```

- Hosted reference models (Claude / GPT / Gemini / DeepSeek) route through OpenRouter — set
  `OPENROUTER_API_KEY`. Any other `--model` value talks to your vLLM endpoint
  (`SANDMLE_MODEL_BASE_URL` / `SANDMLE_SERVED_MODEL_NAME`).
- Results (per-episode rewards, medal tiers, valid-submission) are written under the run's output
  directory.

## Notes

- AIDE / AIRA / MLE-Agent scaffold generalization (paper §6) is not included in this release.
- Metrics follow the MLE-bench protocol (Valid Submission, Above Median, Bronze/Silver/Gold,
  Any Medal); synthetic tasks are graded by each task's `evaluator.py` via `SynScoreTool`.

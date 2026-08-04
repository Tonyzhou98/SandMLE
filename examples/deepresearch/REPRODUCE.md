# Reproducing SandMLE (RL training + evaluation)

This walks through the runtime setup and the main experiments. All site-specific values come
from environment variables — copy `.env.example` to `.env`, edit it, and `source .env` in every
terminal below.

## Prerequisites

Two conda environments (they do different jobs):

- **`rllm`** — runs the rLLM trainer / evaluator (this repo). Install rLLM per its own docs plus
  `examples/deepresearch/requirements-sandmle.txt`.
- **`sandmle`** (`SANDMLE_CONDA_ENV`) — the env the SRUN API uses to execute the *agent's* generated
  code steps. **This env must contain the ML libraries the tasks need**, otherwise agent code fails
  at runtime and every rollout scores ~0. Our runs used a Python 3.12 env with:

  ```
  # core
  numpy==1.26.4  scipy==1.14.1  pandas==2.2.2  polars==1.38.1  pyarrow==17.0.0  numba==0.60.0
  # classic ML
  scikit-learn==1.5.1  scikit-image==0.24.0  xgboost==2.1.1  lightgbm==4.5.0  catboost==1.2.5
  imbalanced-learn==0.12.3  category-encoders  optuna==4.0.0  statsmodels
  # deep learning
  torch==2.2.0  torchvision==0.17.0  torchaudio==2.2.0  torchtext==0.16.2
  tensorflow==2.17.0  keras==3.5.0  timm==0.9.7  accelerate==0.33.0  einops
  # NLP
  transformers==4.44.2  sentence-transformers==3.0.1  datasets==2.1.0  nltk==3.9.1  spacy==3.7.6
  # vision / audio
  pillow==10.4.0  opencv-python==4.10.0.84  albumentations==2.0.8
  librosa==0.10.2.post1  soundfile==0.13.1  audioread==3.1.0
  # graph / viz / utils
  networkx==3.3  matplotlib==3.9.2  seaborn==0.13.2  tqdm==4.66.5
  ```

  (These are the exact libraries/versions we used for training rollouts and evaluation; adjust to
  match the modalities in your task set.)

A Slurm cluster is required: the sandbox runs each agent code step as an `srun` job.

> **Busy-cluster tip.** The SRUN API launches each code step with plain `srun` (default QOS). If
> your cluster is full, those steps queue and rollouts stall. Export a QOS that can allocate in the
> terminal that runs the SRUN API — `srun` inherits it, e.g. `export SLURM_QOS=<your_high_qos>` —
> so per-step jobs get scheduled promptly. (For the MLE-Dojo scaffolds, `SLURM_QOS` is already a
> documented knob in `scaffolds/README.md`.)

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

## Grading an evaluation run (paper tables)

`custom_evaluate.py --dataset easy` writes per-competition `submission.csv` files under its run
directory. To produce the aggregate MLE-bench grade report (valid submissions, above-median, and
bronze/silver/gold counts — the numbers in the main-results tables):

```bash
conda activate rllm
source .env
python -m examples.deepresearch.grade_reports \
    --run-dir examples/deepresearch/output/<your-eval-run> \
    --output-dir outputs/grade_report
```

This collects submissions, writes a JSONL manifest, and runs `mlebench grade` against
`SANDMLE_DATA_ROOT`. The resulting JSON contains `total_valid_submissions`, `total_above_median`,
`total_{bronze,silver,gold}_medals`, and per-competition reports. (Synthetic-task grading instead
uses each task's `evaluator.py` via `SynScoreTool` during evaluation.)

## Notes

- AIDE / AIRA / MLE-Agent scaffold generalization (paper §6) is not included in this release.
- Metrics follow the MLE-bench protocol (Valid Submission, Above Median, Bronze/Silver/Gold,
  Any Medal); synthetic tasks are graded by each task's `evaluator.py` via `SynScoreTool`.

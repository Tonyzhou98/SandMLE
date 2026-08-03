# SandMLE — RL Training + Evaluation for MLE Agents

This directory contains the reinforcement-learning (trajectory-wise GRPO) training and
evaluation code for **SandMLE**, built as an example on top of the [rLLM](https://github.com/agentica-project/rllm)
framework. The ReAct agent and tool interfaces are adapted from
[Tongyi DeepResearch](https://github.com/Alibaba-NLP/DeepResearch) (hence the `deepresearch_*`
filenames, kept for continuity).

## What's here

| File | Role |
| --- | --- |
| `deepresearch_agent.py` | Multi-turn ReAct rollout loop (`<think>`/`<tool_call>`/`<answer>`), context management. |
| `deepresearch_workflow.py` | rLLM workflow wrapper + **dense milestone reward** (`format + execute + median/bronze/silver/gold`) and the **sparse** ablation reward. |
| `deepresearch_tools.py` | Tools: `PythonInterpreter` (runs code via the SRUN API), `Score` (MLE-bench grader), `SynScore` (synthetic `evaluator.py`). |
| `srun_api.py` | FastAPI service that executes each agent code step as a Slurm `srun` job (enforces per-step time limits). |
| `custom_train.py`, `custom_train_megatron.py` | GRPO trainer entry points (FSDP / Megatron). |
| `custom_evaluate.py` | Evaluation harness (MLE-bench-lite `easy`/`hard`, synthetic). |
| `prepare_train_data.py`, `prepare_sft_data.py` | Build GRPO train/val parquet, and SFT data from trajectories. |
| `train_mle_*.sh`, `vllm_serve_*.sh` | Slurm launch configs per model size / ablation, and vLLM serving. |
| `sandmle_config.py` | Central env-driven configuration (no hardcoded paths). |
| `scripts/serve_srun_api.sh` | Launcher for the SRUN execution API. |

## Configuration

All site-specific values (data roots, checkpoints, model paths, hosts, conda env, W&B project)
are read from environment variables. Copy the template and edit it:

```bash
cp ../../.env.example ../../.env
$EDITOR ../../.env
source ../../.env
```

See `sandmle_config.py` for the full list of `SANDMLE_*` variables and their defaults.

## Quickstart

1. **Data** — download the synthetic tasks and (optionally) MLE-bench: see [DATA.md](./DATA.md).
2. **Reproduce** — the 3-terminal runbook (SRUN API → vLLM serve → train/eval) and the exact
   commands for each experiment are in [REPRODUCE.md](./REPRODUCE.md).

## Reward ablation

Set `SANDMLE_REWARD=dense` (default, milestone-based) or `SANDMLE_REWARD=sparse`
(`r = 0.1·format + 0.9·𝟙_gold`). The `*_no_milestone.sh` scripts set the sparse mode.

## Not included

AIDE / AIRA / MLE-Agent scaffold generalization (paper §6) and trained checkpoints are not part
of this release.

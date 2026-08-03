# SandMLE

Public code release for the paper **SandMLE**. SandMLE makes trajectory-wise, on-policy
reinforcement learning feasible for machine-learning-engineering (MLE) agents by training and
evaluating them on diverse, verifiable, *micro-scale* synthetic MLE environments.

This repository releases the **RL training and evaluation** half of SandMLE. It is a
self-contained copy of the [rLLM](https://github.com/agentica-project/rllm) framework with the
SandMLE code under [`examples/deepresearch/`](examples/deepresearch/). (The ReAct agent is
adapted from [Tongyi DeepResearch](https://github.com/Alibaba-NLP/DeepResearch); see `NOTICE`.)

## Start here

- **SandMLE code + overview:** [`examples/deepresearch/README.md`](examples/deepresearch/README.md)
- **Data setup (synthetic tasks + MLE-bench):** [`examples/deepresearch/DATA.md`](examples/deepresearch/DATA.md)
- **Reproduction runbook (train + eval):** [`examples/deepresearch/REPRODUCE.md`](examples/deepresearch/REPRODUCE.md)

## Configuration

All paths, hosts, and credentials are supplied via environment variables — there are no
hardcoded machine- or user-specific values. Copy the template and edit it:

```bash
cp .env.example .env
$EDITOR .env
source .env
```

## What's included / not included

- **Included:** synthetic + MLE-bench evaluation, trajectory-wise GRPO training, the dense
  milestone reward and its sparse ablation, ReAct rollout, and the Slurm-backed execution sandbox.
- **Not included:** the synthetic *environment generation* pipeline, trained checkpoints, and the
  AIDE / AIRA / MLE-Agent scaffold-generalization experiments (paper §6).

## Framework

The `rllm/` tree and other `examples/` are the upstream rLLM framework, redistributed here so the
release runs without an external checkout. Their original licenses apply (see `LICENSE`, `NOTICE`).

# Scaffold Generalization (AIRA / AIDE / MLE-Dojo)

These are **overlays** for the paper's §6 scaffold-generalization experiments — evaluating SandMLE
policies under agent scaffolds other than the training ReAct loop. They are *not* full frameworks:
each folder mirrors the relative paths of an upstream benchmark repo and contains only the
SandMLE-specific config/experiment/run files you drop into a fresh clone.

In every case: **serve your policy first** with vLLM (see
`../examples/deepresearch/vllm_serve_*.sh`), then point the scaffold's `base_url` at that endpoint.
All hosts here are set to `http://localhost:8001/v1` — change them to your serving host. API keys
are read from environment variables (`OPENROUTER_API_KEY` / `OPENAI_API_KEY`); never hardcode them.

---

## AIRA / AIDE — `aira-dojo/`

Upstream: https://github.com/facebookresearch/aira-dojo

```bash
git clone https://github.com/facebookresearch/aira-dojo.git
# install per aira-dojo's README (conda env: aira-dojo), then overlay the SandMLE files:
cp -r scaffolds/aira-dojo/src   aira-dojo/
cp -r scaffolds/aira-dojo/notebooks aira-dojo/

cd aira-dojo
conda activate aira-dojo
# edit src/dojo/configs/solver/client/litellm_qwen3_*.yaml -> set base_url to your vLLM endpoint

# AIDE and AIRA greedy runs (examples):
python -m dojo.main_runner_job_array +_exp=mlebench/aide_greedy_qwen3_30b           logger.use_wandb=False launcher.debug=False
python -m dojo.main_runner_job_array +_exp=mlebench/aira_greedy_qwen3_30b_finetuned logger.use_wandb=False launcher.debug=False

# aggregate results:
python notebooks/analyze_results.py
```

Provided experiment configs (`src/dojo/configs/_exp/mlebench/`): `{aide,aira}_greedy_qwen3_{8b,14b,30b,30b_sft,30b_finetuned}.yaml`,
each wired to a matching `solver/client/litellm_qwen3_*.yaml` client config.

---

## MLE-Dojo — `mle-dojo/`

Upstream: https://github.com/MLE-Dojo/MLE-Dojo

```bash
git clone https://github.com/MLE-Dojo/MLE-Dojo.git
# install per MLE-Dojo's README (conda env: mle-dojo), then overlay the SandMLE files:
cp -r scaffolds/mle-dojo/mledojo MLE-Dojo/
cp scaffolds/mle-dojo/run_dojo_competitions_*.sh MLE-Dojo/

cd MLE-Dojo
conda activate mle-dojo
# edit mledojo/agent/mleagent/config.yaml and mledojo/agent/aide/utils/config.yaml:
#   set base_url to your vLLM endpoint; replace the api_key placeholder
#   "YOUR_OPENROUTER_API_KEY" (or leave it and export OPENAI_API_KEY / OPENROUTER_API_KEY)
export MLE_DOJO_DATA_DIR=/path/to/mle-dojo-data
export SLURM_QOS=<your-slurm-qos>            # optional, defaults to "normal"

./run_dojo_competitions_parallel.sh   # MLE-Agent scaffold
./run_dojo_competitions_aide.sh       # AIDE scaffold
```

`run_dojo_competitions_*.sh` read `COMP_FILE` (default `prepare/dojo_competitions.txt`),
`MLE_DOJO_DATA_DIR`, and `SLURM_QOS` from the environment.

---

## Notes

- Data (MLE-bench-lite for aira-dojo, MLE-Dojo competitions for MLE-Dojo) is downloaded per each
  upstream repo's instructions; it is not included here.
- `model_name` values (e.g. `qwen3_30b_serve_finetuned_step_60`) are just vLLM `--served-model-name`
  labels — match them to what you launch in `vllm_serve_*.sh`.

"""
Central configuration for the SandMLE RL training + evaluation example.

All environment- and site-specific values are read from environment variables so the
published code contains no absolute paths, hostnames, or credentials. Copy `.env.example`
at the repo root, fill in your values, and `source` it (or export the variables) before
running training / evaluation.
"""

import os
from pathlib import Path


def _env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser()


# --- Repo root (used only for sensible relative defaults) -------------------------------
SANDMLE_ROOT = _env_path("SANDMLE_ROOT", str(Path(__file__).resolve().parents[2]))

# --- Datasets ---------------------------------------------------------------------------
# Real MLE-bench data (download via `mlebench`; see docs/DATA.md).
DATA_ROOT = _env_path("SANDMLE_DATA_ROOT", str(SANDMLE_ROOT / "data" / "mle-bench-data"))
# Synthetic SandMLE tasks (download from Google Drive; see docs/DATA.md).
SYN_DATA_ROOT = _env_path("SANDMLE_SYN_DATA_ROOT", str(SANDMLE_ROOT / "data" / "mle-bench-syn"))
# Registered train/val parquet datasets produced by prepare_train_data.py.
DATASETS_DIR = _env_path("SANDMLE_DATASETS_DIR", str(SANDMLE_ROOT / "data" / "datasets"))
# SFT parquet output produced by prepare_sft_data.py.
SFT_OUT_DIR = _env_path("SANDMLE_SFT_OUT_DIR", str(DATASETS_DIR / "mle_bench_sft"))

# Lists of synthetic tasks that passed sanity checks (shipped with the synthetic task bundle).
OK_LIST = _env_path("SANDMLE_OK_LIST", str(SYN_DATA_ROOT / "ok_competitions_sanity.json"))
OK_LIST_SAMPLED = _env_path(
    "SANDMLE_OK_LIST_SAMPLED", str(SYN_DATA_ROOT / "ok_competitions_sanity_random_64.json")
)

# --- Execution sandbox (SRUN API) -------------------------------------------------------
# Conda env in which the SRUN API runs the agent's generated code steps.
EXEC_CONDA_ENV = os.environ.get("SANDMLE_CONDA_ENV", "sandmle")

# --- Model serving ----------------------------------------------------------------------
# Base URL of the vLLM OpenAI-compatible server that serves the (base/finetuned) policy.
MODEL_BASE_URL = os.environ.get("SANDMLE_MODEL_BASE_URL", "http://127.0.0.1:8001/v1")
# The `--served-model-name` used when launching vLLM (see vllm_serve_*.sh).
SERVED_MODEL_NAME = os.environ.get("SANDMLE_SERVED_MODEL_NAME", "sandmle_policy")
# Router for hosted/proprietary reference models (Claude/GPT/Gemini/DeepSeek).
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

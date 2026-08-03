"""Batch-grade an evaluation run into an MLE-bench grade report.

Given an evaluation output directory produced by `custom_evaluate.py` (one subfolder per
competition, each containing a run folder with `submission.csv`), this collects the submissions,
writes a JSONL manifest, and runs `mlebench grade` to produce the aggregate report
(valid submissions, above-median, and bronze/silver/gold medal counts) used for the paper tables.

Usage:
    python -m examples.deepresearch.grade_reports \
        --run-dir examples/deepresearch/output/qwen3-8b-finetuned-<timestamp> \
        --output-dir outputs/grade_report
"""

import argparse
import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union

from . import sandmle_config as cfg
from .custom_evaluate import competition_id_list

logger = logging.getLogger(__name__)


def save_to_jsonl(obj: Union[Dict[str, Any], Iterable[Dict[str, Any]]], path: Union[str, Path]) -> Path:
    """Save one or more submission dicts (each with 'submission_path' and 'competition_id') to JSONL."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = [obj] if isinstance(obj, dict) else list(obj)

    required = {"submission_path", "competition_id"}
    for i, rec in enumerate(records):
        if not isinstance(rec, dict):
            raise TypeError(f"Record at index {i} is not a dict: {type(rec)}")
        if not required.issubset(rec.keys()):
            raise ValueError(f"Record at index {i} missing required keys {required - set(rec.keys())}")

    with p.open("w", encoding="utf-8") as f:
        for rec in records:
            rec_copy = dict(rec)
            if isinstance(rec_copy.get("submission_path"), Path):
                rec_copy["submission_path"] = str(rec_copy["submission_path"])
            f.write(json.dumps(rec_copy, ensure_ascii=False) + "\n")

    logger.info("Wrote %d record(s) to %s", len(records), p)
    return p


def grade_submission(jsonl_path: Union[str, Path], output_dir: Union[str, Path], data_dir: Union[str, Path], timeout: int = 600):
    """Run `mlebench grade --submission <jsonl> --output-dir <dir> --data-dir <dir>`."""
    jsonl_path = Path(jsonl_path)
    output_dir = Path(output_dir)
    data_dir = Path(data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "mlebench", "grade",
        "--submission", str(jsonl_path),
        "--output-dir", str(output_dir),
        "--data-dir", str(data_dir),
    ]

    logger.info("Running command: %s", " ".join(cmd))
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        result = {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}
        if proc.returncode != 0:
            logger.warning("mlebench grade failed (code %d). stderr: %s", proc.returncode, proc.stderr.strip())
        else:
            logger.info("mlebench grade completed successfully.")
        return result
    except subprocess.TimeoutExpired as e:
        logger.error("mlebench grade timed out after %ds", timeout)
        return {"returncode": -1, "stdout": "", "stderr": f"Timeout after {timeout}s: {e}"}
    except Exception as e:
        logger.exception("Unexpected error running mlebench grade: %s", e)
        return {"returncode": -1, "stdout": "", "stderr": str(e)}


def _find_submission_path(result_root: Path, competition_id: str) -> Path:
    comp_dir = result_root / competition_id
    if not comp_dir.exists():
        raise FileNotFoundError(f"Missing competition folder: {comp_dir}")
    subdirs = [p for p in comp_dir.iterdir() if p.is_dir()]
    if len(subdirs) != 1:
        raise ValueError(f"Expected exactly one subfolder under {comp_dir}, found {len(subdirs)}")
    return subdirs[0] / "submission.csv"


def build_submissions(result_root: Union[str, Path], competition_ids: List[str]) -> List[Dict[str, str]]:
    """Collect (submission_path, competition_id) records for each competition in an eval run dir."""
    result_root = Path(result_root)
    return [
        {
            "submission_path": str(_find_submission_path(result_root, competition_id)),
            "competition_id": competition_id,
        }
        for competition_id in competition_ids
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grade an evaluation run into an MLE-bench report.")
    parser.add_argument("--run-dir", required=True, help="Evaluation output dir (one subfolder per competition).")
    parser.add_argument("--output-dir", default="outputs/grade_report", help="Where to write the grade report.")
    parser.add_argument("--data-dir", default=str(cfg.DATA_ROOT), help="MLE-bench data dir (defaults to SANDMLE_DATA_ROOT).")
    parser.add_argument("--jsonl-path", default="submission.jsonl", help="Path for the intermediate JSONL manifest.")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout for `mlebench grade` in seconds.")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    submissions = build_submissions(args.run_dir, competition_id_list)
    jsonl_p = save_to_jsonl(submissions, args.jsonl_path)
    res = grade_submission(jsonl_p, args.output_dir, args.data_dir, timeout=args.timeout)
    print(res)

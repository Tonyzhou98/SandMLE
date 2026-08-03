from dojo.utils.environment import get_log_dir
from dojo.analysis_utils.meta_data_wrangling import (
    collect_all_meta_experiments_in_one_df,
    format_experiment_data,
    filter_dataframe_based_on_data_validity,
    add_node_elapsed_from_first
)
import os
import pandas as pd
from collections.abc import Iterable
import json

from pathlib import Path

user = os.getenv("USER")


def extract_best_node_content(row):
    best_idx = row['best_node_idx']
    if pd.isna(best_idx):
        return row
    return row.apply(
        lambda x: x[int(best_idx)] if isinstance(x, Iterable) and not isinstance(x, str) and int(best_idx) < len(x) else x
    )

# Dictonary with paths to meta-experiments and their corresponding methods (for methods, you can choose any name you like)
log_dir = Path(get_log_dir())
methods = {
    str((log_dir / "aira-dojo" / f"user_{user}_issue_AIDE_GREEDY_qwen3_30b_sft").resolve()): "AIDE_GREEDY_QWEN3_30B_SFT",
    str((log_dir / "aira-dojo" / f"user_{user}_issue_AIDE_GREEDY_qwen3_30b_finetuned_step_60").resolve()): "AIDE_GREEDY_QWEN3_30B_FINETUNED",
    str((log_dir / "aira-dojo" / f"user_{user}_issue_AIRA_GREEDY_qwen3_30b_sft").resolve()): "AIRA_GREEDY_QWEN3_30B_SFT",
    str((log_dir / "aira-dojo" / f"user_{user}_issue_AIRA_GREEDY_qwen3_30b_finetuned_step_60").resolve()): "AIRA_GREEDY_QWEN3_30B_FINETUNED",
}

print(methods)

# Count existing submissions per method based on grading_report.json
submission_counts = {}
grading_counts = {}
for meta_path, method_name in methods.items():
    meta_dir = Path(meta_path)
    if not meta_dir.is_dir():
        submission_counts[method_name] = 0
        grading_counts[method_name] = {
            "submission_exists": 0,
            "valid_submission": 0,
            "any_medal": 0,
            "gold_medal": 0,
            "silver_medal": 0,
            "bronze_medal": 0,
            "above_median": 0,
        }
        continue

    count = 0
    counts = {
        "submission_exists": 0,
        "valid_submission": 0,
        "any_medal": 0,
        "gold_medal": 0,
        "silver_medal": 0,
        "bronze_medal": 0,
        "above_median": 0,
    }
    for subdir in meta_dir.iterdir():
        if not subdir.is_dir():
            continue
        report_path = subdir / "results" / "grading_report.json"
        if report_path.is_file():
            count += 1
            try:
                with report_path.open("r", encoding="utf-8") as f:
                    report = json.load(f)
            except Exception:
                report = {}

            if report.get("submission_exists") is True:
                counts["submission_exists"] += 1
            if report.get("valid_submission") is True:
                counts["valid_submission"] += 1
            if report.get("any_medal") is True:
                counts["any_medal"] += 1
            if report.get("gold_medal") is True:
                counts["gold_medal"] += 1
            if report.get("silver_medal") is True:
                counts["silver_medal"] += 1
            if report.get("bronze_medal") is True:
                counts["bronze_medal"] += 1
            if report.get("above_median") is True:
                counts["above_median"] += 1
    submission_counts[method_name] = count
    grading_counts[method_name] = counts

print("\nSubmission counts by method:")
for method_name, count in submission_counts.items():
    print(f"{method_name}: {count}")

print("\nGrading report counts by method:")
for method_name, counts in grading_counts.items():
    print(
        f"{method_name}: "
        f"submission_exists={counts['submission_exists']}, "
        f"valid_submission={counts['valid_submission']}, "
        f"any_medal={counts['any_medal']}, "
        f"gold={counts['gold_medal']}, "
        f"silver={counts['silver_medal']}, "
        f"bronze={counts['bronze_medal']}, "
        f"above_median={counts['above_median']}"
    )

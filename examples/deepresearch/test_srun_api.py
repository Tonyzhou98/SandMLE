"""Manual smoke test for the SRUN execution API.

Usage:
    export SRUN_API_URL="http://<host>:9000"     # where srun_api.py is serving
    python test_srun_api.py /path/to/script.py

Submits a script to the SRUN API and streams its status/logs until completion.
"""

import os
import sys
import time

import requests

API = os.environ.get("SRUN_API_URL", "http://127.0.0.1:9000")
CONDA_ENV = os.environ.get("SANDMLE_CONDA_ENV", "sandmle")

if len(sys.argv) < 2:
    raise SystemExit("Usage: python test_srun_api.py /path/to/script.py")
SCRIPT_PATH = sys.argv[1]

# -------- submit job --------
print("Submitting job...")
resp = requests.post(
    f"{API}/run",
    json={
        "script_path": SCRIPT_PATH,
        "time": "00:10:00",
        "cpus": 8,
        "mem": "32G",
        "gres_gpus": "gpu:1",
        "conda_env": CONDA_ENV,
    },
    timeout=30,
)
resp.raise_for_status()
job = resp.json()

job_id = job["job_id"]
stdout_path = job["stdout"]
stderr_path = job["stderr"]

print(f"Job submitted: {job_id}")
print(f"stdout: {stdout_path}")
print(f"stderr: {stderr_path}")
print("-" * 60)

# -------- poll until finished --------
while True:
    s = requests.get(f"{API}/status/{job_id}", timeout=30)
    s.raise_for_status()
    status = s.json()

    state = status.get("state")
    print(f"[status] {state}")

    # fetch logs tail every poll
    logs = requests.get(
        f"{API}/logs/{job_id}",
        params={"tail_lines": 40},
        timeout=30,
    ).json()

    stdout_tail = logs.get("stdout_tail", "")
    stderr_tail = logs.get("stderr_tail", "")

    if stdout_tail:
        print("\n--- stdout (tail) ---")
        print(stdout_tail)

    if stderr_tail:
        print("\n--- stderr (tail) ---")
        print(stderr_tail)

    if state not in ("RUNNING", "PENDING"):
        print("\nJob finished.")
        break

    print("-" * 60)
    time.sleep(5)

print("\nFinal log locations:")
print("stdout:", stdout_path)
print("stderr:", stderr_path)

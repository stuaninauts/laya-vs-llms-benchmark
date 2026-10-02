"""Downloads the experiment's models resumably: if the connection drops, it waits and continues.

huggingface_hub keeps partial files (.incomplete) in the cache and resumes where it stopped;
this script just retries with increasing waits. It can be interrupted and rerun at any
time: whatever already finished is not downloaded again.

Usage: .venv/bin/python download_models.py
"""
import os, sys, time

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
from huggingface_hub import snapshot_download

MODELS = [
    # only Laya's English checkpoint (repo root); multilingual and typed-decisions are left out
    ("convaiinnovations/laya", {"ignore_patterns": ["multilingual/*", "typed-decisions/*", "eval/*", "assets/*"]}),
    ("ibm-granite/granite-embedding-small-english-r2", {}),
    ("Qwen/Qwen3-Embedding-0.6B", {}),
    ("answerdotai/ModernBERT-large", {}),
]
MAX_WAIT = 300


def fetch(repo, kw):
    wait, attempt = 10, 0
    while True:
        attempt += 1
        try:
            path = snapshot_download(repo, max_workers=4, **kw)
            print(f"OK {repo} -> {path}", flush=True)
            return
        except KeyboardInterrupt:
            raise
        except Exception as e:  # network drop, timeout, 5xx: wait and retry
            print(f"RETRY {repo} attempt {attempt}: {type(e).__name__}: {str(e)[:160]} "
                  f"(retrying in {wait}s)", flush=True)
            time.sleep(wait)
            wait = min(wait * 2, MAX_WAIT)


for repo, kw in MODELS:
    fetch(repo, kw)
print("DOWNLOAD COMPLETE", flush=True)

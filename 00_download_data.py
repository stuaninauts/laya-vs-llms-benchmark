"""Download the CFPB complaints mirror and keep 2023+ complaints that have a narrative.

The official CFPB bulk download no longer ships the consumer narrative column (observed Sep/2026),
so we use the Hugging Face mirror `sovai/cfpb_complaints` (~4M rows, narratives up to May/2024).
Raw data is not redistributed in this repo — run this script to rebuild it.

Usage: python 00_download_data.py   ->  data/cfpb_2023plus.parquet
"""
import os
import pandas as pd
from huggingface_hub import hf_hub_download

RAW = "data/cfpb_sovai.parquet"
OUT = "data/cfpb_2023plus.parquet"
COLS = ["date", "product", "sub_product", "issue", "sub_issue", "consumer_complaint_narrative",
        "company", "company_response_to_consumer"]

os.makedirs("data", exist_ok=True)
if not os.path.exists(RAW):
    path = hf_hub_download("sovai/cfpb_complaints", "cfpb_complaints.parquet", repo_type="dataset")
    os.symlink(path, RAW)

df = pd.read_parquet(RAW, columns=COLS)
df = df[df.consumer_complaint_narrative.notna() & (df.consumer_complaint_narrative.str.len() > 50)]
df = df[df.date >= "2023-01-01"]
df.to_parquet(OUT)
print(f"{OUT}: {len(df):,} complaints with narrative, {df.date.min():%Y-%m-%d} to {df.date.max():%Y-%m-%d}")

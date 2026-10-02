"""Builds train/test from the CFPB data (2023+): template-letter dedup, product queues, fraud label.

Test: 200 per queue (balanced) with fraud oversampled (up to 30% per queue) to get enough positives.
Train (Laya fine-tuning): disjoint, 400 per queue, same strategy.
"""
import re
import pandas as pd

from common import map_product, fraud_label, PRODUCTS

SEED = 20260929
N_TEST, N_TRAIN, FRAUD_SHARE = 200, 400, 0.30

df = pd.read_parquet("data/cfpb_2023plus.parquet")
df["queue"] = [map_product(p, s) for p, s in zip(df["product"], df["sub_product"])]
df["fraud"] = [fraud_label(i, s) for i, s in zip(df["issue"], df["sub_issue"])]
df = df.dropna(subset=["queue", "fraud"])
df["fraud"] = df["fraud"].astype(bool)

# template letters: drop exact and near-exact duplicates (same normalized opening)
text = df["consumer_complaint_narrative"]
key = text.str.lower().map(lambda t: re.sub(r"[^a-z]+", " ", t)[:300])
df = df[~key.duplicated(keep=False) & (text.str.len() >= 100)]
df = df.rename(columns={"consumer_complaint_narrative": "text"})
df["id"] = range(len(df))
print("pool after cleaning:", len(df))
print(pd.crosstab(df.queue, df.fraud))


def sample(pool, n):
    parts = []
    for q in PRODUCTS:
        g = pool[pool.queue == q]
        pos = g[g.fraud].sample(min(int(n * FRAUD_SHARE), g.fraud.sum()), random_state=SEED)
        neg = g[~g.fraud].sample(n - len(pos), random_state=SEED)
        parts += [pos, neg]
    return pd.concat(parts).sample(frac=1, random_state=SEED)


cols = ["id", "date", "company", "product", "issue", "sub_issue", "queue", "fraud", "text"]
test = sample(df, N_TEST)[cols]
train = sample(df[~df.id.isin(test.id)], N_TRAIN)[cols]
test.to_parquet("data/test.parquet")
train.to_parquet("data/train.parquet")
print("test:", len(test), "fraud:", test.fraud.mean().round(3), "| train:", len(train), "fraud:", train.fraud.mean().round(3))

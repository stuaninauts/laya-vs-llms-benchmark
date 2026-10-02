"""Fixed subsets of the test set for the more expensive LLMs (no cost, deterministic).

- test_s900:   100 per queue (half of the test set), same per-queue fraud rate as the full test set.
               Every system is also evaluated on this subset → fair paired comparison.
- test_lat200: 200 complaints (≈22 per queue) contained in s900, only to measure synchronous latency;
               accuracy of the expensive models comes from the Batch API (50% cheaper, no usable latency).
"""
import pandas as pd

SEED = 20260930
test = pd.read_parquet("data/test.parquet")

s900 = (test.groupby(["queue", "fraud"], group_keys=False)
        .apply(lambda g: g.sample(frac=0.5, random_state=SEED)))
# adjust to exactly 100 per queue (rounding of the fraud strata)
s900 = s900.groupby("queue", group_keys=False).apply(lambda g: g.head(100) if len(g) >= 100 else g)
missing = {q: 100 - n for q, n in s900.queue.value_counts().items() if n < 100}
for q, k in missing.items():
    extra = test[(test.queue == q) & ~test.id.isin(s900.id)].sample(k, random_state=SEED)
    s900 = pd.concat([s900, extra])
s900 = s900.sample(frac=1, random_state=SEED)

lat200 = s900.groupby("queue", group_keys=False).apply(lambda g: g.sample(22, random_state=SEED)).head(200)

s900.to_parquet("data/test_s900.parquet")
lat200.to_parquet("data/test_lat200.parquet")
print(f"s900: {len(s900)} | fraud {s900.fraud.mean():.1%} (full test {test.fraud.mean():.1%}) | "
      f"per queue {s900.queue.value_counts().min()}–{s900.queue.value_counts().max()}")
print(f"lat200: {len(lat200)} | fraud {lat200.fraud.mean():.1%} | contained in s900: {lat200.id.isin(s900.id).all()}")

"""Consolidates whatever exists in results/ — calls no API at all.

Reads results/llm_*.parquet (GPT via API, already paid) and results/sys_*.parquet (local systems, see sysio.py).
- per-system metrics: queue accuracy and macro F1; fraud precision/recall/F1/AUC/ECE
- latency and cost per 1 million complaints
- cascade: the local system decides when confidence >= t; the rest goes to the API LLM
Writes results/summary.md and results/cascade_<system>.csv.
"""
import glob, json, os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

# Cost assumptions for the local systems (stated explicitly in the summary)
WATTS = {"cuda": 170, "cpu": 65}   # TDP RTX 3060 / typical desktop CPU
ENERGY_USD_PER_KWH = 0.15
CLOUD_T4_USD_PER_HOUR = 0.35       # on-demand T4 GPU, GCP reference
CASCADE_LLMS = ["gpt-5.4-mini", "gpt-5.6-luna__full", "gpt-5.6-terra__s900"]   # fallbacks (terra only on s900)

test = pd.read_parquet("data/test.parquet")[["id", "queue", "fraud"]]


def ece(y, p, bins=10):
    """Expected calibration error of the fraud probability."""
    b = np.clip((p * bins).astype(int), 0, bins - 1)
    return sum(abs(y[b == i].mean() - p[b == i].mean()) * (b == i).mean() for i in range(bins) if (b == i).any())


def metrics(d, product, fraud_p):
    ok = d[fraud_p].notna()
    y, p = d.fraud[ok].values.astype(float), d[fraud_p][ok].values.astype(float)
    return {"product_acc": accuracy_score(d.queue, d[product].fillna("")),
            "product_f1_macro": f1_score(d.queue, d[product].fillna(""), average="macro"),
            "fraud_precision": precision_score(y, p >= .5, zero_division=0),
            "fraud_recall": recall_score(y, p >= .5, zero_division=0),
            "fraud_f1": f1_score(y, p >= .5, zero_division=0),
            "fraud_auc": roc_auc_score(y, p), "fraud_ece": ece(y, p)}


rows, frames = [], {}
for path in sorted(glob.glob("results/llm_*.parquet")):
    name = os.path.basename(path)[4:-8]
    d = test.merge(pd.read_parquet(path), on="id")
    frames[name] = d
    d = d.dropna(subset=["llm_product"])
    rows.append({"system": f"{name} (API)", "n": len(d), **metrics(d, "llm_product", "llm_fraud_p"),
                 "p50_ms": d.llm_ms.median(), "p95_ms": d.llm_ms.quantile(.95),
                 "usd_per_1M": d.cost_usd.mean() * 1e6, "usd_per_1M_cloud_T4": np.nan})

local_cost = {}
for path in sorted(glob.glob("results/sys_*.parquet")):
    name = os.path.basename(path)[4:-8]
    d = test.merge(pd.read_parquet(path), on="id")
    frames[name] = d
    meta = json.load(open(f"results/sys_{name}_meta.json"))
    hours_per_1M = meta["batch_ms_per_complaint"] * 1e6 / 3.6e6
    watts = WATTS["cpu" if meta["device"] == "cpu" else "cuda"]
    local_cost[name] = hours_per_1M * watts / 1000 * ENERGY_USD_PER_KWH
    rows.append({"system": f"{name} ({meta['device']})", "n": len(d), **metrics(d, "product", "fraud_p"),
                 "p50_ms": meta["single_p50_ms"], "p95_ms": meta["single_p95_ms"],
                 "usd_per_1M": local_cost[name],
                 "usd_per_1M_cloud_T4": hours_per_1M * CLOUD_T4_USD_PER_HOUR if meta["device"] != "cpu" else np.nan})

summary = pd.DataFrame(rows).sort_values("product_acc", ascending=False)
lines = ["# Results — CFPB complaint triage", "",
         f"Test: {len(test)} complaints, 9 queues (200 each), fraud {test.fraud.mean():.1%}.", "",
         summary.round(4).to_markdown(index=False), "",
         f"Local cost = energy ({WATTS['cuda']} W GPU / {WATTS['cpu']} W CPU, US$ {ENERGY_USD_PER_KWH}/kWh) "
         f"at batch throughput; T4 cloud reference US$ {CLOUD_T4_USD_PER_HOUR}/h. "
         "p50/p95 = isolated decision (1 complaint per call); NaN = run via the Batch API (no latency). "
         "Systems with a different n are not directly comparable — see the paired table below."]

# paired comparison: every system evaluated on the SAME complaints of the s900 subset
if os.path.exists("data/test_s900.parquet"):
    s900 = set(pd.read_parquet("data/test_s900.parquet").id)
    paired = []
    for name, d in frames.items():
        dd = d[d.id.isin(s900)]
        cols = ("llm_product", "llm_fraud_p") if "llm_product" in dd else ("product", "fraud_p")
        dd = dd.dropna(subset=[cols[0]])
        if len(dd) >= 0.95 * len(s900):
            paired.append({"system": name, "n": len(dd), **metrics(dd, *cols)})
    if paired:
        lines += ["", f"## Paired comparison on the s900 subset ({len(s900)} complaints, the same for every system)", "",
                  pd.DataFrame(paired).sort_values("product_acc", ascending=False).round(4).to_markdown(index=False)]

# Cascades: the local system decides when confidence >= t; the rest goes to a fallback LLM.
# Fallback cost = SYNCHRONOUS price per complaint (a cascade is an online system): uses the model's
# __lat200 file when it exists (e.g. luna ran the full test set via batch, at -50%).
cascade_best = []
for fb in CASCADE_LLMS:
    llm = frames.get(fb)
    if llm is None:
        continue
    fb_name = fb.split("__")[0]
    sync = frames.get(f"{fb_name}__lat200")
    llm_cost = (sync if sync is not None else llm).cost_usd.mean() * 1e6
    llm_acc = accuracy_score(llm.queue, llm.llm_product)
    for name, d in frames.items():
        if name not in local_cost:
            continue
        m = d.merge(llm[["id", "llm_product"]], on="id")
        cas = []
        for t in np.round(np.arange(0.30, 1.00, 0.05), 2):
            keep = (m.product_conf >= t).values
            pred = np.where(keep, m["product"], m.llm_product)
            cas.append({"threshold": t, "local_share": keep.mean(),
                        "product_acc": accuracy_score(m.queue, pred),
                        "local_acc_on_kept": accuracy_score(m.queue[keep], m["product"][keep]) if keep.any() else np.nan,
                        "usd_per_1M": local_cost[name] + (1 - keep.mean()) * llm_cost})
        cas = pd.DataFrame(cas)
        cas.to_csv(f"results/cascade_{name}__{fb_name}.csv", index=False)
        b = cas.loc[cas.product_acc.idxmax()]
        cascade_best.append({"local": name, "fallback": fb_name, "best_threshold": b.threshold,
                             "local_share": b.local_share, "product_acc": b.product_acc,
                             "usd_per_1M": b.usd_per_1M, "fallback_alone_acc": llm_acc,
                             "fallback_alone_usd_per_1M": llm_cost})
        lines += ["", f"## Cascade {name} → {fb_name} (n={len(m)})", "", cas.round(4).to_markdown(index=False)]

if cascade_best:
    cb = pd.DataFrame(cascade_best).sort_values("product_acc", ascending=False)
    cb.to_csv("results/cascade_best.csv", index=False)
    lines += ["", "## Summary: best point of each cascade (threshold with the highest final accuracy)", "",
              cb.round(4).to_markdown(index=False)]

open("results/summary.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

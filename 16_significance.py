"""Paired significance tests for the main comparisons (no API calls).

Every system answered the same complaints, so all tests are paired:
- routing accuracy: exact McNemar test (binomial on discordant pairs) + paired bootstrap 95% CI of the
  accuracy difference; Holm correction across the accuracy comparisons;
- fraud AUC and calibration error (ECE): paired bootstrap 95% CI of the difference.
The comparison list is fixed below, before looking at the results.

Writes results/significance.md and results/significance_*.csv.
"""
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score

SEED, N_BOOT, N_BOOT_AUC, THRESHOLD = 20261002, 10_000, 2_000, 0.7
rng = np.random.default_rng(SEED)
test = pd.read_parquet("data/test.parquet")[["id", "queue", "fraud"]]
s900 = set(pd.read_parquet("data/test_s900.parquet").id)


def local(name):
    d = test.merge(pd.read_parquet(f"results/sys_{name}.parquet"), on="id")
    return d.assign(pred=d["product"].fillna(""), p=d.fraud_p)


def api(fname):
    d = test.merge(pd.read_parquet(f"results/llm_{fname}.parquet"), on="id")
    return d.assign(pred=d.llm_product.fillna(""), p=d.llm_fraud_p)


def cascade(local_name, fallback):
    loc, fb = local(local_name), fallback[["id", "pred"]].rename(columns={"pred": "fb"})
    d = loc.merge(fb, on="id")
    return d.assign(pred=np.where(d.product_conf >= THRESHOLD, d.pred, d.fb))


S = {
    "TF-IDF + LR": local("tfidf-logreg"),
    "ModernBERT fine-tuned": local("modernbert-ft"),
    "Laya fine-tuned": local("laya-finetuned"),
    "gpt-5.4-mini": api("gpt-5.4-mini"),
    "gpt-5.6-luna": api("gpt-5.6-luna__full"),
    "gpt-5.6-terra": api("gpt-5.6-terra__s900"),
}
S["Cascade ModernBERT → luna"] = cascade("modernbert-ft", S["gpt-5.6-luna"])
S["Cascade ModernBERT → terra"] = cascade("modernbert-ft", S["gpt-5.6-terra"])

# (A, B, subset) — fixed before looking at the results
ACC_TESTS = [
    ("Cascade ModernBERT → luna", "gpt-5.6-luna", "full"),
    ("Cascade ModernBERT → luna", "ModernBERT fine-tuned", "full"),
    ("gpt-5.6-luna", "ModernBERT fine-tuned", "full"),
    ("gpt-5.6-luna", "gpt-5.4-mini", "full"),
    ("ModernBERT fine-tuned", "Laya fine-tuned", "full"),
    ("ModernBERT fine-tuned", "TF-IDF + LR", "full"),
    ("TF-IDF + LR", "gpt-5.4-mini", "full"),
    ("gpt-5.6-terra", "gpt-5.6-luna", "s900"),
    ("gpt-5.6-terra", "TF-IDF + LR", "s900"),
    ("Cascade ModernBERT → terra", "Cascade ModernBERT → luna", "s900"),
]
FRAUD_TESTS = [
    ("ModernBERT fine-tuned", "gpt-5.4-mini"),
    ("ModernBERT fine-tuned", "Laya fine-tuned"),
    ("Laya fine-tuned", "gpt-5.4-mini"),
    ("TF-IDF + LR", "gpt-5.4-mini"),
    ("ModernBERT fine-tuned", "gpt-5.6-luna"),
]


def paired(a, b, subset):
    m = S[a][["id", "queue", "pred"]].merge(S[b][["id", "pred"]], on="id", suffixes=("_a", "_b"))
    if subset == "s900":
        m = m[m.id.isin(s900)]
    return m


def holm(p):
    order = np.argsort(p)
    adj, running = np.empty(len(p)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


rows = []
for a, b, subset in ACC_TESTS:
    m = paired(a, b, subset)
    ca, cb = (m.pred_a == m.queue).values, (m.pred_b == m.queue).values
    only_a, only_b = int((ca & ~cb).sum()), int((~ca & cb).sum())
    p = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    idx = rng.integers(0, len(m), (N_BOOT, len(m)))
    diffs = ca[idx].mean(1) - cb[idx].mean(1)
    rows.append({"A": a, "B": b, "subset": subset, "n": len(m), "acc_A": ca.mean(), "acc_B": cb.mean(),
                 "diff_pp": 100 * (ca.mean() - cb.mean()),
                 "ci95_low_pp": 100 * np.percentile(diffs, 2.5), "ci95_high_pp": 100 * np.percentile(diffs, 97.5),
                 "only_A_right": only_a, "only_B_right": only_b, "p_mcnemar": p})
acc = pd.DataFrame(rows)
acc["p_holm"] = holm(acc.p_mcnemar.values)
acc["significant_5pct"] = acc.p_holm < 0.05


def ece(y, p, bins=10):
    b = np.clip((p * bins).astype(int), 0, bins - 1)
    return sum(abs(y[b == i].mean() - p[b == i].mean()) * (b == i).mean() for i in range(bins) if (b == i).any())


rows = []
for a, b in FRAUD_TESTS:
    m = S[a][["id", "fraud", "p"]].merge(S[b][["id", "p"]], on="id", suffixes=("_a", "_b")).dropna()
    y, pa, pb = m.fraud.values.astype(float), m.p_a.values, m.p_b.values
    d_auc, d_ece = [], []
    for _ in range(N_BOOT_AUC):
        i = rng.integers(0, len(m), len(m))
        if 0 < y[i].sum() < len(i):
            d_auc.append(roc_auc_score(y[i], pa[i]) - roc_auc_score(y[i], pb[i]))
            d_ece.append(ece(y[i], pa[i]) - ece(y[i], pb[i]))
    rows.append({"A": a, "B": b, "n": len(m),
                 "auc_A": roc_auc_score(y, pa), "auc_B": roc_auc_score(y, pb),
                 "auc_diff": roc_auc_score(y, pa) - roc_auc_score(y, pb),
                 "auc_ci95": (np.percentile(d_auc, 2.5), np.percentile(d_auc, 97.5)),
                 "ece_A": ece(y, pa), "ece_B": ece(y, pb), "ece_diff": ece(y, pa) - ece(y, pb),
                 "ece_ci95": (np.percentile(d_ece, 2.5), np.percentile(d_ece, 97.5))})
fraud = pd.DataFrame(rows)
fraud["auc_ci_excludes_0"] = fraud.auc_ci95.map(lambda c: c[0] > 0 or c[1] < 0)
fraud["ece_ci_excludes_0"] = fraud.ece_ci95.map(lambda c: c[0] > 0 or c[1] < 0)

acc.to_csv("results/significance_accuracy.csv", index=False)
fraud.to_csv("results/significance_fraud.csv", index=False)

fmt_ci = lambda c: f"[{c[0]:+.3f}, {c[1]:+.3f}]"
lines = ["# Paired significance tests", "",
         f"Cascade threshold {THRESHOLD}. Accuracy: exact McNemar + paired bootstrap ({N_BOOT:,} resamples), "
         "Holm-adjusted across the 10 comparisons. Fraud: paired bootstrap "
         f"({N_BOOT_AUC:,} resamples) of the AUC and ECE differences.", "",
         "## Routing accuracy", "",
         "| A | B | n | A | B | Δ (pp) | 95% CI (pp) | p (Holm) | Significant |",
         "|---|---|---|---|---|---|---|---|---|"]
for r in acc.itertuples():
    lines.append(f"| {r.A} | {r.B} | {r.n} | {r.acc_A:.1%} | {r.acc_B:.1%} | {r.diff_pp:+.1f} | "
                 f"[{r.ci95_low_pp:+.1f}, {r.ci95_high_pp:+.1f}] | {r.p_holm:.3g} | {'yes' if r.significant_5pct else 'no'} |")
lines += ["", "## Fraud: AUC and calibration error (ECE, lower is better)", "",
          "| A | B | AUC A | AUC B | Δ AUC, 95% CI | ECE A | ECE B | Δ ECE, 95% CI |",
          "|---|---|---|---|---|---|---|---|"]
for r in fraud.itertuples():
    lines.append(f"| {r.A} | {r.B} | {r.auc_A:.3f} | {r.auc_B:.3f} | {r.auc_diff:+.3f} {fmt_ci(r.auc_ci95)} | "
                 f"{r.ece_A:.3f} | {r.ece_B:.3f} | {r.ece_diff:+.3f} {fmt_ci(r.ece_ci95)} |")
open("results/significance.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

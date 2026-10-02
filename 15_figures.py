"""Figures for the README / post, built from results/ (no API calls).

figs/accuracy_vs_latency.png  routing accuracy vs per-decision latency (log scale)
figs/cascade_cost.png         final accuracy and cost per 1M decisions: LLM alone vs cascades
figs/cost_vs_accuracy.png     cost per 1M decisions vs accuracy, with the cost-efficiency frontier
figs/tradeoff_panels.png      accuracy, latency and cost side by side for the main systems
figs/hype_check.png           TF-IDF vs gpt-5.4-mini, metric by metric
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score

# reference palette (dataviz skill), categorical slots in fixed order
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({"figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": .6, "axes.axisbelow": True,
                     "font.size": 9})

test = pd.read_parquet("data/test.parquet")[["id", "queue"]]


def acc_local(name):
    d = test.merge(pd.read_parquet(f"results/sys_{name}.parquet"), on="id")
    return accuracy_score(d.queue, d["product"].fillna(""))  # unparsed answer counts as wrong


def acc_api(fname):
    d = test.merge(pd.read_parquet(f"results/llm_{fname}.parquet"), on="id").dropna(subset=["llm_product"])
    return accuracy_score(d.queue, d.llm_product)


def p50_local(name):
    return json.load(open(f"results/sys_{name}_meta.json"))["single_p50_ms"]


def p50_api(fname):
    return pd.read_parquet(f"results/llm_{fname}.parquet").llm_ms.median()


# (label, accuracy, p50 ms, group)
pts = [
    ("gpt-5.6-terra*", acc_api("gpt-5.6-terra__s900"), p50_api("gpt-5.6-terra__lat200"), "API LLM"),
    ("gpt-5.6-luna", acc_api("gpt-5.6-luna__full"), p50_api("gpt-5.6-luna__lat200"), "API LLM"),
    ("gpt-5.4-mini", acc_api("gpt-5.4-mini"), p50_api("gpt-5.4-mini"), "API LLM"),
    ("gpt-5.4-nano", acc_api("gpt-5.4-nano"), p50_api("gpt-5.4-nano"), "API LLM"),
    ("Gemma 4 12B", acc_local("llm-gemma4-12b-it-qat"), p50_local("llm-gemma4-12b-it-qat"), "Local LLM"),
    ("Qwen3.5 9B", acc_local("llm-qwen3.5-9b"), p50_local("llm-qwen3.5-9b"), "Local LLM"),
    ("Qwen3.5 4B", acc_local("llm-qwen3.5-4b"), p50_local("llm-qwen3.5-4b"), "Local LLM"),
    ("ModernBERT fine-tuned", acc_local("modernbert-ft"), p50_local("modernbert-ft"), "Trained small model"),
    ("Qwen3-Embedding + LR", acc_local("emb-qwen3emb"), p50_local("emb-qwen3emb"), "Trained small model"),
    ("Granite R2 + LR", acc_local("emb-granite-r2"), p50_local("emb-granite-r2"), "Trained small model"),
    ("TF-IDF + LR", acc_local("tfidf-logreg"), p50_local("tfidf-logreg"), "Trained small model"),
    ("Laya fine-tuned", acc_local("laya-finetuned"), p50_local("laya-finetuned"), "Laya"),
    ("Laya zero-shot", acc_local("laya-zeroshot"), p50_local("laya-zeroshot"), "Laya"),
]
# 3 colors max on scatter plots; Laya keeps the small-model color with a triangle marker
COLORS = {"API LLM": BLUE, "Local LLM": ORANGE, "Trained small model": AQUA, "Laya": AQUA}
MARKERS = {"API LLM": "o", "Local LLM": "o", "Trained small model": "o", "Laya": "^", "Cascade": "D"}
# label offsets in points (dx, dy, ha) — hand-placed to avoid collisions
OFF = {"gpt-5.6-terra*": (8, 4, "left"), "gpt-5.6-luna": (-8, 6, "right"), "gpt-5.4-mini": (-8, -2, "right"),
       "gpt-5.4-nano": (8, -2, "left"), "Gemma 4 12B": (8, 0, "left"), "Qwen3.5 9B": (-8, -4, "right"),
       "Qwen3.5 4B": (-8, 0, "right"), "ModernBERT fine-tuned": (0, 9, "center"),
       "Qwen3-Embedding + LR": (8, 3, "left"), "Granite R2 + LR": (-8, -10, "right"),
       "TF-IDF + LR": (8, 0, "left"), "Laya fine-tuned": (8, -9, "left"), "Laya zero-shot": (8, 0, "left")}

fig, ax = plt.subplots(figsize=(8.2, 5.0))
for g, c in COLORS.items():
    sub = [p for p in pts if p[3] == g]
    ax.scatter([p[2] for p in sub], [p[1] for p in sub], s=52, color=c, marker=MARKERS[g], label=g, zorder=3,
               edgecolor="white", linewidth=1.5)
for lab, acc, ms, g in pts:
    dx, dy, ha = OFF[lab]
    ax.annotate(f"{lab} ({acc:.1%})", (ms, acc), xytext=(dx, dy), textcoords="offset points",
                ha=ha, va="center", fontsize=7.5, color=INK)
ax.set_xscale("log")
ax.set_xlim(0.5, 4000)
ax.set_ylim(0.40, 0.84)
ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
ax.xaxis.set_major_formatter(lambda v, _: f"{v:g} ms")
ax.set_xlabel("Latency per decision, p50 (log scale) — one complaint, one call")
ax.set_ylabel("Routing accuracy (9 queues)")
ax.set_title("Same accuracy band, 10–1000x apart in latency", loc="left", color=INK, fontsize=11, pad=10)
ax.legend(frameon=False, loc="lower right", fontsize=8)
fig.text(0.01, 0.005, "CFPB complaints, n=1,800 (balanced, 200/queue). *terra on the paired 900 subset. "
         "Local models on one RTX 3060 12GB.", fontsize=6.5, color=MUTED)
plt.tight_layout(rect=(0, 0.02, 1, 1))
plt.savefig("figs/accuracy_vs_latency.png", bbox_inches="tight")
plt.close()

# ---- cascade figure: two panels, one axis each (no dual axis)
c_luna = pd.read_parquet("results/llm_gpt-5.6-luna__lat200.parquet").cost_usd.mean() * 1e6
c_mini = pd.read_parquet("results/llm_gpt-5.4-mini.parquet").cost_usd.mean() * 1e6
c_terra = pd.read_parquet("results/llm_gpt-5.6-terra__lat200.parquet").cost_usd.mean() * 1e6
TH = 0.7


def cascade(local):
    r = pd.read_csv(f"results/cascade_{local}__gpt-5.6-luna.csv")
    return r[r.threshold.round(2) == TH].iloc[0]


bars = [("gpt-5.4-mini alone", acc_api("gpt-5.4-mini"), c_mini, BLUE),
        ("gpt-5.6-terra alone*", acc_api("gpt-5.6-terra__s900"), c_terra, BLUE),
        ("gpt-5.6-luna alone", acc_api("gpt-5.6-luna__full"), c_luna, BLUE)]
for local, lab in [("modernbert-ft", "ModernBERT → luna"), ("emb-qwen3emb", "Qwen3-Emb → luna"),
                   ("laya-finetuned", "Laya → luna")]:
    r = cascade(local)
    bars.append((f"{lab} ({r.local_share:.0%} local)", r.product_acc, r.usd_per_1M, AQUA))

fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.0, 3.6), sharey=True)
y = np.arange(len(bars))[::-1]
for yi, (lab, acc, cost, c) in zip(y, bars):
    a1.barh(yi, acc, color=c, height=.62)
    a1.text(acc + .004, yi, f"{acc:.1%}", va="center", fontsize=8, color=INK)
    a2.barh(yi, cost, color=c, height=.62)
    a2.text(cost * 1.12, yi, f"${cost:,.0f}", va="center", fontsize=8, color=INK)
a1.set_yticks(y, [b[0] for b in bars])
a1.set_xlim(0.70, 0.86)
a1.set_xticks([0.70, 0.74, 0.78, 0.82, 0.86])
a1.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
a1.set_title("Final routing accuracy", loc="left", color=INK)
a2.set_xscale("log")
a2.set_xlim(10, 6000)
a2.xaxis.set_major_formatter(lambda v, _: f"${v:,.0f}")
a2.set_title("Cost per 1M complaints (US$, log)", loc="left", color=INK)
for a in (a1, a2):
    a.grid(axis="y", visible=False)
fig.suptitle(f"Cascade: small model decides when confidence ≥ {TH}, the rest goes to the LLM",
             x=0.01, ha="left", fontsize=11, color=INK)
fig.text(0.01, -0.02, "Blue = LLM alone, green = cascade. API cost at synchronous list price; local cost = GPU energy. "
         "*terra on the paired 900 subset.", fontsize=6.5, color=MUTED)
plt.tight_layout()
plt.savefig("figs/cascade_cost.png", bbox_inches="tight")
plt.close()
print("figs/accuracy_vs_latency.png, figs/cascade_cost.png")

# ---- shared metrics for the trade-off figures
WATTS, USD_KWH = {"cpu": 65, "cuda": 170}, 0.15


def local_cost(name):
    m = json.load(open(f"results/sys_{name}_meta.json"))
    hours = m["batch_ms_per_complaint"] * 1e6 / 3.6e6
    return hours * WATTS["cpu" if m["device"] == "cpu" else "cuda"] / 1000 * USD_KWH


def ece(y, p, bins=10):
    b = np.clip((p * bins).astype(int), 0, bins - 1)
    return sum(abs(y[b == i].mean() - p[b == i].mean()) * (b == i).mean() for i in range(bins) if (b == i).any())


def fraud_local(name):
    d = pd.read_parquet("data/test.parquet")[["id", "fraud"]].merge(pd.read_parquet(f"results/sys_{name}.parquet"), on="id")
    return roc_auc_score(d.fraud, d.fraud_p), ece(d.fraud.values.astype(float), d.fraud_p.values)


def fraud_api(fname):
    d = (pd.read_parquet("data/test.parquet")[["id", "fraud"]]
         .merge(pd.read_parquet(f"results/llm_{fname}.parquet"), on="id").dropna(subset=["llm_fraud_p"]))
    return roc_auc_score(d.fraud, d.llm_fraud_p), ece(d.fraud.values.astype(float), d.llm_fraud_p.values)


api_cost = {"gpt-5.4-mini": c_mini, "gpt-5.6-luna": c_luna, "gpt-5.6-terra*": c_terra,
            "gpt-5.4-nano": pd.read_parquet("results/llm_gpt-5.4-nano.parquet").cost_usd.mean() * 1e6}
local_name = {"Gemma 4 12B": "llm-gemma4-12b-it-qat", "Qwen3.5 9B": "llm-qwen3.5-9b", "Qwen3.5 4B": "llm-qwen3.5-4b",
              "ModernBERT fine-tuned": "modernbert-ft", "Qwen3-Embedding + LR": "emb-qwen3emb",
              "Granite R2 + LR": "emb-granite-r2", "TF-IDF + LR": "tfidf-logreg",
              "Laya fine-tuned": "laya-finetuned", "Laya zero-shot": "laya-zeroshot"}
cost = {lab: api_cost.get(lab) if lab in api_cost else local_cost(local_name[lab]) for lab, *_ in pts}
cas_rows = {lab: cascade(loc) for loc, lab in [("modernbert-ft", "ModernBERT → luna"),
                                               ("emb-qwen3emb", "Qwen3-Emb → luna"),
                                               ("tfidf-logreg", "TF-IDF → luna")]}

# ---- figure: cost vs accuracy with the cost-efficiency frontier
fig, ax = plt.subplots(figsize=(8.2, 5.0))
allp = [(lab, acc, cost[lab], g) for lab, acc, ms, g in pts if lab != "Laya zero-shot"]
allp += [(lab, r.product_acc, r.usd_per_1M, "Cascade") for lab, r in cas_rows.items()]
for g in ["API LLM", "Local LLM", "Trained small model", "Laya", "Cascade"]:
    sub = [p for p in allp if p[3] == g]
    ax.scatter([max(p[2], 1e-4) for p in sub], [p[1] for p in sub], s=58 if g == "Cascade" else 50,
               color=AQUA if g == "Cascade" else COLORS[g], marker=MARKERS[g], zorder=3,
               edgecolor=INK if g == "Cascade" else "white", linewidth=1.2 if g == "Cascade" else 1.5,
               label="Cascade (small model → luna)" if g == "Cascade" else g)
# frontier: best accuracy reachable at or below each cost
srt = sorted(allp, key=lambda p: max(p[2], 1e-4))
fx, fy, best = [], [], -1
for lab, acc, c, g in srt:
    if acc > best:
        fx.append(max(c, 1e-4)); fy.append(acc); best = acc
ax.step(fx + [8000], fy + [fy[-1]], where="post", color=MUTED, lw=1, ls="--", zorder=1,
        label="Cost-efficiency frontier")
COFF = {"TF-IDF + LR": (0, 10, "center"), "gpt-5.4-mini": (-8, -9, "right"), "gpt-5.6-terra*": (0, 10, "center"),
        "gpt-5.6-luna": (8, -2, "left"), "ModernBERT fine-tuned": (6, 9, "left"), "ModernBERT → luna": (-10, 6, "right"),
        "TF-IDF → luna": (8, -6, "left"),
        "gpt-5.4-nano": (8, -2, "left"), "Laya fine-tuned": (8, -7, "left"), "Qwen3.5 9B": (8, -2, "left")}
for lab, acc, c, g in allp:
    if lab in COFF:
        dx, dy, ha = COFF[lab]
        ax.annotate(f"{lab} ({acc:.1%})", (max(c, 1e-4), acc), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, va="center", fontsize=7.5, color=INK)
tf, mini = next(p for p in allp if p[0] == "TF-IDF + LR"), next(p for p in allp if p[0] == "gpt-5.4-mini")
ax.annotate("", xy=(mini[2] * 0.75, mini[1]), xytext=(max(tf[2], 1e-4) * 1.6, tf[1]),
            arrowprops=dict(arrowstyle="<->", color=MUTED, lw=.8))
ax.text(0.003, 0.763, f"same accuracy: TF-IDF ≈ \\$0 vs gpt-5.4-mini \\${mini[2]:,.0f} per 1M",
        fontsize=7.5, color=MUTED, ha="center", va="top")
ax.set_xscale("log")
ax.set_xlim(5e-5, 8000)
ax.set_ylim(0.62, 0.84)
ax.xaxis.set_major_formatter(lambda v, _: f"${v:g}")
ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
ax.set_xlabel("Cost per 1M complaints (US$, log scale) — API list price / local GPU energy")
ax.set_ylabel("Routing accuracy (9 queues)")
ax.set_title("Most of the accuracy is cheap. The last points are not.", loc="left", color=INK, fontsize=11, pad=10)
ax.legend(frameon=False, loc="lower right", fontsize=7.5)
fig.text(0.01, 0.005, "Unlabeled: other local LLMs and embedding models (see README). *terra on the paired 900 subset.",
         fontsize=6.5, color=MUTED)
plt.tight_layout(rect=(0, 0.02, 1, 1))
plt.savefig("figs/cost_vs_accuracy.png", bbox_inches="tight")
plt.close()

# ---- figure: accuracy / latency / cost panels for the main systems (one axis per panel)
mb = cas_rows["ModernBERT → luna"]
rows = [  # label, accuracy, p50 ms, cost per 1M, group
    ("TF-IDF + LR", acc_local("tfidf-logreg"), p50_local("tfidf-logreg"), cost["TF-IDF + LR"], "small"),
    ("ModernBERT fine-tuned", acc_local("modernbert-ft"), p50_local("modernbert-ft"), cost["ModernBERT fine-tuned"], "small"),
    ("Laya fine-tuned", acc_local("laya-finetuned"), p50_local("laya-finetuned"), cost["Laya fine-tuned"], "small"),
    ("gpt-5.4-mini", acc_api("gpt-5.4-mini"), p50_api("gpt-5.4-mini"), c_mini, "llm"),
    ("gpt-5.6-luna", acc_api("gpt-5.6-luna__full"), p50_api("gpt-5.6-luna__lat200"), c_luna, "llm"),
    ("gpt-5.6-terra*", acc_api("gpt-5.6-terra__s900"), p50_api("gpt-5.6-terra__lat200"), c_terra, "llm"),
    (f"Cascade ModernBERT → luna", mb.product_acc, p50_local("modernbert-ft"), mb.usd_per_1M, "cascade"),
]
GC = {"small": AQUA, "llm": BLUE, "cascade": AQUA}
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.9), sharey=True)
y = np.arange(len(rows))[::-1]
for yi, (lab, acc, ms, c, g) in zip(y, rows):
    kw = dict(color=GC[g], height=.62, hatch="///" if g == "cascade" else None, edgecolor="white", linewidth=0)
    axes[0].barh(yi, acc, **kw)
    axes[0].text(acc + .003, yi, f"{acc:.1%}", va="center", fontsize=8, color=INK)
    axes[1].barh(yi, ms, **kw)
    axes[1].text(ms * 1.15, yi, f"{ms:,.0f} ms" if ms >= 1 else f"{ms:.1f} ms", va="center", fontsize=8, color=INK)
    axes[2].barh(yi, max(c, 1e-3), **kw)
    axes[2].text(max(c, 1e-3) * 1.2, yi, f"${c:,.0f}" if c >= 1 else ("≈ $0" if c < 0.01 else f"${c:.2f}"),
                 va="center", fontsize=8, color=INK)
axes[0].set_yticks(y, [r[0] for r in rows])
axes[0].set_xlim(0.70, 0.86)
axes[0].set_xticks([0.70, 0.74, 0.78, 0.82, 0.86])
axes[0].xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
axes[0].set_title("Routing accuracy", loc="left", color=INK)
axes[1].set_xscale("log"); axes[1].set_xlim(0.5, 8000); axes[1].set_xticks([1, 10, 100, 1000])
axes[1].xaxis.set_major_formatter(lambda v, _: f"{v:g} ms")
axes[1].set_title("Latency p50 (log)", loc="left", color=INK)
axes[2].set_xscale("log"); axes[2].set_xlim(1e-3, 30000)
axes[2].xaxis.set_major_formatter(lambda v, _: f"${v:g}")
axes[2].set_title("Cost per 1M (US$, log)", loc="left", color=INK)
for a in axes:
    a.grid(axis="y", visible=False)
fig.suptitle("Accuracy is a tie. Latency and cost differ by orders of magnitude.", x=0.01, ha="left",
             fontsize=11, color=INK)
fig.text(0.01, -0.03, "Green = trained small model, blue = LLM via API, hatched = cascade (74% decided by ModernBERT; "
         "its p50 is the local model's). *terra on the paired 900 subset.", fontsize=6.5, color=MUTED)
plt.tight_layout()
plt.savefig("figs/tradeoff_panels.png", bbox_inches="tight")
plt.close()

# ---- figure: hype check, TF-IDF vs gpt-5.4-mini
auc_t, ece_t = fraud_local("tfidf-logreg")
auc_m, ece_m = fraud_api("gpt-5.4-mini")
metrics = [  # name, tf-idf value, llm value, formatter, higher_is_better
    ("Routing accuracy", acc_local("tfidf-logreg"), acc_api("gpt-5.4-mini"), lambda v: f"{v:.1%}", True),
    ("Fraud AUC", auc_t, auc_m, lambda v: f"{v:.3f}", True),
    ("Calibration error (ECE)", ece_t, ece_m, lambda v: f"{v:.3f}", False),
    ("Latency per decision (p50)", p50_local("tfidf-logreg"), p50_api("gpt-5.4-mini"), lambda v: f"{v:,.0f} ms", False),
    ("Cost per 1M complaints", cost["TF-IDF + LR"], c_mini, lambda v: "≈ $0" if v < 0.01 else f"${v:,.0f}", False),
]
fig, ax = plt.subplots(figsize=(8.0, 4.2))
ax.axis("off")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.text(0.0, 0.97, "Hype check: a 20-line baseline vs an LLM", fontsize=13, color=INK, va="top")
ax.text(0.0, 0.89, "Same 1,800 complaints, same 9 queues. TF-IDF trained on 3,600 labeled examples, seconds on a CPU.",
        fontsize=8, color=MUTED, va="top")
cx = {"t": 0.60, "m": 0.86}
for key, lab, col in [("t", "TF-IDF + LR", AQUA), ("m", "gpt-5.4-mini", BLUE)]:
    ax.add_patch(plt.Rectangle((cx[key] - 0.11, 0.735), 0.22, 0.012, color=col))
    ax.text(cx[key], 0.775, lab, fontsize=10, color=INK, ha="center", va="bottom")
for i, (name, vt, vm, fmt, hib) in enumerate(metrics):
    yy = 0.63 - i * 0.135
    ax.text(0.0, yy, name, fontsize=10, color=INK, va="center")
    tie = abs(vt - vm) / max(abs(vm), 1e-9) < 0.01
    win_t = (vt > vm) == hib and not tie
    for key, v, win in [("t", vt, win_t), ("m", vm, (not win_t) and not tie)]:
        ax.text(cx[key], yy, fmt(v), fontsize=15 if win else 13, color=INK, ha="center", va="center",
                fontweight="bold" if win else "normal")
    if tie:
        ax.text(0.98, yy, "tie", fontsize=8, color=MUTED, ha="right", va="center")
    ax.plot([0, 1], [yy - 0.067, yy - 0.067], color=GRID, lw=.8)
ax.text(0.0, -0.02, "Bold = better. 'tie' = within 1%. Cost: API list price vs CPU energy.",
        fontsize=7, color=MUTED, va="top")
plt.savefig("figs/hype_check.png", bbox_inches="tight")
plt.close()
print("figs/cost_vs_accuracy.png, figs/tradeoff_panels.png, figs/hype_check.png")

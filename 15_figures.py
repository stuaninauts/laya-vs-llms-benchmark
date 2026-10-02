"""Figures for the README / post, built from results/ (no API calls).

figs/accuracy_vs_latency.png  routing accuracy vs per-decision latency (log scale)
figs/cascade_cost.png         final accuracy and cost per 1M decisions: LLM alone vs cascades
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# reference palette (dataviz skill), categorical slots in fixed order
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
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
COLORS = {"API LLM": BLUE, "Local LLM": ORANGE, "Trained small model": AQUA, "Laya": VIOLET}
# label offsets in points (dx, dy, ha) — hand-placed to avoid collisions
OFF = {"gpt-5.6-terra*": (8, 4, "left"), "gpt-5.6-luna": (-8, 6, "right"), "gpt-5.4-mini": (-8, -2, "right"),
       "gpt-5.4-nano": (8, -2, "left"), "Gemma 4 12B": (8, 0, "left"), "Qwen3.5 9B": (-8, -4, "right"),
       "Qwen3.5 4B": (-8, 0, "right"), "ModernBERT fine-tuned": (0, 9, "center"),
       "Qwen3-Embedding + LR": (8, 3, "left"), "Granite R2 + LR": (-8, -10, "right"),
       "TF-IDF + LR": (8, 0, "left"), "Laya fine-tuned": (8, -9, "left"), "Laya zero-shot": (8, 0, "left")}

fig, ax = plt.subplots(figsize=(8.2, 5.0))
for g, c in COLORS.items():
    sub = [p for p in pts if p[3] == g]
    ax.scatter([p[2] for p in sub], [p[1] for p in sub], s=46, color=c, label=g, zorder=3,
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

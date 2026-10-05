"""Hero figure (1080x1350) for the README and the LinkedIn post, built from results/ — no API calls.

Usage: python 17_hero_figure.py [repo-name]  ->  figs/hero.png
"""
import json, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.metrics import accuracy_score

REPO = sys.argv[1] if len(sys.argv) > 1 else "laya-vs-llms-benchmark"
for f in ["Lato-Regular", "Lato-Bold", "Lato-Black", "Lato-Semibold"]:
    try:
        font_manager.fontManager.addfont(f"/usr/share/fonts/truetype/lato/{f}.ttf")
    except FileNotFoundError:
        pass

# categorical slots in fixed order: 1 blue = LLM, 2 orange = Laya, 3 aqua = classic / small model
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, SECOND, MUTED, SURFACE, TINT, GTINT = "#0b0b0b", "#52514e", "#8a8984", "#fcfcfb", "#fdf0ea", "#e8f6ef"
plt.rcParams.update({"font.family": "Lato", "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE})

test = pd.read_parquet("data/test.parquet")[["id", "queue"]]


def local(name):
    d = test.merge(pd.read_parquet(f"results/sys_{name}.parquet"), on="id")
    m = json.load(open(f"results/sys_{name}_meta.json"))
    cost = m["batch_ms_per_complaint"] * 1e6 / 3.6e6 * (65 if m["device"] == "cpu" else 170) / 1000 * 0.15
    return dict(acc=accuracy_score(d.queue, d["product"].fillna("")), ms=m["single_p50_ms"], cost=cost)


def api(acc_file, lat_file):
    d = test.merge(pd.read_parquet(f"results/llm_{acc_file}.parquet"), on="id").dropna(subset=["llm_product"])
    lat = pd.read_parquet(f"results/llm_{lat_file}.parquet")
    return dict(acc=accuracy_score(d.queue, d.llm_product), ms=lat.llm_ms.median(), cost=lat.cost_usd.mean() * 1e6)


cas = pd.read_csv("results/cascade_modernbert-ft__gpt-5.6-luna.csv")
cas = cas[cas.threshold.round(2) == 0.7].iloc[0]
mb = local("modernbert-ft")
rows = [  # label, metrics, color, band
    ("Laya, zero-shot", local("laya-zeroshot"), ORANGE, "laya"),
    ("Laya, fine-tuned", local("laya-finetuned"), ORANGE, "laya"),
    ("ModernBERT, fine-tuned", mb, AQUA, None),
    ("TF-IDF + LR", local("tfidf-logreg"), AQUA, None),
    ("gpt-5.4-mini", api("gpt-5.4-mini", "gpt-5.4-mini"), BLUE, None),
    ("gpt-5.6-luna", api("gpt-5.6-luna__full", "gpt-5.6-luna__lat200"), BLUE, None),
    ("ModernBERT + luna", dict(acc=cas.product_acc, ms=mb["ms"], cost=cas.usd_per_1M), AQUA, "hybrid"),
]


def money(v):
    return "≈$0" if v < 0.01 else (f"${v:.2f}" if v < 1 else f"${v:,.0f}")


fig = plt.figure(figsize=(7.2, 9.0), dpi=150)
T = fig.text
T(0.06, 0.955, "I tested the open-source Jev.", fontsize=29, fontweight="black", color=INK, va="top")
T(0.06, 0.895, "Laya vs classic ML vs GPT · 1,800 real financial complaints · 9 teams",
  fontsize=12, color=SECOND, va="top")

TOP, BOT = 0.805, 0.33
n = len(rows)
rh = (TOP - BOT) / n
# row bands (behind everything)
for i, (_, _, _, band) in enumerate(rows):
    y0 = TOP - (i + 1) * rh
    if band == "laya":
        fig.patches.append(plt.Rectangle((0.04, y0), 0.92, rh, color=TINT, transform=fig.transFigure, figure=fig,
                                         zorder=-1))
    if band == "hybrid":
        fig.patches.append(plt.Rectangle((0.04, y0 + 0.004), 0.92, rh - 0.008, facecolor=GTINT, edgecolor=AQUA,
                                         linewidth=1.3, transform=fig.transFigure, figure=fig, zorder=-1))
for i, (lab, m, c, band) in enumerate(rows):
    yc = TOP - (i + 0.5) * rh
    T(0.06, yc, lab, fontsize=12.5, fontweight="bold" if band else "normal", color=INK, va="center")
    if band == "hybrid":
        T(0.06, yc - 0.022, "small model + LLM", fontsize=9.5, color=SECOND, va="center")

# three metric columns, one axis each
cols = [("Accuracy", "higher is better", "acc", (0.40, 1.08), False, lambda v: f"{v:.1%}"),
        ("Latency", "per decision", "ms", (0.5, 2e5), True, lambda v: f"{v:,.0f} ms"),
        ("Cost / 1M", "complaints", "cost", (2e-3, 2e6), True, money)]
x0, cw, gap = 0.335, 0.195, 0.015
for j, (title, sub, key, lim, log, fmt) in enumerate(cols):
    ax = fig.add_axes([x0 + j * (cw + gap), BOT, cw, TOP - BOT])
    ax.set_facecolor("none")
    for i, (_, m, c, band) in enumerate(rows):
        yi = n - 1 - i
        v = max(m[key], lim[0] * 1.05)
        ax.barh(yi, v - (0 if log else lim[0]), left=0 if log else lim[0], color=c, height=0.5,
                hatch="///" if band == "hybrid" else None, edgecolor=SURFACE, linewidth=0)
        ax.text(v * 1.35 if log else v + 0.012, yi, fmt(m[key]), va="center", fontsize=11, color=INK,
                fontweight="bold" if key == "acc" else "normal")
    if log:
        ax.set_xscale("log")
    ax.set_xlim(*lim)
    ax.set_ylim(-0.5, n - 0.5)
    ax.axis("off")
    T(x0 + j * (cw + gap), TOP + 0.03, title, fontsize=13, fontweight="bold", color=INK, va="center")
    T(x0 + j * (cw + gap), TOP + 0.009, sub, fontsize=9.5, color=MUTED, va="center")

# takeaways
lt = rows[1][1]
mini = rows[4][1]
takeaways = [
    (ORANGE, "Out of the box, Laya isn't ready.", "45% accuracy without training."),
    (ORANGE, "Fine-tuned, it delivers.", f"GPT-level accuracy, better calibrated, {mini['ms'] / lt['ms']:.0f}x faster."),
    (AQUA, "The newest option wasn't the best.", "A plain fine-tune and even TF-IDF matched it."),
    (AQUA, "Combining models won.", f"{cas.product_acc:.1%}: above the same LLM alone ({rows[5][1]['acc']:.1%}), at ${cas.usd_per_1M:,.0f} per 1M."),
]
for k, (c, a, b) in enumerate(takeaways):
    yy = 0.285 - k * 0.058
    fig.patches.append(plt.Rectangle((0.06, yy - 0.019), 0.008, 0.038, color=c, transform=fig.transFigure, figure=fig))
    T(0.08, yy + 0.009, a, fontsize=12.5, fontweight="bold", color=INK, va="center")
    T(0.08, yy - 0.012, b, fontsize=11, color=SECOND, va="center")

T(0.06, 0.03, f"Latency and cost in log scale · paired significance tests · github → {REPO}", fontsize=8.5, color=MUTED)
plt.savefig("figs/hero.png", dpi=150)
print("figs/hero.png")

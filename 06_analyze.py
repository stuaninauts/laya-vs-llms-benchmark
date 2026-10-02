"""Consolida tudo o que existir em results/ — não chama API nenhuma.

Lê results/llm_*.parquet (GPT via API, já pago) e results/sys_*.parquet (sistemas locais, ver sysio.py).
- métricas por sistema: acurácia e F1 macro de fila; precisão/recall/F1/AUC/ECE de fraude
- latência e custo por 1 milhão de reclamações
- cascata: sistema local decide quando confiança >= t; o resto vai para o LLM de API
Grava results/summary.md e results/cascade_<sistema>.csv.
"""
import glob, json, os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

# Premissas de custo dos sistemas locais (ficam explícitas no summary)
WATTS = {"cuda": 170, "cpu": 65}   # TDP RTX 3060 / CPU desktop típica
ENERGY_USD_PER_KWH = 0.15
CLOUD_T4_USD_PER_HOUR = 0.35       # GPU T4 sob demanda, referência GCP
CASCADE_LLMS = ["gpt-5.4-mini", "gpt-5.6-luna__full", "gpt-5.6-terra__s900"]   # reservas (terra só na s900)

test = pd.read_parquet("data/test.parquet")[["id", "queue", "fraud"]]


def ece(y, p, bins=10):
    """Erro de calibração esperado da probabilidade de fraude."""
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
lines = ["# Resultados — triagem de reclamações CFPB", "",
         f"Teste: {len(test)} reclamações, 9 filas (200 cada), fraude {test.fraud.mean():.1%}.", "",
         summary.round(4).to_markdown(index=False), "",
         f"Custo local = energia ({WATTS['cuda']} W GPU / {WATTS['cpu']} W CPU, US$ {ENERGY_USD_PER_KWH}/kWh) "
         f"no throughput em lote; referência de nuvem T4 US$ {CLOUD_T4_USD_PER_HOUR}/h. "
         "p50/p95 = decisão isolada (1 reclamação por chamada); NaN = rodado via Batch API (sem latência). "
         "Sistemas com n diferente não são diretamente comparáveis — ver a tabela pareada abaixo."]

# comparação pareada: todo sistema avaliado nas MESMAS reclamações da amostra s900
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
        lines += ["", f"## Comparação pareada na amostra s900 ({len(s900)} reclamações, as mesmas para todos)", "",
                  pd.DataFrame(paired).sort_values("product_acc", ascending=False).round(4).to_markdown(index=False)]

# Cascatas: sistema local decide quando confiança >= t; o resto vai para um LLM de reserva.
# Custo da reserva = preço SÍNCRONO por reclamação (cascata é um sistema online): usa o arquivo
# __lat200 do modelo quando existir (ex.: luna rodou o teste completo via batch, a -50%).
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
        cascade_best.append({"local": name, "reserva": fb_name, "melhor_threshold": b.threshold,
                             "local_share": b.local_share, "product_acc": b.product_acc,
                             "usd_per_1M": b.usd_per_1M, "reserva_sozinha_acc": llm_acc,
                             "reserva_sozinha_usd_per_1M": llm_cost})
        lines += ["", f"## Cascata {name} → {fb_name} (n={len(m)})", "", cas.round(4).to_markdown(index=False)]

if cascade_best:
    cb = pd.DataFrame(cascade_best).sort_values("product_acc", ascending=False)
    cb.to_csv("results/cascade_best.csv", index=False)
    lines += ["", "## Resumo: melhor ponto de cada cascata (threshold com maior acurácia final)", "",
              cb.round(4).to_markdown(index=False)]

open("results/summary.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

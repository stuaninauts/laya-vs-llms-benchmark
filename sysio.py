"""Formato único de resultado para os sistemas locais (Laya, encoder, embeddings, LLM local, TF-IDF).

results/sys_<nome>.parquet   colunas: id, product, product_conf, fraud_p
results/sys_<nome>_meta.json {system, device, single_p50_ms, single_p95_ms, batch_ms_per_complaint, notes}
"""
import json, os, time
import pandas as pd

# SMOKE=1: teste rápido (5 reclamações, treino mínimo), resultados em results/smoke/
SMOKE = os.environ.get("SMOKE") == "1"
RESULTS = "results/smoke" if SMOKE else "results"
os.makedirs(RESULTS, exist_ok=True)


def load_test():
    import pandas as pd
    t = pd.read_parquet("data/test.parquet")
    return t.head(5) if SMOKE else t


def load_train():
    import pandas as pd
    t = pd.read_parquet("data/train.parquet")
    # no teste rápido: 10 por fila, para ainda haver todas as classes e positivos de fraude
    return t.groupby("queue", group_keys=False).head(10) if SMOKE else t


def save(name: str, ids, product, product_conf, fraud_p, device: str, single_ms, batch_ms: float, notes: str = ""):
    pd.DataFrame({"id": list(ids), "product": list(product), "product_conf": list(product_conf),
                  "fraud_p": list(fraud_p)}).to_parquet(f"{RESULTS}/sys_{name}.parquet")
    s = pd.Series(single_ms, dtype=float)
    meta = {"system": name, "device": device, "single_p50_ms": float(s.median()),
            "single_p95_ms": float(s.quantile(.95)), "batch_ms_per_complaint": float(batch_ms), "notes": notes}
    json.dump(meta, open(f"{RESULTS}/sys_{name}_meta.json", "w"), indent=2)
    return meta


def time_single(fn, items, n=50):
    """Latência de uma decisão isolada (1 reclamação chega, 1 chamada). Aquece antes."""
    fn(items[0])
    out = []
    for x in items[:n]:
        t0 = time.perf_counter()
        fn(x)
        out.append((time.perf_counter() - t0) * 1000)
    return out

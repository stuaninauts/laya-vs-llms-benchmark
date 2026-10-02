"""Embeddings congelados + regressão logística: "um encoder bom sem ajuste nenhum já basta?"

Uso: .venv/bin/python 09_embeddings_logreg.py <modelo_hf> <nome_curto> [device]
  ex.: Qwen/Qwen3-Embedding-0.6B qwen3emb
       ibm-granite/granite-embedding-small-english-r2 granite-r2
Grava results/sys_emb-<nome_curto>.parquet. Custo: zero (GPU local ou CPU).
Requer: sentence-transformers (ver 02-modelos-locais-mapeados.md).
"""
import sys, time
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegressionCV

from common import MAX_CHARS
import sysio

MODEL_ID, NAME = sys.argv[1], sys.argv[2]
DEVICE = sys.argv[3] if len(sys.argv) > 3 else ("cuda" if torch.cuda.is_available() else "cpu")
SEED = 20260929
# Qwen3-Embedding é instruído: o texto a classificar vai como "query" com a tarefa descrita
PROMPTS = {"Qwen/Qwen3-Embedding-0.6B":
           "Instruct: Classify this consumer financial complaint by product and by whether it reports fraud\nQuery: "}

train = sysio.load_train()
test = sysio.load_test()
CV = 3 if sysio.SMOKE else 5
enc = SentenceTransformer(MODEL_ID, device=DEVICE, model_kwargs={"dtype": torch.float16} if DEVICE == "cuda" else {})
enc.max_seq_length = 512
prompt = PROMPTS.get(MODEL_ID)


def embed(texts, bs=32):
    return enc.encode([t[:MAX_CHARS] for t in texts], batch_size=bs, prompt=prompt,
                      normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)


t0 = time.time()
Xtr = embed(train.text)
print(f"embeddings de treino em {time.time()-t0:.0f}s, dim {Xtr.shape[1]}")

clf_q = LogisticRegressionCV(Cs=6, cv=CV, max_iter=3000, random_state=SEED).fit(Xtr, train.queue)
clf_f = LogisticRegressionCV(Cs=6, cv=CV, max_iter=3000, random_state=SEED, scoring="roc_auc").fit(Xtr, train.fraud)

texts = list(test.text)
single = sysio.time_single(lambda t: (clf_q.predict_proba(embed([t], 1)), clf_f.predict_proba(embed([t], 1))), texts)
t1 = time.perf_counter()
Xte = embed(texts)
pq = clf_q.predict_proba(Xte)
pf = clf_f.predict_proba(Xte)[:, list(clf_f.classes_).index(True)]
batch_ms = (time.perf_counter() - t1) * 1000 / len(texts)

pred = clf_q.classes_[pq.argmax(1)]
meta = sysio.save(f"emb-{NAME}", test.id, pred, pq.max(1), pf, DEVICE, single, batch_ms,
                  notes=f"{MODEL_ID} congelado + LogisticRegressionCV")
print(f"emb-{NAME}: acc fila {(pred == test.queue.values).mean():.3f} | p50 {meta['single_p50_ms']:.0f}ms "
      f"| lote {batch_ms:.1f}ms")

"""Piso clássico: TF-IDF + regressão logística (o mesmo da EDA), no formato padrão de resultado.

Uso: .venv/bin/python 11_tfidf.py      Custo: zero, segundos em CPU.
"""
import time
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from common import MAX_CHARS
import sysio

train = sysio.load_train()
test = sysio.load_test()
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=50000, sublinear_tf=True,
                      token_pattern=r"(?u)\b[a-wyz][a-z]{2,}\b")  # ignora as redações XXXX
Xtr = vec.fit_transform(train.text.str[:MAX_CHARS])
clf_q = LogisticRegression(max_iter=2000, C=5).fit(Xtr, train.queue)
clf_f = LogisticRegression(max_iter=2000, C=5, class_weight="balanced").fit(Xtr, train.fraud)

texts = [t[:MAX_CHARS] for t in test.text]
single = sysio.time_single(lambda t: (clf_q.predict_proba(vec.transform([t])), clf_f.predict_proba(vec.transform([t]))), texts)
t0 = time.perf_counter()
X = vec.transform(texts)
pq, pf = clf_q.predict_proba(X), clf_f.predict_proba(X)[:, 1]
batch_ms = (time.perf_counter() - t0) * 1000 / len(texts)

pred = clf_q.classes_[pq.argmax(1)]
meta = sysio.save("tfidf-logreg", test.id, pred, pq.max(1), pf, "cpu", single, batch_ms,
                  notes="TF-IDF 1-2gram + LogisticRegression, class_weight balanced na fraude")
print(f"tfidf-logreg: acc fila {(pred == test.queue.values).mean():.3f} | p50 {meta['single_p50_ms']:.2f}ms "
      f"| lote {batch_ms:.3f}ms")

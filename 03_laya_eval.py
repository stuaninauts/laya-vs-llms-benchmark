"""Roda o Laya (zero-shot ou fine-tuned) no conjunto de teste.

Uso: .venv/bin/python 03_laya_eval.py <tag> [checkpoint] [device]
  tag        nome do resultado (ex.: zeroshot, finetuned)
  checkpoint repo HF ou pasta local (default: convaiinnovations/laya)
  device     cuda | cpu (default: cuda se disponível)
Grava results/sys_laya-<tag>.parquet (+ _meta.json).
"""
import sys, time
import pandas as pd
import torch
import laya

from common import QUESTIONS, MAX_CHARS
import sysio

TAG = sys.argv[1]
CKPT = sys.argv[2] if len(sys.argv) > 2 else "convaiinnovations/laya"
DEVICE = sys.argv[3] if len(sys.argv) > 3 else ("cuda" if torch.cuda.is_available() else "cpu")
BATCH = 32 if DEVICE == "cuda" else 8

test = sysio.load_test()
agent = laya.load(CKPT, device=DEVICE)
states = [t[:MAX_CHARS] for t in test.text]

single = sysio.time_single(lambda s: agent.predict(s, QUESTIONS), states)

# throughput em lote (caso batch: fila de reclamações do dia)
t0 = time.perf_counter()
res = agent.predict_batch(states, QUESTIONS, batch_size=BATCH, sort_by_length=True)
batch_ms = (time.perf_counter() - t0) * 1000 / len(states)

meta = sysio.save(f"laya-{TAG}", test.id,
                  [r["answers"]["product"]["choice"] for r in res],
                  [r["answers"]["product"]["confidence"] for r in res],
                  [r["answers"]["fraud"]["noul"] for r in res],
                  DEVICE, single, batch_ms, notes=f"checkpoint={CKPT}")
acc = (pd.read_parquet(f"{sysio.RESULTS}/sys_laya-{TAG}.parquet")["product"].values == test.queue.values).mean()
print(f"laya {TAG} [{DEVICE}]: acc produto {acc:.3f} | single p50 {meta['single_p50_ms']:.0f}ms "
      f"| lote {batch_ms:.1f}ms/reclamação")

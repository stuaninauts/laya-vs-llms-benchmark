"""Baixa os modelos do experimento de forma retomável: se a internet cair, espera e continua.

O huggingface_hub mantém os arquivos parciais (.incomplete) no cache e retoma de onde parou;
este script só repete as tentativas com espera crescente. Pode ser interrompido e rodado de
novo a qualquer momento: o que já terminou não é baixado outra vez.

Uso: .venv/bin/python download_models.py
"""
import os, sys, time

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
from huggingface_hub import snapshot_download

MODELS = [
    # só o checkpoint inglês do Laya (raiz do repo); multilíngue e typed-decisions ficam de fora
    ("convaiinnovations/laya", {"ignore_patterns": ["multilingual/*", "typed-decisions/*", "eval/*", "assets/*"]}),
    ("ibm-granite/granite-embedding-small-english-r2", {}),
    ("Qwen/Qwen3-Embedding-0.6B", {}),
    ("answerdotai/ModernBERT-large", {}),
]
MAX_WAIT = 300


def fetch(repo, kw):
    wait, attempt = 10, 0
    while True:
        attempt += 1
        try:
            path = snapshot_download(repo, max_workers=4, **kw)
            print(f"OK {repo} -> {path}", flush=True)
            return
        except KeyboardInterrupt:
            raise
        except Exception as e:  # queda de rede, timeout, 5xx: espera e tenta de novo
            print(f"RETRY {repo} tentativa {attempt}: {type(e).__name__}: {str(e)[:160]} "
                  f"(nova tentativa em {wait}s)", flush=True)
            time.sleep(wait)
            wait = min(wait * 2, MAX_WAIT)


for repo, kw in MODELS:
    fetch(repo, kw)
print("DOWNLOAD COMPLETO", flush=True)

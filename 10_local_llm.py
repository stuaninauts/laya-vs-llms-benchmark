"""LLM local via Ollama, na mesma GPU do Laya: isola arquitetura de preço de API.

Mesmas opções do GPT (04_llm_baseline.py), mas o modelo responde só "<letra> <Y|N>" e lemos os
logprobs do token da letra e do token Y/N — assim temos probabilidade de verdade (não a
"falada"), comparável com a do Laya na curva de calibração.

Uso: .venv/bin/python 10_local_llm.py <modelo_ollama> [limite]   ex.: qwen3.5:9b
Requer Ollama >= 0.12.11 (logprobs) e `ollama pull <modelo>`. Custo: zero (GPU local).
Grava results/sys_llm-<modelo>.parquet.
"""
import math, os, sys, time
import pandas as pd
import requests

from common import PRODUCTS, FRAUD_QUESTION, MAX_CHARS
import sysio

MODEL = sys.argv[1]
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else None
HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
QUEUES = list(PRODUCTS)
LETTERS = "ABCDEFGHI"
L2Q = dict(zip(LETTERS, QUEUES))

SYSTEM = (
    "You triage consumer complaints received by a US financial institution.\n"
    "Which team must handle the complaint?\n"
    + "\n".join(f"{l}) {q}: {PRODUCTS[q]}" for l, q in zip(LETTERS, QUEUES))
    + f"\n\nAlso answer: {FRAUD_QUESTION['instructions']} (Y = yes, N = no)\n"
    "Reply with exactly one letter A-I, a space, then Y or N. Nothing else. Example: C N"
)


def check_server():
    v = requests.get(f"{HOST}/api/version", timeout=5).json()["version"]
    major, minor, patch = (int(x) for x in v.split("-")[0].split(".")[:3])
    if (major, minor, patch) < (0, 12, 11):
        sys.exit(f"Ollama {v} não expõe logprobs; atualizar (ver 02-modelos-locais-mapeados.md)")
    names = [m["name"] for m in requests.get(f"{HOST}/api/tags", timeout=5).json()["models"]]
    if not any(n == MODEL or n.startswith(MODEL + ":") for n in names):
        sys.exit(f"modelo {MODEL} não baixado: ollama pull {MODEL}")
    return v


def dist_from(top, allowed):
    """Probabilidades renormalizadas sobre os rótulos permitidos, a partir dos top_logprobs."""
    p = {}
    for t in top:
        k = t["token"].strip().upper()
        if k in allowed:
            p[k] = p.get(k, 0.0) + math.exp(t["logprob"])
    s = sum(p.values())
    return {k: v / s for k, v in p.items()} if s > 0 else {}


def post_chat(payload, tries=5):
    """POST com nova tentativa: se o servidor do Ollama cair/reiniciar, espera e tenta de novo."""
    for i in range(tries):
        try:
            return requests.post(f"{HOST}/api/chat", timeout=120, json=payload).json()
        except requests.exceptions.ConnectionError:
            if i == tries - 1:
                raise
            time.sleep(10 * (i + 1))


def ask(text):
    r = post_chat({
        "model": MODEL, "stream": False, "think": False, "logprobs": True, "top_logprobs": 20,
        "options": {"temperature": 0, "num_predict": 6, "seed": 0},
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": text[:MAX_CHARS]}]})
    toks = r.get("logprobs") or []
    pq, pf = {}, {}
    for t in toks:  # primeiro token que é letra de fila, depois o primeiro Y/N
        k = t["token"].strip().upper()
        if not pq and k in LETTERS:
            pq = dist_from(t.get("top_logprobs", []), set(LETTERS))
        elif pq and not pf and k in {"Y", "N"}:
            pf = dist_from(t.get("top_logprobs", []), {"Y", "N"})
    content = r.get("message", {}).get("content", "").strip().upper()
    if not pq and content[:1] in LETTERS:        # fallback: sem logprob, confiança 1.0
        pq = {content[0]: 1.0}
    if not pf and content[-1:] in {"Y", "N"}:
        pf = {content[-1]: 1.0}
    best = max(pq, key=pq.get) if pq else None
    return {"product": L2Q.get(best), "product_conf": pq.get(best, float("nan")),
            "fraud_p": pf.get("Y", 1.0 - pf.get("N", 0.5)) if pf else float("nan"), "raw": content}


def main():
    v = check_server()
    test = sysio.load_test()
    if LIMIT:
        test = test.head(LIMIT)
    ask(test.text.iloc[0])  # carrega o modelo na GPU
    rows, lat = [], []
    t0 = time.perf_counter()
    for t in test.text:  # sequencial: é o caso online e o que o Ollama serve por padrão
        t1 = time.perf_counter()
        rows.append(ask(t))
        lat.append((time.perf_counter() - t1) * 1000)
    batch_ms = (time.perf_counter() - t0) * 1000 / len(test)
    out = pd.DataFrame(rows)
    name = "llm-" + MODEL.replace(":", "-").replace("/", "-")
    meta = sysio.save(name, test.id, out["product"], out.product_conf, out.fraud_p, "cuda (ollama)", lat, batch_ms,
                      notes=f"ollama {v}; {out['product'].isna().sum()} respostas não parseadas")
    acc = (out["product"].values == test.queue.values).mean()
    print(f"{name}: acc fila {acc:.3f} | p50 {meta['single_p50_ms']:.0f}ms | não parseadas {out['product'].isna().sum()}")


if __name__ == "__main__":
    main()

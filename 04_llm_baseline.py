"""LLM baseline: the same two decisions (product + fraud) with a small GPT, structured output.

COSTS MONEY (OpenAI API). Only runs with an explicit --spend and never overwrites an existing result.
Usage: .venv/bin/python 04_llm_baseline.py gpt-5.4-nano --spend [limit]
Writes results/llm_<model>.parquet with prediction, tokens, latency and cost.
Already run on 2026-09-29: nano US$ 0.246, mini US$ 0.921 (1,800 complaints each).
"""
import asyncio, json, os, sys, time
import pandas as pd
from openai import AsyncOpenAI

from common import PRODUCTS, FRAUD_QUESTION, MAX_CHARS

# US$ per 1M tokens (input, output) — public OpenAI price list, Sep/2026
PRICES = {"gpt-5.4-nano": (0.20, 1.25), "gpt-5.4-mini": (0.75, 4.50), "gpt-4.1-nano": (0.10, 0.40)}

args = [a for a in sys.argv[1:] if a != "--spend"]
MODEL = args[0]
LIMIT = int(args[1]) if len(args) > 1 else None
CONCURRENCY = 16
OUT = f"results/llm_{MODEL}.parquet"

if "--spend" not in sys.argv:
    sys.exit(f"Refused: this script calls a paid API. Run with --spend to authorize ({MODEL}).")
if os.path.exists(OUT):
    sys.exit(f"{OUT} already exists — not paying again. Delete the file if you want to rerun.")

SYSTEM = (
    "You triage consumer complaints received by a US financial institution.\n"
    "Return JSON only.\n"
    "product: the queue that must handle the complaint. Options:\n"
    + "\n".join(f"- {k}: {v}" for k, v in PRODUCTS.items())
    + f"\nfraud: {FRAUD_QUESTION['instructions']} "
    f"(true = {FRAUD_QUESTION['criteria']['true']}; false = {FRAUD_QUESTION['criteria']['false']})\n"
    "fraud_probability: your probability (0-1) that fraud is true."
)

SCHEMA = {
    "type": "object",
    "properties": {
        "product": {"type": "string", "enum": list(PRODUCTS)},
        "fraud": {"type": "boolean"},
        "fraud_probability": {"type": "number"},
    },
    "required": ["product", "fraud", "fraud_probability"],
    "additionalProperties": False,
}

client = AsyncOpenAI()


async def classify(sem, row):
    extra = {"reasoning_effort": "none"} if MODEL.startswith("gpt-5") else {"temperature": 0}
    async with sem:
        for attempt in range(5):
            try:
                t0 = time.perf_counter()
                r = await client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": row.text[:MAX_CHARS]}],
                    response_format={"type": "json_schema",
                                     "json_schema": {"name": "triage", "strict": True, "schema": SCHEMA}},
                    **extra,
                )
                dt = (time.perf_counter() - t0) * 1000
                out = json.loads(r.choices[0].message.content)
                pin, pout = PRICES[MODEL]
                cost = (r.usage.prompt_tokens * pin + r.usage.completion_tokens * pout) / 1e6
                return {"id": row.id, "llm_product": out["product"], "llm_fraud": out["fraud"],
                        "llm_fraud_p": out["fraud_probability"], "llm_ms": dt,
                        "tok_in": r.usage.prompt_tokens, "tok_out": r.usage.completion_tokens, "cost_usd": cost}
            except Exception as e:  # rate limit / transient
                err = e
                await asyncio.sleep(2 ** attempt)
        return {"id": row.id, "error": repr(err)}


async def main():
    test = pd.read_parquet("data/test.parquet")
    if LIMIT:
        test = test.head(LIMIT)
    sem = asyncio.Semaphore(CONCURRENCY)
    t0 = time.time()
    rows = await asyncio.gather(*(classify(sem, r) for r in test.itertuples()))
    out = pd.DataFrame(rows)
    out.to_parquet(f"results/llm_{MODEL}.parquet")
    ok = out.dropna(subset=["llm_product"]) if "llm_product" in out else out
    print(f"{MODEL}: {len(ok)}/{len(out)} ok in {time.time()-t0:.0f}s | total cost US$ {ok.cost_usd.sum():.3f} "
          f"| p50 {ok.llm_ms.median():.0f}ms p95 {ok.llm_ms.quantile(.95):.0f}ms")


asyncio.run(main())

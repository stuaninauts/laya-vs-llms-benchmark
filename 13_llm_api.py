"""API LLMs (OpenAI and Anthropic) with the same prompt/schema as the already-paid baseline.

COSTS MONEY in sync and submit modes — they only run with --spend, and never overwrite a result.
The estimate mode is local and free.

Usage:
  .venv/bin/python 13_llm_api.py estimate                              cost table (free)
  .venv/bin/python 13_llm_api.py probe  openai <model> --spend         3 calls (~US$ 0.001): confirms
                                         real token counts (including reasoning) and re-estimates cost
  .venv/bin/python 13_llm_api.py sync   <provider> <model> <sample> --spend
  .venv/bin/python 13_llm_api.py submit <provider> <model> <sample> --spend   (Batch API, -50%)
  .venv/bin/python 13_llm_api.py collect <provider> <model> <sample>          (downloads the batch; free)
  provider: openai | anthropic     sample: full (1,800) | s900 | lat200

sync measures real latency (use lat200); submit/collect gives cheap accuracy (use s900 or full).
Writes results/llm_<model>__<sample>.parquet in the same format as the baseline (read by 06_analyze.py).
Keys: OPENAI_API_KEY; ANTHROPIC_API_KEY (or `ant auth login`).
"""
import asyncio, json, os, sys, time
import pandas as pd

from common import MAX_CHARS
from llm_prompt import SYSTEM, SCHEMA, PRICES, BATCH_DISCOUNT, estimate

BUDGET_USD = float(os.environ.get("BUDGET_USD", "2.0"))  # budget cap for this stage (POST-LAYA key)
BRL_PER_USD = 5.40  # reference exchange rate; adjust
SAMPLES = {"full": "data/test.parquet", "s900": "data/test_s900.parquet", "lat200": "data/test_lat200.parquet"}
CONCURRENCY = 8
BATCH_DIR = "results/batches"


def load_env_key():
    """Uses the key from the project's .env (a hyphenated name can't be a shell variable). Never prints it."""
    if not os.path.exists(".env"):
        return False
    for line in open(".env"):
        k, _, v = line.strip().partition("=")
        if "OPENAI" in k.upper() and v:
            os.environ["OPENAI_API_KEY"] = v.strip().strip('"').strip("'")
            return True
    return False


def spent_this_stage():
    """Spend already recorded in this stage: new results (llm_<model>__<sample>) + submitted batches."""
    import glob
    total = sum(pd.read_parquet(f)["cost_usd"].sum() for f in glob.glob("results/llm_*__*.parquet"))
    for st in [f for f in glob.glob(f"{BATCH_DIR}/*.json") if not os.path.basename(f).startswith("probe_")]:
        d = json.load(open(st))
        if not os.path.exists(out_path(d["model"], d["sample"])):
            total += d.get("estimate_usd", 0.0)   # batch in progress: count its estimate
    total += sum(json.load(open(f)).get("cost_usd", 0.0) for f in glob.glob(f"{BATCH_DIR}/probe_*.json"))
    return total


def check_budget(est):
    spent = spent_this_stage()
    if spent + est > BUDGET_USD:
        sys.exit(f"Refused by budget cap: already spent US$ {spent:.3f} + estimate US$ {est:.3f} > US$ {BUDGET_USD:.2f}.")
    print(f"budget: spent US$ {spent:.3f} + estimate US$ {est:.3f} ≤ cap US$ {BUDGET_USD:.2f}")


def out_path(model, sample):
    return f"results/llm_{model}__{sample}.parquet"


def cost(provider, model, tin, tout, batch):
    pin, pout = PRICES[(provider, model)]
    return (tin * pin + tout * pout) / 1e6 * (BATCH_DISCOUNT if batch else 1.0)


def parse(text):
    d = json.loads(text)
    return {"llm_product": d["product"], "llm_fraud": bool(d["fraud"]), "llm_fraud_p": float(d["fraud_probability"])}


# ---------------------------------------------------------------- OpenAI
def openai_params(model, text):
    p = {"model": model,
         "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text[:MAX_CHARS]}],
         "response_format": {"type": "json_schema", "json_schema": {"name": "triage", "strict": True, "schema": SCHEMA}}}
    if model.startswith("gpt-5"):
        p["reasoning_effort"] = os.environ.get("OPENAI_REASONING_EFFORT", "none")
    else:
        p["temperature"] = 0
    return p


async def openai_sync(model, df):
    from openai import AsyncOpenAI
    client, sem = AsyncOpenAI(), asyncio.Semaphore(CONCURRENCY)

    async def one(row):
        async with sem:
            for attempt in range(5):
                try:
                    t0 = time.perf_counter()
                    r = await client.chat.completions.create(**openai_params(model, row.text))
                    ms = (time.perf_counter() - t0) * 1000
                    return {"id": row.id, **parse(r.choices[0].message.content), "llm_ms": ms,
                            "tok_in": r.usage.prompt_tokens, "tok_out": r.usage.completion_tokens,
                            "cost_usd": cost("openai", model, r.usage.prompt_tokens, r.usage.completion_tokens, False)}
                except Exception as e:
                    err = e
                    await asyncio.sleep(2 ** attempt)
            return {"id": row.id, "error": repr(err)}
    return await asyncio.gather(*(one(r) for r in df.itertuples()))


def openai_submit(model, df, tag):
    from openai import OpenAI
    client = OpenAI()
    path = f"{BATCH_DIR}/{tag}.jsonl"
    with open(path, "w") as f:
        for row in df.itertuples():
            f.write(json.dumps({"custom_id": str(row.id), "method": "POST", "url": "/v1/chat/completions",
                                "body": openai_params(model, row.text)}) + "\n")
    fid = client.files.create(file=open(path, "rb"), purpose="batch").id
    b = client.batches.create(input_file_id=fid, endpoint="/v1/chat/completions", completion_window="24h")
    return b.id


def openai_collect(model, batch_id):
    from openai import OpenAI
    client = OpenAI()
    b = client.batches.retrieve(batch_id)
    if b.status != "completed":
        sys.exit(f"batch {batch_id} still {b.status} ({b.request_counts})")
    rows = []
    for line in client.files.content(b.output_file_id).text.splitlines():
        r = json.loads(line)
        body = r["response"]["body"]
        try:
            u = body["usage"]
            rows.append({"id": int(r["custom_id"]), **parse(body["choices"][0]["message"]["content"]), "llm_ms": float("nan"),
                         "tok_in": u["prompt_tokens"], "tok_out": u["completion_tokens"],
                         "cost_usd": cost("openai", model, u["prompt_tokens"], u["completion_tokens"], True)})
        except Exception as e:
            rows.append({"id": int(r["custom_id"]), "error": repr(e)})
    return rows


# ---------------------------------------------------------------- Anthropic
def anthropic_params(model, text):
    p = {"model": model, "max_tokens": 1024 if model == "claude-opus-5-5" else 256,
         "system": SYSTEM, "messages": [{"role": "user", "content": text[:MAX_CHARS]}],
         "output_config": {"format": {"type": "json_schema", "schema": SCHEMA}}}
    if model == "claude-sonnet-5-5":
        p["thinking"] = {"type": "between_tools"}          # turns reasoning off ("disabled" returns 400 on 5.5)
    elif model == "claude-opus-5-5":
        p["output_config"]["effort"] = "low"                # reasoning can't be turned off on Opus 5.5
    return p                                                 # Haiku 4.5: no thinking by default


def anthropic_text(msg):
    return next(b.text for b in msg.content if b.type == "text")


async def anthropic_sync(model, df):
    import anthropic
    client, sem = anthropic.AsyncAnthropic(max_retries=5), asyncio.Semaphore(CONCURRENCY)

    async def one(row):
        async with sem:
            try:
                t0 = time.perf_counter()
                m = await client.messages.create(**anthropic_params(model, row.text))
                ms = (time.perf_counter() - t0) * 1000
                if m.stop_reason == "refusal":
                    return {"id": row.id, "error": "refusal"}
                return {"id": row.id, **parse(anthropic_text(m)), "llm_ms": ms,
                        "tok_in": m.usage.input_tokens, "tok_out": m.usage.output_tokens,
                        "cost_usd": cost("anthropic", model, m.usage.input_tokens, m.usage.output_tokens, False)}
            except Exception as e:
                return {"id": row.id, "error": repr(e)}
    return await asyncio.gather(*(one(r) for r in df.itertuples()))


def anthropic_submit(model, df, tag):
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request
    client = anthropic.Anthropic()
    b = client.messages.batches.create(requests=[
        Request(custom_id=str(row.id), params=MessageCreateParamsNonStreaming(**anthropic_params(model, row.text)))
        for row in df.itertuples()])
    return b.id


def anthropic_collect(model, batch_id):
    import anthropic
    client = anthropic.Anthropic()
    b = client.messages.batches.retrieve(batch_id)
    if b.processing_status != "ended":
        sys.exit(f"batch {batch_id} still {b.processing_status} ({b.request_counts})")
    rows = []
    for r in client.messages.batches.results(batch_id):   # arbitrary order: custom_id is the key
        if r.result.type != "succeeded":
            rows.append({"id": int(r.custom_id), "error": r.result.type})
            continue
        m = r.result.message
        try:
            rows.append({"id": int(r.custom_id), **parse(anthropic_text(m)), "llm_ms": float("nan"),
                         "tok_in": m.usage.input_tokens, "tok_out": m.usage.output_tokens,
                         "cost_usd": cost("anthropic", model, m.usage.input_tokens, m.usage.output_tokens, True)})
        except Exception as e:
            rows.append({"id": int(r.custom_id), "error": repr(e)})
    return rows


# ---------------------------------------------------------------- CLI
def save(rows, model, sample):
    out = pd.DataFrame(rows)
    out.to_parquet(out_path(model, sample))
    ok = out.dropna(subset=["llm_product"]) if "llm_product" in out else out.iloc[0:0]
    print(f"{model} [{sample}]: {len(ok)}/{len(out)} ok | cost US$ {ok.get('cost_usd', pd.Series()).sum():.3f}"
          + (f" | p50 {ok.llm_ms.median():.0f}ms p95 {ok.llm_ms.quantile(.95):.0f}ms" if ok.llm_ms.notna().any() else ""))


def cmd_estimate():
    rows = []
    for (prov, model) in PRICES:
        rows.append({"provider": prov, "model": model,
                     "lat200 sync": estimate(prov, model, 200, False),
                     "s900 batch": estimate(prov, model, 900, True),
                     "full batch": estimate(prov, model, 1800, True),
                     "full sync": estimate(prov, model, 1800, False)})
    print(pd.DataFrame(rows).round(3).to_markdown(index=False))
    print("\nEstimate using tokens measured in the baseline (518 input / 30 output; Anthropic +20%; Opus 250 output).")
    import glob
    spent = sum(pd.read_parquet(f)["cost_usd"].sum() for f in glob.glob("results/llm_*.parquet"))
    print(f"\nAlready spent on APIs (sum of results/llm_*.parquet): US$ {spent:.2f} ≈ R$ {spent * BRL_PER_USD:.2f} "
          f"(exchange rate {BRL_PER_USD})")


def cmd_probe(prov, model):
    """3 synchronous calls to measure real token counts before spending on the batch."""
    assert prov == "openai", "only OpenAI in the current scope"
    if "--spend" not in sys.argv:
        sys.exit("Refused: probe makes 3 paid calls (~US$ 0.001). Run with --spend.")
    load_env_key()
    check_budget(0.01)
    from openai import OpenAI
    client = OpenAI()
    df = pd.read_parquet(SAMPLES["lat200"]).head(3)
    tin, tout, treason = [], [], []
    for row in df.itertuples():
        try:
            r = client.chat.completions.create(**openai_params(model, row.text))
        except Exception as e:
            sys.exit(f"{model} rejected the parameters ({e}). Try OPENAI_REASONING_EFFORT=minimal or low.")
        tin.append(r.usage.prompt_tokens)
        tout.append(r.usage.completion_tokens)
        det = getattr(r.usage, "completion_tokens_details", None)
        treason.append(getattr(det, "reasoning_tokens", 0) or 0)
        parse(r.choices[0].message.content)  # fails here if the output doesn't match the schema
    avg_in, avg_out = sum(tin) / 3, sum(tout) / 3
    per = cost("openai", model, avg_in, avg_out, False)
    os.makedirs(BATCH_DIR, exist_ok=True)
    json.dump({"model": model, "cost_usd": per * 3, "tok_in": avg_in, "tok_out": avg_out,
               "reasoning_tokens": sum(treason) / 3}, open(f"{BATCH_DIR}/probe_{model}.json", "w"), indent=2)
    print(f"{model}: input {avg_in:.0f} | output {avg_out:.0f} (reasoning {sum(treason)/3:.0f}) tokens/complaint")
    print(f"  estimated real cost: s900 batch US$ {per*900*BATCH_DISCOUNT:.3f} | full batch US$ {per*1800*BATCH_DISCOUNT:.3f} "
          f"| lat200 sync US$ {per*198:.3f}")


def main():
    args = [a for a in sys.argv[1:] if a != "--spend"]
    if not args or args[0] == "estimate":
        return cmd_estimate()
    if args[0] == "probe":
        return cmd_probe(args[1], args[2])
    mode, prov, model, sample = args[:4]
    assert (prov, model) in PRICES, f"model with no registered price: {prov}/{model}"
    tag = f"{model}__{sample}"
    os.makedirs(BATCH_DIR, exist_ok=True)
    state = f"{BATCH_DIR}/{tag}.json"

    if mode == "collect":
        if os.path.exists(out_path(model, sample)):
            sys.exit(f"{out_path(model, sample)} already exists")
        load_env_key()
        bid = json.load(open(state))["batch_id"]
        rows = openai_collect(model, bid) if prov == "openai" else anthropic_collect(model, bid)
        return save(rows, model, sample)

    df = pd.read_parquet(SAMPLES[sample])
    est = estimate(prov, model, len(df), batch=(mode == "submit"))
    if "--spend" not in sys.argv:
        sys.exit(f"Refused: paid call. Estimate {prov}/{model} [{sample}, {mode}]: US$ {est:.2f}. "
                 "Run with --spend to authorize.")
    if os.path.exists(out_path(model, sample)) or (mode == "submit" and os.path.exists(state)):
        sys.exit(f"{tag} already run/submitted — not paying again.")
    load_env_key()
    check_budget(est)
    print(f"{prov}/{model} [{sample}, {mode}] — estimate US$ {est:.2f}")

    if mode == "sync":
        rows = asyncio.run(openai_sync(model, df) if prov == "openai" else anthropic_sync(model, df))
        return save(rows, model, sample)
    if mode == "submit":
        bid = openai_submit(model, df, tag) if prov == "openai" else anthropic_submit(model, df, tag)
        json.dump({"batch_id": bid, "provider": prov, "model": model, "sample": sample, "n": len(df),
                   "estimate_usd": est, "submitted": time.strftime("%Y-%m-%d %H:%M")}, open(state, "w"), indent=2)
        print(f"batch submitted: {bid} — then: 13_llm_api.py collect {prov} {model} {sample}")


if __name__ == "__main__":
    main()

"""Controle do Laya: ModernBERT-large (a mesma base do Laya inglês) ajustado do jeito tradicional.

Um encoder, duas cabeças lineares (fila: 9 classes; fraude: 2 classes), cross-entropy somada,
mesmos 3.600 exemplos e mesma fatia de calibração (temperatura por cabeça) do 05_finetune.py.
Se empatar com o Laya fine-tuned, o ganho vem do encoder, não do método RLCD.

Uso: .venv/bin/python 08_encoder_finetune.py [modelo_hf]   (default: answerdotai/ModernBERT-large)
Grava results/sys_modernbert-ft.parquet e models/modernbert-cfpb/. Custo: zero (GPU local).
"""
import os, random, sys, time
import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

from common import PRODUCTS, MAX_CHARS
import sysio

MODEL_ID = sys.argv[1] if len(sys.argv) > 1 else "answerdotai/ModernBERT-large"
OUT_DIR = "models/smoke-modernbert" if sysio.SMOKE else "models/modernbert-cfpb"
SEED = 20260929
MAX_LEN = 512           # mesmo teto de tokens do Laya (que ainda gasta parte com pergunta e opções)
EPOCHS, MICRO_BATCH, GRAD_ACCUM = (1 if sysio.SMOKE else 3), 8, 4   # batch efetivo 32
LR_ENCODER, LR_HEAD = 3e-5, 1e-3
CALIB_N = 20 if sysio.SMOKE else 200           # reclamações fora do treino para calibração (igual ao Laya: 400 itens / 2 perguntas)
QUEUES = list(PRODUCTS)


class TwoHeads(nn.Module):
    def __init__(self, model_id):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_id)
        h = self.encoder.config.hidden_size
        self.product = nn.Linear(h, len(QUEUES))
        self.fraud = nn.Linear(h, 2)

    def forward(self, input_ids, attention_mask):
        x = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).to(x.dtype)
        pooled = (x * m).sum(1) / m.sum(1).clamp(min=1)   # mean pooling
        return self.product(pooled), self.fraud(pooled)


def batches(df, tok, bs, shuffle):
    idx = np.arange(len(df))
    if shuffle:
        np.random.shuffle(idx)
    for i in range(0, len(idx), bs):
        part = df.iloc[idx[i:i + bs]]
        enc = tok([t[:MAX_CHARS] for t in part.text], truncation=True, max_length=MAX_LEN,
                  padding=True, return_tensors="pt")
        yield enc, part


def fit_temperature(logits, labels):
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)
    ce = nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        loss = ce(logits / log_t.exp(), labels)
        loss.backward()
        return loss
    opt.step(closure)
    return float(log_t.exp().clamp(0.1, 10).item())


@torch.no_grad()
def predict(model, tok, df, device, bs=32):
    model.eval()
    lp, lf = [], []
    for enc, _ in batches(df, tok, bs, shuffle=False):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p, f = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
        lp.append(p.float().cpu()); lf.append(f.float().cpu())
    return torch.cat(lp), torch.cat(lf)


def main():
    assert torch.cuda.is_available(), "sem CUDA — ver RUNBOOK.md (nvidia_uvm)"
    device = torch.device("cuda")
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    train = sysio.load_train()
    test = sysio.load_test()
    calib = train.sample(n=CALIB_N, random_state=SEED)
    fit = train[~train.id.isin(calib.id)]
    qidx = {q: i for i, q in enumerate(QUEUES)}

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = TwoHeads(MODEL_ID).to(device)
    model.encoder.gradient_checkpointing_enable()
    opt = torch.optim.AdamW([{"params": model.encoder.parameters(), "lr": LR_ENCODER},
                             {"params": list(model.product.parameters()) + list(model.fraud.parameters()),
                              "lr": LR_HEAD}], weight_decay=0.01)
    steps = EPOCHS * int(np.ceil(len(fit) / MICRO_BATCH / GRAD_ACCUM))
    sched = get_cosine_schedule_with_warmup(opt, int(0.06 * steps), steps)
    ce = nn.CrossEntropyLoss()

    t0 = time.time()
    for epoch in range(EPOCHS):
        model.train()
        opt.zero_grad(set_to_none=True)
        for step, (enc, part) in enumerate(batches(fit, tok, MICRO_BATCH, shuffle=True), 1):
            yp = torch.tensor([qidx[q] for q in part.queue], device=device)
            yf = torch.tensor(part.fraud.astype(int).values, device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p, f = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
            loss = (ce(p.float(), yp) + ce(f.float(), yf)) / GRAD_ACCUM
            loss.backward()
            if step % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            if step % 100 == 0:
                print(f"  época {epoch+1} passo {step} loss {loss.item()*GRAD_ACCUM:.4f}", flush=True)
        print(f"=== época {epoch+1} ok em {time.time()-t0:.0f}s", flush=True)

    # calibração de temperatura por cabeça, na fatia que não treinou
    cp, cf = predict(model, tok, calib, device)
    t_prod = fit_temperature(cp, torch.tensor([qidx[q] for q in calib.queue]))
    t_fraud = fit_temperature(cf, torch.tensor(calib.fraud.astype(int).values))
    print(f"temperaturas: fila {t_prod:.3f} | fraude {t_fraud:.3f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "t_prod": t_prod, "t_fraud": t_fraud,
                "queues": QUEUES, "base": MODEL_ID}, os.path.join(OUT_DIR, "model.pt"))

    # avaliação no teste
    texts = list(test.text)

    def one(t):
        enc = tok(t[:MAX_CHARS], truncation=True, max_length=MAX_LEN, return_tensors="pt")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
        torch.cuda.synchronize()

    model.eval()
    single = sysio.time_single(one, texts)
    t1 = time.perf_counter()
    lp, lf = predict(model, tok, test, device)
    batch_ms = (time.perf_counter() - t1) * 1000 / len(test)
    pp = torch.softmax(lp / t_prod, -1)
    pf = torch.softmax(lf / t_fraud, -1)[:, 1]
    meta = sysio.save("modernbert-ft", test.id, [QUEUES[i] for i in pp.argmax(-1)], pp.max(-1).values.numpy(),
                      pf.numpy(), "cuda", single, batch_ms, notes=f"{MODEL_ID}, CE multitarefa, {EPOCHS} épocas")
    acc = (np.array([QUEUES[i] for i in pp.argmax(-1)]) == test.queue.values).mean()
    print(f"modernbert-ft: acc fila {acc:.3f} | p50 {meta['single_p50_ms']:.0f}ms | lote {batch_ms:.1f}ms")


if __name__ == "__main__":
    main()

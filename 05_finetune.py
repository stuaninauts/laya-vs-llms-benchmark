"""Laya fine-tuning (English, 421M) on complaint triage — 1 GPU (RTX 3060 12GB).

Adapted from the official notebook (laya_finetune_ref.ipynb, 2xT4 DDP) to a single GPU:
same RLCD objective (policy gradient with a proper scoring rule + soft cross-entropy),
same temperature calibration on a held-out slice, no DDP.

Usage: .venv/bin/python 05_finetune.py [output_dir]   (default: models/laya-cfpb)
Cost: zero (local GPU). Expected time: tens of minutes.
"""
import json, os, random, sys, time
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer
from laya.agent import _fix_tokenizer_config
from laya.common import build_model, build_sequence, proper_reward, render_options, QTYPES

from common import QUESTIONS, MAX_CHARS, PRODUCTS
import sysio

OUTPUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "models/laya-cfpb"
MODEL_ID = "convaiinnovations/laya"
SEED = 20260929

EPOCHS = 1 if sysio.SMOKE else 3
MICRO_BATCH = 8       # sequences per forward pass (12GB with gradient checkpointing + fp16)
GRAD_ACCUM = 8        # effective batch = 64, same as the notebook (8 * 2 GPUs * 4)
GROUP_SIZE = 4
LR_ENCODER, LR_HEAD = 2.5e-5, 1.0e-4
SIGMA_START, SIGMA_END = 0.4, 0.1
CALIB_MAX = 400


def targets(row):
    """(question, target) for the two decisions of a complaint."""
    prod = [1.0 if k == row.queue else 0.0 for k in PRODUCTS]
    fraud = [0.0, 1.0] if row.fraud else [1.0, 0.0]  # criteria order: false, true
    return [("product", prod), ("fraud", fraud)]


def build_items(df, tok, cfg):
    items = []
    for row in df.itertuples():
        state = row.text[:MAX_CHARS]
        for qid, target in targets(row):
            q = QUESTIONS[qid]
            iq = {"t": q["type"], "ins": q["instructions"], "crit": q["criteria"]}
            seq, markers = build_sequence(tok, state, iq, cfg["max_len"], cfg["head_max_len"])
            if len(markers) != len(render_options(iq)):
                continue
            items.append({"ids": seq, "markers": markers, "qtype": QTYPES[q["type"]],
                          "target": target, "label": target.index(max(target))})
    return items


def collate(items, pad_id):
    n, L = len(items), max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    target = torch.zeros((n, kmax), dtype=torch.float32)
    for i, it in enumerate(items):
        ids[i, :len(it["ids"])] = torch.tensor(it["ids"])
        att[i, :len(it["ids"])] = 1
        k = len(it["markers"])
        mpos[i, :k] = torch.tensor(it["markers"])
        mmask[i, :k] = True
        target[i, :k] = torch.tensor(it["target"], dtype=torch.float32)
    return {"input_ids": ids, "attention_mask": att, "marker_pos": mpos, "marker_mask": mmask,
            "target": target, "qtype": torch.tensor([it["qtype"] for it in items])}


def fit_one_temp(sel):
    if len(sel) < 10:
        return 1.0
    kmax = max(len(z) for z, _ in sel)
    Z = torch.full((len(sel), kmax), -1e4)
    T = torch.zeros((len(sel), kmax))
    for i, (z, t) in enumerate(sel):
        Z[i, :len(z)] = torch.tensor(z)
        T[i, :len(t)] = torch.tensor(t, dtype=torch.float32)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = -(T * torch.log_softmax(Z / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss
    opt.step(closure)
    return float(torch.clamp(log_t.exp(), 0.1, 10.0).item())


def forward(model, b, device):
    with torch.autocast("cuda", dtype=torch.float16):
        logits, act = model(b["input_ids"].to(device), b["attention_mask"].to(device),
                            b["marker_pos"].to(device), b["marker_mask"].to(device), b["qtype"].to(device))
    return logits.float(), act


def main():
    assert torch.cuda.is_available(), "no CUDA — see Troubleshooting in README.md"
    device = torch.device("cuda")
    random.seed(SEED)
    torch.manual_seed(SEED)

    model_dir = snapshot_download(MODEL_ID)
    _fix_tokenizer_config(model_dir)
    with open(os.path.join(model_dir, "rl_agent_config.json")) as f:
        cfg = json.load(f)
    cfg["gradient_checkpointing"] = True
    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))

    train = sysio.load_train()
    # calibration split PER COMPLAINT (not per item) so the two decisions for the same
    # text never land one in training and the other in calibration
    calib_ids = set(train.sample(n=min(CALIB_MAX // 2, len(train) // 10), random_state=SEED).id)
    train_items = build_items(train[~train.id.isin(calib_ids)], tok, cfg)
    calib_items = build_items(train[train.id.isin(calib_ids)], tok, cfg)
    print(f"{len(train_items)} training items | {len(calib_items)} calibration items")

    model = build_model(cfg, encoder_dir=os.path.join(model_dir, "encoder"))
    model.load_state_dict(load_file(os.path.join(model_dir, "model.safetensors")), strict=True)
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.head_checkpointing = True
    model.to(device).train()

    enc = [p for n, p in model.named_parameters() if "encoder." in n]
    head = [p for n, p in model.named_parameters() if "encoder." not in n]
    opt = torch.optim.AdamW([{"params": enc, "lr": LR_ENCODER}, {"params": head, "lr": LR_HEAD}],
                            weight_decay=0.01)
    total = max(1, len(train_items) // (MICRO_BATCH * GRAD_ACCUM) * EPOCHS)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda")

    t0 = time.time()
    for epoch in range(EPOCHS):
        random.shuffle(train_items)
        sigma = SIGMA_START + (SIGMA_END - SIGMA_START) * epoch / max(1, EPOCHS - 1)
        opt.zero_grad(set_to_none=True)
        ep_loss, nb = 0.0, 0
        for step, i in enumerate(range(0, len(train_items), MICRO_BATCH), 1):
            b = collate(train_items[i:i + MICRO_BATCH], tok.pad_token_id)
            logits, act = forward(model, b, device)
            mask = b["marker_mask"].to(device)
            k = mask.sum(-1, keepdim=True).float()
            target = b["target"].to(device)
            qtype = b["qtype"].to(device)

            eps = torch.randn((GROUP_SIZE,) + logits.shape, device=device) * sigma * mask
            eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
            z = logits.detach().unsqueeze(0) + eps
            q = torch.softmax(z.masked_fill(~mask, -1e4), -1)
            with torch.no_grad():
                r = proper_reward(q, target.unsqueeze(0), qtype, mask, w_sph=0.75, w_rps=1.0)
                adv = (r - r.mean(0, keepdim=True)) / ((r - r.mean(0, keepdim=True)).std() + 1e-6)
            logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
            loss_rl = -(adv * logp).mean()
            loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
            loss = (loss_rl + loss_ce) / GRAD_ACCUM + 0.0 * act.sum()
            scaler.scale(loss).backward()

            if step % GRAD_ACCUM == 0 or i + MICRO_BATCH >= len(train_items):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad(set_to_none=True)
            ep_loss += loss.item() * GRAD_ACCUM
            nb += 1
            if nb % 50 == 0:
                print(f"  epoch {epoch+1}/{EPOCHS} step {nb} loss {loss.item()*GRAD_ACCUM:.4f} "
                      f"reward {r.mean().item():.3f}", flush=True)
        print(f"=== epoch {epoch+1} done in {time.time()-t0:.0f}s | mean loss {ep_loss/max(1,nb):.4f}", flush=True)

    # temperature calibration on the held-out slice
    model.eval()
    del opt, scaler, sched
    torch.cuda.empty_cache()
    preds = []
    with torch.no_grad():
        for i in range(0, len(calib_items), 16):
            chunk = calib_items[i:i + 16]
            logits, _ = forward(model, collate(chunk, tok.pad_token_id), device)
            ln = logits.cpu().numpy()
            for r_, it in enumerate(chunk):
                preds.append((it["qtype"], ln[r_, :len(it["markers"])], it["target"]))
    temps = [1.2, 1.2, 1.2]
    for qt in range(3):
        sel = [(z, t) for q_, z, t in preds if q_ == qt]
        if sel:
            temps[qt] = fit_one_temp(sel)
    print("temperatures (choice, score, noul):", [round(t, 3) for t in temps])

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_file({k: v.half().contiguous().cpu() for k, v in model.state_dict().items()},
              os.path.join(OUTPUT_DIR, "model.safetensors"))
    model.encoder.config.save_pretrained(os.path.join(OUTPUT_DIR, "encoder"))
    tok.save_pretrained(os.path.join(OUTPUT_DIR, "tokenizer"))
    cfg.update({"fine_tuned": True, "model_name": "laya-cfpb-triage", "temperature": temps})
    cfg.pop("temperature_by_options", None)
    with open(os.path.join(OUTPUT_DIR, "rl_agent_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"saved to {OUTPUT_DIR} | total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
# Local pipeline (zero API cost). The GPT baseline is already in results/ and does not run here.
# On the 12GB GPU, jobs run ONE AT A TIME (measured: Laya batch inference ~5GB,
# fine-tuning ~9GB — together they run out of memory). Only CPU jobs run in parallel.
#   phase 1: TF-IDF (CPU) ‖ [Laya zero-shot → Laya fine-tuning]
#   phase 2: fine-tuned Laya → ModernBERT fine-tuning + evaluation
#   phase 3: Qwen3 embeddings → Granite → local LLMs on Ollama
#   end:     06_analyze.py
# Usage: ./run.sh            (all phases)
#        ./run.sh 2 3        (only the given phases)
set -uo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
export USE_TF=0 TF_CPP_MIN_LOG_LEVEL=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # the venv sees the system TensorFlow; without this, transformers breaks
LLM_LOCAL=${LLM_LOCAL:-"qwen3.5:9b gemma4:12b-it-qat qwen3.5:4b"}   # run sequentially (~35 min each)
mkdir -p results models logs
PHASES=${*:-1 2 3}

$PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 'no CUDA: see Troubleshooting in README.md')" || exit 1

run() {  # run <log> <command...>  — in the background, records the PID
  local log=$1; shift
  ( "$@" ) > "logs/$log.log" 2>&1 &
  PIDS+=("$!:$log")
}
wait_all() {
  for p in "${PIDS[@]}"; do
    wait "${p%%:*}" && echo "  ok     ${p#*:}" || echo "  FAILED ${p#*:} (logs/${p#*:}.log)"
  done
  PIDS=()
}

PIDS=()
for f in $PHASES; do
  echo "== phase $f"
  case $f in
    1) run tfidf $PY 11_tfidf.py
       run laya-phase1 bash -c "$PY 03_laya_eval.py zeroshot > logs/laya-zeroshot.log 2>&1; \
                               $PY 05_finetune.py models/laya-cfpb > logs/laya-finetune.log 2>&1" ;;
    2) run gpu-phase2 bash -c "$PY 03_laya_eval.py finetuned models/laya-cfpb > logs/laya-finetuned-eval.log 2>&1; \
                              $PY 08_encoder_finetune.py > logs/modernbert.log 2>&1" ;;
    3) run gpu-phase3 bash -c "$PY 09_embeddings_logreg.py Qwen/Qwen3-Embedding-0.6B qwen3emb; \
                              $PY 09_embeddings_logreg.py ibm-granite/granite-embedding-small-english-r2 granite-r2; \
                              for m in $LLM_LOCAL; do $PY 10_local_llm.py \$m || echo FAILED \$m; done" ;;
  esac
  wait_all
done

$PY 06_analyze.py

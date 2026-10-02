#!/usr/bin/env bash
# Pipeline local (custo zero de API). O baseline GPT já está em results/ e não roda aqui.
# Na GPU de 12GB os trabalhos rodam UM DE CADA VEZ (medido: inferência do Laya em lote ~5GB,
# fine-tuning ~9GB — juntos estouram). Só o que é CPU roda em paralelo.
#   fase 1: TF-IDF (CPU) ‖ [Laya zero-shot → fine-tuning do Laya]
#   fase 2: Laya fine-tuned → ModernBERT fine-tuning + avaliação
#   fase 3: embeddings Qwen3 → Granite → LLMs locais no Ollama
#   fim:    06_analyze.py
# Uso: ./run.sh            (todas as fases)
#      ./run.sh 2 3        (só as fases indicadas)
set -uo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
export USE_TF=0 TF_CPP_MIN_LOG_LEVEL=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # o venv enxerga o TensorFlow do sistema; sem isto o transformers quebra
LLM_LOCAL=${LLM_LOCAL:-"qwen3.5:9b gemma4:12b-it-qat qwen3.5:4b"}   # rodam em sequência (~35 min cada)
mkdir -p results models logs
PHASES=${*:-1 2 3}

$PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 'sem CUDA: ver RUNBOOK.md')" || exit 1

run() {  # run <log> <comando...>  — em segundo plano, registra o PID
  local log=$1; shift
  ( "$@" ) > "logs/$log.log" 2>&1 &
  PIDS+=("$!:$log")
}
wait_all() {
  for p in "${PIDS[@]}"; do
    wait "${p%%:*}" && echo "  ok     ${p#*:}" || echo "  FALHOU ${p#*:} (logs/${p#*:}.log)"
  done
  PIDS=()
}

PIDS=()
for f in $PHASES; do
  echo "== fase $f"
  case $f in
    1) run tfidf $PY 11_tfidf.py
       run laya-fase1 bash -c "$PY 03_laya_eval.py zeroshot > logs/laya-zeroshot.log 2>&1; \
                               $PY 05_finetune.py models/laya-cfpb > logs/laya-finetune.log 2>&1" ;;
    2) run gpu-fase2 bash -c "$PY 03_laya_eval.py finetuned models/laya-cfpb > logs/laya-finetuned-eval.log 2>&1; \
                              $PY 08_encoder_finetune.py > logs/modernbert.log 2>&1" ;;
    3) run gpu-fase3 bash -c "$PY 09_embeddings_logreg.py Qwen/Qwen3-Embedding-0.6B qwen3emb; \
                              $PY 09_embeddings_logreg.py ibm-granite/granite-embedding-small-english-r2 granite-r2; \
                              for m in $LLM_LOCAL; do $PY 10_local_llm.py \$m || echo FALHOU \$m; done" ;;
  esac
  wait_all
done

$PY 06_analyze.py

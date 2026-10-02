#!/usr/bin/env bash
# API plan (OpenAI only), step by step. Each step shows the estimate and asks for confirmation.
# Accuracy via the Batch API (-50%); latency from a few synchronous calls (lat200 = 198).
# Target budget for the whole experiment: R$ 10–20 (already spent: ~R$ 6.30 on the nano/mini baseline).
#
#   ./14_api_plan.sh 1   luna + terra probe (6 calls, ~US$ 0.002) — confirms real token counts
#   ./14_api_plan.sh 2   luna: batch on the full test set (~US$ 0.13)
#   ./14_api_plan.sh 3   luna: synchronous latency on lat200 (~US$ 0.03)
#   ./14_api_plan.sh 4   terra (ceiling): batch on s900 (~US$ 0.63)
#   ./14_api_plan.sh 5   terra: synchronous latency on lat200 (~US$ 0.28)
#   ./14_api_plan.sh collect   downloads the batches that have finished (free)
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
ok() { read -r -p "$1 — confirm spend? [y/N] " a; [ "$a" = "y" ]; }

case "${1:-}" in
  1) ok "luna+terra probe ~US\$ 0.002" && { $PY 13_llm_api.py probe openai gpt-5.6-luna --spend; $PY 13_llm_api.py probe openai gpt-5.6-terra --spend; } ;;
  2) ok "luna full batch ~US\$ 0.13"      && $PY 13_llm_api.py submit openai gpt-5.6-luna full --spend ;;
  3) ok "luna lat200 sync ~US\$ 0.03"     && $PY 13_llm_api.py sync   openai gpt-5.6-luna lat200 --spend ;;
  4) ok "terra s900 batch ~US\$ 0.63"     && $PY 13_llm_api.py submit openai gpt-5.6-terra s900 --spend ;;
  5) ok "terra lat200 sync ~US\$ 0.28"    && $PY 13_llm_api.py sync   openai gpt-5.6-terra lat200 --spend ;;
  collect)
     for t in "gpt-5.6-luna full" "gpt-5.6-terra s900"; do set -- $t
       [ -f "results/batches/$1__$2.json" ] && [ ! -f "results/llm_$1__$2.parquet" ] && $PY 13_llm_api.py collect openai $1 $2 || true
     done ;;
  *) sed -n 2,11p "$0" ;;
esac
$PY 13_llm_api.py estimate | tail -1

#!/usr/bin/env bash
# Plano de API (só OpenAI), por partes. Cada passo mostra a estimativa e pede confirmação.
# Acurácia via Batch API (-50%); latência com poucas chamadas síncronas (lat200 = 198).
# Orçamento-alvo do experimento inteiro: R$ 10–20 (já gasto: ~R$ 6,30 no baseline nano/mini).
#
#   ./14_api_plan.sh 1   sondagem luna + terra (6 chamadas, ~US$ 0,002) — confirma tokens reais
#   ./14_api_plan.sh 2   luna: batch no teste completo (~US$ 0,13)
#   ./14_api_plan.sh 3   luna: latência síncrona lat200 (~US$ 0,03)
#   ./14_api_plan.sh 4   terra (teto): batch na s900 (~US$ 0,63)
#   ./14_api_plan.sh 5   terra: latência síncrona lat200 (~US$ 0,28)
#   ./14_api_plan.sh collect   baixa os batches que já terminaram (grátis)
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
ok() { read -r -p "$1 — confirmar gasto? [s/N] " a; [ "$a" = "s" ]; }

case "${1:-}" in
  1) ok "sondagem luna+terra ~US\$ 0,002" && { $PY 13_llm_api.py probe openai gpt-5.6-luna --spend; $PY 13_llm_api.py probe openai gpt-5.6-terra --spend; } ;;
  2) ok "luna full batch ~US\$ 0,13"      && $PY 13_llm_api.py submit openai gpt-5.6-luna full --spend ;;
  3) ok "luna lat200 sync ~US\$ 0,03"     && $PY 13_llm_api.py sync   openai gpt-5.6-luna lat200 --spend ;;
  4) ok "terra s900 batch ~US\$ 0,63"     && $PY 13_llm_api.py submit openai gpt-5.6-terra s900 --spend ;;
  5) ok "terra lat200 sync ~US\$ 0,28"    && $PY 13_llm_api.py sync   openai gpt-5.6-terra lat200 --spend ;;
  collect)
     for t in "gpt-5.6-luna full" "gpt-5.6-terra s900"; do set -- $t
       [ -f "results/batches/$1__$2.json" ] && [ ! -f "results/llm_$1__$2.parquet" ] && $PY 13_llm_api.py collect openai $1 $2 || true
     done ;;
  *) sed -n 2,11p "$0" ;;
esac
$PY 13_llm_api.py estimate | tail -1

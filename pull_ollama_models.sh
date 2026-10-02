#!/usr/bin/env bash
# Downloads the experiment's local LLMs, resumably: `ollama pull` continues where it stopped, and this
# script retries with increasing waits if the connection drops. Can be interrupted and rerun.
# Requires Ollama >= 0.12.11 (logprobs). Usage: ./pull_ollama_models.sh [model ...]
set -u
MODELS=${*:-qwen3.5:9b gemma4:12b-it-qat qwen3.5:4b}

v=$(ollama --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
if [ -z "$v" ] || [ "$(printf '%s\n0.12.11\n' "$v" | sort -V | head -1)" != "0.12.11" ]; then
  echo "Ollama $v is too old (needs >= 0.12.11): curl -fsSL https://ollama.com/install.sh | sh"; exit 1
fi

for m in $MODELS; do
  wait=10
  until ollama pull "$m"; do
    echo "RETRY $m in ${wait}s"; sleep "$wait"; wait=$(( wait < 300 ? wait * 2 : 300 ))
  done
  echo "OK $m"
done
ollama list

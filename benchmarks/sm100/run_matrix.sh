#!/usr/bin/env bash
set -euo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ $# -lt 1 || $# -gt 2 || $1 == --help ]]; then
  printf 'Usage: %s RESULTS_DIR [weft|megatron|sglang|all]\n' "$0"
  exit 2
fi
results=$1
frameworks=${2:-all}
case "$frameworks" in
  weft|megatron|sglang|all) ;;
  *) printf 'Unknown framework: %s\n' "$frameworks" >&2; exit 2 ;;
esac
dispatcher=${MEGATRON_DISPATCHER:-alltoall}
if [[ $frameworks != weft && $frameworks != sglang && $dispatcher != alltoall && $dispatcher != flex ]]; then
  printf 'Unknown Megatron dispatcher: %s\n' "$dispatcher" >&2
  exit 2
fi
mkdir -p -- "$results"
node_rank=${NODE_RANK:-0}
port=${MASTER_PORT:-29531}
if [[ ${NNODES:-1} == 1 ]]; then
  node_rank=0
fi

run() {
  local framework=$1 model=$2 precision=$3 mode=$4
  shift 4
  local log="$results/$framework.$model.$precision.$mode.node$node_rank.log"
  printf 'Running %s\n' "$log"
  MASTER_PORT=$port "$here/run.sh" "$framework" --model "$model" --precision "$precision" "$@" > "$log" 2>&1
  port=$((port + 1))
}

for model in qwen35b qwen122b flash; do
  if [[ $frameworks == weft ]]; then
    for precision in bf16 fp4; do
      ref=()
      if [[ $precision == fp4 && ${NNODES:-1} == 1 ]]; then
        ref=(--mega-mixed)
      fi
      run weft "$model" "$precision" decode --mode decode "${ref[@]}"
      run weft "$model" "$precision" prefill --mode prefill "${ref[@]}"
      run weft-training "$model" "$precision" forward --forward-only --samples 30
      run weft-training "$model" "$precision" backward --backward-only --samples 30
      run weft-training "$model" "$precision" cycle --samples 30
    done
    continue
  fi
  for precision in bf16 fp4; do
    if [[ $frameworks != sglang ]]; then
      run megatron "$model" "$precision" cycle --dispatcher "$dispatcher" --samples 30
      run megatron "$model" "$precision" forward --dispatcher "$dispatcher" --samples 30 --forward-only
      run megatron "$model" "$precision" backward --dispatcher "$dispatcher" --samples 30 --backward-only
    fi
    if [[ $frameworks != megatron ]]; then
      run sglang "$model" "$precision" decode
    fi
  done
done

#!/usr/bin/env bash
set -euo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
extra=()
case "${1:-}" in
  weft) entry=torchrun_forward.py ;;
  weft-tile-m) entry=torchrun_tile_m.py ;;
  weft-gin) entry=torchrun_gin.py ;;
  weft-training) entry=torchrun_training.py; extra=(--backend weft) ;;
  megatron) entry=torchrun_training.py ;;
  sglang) entry=torchrun_sglang_decode.py ;;
  *)
    printf 'Usage: %s {weft|weft-training|weft-tile-m|weft-gin|megatron|sglang} [benchmark arguments]\n' "$0" >&2
    exit 2
    ;;
esac
backend=$1
shift

nnodes=${NNODES:-1}
nproc=${NPROC_PER_NODE:-8}
if [[ $nnodes == 1 ]]; then
  launch=(--standalone)
else
  if [[ $backend == weft* && -e /sys/class/infiniband/mlx5_bond_7 ]]; then
    # B300 RoCE nodes expose each pair of 400G ports as one mlx5 bond.
    # GDAKI uses the private RoCE entropy fix when installed. The stock
    # NCCL IB Proxy backend also stripes reliably with independent verbs QPs.
    export NCCL_CUMEM_ENABLE=${NCCL_CUMEM_ENABLE:-1}
    export NCCL_IB_MERGE_NICS=${NCCL_IB_MERGE_NICS:-0}
    gdaki_prefix=${WEFTMOE_GDAKI_NCCL_PREFIX:-"$here/../../build/sm100-b300/nccl-gdaki"}
    if [[ -f $gdaki_prefix/lib/libnccl.so.2 && ${NCCL_GIN_TYPE:-3} == 3 ]]; then
      if [[ ${LD_PRELOAD:-} != *libnccl.so* ]]; then
        export LD_PRELOAD="$gdaki_prefix/lib/libnccl.so.2${LD_PRELOAD:+:$LD_PRELOAD}"
      fi
      export NCCL_GIN_TYPE=3
    else
      export NCCL_GIN_TYPE=${NCCL_GIN_TYPE:-2}
    fi
    export NCCL_GIN_NCONNECTIONS=${NCCL_GIN_NCONNECTIONS:-4}
    export NCCL_IB_QPS_PER_CONNECTION=${NCCL_IB_QPS_PER_CONNECTION:-4}
    export NCCL_IB_SPLIT_DATA_ON_QPS=${NCCL_IB_SPLIT_DATA_ON_QPS:-1}
    if [[ -z ${NCCL_IB_HCA:-} ]]; then
      hcas=(/sys/class/infiniband/mlx5_bond_*)
      hcas=("${hcas[@]##*/}")
      export NCCL_IB_HCA="=$(IFS=,; echo "${hcas[*]/%/:1}")"
    fi
  fi
  launch=(--nnodes="$nnodes" --node-rank="${NODE_RANK:?Set NODE_RANK on each node}"
          --master-addr="${MASTER_ADDR:?Set MASTER_ADDR to node 0}"
          --master-port="${MASTER_PORT:-29531}")
fi

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
exec "${PYTHON:-python3}" -m torch.distributed.run "${launch[@]}" \
  --nproc-per-node="$nproc" "$here/$entry" "${extra[@]}" "$@"

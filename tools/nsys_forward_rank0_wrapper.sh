#!/usr/bin/env bash
set -euo pipefail

readonly script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly repo_root="$(cd -- "${script_dir}/.." && pwd)"
readonly real_binary="${MEGAMOE_NSYS_REAL_BINARY:-${repo_root}/build/test_cuda_notify}"
rank=-1
bootstrap_fd=-1
previous=
for argument in "$@"; do
  if [[ "${previous}" == "--rank" ]]; then
    rank="${argument}"
  elif [[ "${previous}" == "--bootstrap-output-fd" ]]; then
    bootstrap_fd="${argument}"
  fi
  previous="${argument}"
done

# The distributed launcher invokes this wrapper for every local rank.  Trace
# rank 0 only: profiling all eight processes multiplies overhead and makes all
# of them race on one report path.
if [[ "${rank}" != "0" ]]; then
  exec "${real_binary}" "$@"
fi

# The distributed launcher deliberately waits for EOF after rank 0 writes its
# 128-byte NCCL bootstrap frame.  Nsight otherwise inherits that pipe and keeps
# it open until report conversion finishes, so no peer rank is ever launched.
# Relay only the fixed frame through a FIFO, then close the launcher's original
# descriptor before starting Nsight.  The profiled child receives the FIFO on
# the same numeric descriptor named in its argv.
if [[ ! "${bootstrap_fd}" =~ ^[0-9]+$ ]] || (( bootstrap_fd < 3 )); then
  echo "invalid --bootstrap-output-fd for Nsight wrapper: ${bootstrap_fd}" >&2
  exit 2
fi
readonly bootstrap_fd_value="${bootstrap_fd}"
readonly relay_dir="$(mktemp -d /tmp/megamoe-nsys-bootstrap.XXXXXX)"
readonly relay_fifo="${relay_dir}/frame"
mkfifo "${relay_fifo}"
cleanup_relay() {
  rm -f -- "${relay_fifo}"
  rmdir -- "${relay_dir}" 2>/dev/null || true
}
trap cleanup_relay EXIT
dd if="${relay_fifo}" bs=128 count=1 iflag=fullblock status=none \
  >&"${bootstrap_fd_value}" &
relay_pid=$!
eval "exec ${bootstrap_fd_value}>&-"

readonly report_prefix="${MEGAMOE_NSYS_OUTPUT:-${repo_root}/artifacts/forward_rank0_current}"
readonly profiler_log="${report_prefix}.nsys.log"
set +e
/opt/nvidia/nsight-compute/2025.2.1/host/target-linux-x64/nsys profile \
  --trace=cuda,nvtx \
  --cuda-event-trace=false \
  --sample=none \
  --gpu-metrics-devices=0 \
  --gpu-metrics-set=gh100-ct \
  --gpu-metrics-frequency=5000 \
  --force-overwrite=true \
  --export=none \
  --output="${report_prefix}" \
  /usr/bin/env bash -c \
    'fd=$1; fifo=$2; binary=$3; shift 3; eval "exec ${fd}>\"${fifo}\""; exec "${binary}" "$@"' \
    megamoe-nsys-target "${bootstrap_fd_value}" "${relay_fifo}" \
    "${real_binary}" "$@" 2>"${profiler_log}"
profile_status=$?
set -e
kill "${relay_pid}" 2>/dev/null || true
wait "${relay_pid}" 2>/dev/null || true
exit "${profile_status}"

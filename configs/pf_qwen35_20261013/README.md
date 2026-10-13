# Qwen3.5-35B-A3B PF training tune result

Measured on `main@bb444c2`, NVIDIA L20X, SM clock locked to 1620 MHz.
Routed MoE only: BF16, H2048/I512/E256/top-k8, DP-layout inputs, TP/ETP1.
EP8 is one eight-GPU node; EP16 is two eight-GPU nodes. This is not a
full-model or inference-only performance claim.

## Selected configuration

Both topologies selected compiled ID `eab65a9995ff`, config ID `481cffe7fbc7`:

- FWD: M128/N256/K64, three stages, cluster-N2.
- BWD dU/dX: M128/N256/K64, three stages, cluster-N2.
- BWD dW1/dW2: M128/N256/K64, three stages; **chunk 1/1**, previously 4/1.
- Ring: B2048/D2. Network lanes 132, dispatch/return packet rows 32/128.
- Required measured environment: `WEFTMOE_PF_MIN_RING_DEPTH=2`,
  `WEFTMOE_PF_CREDIT_WINDOWS=2`.

FWD/activation tile and pipeline changes did not pass the 1% improvement
threshold. This is the measured winner of a bounded staged search, not a
global optimum. No production kernel or other model/DT default was changed.
The generated header freezes the combined parameter format, including legacy
compatibility fields; those fields are not all active search dimensions.

## Confirmed full-cycle performance

Random token counts (10% normal variation) and biased random routes, seed 123.
Each run primes all 20 steps, then uses 4 warmups and 16 timed samples. The
table averages two reversed-order confirmation runs, each reporting rank-max
p50. Timing covers eager cached FWD+BWD+release+synchronize, not the sum of
separately measured FWD and BWD medians. Fixed pool/IPC initialization and input
generation are excluded; per-call weight preparation/cache allocations remain.

| Topology | Nominal tokens/rank | Original (ms) | Tuned (ms) | Latency reduction |
| --- | ---: | ---: | ---: | ---: |
| EP8 | 8192 | 10.958 | 10.639 | 2.91% |
| EP8 | 20480 | 32.123 | 30.609 | 4.71% |
| EP16 | 8192 | 15.007 | 14.881 | 0.84% |
| EP16 | 20480 | 33.317 | 31.539 | 5.34% |

These are the dW confirmation pairs; subsequent ring searches retained the
same configuration. Joint 8K/20K geometric-mean latency reductions were
3.81% (EP8) and 3.12% (EP16). See the
[full record](../../docs/pf_backward_20261012.md) and
[machine-readable results](../../benchmarks/results/qwen35_pf_autotune_20261013.json)
for candidate scores, independent FWD/BWD checks and baseline comparisons.

The independent nonuniform-route BWD checks improved EP8 8K/20K latency by
5.0%/7.5%, and EP16 20K by 7.7%; EP16 8K and FWD were effectively unchanged.
This is not a universal win: EP8 balanced-route 8K BWD regressed 2.7% in two
reversed-order runs; EP16 balanced 20K was 1.5% slower in one run. The selected
profile targets the nonuniform training trace. Single-node 20K still trails
the previously measured Megatron baseline.

## Build and load

Use the repository's CUDA/CUTLASS/NCCL build environment:

```bash
export NVCCFLAGS="-lineinfo -DWEFTMOE_FORWARD_USE_GENERATED_TUNING=1 -I${PWD}/configs/pf_qwen35_20261013/eab65a9995ff"
python build_native.py lib --output build/qwen35_pf/libweftmoe.so --jobs 4
python tools/build_tensor_runtime.py --library build/qwen35_pf/libweftmoe.so

# Set WEFTMOE_TENSOR_EXTENSION to the absolute extension path printed above.
export WEFTMOE_AUTOTUNE_TABLE="${PWD}/configs/pf_qwen35_20261013/table.json"
export WEFTMOE_PF_MIN_RING_DEPTH=2
export WEFTMOE_PF_CREDIT_WINDOWS=2
```

`table.json` is an exact prepared-capacity lookup, not first-call JIT or runtime
autotuning. Its `training=true` entries use max_tokens 10055/25820 for EP8 and
10277/25820 for EP16, the actual capacities of the nominal 8K/20K traces.
Other prepared capacities require separately validated table entries; do not
silently relabel these measurements. Both topologies use the same native
library. A mismatched compiled ID is rejected collectively before allocation.

## Reproduce the search

```bash
python tools/autotune_training.py \
  --config configs/pf_qwen35_training_autotune.json \
  --hidden 2048 --intermediate 512 --experts 256 --top-k 8 \
  --tokens 8192 20480 --gpus-per-node 8 --jobs 8 \
  --output build/qwen35-tune-ep8

# For EP16 add --remotes root@HOST --master-addr LOCAL_IP --ssh-port 2222
# and use a different output directory. Lock clocks before benchmarking.
```

The original search included an extra EP16 B2048/D4 duplicate control. Network
PF derives contribution capacity from the complete invocation, so D-only changes
do not alter that data path. The duplicate was not selected; the shipped search
config removes it. Original logs retain it for audit. Raw runs and confirmations
are in `build/qwen35_tune_20261013/ep{8,16}/`; unique compiled artifacts are in
`build/qwen35_perf_20261013/artifacts/`.

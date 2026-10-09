# Autotune Configuration Reference

The configuration files define kernel hyperparameter search spaces for forward
and backward execution on single-node and multi-node systems. Input dimensions
and parallel topology are supplied separately by the caller. Compiler, library,
and network settings are not kernel hyperparameters.

`pf_training_autotune.json` is the PF **full FWD+BWD** search. It uses staged
ring/FWD/BWD candidates and a variable-token, random-route workload rather than
the independent phase grids described below. The same file accepts different
model dimensions and local/multi-node topologies through
`tools/autotune_training.py`; see [training autotune](../docs/autotune.md).
Its search candidates are not a claim of measured optimal parameters.

The single-node Flash **inference-only** profile is in
[`pf_inference_flash_20261008`](pf_inference_flash_20261008/weftmoe_forward_tuning_generated.hpp).
It enables the expert-global PF pipeline for BF16 H4096/I2048/E256/top-k6,
using M128/N256/K64, three shared input stages, two-CTA multicast, and
optional fast SwiGLU. It retains bounded X/S rings but reserves full source
contributions. In inference no activations or routes are cached between calls.
The same local main also supports a training tape: count-only mini-notify and
exact-size allocation precede main, which saves X and BF16 gate/up in the
existing Backward layout. The existing training, multi-node and DT profiles
are not replaced by this profile.
The accompanying lookup table has `training=false`, EP8, NVIDIA L20X entries
at exact prepared token capacities 10055 and 25820; these are measured trace
maxima, not bucket sizes or maximum supported lengths.

Build with the existing CUDA/CUTLASS/NCCL environment:

```bash
export NVCCFLAGS="-lineinfo -DWEFTMOE_FORWARD_USE_GENERATED_TUNING=1 -I${PWD}/configs/pf_inference_flash_20261008"
python build_native.py lib --output build/flash_pf_inference/libweftmoe.so --jobs 4
python tools/build_tensor_runtime.py --library build/flash_pf_inference/libweftmoe.so
```

Select the printed extension with `WEFTMOE_TENSOR_EXTENSION`. The optional
`WEFTMOE_AUTOTUNE_TABLE=configs/pf_inference_flash_20261008/table.json` performs
exact-capacity lookup, not runtime tuning or compilation. Without that table,
set `WEFTMOE_STATIC_RING_TOKENS=128` and
`WEFTMOE_STATIC_RING_CONTRIBUTION_WINDOWS=2` before preparing the runtime.
Eager inference may pass `layer=...` to prepare immutable weights once; call
`invalidate_inference_layer(layer)` before changing those weights, including
in-place updates. This is weight preparation, not saved intermediate results.
The benchmark option `--prepared-weights` exercises this path. Its wall timing
includes input staging, routing, communication, both GEMMs, reduce, and device
synchronization. Measurements and negative ablations are in the
[PF report](../docs/static_forward_v4_training_record.md).

To measure the cost of retaining activations on this local profile, use
`benchmarks/torchrun_megamoe_prefill.py --backend pf --forward-cache off|on`
with the same dimensions, `--ring-tokens 128 --ring-depth 2`, and
`--prepared-weights`. Both modes use the same native forward/tape API and
include tape release in wall timing. Omit the inference-only lookup table
when constructing a training runtime. The explicit layer option assumes
immutable weights; normal `TensorRuntime.forward` does not pass it and still
publishes the current training weights on every call. Forward-only cache
overhead does not establish end-to-end FWD+BWD performance.

The small-batch Flash DT comparison profile is in
[`dt_flash_probe_20261009`](dt_flash_probe_20261009/weftmoe_forward_tuning_generated.hpp).
It preserves the existing DT arithmetic and epilogue; it is not an autotuned
replacement or a deployment lookup table. The T256 measurement additionally
uses an isolated capacity-guard patch; production DT selection remains at
capacity <=64. See the [DT/PF comparison](../docs/static_forward_v4_decode_record.md)
for exact kernel traces, graph timing, and the patch needed to reproduce T256.

In the independent phase grids, each parameter contains a nonempty array of candidate values. The first values
form the incumbent configuration and the defaults used when autotuning is
disabled; the Cartesian product defines the search space. A single value fixes a parameter for that search. Supported values and
parameter combinations are checked by the autotuner before benchmarking.

FWD denotes forward execution; BWD denotes backward execution. A CTA is a CUDA
thread block. A warp contains 32 threads, and a warpgroup contains four warps.
An M-block is a block of routed rows belonging to one expert.

The public grids now select unified workers: every CTA contains math and
communication roles. Single-node FWD has 8 candidates, multi-node FWD has 72,
and BWD has 12 single-node or 36 multi-node resource-valid candidates.

The FWD-only `gin_channel_pack_limit` axis is `[3, 1, 2]` for multi-node and
`[3]` for single-node, default 3, with
valid values 1–8. It limits simultaneous packing reservations for one channel.
Value 1 selects the original single-reservation path at compile time; larger
values remain bounded by context slots and ring credit. The default equal to
the context slot count adds no per-channel counting. Communication workers use this policy; ordered submission and queue ownership
stay unchanged. Old FWD plans missing this key upgrade to 3; BWD does not
accept it. These GIN-only axes, context probes, context packing slots and
GIN local-transport policy are singletons in the public single-node grid,
whose `StaticLocal` specialization never executes GIN. Custom JSON plans
retain the full parameter ranges; only the public search space is narrowed.
See the
[worker architecture](../docs/full_cta_all_worker.md) for phase resource
parameters and the legacy-only transport settings fixed in these grids.

`gin_ring_depth` controls the physical RDMA TX/RX ring slots **per channel**,
independently of the context's `unified_gin_pack_slots` packing reservations.
The public multi-node axis is `[8, 4, 16]`; single-node fixes it to `[8]` because
local execution does not use GIN. It changes persistent device-memory storage,
local-completion counters and credit windows, without adding shared memory or
changing the payload chunk size. Forward and Backward must use the same value
in a combined binary. Increasing depth enlarges the credit window and the
final drain checks; normal ACK polling uses a dirty bitmap instead of scanning
every physical slot. Retune rather than assuming a larger value always helps.

Bare builds default to 8. New generated headers explicitly define
`WEFTMOE_GIN_RING_DEPTH`. Old plans and build records missing the key, and old
generated headers missing the macro, retain depth 4 when rebuilt. Headers
passed through `-include` are recognized by their pre-existing Forward or
Backward model-dimension macros; direct compiler dimension overrides without
an explicit depth conservatively follow the same compatibility rule. To
change an older profile, define `WEFTMOE_GIN_RING_DEPTH=8` or `16` explicitly.
This is a compile-time wire-layout setting: every rank must use the same
compiled depth, and an existing resource pool cannot change it at runtime.

The selected all-CTA Qwen and Flash profiles with eight physical slots are in
[`full_cta_all_worker_20260915`](full_cta_all_worker_20260915/README.md).
They copy the actual final build headers and retain all other September 14
kernel tuning values. The source now uses bitmap ACK without an optional scan
branch. Their README distinguishes final native and EP16/E256/8K/64K FWD/BWD
checks from earlier EP8 and Graph evidence, including identical final kernel
machine code. Formal EP8 performance was not rerun. Full samples are in the
[communication report](../benchmarks/ring_depth_20260915.md).
The [September 14 all-CTA profiles](full_cta_all_worker_20260914/README.md)
remain unchanged with their original four-slot wire layout.

ACK uses a per-channel dirty bitmap to identify completed RX slot generations.
Consumers publish a bit only after retiring the original payload work; the
leased ACK owner acquires and clears the current mask, then returns one credit
for each captured bit. Empty polls do not scan every physical slot. The bitmap
occupies existing channel padding, so registered-window sizes and slot offsets
remain unchanged. Final drain checks still verify the durable per-slot counts.
This reduces control overhead without changing payload bytes or completion
ordering. Forward can now assign the final warp of RDMA CTAs to independent GIN control.

`unified_rdma_ctas` selects Forward communication ownership, independently of
model dimensions. Zero retains mixed workers; a positive value reserves that
many CTA communication domains for RDMA packing/control, clamped to leave at
least one relay CTA. All CTAs retain GEMM workers. Single-node execution always
uses relay roles. Bare or macro-absent headers default to 16; legacy plans and
the public grids explicitly retain 0 until the new path is tuned. Backward
uses its own phase-specific sender quota: a positive value selects that count,
also leaving at least one receiver CTA; 0 in historical plans selects 16.
Backward no longer borrows Forward's quota, so disabling legacy Forward senders
for static inference does not collapse training to a single network sender.
Forward Return notifications use a dedicated full-warp single consumer in both
layouts. The staged evidence is in
[the communication partition record](../benchmarks/communication_partition_20260915.json).

The experimental Flash **network inference** profile is
[`pf_network_flash_20261009`](pf_network_flash_20261009/weftmoe_forward_tuning_generated.hpp).
It reuses the local PF GEMM with node-deduplicated Get dispatch and direct Put
return. Build with `WEFTMOE_PF_NETWORK_WORKERS=1` for selected dedicated warps,
or `=2` for the WG3 208/48/48 experiment. Existing profiles/default builds and
training selection are unchanged. See the [network report](../docs/pf_network_20261009.md)
for commands, buffer lifetimes, measured performance and validation scope.

## GEMM Parameters (`1_gemm`)

Tile dimensions determine work granularity and data reuse. Pipeline stages
control how many input buffers can overlap data movement with computation.
Larger tiles or deeper pipelines can improve reuse, but also increase resource
requirements and reduce scheduling flexibility.

Both phases require two-CTA compute clusters and use TMA multicast. Backward
activation GEMMs pair two complete M128 tiles along N when both dU and dX
have even N-tile counts. Both math warpgroups then run, and TMA multicasts
the activation tile. Odd N-tile counts retain the two M64 row halves and
multicast the weights. Both CTAs finish before their cluster task publishes
completion. Weight-gradient GEMMs prefer sharing the larger input operand,
changing pairing direction for odd tile counts; a singleton tail participates
in the cluster pipeline but only its valid tile writes a result.

| Parameter | Phase | Purpose and guidance |
|---|---|---|
| `tile_m` | FWD/BWD | Rows in a logical activation tile, currently fixed at 128. Backward uses full M128 CTAs for even N extents and M64 halves otherwise. Padding and the number of ready tiles affect utilization. |
| `tile_k` | FWD/BWD | Reduction-dimension step of the GEMM pipeline. The current implementation uses 64 to match its shared-memory layout. |
| `fc1_tile_n`, `fc2_tile_n` | FWD | Output-column dimensions of the two forward GEMMs. Start with FC1 N128 and FC2 N256 on Hopper. FC1 holds two projection accumulators, so increasing its N tile can be costly even when the shared-memory buffers fit. Keep FC2 N128 as an alternative for distributed tuning. |
| `dab_tile_n`, `dx_tile_n` | BWD | Output-column dimensions of the activation-gradient and input-gradient GEMMs. Start with 256 for both; evaluate narrower tiles when dimensions or task parallelism warrant it. Narrower tiles do not necessarily reduce register usage. |
| `dw_tile_m`, `dw_tile_n` | BWD | Output-tile dimensions of the weight-gradient GEMMs. Start with M128/N256. Larger tiles reuse more input data but require more accumulator storage. |
| `fc1_stages`, `fc2_stages` | FWD | Number of input-buffer stages for each forward GEMM. Start with 4/3. Each GEMM supports 2–4 stages, including partial stage cycles. The input and output buffers must fit the device shared-memory limit. |
| `fc2_claim_tiles` | FWD | Ready N tiles processed per CTA before returning to the scheduler: 1, 2, 4 or 8. Start with 4. Larger claims overlap prefetch and output stores, with a separate output tile; claims stop at the current M-block boundary. |
| `dab_stages`, `dx_stages` | BWD | Input-buffer stages for activation-gradient and input-gradient GEMMs. These values must match in the current implementation. Start with 4. |
| `dw1_stages`, `dw2_stages` | BWD | Input-buffer stages for the two weight-gradient GEMMs. These values must match in the current implementation. Start with 3. |
| `compute_threads` | FWD/BWD | Threads per compute CTA. The current implementation requires 384. |
| `math_warpgroups` | FWD/BWD | Warpgroups assigned to matrix computation. The current implementation requires two. |
| `producer_warpgroups` | FWD/BWD | Warpgroups assigned to staging operands. The current implementation requires one. |
| `math_registers`, `producer_registers` | FWD/BWD | Per-thread register budget requests for the corresponding roles. These are allocation requests, not measured register counts; inspect compiler resource usage when changing them. |
| `wgmma_wait_group` | FWD | Number of K iterations allowed to remain outstanding: 0 or 1. FC1 commits two WGMMA groups per iteration and FC2 commits one. Mode 1 releases only the previous shared-memory stage and drains the final iteration before the epilogue. Start with 0; additional overlap is not always faster. |
| `fc2_store_atom_n` | FWD | Column width of the FC2 output TMA store atom. Determines the granularity of output stores. |
| `cluster_size` | FWD | Fixed at two CTAs per compute cluster. TMA multicast is required; select N tiles that give even FC1 and FC2 tile counts. |
| `fc2_tma_store` | FWD | Required for the multicast compute path. Full tiles use shared-memory staging and TMA; partial expert tails use bounds-checked stores. |

## Transport Parameters (`2_transport`)

Relay CTAs handle node-local data movement and associated progress work. RDMA
CTAs handle inter-node GIN communication. Increasing transport concurrency can
improve operand delivery and return progress, but leaves fewer SMs available
for compute workers.

| Parameter | Phase | Purpose and guidance |
|---|---|---|
| `rdma_ctas` | FWD/BWD | Number of inter-node GIN communication CTAs. Single-node execution sets this to 0. The current GIN implementation uses 8; this is not the number of nodes or the EP size. |
| `relay_ctas` | FWD/BWD | Number of CTAs assigned to node-local transport. For small hidden dimensions, include larger values such as 24, 32, and 40: transport progress can limit throughput even on one node. Stop increasing the count when reduced compute capacity outweighs faster transport. Counts remain subject to alignment and residency constraints. |
| `local_dispatch_tma_warps` | FWD | Warps per relay CTA used for node-local dispatch copies. More copy streams increase concurrency but compete with progress and readiness publication work. |
| `local_return_tma_warps` | FWD | Warps used for node-local return TMA copies. A value of 0 selects vector copies. Comparing 0 and 8 is a useful initial search; more TMA streams are not always faster end to end. |
| `relay_scatter_warps` | FWD | Warps assigned to relay scatter copies. Controls concurrent TMA copy streams within a received dispatch chunk. Multiple relay CTAs can consume the same channel concurrently. Shared-memory usage grows with the row width and stream count. |
| `local_return_batch_rows` | FWD | Maximum published rows claimed in one local return batch. Larger batches amortize scheduling overhead but can delay service of other ready work. |
| `dispatch_publish_batch_rows` | FWD | Row granularity for publishing dispatch arrival progress. Smaller batches expose compute work sooner; larger batches reduce publication overhead. |
| `relay_local_return` | FWD | Assigns local return processing to relay CTAs. Keep enabled; the current supported path requires it. |
| `dispatch_batch_rows` | BWD | Routed rows claimed in one relay dispatch batch. Start with 16 or 32 for node-local execution. Larger batches can reduce scheduling overhead, while smaller batches improve responsiveness to return work. |
| `setup_threads` | BWD | Threads per setup CTA. Controls parallelism in pipeline initialization. |
| `setup_idle_nanosleep` | BWD | Backoff hint while setup waits for progress. Keep the default unless initialization waiting is significant. |

## Scheduler Parameters (`3_scheduler`)

Scheduler parameters balance queue contention, weight reuse, and how quickly
newly ready work can execute. Larger batches or locality groups can reduce
scheduling overhead while increasing the time other tasks spend waiting.

Backward activation readiness has a second bitmap level, skipping 64 empty
owner words per probe while preserving expert-major claim order. Retirement
and DAB phase repair update both levels to avoid lost wakeups. Local dX Return
uses the same append-only queue as dW scheduling, with one task per M-block;
GIN retains its expert-major scan. Neither changes the arithmetic reduction
order of dX, dGate, or the single-writer dW tiles.

| Parameter | Phase | Purpose and guidance |
|---|---|---|
| `local_route_batch_rows` | FWD | Number of local dispatch routes assigned per batch. Larger batches reduce claim overhead but can increase load imbalance. |
| `startup_mblocks` | FWD | Initial M-blocks per expert published without waiting for subsequent locality groups. Controls how quickly the first compute work becomes available. |
| `fc1_locality_group` | FWD | Groups adjacent M-blocks when publishing FC1 work to encourage reuse of the same weight tiles in L2. Larger groups can improve reuse but delay FC2 and return progress. |
| `fc1_locality_group4_begin`, `fc1_locality_group4_end` | FWD | Expert-local M-block interval `[begin, end)` that uses groups of four when the base locality group is two. These are row-block indices, not expert IDs or GPU groups. A bounded middle interval preserves finer scheduling near startup and completion. |
| `fc2_priority_burst` | FWD | Limit on consecutive priority FC2 task claims. A value of 0 leaves this priority burst unlimited. Smaller finite bursts allow FC1 work to receive service more frequently. |
| `task_claim_chunk` | FWD | Compute tasks claimed per queue operation. The current implementation supports one task per claim. |
| `task_idle_nanosleep` | FWD | Backoff hint for idle compute-task polling. |
| `transport_idle_nanosleep` | FWD | Backoff hint for idle transport polling. |
| `setup_nanosleep` | FWD | Backoff hint during setup waits. |
| `queue_idle_nanosleep` | BWD | Backoff hint when no compute-ready task is available. |
| `ready_word_stagger` | BWD | Starts different CTAs at different ready-bitmap words when enabled. Can reduce contention, but may weaken locality between related tasks. |
| `activation_owner_group` | BWD | Group size for distributing activation-gradient task claims among neighboring ready M-block owners. These are compute-work owners, not communication ranks. Start with 4 to balance contention and weight reuse. |
| `epilogue_rows_per_task` | BWD | Rows processed per CTA in an activation epilogue task. A cluster processes two such batches. Controls the granularity of non-GEMM activation-gradient work. |
| `dw2_claim_tiles`, `dw1_claim_tiles` | BWD | Weight-gradient tiles per CTA in a claimed range. A cluster claims this many tile pairs. Larger claims amortize scheduling overhead; while activation work can still arrive, claims are limited to one pair to preserve queue priority. |

Nanosleep values are GPU backoff hints, not guaranteed delays. Lower values may
improve responsiveness while increasing polling contention. Tune these after
transport concurrency and GEMM tile choices.

## Completion and Reduction Parameters (`4_completion_reduce`)

In FWD, return processing makes each routed contribution available at its
source token. Once all contributions for a token are ready, the token can enter
the reduction queue. Idle CTAs may consume ready tokens and accumulate their
contributions in a fixed top-k order. Inter-node return uses one GIN strong
signal per sender to publish arrived row notifications. Relay workers claim
only these confirmed rows before publishing route readiness.

BWD uses a separate input-gradient reduction kernel after persistent transport
completes. This reduction can overlap remaining weight-gradient computation.
Its thread-count setting does not change the main compute-worker allocation.

| Parameter | Phase | Purpose and guidance |
|---|---|---|
| `completion_peer_probes` | FWD | Sender queues probed for newly arrived return rows in one progress pass. Larger values discover arrivals sooner but reduce time available for other transport work. |
| `reduce_scheduler` | FWD | Selects how ready reduction work is discovered and scheduled. Use `queue` as the starting point. |
| `reduce_warps` | FWD | Warps participating when a CTA executes reduction work. Controls parallelism within the reduction role. |
| `reduce_tokens_per_claim` | FWD | Tokens claimed in one reduction queue operation. Must not exceed `reduce_warps`. Larger claims reduce queue overhead but can reduce load balance near completion. |
| `reduce_registers` | FWD | Per-thread register budget request when a CTA enters its reduction role. Inspect compiled resource usage when changing it. |
| `return_idle_nanosleep` | FWD | Backoff hint while waiting for return progress. |
| `reduce_queue_nanosleep` | FWD | Backoff hint while waiting for reduction work. |
| `finalize_nanosleep` | FWD | Backoff hint while waiting for final completion. |
| `reduce_threads` | BWD | Threads per input-gradient reduction CTA. Each warp handles one token in the current implementation. Start with 256; larger CTAs may reduce scheduling flexibility for short reductions. |
| `reduce_idle_nanosleep` | BWD | Backoff hint while waiting for input-gradient route contributions. |

Earlier Qwen3.5-35B-A3B and DeepSeek-V4-Flash GEMM optimization profiles are preserved in
[`performance_20260914/profiles.json`](performance_20260914/profiles.json).
FC2 uses three input stages and `fc2_claim_tiles=4` on both models. This
parameter accepts 1, 2, 4 or 8 tiles per CTA, clamped to the remaining N tiles
of one ready M block. Values above one reserve a separate output tile so
prefetch and output stores can overlap. The autotuner checks shared-memory
capacity; four stages with N256 and a separate output tile do not fit Hopper.
K iterations need not divide evenly into the stage count.
Each referenced header contains both Forward and Backward settings. For example,
build a production library from the repository root with:

```sh
NVCCFLAGS="-include $PWD/configs/performance_20260914/qwen_ep8.hpp" \
  python3 build_native.py lib --output build/libweftmoe_qwen_ep8.so --jobs 16
```

The Qwen EP8 64K case uses `qwen_ep8_64k.hpp`; the other three model/topology
combinations share one profile across 8K and 64K. Tokens are per rank. These
profiles do not change the model or topology interfaces. Retune other
supported aligned dimensions and hardware with the
existing autotuners instead of applying these model dimensions to other layers.
